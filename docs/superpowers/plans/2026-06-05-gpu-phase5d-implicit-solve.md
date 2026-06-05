# GPU Phase 5d — Implicit Field Solve (BC rows + batched solve) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Perform one field's CNAB2 **implicit solve** on a single GPU — set the boundary rows of the RHS to the prescribed BC values, then batched-solve the per-mode banded systems — matching the CPU `_solver_solve_scalar_implicit_step!` exactly. This composes Phase 5c's RHS + Phase 4's batched solve into one validated implicit half-step.

**Architecture:** From the CPU (`src/timestep/imex.jl:343-366`): for each mode `(l,m)`, gather the RHS radial profile, **overwrite row 1 with the inner BC value and row `nr` with the outer BC value** (the system matrix already has identity/BC rows embedded at those positions), then `solve_banded!` in place with the degree-`l` LU factor, and scatter back. On the dense `(nl, nm, nr)` GPU layout this is: a **BC-row write** (set `x[:,:,1]` and `x[:,:,nr]` to per-mode BC arrays) + Phase 4's `gpu_batched_banded_solve!` (in-place `X===B` is supported). The BC-row write is the only new piece (a broadcast/view assignment); everything else reuses Phase 4.

**Tech Stack:** Julia, broadcast (BC-row write), reuses Phase 4 `gpu_batched_banded_solve!` + `gpu_pack_banded_lu`, Phase-0 `on_architecture`. No KA kernel, no CUDA extension.

---

## Background (CPU reference — `src/timestep/imex.jl:337-367`)

```
per mode (l,m):  tmp = rhs[mode, :]           # length nr
  tmp_real[1]  = bc_inner[lm_idx]   ; tmp_imag[1]  = bc_inner_imag[lm_idx]
  tmp_real[nr] = bc_outer[lm_idx]   ; tmp_imag[nr] = bc_outer_imag[lm_idx]
  solve_banded!(tmp, factorizations[l], tmp)   # in-place, per-degree-l LU
  solution[mode, :] = tmp
```
BC values are per `(l,m)` mode (often 0 for perturbation fields, or a fixed boundary value). The matrix rows 1/`nr` are the embedded BC equations (identity for Dirichlet), so writing the RHS boundary rows + solving imposes the BC.

## Testing without a local GPU

- **[LOCAL]** — the BC-row write + Phase-4 solve run on Array; tests assert the result **equals** the CPU per-mode reference (`tmp[1]=bc_in; tmp[nr]=bc_out; solve_banded!`) — exact `==`. Real verification.
- **[GPU-BOX]** — same on `CuArray`; guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu` (branch `feat/gpu-phase0`, on Phases 0–5c). **Never pipe test runs through `tail`.**

## File Structure

- **Create** `src/gpu/implicit_solve.jl` — `gpu_apply_bc_rows!`, `gpu_implicit_solve_field!`.
- **Modify** `src/GeoDynamo.jl` — `include("gpu/implicit_solve.jl")` (after `gpu/cnab2_rhs.jl`); export both.
- **Create** `test/gpu_phase5d_implicit_solve.jl` — `[LOCAL]` + `[GPU-BOX]` tests.
- **Modify** `test/runtests.jl` — register.

Locked interfaces:

```julia
gpu_apply_bc_rows!(x_r, x_i, bc_in_r, bc_in_i, bc_out_r, bc_out_i)
    # x_r[:,:,1] .= bc_in_r ; x_i[:,:,1] .= bc_in_i ; x_r[:,:,nr] .= bc_out_r ; x_i[:,:,nr] .= bc_out_i
gpu_implicit_solve_field!(x_r, x_i, lu_batched, bc_in_r, bc_in_i, bc_out_r, bc_out_i, bw)
    # x holds RHS on entry; applies BC rows, batched-solves in place; x holds the solution on exit
```

`x_*` are `(nl,nm,nr)` (RHS in, solution out); `bc_*` are `(nl,nm)` (per-mode boundary values); `lu_batched` `(2bw+1,nr,nl)` per-l LU factors (from Phase 4 `gpu_pack_banded_lu`); same backend.

---

## Task 1: `gpu_apply_bc_rows!`

**Files:** Create `src/gpu/implicit_solve.jl`; Modify `src/GeoDynamo.jl`; Test `test/gpu_phase5d_implicit_solve.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase5d_implicit_solve.jl`:

```julia
using Test
using GeoDynamo
using Random

@testset "GPU Phase 5d — Implicit Solve" begin
    @testset "apply BC rows [LOCAL]" begin
        nl, nm, nr = 4, 3, 6
        x_r = rand(MersenneTwister(1), nl, nm, nr); x_i = rand(MersenneTwister(2), nl, nm, nr)
        bir = rand(MersenneTwister(3), nl, nm); bii = rand(MersenneTwister(4), nl, nm)
        bor = rand(MersenneTwister(5), nl, nm); boi = rand(MersenneTwister(6), nl, nm)
        x0r = copy(x_r); x0i = copy(x_i)
        GeoDynamo.gpu_apply_bc_rows!(x_r, x_i, bir, bii, bor, boi)
        @test x_r[:, :, 1] == bir && x_i[:, :, 1] == bii          # inner row set
        @test x_r[:, :, nr] == bor && x_i[:, :, nr] == boi        # outer row set
        @test x_r[:, :, 2:(nr-1)] == x0r[:, :, 2:(nr-1)]          # interior untouched
        @test x_i[:, :, 2:(nr-1)] == x0i[:, :, 2:(nr-1)]
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5d_implicit_solve.jl")'`
Expected: FAIL — `gpu_apply_bc_rows!` undefined.

- [ ] **Step 3: Implement**

Create `src/gpu/implicit_solve.jl`:

```julia
# =============================================================================
# GPU Phase 5d — one field's CNAB2 implicit solve: set the RHS boundary rows to
# the prescribed BC values (the system matrix has the BC equations embedded at
# rows 1 and nr), then batched-solve the per-mode banded systems (Phase 4).
# Mirrors _solver_solve_scalar_implicit_step! (imex.jl:343-366). Composes Phase 5c
# (RHS) + Phase 4 (solve). Broadcast + reused solve → runs on Array (locally
# testable) and CuArray.
# =============================================================================

"""
    gpu_apply_bc_rows!(x_r, x_i, bc_in_r, bc_in_i, bc_out_r, bc_out_i) -> nothing

Overwrite the boundary rows of the per-mode radial RHS with the prescribed BC
values: row 1 (inner) ← `bc_in_*`, row `nr` (outer) ← `bc_out_*`.  `x_*` are
`(nl,nm,nr)`; `bc_*` are `(nl,nm)` (per-`(l,m)` boundary value).
"""
function gpu_apply_bc_rows!(x_r, x_i, bc_in_r, bc_in_i, bc_out_r, bc_out_i)
    nr = size(x_r, 3)
    @views x_r[:, :, 1]  .= bc_in_r
    @views x_i[:, :, 1]  .= bc_in_i
    @views x_r[:, :, nr] .= bc_out_r
    @views x_i[:, :, nr] .= bc_out_i
    return nothing
end
```

- [ ] **Step 4: Include + export**

In `src/GeoDynamo.jl`, after `include("gpu/cnab2_rhs.jl")` add `include("gpu/implicit_solve.jl")`. Add export:
```julia
export gpu_apply_bc_rows!, gpu_implicit_solve_field!
```

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5d_implicit_solve.jl")'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/implicit_solve.jl src/GeoDynamo.jl test/gpu_phase5d_implicit_solve.jl
git commit -m "feat(gpu): gpu_apply_bc_rows! (set implicit RHS boundary rows) (Phase 5d)"
```

---

## Task 2: `gpu_implicit_solve_field!` (BC + batched solve)

**Files:** Modify `src/gpu/implicit_solve.jl`; Test `test/gpu_phase5d_implicit_solve.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase5d_implicit_solve.jl`:

```julia
@testset "implicit field solve == CPU implicit step [LOCAL]" begin
    nr, bw, nl, nm = 8, 2, 4, 3
    # per-l non-singular (diagonally dominant) banded matrices → LU factors
    function band(::Type{T}, N, bw; seed) where {T}
        rng = MersenneTwister(seed); d = zeros(T, 2bw+1, N)
        for j in 1:N, i in max(1,j-bw):min(N,j+bw)
            d[bw+1+i-j,j] = (i==j) ? (T(2bw)+rand(rng,T)) : (rand(rng,T)-T(0.5))
        end
        GeoDynamo.BandedMatrix{T}(d, bw, N)
    end
    lus = [GeoDynamo.factorize_banded(band(Float64, nr, bw; seed = 90 + l)) for l in 1:nl]
    lub = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
    rng = MersenneTwister(91)
    rhs_r = rand(rng, nl, nm, nr); rhs_i = rand(rng, nl, nm, nr)
    bir = rand(rng, nl, nm); bii = rand(rng, nl, nm); bor = rand(rng, nl, nm); boi = rand(rng, nl, nm)
    x_r = copy(rhs_r); x_i = copy(rhs_i)
    GeoDynamo.gpu_implicit_solve_field!(x_r, x_i, lub, bir, bii, bor, boi, bw)
    # CPU reference: per mode, set BC rows then solve_banded!
    for l in 1:nl, m in 1:nm
        tr = collect(rhs_r[l, m, :]); ti = collect(rhs_i[l, m, :])
        tr[1] = bir[l, m]; ti[1] = bii[l, m]; tr[nr] = bor[l, m]; ti[nr] = boi[l, m]
        GeoDynamo.solve_banded!(tr, lus[l], tr); GeoDynamo.solve_banded!(ti, lus[l], ti)
        @test x_r[l, m, :] == tr
        @test x_i[l, m, :] == ti
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5d_implicit_solve.jl")'`
Expected: FAIL — `gpu_implicit_solve_field!` undefined.

- [ ] **Step 3: Implement**

Append to `src/gpu/implicit_solve.jl`:

```julia
"""
    gpu_implicit_solve_field!(x_r, x_i, lu_batched, bc_in_r, bc_in_i, bc_out_r, bc_out_i, bw) -> nothing

One field's CNAB2 implicit solve: `x` holds the RHS on entry; this sets the BC
boundary rows (`gpu_apply_bc_rows!`) then batched-solves the per-mode banded
systems in place (Phase 4 `gpu_batched_banded_solve!`, `X===B` supported), leaving
the solution in `x`.  `lu_batched` `(2bw+1,nr,nl)` are the per-l LU factors.
"""
function gpu_implicit_solve_field!(x_r, x_i, lu_batched, bc_in_r, bc_in_i, bc_out_r, bc_out_i, bw::Int)
    gpu_apply_bc_rows!(x_r, x_i, bc_in_r, bc_in_i, bc_out_r, bc_out_i)
    gpu_batched_banded_solve!(x_r, x_r, lu_batched, bw)   # in-place: solution overwrites RHS
    gpu_batched_banded_solve!(x_i, x_i, lu_batched, bw)
    return nothing
end
```

- [ ] **Step 4: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5d_implicit_solve.jl")'`
Expected: PASS — the GPU implicit solve matches the CPU per-mode reference exactly.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/implicit_solve.jl test/gpu_phase5d_implicit_solve.jl
git commit -m "feat(gpu): gpu_implicit_solve_field! (BC rows + batched solve) (Phase 5d)"
```

---

## Task 3: GPU-box gate + register + regression

**Files:** Test `test/gpu_phase5d_implicit_solve.jl`, `test/runtests.jl`

- [ ] **Step 1: Add the GPU-box gate** `[GPU-BOX]`

Add to `test/gpu_phase5d_implicit_solve.jl`:

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-5d gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        nr, bw, nl, nm = 10, 2, 4, 3
        function band(::Type{T}, N, bw; seed) where {T}
            rng = MersenneTwister(seed); dd = zeros(T, 2bw+1, N)
            for j in 1:N, i in max(1,j-bw):min(N,j+bw)
                dd[bw+1+i-j,j] = (i==j) ? (T(2bw)+rand(rng,T)) : (rand(rng,T)-T(0.5))
            end
            GeoDynamo.BandedMatrix{T}(dd, bw, N)
        end
        lus = [GeoDynamo.factorize_banded(band(Float64, nr, bw; seed = 95 + l)) for l in 1:nl]
        rng = MersenneTwister(96)
        rhs_r = rand(rng,nl,nm,nr); rhs_i = rand(rng,nl,nm,nr)
        bir = rand(rng,nl,nm); bii = rand(rng,nl,nm); bor = rand(rng,nl,nm); boi = rand(rng,nl,nm)
        # CPU
        club = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
        cxr = copy(rhs_r); cxi = copy(rhs_i)
        GeoDynamo.gpu_implicit_solve_field!(cxr, cxi, club, bir, bii, bor, boi, bw)
        # GPU
        d(x) = GeoDynamo.on_architecture(GPU(), x)
        glub = GeoDynamo.gpu_pack_banded_lu(lus, GPU())
        gxr = d(copy(rhs_r)); gxi = d(copy(rhs_i))
        GeoDynamo.gpu_implicit_solve_field!(gxr, gxi, glub, d(bir), d(bii), d(bor), d(boi), bw)
        @test gxr isa CUDA.CuArray
        @test isapprox(Array(gxr), cxr; atol = 1e-12, rtol = 1e-10)
        @test isapprox(Array(gxi), cxi; atol = 1e-12, rtol = 1e-10)
    end
end
```

- [ ] **Step 2: Run locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5d_implicit_solve.jl")'`
Expected: `[LOCAL]` testsets pass; the gate skips.

- [ ] **Step 3: Register**

In `test/runtests.jl`, add `"gpu_phase5d_implicit_solve.jl"` (next to the Phase 5c entry).

- [ ] **Step 4: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5c_cnab2_rhs.jl")'` then separately `… -e 'using Test, GeoDynamo, MPI; include("test/allocation_runtime_checks.jl")'`
Expected: Phase 5c green; allocation guards 39/39.

- [ ] **Step 5: Commit**

```bash
git add test/gpu_phase5d_implicit_solve.jl test/runtests.jl
git commit -m "test(gpu): Phase-5d GPU-box gate + register implicit solve"
```

---

## GPU-box validation handoff

On the GPU box:
```julia
using CUDA, Test, GeoDynamo, Random
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase5d_implicit_solve.jl")    # the [GPU-BOX] gate must PASS
```
**Phase 5d passes when:** the BC-row write + batched solve on `CuArray` match the CPU implicit step to ~1e-12.

---

## What this unblocks / what's next

The **implicit half-step** (RHS → BC rows → solve → updated field) now runs on GPU and matches the CPU per-field implicit step. With Phases 1–5d, the implicit side of a timestep is fully on GPU. Remaining:
- **Phase 5e — the EXPLICIT half + full `solver_step!` orchestration**: per field, transform → curl/gradient (5a/5b) → nonlinear products (2) → analyze → `gpu_build_rhs_cnab2!` (5c) → `gpu_implicit_solve_field!` (5d), wired across all fields (T, C, velocity, magnetic) with their couplings; the **GPU≈CPU full-step gate** lives here. This needs GPU field-container plumbing (a device `SolverState`-equivalent) + per-l matrix/LU caches on device.
- **Phase 6 — `run!`/`Simulation` device-resident loop + IO host-gather.**

---

## Self-Review

**Spec coverage:** the timestep's implicit solve needs BC-row imposition + per-mode banded solve. This delivers it: `gpu_apply_bc_rows!` (Task 1) + `gpu_implicit_solve_field!` composing it with Phase 4 (Task 2), GPU gate + regression (Task 3). The explicit half + full orchestration + the device field plumbing are the explicit next increment. Covered for the implicit solve.

**Placeholder scan:** none — every code step has complete code; every run step has the exact command + expected result. The `band` helper is fully defined inline.

**Type consistency:** `gpu_apply_bc_rows!(x_r,x_i, bc_in_r,bc_in_i, bc_out_r,bc_out_i)`, `gpu_implicit_solve_field!(x_r,x_i, lu_batched, bc_in_r,bc_in_i, bc_out_r,bc_out_i, bw)` — consistent across tasks and the interface block. BC rows at index 1 (inner) and `nr` (outer), matching `imex.jl:358-361`. Reuses Phase 4 `gpu_batched_banded_solve!` (in-place `X===B`, verified Phase 4) + `gpu_pack_banded_lu`, and `GeoDynamo.factorize_banded`/`solve_banded!`/`BandedMatrix` (test references). `x_*` `(nl,nm,nr)`, `bc_*` `(nl,nm)`. The `@views x[:,:,1] .= bc` broadcast-assign works on Array and CuArray (in-place device slice write).
