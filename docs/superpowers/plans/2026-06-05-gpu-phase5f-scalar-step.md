# GPU Phase 5f — Full Scalar Field Step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Advance one scalar field (temperature/composition) by one CNAB2 timestep on a single GPU — nonlinear term → RHS → implicit solve → field update + `nl_prev` rollover — composing the verified GPU pieces (5e, 5c, 5d), matching the CPU `apply_temperature_implicit_update!` + history rollover.

**Architecture:** From the CPU (`src/physics/temperature/solver.jl:121-208` + `roll_solver_histories!`): (1) the nonlinear term `nl` is computed (Phase 5e); (2) `build_rhs_cnab2!` forms the RHS from the OLD field, `nl`, `nl_prev`; (3) `solve_temperature_implicit_step!` imposes BCs + solves, overwriting the field's spectral storage; (4) at step end, `nl → nl_prev` (a copy). The GPU `gpu_scalar_field_step!` chains: `gpu_scalar_nonlinear!` (5e) → `gpu_build_rhs_cnab2!` (5c) → `gpu_implicit_solve_field!` (5d, in-place, solution overwrites the RHS) → `spec .= solution`; `prev_nl .= nl`. Ordering is safe: `build_rhs` reads the OLD `spec` before the solve produces the new one; `spec` is overwritten only at the end.

**Tech Stack:** Julia, reuses Phase 5e `gpu_scalar_nonlinear!`, Phase 5c `gpu_build_rhs_cnab2!`, Phase 5d `gpu_implicit_solve_field!`, Phase 0 `arch_of`. No new kernel.

---

## Background (CPU reference — `src/physics/temperature/solver.jl:121-208`)

```
# (nl already computed in compute_solver_nonlinear_terms!)
build_rhs_cnab2!(work_spectral, spectral, nonlinear, prev_nonlinear, dt, matrices)  # RHS
solve_temperature_implicit_step!(spectral, work_spectral, matrices; bc_inner, bc_outer, ...)  # overwrites spectral
# at step end: roll_solver_histories! copies nonlinear → prev_nonlinear
```
`build_rhs` uses the OLD `spectral` (`inv_dt·u + 1.5·nl − 0.5·nl_prev + (1−θ)·L·u`); the solve produces the new field; the rollover saves `nl` as the next step's `nl_prev`.

## Testing without a local GPU

- **[LOCAL]** — the whole step runs on Array (transforms via SHTnsKit fallback). The test asserts the step's outputs (`spec`, `prev_nl`) **equal a manual chain** of `gpu_scalar_nonlinear!` → `gpu_build_rhs_cnab2!` → `gpu_implicit_solve_field!` → copies (exact `==`) — verifying step wiring + ordering. Sub-pieces verified per phase.
- **[GPU-BOX]** — same on `CuArray`; guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu` (branch `feat/gpu-phase0`, on Phases 0–5e). **Never pipe test runs through `tail`.**

## File Structure

- **Create** `src/gpu/scalar_step.jl` — `gpu_scalar_field_step!`.
- **Modify** `src/GeoDynamo.jl` — `include("gpu/scalar_step.jl")` (after `gpu/scalar_nonlinear.jl`); export it.
- **Create** `test/gpu_phase5f_scalar_step.jl` — `[LOCAL]` + `[GPU-BOX]` tests.
- **Modify** `test/runtests.jl` — register.

Locked interface:

```julia
gpu_scalar_field_step!(spec_r, spec_i, prev_nl_r, prev_nl_i, u_r, u_θ, u_φ, config,
                       d1, mvals, rinv, lin_batched, lu_batched,
                       bc_in_r, bc_in_i, bc_out_r, bc_out_i, inv_dt, linear_weight, lmax, bw)
    # nl = nonlinear(spec,u) [5e]; rhs = build_rhs(spec,nl,prev_nl) [5c];
    # solve(rhs)→solution [5d]; spec .= solution; prev_nl .= nl
```

`spec_*`/`prev_nl_*` dense `(nl,nm,nr)` (spec: field in/out; prev_nl: in=nl_{n-1}, out=nl_n); `u_*` physical `(nlat,nlon,nr)`; `d1` `(2bw+1,nr)`; `mvals` len-`nm`; `rinv` len-`nr`; `lin_batched`/`lu_batched` `(2bw+1,nr,nl)` (per-l linear operator + per-l LU of the system matrix); `bc_*` `(nl,nm)`; `inv_dt`/`linear_weight` scalars; same backend.

---

## Task 1: `gpu_scalar_field_step!`

**Files:** Create `src/gpu/scalar_step.jl`; Modify `src/GeoDynamo.jl`; Test `test/gpu_phase5f_scalar_step.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase5f_scalar_step.jl`:

```julia
using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5f — Full Scalar Field Step" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 4)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 4
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    function band(::Type{T}, N, bw; seed, dd = false) where {T}
        rng = MersenneTwister(seed); d = zeros(T, 2bw+1, N)
        for j in 1:N, i in max(1,j-bw):min(N,j+bw)
            d[bw+1+i-j,j] = (dd && i==j) ? (T(2bw)+rand(rng,T)) : (rand(rng,T)-T(0.5))
        end
        GeoDynamo.BandedMatrix{T}(d, bw, N)
    end
    d1 = band(Float64, nr, bw; seed = 1).data
    mvals = Float64.(0:(nm-1)); rinv = [1.0/(0.5+0.1k) for k in 1:nr]
    # per-l linear operators (L) + system LU (factorized, non-singular)
    linmats = [band(Float64, nr, bw; seed = 10 + l) for l in 1:nl]
    lin = zeros(Float64, 2bw+1, nr, nl); for l in 1:nl; lin[:,:,l] .= linmats[l].data; end
    sysmats = [band(Float64, nr, bw; seed = 20 + l, dd = true) for l in 1:nl]
    lus = [GeoDynamo.factorize_banded(m) for m in sysmats]
    lub = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
    rng = MersenneTwister(2)
    spec_r = zeros(nl,nm,nr); spec_i = zeros(nl,nm,nr)
    for mi in 1:nm, li in mi:nl, r in 1:nr; spec_r[li,mi,r]=rand(rng); spec_i[li,mi,r]=rand(rng); end
    pnl_r = rand(rng,nl,nm,nr); pnl_i = rand(rng,nl,nm,nr)
    u_r = rand(rng,nlat,nlon,nr); u_θ = rand(rng,nlat,nlon,nr); u_φ = rand(rng,nlat,nlon,nr)
    bir = rand(rng,nl,nm); bii = rand(rng,nl,nm); bor = rand(rng,nl,nm); boi = rand(rng,nl,nm)
    inv_dt = 1.0/0.01; lw = 0.5

    @testset "step == manual chain [LOCAL]" begin
        # GPU step (copies of mutable inputs)
        sr = copy(spec_r); si = copy(spec_i); pr = copy(pnl_r); pi_ = copy(pnl_i)
        GeoDynamo.gpu_scalar_field_step!(sr, si, pr, pi_, u_r, u_θ, u_φ, cfg, d1, mvals, rinv,
                                         lin, lub, bir, bii, bor, boi, inv_dt, lw, cfg.lmax, bw)
        # manual chain
        msr = copy(spec_r); msi = copy(spec_i); mpr = copy(pnl_r); mpi = copy(pnl_i)
        mnl_r = zeros(nl,nm,nr); mnl_i = zeros(nl,nm,nr)
        GeoDynamo.gpu_scalar_nonlinear!(mnl_r, mnl_i, msr, msi, u_r, u_θ, u_φ, cfg, d1, mvals, rinv, cfg.lmax, bw)
        rhs_r = zeros(nl,nm,nr); rhs_i = zeros(nl,nm,nr)
        GeoDynamo.gpu_build_rhs_cnab2!(rhs_r, rhs_i, msr, msi, mnl_r, mnl_i, mpr, mpi, lin, inv_dt, lw, bw)
        GeoDynamo.gpu_implicit_solve_field!(rhs_r, rhs_i, lub, bir, bii, bor, boi, bw)
        msr .= rhs_r; msi .= rhs_i; mpr .= mnl_r; mpi .= mnl_i

        @test sr == msr && si == msi           # updated field
        @test pr == mpr && pi_ == mpi          # rolled-over nl_prev (= this step's nl)
        @test all(isfinite, sr) && all(isfinite, pr)
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5f_scalar_step.jl")'`
Expected: FAIL — `gpu_scalar_field_step!` undefined.

- [ ] **Step 3: Implement**

Create `src/gpu/scalar_step.jl`:

```julia
# =============================================================================
# GPU Phase 5f — one scalar field's full CNAB2 timestep, composing the verified
# pieces: nonlinear (5e) → RHS (5c) → implicit solve (5d) → field update +
# nl_prev rollover.  Mirrors apply_temperature_implicit_update! + the CNAB2
# history rollover (temperature/solver.jl:121-208).  Runs on Array (locally
# testable) and CuArray.  (Per-call scratch — Phase-6 may cache.)
# =============================================================================

"""
    gpu_scalar_field_step!(spec_r, spec_i, prev_nl_r, prev_nl_i, u_r, u_θ, u_φ, config,
                           d1, mvals, rinv, lin_batched, lu_batched,
                           bc_in_r, bc_in_i, bc_out_r, bc_out_i, inv_dt, linear_weight, lmax, bw) -> nothing

Advance one scalar field one CNAB2 step.  On entry `spec_*` is the field and
`prev_nl_*` the previous nonlinear term; on exit `spec_*` is the updated field and
`prev_nl_*` holds THIS step's nonlinear term (rolled over).  `lin_batched` are the
per-l linear operators `L`, `lu_batched` the per-l LU factors of the system matrix
`(I−θ·dt·L)`; `bc_*` the per-mode BC values; `inv_dt = mass_coeff/dt`,
`linear_weight = 1−θ`.  All arrays on the same backend.
"""
function gpu_scalar_field_step!(spec_r, spec_i, prev_nl_r, prev_nl_i, u_r, u_θ, u_φ, config,
        d1, mvals, rinv, lin_batched, lu_batched, bc_in_r, bc_in_i, bc_out_r, bc_out_i,
        inv_dt, linear_weight, lmax::Int, bw::Int)
    # 1. nonlinear term (5e)
    nl_r = similar(spec_r); nl_i = similar(spec_i)
    gpu_scalar_nonlinear!(nl_r, nl_i, spec_r, spec_i, u_r, u_θ, u_φ, config, d1, mvals, rinv, lmax, bw)
    # 2. CNAB2 RHS from the OLD field, nl, prev_nl (5c)
    rhs_r = similar(spec_r); rhs_i = similar(spec_i)
    gpu_build_rhs_cnab2!(rhs_r, rhs_i, spec_r, spec_i, nl_r, nl_i, prev_nl_r, prev_nl_i,
                         lin_batched, inv_dt, linear_weight, bw)
    # 3. implicit solve (BC rows + batched solve, in-place → solution in rhs) (5d)
    gpu_implicit_solve_field!(rhs_r, rhs_i, lu_batched, bc_in_r, bc_in_i, bc_out_r, bc_out_i, bw)
    # 4. update the field with the solution
    spec_r .= rhs_r
    spec_i .= rhs_i
    # 5. roll the history: prev_nl ← this step's nl
    prev_nl_r .= nl_r
    prev_nl_i .= nl_i
    return nothing
end
```

- [ ] **Step 4: Include + export**

In `src/GeoDynamo.jl`, after `include("gpu/scalar_nonlinear.jl")` add `include("gpu/scalar_step.jl")`. Add export `gpu_scalar_field_step!`.

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5f_scalar_step.jl")'`
Expected: PASS — the step equals the manual chain (field + nl_prev), finite.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/scalar_step.jl src/GeoDynamo.jl test/gpu_phase5f_scalar_step.jl
git commit -m "feat(gpu): gpu_scalar_field_step! (full CNAB2 scalar step: 5e→5c→5d + rollover) (Phase 5f)"
```

---

## Task 2: GPU-box gate + register + regression

**Files:** Test `test/gpu_phase5f_scalar_step.jl`, `test/runtests.jl`

- [ ] **Step 1: Add the GPU-box gate** `[GPU-BOX]`

Add to `test/gpu_phase5f_scalar_step.jl` (inside the outer testset, reusing the setup vars):

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-5f gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        # CPU reference
        csr = copy(spec_r); csi = copy(spec_i); cpr = copy(pnl_r); cpi = copy(pnl_i)
        GeoDynamo.gpu_scalar_field_step!(csr, csi, cpr, cpi, u_r, u_θ, u_φ, cfg, d1, mvals, rinv,
                                         lin, lub, bir, bii, bor, boi, inv_dt, lw, cfg.lmax, bw)
        # GPU
        d(x) = GeoDynamo.on_architecture(GPU(), x)
        glub = GeoDynamo.gpu_pack_banded_lu(lus, GPU())
        gsr=d(copy(spec_r)); gsi=d(copy(spec_i)); gpr=d(copy(pnl_r)); gpi=d(copy(pnl_i))
        GeoDynamo.gpu_scalar_field_step!(gsr, gsi, gpr, gpi, d(u_r), d(u_θ), d(u_φ), cfg,
                                         d(d1), d(mvals), d(rinv), d(lin), glub,
                                         d(bir), d(bii), d(bor), d(boi), inv_dt, lw, cfg.lmax, bw)
        @test gsr isa CUDA.CuArray
        @test isapprox(Array(gsr), csr; atol = 1e-9, rtol = 1e-8)
        @test isapprox(Array(gpr), cpr; atol = 1e-9, rtol = 1e-8)
    end
end
```

(Tolerance 1e-9 — the step runs the transform path; errors accumulate through nonlinear+RHS+solve.)

- [ ] **Step 2: Run locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5f_scalar_step.jl")'`
Expected: `[LOCAL]` testsets pass; the gate skips.

- [ ] **Step 3: Register**

In `test/runtests.jl`, add `"gpu_phase5f_scalar_step.jl"` (next to the Phase 5e entry).

- [ ] **Step 4: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5e_scalar_nonlinear.jl")'` then separately `… -e 'using Test, GeoDynamo, MPI; include("test/allocation_runtime_checks.jl")'`
Expected: Phase 5e green; allocation guards 39/39.

- [ ] **Step 5: Commit**

```bash
git add test/gpu_phase5f_scalar_step.jl test/runtests.jl
git commit -m "test(gpu): Phase-5f GPU-box gate + register scalar field step"
```

---

## GPU-box validation handoff

On the GPU box:
```julia
using CUDA, Test, GeoDynamo, MPI, Random
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase5f_scalar_step.jl")    # the [GPU-BOX] gate must PASS
```
**Phase 5f passes when:** the full scalar step on `CuArray` matches the CPU(Array) result (field + nl_prev) to ~1e-9.

---

## What this unblocks / what's next

A full scalar field step (temperature/composition) now runs on GPU. Remaining toward the full solver:
- **Phase 5g — vector fields** (velocity, magnetic): vector transform (3) + curls (5a) + cross-product nonlinears (2: u×ω, J×B, u×B) + the field-specific BC handling (velocity `l=1,m=0` rotation; poloidal influence-matrix correction; magnetic conducting inner-core reconstruction) + the vector field step.
- **Phase 5h — the full multi-field `gpu_solver_step!`**: velocity→magnetic→temperature→composition (dependency order, velocity nonlinear first = shared advecting flow), device `SolverState` plumbing (all fields + caches + BC on device), GPU≈CPU full-step gate.
- **Phase 6 — `run!`/`Simulation` loop + IO host-gather.**

---

## Self-Review

**Spec coverage:** one scalar field's CNAB2 step — `gpu_scalar_field_step!` composing nonlinear (5e) → RHS (5c) → implicit solve (5d) → field update + `nl_prev` rollover (Task 1); GPU gate + regression (Task 2). Matches `apply_temperature_implicit_update!` + `roll_solver_histories!`. The nonlinear-term computation happens INSIDE the step here (the CPU computes it earlier in `compute_solver_nonlinear_terms!`); for a single isolated scalar field this is equivalent (velocity supplied physical). The full multi-field ordering is Phase 5h. Covered for the scalar step.

**Placeholder scan:** none — complete code; exact commands + expected results. `band` helper (with `dd` for diagonal-dominant system matrices) fully defined.

**Type consistency:** `gpu_scalar_field_step!(spec_r,spec_i, prev_nl_r,prev_nl_i, u_r,u_θ,u_φ, config, d1, mvals, rinv, lin_batched, lu_batched, bc_in_r,bc_in_i, bc_out_r,bc_out_i, inv_dt, linear_weight, lmax, bw)` — consistent across the task and the interface block. Reuses `gpu_scalar_nonlinear!` (5e), `gpu_build_rhs_cnab2!` (5c: `(rr,ri, ur,ui, nr_,ni_, pr,pi_, lin, inv_dt, lw, bw)`), `gpu_implicit_solve_field!` (5d: `(x_r,x_i, lu, bc..., bw)`). Ordering: `build_rhs` reads old `spec` before the solve writes the new field; `spec .= rhs` only after; `prev_nl .= nl` last. The test's manual chain is the reference. `lin_batched`=per-l L, `lu_batched`=per-l LU of `(I−θdt·L)` — distinct, both `(2bw+1,nr,nl)`.
