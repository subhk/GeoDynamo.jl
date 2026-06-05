# GPU Phase 5j — Velocity Poloidal Influence-Matrix 2×2 Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the post-solve 2×2 influence-matrix (Green's-function) endpoint correction for poloidal velocity to a GPU kernel, bit-exact against the CPU reference.

**Architecture:** A new file `src/gpu/influence_correction.jl` with (1) a host-side packer that flattens the per-degree `ERK2InfluenceOp` (`Gre` `(nr×2)`, `invG` `(2×2)`) into batched device arrays `Gre_b` `(nr,2,nl)` / `invG_b` `(2,2,nl)` (zeros for degrees with no correction → exact no-op), (2) a KernelAbstractions kernel `_influence_correction_kernel!` (one workitem per `(l,m)` mode; reads the two endpoints into registers, forms the two correction coefficients, subtracts the rank-2 Green's combination along radius), and (3) a driver `gpu_velocity_poloidal_influence_correction!` that applies it to real and imag parts. Runs on the CPU backend (Array, locally testable) and CUDA (CuArray).

**Tech Stack:** Julia, KernelAbstractions (already a GeoDynamo dep), the existing `on_architecture`/`arch` helpers in `src/core/architecture.jl`.

---

## Background — the CPU reference (read, do not modify)

`src/timestep/erk2.jl:1795-1813` — the per-mode correction:

```julia
function apply_solver_influence_matrix_correction!(
        result::AbstractVector{T}, influence::ERK2InfluenceOp{T},
        bc_inner_val::T = zero(T), bc_outer_val::T = zero(T)) where {T}
    nr = length(result)
    delta_inner = result[1]  - bc_inner_val
    delta_outer = result[nr] - bc_outer_val
    c1 = influence.invG[1,1]*delta_inner + influence.invG[1,2]*delta_outer
    c2 = influence.invG[2,1]*delta_inner + influence.invG[2,2]*delta_outer
    @inbounds for i in 1:nr
        result[i] -= c1*influence.Gre[i,1] + c2*influence.Gre[i,2]
    end
    return result
end
```

`src/timestep/erk2.jl:1821-1861` — applied to every local mode with `l ≥ 1` that has an influence op; `bc_inner_val = bc_outer_val = 0`; real and imag treated identically. The struct (`src/solver/state.jl:61-65`):

```julia
struct ERK2InfluenceOp{T}
    Gre::Matrix{T}   # (nr × 2)
    invG::Matrix{T}  # (2 × 2)
    l::Int
end
```

**GPU layout mapping:** the dense GPU spectral field is `(nl,nm,nr)` = `(lmax+1, mmax+1, nr)`; dim-1 slot `li` ↔ degree `l = li-1`, dim-2 slot `mi` ↔ order `m = mi-1`. The correction depends only on the degree, so all `m`-slots of a given `li` use the same `Gre`/`invG`. Empty (`m>l`) modes hold zeros → `delta=0` → zero correction (matches CPU, which never touches them). Degrees with no op (incl `l=0`) get zero `Gre`/`invG` in the packed arrays → exact no-op.

**Aliasing safety:** the correction is in-place on `result`. Each workitem owns one `(li,mi)` radial column entirely; it reads the two endpoints into registers (`di`,`do_`) and forms `c1`,`c2` BEFORE writing any element — so the later writes to `result[…,1]`/`result[…,nr]` cannot corrupt the coefficients. Single pass per workitem; no cross-thread coordination. This mirrors the `_banded_solve_kernel!` discipline in `src/gpu/banded_solve.jl`.

---

## Task 1: Packer + kernel + driver

**Files:**
- Create: `src/gpu/influence_correction.jl`
- Modify: `src/GeoDynamo.jl` (add `include("gpu/influence_correction.jl")` after the `gpu/magnetic_nonlinear.jl` include at line 543; add an `export` line near the other `gpu_*` exports ~line 496)
- Test: `test/gpu_phase5j_influence_correction.jl`

- [ ] **Step 1: Write the failing test**

Create `test/gpu_phase5j_influence_correction.jl`:

```julia
using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5j — Velocity Poloidal Influence Correction (2×2)" begin
    nl, nm, nr = 5, 4, 6          # degrees 0..4, orders 0..3
    bw_unused = 0                  # not a banded op; kept for symmetry of mental model
    rng = MersenneTwister(5)

    # Per-degree influence ops for degrees 1,2,3 (NOT 0, NOT 4 → those stay no-op).
    influence = Dict{Int, GeoDynamo.ERK2InfluenceOp{Float64}}()
    for l in (1, 2, 3)
        Gre  = rand(rng, nr, 2) .- 0.5
        invG = rand(rng, 2, 2) .- 0.5
        influence[l] = GeoDynamo.ERK2InfluenceOp{Float64}(Gre, invG, l)
    end

    # Random dense spectral field (real + imag), all slots filled.
    x_r0 = rand(rng, nl, nm, nr) .- 0.5
    x_i0 = rand(rng, nl, nm, nr) .- 0.5

    # CPU reference: per (li,mi) the degree is li-1; apply the op if present, else leave.
    function cpu_reference(x0)
        x = copy(x0)
        tmp = Vector{Float64}(undef, nr)
        for li in 1:nl, mi in 1:nm
            l = li - 1
            haskey(influence, l) || continue
            for ir in 1:nr; tmp[ir] = x[li, mi, ir]; end
            GeoDynamo.apply_solver_influence_matrix_correction!(tmp, influence[l], 0.0, 0.0)
            for ir in 1:nr; x[li, mi, ir] = tmp[ir]; end
        end
        return x
    end
    ref_r = cpu_reference(x_r0)
    ref_i = cpu_reference(x_i0)

    @testset "packer shape + zero-fill for missing degrees [LOCAL]" begin
        Gre_b, invG_b = GeoDynamo.gpu_pack_influence(influence, nl, nr, CPU())
        @test size(Gre_b) == (nr, 2, nl)
        @test size(invG_b) == (2, 2, nl)
        # degree 0 (slot 1) and degree 4 (slot 5) have no op → all-zero packed columns
        @test all(==(0.0), Gre_b[:, :, 1]) && all(==(0.0), invG_b[:, :, 1])
        @test all(==(0.0), Gre_b[:, :, 5]) && all(==(0.0), invG_b[:, :, 5])
        # degree 2 (slot 3) carries the op's data exactly
        @test Gre_b[:, :, 3]  == influence[2].Gre
        @test invG_b[:, :, 3] == influence[2].invG
    end

    @testset "correction == CPU reference (exact) [LOCAL]" begin
        Gre_b, invG_b = GeoDynamo.gpu_pack_influence(influence, nl, nr, CPU())
        x_r = copy(x_r0); x_i = copy(x_i0)
        GeoDynamo.gpu_velocity_poloidal_influence_correction!(x_r, x_i, Gre_b, invG_b)
        @test x_r == ref_r
        @test x_i == ref_i
        @test all(isfinite, x_r) && all(isfinite, x_i)
    end

    @testset "missing-degree modes are untouched [LOCAL]" begin
        Gre_b, invG_b = GeoDynamo.gpu_pack_influence(influence, nl, nr, CPU())
        x_r = copy(x_r0); x_i = copy(x_i0)
        GeoDynamo.gpu_velocity_poloidal_influence_correction!(x_r, x_i, Gre_b, invG_b)
        # degree 0 (slot 1) and degree 4 (slot 5) unchanged
        @test x_r[1, :, :] == x_r0[1, :, :]
        @test x_r[5, :, :] == x_r0[5, :, :]
        @test x_i[1, :, :] == x_i0[1, :, :]
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5j gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            Gre_b, invG_b = GeoDynamo.gpu_pack_influence(influence, nl, nr, GPU())
            gx_r = d(copy(x_r0)); gx_i = d(copy(x_i0))
            GeoDynamo.gpu_velocity_poloidal_influence_correction!(gx_r, gx_i, Gre_b, invG_b)
            @test gx_r isa CUDA.CuArray
            @test isapprox(Array(gx_r), ref_r; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gx_i), ref_i; atol = 1e-12, rtol = 1e-10)
        end
    end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5j_influence_correction.jl")'
```
Expected: FAIL — `UndefVarError: gpu_pack_influence` / `gpu_velocity_poloidal_influence_correction!` not defined.

- [ ] **Step 3: Write the implementation**

Create `src/gpu/influence_correction.jl`:

```julia
# =============================================================================
# GPU Phase 5j — velocity poloidal influence-matrix 2×2 (Green's-function)
# endpoint correction. Post-solve, projects each poloidal radial profile back
# onto zero endpoints using a precomputed two-column influence operator per
# degree l. Mirrors apply_solver_influence_matrix_correction! (erk2.jl:1795) +
# apply_solver_velocity_poloidal_influence_correction! (erk2.jl:1821). The
# per-degree Gre (nr×2) / invG (2×2) are packed host-side into batched arrays;
# the KA kernel does one rank-2 subtract per (l,m) mode. Runs on Array + CuArray.
# =============================================================================

"""
    gpu_pack_influence(influence, nl, nr, arch) -> (Gre_b, invG_b)

Flatten the per-degree `ERK2InfluenceOp` correction operators into batched arrays
on `arch`'s backend: `Gre_b` is `(nr,2,nl)` and `invG_b` is `(2,2,nl)`, indexed by
dim-3 = dense degree slot `li` (degree `l = li-1`).  Degrees absent from
`influence` (including `l=0`) get all-zero columns, so the kernel applies an exact
no-op to those modes — matching the CPU path, which skips them.

`influence` is the `Dict{Int,ERK2InfluenceOp{T}}` keyed by degree `l` (0-based).
`nl` is the number of degree slots (`lmax+1`); `nr` the radial size.
"""
function gpu_pack_influence(influence::AbstractDict{Int, ERK2InfluenceOp{T}},
        nl::Int, nr::Int, arch::AbstractArchitecture) where {T}
    Gre_b = zeros(T, nr, 2, nl)
    invG_b = zeros(T, 2, 2, nl)
    for (l, op) in influence
        slot = l + 1                      # degree l (0-based) → dim-3 slot
        (1 <= slot <= nl) || continue
        size(op.Gre, 1) == nr ||
            throw(ArgumentError("gpu_pack_influence: Gre has $(size(op.Gre,1)) rows, expected nr=$nr"))
        Gre_b[:, :, slot] .= op.Gre
        invG_b[:, :, slot] .= op.invG
    end
    return on_architecture(arch, Gre_b), on_architecture(arch, invG_b)
end

# One workitem per (l,m) mode. Reads the two endpoints into registers, forms the
# two correction coefficients from the degree's 2×2 invG, then subtracts the
# rank-2 Green's combination along radius. Mirrors apply_solver_influence_matrix_
# correction! (erk2.jl:1808) with bc_inner=bc_outer=0.
#
# Aliasing: di/do_ and c1/c2 are captured in registers BEFORE the write loop, so
# the in-place writes to R[li,mi,1] and R[li,mi,nr] cannot corrupt the
# coefficients. Each workitem owns its full radial column → no cross-thread races.
@kernel function _influence_correction_kernel!(R, @Const(Gre_b), @Const(invG_b), nr::Int)
    li, mi = @index(Global, NTuple)
    @inbounds begin
        di  = R[li, mi, 1]            # delta_inner (bc_inner = 0)
        do_ = R[li, mi, nr]           # delta_outer (bc_outer = 0)
        c1 = invG_b[1, 1, li] * di + invG_b[1, 2, li] * do_
        c2 = invG_b[2, 1, li] * di + invG_b[2, 2, li] * do_
        for i in 1:nr
            R[li, mi, i] -= c1 * Gre_b[i, 1, li] + c2 * Gre_b[i, 2, li]
        end
    end
end

"""
    gpu_velocity_poloidal_influence_correction!(x_r, x_i, Gre_b, invG_b) -> nothing

Apply the velocity-poloidal endpoint influence correction in-place to the real
(`x_r`) and imaginary (`x_i`) parts of a dense `(nl,nm,nr)` spectral field, using
the batched operators from [`gpu_pack_influence`](@ref).  Backend (CPU/CUDA) is
inferred from `x_r`; `x_r`, `x_i`, `Gre_b`, `invG_b` must all be on the same backend.
"""
function gpu_velocity_poloidal_influence_correction!(x_r, x_i, Gre_b, invG_b)
    nl, nm, nr = size(x_r)
    backend = KernelAbstractions.get_backend(x_r)
    _influence_correction_kernel!(backend)(x_r, Gre_b, invG_b, nr; ndrange = (nl, nm))
    _influence_correction_kernel!(backend)(x_i, Gre_b, invG_b, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)  # eager sync; Phase-5n may hoist for pipelining
    return nothing
end
```

Modify `src/GeoDynamo.jl` — add the include after `include("gpu/magnetic_nonlinear.jl")` (line 543):

```julia
include("gpu/influence_correction.jl")
```

And add the export near the other `gpu_*` exports (after `export gpu_magnetic_nonlinear!`, ~line 496):

```julia
export gpu_pack_influence, gpu_velocity_poloidal_influence_correction!
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5j_influence_correction.jl")'
```
Expected: the three `[LOCAL]` testsets PASS (packer shape/zero-fill, exact `==`, missing-degree untouched); the `[GPU-BOX]` testset shows 1 Broken (`@test_skip`) on this Apple-Silicon box.

- [ ] **Step 5: Verify the module still loads**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; println("LOAD OK")'
```
Expected: `LOAD OK`.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/influence_correction.jl src/GeoDynamo.jl test/gpu_phase5j_influence_correction.jl
git commit -m "feat(gpu): Phase 5j velocity poloidal influence-matrix 2×2 correction

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Register the test + regression check

**Files:**
- Modify: `test/runtests.jl` (add the Phase 5j entry after the Phase 5i entry)

- [ ] **Step 1: Add the test to the suite**

In `test/runtests.jl`, find the line that includes `"gpu_phase5i_coupled_velocity.jl"` and add immediately after it (same construct/indentation as the surrounding `gpu_phase5*` entries):

```julia
        "gpu_phase5j_influence_correction.jl",
```

- [ ] **Step 2: Confirm the new test still passes in isolation**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5j_influence_correction.jl")' > /tmp/phase5j.log 2>&1; echo "exit=$?"; tail -25 /tmp/phase5j.log
```
Expected: `exit=0`, the three `[LOCAL]` testsets pass, 1 Broken for the GPU-box gate.

- [ ] **Step 3: Confirm the allocation guards still pass**

Run (these guard the documented hot-path allocation budget; Phase 5j adds no hot-path CPU change):
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/allocation_guards.jl")' > /tmp/allocguards.log 2>&1; echo "exit=$?"; tail -8 /tmp/allocguards.log
```
Expected: `exit=0`, 39/39 (unchanged). (If the file name differs, locate it with `grep -rl "alloc" test/ | head`.)

- [ ] **Step 4: Commit**

```bash
git add test/runtests.jl
git commit -m "test(gpu): register Phase 5j influence-correction in suite

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** packer flattens `Dict{Int,ERK2InfluenceOp}` → batched `(nr,2,nl)`/`(2,2,nl)` with zero-fill for missing/`l=0` degrees ✓; kernel reproduces the exact CPU formula (`delta = endpoint − 0`; `c = invG·delta`; `result -= c1·Gre[:,1] + c2·Gre[:,2]`) ✓; driver applies to real + imag ✓; degree→slot mapping `l → l+1` matches the dense layout ✓; runs on Array + CuArray via `KernelAbstractions.get_backend` ✓.

**Placeholder scan:** none — every step has full code/commands.

**Type consistency:** `gpu_pack_influence` / `gpu_velocity_poloidal_influence_correction!` named identically across tasks; `ERK2InfluenceOp{T}` fields `Gre`/`invG` used as `(nr×2)`/`(2×2)`; kernel arg order `(R, Gre_b, invG_b, nr)` matches the driver's launch.

**Aliasing:** endpoints captured in registers before the in-place write loop — documented in the kernel comment, mirrors the `_banded_solve_kernel!` rule.
