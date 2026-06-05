# GPU Phase 5c — CNAB2 RHS Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the CNAB2 implicit right-hand side on a single GPU — `rhs = inv_dt·u + 1.5·nl − 0.5·nl_prev + (1−θ)·L·u`, where `L·u` is a **per-degree-`l`** banded mat-vec (the linear diffusion operator) — matching the CPU `build_rhs_cnab2!` exactly. This is the first piece of the full-step integration (Phase 5c-step assembles the rest).

**Architecture:** From the CPU (`src/timestep/implicit.jl:179-256`): for each mode `(l,m)`, `rhs = inv_dt·u + 1.5·nl − 0.5·nl_prev + linear_weight·(L_l·u)`, with `inv_dt = mass_coeff/dt`, `linear_weight = 1−θ` (θ=0.5), and `L_l·u` a banded mat-vec using the **per-l** linear matrix (`linear_matrices[l]`, includes diffusivity + `l(l+1)/r²` + radial Laplacian). `apply_banded_full!` (`numerics.jl:1765`) is the same banded mat-vec as Phase 5a's `apply_radial_derivative!` but with a per-l matrix. So Phase 5c-RHS = a **per-l batched banded mat-vec** (new — like Phase 5a's but the matrix is indexed by `l`) + the element-wise CNAB2 combination (broadcast). Both are KA/broadcast → **locally testable** against `apply_banded_full!` + the formula.

**Tech Stack:** Julia, KernelAbstractions (per-l mat-vec kernel), broadcast (the combination). Reuses Phase-0 `on_architecture`, the banded-storage convention. No CUDA extension methods.

---

## Background (CPU reference — `src/timestep/implicit.jl:216-252`)

```
inv_dt = mass_coeff/dt ;  three_halves = 1.5 ;  half = 0.5
linear_weight = 1 − θ  (θ = matrices.theta = 0.5)
per mode (l,m), per r:
  value = inv_dt·u[r] + 1.5·nl[r] − 0.5·nl_prev[r]
  if linear_weight ≠ 0:  value += linear_weight · (L_l·u)[r]
```
`L_l·u = apply_banded_full!(linear_matrices[l], u)` (banded mat-vec; `out[i]=Σ_{|i-j|≤bw} mat[bw+1+i-j,j]·u[j]`, ascending j). Real and imag assembled identically.

## Testing without a local GPU

- **[LOCAL]** — the per-l mat-vec runs on Array; tests assert it **equals `apply_banded_full!`** per `(l,m)` column (exact `==`), and the RHS equals the formula reference (exact `==`). Real verification.
- **[GPU-BOX]** — same on `CuArray`; guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu` (branch `feat/gpu-phase0`, on Phases 0–5b). **Never pipe test runs through `tail`.**

## File Structure

- **Create** `src/gpu/cnab2_rhs.jl` — `gpu_batched_banded_matvec_perl!` (KA kernel + driver), `gpu_build_rhs_cnab2!`.
- **Modify** `src/GeoDynamo.jl` — `include("gpu/cnab2_rhs.jl")` (after `gpu/scalar_gradient.jl`); export both.
- **Create** `test/gpu_phase5c_cnab2_rhs.jl` — `[LOCAL]` + `[GPU-BOX]` tests.
- **Modify** `test/runtests.jl` — register.

Locked interfaces:

```julia
gpu_batched_banded_matvec_perl!(Y, X, mat_batched, bw)   # Y[l,m,:] = mat_batched[:,:,l] · X[l,m,:]  (per-l matrix); Y ≠ X
gpu_build_rhs_cnab2!(rr, ri, ur, ui, nr_, ni_, pr, pi, lin_batched, inv_dt, linear_weight, bw)
    # rhs = inv_dt·u + 1.5·nl − 0.5·nl_prev + linear_weight·(lin·u)
```

`Y`/`X`/`r*`/`u*`/`n*`/`p*` are `(nl,nm,nr)`; `mat_batched`/`lin_batched` are `(2bw+1,nr,nl)` (degree `l` = dim-3 slice `[:,:,l]`, l-slot index); `inv_dt`/`linear_weight` scalars; same backend; outputs distinct from inputs.

---

## Task 1: `gpu_batched_banded_matvec_perl!` (per-l mat-vec)

**Files:** Create `src/gpu/cnab2_rhs.jl`; Modify `src/GeoDynamo.jl`; Test `test/gpu_phase5c_cnab2_rhs.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase5c_cnab2_rhs.jl`:

```julia
using Test
using GeoDynamo
using Random

# per-l banded matrices stacked into (2bw+1, nr, nl)
function _band(::Type{T}, N, bw; seed) where {T}
    rng = MersenneTwister(seed); d = zeros(T, 2bw+1, N)
    for j in 1:N, i in max(1,j-bw):min(N,j+bw); d[bw+1+i-j,j] = rand(rng,T)-T(0.5); end
    GeoDynamo.BandedMatrix{T}(d, bw, N)
end

@testset "GPU Phase 5c — CNAB2 RHS" begin
    @testset "per-l batched mat-vec == apply_banded_full! [LOCAL]" begin
        N, bw, nl, nm = 10, 2, 4, 3
        mats = [_band(Float64, N, bw; seed = 50 + l) for l in 1:nl]
        matb = zeros(Float64, 2bw+1, N, nl)
        for l in 1:nl; matb[:,:,l] .= mats[l].data; end
        X = rand(MersenneTwister(60), nl, nm, N)
        Y = zeros(nl, nm, N)
        GeoDynamo.gpu_batched_banded_matvec_perl!(Y, X, matb, bw)
        for l in 1:nl, m in 1:nm
            ref = zeros(N)
            GeoDynamo.apply_banded_full!(ref, mats[l], collect(X[l,m,:]))
            @test Y[l,m,:] == ref
        end
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5c_cnab2_rhs.jl")'`
Expected: FAIL — `gpu_batched_banded_matvec_perl!` undefined.

- [ ] **Step 3: Implement**

Create `src/gpu/cnab2_rhs.jl`:

```julia
# =============================================================================
# GPU Phase 5c — CNAB2 implicit RHS assembly.  The linear term L_l·u uses a
# PER-DEGREE-l banded matrix (linear_matrices[l]), so this needs a per-l batched
# mat-vec (like Phase 5a's, but the matrix is indexed by l = dim-3 slice).
# Mirrors apply_banded_full! (numerics.jl:1765) + build_rhs_cnab2! (implicit.jl:216).
# KA + broadcast → runs on Array (locally testable) and CuArray.
# =============================================================================

# One workitem per (l,m). Y[li,mi,i] = Σ_{j∈[max(1,i-bw),min(nr,i+bw)]}
# mat_batched[bw+1+i-j, j, li] · X[li,mi,j].  Same ascending-j accumulation as
# apply_banded_full! → exact == on CPU.  Y ≠ X.
@kernel function _perl_matvec_kernel!(Y, @Const(X), @Const(mat_batched), bw::Int, nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(Y)
    @inbounds for i in 1:nr
        s = zero(T)
        for j in max(1, i - bw):min(nr, i + bw)
            s += mat_batched[bw + 1 + i - j, j, li] * X[li, mi, j]
        end
        Y[li, mi, i] = s
    end
end

"""
    gpu_batched_banded_matvec_perl!(Y, X, mat_batched, bw) -> Y

Per-degree-`l` banded mat-vec: `Y[l,m,:] = mat_batched[:,:,l] · X[l,m,:]`, where
`mat_batched` is `(2bw+1, nr, nl)` (degree `l` = dim-3 slice).  `Y`/`X` are
`(nl,nm,nr)`; `Y` must NOT alias `X`.  Backend inferred from `Y`.
"""
function gpu_batched_banded_matvec_perl!(Y, X, mat_batched, bw::Int)
    nl, nm, nr = size(Y)
    backend = KernelAbstractions.get_backend(Y)
    _perl_matvec_kernel!(backend)(Y, X, mat_batched, bw, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)  # eager sync; Phase-5c-step: hoist to caller
    return Y
end
```

- [ ] **Step 4: Include + export**

In `src/GeoDynamo.jl`, after `include("gpu/scalar_gradient.jl")` add `include("gpu/cnab2_rhs.jl")`. Add export:
```julia
export gpu_batched_banded_matvec_perl!, gpu_build_rhs_cnab2!
```

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5c_cnab2_rhs.jl")'`
Expected: PASS — every column equals `apply_banded_full!`.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/cnab2_rhs.jl src/GeoDynamo.jl test/gpu_phase5c_cnab2_rhs.jl
git commit -m "feat(gpu): per-l batched banded mat-vec kernel (Phase 5c)"
```

---

## Task 2: `gpu_build_rhs_cnab2!`

**Files:** Modify `src/gpu/cnab2_rhs.jl`; Test `test/gpu_phase5c_cnab2_rhs.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase5c_cnab2_rhs.jl`:

```julia
@testset "CNAB2 RHS == formula [LOCAL]" begin
    N, bw, nl, nm = 10, 2, 4, 3
    mats = [_band(Float64, N, bw; seed = 70 + l) for l in 1:nl]
    lin = zeros(Float64, 2bw+1, N, nl); for l in 1:nl; lin[:,:,l] .= mats[l].data; end
    rng = MersenneTwister(71)
    ur = rand(rng,nl,nm,N); ui = rand(rng,nl,nm,N)
    nr_ = rand(rng,nl,nm,N); ni_ = rand(rng,nl,nm,N)
    pr = rand(rng,nl,nm,N); pi_ = rand(rng,nl,nm,N)
    inv_dt = 1.0 / 0.01; lw = 0.5
    rr = zeros(nl,nm,N); ri = zeros(nl,nm,N)
    GeoDynamo.gpu_build_rhs_cnab2!(rr, ri, ur, ui, nr_, ni_, pr, pi_, lin, inv_dt, lw, bw)
    for l in 1:nl, m in 1:nm
        Lur = zeros(N); Lui = zeros(N)
        GeoDynamo.apply_banded_full!(Lur, mats[l], collect(ur[l,m,:]))
        GeoDynamo.apply_banded_full!(Lui, mats[l], collect(ui[l,m,:]))
        for r in 1:N
            @test rr[l,m,r] == inv_dt*ur[l,m,r] + 1.5*nr_[l,m,r] - 0.5*pr[l,m,r] + lw*Lur[r]
            @test ri[l,m,r] == inv_dt*ui[l,m,r] + 1.5*ni_[l,m,r] - 0.5*pi_[l,m,r] + lw*Lui[r]
        end
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5c_cnab2_rhs.jl")'`
Expected: FAIL — `gpu_build_rhs_cnab2!` undefined.

- [ ] **Step 3: Implement**

Append to `src/gpu/cnab2_rhs.jl`:

```julia
"""
    gpu_build_rhs_cnab2!(rr, ri, ur, ui, nr_, ni_, pr, pi_, lin_batched, inv_dt, linear_weight, bw) -> nothing

Assemble the CNAB2 implicit RHS (split real/imag):
`rhs = inv_dt·u + 1.5·nl − 0.5·nl_prev + linear_weight·(lin·u)`, where `lin·u` is the
per-l banded mat-vec of the linear operator (`lin_batched` `(2bw+1,nr,nl)`).
`inv_dt = mass_coeff/dt`, `linear_weight = 1−θ`.  All arrays `(nl,nm,nr)` on the
same backend; outputs distinct from inputs.
"""
function gpu_build_rhs_cnab2!(rr, ri, ur, ui, nr_, ni_, pr, pi_, lin_batched, inv_dt, linear_weight, bw::Int)
    Lur = similar(ur); Lui = similar(ui)
    gpu_batched_banded_matvec_perl!(Lur, ur, lin_batched, bw)
    gpu_batched_banded_matvec_perl!(Lui, ui, lin_batched, bw)
    T = eltype(rr)
    a = T(inv_dt); lw = T(linear_weight); c32 = T(1.5); c12 = T(0.5)
    @. rr = a * ur + c32 * nr_ - c12 * pr + lw * Lur
    @. ri = a * ui + c32 * ni_ - c12 * pi_ + lw * Lui
    return nothing
end
```

- [ ] **Step 4: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5c_cnab2_rhs.jl")'`
Expected: PASS — RHS matches the formula exactly. (For Float64, `T(1.5)·x == 1.5·x`, `T(inv_dt)==inv_dt`, so the `==` holds against the test's Float64 reference.)

- [ ] **Step 5: Commit**

```bash
git add src/gpu/cnab2_rhs.jl test/gpu_phase5c_cnab2_rhs.jl
git commit -m "feat(gpu): gpu_build_rhs_cnab2! (CNAB2 implicit RHS) (Phase 5c)"
```

---

## Task 3: GPU-box gate + register + regression

**Files:** Test `test/gpu_phase5c_cnab2_rhs.jl`, `test/runtests.jl`

- [ ] **Step 1: Add the GPU-box gate** `[GPU-BOX]`

Add to `test/gpu_phase5c_cnab2_rhs.jl`:

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-5c gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        N, bw, nl, nm = 12, 2, 4, 3
        mats = [_band(Float64, N, bw; seed = 80 + l) for l in 1:nl]
        lin = zeros(Float64, 2bw+1, N, nl); for l in 1:nl; lin[:,:,l] .= mats[l].data; end
        rng = MersenneTwister(81)
        ur=rand(rng,nl,nm,N); ui=rand(rng,nl,nm,N); nr_=rand(rng,nl,nm,N); ni_=rand(rng,nl,nm,N); pr=rand(rng,nl,nm,N); pi_=rand(rng,nl,nm,N)
        inv_dt = 1.0/0.01; lw = 0.5
        crr=zeros(nl,nm,N); cri=zeros(nl,nm,N)
        GeoDynamo.gpu_build_rhs_cnab2!(crr,cri, ur,ui, nr_,ni_, pr,pi_, lin, inv_dt, lw, bw)
        d(x) = GeoDynamo.on_architecture(GPU(), x)
        grr=d(zeros(nl,nm,N)); gri=d(zeros(nl,nm,N))
        GeoDynamo.gpu_build_rhs_cnab2!(grr,gri, d(ur),d(ui), d(nr_),d(ni_), d(pr),d(pi_), d(lin), inv_dt, lw, bw)
        @test grr isa CUDA.CuArray
        @test isapprox(Array(grr), crr; atol = 1e-12, rtol = 1e-10)
        @test isapprox(Array(gri), cri; atol = 1e-12, rtol = 1e-10)
    end
end
```

- [ ] **Step 2: Run locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5c_cnab2_rhs.jl")'`
Expected: `[LOCAL]` testsets pass; the gate skips.

- [ ] **Step 3: Register**

In `test/runtests.jl`, add `"gpu_phase5c_cnab2_rhs.jl"` (next to the Phase 5b entry).

- [ ] **Step 4: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5b_scalar_gradient.jl")'` then separately `… -e 'using Test, GeoDynamo, MPI; include("test/allocation_runtime_checks.jl")'`
Expected: Phase 5b green; allocation guards 39/39.

- [ ] **Step 5: Commit**

```bash
git add test/gpu_phase5c_cnab2_rhs.jl test/runtests.jl
git commit -m "test(gpu): Phase-5c GPU-box gate + register CNAB2 RHS"
```

---

## GPU-box validation handoff

On the GPU box:
```julia
using CUDA, Test, GeoDynamo, Random
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase5c_cnab2_rhs.jl")    # the [GPU-BOX] gate must PASS
```
**Phase 5c (RHS) passes when:** the per-l mat-vec and the CNAB2 RHS on `CuArray` match the CPU result to ~1e-12.

---

## What this unblocks / what's next

The CNAB2 RHS now exists on GPU. With Phases 1–5b, every per-component operation of one timestep is available on GPU: transform (1/3), curl/gradient (5a/5b), nonlinear (2), per-l mat-vec + RHS (5c), batched solve (4). Remaining:
- **Phase 5c-step — the full GPU `solver_step!` orchestration**: per field (T, C, vel-tor/pol, mag-tor/pol), drive transform → curl/gradient → nonlinear products → analyze → `gpu_build_rhs_cnab2!` → `gpu_batched_banded_solve!` → update; plus **boundary-condition application** (embedded in the per-l matrices + boundary rows — needs the BC-row assembly on GPU). The **GPU≈CPU full-step gate** lives here.
- **Phase 6 — `run!`/`Simulation` device-resident loop + IO host-gather.**

⚠️ The full-step orchestration integrates 8 unvalidated increments — a GPU-box pass of the per-increment `[GPU-BOX]` gates first would isolate any GPU-only bug before it surfaces as a full-step mismatch.

---

## Self-Review

**Spec coverage:** the timestep needs the CNAB2 RHS. This delivers it: the per-l banded mat-vec (`L_l·u`, Task 1) + the CNAB2 combination (Task 2), GPU gate + regression (Task 3). The full-step orchestration + BCs are the explicit next increment. Covered for the RHS.

**Placeholder scan:** none — every code step has complete code; every run step has the exact command + expected result. `_band` helper fully defined.

**Type consistency:** `gpu_batched_banded_matvec_perl!(Y, X, mat_batched, bw)`, `gpu_build_rhs_cnab2!(rr,ri, ur,ui, nr_,ni_, pr,pi_, lin_batched, inv_dt, linear_weight, bw)` — consistent across tasks and the interface block. Per-l mat-vec mirrors `apply_banded_full!` (`numerics.jl:1765`, ascending-j) — exact `==`. RHS matches `build_rhs_cnab2!` (`implicit.jl:238-247`): `inv_dt·u + 1.5·nl − 0.5·nl_prev + linear_weight·(lin·u)`. `lin_batched` degree-l = dim-3 slice (l-slot). Reuses `GeoDynamo.BandedMatrix`/`apply_banded_full!` (test references) + Phase-0 `on_architecture`. Outputs distinct from inputs (mat-vec reads other-radius). `1.5`/`0.5`/`inv_dt`/`linear_weight` cast to `eltype` for Float32-safety; bitwise-identical to the Float64 reference for Float64 arrays.
