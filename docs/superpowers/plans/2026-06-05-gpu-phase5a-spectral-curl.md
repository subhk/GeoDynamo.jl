# GPU Phase 5a — Batched Banded Mat-Vec & Spectral Curl Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the spectral **curl** of a toroidal–poloidal field on a single GPU — vorticity `ω = ∇×u` and current `J = ∇×B` (the *same* operator, different input fields) — via a reusable **batched banded radial mat-vec** (the `∂/∂r`, `∂²/∂r²` operators) plus element-wise `l(l+1)/r²` algebra, matching the CPU `spectral_curl_torpol!` exactly. This unblocks the velocity-advection (`u×ω`) and Lorentz (`J×B`) nonlinear terms (Phase 2 kernels consume `ω`/`J`).

**Architecture:** From the CPU map: the radial derivative matrices `d1`/`d2` are **banded `(2bw+1, nr)`, l-independent** (one matrix for all modes), applied by `apply_radial_derivative!` = a banded mat-vec (`out[i] = Σ_{|i−j|≤bw} mat[bw+1+i-j, j]·in[j]`). The curl is `dst_tor = l(l+1)/r²·P − d²P/dr² − (2/r)·dP/dr` and `dst_pol = −l(l+1)/r²·T`, where `dP/dr`/`d²P/dr²` are mat-vecs of the poloidal `P`, and `l(l+1)`/`1/r`/`1/r²` are per-mode/per-level scalars. We implement the mat-vec as a **KernelAbstractions `@kernel`** (one workitem per `(l,m)`, mirroring Phase 4's structure but accumulating `out` instead of solving) — portable, **locally testable** against `apply_radial_derivative!`. The curl assembly is two mat-vecs + element-wise broadcasts (Phase-2 style). Curl is real-linear → applied to real/imag parts independently.

**Tech Stack:** Julia, KernelAbstractions (GeoDynamo dep), broadcast. Reuses Phase-0 `on_architecture`, Phase-4's banded storage convention. No CUDA extension methods.

---

## Background (CPU reference)

- `apply_radial_derivative!(out, mat, in)` (`src/solver/numerics.jl:1026-1045`): banded mat-vec. Storage `mat.data` is `(2bw+1, N)`; `A[i,j]→mat.data[bw+1+i-j, j]`. For each output `i`, `out[i] = Σ_{j=max(1,i-bw)}^{min(N,i+bw)} mat.data[bw+1+i-j, j]·in[j]` (accumulated in ascending `j`).
- `spectral_curl_torpol!` (`numerics.jl:1383-1471`) / `compute_vorticity_spectral!` (`numerics.jl:1047-1158`):
  - `dst_tor = l_factor·r⁻²·P − d²P/dr² − 2·r⁻¹·dP/dr` (P = source poloidal).
  - `dst_pol = −l_factor·r⁻²·T` (T = source toroidal).
  - `l_factor = l(l+1)` (per mode); `r⁻¹`, `r⁻²` per radial level (`domain.r[r,3]`, `domain.r[r,2]`).
  - Vorticity = curl of velocity (T,P); current = curl of magnetic (T,P). **Same operator.**

## Testing without a local GPU

- **[LOCAL]** — the KA mat-vec runs on the CPU backend (Array); tests assert it **equals `apply_radial_derivative!`** per `(l,m)` column (exact `==` — same ascending-`j` accumulation), and the curl equals an independent reference (`apply_radial_derivative!` + the formula). Real verification.
- **[GPU-BOX]** — same on `CuArray`; guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu` (branch `feat/gpu-phase0`, on Phases 0–4). **Never pipe test runs through `tail`.**

## File Structure

- **Create** `src/gpu/spectral_curl.jl` — `gpu_batched_banded_matvec!` (KA kernel + driver), `gpu_spectral_curl!`.
- **Modify** `src/GeoDynamo.jl` — `include("gpu/spectral_curl.jl")` (after `gpu/banded_solve.jl`); export both.
- **Create** `test/gpu_phase5a_spectral_curl.jl` — `[LOCAL]` + `[GPU-BOX]` tests.
- **Modify** `test/runtests.jl` — register.

Locked interfaces:

```julia
gpu_batched_banded_matvec!(Y, X, mat, bw)   # Y[l,m,:] = mat · X[l,m,:]  (mat banded (2bw+1,nr)); Y ≠ X
gpu_spectral_curl!(dtr, dti, dpr, dpi, str, sti, spr, spi, d1, d2, lfac, rinv, rinv2, bw)
    # dst_tor = lfac·rinv2·P − d2·P − 2·rinv·d1·P ;  dst_pol = −lfac·rinv2·T   (P=src_pol, T=src_tor)
```

`X`/`Y`/`s*`/`d*` are `(nl,nm,nr)`; `d1`/`d2` are `(2bw+1,nr)`; `lfac` length-`nl` (`l(l+1)`); `rinv`/`rinv2` length-`nr`; all on the same backend. Vorticity: pass velocity (T,P); current: pass magnetic (T,P).

---

## Task 1: `gpu_batched_banded_matvec!`

**Files:** Create `src/gpu/spectral_curl.jl`; Modify `src/GeoDynamo.jl`; Test `test/gpu_phase5a_spectral_curl.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase5a_spectral_curl.jl`:

```julia
using Test
using GeoDynamo
using Random

# A banded matrix in (2bw+1, N) storage (NOT necessarily invertible — it's a derivative op).
function _rand_band_mat(::Type{T}, N, bw; seed) where {T}
    rng = MersenneTwister(seed)
    data = zeros(T, 2bw+1, N)
    for j in 1:N, i in max(1,j-bw):min(N,j+bw)
        data[bw+1+i-j, j] = rand(rng, T) - T(0.5)
    end
    return GeoDynamo.BandedMatrix{T}(data, bw, N)
end

@testset "GPU Phase 5a — Spectral Curl" begin
    @testset "batched banded mat-vec == apply_radial_derivative! [LOCAL]" begin
        N, bw, nl, nm = 10, 2, 3, 2
        mat = _rand_band_mat(Float64, N, bw; seed = 1)
        X = rand(MersenneTwister(2), Float64, nl, nm, N)
        Y = zeros(Float64, nl, nm, N)
        GeoDynamo.gpu_batched_banded_matvec!(Y, X, mat.data, bw)
        for l in 1:nl, m in 1:nm
            ref = zeros(Float64, N)
            GeoDynamo.apply_radial_derivative!(ref, mat, collect(X[l, m, :]))
            @test Y[l, m, :] == ref
        end
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5a_spectral_curl.jl")'`
Expected: FAIL — `gpu_batched_banded_matvec!` undefined.

- [ ] **Step 3: Implement**

Create `src/gpu/spectral_curl.jl`:

```julia
# =============================================================================
# GPU Phase 5a — batched banded radial mat-vec + spectral curl (vorticity ω=∇×u,
# current J=∇×B; the SAME operator on velocity vs magnetic (T,P)).  The derivative
# matrices d1=∂/∂r, d2=∂²/∂r² are banded (2bw+1,nr), l-independent.  A KA kernel
# applies one to each mode's radial profile (mirrors apply_radial_derivative!,
# numerics.jl:1026-1045); the curl is then 2 mat-vecs + element-wise l(l+1)/r terms.
# Runs on Array (locally testable) and CuArray.  Curl is real-linear → real/imag
# handled independently.
# =============================================================================

# One workitem per (l,m). Y[li,mi,i] = Σ_{j∈[max(1,i-bw),min(nr,i+bw)]} mat[bw+1+i-j,j]·X[li,mi,j].
# Same ascending-j accumulation as apply_radial_derivative! (so exact == on CPU). Y ≠ X.
@kernel function _banded_matvec_kernel!(Y, @Const(X), @Const(mat), bw::Int, nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(Y)
    @inbounds for i in 1:nr
        s = zero(T)
        for j in max(1, i - bw):min(nr, i + bw)
            s += mat[bw + 1 + i - j, j] * X[li, mi, j]
        end
        Y[li, mi, i] = s
    end
end

"""
    gpu_batched_banded_matvec!(Y, X, mat, bw) -> Y

Apply the banded radial operator `mat` (`(2bw+1,nr)`) to every mode's radial
profile: `Y[l,m,:] = mat · X[l,m,:]`.  `Y`/`X` are `(nl,nm,nr)`.  `Y` must NOT
alias `X` (an output point reads input points at other radii).  Backend inferred
from `Y`.
"""
function gpu_batched_banded_matvec!(Y, X, mat, bw::Int)
    nl, nm, nr = size(Y)
    backend = KernelAbstractions.get_backend(Y)
    _banded_matvec_kernel!(backend)(Y, X, mat, bw, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)
    return Y
end
```

- [ ] **Step 4: Include + export**

In `src/GeoDynamo.jl`, after `include("gpu/banded_solve.jl")` add `include("gpu/spectral_curl.jl")`. Add export:
```julia
export gpu_batched_banded_matvec!, gpu_spectral_curl!
```

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5a_spectral_curl.jl")'`
Expected: PASS — every column equals `apply_radial_derivative!`.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/spectral_curl.jl src/GeoDynamo.jl test/gpu_phase5a_spectral_curl.jl
git commit -m "feat(gpu): batched banded radial mat-vec kernel (Phase 5a)"
```

---

## Task 2: `gpu_spectral_curl!` (vorticity / current)

**Files:** Modify `src/gpu/spectral_curl.jl`; Test `test/gpu_phase5a_spectral_curl.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase5a_spectral_curl.jl`:

```julia
@testset "spectral curl == apply_radial_derivative! + formula [LOCAL]" begin
    N, bw, nl, nm = 10, 2, 4, 3
    d1 = _rand_band_mat(Float64, N, bw; seed = 11)
    d2 = _rand_band_mat(Float64, N, bw; seed = 12)
    rng = MersenneTwister(13)
    str = rand(rng, nl, nm, N); sti = rand(rng, nl, nm, N)   # source toroidal (T)
    spr = rand(rng, nl, nm, N); spi = rand(rng, nl, nm, N)   # source poloidal (P)
    lfac = Float64[l * (l + 1) for l in 0:(nl - 1)]
    rinv = [1.0 / (0.5 + 0.1k) for k in 1:N]
    rinv2 = rinv .^ 2
    dtr = zeros(nl,nm,N); dti = zeros(nl,nm,N); dpr = zeros(nl,nm,N); dpi = zeros(nl,nm,N)
    GeoDynamo.gpu_spectral_curl!(dtr, dti, dpr, dpi, str, sti, spr, spi, d1.data, d2.data, lfac, rinv, rinv2, bw)
    # independent reference per (l,m): d1·P, d2·P via apply_radial_derivative!, then the formula
    for l in 1:nl, m in 1:nm
        d1P = zeros(N); d2P = zeros(N)
        GeoDynamo.apply_radial_derivative!(d1P, d1, collect(spr[l,m,:]))
        GeoDynamo.apply_radial_derivative!(d2P, d2, collect(spr[l,m,:]))
        for r in 1:N
            tor_ref = lfac[l]*rinv2[r]*spr[l,m,r] - d2P[r] - 2.0*rinv[r]*d1P[r]
            pol_ref = -lfac[l]*rinv2[r]*str[l,m,r]
            @test dtr[l,m,r] == tor_ref
            @test dpr[l,m,r] == pol_ref
        end
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5a_spectral_curl.jl")'`
Expected: FAIL — `gpu_spectral_curl!` undefined.

- [ ] **Step 3: Implement**

Append to `src/gpu/spectral_curl.jl`:

```julia
"""
    gpu_spectral_curl!(dst_tor_r, dst_tor_i, dst_pol_r, dst_pol_i,
                       src_tor_r, src_tor_i, src_pol_r, src_pol_i,
                       d1, d2, lfac, rinv, rinv2, bw) -> nothing

Spectral curl of a toroidal–poloidal field (vorticity `∇×u` from velocity, or
current `∇×B` from magnetic — the same operator):
  `dst_tor = lfac·rinv2·P − d2·P − 2·rinv·d1·P`,  `dst_pol = −lfac·rinv2·T`,
with `P`=`src_pol`, `T`=`src_tor`.  `lfac[l+1]=l(l+1)` (length `nl`); `rinv`/`rinv2`
length `nr`; `d1`/`d2` banded `(2bw+1,nr)`.  All arrays on the same backend.
Real/imag handled independently (curl is real-linear).
"""
function gpu_spectral_curl!(dst_tor_r, dst_tor_i, dst_pol_r, dst_pol_i,
        src_tor_r, src_tor_i, src_pol_r, src_pol_i, d1, d2, lfac, rinv, rinv2, bw::Int)
    d1Pr = similar(src_pol_r); d1Pi = similar(src_pol_i)
    d2Pr = similar(src_pol_r); d2Pi = similar(src_pol_i)
    gpu_batched_banded_matvec!(d1Pr, src_pol_r, d1, bw)
    gpu_batched_banded_matvec!(d1Pi, src_pol_i, d1, bw)
    gpu_batched_banded_matvec!(d2Pr, src_pol_r, d2, bw)
    gpu_batched_banded_matvec!(d2Pi, src_pol_i, d2, bw)
    lf  = reshape(lfac, :, 1, 1)
    ri  = reshape(rinv, 1, 1, :)
    ri2 = reshape(rinv2, 1, 1, :)
    @. dst_tor_r = lf * ri2 * src_pol_r - d2Pr - 2.0 * ri * d1Pr
    @. dst_tor_i = lf * ri2 * src_pol_i - d2Pi - 2.0 * ri * d1Pi
    @. dst_pol_r = -lf * ri2 * src_tor_r
    @. dst_pol_i = -lf * ri2 * src_tor_i
    return nothing
end
```

- [ ] **Step 4: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5a_spectral_curl.jl")'`
Expected: PASS — toroidal/poloidal curl match the independent reference exactly.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/spectral_curl.jl test/gpu_phase5a_spectral_curl.jl
git commit -m "feat(gpu): gpu_spectral_curl! (vorticity/current) (Phase 5a)"
```

---

## Task 3: GPU-box gate + register + regression

**Files:** Test `test/gpu_phase5a_spectral_curl.jl`, `test/runtests.jl`

- [ ] **Step 1: Add the GPU-box gate** `[GPU-BOX]`

Add to `test/gpu_phase5a_spectral_curl.jl`:

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-5a gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        N, bw, nl, nm = 12, 2, 4, 3
        d1 = _rand_band_mat(Float64, N, bw; seed = 21); d2 = _rand_band_mat(Float64, N, bw; seed = 22)
        rng = MersenneTwister(23)
        str = rand(rng, nl,nm,N); sti = rand(rng, nl,nm,N); spr = rand(rng, nl,nm,N); spi = rand(rng, nl,nm,N)
        lfac = Float64[l*(l+1) for l in 0:(nl-1)]; rinv = [1.0/(0.5+0.1k) for k in 1:N]; rinv2 = rinv.^2
        # CPU reference
        z() = zeros(Float64, nl,nm,N)
        cdtr,cdti,cdpr,cdpi = z(),z(),z(),z()
        GeoDynamo.gpu_spectral_curl!(cdtr,cdti,cdpr,cdpi, str,sti,spr,spi, d1.data,d2.data, lfac,rinv,rinv2, bw)
        # GPU
        d(x) = GeoDynamo.on_architecture(GPU(), x)
        gdtr,gdti,gdpr,gdpi = d(z()),d(z()),d(z()),d(z())
        GeoDynamo.gpu_spectral_curl!(gdtr,gdti,gdpr,gdpi, d(str),d(sti),d(spr),d(spi),
                                     d(d1.data),d(d2.data), d(lfac),d(rinv),d(rinv2), bw)
        @test gdtr isa CUDA.CuArray
        @test isapprox(Array(gdtr), cdtr; atol = 1e-12, rtol = 1e-10)
        @test isapprox(Array(gdpr), cdpr; atol = 1e-12, rtol = 1e-10)
    end
end
```

- [ ] **Step 2: Run locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5a_spectral_curl.jl")'`
Expected: `[LOCAL]` testsets pass; the gate skips.

- [ ] **Step 3: Register**

In `test/runtests.jl`, add `"gpu_phase5a_spectral_curl.jl"` (next to the Phase 4 entry).

- [ ] **Step 4: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI; include("test/gpu_phase4_banded_solve.jl")'` (then separately) `… include("test/allocation_runtime_checks.jl")`
Expected: Phase 4 green; allocation guards 39/39.

- [ ] **Step 5: Commit**

```bash
git add test/gpu_phase5a_spectral_curl.jl test/runtests.jl
git commit -m "test(gpu): Phase-5a GPU-box gate + register spectral curl"
```

---

## GPU-box validation handoff

On the GPU box:
```julia
using CUDA, Test, GeoDynamo, Random
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase5a_spectral_curl.jl")    # the [GPU-BOX] gate must PASS
```
**Phase 5a passes when:** the batched mat-vec and the curl on `CuArray` match the CPU result to ~1e-12. Report any failure (a KA mat-vec index surprise, or a curl broadcast/reshape issue) before the next increment.

---

## What this unblocks / what's next

This completes the **deferred curls**. The remaining GPU port pieces:
- **Phase 5b — scalar gradient `∇θ`** (radial `∂θ/∂r` via this mat-vec; θ-component via the Legendre `(l±1,m)` recurrence — a neighbor-coupled kernel; φ-component `i·m·θ` element-wise; ×`1/r` geometric). Unblocks scalar advection.
- **Phase 5c — RHS assembly (CNAB2) + the full GPU `solver_step!`** (transform → curl/gradient → nonlinear → RHS → batched solve → update → BCs) + the GPU≈CPU full-step gate.
- **Phase 6 — `run!`/`Simulation` device-resident loop + IO host-gather.**

---

## Self-Review

**Spec coverage:** the design-doc Phase 3 phrase "v_r/curl assembly" deferred the curls; the Phase-2 map deferred ω/J. This plan delivers them: the reusable batched banded mat-vec (Task 1) + the curl operator serving BOTH vorticity and current (Task 2, the same `spectral_curl_torpol!` formula), GPU gate + regression (Task 3). The scalar gradient + RHS + full step are explicitly the next increments. Covered for this increment.

**Placeholder scan:** none — every code step has complete code; every run step has the exact command + expected result. `_rand_band_mat` test helper fully defined.

**Type consistency:** `gpu_batched_banded_matvec!(Y, X, mat, bw)`, `gpu_spectral_curl!(dtr,dti,dpr,dpi, str,sti,spr,spi, d1,d2, lfac,rinv,rinv2, bw)` — consistent across tasks and the interface block. Mat-vec kernel mirrors `apply_radial_derivative!` (band index `bw+1+i-j`, ascending-`j` accumulation → exact `==`). Curl formula matches `spectral_curl_torpol!` (`dst_tor=lfac·r⁻²·P − d²P − 2r⁻¹·d¹P`, `dst_pol=−lfac·r⁻²·T`). Reuses `GeoDynamo.BandedMatrix`/`apply_radial_derivative!` (test reference) + Phase-0 `on_architecture`. `lfac` `(nl,1,1)`, `rinv`/`rinv2` `(1,1,nr)` reshape axes. `Y ≠ X` documented (mat-vec reads other-radius inputs — not in-place safe, unlike the solve).
