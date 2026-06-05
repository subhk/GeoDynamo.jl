# GPU Phase 3 — Vector Transform (velocity / magnetic) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform a device-resident toroidal–poloidal vector field (velocity or magnetic) between its dense spectral representation (two `(lmax+1, mmax+1, nr)` scalar fields: toroidal `T`, poloidal `P`) and its physical components (`v_r, v_θ, v_φ`, each `(nlat, nlon, nr)`) on a single GPU, matching the CPU serial transform (~1e-12).

**Architecture:** The CPU vector transform (mapped from `src/solver/numerics.jl:846-966`) is **purely algebraic** — no radial derivative. Forward: feed the poloidal `P` as the spheroidal `S` and the toroidal `T` directly into `synthesis_sphtor` → tangential `(v_θ, v_φ)`; assemble `v_r` by scaling the poloidal per `(l,r)` (`P·l(l+1)/r_val` solver path; `/r_val²` MIE path) and scalar-synthesizing it. Inverse: `analysis_sphtor(v_θ, v_φ) → (S=P, T)` directly (v_r is redundant for a solenoidal field). This mirrors Phase 1's dispatch-helper pattern: a `_vector_synth_sphtor`/`_vector_anal_sphtor` helper whose `AbstractMatrix` method calls always-available `SHTnsKit.synthesis_sphtor`/`analysis_sphtor` (CPU) and whose `CuArray` method (ext-only) calls `gpu_synthesis_sphtor`/`gpu_analysis_sphtor`. So the per-level loop logic is locally testable; only `CuArray` execution is GPU-gated. The `v_r` scaling is an element-wise broadcast (Phase-2 style). Confirmed orientations: `synthesis_sphtor(cfg,S,T)→(vt,vp)` each `(nlat,nlon)` (direct, no transpose); `analysis_sphtor(cfg,vt,vp)→(S,T)` dense `(lmax+1,mmax+1)`.

**Tech Stack:** Julia, SHTnsKit (`synthesis_sphtor`/`analysis_sphtor` core; `gpu_synthesis_sphtor`/`gpu_analysis_sphtor` in `SHTnsKitGPUExt`), CUDA.jl (weakdep). Reuses Phase 1's `_scalar_synth` for v_r.

---

## Prerequisite

Builds on Phases 0–2 (branch `feat/gpu-phase0`, worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu`). Uses Phase 0's `GPUSpectralField` (dense `(lmax+1,mmax+1,nr)`) and `GPUPhysicalField` (`(nlat,nlon,nr)`), and Phase 1's `_scalar_synth`.

## Testing without a local GPU

- **[LOCAL]** — the sphtor helper's CPU method calls `SHTnsKit.synthesis_sphtor`/`analysis_sphtor` (always available), so the drivers RUN on `Array`-backed fields and assert real results (per-level parity vs direct sphtor; band-limited roundtrip). Real verification.
- **[GPU-BOX]** — `CuArray` execution via `gpu_*_sphtor`; guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from the worktree. **Never pipe test runs through `tail`.**

## File Structure

- **Create** `src/gpu/vector_transform.jl` — `_vector_synth_sphtor`/`_vector_anal_sphtor` (core) + `gpu_vr_scale!` + the two drivers.
- **Modify** `ext/GeoDynamoCUDAExt.jl` — `CuArray` methods of `_vector_synth_sphtor`/`_vector_anal_sphtor` → `gpu_synthesis_sphtor`/`gpu_analysis_sphtor`.
- **Modify** `src/GeoDynamo.jl` — `include("gpu/vector_transform.jl")` (after `gpu/nonlinear.jl`); export the drivers + `gpu_vr_scale!`.
- **Create** `test/gpu_phase3_vector_transform.jl` — `[LOCAL]` + `[GPU-BOX]` tests.
- **Modify** `test/runtests.jl` — register.

Locked interfaces:

```julia
_vector_synth_sphtor(cfg_sht, S::AbstractMatrix, T::AbstractMatrix)   # → (vt, vp)  each (nlat,nlon); ext CuArray→gpu_synthesis_sphtor
_vector_anal_sphtor(cfg_sht, vt::AbstractMatrix, vp::AbstractMatrix)  # → (S, T)    each (lmax+1,mmax+1); ext CuArray→gpu_analysis_sphtor
gpu_vr_scale!(vr_alm_r, vr_alm_i, pol_r, pol_i, lfac, rscale)          # vr_alm = pol · lfac[l] · rscale[r]  (split real/imag)
gpu_vector_spectral_to_physical!(vr::GPUPhysicalField, vθ::GPUPhysicalField, vφ::GPUPhysicalField,
                                 tor::GPUSpectralField, pol::GPUSpectralField, config, lfac, rscale)
gpu_vector_physical_to_spectral!(tor::GPUSpectralField, pol::GPUSpectralField,
                                 vθ::GPUPhysicalField, vφ::GPUPhysicalField, config)
```

`lfac` is length-`(lmax+1)` with `lfac[l+1]=l(l+1)`; `rscale` is length-`nr` (`1/r_val` solver path, `1/r_val²` MIE) — caller supplies, on the same backend as the fields. v_r is not consumed by the inverse (solenoidal).

---

## Task 1: sphtor dispatch helpers (CPU + CUDA)

**Files:** Create `src/gpu/vector_transform.jl`; Modify `ext/GeoDynamoCUDAExt.jl`, `src/GeoDynamo.jl`; Test `test/gpu_phase3_vector_transform.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase3_vector_transform.jl`:

```julia
using Test
using GeoDynamo
import SHTnsKit

@testset "GPU Phase 3 — Vector Transform" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 3)
    sht = cfg.sht_config
    nl, nm = cfg.lmax + 1, cfg.mmax + 1

    @testset "sphtor helper CPU method [LOCAL]" begin
        S = zeros(ComplexF64, nl, nm); T = zeros(ComplexF64, nl, nm)
        S[3, 1] = 1.0; T[4, 2] = 0.5 - 0.25im
        vt, vp = GeoDynamo._vector_synth_sphtor(sht, S, T)
        @test size(vt) == (cfg.nlat, cfg.nlon) && size(vp) == (cfg.nlat, cfg.nlon)
        rt, rp = SHTnsKit.synthesis_sphtor(sht, S, T; real_output = true)
        @test vt == rt && vp == rp
        S2, T2 = GeoDynamo._vector_anal_sphtor(sht, vt, vp)
        rS, rT = SHTnsKit.analysis_sphtor(sht, vt, vp)
        @test S2 == rS && T2 == rT
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase3_vector_transform.jl")'`
Expected: FAIL — `_vector_synth_sphtor` undefined.

- [ ] **Step 3: Implement the core helpers**

Create `src/gpu/vector_transform.jl`:

```julia
# =============================================================================
# GPU Phase 3 — vector (velocity/magnetic, toroidal-poloidal) transform.
# Mirrors the CPU transform (numerics.jl:846-966), which is PURELY ALGEBRAIC:
#   tangential (v_θ,v_φ) = synthesis_sphtor(S=poloidal, T=toroidal)   [no ∂/∂r]
#   radial     v_r       = scalar_synth(poloidal · l(l+1)/r)          [per-(l,r) factor]
# The AbstractMatrix methods call the always-available serial CPU sphtor transform;
# the CUDA extension adds CuArray methods → gpu_*_sphtor. v_r reuses Phase 1's
# _scalar_synth. Curls (vorticity/current) are a SEPARATE later phase — not here.
# =============================================================================

# CPU path (always available). Returns (vt, vp) / (S, T).
_vector_synth_sphtor(cfg_sht, S::AbstractMatrix, T::AbstractMatrix) =
    SHTnsKit.synthesis_sphtor(cfg_sht, S, T; real_output = true)
_vector_anal_sphtor(cfg_sht, vt::AbstractMatrix, vp::AbstractMatrix) =
    SHTnsKit.analysis_sphtor(cfg_sht, vt, vp)
```

- [ ] **Step 4: Add the CUDA override**

In `ext/GeoDynamoCUDAExt.jl`, after the Phase-1 `_scalar_*` overrides, add:

```julia
# Phase 3: route CuArray sphtor coefficient/spatial matrices through SHTnsKit's GPU transform.
GeoDynamo._vector_synth_sphtor(cfg_sht, S::CUDA.CuArray, T::CUDA.CuArray) =
    SHTnsKit.gpu_synthesis_sphtor(cfg_sht, S, T; real_output = true)
GeoDynamo._vector_anal_sphtor(cfg_sht, vt::CUDA.CuArray, vp::CUDA.CuArray) =
    SHTnsKit.gpu_analysis_sphtor(cfg_sht, vt, vp)
```

- [ ] **Step 5: Include in the module**

In `src/GeoDynamo.jl`, after `include("gpu/nonlinear.jl")` add `include("gpu/vector_transform.jl")`.

- [ ] **Step 6: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase3_vector_transform.jl")'`
Expected: PASS — helper equals a direct `SHTnsKit.synthesis_sphtor`/`analysis_sphtor`.

- [ ] **Step 7: Commit**

```bash
git add src/gpu/vector_transform.jl ext/GeoDynamoCUDAExt.jl src/GeoDynamo.jl test/gpu_phase3_vector_transform.jl
git commit -m "feat(gpu): sphtor vector transform dispatch helper (CPU + CUDA) (Phase 3)"
```

---

## Task 2: `gpu_vr_scale!` (poloidal → v_r coefficients)

**Files:** Modify `src/gpu/vector_transform.jl`, `src/GeoDynamo.jl` (export); Test `test/gpu_phase3_vector_transform.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase3_vector_transform.jl`:

```julia
@testset "gpu_vr_scale! [LOCAL]" begin
    nr = 3
    pr = rand(Float64, nl, nm, nr); pi_ = rand(Float64, nl, nm, nr)
    lfac = Float64[l * (l + 1) for l in 0:cfg.lmax]      # length nl
    rscale = [1.0 / (0.5 + 0.1k) for k in 1:nr]           # length nr (solver path 1/r)
    vr = zeros(Float64, nl, nm, nr); vi = zeros(Float64, nl, nm, nr)
    GeoDynamo.gpu_vr_scale!(vr, vi, pr, pi_, lfac, rscale)
    refr = similar(vr); refi = similar(vi)
    @inbounds for k in 1:nr, m in 1:nm, l in 1:nl
        f = lfac[l] * rscale[k]
        refr[l,m,k] = pr[l,m,k] * f
        refi[l,m,k] = pi_[l,m,k] * f
    end
    @test vr == refr && vi == refi
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase3_vector_transform.jl")'`
Expected: FAIL — `gpu_vr_scale!` undefined.

- [ ] **Step 3: Implement**

Append to `src/gpu/vector_transform.jl`. `lfac` (length-`nl`) → `(nl,1,1)`; `rscale` (length-`nr`) → `(1,1,nr)`:

```julia
"""
    gpu_vr_scale!(vr_alm_r, vr_alm_i, pol_r, pol_i, lfac, rscale) -> nothing

Scale the (split-complex) poloidal coefficients into the v_r source coefficients:
`vr_alm[l,m,r] = pol[l,m,r] · lfac[l] · rscale[r]`.  `lfac[l+1]=l(l+1)` (length
`lmax+1`); `rscale` is `1/r_val` (solver) or `1/r_val²` (MIE), length `nr`.
"""
function gpu_vr_scale!(vr_alm_r, vr_alm_i, pol_r, pol_i, lfac, rscale)
    lf = reshape(lfac, :, 1, 1)
    rs = reshape(rscale, 1, 1, :)
    @. vr_alm_r = pol_r * lf * rs
    @. vr_alm_i = pol_i * lf * rs
    return nothing
end
```

- [ ] **Step 4: Export**

In `src/GeoDynamo.jl`, add a GPU export line:
```julia
export gpu_vr_scale!, gpu_vector_spectral_to_physical!, gpu_vector_physical_to_spectral!
```
(`gpu_vector_*` are defined in Tasks 3–4; exporting now is legal.)

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase3_vector_transform.jl")'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/vector_transform.jl src/GeoDynamo.jl test/gpu_phase3_vector_transform.jl
git commit -m "feat(gpu): gpu_vr_scale! poloidal→v_r coefficient scaling (Phase 3)"
```

---

## Task 3: `gpu_vector_spectral_to_physical!`

**Files:** Modify `src/gpu/vector_transform.jl`; Test `test/gpu_phase3_vector_transform.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase3_vector_transform.jl`:

```julia
@testset "vector spectral_to_physical [LOCAL]" begin
    nr = 3
    arch = CPU()
    tor = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr)
    pol = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr)
    vr = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
    vθ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
    vφ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
    for k in 1:nr
        pol.data_real[3,1,k] = Float64(k); tor.data_real[4,2,k] = 0.5; tor.data_imag[4,2,k] = -0.25
    end
    lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
    rscale = [1.0/(0.5 + 0.1k) for k in 1:nr]
    GeoDynamo.gpu_vector_spectral_to_physical!(vr, vθ, vφ, tor, pol, cfg, lfac, rscale)
    @test size(vr.data) == (cfg.nlat, cfg.nlon, nr)
    for k in 1:nr
        S_k = complex.(pol.data_real[:,:,k], pol.data_imag[:,:,k])
        T_k = complex.(tor.data_real[:,:,k], tor.data_imag[:,:,k])
        rt, rp = SHTnsKit.synthesis_sphtor(cfg.sht_config, S_k, T_k; real_output = true)
        @test vθ.data[:,:,k] == rt && vφ.data[:,:,k] == rp
        vra = S_k .* reshape(lfac, :, 1) .* rscale[k]
        @test vr.data[:,:,k] == SHTnsKit.synthesis(cfg.sht_config, vra; real_output = true)
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase3_vector_transform.jl")'`
Expected: FAIL — `gpu_vector_spectral_to_physical!` undefined.

- [ ] **Step 3: Implement**

Append to `src/gpu/vector_transform.jl`:

```julia
"""
    gpu_vector_spectral_to_physical!(vr, vθ, vφ, tor, pol, config, lfac, rscale) -> nothing

Synthesize a toroidal–poloidal vector field to physical components.  Tangential
`(vθ, vφ)` per level via `synthesis_sphtor(poloidal, toroidal)`; radial `vr` per
level via scalar synthesis of `poloidal · lfac[l] · rscale[r]` (see `gpu_vr_scale!`).
"""
function gpu_vector_spectral_to_physical!(vr::GPUPhysicalField, vθ::GPUPhysicalField,
        vφ::GPUPhysicalField, tor::GPUSpectralField, pol::GPUSpectralField, config, lfac, rscale)
    sht = config.sht_config
    nr = pol.nr
    # v_r source coefficients (whole field), then per-level scalar synthesis.
    vr_alm_r = similar(pol.data_real); vr_alm_i = similar(pol.data_imag)
    gpu_vr_scale!(vr_alm_r, vr_alm_i, pol.data_real, pol.data_imag, lfac, rscale)
    for k in 1:nr
        S_k = complex.(view(pol.data_real, :, :, k), view(pol.data_imag, :, :, k))
        T_k = complex.(view(tor.data_real, :, :, k), view(tor.data_imag, :, :, k))
        vt, vp = _vector_synth_sphtor(sht, S_k, T_k)
        vθ.data[:, :, k] .= vt
        vφ.data[:, :, k] .= vp
        vra_k = complex.(view(vr_alm_r, :, :, k), view(vr_alm_i, :, :, k))
        vr.data[:, :, k] .= _scalar_synth(sht, vra_k)
    end
    return nothing
end
```

- [ ] **Step 4: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase3_vector_transform.jl")'`
Expected: PASS — tangential matches direct sphtor; v_r matches scaled scalar synth.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/vector_transform.jl test/gpu_phase3_vector_transform.jl
git commit -m "feat(gpu): gpu_vector_spectral_to_physical! (sphtor + v_r) (Phase 3)"
```

---

## Task 4: `gpu_vector_physical_to_spectral!` + roundtrip

**Files:** Modify `src/gpu/vector_transform.jl`; Test `test/gpu_phase3_vector_transform.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase3_vector_transform.jl`:

```julia
@testset "vector physical_to_spectral + roundtrip [LOCAL]" begin
    nr = 3
    arch = CPU()
    tor = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr)
    pol = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr)
    vr = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
    vθ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
    vφ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
    for k in 1:nr
        pol.data_real[3,1,k] = Float64(k); tor.data_real[4,2,k] = 0.5; tor.data_imag[4,2,k] = -0.25
    end
    pol0r = copy(pol.data_real); tor0r = copy(tor.data_real); tor0i = copy(tor.data_imag)
    lfac = Float64[l*(l+1) for l in 0:cfg.lmax]; rscale = [1.0/(0.5+0.1k) for k in 1:nr]
    GeoDynamo.gpu_vector_spectral_to_physical!(vr, vθ, vφ, tor, pol, cfg, lfac, rscale)
    fill!(pol.data_real,0.0); fill!(pol.data_imag,0.0); fill!(tor.data_real,0.0); fill!(tor.data_imag,0.0)
    GeoDynamo.gpu_vector_physical_to_spectral!(tor, pol, vθ, vφ, cfg)
    # analysis_sphtor recovers (S=poloidal, T=toroidal) from (vθ,vφ); v_r not consumed.
    @test isapprox(pol.data_real, pol0r; atol = 1e-10)
    @test isapprox(tor.data_real, tor0r; atol = 1e-10)
    @test isapprox(tor.data_imag, tor0i; atol = 1e-10)
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase3_vector_transform.jl")'`
Expected: FAIL — `gpu_vector_physical_to_spectral!` undefined.

- [ ] **Step 3: Implement**

Append to `src/gpu/vector_transform.jl`:

```julia
"""
    gpu_vector_physical_to_spectral!(tor, pol, vθ, vφ, config) -> nothing

Analyze the tangential physical components `(vθ, vφ)` into the toroidal `tor` and
poloidal `pol` spectral fields, per level, via `analysis_sphtor` (`S→pol`, `T→tor`).
`v_r` is not consumed (redundant for a solenoidal field), matching the CPU.
"""
function gpu_vector_physical_to_spectral!(tor::GPUSpectralField, pol::GPUSpectralField,
        vθ::GPUPhysicalField, vφ::GPUPhysicalField, config)
    sht = config.sht_config
    nr = pol.nr
    for k in 1:nr
        # Plain indexing (NOT @view): a @view SubArray would miss the ::CuArray
        # sphtor method and silently run on CPU against device data (see Phase 1).
        vt_k = vθ.data[:, :, k]
        vp_k = vφ.data[:, :, k]
        S_k, T_k = _vector_anal_sphtor(sht, vt_k, vp_k)
        pol.data_real[:, :, k] .= real.(S_k)
        pol.data_imag[:, :, k] .= imag.(S_k)
        tor.data_real[:, :, k] .= real.(T_k)
        tor.data_imag[:, :, k] .= imag.(T_k)
    end
    return nothing
end
```

- [ ] **Step 4: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase3_vector_transform.jl")'`
Expected: PASS — band-limited roundtrip recovers toroidal/poloidal to 1e-10.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/vector_transform.jl test/gpu_phase3_vector_transform.jl
git commit -m "feat(gpu): gpu_vector_physical_to_spectral! + roundtrip (Phase 3)"
```

---

## Task 5: GPU-box gate + register + regression

**Files:** Modify `test/gpu_phase3_vector_transform.jl`, `test/runtests.jl`

- [ ] **Step 1: Add the GPU-box gate** `[GPU-BOX]`

Add to `test/gpu_phase3_vector_transform.jl`:

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-3 gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        nr = 3
        lfac = Float64[l*(l+1) for l in 0:cfg.lmax]; rscale = [1.0/(0.5+0.1k) for k in 1:nr]
        mk(arch) = (GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr),
                    GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr),
                    GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr),
                    GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr),
                    GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr))
        ctor, cpol, cvr, cvθ, cvφ = mk(CPU())
        for k in 1:nr
            cpol.data_real[3,1,k] = Float64(k); ctor.data_real[4,2,k] = 0.5; ctor.data_imag[4,2,k] = -0.25
        end
        GeoDynamo.gpu_vector_spectral_to_physical!(cvr, cvθ, cvφ, ctor, cpol, cfg, lfac, rscale)  # CPU ref

        gtor, gpol, gvr, gvθ, gvφ = mk(GPU())
        d!(dst, src) = (copyto!(dst.data_real, src.data_real); copyto!(dst.data_imag, src.data_imag))
        d!(gtor, ctor); d!(gpol, cpol)
        glfac = GeoDynamo.on_architecture(GPU(), lfac); grscale = GeoDynamo.on_architecture(GPU(), rscale)
        GeoDynamo.gpu_vector_spectral_to_physical!(gvr, gvθ, gvφ, gtor, gpol, cfg, glfac, grscale)  # GPU
        @test gvθ.data isa CUDA.CuArray
        @test isapprox(Array(gvr.data), cvr.data; atol = 1e-12, rtol = 1e-10)
        @test isapprox(Array(gvθ.data), cvθ.data; atol = 1e-12, rtol = 1e-10)
        @test isapprox(Array(gvφ.data), cvφ.data; atol = 1e-12, rtol = 1e-10)
    end
end
```

- [ ] **Step 2: Run locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase3_vector_transform.jl")'`
Expected: `[LOCAL]` testsets pass; the gate skips. Mark **"implemented; awaiting GPU-box parity."**

- [ ] **Step 3: Register**

In `test/runtests.jl`, add `"gpu_phase3_vector_transform.jl"` (next to the Phase 2 entry).

- [ ] **Step 4: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI; include("test/gpu_phase2_nonlinear.jl"); include("test/allocation_runtime_checks.jl")'`
Expected: Phase 2 green; allocation guards 39/39.

- [ ] **Step 5: Commit**

```bash
git add test/gpu_phase3_vector_transform.jl test/runtests.jl
git commit -m "test(gpu): Phase-3 GPU-box gate + register vector transform"
```

---

## GPU-box validation handoff

On the GPU box:
```julia
using CUDA, Test, GeoDynamo
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase3_vector_transform.jl")    # the [GPU-BOX] gate must PASS
```
**Phase 3 passes when:** the vector synthesis (`v_r, v_θ, v_φ`) on `CuArray` matches the CPU(Array) result to ~1e-12, and the band-limited roundtrip recovers toroidal/poloidal to ~1e-10. Report any failure (a `gpu_*_sphtor` shape/orientation surprise — esp. if `gpu_synthesis_sphtor` returns `(nlon,nlat)` needing a transpose unlike the serial path; a normalization mismatch; or a `gpu_vr_scale!` broadcast issue) before Phase 4.

---

## Self-Review

**Spec coverage (design-doc Phase 3: "vector transform (vel/mag via gpu_*_sphtor) + v_r/curl assembly; gate vector roundtrip + vel nonlinear ≈"):** sphtor dispatch (Task 1), v_r scaling (Task 2), spectral→physical (Task 3), physical→spectral + roundtrip (Task 4), GPU gate + regression (Task 5). The "curl assembly" in the design phrase refers to vorticity/current — these are a SEPARATE spectral-stencil computation (confirmed by the CPU map: the plain vector transform has NO radial derivative) and are explicitly deferred to a later phase (they need a banded `∂/∂r` operator). The "vel nonlinear ≈" integration check (feeding real transformed velocity into Phase-2 kernels) is a Phase-5 integration concern; Phase 3 validates the transform itself. Covered.

**Placeholder scan:** none — every code step has complete code; every run step has the exact command + expected result.

**Type consistency:** `_vector_synth_sphtor(cfg_sht, S,T)→(vt,vp)`, `_vector_anal_sphtor(cfg_sht, vt,vp)→(S,T)`, `gpu_vr_scale!(vr_alm_r,vr_alm_i, pol_r,pol_i, lfac,rscale)`, `gpu_vector_spectral_to_physical!(vr,vθ,vφ, tor,pol, config, lfac,rscale)`, `gpu_vector_physical_to_spectral!(tor,pol, vθ,vφ, config)` — names + arg orders consistent across tasks and the interface block. Uses Phase 0 `GPUSpectralField`(`data_real`/`data_imag`/`nr`)/`GPUPhysicalField`(`data`) and Phase 1 `_scalar_synth`. The `physical_to_spectral!` plain-indexing (not `@view`) follows the Phase-1 GPU-dispatch rule. `synthesis_sphtor(cfg,S,T)→(nlat,nlon)` / `analysis_sphtor(cfg,vt,vp)→(lmax+1,mmax+1)` confirmed by runtime check.
