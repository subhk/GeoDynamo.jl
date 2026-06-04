# GPU Phase 1 — Scalar Transform (T/C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform a device-resident scalar field (temperature/composition) between its dense spectral `(lmax+1, mmax+1, nr)` representation and its physical `(nlat, nlon, nr)` representation on a single GPU, by looping radial levels through SHTnsKit's serial GPU transform (`gpu_synthesis`/`gpu_analysis`), with results equivalent (~1e-12) to the CPU serial transform.

**Architecture:** Phase 0 gave dense `GPUSpectralField{T,A}` (`data_real`/`data_imag` are `(lmax+1, mmax+1, nr)`) and `GPUPhysicalField{T,A}` (`data` is `(nlat, nlon, nr)`). SHTnsKit's `gpu_synthesis(cfg::SHTConfig, alm)`/`gpu_analysis(cfg, f)` operate on ONE dense `(lmax+1, mmax+1)` ↔ `(nlat, nlon)` level at a time (and fall back to `SHTnsKit.synthesis`/`analysis` on CPU). Phase 1 adds a transform-dispatch helper (`_scalar_synth`/`_scalar_anal`) whose `AbstractMatrix` method calls the always-available `SHTnsKit.synthesis`/`analysis` (CPU) and whose `CuArray` method — defined ONLY in the CUDA extension — calls `gpu_synthesis`/`gpu_analysis`. The two driver functions loop levels through that helper. Because the CPU method is always available, the loop logic is fully testable on the CPU backend (Array-backed GPU fields) without a GPU; only `CuArray` execution is GPU-gated.

**Tech Stack:** Julia, SHTnsKit (`synthesis`/`analysis` in core; `gpu_synthesis`/`gpu_analysis` in `SHTnsKitGPUExt`), CUDA.jl (weakdep via `GeoDynamoCUDAExt`).

---

## Prerequisite

Phase 0 is implemented on branch `feat/gpu-phase0` (worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu`), with `GPUSpectralField` in the DENSE `(lmax+1, mmax+1, nr)` layout (commit `ebf6cde`). Build Phase 1 on that branch (or a `feat/gpu-phase1` branched from it). Phase 0's GPU-box roundtrip gate should ideally be validated first, but Phase 1's CPU-backend logic does not depend on that.

## Testing without a local GPU (same convention as Phase 0)

- **[LOCAL]** — runs on any machine. **NEW for Phase 1:** because the transform helper's CPU method calls `SHTnsKit.synthesis`/`analysis` (always available), the driver functions RUN on Array-backed GPU fields locally. So Phase 1's *logic* is verified locally with real assertions (roundtrip + parity vs a direct `SHTnsKit.synthesis` call) — not skipped.
- **[GPU-BOX]** — requires a functional CUDA GPU; verifies the `CuArray` execution path (`gpu_synthesis`/`gpu_analysis`). Guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from the worktree. **Never pipe test runs through `tail`.**

## File Structure

- **Create** `src/gpu/scalar_transform.jl` — `_scalar_synth`/`_scalar_anal` (core CPU methods) + `gpu_scalar_spectral_to_physical!`/`gpu_scalar_physical_to_spectral!` drivers.
- **Modify** `ext/GeoDynamoCUDAExt.jl` — `CuArray` methods of `_scalar_synth`/`_scalar_anal` → `gpu_synthesis`/`gpu_analysis`.
- **Modify** `src/GeoDynamo.jl` — `include("gpu/scalar_transform.jl")` (after `include("gpu/fields.jl")`); export the two drivers.
- **Create** `test/gpu_phase1_scalar_transform.jl` — `[LOCAL]` (Array backend) + `[GPU-BOX]` (CuArray) tests.
- **Modify** `test/runtests.jl` — register the new test file.

Locked interfaces:

```julia
_scalar_synth(cfg_sht, alm::AbstractMatrix)::AbstractMatrix     # (lmax+1,mmax+1) complex -> (nlat,nlon) real ; core→SHTnsKit.synthesis, ext CuArray→gpu_synthesis
_scalar_anal(cfg_sht, f::AbstractMatrix)::AbstractMatrix        # (nlat,nlon) real -> (lmax+1,mmax+1) complex ; core→SHTnsKit.analysis, ext CuArray→gpu_analysis
gpu_scalar_spectral_to_physical!(phys::GPUPhysicalField, spec::GPUSpectralField, config)::GPUPhysicalField
gpu_scalar_physical_to_spectral!(spec::GPUSpectralField, phys::GPUPhysicalField, config)::GPUSpectralField
```

`config` here is the GeoDynamo `SHTnsKitConfig`; pass `config.sht_config` (a `SHTnsKit.SHTConfig`) to the helpers.

---

## Task 1: transform-dispatch helper (CPU method + CUDA override)

**Files:**
- Create: `src/gpu/scalar_transform.jl`
- Modify: `ext/GeoDynamoCUDAExt.jl`
- Modify: `src/GeoDynamo.jl` (include)
- Test: `test/gpu_phase1_scalar_transform.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase1_scalar_transform.jl`:

```julia
using Test
using GeoDynamo
import SHTnsKit

@testset "GPU Phase 1 — Scalar Transform" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 3)
    sht = cfg.sht_config

    @testset "transform helper CPU method [LOCAL]" begin
        alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        alm[3, 1] = 1.0 + 0.0im                       # a single (l=2,m=0) mode
        f = GeoDynamo._scalar_synth(sht, alm)         # -> (nlat,nlon) real
        @test size(f) == (cfg.nlat, cfg.nlon)
        @test eltype(f) <: Real
        # matches a direct SHTnsKit synthesis exactly (helper is a thin pass-through on CPU)
        @test f == SHTnsKit.synthesis(sht, alm; real_output = true)
        alm2 = GeoDynamo._scalar_anal(sht, f)         # back to coeffs
        @test size(alm2) == (cfg.lmax + 1, cfg.mmax + 1)
        @test alm2 == SHTnsKit.analysis(sht, f)
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase1_scalar_transform.jl")'`
Expected: FAIL — `GeoDynamo._scalar_synth` undefined.

- [ ] **Step 3: Implement the core helper**

Create `src/gpu/scalar_transform.jl`:

```julia
# =============================================================================
# GPU Phase 1 — scalar (T/C) spectral<->physical transform, per radial level.
# The transform itself is reused from SHTnsKit: the AbstractMatrix methods below
# call the always-available serial CPU transform; the CUDA extension adds CuArray
# methods that call SHTnsKit's gpu_synthesis/gpu_analysis. Both consume/produce a
# DENSE (lmax+1, mmax+1) coefficient matrix and an (nlat, nlon) spatial matrix.
# =============================================================================

using SHTnsKit

# CPU path (always available; no CUDA needed). `cfg_sht` is a SHTnsKit.SHTConfig.
_scalar_synth(cfg_sht, alm::AbstractMatrix) = SHTnsKit.synthesis(cfg_sht, alm; real_output = true)
_scalar_anal(cfg_sht, f::AbstractMatrix)    = SHTnsKit.analysis(cfg_sht, f)
```

- [ ] **Step 4: Add the CUDA override**

In `ext/GeoDynamoCUDAExt.jl`, after the existing `on_architecture`/`gpu_functional` overrides, add:

```julia
# Phase 1: route CuArray coefficient/spatial matrices through SHTnsKit's GPU transform.
GeoDynamo._scalar_synth(cfg_sht, alm::CUDA.CuArray) = SHTnsKit.gpu_synthesis(cfg_sht, alm; real_output = true)
GeoDynamo._scalar_anal(cfg_sht, f::CUDA.CuArray)    = SHTnsKit.gpu_analysis(cfg_sht, f)
```

- [ ] **Step 5: Include in the module**

In `src/GeoDynamo.jl`, immediately after the `include("gpu/fields.jl")` line, add:

```julia
include("gpu/scalar_transform.jl")
```

- [ ] **Step 6: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase1_scalar_transform.jl")'`
Expected: PASS — the CPU helper equals a direct `SHTnsKit.synthesis`/`analysis` call.

- [ ] **Step 7: Commit**

```bash
git add src/gpu/scalar_transform.jl ext/GeoDynamoCUDAExt.jl src/GeoDynamo.jl test/gpu_phase1_scalar_transform.jl
git commit -m "feat(gpu): scalar transform dispatch helper (CPU + CUDA gpu_* methods) (Phase 1)"
```

---

## Task 2: `gpu_scalar_spectral_to_physical!`

**Files:**
- Modify: `src/gpu/scalar_transform.jl`
- Modify: `src/GeoDynamo.jl` (export)
- Test: `test/gpu_phase1_scalar_transform.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase1_scalar_transform.jl`:

```julia
@testset "spectral_to_physical [LOCAL]" begin
    arch = CPU()                                            # Array-backed GPU fields → CPU transform path
    spec = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, cfg.nr)
    phys = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, cfg.nr)
    # band-limited spectral content per level
    for k in 1:cfg.nr
        spec.data_real[3, 1, k] = Float64(k)               # (l=2,m=0) real
        spec.data_real[4, 2, k] = 0.5                       # (l=3,m=1) real
        spec.data_imag[4, 2, k] = -0.25                     # (l=3,m=1) imag
    end
    GeoDynamo.gpu_scalar_spectral_to_physical!(phys, spec, cfg)
    @test size(phys.data) == (cfg.nlat, cfg.nlon, cfg.nr)
    # each level must equal a direct synthesis of that level's complex coeffs
    for k in 1:cfg.nr
        alm_k = complex.(spec.data_real[:, :, k], spec.data_imag[:, :, k])
        @test phys.data[:, :, k] == SHTnsKit.synthesis(cfg.sht_config, alm_k; real_output = true)
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase1_scalar_transform.jl")'`
Expected: FAIL — `gpu_scalar_spectral_to_physical!` undefined.

- [ ] **Step 3: Implement the driver**

Append to `src/gpu/scalar_transform.jl`:

```julia
"""
    gpu_scalar_spectral_to_physical!(phys, spec, config) -> phys

Synthesize each radial level of the dense spectral field `spec`
(`(lmax+1, mmax+1, nr)` split real/imag) into the physical field `phys`
(`(nlat, nlon, nr)`), via SHTnsKit's per-level transform (GPU on `CuArray`s,
CPU otherwise).  `config` is the GeoDynamo `SHTnsKitConfig`.
"""
function gpu_scalar_spectral_to_physical!(phys::GPUPhysicalField, spec::GPUSpectralField, config)
    sht = config.sht_config
    nr = spec.nr
    @inbounds for k in 1:nr
        alm_k = complex.(view(spec.data_real, :, :, k), view(spec.data_imag, :, :, k))
        f_k = _scalar_synth(sht, alm_k)              # (nlat, nlon)
        phys.data[:, :, k] .= f_k
    end
    return phys
end
```

- [ ] **Step 4: Export**

In `src/GeoDynamo.jl`, add `gpu_scalar_spectral_to_physical!` to the GPU export line (the one with `field_to_host, field_to_device`).

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase1_scalar_transform.jl")'`
Expected: PASS — each physical level equals the direct per-level synthesis.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/scalar_transform.jl src/GeoDynamo.jl test/gpu_phase1_scalar_transform.jl
git commit -m "feat(gpu): gpu_scalar_spectral_to_physical! per-level synthesis (Phase 1)"
```

---

## Task 3: `gpu_scalar_physical_to_spectral!` + roundtrip

**Files:**
- Modify: `src/gpu/scalar_transform.jl`
- Modify: `src/GeoDynamo.jl` (export)
- Test: `test/gpu_phase1_scalar_transform.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase1_scalar_transform.jl`:

```julia
@testset "physical_to_spectral + roundtrip [LOCAL]" begin
    arch = CPU()
    spec = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, cfg.nr)
    phys = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, cfg.nr)
    # band-limited start (so analysis∘synthesis is the identity to ~1e-12)
    for k in 1:cfg.nr
        spec.data_real[3, 1, k] = Float64(k)
        spec.data_real[4, 2, k] = 0.5
        spec.data_imag[4, 2, k] = -0.25
    end
    spec0_r = copy(spec.data_real); spec0_i = copy(spec.data_imag)

    GeoDynamo.gpu_scalar_spectral_to_physical!(phys, spec, cfg)
    # zero spec, then analyze physical back into it
    fill!(spec.data_real, 0.0); fill!(spec.data_imag, 0.0)
    GeoDynamo.gpu_scalar_physical_to_spectral!(spec, phys, cfg)

    @test size(spec.data_real) == (cfg.lmax + 1, cfg.mmax + 1, cfg.nr)
    # roundtrip: recovered coeffs ≈ original (only the populated band; high-l slots stay ~0)
    @test isapprox(spec.data_real, spec0_r; atol = 1e-10)
    @test isapprox(spec.data_imag, spec0_i; atol = 1e-10)
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase1_scalar_transform.jl")'`
Expected: FAIL — `gpu_scalar_physical_to_spectral!` undefined.

- [ ] **Step 3: Implement the driver**

Append to `src/gpu/scalar_transform.jl`:

```julia
"""
    gpu_scalar_physical_to_spectral!(spec, phys, config) -> spec

Analyze each radial level of the physical field `phys` (`(nlat, nlon, nr)`) into
the dense spectral field `spec` (`(lmax+1, mmax+1, nr)` split real/imag), via
SHTnsKit's per-level transform (GPU on `CuArray`s, CPU otherwise).
"""
function gpu_scalar_physical_to_spectral!(spec::GPUSpectralField, phys::GPUPhysicalField, config)
    sht = config.sht_config
    nr = spec.nr
    @inbounds for k in 1:nr
        f_k = phys.data[:, :, k]                      # materialize the level (contiguous for the transform)
        alm_k = _scalar_anal(sht, f_k)                # (lmax+1, mmax+1) complex
        spec.data_real[:, :, k] .= real.(alm_k)
        spec.data_imag[:, :, k] .= imag.(alm_k)
    end
    return spec
end
```

- [ ] **Step 4: Export**

In `src/GeoDynamo.jl`, add `gpu_scalar_physical_to_spectral!` to the GPU export line.

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase1_scalar_transform.jl")'`
Expected: PASS — band-limited roundtrip recovers the coefficients to 1e-10.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/scalar_transform.jl src/GeoDynamo.jl test/gpu_phase1_scalar_transform.jl
git commit -m "feat(gpu): gpu_scalar_physical_to_spectral! + roundtrip (Phase 1)"
```

---

## Task 4: GPU-box gate (CuArray execution + GPU≈CPU parity)

**Files:**
- Test: `test/gpu_phase1_scalar_transform.jl`

- [ ] **Step 1: Write the gate test** `[GPU-BOX] (the Phase-1 gate)`

Add to `test/gpu_phase1_scalar_transform.jl`:

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-1 gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        # Build identical content on CPU (Array) and GPU (CuArray) fields.
        cspec = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, CPU(), cfg, cfg.nr)
        cphys = GeoDynamo.allocate_gpu_physical_field(Float64, CPU(), cfg, cfg.nr)
        for k in 1:cfg.nr
            cspec.data_real[3, 1, k] = Float64(k)
            cspec.data_real[4, 2, k] = 0.5
            cspec.data_imag[4, 2, k] = -0.25
        end
        GeoDynamo.gpu_scalar_spectral_to_physical!(cphys, cspec, cfg)   # CPU reference

        gspec = GeoDynamo.field_to_device(GPU(), (copy(cspec.data_real), copy(cspec.data_imag)), cfg, cfg.nr)
        gphys = GeoDynamo.allocate_gpu_physical_field(Float64, GPU(), cfg, cfg.nr)
        GeoDynamo.gpu_scalar_spectral_to_physical!(gphys, gspec, cfg)   # GPU path
        @test gphys.data isa CUDA.CuArray
        # GPU ≈ CPU synthesis (reduction reorder → tolerance, not bitwise)
        @test isapprox(Array(gphys.data), cphys.data; atol = 1e-12, rtol = 1e-10)

        # analysis parity + GPU roundtrip
        gspec2 = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, GPU(), cfg, cfg.nr)
        GeoDynamo.gpu_scalar_physical_to_spectral!(gspec2, gphys, cfg)
        @test isapprox(Array(gspec2.data_real), cspec.data_real; atol = 1e-10)
        @test isapprox(Array(gspec2.data_imag), cspec.data_imag; atol = 1e-10)
    end
end
```

- [ ] **Step 2: Run it locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase1_scalar_transform.jl")'`
Expected: the gate testset records a **skip** (no GPU); all `[LOCAL]` testsets pass. Mark **"implemented; awaiting GPU-box parity confirmation."**

- [ ] **Step 3: Commit**

```bash
git add test/gpu_phase1_scalar_transform.jl
git commit -m "test(gpu): Phase-1 GPU-box gate (CuArray execution + GPU≈CPU parity)"
```

---

## Task 5: register test + CPU regression

**Files:**
- Modify: `test/runtests.jl`

- [ ] **Step 1: Register**

In `test/runtests.jl`, add `"gpu_phase1_scalar_transform.jl"` to the test-file list (next to `"gpu_phase0_foundation.jl"`).

- [ ] **Step 2: Run via the suite entry** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase1_scalar_transform.jl")'`
Expected: PASS — `[LOCAL]` testsets pass; the `[GPU-BOX]` gate skips.

- [ ] **Step 3: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI; include("test/gpu_phase0_foundation.jl"); include("test/allocation_runtime_checks.jl")'`
Expected: Phase 0 green-with-skips; allocation guards 39/39 (Phase 1 added only new files + include/export; no CPU compute path changed).

- [ ] **Step 4: Commit**

```bash
git add test/runtests.jl
git commit -m "test(gpu): register Phase-1 scalar transform tests"
```

---

## GPU-box validation handoff

On the GPU box (CUDA loaded):
```julia
using CUDA, Test, GeoDynamo
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase1_scalar_transform.jl")     # the [GPU-BOX] gate must now PASS
```
**Phase 1 passes when:** on the GPU box, `gpu_scalar_spectral_to_physical!`/`..._physical_to_spectral!` run on `CuArray` fields and produce results matching the CPU path within `atol=1e-12` (synthesis) / `1e-10` (analysis roundtrip). Report any failure (scalar-indexing error inside the per-level loop, a `gpu_synthesis` shape/normalization mismatch, or a tolerance miss) for fixing before Phase 2.

---

## Self-Review

**Spec coverage (design-doc Phase 1: "scalar transform (T/C) via SHTnsKit gpu_* on CuArray fields; gate: scalar roundtrip GPU≈CPU ~1e-12, dealiased"):** Task 1 (gpu_* dispatch), Task 2 (spectral→physical), Task 3 (physical→spectral + roundtrip), Task 4 (GPU≈CPU parity gate), Task 5 (registration + regression). nlat=24/nlon=48 with lmax=mmax=8 is dealiased (nlon=48 > 2·mmax+1=17; 3·mmax=24 ≤ 48). Covered.

**Placeholder scan:** none — every code step has complete code; every run step has the exact command + expected result.

**Type consistency:** `_scalar_synth(cfg_sht, alm::AbstractMatrix)` / `_scalar_anal(cfg_sht, f::AbstractMatrix)` (core) and the `::CuArray` ext methods; `gpu_scalar_spectral_to_physical!(phys::GPUPhysicalField, spec::GPUSpectralField, config)` and `gpu_scalar_physical_to_spectral!(spec::GPUSpectralField, phys::GPUPhysicalField, config)` — names + arg order consistent across tasks and the interface block. Uses Phase 0's `GPUSpectralField` (dense `(lmax+1,mmax+1,nr)`, fields `data_real`/`data_imag`/`nr`) and `GPUPhysicalField` (`data` `(nlat,nlon,nr)`/`nr`) exactly as defined. `config.sht_config` is a `SHTnsKit.SHTConfig` (verified).
