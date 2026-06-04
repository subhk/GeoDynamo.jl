# GPU Phase 0 — Foundation & Fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allocate GeoDynamo physical & spectral fields on a single NVIDIA GPU and move them host↔device bit-identically, behind the CUDA package extension, with the CPU path untouched.

**Architecture:** Build on the existing (partly-orphaned) `AbstractArchitecture` scaffolding (`GPU{B}` carrying a KernelAbstractions backend; `arch_zeros`, `on_architecture`, `get_backend` already defined in `src/core/architecture.jl` and overridden in `ext/GeoDynamoCUDAExt.jl`). Add a `gpu_functional()` capability gate, a `GPU()` convenience constructor, device-backed field containers (`GPUPhysicalField`/`GPUSpectralField`), and field-level host↔device transfer. CUDA-specific code lives only in the extension (loaded when `CUDA` is present); the core defines interfaces + CPU-safe fallbacks.

**Tech Stack:** Julia, CUDA.jl (weakdep, via `GeoDynamoCUDAExt`), KernelAbstractions, the existing `SHTnsKitGPUExt` device utilities.

---

## CRITICAL: testing without a local GPU

The development machine has **no NVIDIA GPU** (Apple Silicon). CUDA.jl loads but `CUDA.functional()` is `false`, so no kernel/`CuArray` op runs locally. Every test below is one of:

- **[LOCAL]** — runs and must pass on any machine (interface exists, CPU dispatch, `gpu_functional()==false`, package precompiles, CPU regression). The implementing agent runs these.
- **[GPU-BOX]** — requires a functional CUDA GPU; written now, but **skipped automatically** when `!GeoDynamo.gpu_functional()` (via `@test_skip`). The **user runs these on their GPU box** and reports results.

Every `[GPU-BOX]` test uses this guard so the suite stays green locally:

```julia
if !GeoDynamo.gpu_functional()
    @test_skip "requires a functional CUDA GPU"
else
    # real CuArray assertions
end
```

The implementing agent must NOT mark a `[GPU-BOX]` task "done/passing" — only "implemented; awaiting GPU-box validation."

---

## File Structure

- **Create** `src/gpu/device.jl` — capability gate (`gpu_functional`), `GPU()` convenience constructor, device synchronize. Core-side stubs; CUDA ext fills them.
- **Create** `src/gpu/fields.jl` — `GPUPhysicalField`, `GPUSpectralField` container structs + allocators (`allocate_gpu_physical_field`, `allocate_gpu_spectral_field`) + field transfer (`field_to_host`, `field_to_device`).
- **Modify** `ext/GeoDynamoCUDAExt.jl` — implement `gpu_functional`, `_gpu_default_backend`, `gpu_synchronize` with CUDA.
- **Modify** `src/GeoDynamo.jl` — `include` the two new files (after `core/architecture.jl`, before `fields/containers.jl`) and export the new public names.
- **Create** `test/gpu_phase0_foundation.jl` — `[LOCAL]` + `[GPU-BOX]` tests.
- **Modify** `test/runtests.jl` — register the new test file.

Interfaces (locked here; later tasks must match exactly):

```julia
gpu_functional()::Bool                                   # false unless CUDA functional
GPU()::GPU                                                # convenience: functional CUDA backend (errors if unavailable)
gpu_synchronize()::Nothing                               # device barrier (no-op on CPU)
struct GPUPhysicalField{T,A}  config; nlat; nlon; nr; data::A   end
struct GPUSpectralField{T,A}  config; nlm; nr; data_real::A; data_imag::A   end
allocate_gpu_physical_field(::Type{T}, arch, config, nr)::GPUPhysicalField
allocate_gpu_spectral_field(::Type{T}, arch, config, nr)::GPUSpectralField
field_to_host(f)        # GPU field -> NamedTuple of host Arrays
field_to_device(arch, host_arrays..., config, nr)  # host -> GPU field
```

---

## Task 1: `gpu_functional()` capability gate

**Files:**
- Create: `src/gpu/device.jl`
- Modify: `ext/GeoDynamoCUDAExt.jl`
- Modify: `src/GeoDynamo.jl:515` (include + export)
- Test: `test/gpu_phase0_foundation.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase0_foundation.jl`:

```julia
using Test
using GeoDynamo

@testset "GPU Phase 0 — Foundation" begin
    @testset "gpu_functional gate [LOCAL]" begin
        @test GeoDynamo.gpu_functional() isa Bool
        # No CUDA GPU in CI / dev machine → false. On a GPU box this flips true.
        @test Base.isexported(GeoDynamo, :gpu_functional)
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase0_foundation.jl")'`
Expected: FAIL — `UndefVarError: gpu_functional` (not defined yet).

- [ ] **Step 3: Define the core stub**

Create `src/gpu/device.jl`:

```julia
# =============================================================================
# GPU Phase 0 — device capability gate + constructor (core-side interface).
# CUDA-specific behaviour is provided by ext/GeoDynamoCUDAExt.jl when CUDA loads.
# =============================================================================

"""
    gpu_functional() -> Bool

`true` only when a CUDA-capable GPU is present AND the `GeoDynamoCUDAExt`
extension is loaded (i.e. `CUDA.functional()`).  `false` otherwise, including on
machines with no GPU.  Use this to gate GPU code paths and tests.
"""
gpu_functional() = false

"""
    gpu_synchronize()

Block until all queued GPU work completes.  No-op when no GPU backend is active.
"""
gpu_synchronize() = nothing
```

- [ ] **Step 4: Include + export in the module**

In `src/GeoDynamo.jl`, immediately after the `include("core/architecture.jl")` line (currently line 515), add:

```julia
include("gpu/device.jl")
```

In the export block at `src/GeoDynamo.jl:482` (the line exporting `AbstractArchitecture, CPU, GPU, ...`), append `gpu_functional, gpu_synchronize`:

```julia
export AbstractArchitecture, CPU, GPU, arch_zeros, on_architecture, get_backend, gpu_functional, gpu_synchronize
```

- [ ] **Step 5: Implement the CUDA override**

In `ext/GeoDynamoCUDAExt.jl`, after the existing `GeoDynamo.on_architecture(::GeoDynamo.GPU, a) = CUDA.cu(a)` line, add:

```julia
GeoDynamo.gpu_functional() = CUDA.functional()
GeoDynamo.gpu_synchronize() = (CUDA.synchronize(); nothing)
```

- [ ] **Step 6: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase0_foundation.jl")'`
Expected: PASS (`gpu_functional()` returns `false` locally; the value is a `Bool`; symbol exported).

- [ ] **Step 7: Commit**

```bash
git add src/gpu/device.jl src/GeoDynamo.jl ext/GeoDynamoCUDAExt.jl test/gpu_phase0_foundation.jl
git commit -m "feat(gpu): add gpu_functional capability gate (Phase 0)"
```

---

## Task 2: `GPU()` convenience constructor

**Files:**
- Modify: `src/gpu/device.jl`
- Modify: `ext/GeoDynamoCUDAExt.jl`
- Test: `test/gpu_phase0_foundation.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase0_foundation.jl` inside the outer `@testset`:

```julia
@testset "GPU() constructor [LOCAL/GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        # Without a functional GPU, GPU() must error clearly, not silently fall back.
        @test_throws ErrorException GPU()
    else
        a = GPU()
        @test a isa GPU
        @test get_backend(a) !== KernelAbstractions.CPU()
    end
end
```

(Add `import KernelAbstractions` to the test file's `using`/`import` block.)

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase0_foundation.jl")'`
Expected: FAIL — `GPU()` (zero-arg) is not defined (`GPU` currently requires a backend arg), so calling it errors with `MethodError`, not the expected `ErrorException`.

- [ ] **Step 3: Define the core constructor stub**

Append to `src/gpu/device.jl`:

```julia
"""
    GPU() -> GPU

Construct a `GPU` architecture bound to the default functional GPU backend.
Errors if no GPU backend is available (CUDA extension not loaded / no device).
The CUDA extension overrides `_gpu_default_backend()` to return a `CUDABackend`.
"""
GPU() = GPU(_gpu_default_backend())

_gpu_default_backend() = error(
    "GPU() requires a functional CUDA GPU and the GeoDynamoCUDAExt extension " *
    "(load CUDA.jl on a machine with a CUDA device).")
```

- [ ] **Step 4: Implement the CUDA override**

In `ext/GeoDynamoCUDAExt.jl`, after the `gpu_functional` override from Task 1, add:

```julia
function GeoDynamo._gpu_default_backend()
    CUDA.functional() || error("GPU() called but CUDA.functional() is false (no usable CUDA device).")
    return CUDA.CUDABackend()
end
```

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase0_foundation.jl")'`
Expected: PASS — locally `GPU()` throws `ErrorException` (caught by `@test_throws`); the GPU-box branch is dormant.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/device.jl ext/GeoDynamoCUDAExt.jl test/gpu_phase0_foundation.jl
git commit -m "feat(gpu): GPU() convenience constructor (default CUDA backend)"
```

---

## Task 3: GPU physical field container + allocator

**Files:**
- Create: `src/gpu/fields.jl`
- Modify: `src/GeoDynamo.jl` (include + export)
- Test: `test/gpu_phase0_foundation.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL + GPU-BOX]`

Add to `test/gpu_phase0_foundation.jl`:

```julia
@testset "GPU physical field allocation [GPU-BOX]" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 16, nlon = 32, nr = 4)
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        f = GeoDynamo.allocate_gpu_physical_field(Float64, GPU(), cfg, 4)
        @test f isa GeoDynamo.GPUPhysicalField
        @test size(f.data) == (cfg.nlat, cfg.nlon, 4)
        @test f.data isa CUDA.CuArray          # CUDA in scope on the GPU box test run
        @test all(Array(f.data) .== 0)         # zero-initialised
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase0_foundation.jl")'`
Expected: locally the `@test_skip` records a skip but the testset references `GeoDynamo.GPUPhysicalField` / `allocate_gpu_physical_field`, which are undefined → `UndefVarError` at parse/eval of the testset. FAIL.

- [ ] **Step 3: Implement the container + allocator**

Create `src/gpu/fields.jl`:

```julia
# =============================================================================
# GPU Phase 0 — device-resident field containers (single GPU, no MPI/pencils).
# Shapes mirror the CPU containers: physical (nlat, nlon, nr); spectral (nlm, nr).
# Backing arrays are allocated through arch_zeros(arch, ...) → CuArray on a GPU
# backend, plain Array on CPU. No PencilArrays (single GPU has no decomposition).
# =============================================================================

"""
    GPUPhysicalField{T,A}

Device-resident physical field: `data` is an `(nlat, nlon, nr)` array on the
architecture's backend.
"""
struct GPUPhysicalField{T, A}
    config::Any
    nlat::Int
    nlon::Int
    nr::Int
    data::A
end

"""
    allocate_gpu_physical_field(T, arch, config, nr) -> GPUPhysicalField

Allocate a zero-filled `(nlat, nlon, nr)` physical field on `arch`'s backend.
"""
function allocate_gpu_physical_field(::Type{T}, arch::AbstractArchitecture, config, nr::Int) where {T}
    nlat = config.nlat
    nlon = config.nlon
    data = arch_zeros(arch, T, nlat, nlon, nr)
    return GPUPhysicalField{T, typeof(data)}(config, nlat, nlon, nr, data)
end
```

- [ ] **Step 4: Include + export**

In `src/GeoDynamo.jl`, immediately after the `include("gpu/device.jl")` line added in Task 1, add:

```julia
include("gpu/fields.jl")
```

Append to the export line (the one ending `..., gpu_functional, gpu_synchronize`):

```julia
       GPUPhysicalField, GPUSpectralField, allocate_gpu_physical_field,
       allocate_gpu_spectral_field, field_to_host, field_to_device
```

(`GPUSpectralField`, `allocate_gpu_spectral_field`, `field_to_host`, `field_to_device` are created in Tasks 4–5; exporting them now is harmless since the names resolve once those tasks land. If running tasks strictly in order and the export errors on an undefined name, add each name in its own task — but Julia `export` of a not-yet-defined name is legal, so export all here.)

- [ ] **Step 5: Run the test** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase0_foundation.jl")'`
Expected: PASS locally — the physical-field testset records a **skip** (no GPU); the names now resolve so no `UndefVarError`. Mark the task **"implemented; GPU-box validation pending."**

- [ ] **Step 6: Commit**

```bash
git add src/gpu/fields.jl src/GeoDynamo.jl test/gpu_phase0_foundation.jl
git commit -m "feat(gpu): GPUPhysicalField container + device allocator (Phase 0)"
```

---

## Task 4: GPU spectral field container + allocator

**Files:**
- Modify: `src/gpu/fields.jl`
- Test: `test/gpu_phase0_foundation.jl`

- [ ] **Step 1: Write the failing test** `[GPU-BOX]`

Add to `test/gpu_phase0_foundation.jl`:

```julia
@testset "GPU spectral field allocation [GPU-BOX]" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 16, nlon = 32, nr = 4)
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        f = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, GPU(), cfg, 4)
        @test f isa GeoDynamo.GPUSpectralField
        @test size(f.data_real) == (cfg.nlm, 4)
        @test size(f.data_imag) == (cfg.nlm, 4)
        @test f.data_real isa CUDA.CuArray
        @test all(Array(f.data_real) .== 0) && all(Array(f.data_imag) .== 0)
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase0_foundation.jl")'`
Expected: FAIL — `GeoDynamo.GPUSpectralField` / `allocate_gpu_spectral_field` undefined.

- [ ] **Step 3: Implement the container + allocator**

Append to `src/gpu/fields.jl`:

```julia
"""
    GPUSpectralField{T,A}

Device-resident spectral field: `data_real`/`data_imag` are `(nlm, nr)` real
arrays on the architecture's backend (split real/imag mirrors the CPU container).
`T` is the real element type (e.g. `Float64`); pass `ComplexF64` as the alloc
element type to select `Float64` storage.
"""
struct GPUSpectralField{T, A}
    config::Any
    nlm::Int
    nr::Int
    data_real::A
    data_imag::A
end

"""
    allocate_gpu_spectral_field(CT, arch, config, nr) -> GPUSpectralField

Allocate a zero-filled `(nlm, nr)` split-complex spectral field on `arch`'s
backend.  `CT` is the complex element type (`ComplexF64`); storage is its real
part type (`Float64`).
"""
function allocate_gpu_spectral_field(::Type{CT}, arch::AbstractArchitecture, config, nr::Int) where {CT}
    RT = real(CT)
    nlm = config.nlm
    dr = arch_zeros(arch, RT, nlm, nr)
    di = arch_zeros(arch, RT, nlm, nr)
    return GPUSpectralField{RT, typeof(dr)}(config, nlm, nr, dr, di)
end
```

- [ ] **Step 4: Run the test** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase0_foundation.jl")'`
Expected: PASS locally — testset records a **skip**; names resolve. Mark **"GPU-box validation pending."**

- [ ] **Step 5: Commit**

```bash
git add src/gpu/fields.jl test/gpu_phase0_foundation.jl
git commit -m "feat(gpu): GPUSpectralField container + device allocator (Phase 0)"
```

---

## Task 5: host↔device field transfer + roundtrip gate

**Files:**
- Modify: `src/gpu/fields.jl`
- Test: `test/gpu_phase0_foundation.jl`

- [ ] **Step 1: Write the failing test** `[GPU-BOX] (the Phase-0 gate)`

Add to `test/gpu_phase0_foundation.jl`:

```julia
@testset "host<->device field roundtrip (Phase-0 gate) [GPU-BOX]" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 16, nlon = 32, nr = 4)
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        # physical
        host_phys = rand(Float64, cfg.nlat, cfg.nlon, 4)
        gf = GeoDynamo.field_to_device(GPU(), host_phys, cfg, 4)            # host -> device
        back = GeoDynamo.field_to_host(gf)                                   # device -> host
        @test back.data == host_phys                                        # BIT-IDENTICAL

        # spectral
        hr = rand(Float64, cfg.nlm, 4); hi = rand(Float64, cfg.nlm, 4)
        gs = GeoDynamo.field_to_device(GPU(), (hr, hi), cfg, 4)
        bs = GeoDynamo.field_to_host(gs)
        @test bs.data_real == hr && bs.data_imag == hi
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase0_foundation.jl")'`
Expected: FAIL — `field_to_device`/`field_to_host` undefined.

- [ ] **Step 3: Implement transfer**

Append to `src/gpu/fields.jl` (uses the existing `on_architecture` both directions: `on_architecture(::CPU, a)` → host `Array`, `on_architecture(::GPU, a)` → `CUDA.cu`):

```julia
"""
    field_to_host(f) -> NamedTuple

Copy a GPU field's device arrays back to host `Array`s.  Returns
`(; data)` for a `GPUPhysicalField`, `(; data_real, data_imag)` for a
`GPUSpectralField`.
"""
field_to_host(f::GPUPhysicalField) = (; data = on_architecture(CPU(), f.data))
function field_to_host(f::GPUSpectralField)
    return (; data_real = on_architecture(CPU(), f.data_real),
              data_imag = on_architecture(CPU(), f.data_imag))
end

"""
    field_to_device(arch, host_phys::AbstractArray, config, nr) -> GPUPhysicalField
    field_to_device(arch, (hr, hi)::Tuple, config, nr)          -> GPUSpectralField

Copy host data onto `arch`'s backend, wrapped in the matching GPU field.
"""
function field_to_device(arch::AbstractArchitecture, host_phys::AbstractArray{T, 3}, config, nr::Int) where {T}
    data = on_architecture(arch, host_phys)
    return GPUPhysicalField{T, typeof(data)}(config, size(host_phys, 1), size(host_phys, 2), nr, data)
end

function field_to_device(arch::AbstractArchitecture, host_spec::Tuple{<:AbstractArray, <:AbstractArray}, config, nr::Int)
    hr, hi = host_spec
    dr = on_architecture(arch, hr)
    di = on_architecture(arch, hi)
    return GPUSpectralField{eltype(hr), typeof(dr)}(config, size(hr, 1), nr, dr, di)
end
```

- [ ] **Step 4: Run the test** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase0_foundation.jl")'`
Expected: PASS locally — gate testset records a **skip**; names resolve. This is THE Phase-0 gate — mark **"implemented; awaiting GPU-box bit-identical roundtrip confirmation."**

- [ ] **Step 5: Commit**

```bash
git add src/gpu/fields.jl test/gpu_phase0_foundation.jl
git commit -m "feat(gpu): host<->device field transfer + Phase-0 roundtrip gate"
```

---

## Task 6: register test + CPU regression

**Files:**
- Modify: `test/runtests.jl`

- [ ] **Step 1: Register the test file**

In `test/runtests.jl`, add `"gpu_phase0_foundation.jl"` to the list of included test files (follow the existing list format — append an entry alongside e.g. `"gpu_backend.jl"`).

- [ ] **Step 2: Run the new file via the suite entry** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase0_foundation.jl")'`
Expected: PASS with skips (all `[GPU-BOX]` testsets skipped; `[LOCAL]` testsets pass).

- [ ] **Step 3: CPU regression — confirm the CPU path is untouched** `[LOCAL]`

Run the existing GPU-arch test plus a representative transform test to confirm no regression:
`~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI; include("test/gpu_backend.jl"); include("test/allocation_runtime_checks.jl")'`
Expected: `gpu_backend.jl` green; allocation guards 39/39. (Phase 0 adds only new files + include/export lines; no CPU code changed.)

- [ ] **Step 4: Commit**

```bash
git add test/runtests.jl
git commit -m "test(gpu): register Phase-0 foundation tests"
```

---

## GPU-box validation handoff (user runs on the GPU machine)

After the 6 tasks land (green locally, all `[GPU-BOX]` skipped), the user runs the suite on the GPU box with CUDA loaded:

```julia
using CUDA, Test, GeoDynamo
@assert GeoDynamo.gpu_functional()     # must be true on the GPU box
include("test/gpu_phase0_foundation.jl")
```

**Phase 0 passes when, on the GPU box:** `gpu_functional()==true`, `GPU()` builds, physical & spectral fields allocate as zero `CuArray`s of the right shape, and the **host→device→host roundtrip is bit-identical** for both field kinds. Report any failure (esp. scalar-indexing errors or shape mismatches) for fixes before Phase 1.

---

## Self-Review

**Spec coverage (Phase 0 row of the design doc):** "CUDA wiring, CuArray field containers, host↔device transfer, device mgmt" → Task 1 (`gpu_functional` + ext wiring), Task 2 (`GPU()` device selection), Tasks 3–4 (containers + device allocation), Task 5 (transfer), Task 6 (regression). Gate "field host→device→host bit-identical; CPU path unchanged" → Task 5 (roundtrip gate) + Task 6 (CPU regression). Covered.

**Placeholder scan:** none — every code step has complete code; every run step has the exact command + expected result.

**Type consistency:** `gpu_functional`, `GPU()`, `_gpu_default_backend`, `GPUPhysicalField{T,A}(config,nlat,nlon,nr,data)`, `GPUSpectralField{T,A}(config,nlm,nr,data_real,data_imag)`, `allocate_gpu_physical_field(T,arch,config,nr)`, `allocate_gpu_spectral_field(CT,arch,config,nr)`, `field_to_host`/`field_to_device` — names + signatures consistent across tasks and the File-Structure interface block. `on_architecture`/`arch_zeros`/`get_backend`/`GPU{B}` reused from `core/architecture.jl` as-is.
