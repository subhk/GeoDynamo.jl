# GPU Architecture Design — Oceananigans Style

**Date:** 2026-04-26  
**Status:** Approved

---

## Goal

Replace GeoDynamo's current GPU backend (a `GPUBackendState` struct with 16 `Function`-typed fields, a global `Ref`, and runtime `Symbol` comparisons) with an Oceananigans-style architecture type hierarchy. GPU dispatch becomes Julia multiple dispatch on concrete architecture types, eliminating all global state and type instability in the GPU code path.

## Target GPU Support

- **CPU** — always available, default
- **CUDA** — via `CUDA.jl` + native `SHTnsKit.jl` GPU transforms
- **Abstract GPU** (Metal, ROCm, etc.) — via `KernelAbstractions.jl`; SHT transforms error on non-CUDA GPU until SHTnsKit gains broader backend support

## Architecture

### 1. Architecture Types — `src/core/architecture.jl`

New file. Foundation for all GPU dispatch.

```julia
abstract type AbstractArchitecture end

struct CPU <: AbstractArchitecture end

struct GPU{B} <: AbstractArchitecture
    backend::B  # KernelAbstractions.Backend instance
end

# Allocation
arch_zeros(::CPU, FT, dims...) = zeros(FT, dims...)
arch_zeros(g::GPU, FT, dims...) = KernelAbstractions.zeros(g.backend, FT, dims...)

# Array movement
on_architecture(::CPU, a) = Array(a)
on_architecture(g::GPU, a) = Adapt.adapt(g.backend, a)

# Backend extraction (for @kernel dispatch)
get_backend(::CPU) = KernelAbstractions.CPU()
get_backend(g::GPU) = g.backend
```

`GPU{B}` carries the concrete backend as a type parameter so Julia specializes at compile time. No global state, no `Ref`, no `Function` fields.

### 2. Struct Parameterization — `src/solver/backend.jl`

Three structs gain an architecture type parameter:

**`SolverBackend{A<:AbstractArchitecture}`**
- `architecture::A` (was `architecture::Symbol`)
- All other fields unchanged

**`TransformWorkspace{T, A<:AbstractArchitecture}`**
- `arch::A` (was `device::Symbol`)
- `buffers::SolverTransformBuffers{T}` unchanged
- Constructor: `TransformWorkspace{T}(arch::A) where {T,A} = TransformWorkspace{T,A}(arch, SolverTransformBuffers{T}())`

**`SolverRuntime{T, A<:AbstractArchitecture}`**
- Holds `transform_workspace::TransformWorkspace{T,A}`
- `SolverState{T,A}` follows transitively (holds `SolverRuntime{T,A}`)

`create_transform_workspace` passes `backend.architecture` directly (already an `AbstractArchitecture` instance — no Symbol-to-device conversion needed).

### 3. GPU Global State Removal

**Deleted from `src/GeoDynamo.jl`:**
- `struct GPUBackendState` (16 `Function`-typed fields)
- `const _GPU_BACKEND = Ref{GPUBackendState}(...)`
- `register_gpu_backend!`, `restore_gpu_backend!`, `gpu_backend_state`, `with_gpu_backend`
- All 12 `gpu_scalar_synthesis`, `gpu_scalar_analysis`, ... forwarding functions
- Export of all the above

**Deleted from `src/solver/interop.jl`:**
- 13 `const SOLVER_GPU_*` aliases
- 12 inline `solver_gpu_*()` wrapper functions

### 4. SHT Dispatch — `src/solver/numerics.jl`

Replace 8 `if uses_gpu(config) && solver_gpu_device() !== :cuda` blocks with architecture-dispatched functions. `uses_gpu` and `solver_gpu_device` are deleted.

**Scalar synthesis:**
```julia
sht_synthesis!(::CPU, config, coeffs, output) =
    SHTnsKit.synthesis!(config.sht_config, output, coeffs)
sht_synthesis!(::GPU{<:CUDA.CUDABackend}, config, coeffs, output) =
    SHTnsKit.gpu_synthesis_safe(config.sht_config, coeffs; real_output=output)
sht_synthesis!(::GPU, config, coeffs, output) =
    SHTnsKit.synthesis!(config.sht_config, output, coeffs)  # CPU SHTns; data moves CPU↔GPU around each call
```

Same pattern for `sht_analysis!`, `sht_vector_synthesis!`, `sht_vector_analysis!`.

For non-CUDA GPU, `solver_create_shtns_config` creates a CPU SHTns config (SHTnsKit has no native Metal/ROCm support), so the transforms correctly use the CPU overload above. Physical data arrays live on the GPU device; the transform layer copies slices to CPU, transforms, and copies back. This is correct but slower than native GPU transforms — accepted until SHTnsKit gains broader backend support.

**Buffer operations** (`fill_scalar_coeff_buffer`, `extract_physical_slice`, `store_physical_slice`, `fill_vector_coeff_buffer`, `store_vector_coefficients`, `extract_vector_component`, `store_vector_components`):
- CPU overloads call existing `solver_cpu_*` implementations
- CUDA overloads call `SHTnsKit.gpu_*` equivalents
- Non-CUDA GPU overloads `error(...)` until implemented

**Allocation:** `solver_workspace_zeros(config, FT, dims...)` replaced by `arch_zeros(arch, FT, dims...)` at call sites. `arch` obtained from `runtime.transform_workspace.arch`.

### 5. `create_solver_backend` update

```julia
function create_solver_backend(arch::A, params::SolverParameters) where {A<:AbstractArchitecture}
    # was: architecture::Symbol + if architecture === :gpu checks
    if arch isa GPU && !CUDA.functional()
        @warn "GPU architecture requested but CUDA not functional; falling back to CPU"
        arch = CPU()
    end
    shtns_config = solver_create_shtns_config(arch, params)
    ...
    return SolverBackend{A}(arch, shtns_config, ...)
end
```

`solver_create_shtns_config` dispatches on `arch`:
```julia
solver_create_shtns_config(::CPU, params) = create_shtnskit_config(:cpu, ...)
solver_create_shtns_config(::GPU{<:CUDA.CUDABackend}, params) = create_shtnskit_config(:cuda, ...)
solver_create_shtns_config(::GPU, params) = create_shtnskit_config(:cpu, ...)  # SHT always CPU for non-CUDA
```

---

## File Map

| File | Change |
|---|---|
| `src/core/architecture.jl` | **New** — `AbstractArchitecture`, `CPU`, `GPU{B}`, `arch_zeros`, `on_architecture`, `get_backend` |
| `src/GeoDynamo.jl` | Delete `GPUBackendState`, `_GPU_BACKEND`, all forwarding functions; include `architecture.jl`; export `CPU`, `GPU`, `AbstractArchitecture`, `arch_zeros`, `on_architecture` |
| `src/solver/backend.jl` | Parameterize `SolverBackend{A}`, `TransformWorkspace{T,A}`, `SolverRuntime{T,A}`; update `create_solver_backend`, `create_transform_workspace` |
| `src/solver/state.jl` | `SolverState{T,A}` (transitively from `SolverRuntime`) |
| `src/solver/mainloop.jl` | Thread `arch` through `initialize_solver_state` |
| `src/solver/interop.jl` | Delete GPU alias section (lines 25–94) |
| `src/solver/numerics.jl` | Replace 8 `if uses_gpu` blocks with `sht_*!` / buffer operation dispatch; delete `uses_gpu`, `solver_gpu_device` |
| `src/physics/nonlinear.jl` | Replace `solver_workspace_zeros(config, FT, dims...)` call sites with `arch_zeros(arch, FT, dims...)` |
| `test/gpu_backend.jl` | Rewrite — remove `with_gpu_backend` tests; add architecture type and struct field tests |
| `test/stability_regressions.jl` | Add: `!hasfield(SolverBackend, :architecture) || fieldtype(SolverBackend{CPU}, :architecture) === CPU` |

---

## Acceptance Criteria

- `GPUBackendState` is not defined anywhere in the codebase
- `_GPU_BACKEND` Ref does not exist
- `register_gpu_backend!` and `with_gpu_backend` are not exported or defined
- `SolverBackend{CPU}` and `SolverBackend{GPU{B}}` both construct without error
- `TransformWorkspace{Float64, CPU}` has field `arch::CPU`, not `device::Symbol`
- `arch_zeros(CPU(), Float64, 3, 4) == zeros(Float64, 3, 4)`
- `on_architecture(CPU(), cu_array)` returns a CPU `Array`
- All existing CPU tests pass unchanged
- `grep -rn "_GPU_BACKEND\|GPUBackendState\|register_gpu_backend\|solver_gpu_device\|uses_gpu" src/` → empty

---

## What Is Not Changed

- `SHTnsKitConfig` and `SHTnsBuffers` — unchanged (Task 2 already complete)
- `TimestepCaches{T}`, `SolverTransformBuffers{T}` — unchanged
- `solver_cpu_*` CPU implementations in `numerics.jl` — kept as-is, called by CPU dispatch overloads
- MPI / PencilArrays integration — unaffected
- `SolverParameters.architecture::Symbol` (user-facing config) — kept as `:cpu`/`:gpu`; converted to `CPU()`/`GPU(CUDA.CUDABackend())` at `initialize_simulation` time. Users who need a non-CUDA GPU backend pass an `AbstractArchitecture` instance directly to `initialize_simulation` instead of using `SolverParameters`.
