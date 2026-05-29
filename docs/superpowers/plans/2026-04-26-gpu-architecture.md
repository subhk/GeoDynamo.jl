# GPU Architecture (Oceananigans Style) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GeoDynamo's `GPUBackendState` / `Ref{GPUBackendState}` GPU system with an Oceananigans-style architecture type hierarchy (`AbstractArchitecture`, `CPU`, `GPU{B}`) so GPU dispatch happens through Julia's type system instead of 16 `Function`-typed fields.

**Architecture:** Define `CPU`/`GPU{B<:KernelAbstractions.Backend}` concrete types. Parameterize `SolverBackend{A}`, `TransformWorkspace{T,A}`, `SolverRuntime{T,A}`, and `SolverState{T,A}` by architecture. Replace all `if uses_gpu(config) && solver_gpu_device() !== :cuda` runtime branches with architecture-dispatched function overloads. Delete the global `_GPU_BACKEND` Ref, `GPUBackendState`, and all interop aliases that wrap it.

**Tech Stack:** Julia 1.x, KernelAbstractions.jl, Adapt.jl, CUDA.jl (optional, loaded by user), SHTnsKit.jl

---

## File Map

| File | Change |
|---|---|
| `src/core/architecture.jl` | **New** — `AbstractArchitecture`, `CPU`, `GPU{B}`, `arch_zeros`, `on_architecture`, `get_backend` |
| `src/GeoDynamo.jl` | Add include + exports for architecture types; delete GPUBackendState block (lines 19–244) and GPU exports (lines 397–398) |
| `src/solver/backend.jl` | Parameterize `SolverBackend{A}`, `TransformWorkspace{T,A}`, `SolverRuntime{T,A}`; update `create_solver_backend`, `solver_create_shtns_config`, `create_transform_workspace`, `create_solver_runtime` |
| `src/solver/state.jl` | Parameterize `SolverState{T,A}`; update field types and `show` method |
| `src/solver/mainloop.jl` | Update `initialize_solver_state` to convert `params.architecture::Symbol` → arch instance |
| `src/solver/interop.jl` | Delete GPU alias section (lines 25–43) and GPU wrapper functions (lines 65–94) |
| `src/solver/numerics.jl` | Add arch-dispatched overloads of `sht_synthesis`, `sht_analysis`, `sht_vector_synthesis`, `sht_vector_analysis` and 4 buffer operations; delete `uses_gpu`; update `solver_workspace_zeros` |
| `test/gpu_backend.jl` | Rewrite — remove `_GPU_BACKEND` / `with_gpu_backend` tests; add architecture type + struct field tests |
| `test/stability_regressions.jl` | Add: `!hasfield(SolverBackend{CPU}, something_old)`, `fieldtype(SolverBackend{CPU}, :architecture) === CPU` |

---

## Task 1: Define `AbstractArchitecture`, `CPU`, `GPU{B}` in `src/core/architecture.jl`

**Goal:** Create the foundation architecture type hierarchy with allocation and array-movement primitives; export them from the module.

**Files:**
- Create: `src/core/architecture.jl`
- Modify: `src/GeoDynamo.jl`
- Test: `test/gpu_backend.jl`

**Acceptance Criteria:**
- [ ] `CPU <: AbstractArchitecture` is defined
- [ ] `GPU{B} <: AbstractArchitecture` is defined with a `backend::B` field
- [ ] `arch_zeros(CPU(), Float64, 2, 3) == zeros(Float64, 2, 3)`
- [ ] `on_architecture(CPU(), [1,2,3]) isa Array`
- [ ] `get_backend(CPU()) isa KernelAbstractions.CPU`
- [ ] `GPU`, `CPU`, `AbstractArchitecture`, `arch_zeros`, `on_architecture`, `get_backend` are exported from `GeoDynamo`

**Verify:** `grep -n "AbstractArchitecture\|^struct CPU\|^struct GPU" src/core/architecture.jl` → 3 hits; run `julia --project -e 'using GeoDynamo; @assert CPU() isa AbstractArchitecture; @assert arch_zeros(CPU(), Float64, 2) == zeros(Float64, 2); println("ok")'` → `ok`

**Steps:**

- [ ] **Step 1: Create `src/core/architecture.jl`**

```julia
using KernelAbstractions
using Adapt

abstract type AbstractArchitecture end

struct CPU <: AbstractArchitecture end

struct GPU{B} <: AbstractArchitecture
    backend::B
end

"""
    arch_zeros(arch, FT, dims...)

Allocate a zero-filled array on the given architecture.
"""
arch_zeros(::CPU, FT::DataType, dims...) = zeros(FT, dims...)
arch_zeros(g::GPU, FT::DataType, dims...) = KernelAbstractions.zeros(g.backend, FT, dims...)

"""
    on_architecture(arch, array)

Move `array` to the device associated with `arch`.
"""
on_architecture(::CPU, a) = Array(a)
on_architecture(g::GPU, a) = Adapt.adapt(g.backend, a)

"""
    get_backend(arch)

Return the `KernelAbstractions.Backend` for the given architecture.
"""
get_backend(::CPU) = KernelAbstractions.CPU()
get_backend(g::GPU) = g.backend
```

- [ ] **Step 2: Add include in `src/GeoDynamo.jl`**

Find the line `using JLD2` (line 14). After it, add:

```julia
    include("core/architecture.jl")
```

Make sure `src/core/` directory exists first (check with `ls src/core/` or `isdir("src/core")`; create if absent).

- [ ] **Step 3: Add exports in `src/GeoDynamo.jl`**

Find the exports block (around line 256 where `SHTnsKitConfig` is exported). Add a new export line near the top of the exports:

```julia
    export AbstractArchitecture, CPU, GPU, arch_zeros, on_architecture, get_backend
```

- [ ] **Step 4: Write failing test in `test/gpu_backend.jl`**

Replace the entire file content with:

```julia
using Test

@testset "Architecture types" begin
    @testset "CPU <: AbstractArchitecture" begin
        @test CPU() isa AbstractArchitecture
        @test CPU() isa CPU
    end

    @testset "GPU{B} <: AbstractArchitecture" begin
        @test GPU{Nothing} <: AbstractArchitecture
        g = GPU(nothing)
        @test g isa GPU
        @test g.backend === nothing
    end

    @testset "arch_zeros on CPU" begin
        a = arch_zeros(CPU(), Float64, 3, 4)
        @test a == zeros(Float64, 3, 4)
        @test a isa Matrix{Float64}
    end

    @testset "on_architecture CPU returns Array" begin
        a = [1, 2, 3]
        @test on_architecture(CPU(), a) isa Array
        @test on_architecture(CPU(), a) == a
    end

    @testset "get_backend CPU" begin
        import KernelAbstractions
        @test get_backend(CPU()) isa KernelAbstractions.CPU
    end

    @testset "Exports from GeoDynamo" begin
        @test Base.isexported(GeoDynamo, :AbstractArchitecture)
        @test Base.isexported(GeoDynamo, :CPU)
        @test Base.isexported(GeoDynamo, :GPU)
        @test Base.isexported(GeoDynamo, :arch_zeros)
        @test Base.isexported(GeoDynamo, :on_architecture)
        @test Base.isexported(GeoDynamo, :get_backend)
    end
end
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd /Users/subha/Documents/GitHub/GeoDynamo.jl && julia --project -e 'include("test/gpu_backend.jl")'`

Expected: errors about `CPU` not defined (before implementation) or import errors.

- [ ] **Step 6: Verify tests pass after implementation**

After creating the file and adding include/exports, re-run:

`julia --project -e 'using GeoDynamo; include("test/gpu_backend.jl")'`

Expected: all tests pass, `ok` printed.

---

## Task 2: Parameterize `SolverBackend{A}`, `TransformWorkspace{T,A}`, `SolverRuntime{T,A}`, `SolverState{T,A}`

**Goal:** Thread the architecture type parameter through the four key solver structs so Julia can specialize solver dispatch on architecture at compile time.

**Files:**
- Modify: `src/solver/backend.jl`
- Modify: `src/solver/state.jl`
- Modify: `src/solver/mainloop.jl`

**Acceptance Criteria:**
- [ ] `SolverBackend{CPU}` constructs and has field `architecture::CPU`
- [ ] `TransformWorkspace{Float64, CPU}` constructs and has field `arch::CPU` (not `device::Symbol`)
- [ ] `SolverRuntime{Float64, CPU}` constructs
- [ ] `SolverState{Float64, CPU}` constructs
- [ ] `solver_create_shtns_config` dispatches on arch type (no `solver_gpu_device()` call)
- [ ] `create_transform_workspace` takes arch directly (no `backend.architecture === :gpu` branch)

**Verify:** `grep -n "architecture === :gpu\|solver_gpu_device\|device::Symbol" src/solver/backend.jl` → empty

**Steps:**

- [ ] **Step 1: Update `SolverBackend` in `src/solver/backend.jl`**

Find `struct SolverBackend` (line 10). Replace:
```julia
struct SolverBackend
    parameters::SolverParameters
    architecture::Symbol
    shtns_config::SHTnsConfigType
    outer_core_domain::RadialDomainType
    inner_core_domain::Union{RadialDomainType, Nothing}
    rank::Int
    process_count::Int
end
```
with:
```julia
struct SolverBackend{A<:AbstractArchitecture}
    parameters::SolverParameters
    architecture::A
    shtns_config::SHTnsConfigType
    outer_core_domain::RadialDomainType
    inner_core_domain::Union{RadialDomainType, Nothing}
    rank::Int
    process_count::Int
end
```

- [ ] **Step 2: Update `TransformWorkspace` in `src/solver/backend.jl`**

Find `struct TransformWorkspace{T}` (line 97). Replace:
```julia
struct TransformWorkspace{T}
    device::Symbol
    buffers::SolverTransformBuffers{T}
end

TransformWorkspace{T}(device::Symbol) where {T} =
    TransformWorkspace{T}(device, SolverTransformBuffers{T}())
```
with:
```julia
struct TransformWorkspace{T, A<:AbstractArchitecture}
    arch::A
    buffers::SolverTransformBuffers{T}
end

TransformWorkspace{T}(arch::A) where {T, A<:AbstractArchitecture} =
    TransformWorkspace{T,A}(arch, SolverTransformBuffers{T}())
```

- [ ] **Step 3: Update `SolverRuntime` in `src/solver/backend.jl`**

Find `struct SolverRuntime{T}` (line 113). Replace:
```julia
struct SolverRuntime{T}
    velocity::VelocityFieldsType{T}
    magnetic::MagneticFieldsType{T}
    temperature::TemperatureFieldType{T}
    composition::Union{CompositionFieldType{T}, Nothing}
    gradient_workspace::SolverGradientWorkspace{T}
    transform_workspace::TransformWorkspace{T}
    shtns_config::SHTnsConfigType
    𝒟ᵒᶜ::RadialDomainType
    𝒟ⁱᶜ::RadialDomainType
    timestep_state::SolverTimestepState
end
```
with:
```julia
struct SolverRuntime{T, A<:AbstractArchitecture}
    velocity::VelocityFieldsType{T}
    magnetic::MagneticFieldsType{T}
    temperature::TemperatureFieldType{T}
    composition::Union{CompositionFieldType{T}, Nothing}
    gradient_workspace::SolverGradientWorkspace{T}
    transform_workspace::TransformWorkspace{T,A}
    shtns_config::SHTnsConfigType
    𝒟ᵒᶜ::RadialDomainType
    𝒟ⁱᶜ::RadialDomainType
    timestep_state::SolverTimestepState
end
```

- [ ] **Step 4: Update `SolverState` in `src/solver/state.jl`**

Find `mutable struct SolverState{T}` (line 149). Replace:
```julia
mutable struct SolverState{T}
    parameters::SolverParameters
    backend::SolverBackend
    fields::SolverFields{T}
    topography::SolverTopographyState{T}
    runtime::SolverRuntime{T}
    implicit_matrices::Dict{Symbol, ImplicitMatrixSet{T}}
    timestep_caches::TimestepCaches{T}
    energy_tracker::SolverEnergyTracker
    solenoidal_monitor::SolverSolenoidalMonitor
    time::Float64
    step::Int
    is_initialized::Bool
end
```
with:
```julia
mutable struct SolverState{T, A<:AbstractArchitecture}
    parameters::SolverParameters
    backend::SolverBackend{A}
    fields::SolverFields{T}
    topography::SolverTopographyState{T}
    runtime::SolverRuntime{T,A}
    implicit_matrices::Dict{Symbol, ImplicitMatrixSet{T}}
    timestep_caches::TimestepCaches{T}
    energy_tracker::SolverEnergyTracker
    solenoidal_monitor::SolverSolenoidalMonitor
    time::Float64
    step::Int
    is_initialized::Bool
end
```

Also update `SolverState.show` (line 164): the line `_solver_print_row(io, "architecture", state.backend.architecture)` now prints an `AbstractArchitecture` instance — this will print its type name via Julia's default `show`, which is fine.

Also update any functions dispatching on `SolverState{T}` in `state.jl`:
- `_synchronize_solver_views!(state::SolverState{T}) where T` → `where {T,A}`
- `GeoDynamo.extract_all_fields(state::SolverState{T}) where {T}` → `where {T,A}`

- [ ] **Step 5: Update `solver_create_shtns_config` in `src/solver/backend.jl`**

Find `function solver_create_shtns_config(params::SolverParameters)` (line 158). Replace the entire function with arch-dispatched overloads:

```julia
function solver_create_shtns_config(::CPU, params::SolverParameters)
    return SOLVER_SHTNS_CONFIG_BUILDER(
        lmax=params.lmax,
        mmax=params.mmax,
        nlat=params.nlat,
        nlon=params.nlon,
        nr=params.nr,
        optimize_decomp=true,
        device=:cpu,
    )
end

function solver_create_shtns_config(::GPU, params::SolverParameters)
    # Non-CUDA GPU: SHTnsKit has no native support; use CPU SHTns config.
    # Transforms run on CPU; physical data arrays live on the GPU device.
    return SOLVER_SHTNS_CONFIG_BUILDER(
        lmax=params.lmax,
        mmax=params.mmax,
        nlat=params.nlat,
        nlon=params.nlon,
        nr=params.nr,
        optimize_decomp=true,
        device=:cpu,
    )
end

# CUDA-specific overload — uses SHTnsKit's native CUDA config
function solver_create_shtns_config(::GPU{<:Any}, params::SolverParameters)
    # Default GPU uses CPU SHTns; CUDA extension overrides this if CUDA.jl is loaded.
    return solver_create_shtns_config(CPU(), params)
end
```

Note: A CUDA.jl extension can add:
```julia
function GeoDynamo.solver_create_shtns_config(::GPU{CUDA.CUDABackend}, params)
    return SOLVER_SHTNS_CONFIG_BUILDER(..., device=:cuda)
end
```

- [ ] **Step 6: Update `create_transform_workspace` in `src/solver/backend.jl`**

Find `function create_transform_workspace(::Type{T}, backend::SolverBackend)` (line 348). Replace:
```julia
function create_transform_workspace(::Type{T}, backend::SolverBackend) where T
    device = backend.architecture === :gpu ? solver_gpu_device() : :cpu
    return TransformWorkspace{T}(device)
end
```
with:
```julia
function create_transform_workspace(::Type{T}, backend::SolverBackend{A}) where {T, A}
    return TransformWorkspace{T}(backend.architecture)
end
```

- [ ] **Step 7: Update `create_solver_backend` in `src/solver/backend.jl`**

Find `function create_solver_backend(params::SolverParameters)` (line 204). Replace with an arch-parameterized version:

```julia
function create_solver_backend(arch::AbstractArchitecture, params::SolverParameters)
    cfg = solver_create_shtns_config(arch, params)
    outer_core_domain, inner_core_domain = solver_create_radial_domains(params)

    return SolverBackend(
        params,
        arch,
        cfg,
        outer_core_domain,
        inner_core_domain,
        solver_backend_rank(),
        solver_backend_process_count(),
    )
end

# Convenience: convert params.architecture::Symbol → AbstractArchitecture
function create_solver_backend(params::SolverParameters)
    arch = params.architecture === :gpu ? GPU(nothing) : CPU()
    return create_solver_backend(arch, params)
end
```

(The `GPU(nothing)` is a placeholder that works without CUDA. CUDA.jl extension provides the real arch.)

Also delete the validation block that called `gpu_backend_loaded()` and `gpu_backend_available()` — those functions will be deleted in Task 3.

- [ ] **Step 8: Update `create_solver_runtime` in `src/solver/backend.jl`**

Find `function create_solver_runtime` (line 411). Update the `SolverRuntime{T}(...)` constructor call to `SolverRuntime{T,A}(...)` where `A = typeof(backend.architecture)`. Also update `load_solver_file_bcs!(runtime::SolverRuntime{T}, ...)` to `SolverRuntime{T,<:AbstractArchitecture}`:

```julia
function create_solver_runtime(::Type{T}, backend::SolverBackend{A};
                               auto_optimize::Bool=false,
                               adaptive_threading::Bool=false) where {T, A}
    solver_backend_ensure_mpi!()

    velocity, magnetic, temperature, composition = create_solver_fields(T, backend)
    gradient_workspace = create_solver_gradient_workspace(T, backend)
    transform_workspace = create_transform_workspace(T, backend)
    timestep_state = create_solver_timestep_state(backend)
    backend.shtns_config._buffers.solver_transform_workspace = transform_workspace
    # Store arch in SHTnsBuffers for use by SHT dispatch functions (Task 3)
    backend.shtns_config._buffers.transform_device = backend.architecture

    runtime = SolverRuntime{T,A}(
        velocity,
        magnetic,
        temperature,
        composition,
        gradient_workspace,
        transform_workspace,
        backend.shtns_config,
        backend.outer_core_domain,
        isnothing(backend.inner_core_domain) ? backend.outer_core_domain : backend.inner_core_domain,
        timestep_state,
    )

    load_solver_file_bcs!(runtime, backend.parameters, backend.rank)
    return runtime
end
```

Also update `load_solver_file_bcs!` signature:
```julia
function load_solver_file_bcs!(runtime::SolverRuntime{T,<:AbstractArchitecture}, params::SolverParameters, rank::Int) where T
```

- [ ] **Step 9: Update `initialize_solver_state` in `src/solver/mainloop.jl`**

Find `function initialize_solver_state`. Update the `SolverState{T}(...)` call to `SolverState{T,A}(...)`. The call to `create_solver_backend(params)` already handles the arch conversion. Just ensure the return type annotation if any is updated.

Also update function signatures that dispatch on `SolverState{T}`:
```julia
function advance_solver_step!(state::SolverState{T,<:AbstractArchitecture}) where T
function run_solver!(state::SolverState{T,<:AbstractArchitecture}) where T
function run_simulation!(state::SolverState{T,<:AbstractArchitecture}; ...) where T
```

- [ ] **Step 10: Verify**

Run:
```bash
grep -n "architecture === :gpu\|solver_gpu_device\|device::Symbol" \
    src/solver/backend.jl src/solver/state.jl src/solver/mainloop.jl
```
Expected: empty.

```bash
julia --project -e '
using GeoDynamo
b = GeoDynamo.create_solver_backend(GeoDynamo.CPU(), GeoDynamo.SolverParameters())
@assert b.architecture isa GeoDynamo.CPU
tw = GeoDynamo.TransformWorkspace{Float64}(GeoDynamo.CPU())
@assert tw.arch isa GeoDynamo.CPU
@assert !hasproperty(tw, :device)
println("ok")
'
```
Expected: `ok`.

---

## Task 3: Replace SHT dispatch + delete old GPU system

**Goal:** Replace all 8 `if uses_gpu && solver_gpu_device()` branches with arch-dispatched overloads; delete `GPUBackendState`, `_GPU_BACKEND`, and all interop wrappers.

**Files:**
- Modify: `src/solver/numerics.jl`
- Modify: `src/solver/interop.jl`
- Modify: `src/GeoDynamo.jl`
- Modify: `src/physics/nonlinear.jl`

**Acceptance Criteria:**
- [ ] `grep -rn "_GPU_BACKEND\|GPUBackendState\|register_gpu_backend\|with_gpu_backend" src/` → empty
- [ ] `grep -rn "uses_gpu\|solver_gpu_device\b" src/` → empty
- [ ] `grep -rn "SOLVER_GPU_BACKEND\|SOLVER_GPU_SCALAR\|SOLVER_GPU_VECTOR\|SOLVER_GPU_SCRATCH" src/` → empty
- [ ] CPU synthesis/analysis work: `sht_synthesis(config, coeffs)` correctly delegates to `SHTnsKit.synthesis` for CPU arch

**Verify:** `grep -rn "_GPU_BACKEND\|GPUBackendState\|register_gpu_backend\|uses_gpu\b\|solver_gpu_device\b" src/` → empty

**Steps:**

- [ ] **Step 1: Add arch-dispatched SHT overloads in `src/solver/numerics.jl`**

Find `@inline uses_gpu(config) = SHTnsKit.is_gpu_config(config.sht_config)` (line 357). Delete this line.

Find `@inline function sht_synthesis(config, coeffs_matrix)` (line 364). Replace the entire function body with:

```julia
@inline function sht_synthesis(config, coeffs_matrix)
    arch = config._buffers.transform_device::AbstractArchitecture
    return _sht_synthesis(arch, config, coeffs_matrix)
end

@inline function _sht_synthesis(::CPU, config, coeffs_matrix)
    return SHTnsKit.synthesis(config.sht_config, coeffs_matrix; real_output=true)
end

@inline function _sht_synthesis(::GPU, config, coeffs_matrix)
    # Non-CUDA GPU: SHTns config is CPU-based; run CPU synthesis.
    return SHTnsKit.synthesis(config.sht_config, coeffs_matrix; real_output=true)
end
```

Note: The CUDA overload lives in the CUDA.jl extension:
```julia
@inline function GeoDynamo._sht_synthesis(::GPU{<:CUDA.CUDABackend}, config, coeffs_matrix)
    return SHTnsKit.gpu_synthesis_safe(config.sht_config, coeffs_matrix;
        device=SHTnsKit.CUDA_DEVICE, real_output=true)
end
```

Find `@inline function sht_analysis(config, phys_slice)` (line 384). Replace body:

```julia
@inline function sht_analysis(config, phys_slice)
    arch = config._buffers.transform_device::AbstractArchitecture
    return _sht_analysis(arch, config, phys_slice)
end

@inline _sht_analysis(::CPU, config, phys_slice) =
    SHTnsKit.analysis(config.sht_config, phys_slice)

@inline _sht_analysis(::GPU, config, phys_slice) =
    SHTnsKit.analysis(config.sht_config, phys_slice)
```

Find `@inline function sht_vector_synthesis(config, pol_coeffs, tor_coeffs)` (line 410). Replace body:

```julia
@inline function sht_vector_synthesis(config, pol_coeffs, tor_coeffs)
    arch = config._buffers.transform_device::AbstractArchitecture
    return _sht_vector_synthesis(arch, config, pol_coeffs, tor_coeffs)
end

@inline _sht_vector_synthesis(::CPU, config, pol_coeffs, tor_coeffs) =
    SHTnsKit.synthesis_sphtor(config.sht_config, pol_coeffs, tor_coeffs; real_output=true)

@inline _sht_vector_synthesis(::GPU, config, pol_coeffs, tor_coeffs) =
    SHTnsKit.synthesis_sphtor(config.sht_config, pol_coeffs, tor_coeffs; real_output=true)
```

Find `@inline function sht_vector_analysis(config, vt_field, vp_field)` (line 442). Replace body:

```julia
@inline function sht_vector_analysis(config, vt_field, vp_field)
    arch = config._buffers.transform_device::AbstractArchitecture
    return _sht_vector_analysis(arch, config, vt_field, vp_field)
end

@inline _sht_vector_analysis(::CPU, config, vt_field, vp_field) =
    SHTnsKit.analysis_sphtor(config.sht_config, vt_field, vp_field)

@inline _sht_vector_analysis(::GPU, config, vt_field, vp_field) =
    SHTnsKit.analysis_sphtor(config.sht_config, vt_field, vp_field)
```

- [ ] **Step 2: Replace buffer operation dispatch in `src/solver/numerics.jl`**

Find `function fill_vector_coeff_buffer!(coeffs_buffer, spec_real, spec_imag, r_local, config)` (line 490). Replace the `if uses_gpu(config) && solver_gpu_device() !== :cuda` block:

```julia
function fill_vector_coeff_buffer!(coeffs_buffer, spec_real, spec_imag, r_local, config)
    return cpu_fill_vector_coeff_buffer!(coeffs_buffer, spec_real, spec_imag, r_local, config)
end
```

(The CPU path is always correct for both CPU and non-CUDA GPU; CUDA.jl extension can add GPU overload.)

Find `function store_vector_coefficients!(spec_real, spec_imag, coeffs_matrix, r_local, config)` (line 513). Replace body:

```julia
function store_vector_coefficients!(spec_real, spec_imag, coeffs_matrix, r_local, config)
    return cpu_store_vector_coefficients!(spec_real, spec_imag, coeffs_matrix, r_local, config)
end
```

Find `function extract_vector_component!(component_buffer, v_data, r_local, config; ...)` (line 556). Replace body:

```julia
function extract_vector_component!(
    component_buffer::Matrix{T},
    v_data,
    r_local,
    config;
    axes_local::Union{Nothing,Tuple}=nothing,
) where T
    return cpu_extract_vector_component!(component_buffer, v_data, r_local, config; axes_local=axes_local)
end
```

Find `function store_vector_components!(v_theta, v_phi, vt_field, vp_field, r_local, config; ...)` (line 620). Replace body:

```julia
function store_vector_components!(
    v_theta,
    v_phi,
    vt_field,
    vp_field,
    r_local,
    config;
    axes_local::Union{Nothing,Tuple}=nothing,
)
    return cpu_store_vector_components!(v_theta, v_phi, vt_field, vp_field, r_local, config; axes_local=axes_local)
end
```

- [ ] **Step 3: Update `solver_workspace_zeros` in `src/physics/nonlinear.jl`**

Find `@inline function solver_workspace_zeros(config, ::Type{T}, dims...) where {T}` (line 547). Replace:

```julia
@inline function solver_workspace_zeros(config, ::Type{T}, dims...) where {T}
    arch = config._buffers.transform_device::AbstractArchitecture
    return arch_zeros(arch, T, dims...)
end
```

(This replaces the `if workspace isa TransformWorkspace && workspace.device != :cpu` branch entirely.)

- [ ] **Step 4: Delete GPU aliases from `src/solver/interop.jl`**

Delete lines 25–43 (the 19 `const SOLVER_GPU_*` aliases):
```julia
const SOLVER_GPU_BACKEND_LOADED = ...
const SOLVER_GPU_BACKEND_AVAILABLE = ...
...
const SOLVER_GPU_UNAVAILABLE_ERROR = ...
```

Delete lines 65–94 (the inline GPU wrapper functions):
```julia
@inline solver_gpu_device() = ...
@inline solver_gpu_scalar_synthesis(...) = ...
...
@inline solver_gpu_unavailable_error() = ...
```

- [ ] **Step 5: Delete `GPUBackendState` block from `src/GeoDynamo.jl`**

Delete lines 19–244:
- `struct GPUBackendState` definition
- All `_default_gpu_*` functions
- `const _GPU_BACKEND = Ref{GPUBackendState}(...)`
- `register_gpu_backend!`, `restore_gpu_backend!`, `with_gpu_backend`, `gpu_backend_state`
- All 12 `gpu_*` forwarding functions

Delete lines 397–398 (GPU exports):
```julia
    export GPUBackendState, register_gpu_backend!, gpu_backend_state, restore_gpu_backend!, with_gpu_backend
    export gpu_backend_loaded, gpu_backend_available, gpu_backend_device
```

Also delete `_gpu_backend_not_loaded_error` and `_gpu_backend_unavailable_error` (lines 38–48) if not already deleted above.

- [ ] **Step 6: Verify grep guard**

```bash
grep -rn "_GPU_BACKEND\|GPUBackendState\|register_gpu_backend\|with_gpu_backend\|uses_gpu\b\|solver_gpu_device\b\|SOLVER_GPU_BACKEND\|SOLVER_GPU_SCALAR\|SOLVER_GPU_VECTOR" src/
```
Expected: empty (zero matches).

Also:
```bash
grep -rn "solver_gpu_not_loaded_error\|solver_gpu_unavailable_error\|gpu_backend_loaded\|gpu_backend_available" src/
```
Expected: empty.

---

## Task 4: Rewrite `test/gpu_backend.jl` and add regression tests

**Goal:** Replace the now-deleted `_GPU_BACKEND` / `with_gpu_backend` tests with architecture type verification; add stability regressions.

**Files:**
- Modify: `test/gpu_backend.jl`
- Modify: `test/stability_regressions.jl`

**Acceptance Criteria:**
- [ ] `test/gpu_backend.jl` passes with zero skipped tests
- [ ] No references to `_GPU_BACKEND`, `GPUBackendState`, `register_gpu_backend`, `with_gpu_backend` in tests
- [ ] Regression test asserts `SolverBackend{CPU}.architecture === CPU`
- [ ] Regression test asserts `TransformWorkspace{Float64, CPU}` has `arch` field not `device`

**Verify:** `julia --project -e 'using GeoDynamo; include("test/gpu_backend.jl"); include("test/stability_regressions.jl")'` → no errors

**Steps:**

- [ ] **Step 1: Rewrite `test/gpu_backend.jl`**

Replace the entire file with:

```julia
using Test

@testset "GPU Architecture (Oceananigans style)" begin
    @testset "Architecture type hierarchy" begin
        @test CPU() isa AbstractArchitecture
        @test GPU{Nothing} <: AbstractArchitecture

        # Old global state is gone
        @test !isdefined(GeoDynamo, :_GPU_BACKEND)
        @test !isdefined(GeoDynamo, :GPUBackendState)
        @test !Base.isexported(GeoDynamo, :register_gpu_backend!)
        @test !Base.isexported(GeoDynamo, :with_gpu_backend)
        @test !Base.isexported(GeoDynamo, :gpu_backend_loaded)
        @test !Base.isexported(GeoDynamo, :gpu_backend_available)
    end

    @testset "arch_zeros" begin
        a = arch_zeros(CPU(), Float64, 3, 4)
        @test a isa Matrix{Float64}
        @test all(iszero, a)
        @test size(a) == (3, 4)
    end

    @testset "on_architecture CPU" begin
        src = [1.0, 2.0, 3.0]
        dst = on_architecture(CPU(), src)
        @test dst isa Vector{Float64}
        @test dst == src
    end

    @testset "SolverBackend carries architecture" begin
        @test fieldtype(GeoDynamo.SolverBackend{CPU}, :architecture) === CPU
        params = GeoDynamo.SolverParameters()
        b = GeoDynamo.create_solver_backend(CPU(), params)
        @test b.architecture isa CPU
        @test !hasproperty(b, :loaded)
        @test !hasproperty(b, :available)
    end

    @testset "TransformWorkspace carries arch not device symbol" begin
        tw = GeoDynamo.TransformWorkspace{Float64}(CPU())
        @test tw.arch isa CPU
        @test !hasproperty(tw, :device)
    end
end
```

- [ ] **Step 2: Add regression tests to `test/stability_regressions.jl`**

Find the existing type-stability testsets (search for `"TimestepCaches replaces dict caches"` or `"SHTnsBuffers replaces _buffer_cache"`). After the last existing testset in the file, add:

```julia
@testset "Oceananigans-style GPU architecture" begin
    # Architecture types
    @test CPU <: AbstractArchitecture
    @test GPU{Nothing} <: AbstractArchitecture
    @test GPU{Nothing}(nothing) isa GPU

    # Old GPU global state absent
    @test !isdefined(GeoDynamo, :_GPU_BACKEND)
    @test !isdefined(GeoDynamo, :GPUBackendState)
    @test !isdefined(GeoDynamo, :register_gpu_backend!)

    # SolverBackend parameterized by architecture
    @test GeoDynamo.SolverBackend{CPU} isa DataType
    @test fieldtype(GeoDynamo.SolverBackend{CPU}, :architecture) === CPU

    # TransformWorkspace has arch not device
    @test hasfield(GeoDynamo.TransformWorkspace{Float64, CPU}, :arch)
    @test !hasfield(GeoDynamo.TransformWorkspace{Float64, CPU}, :device)
    @test fieldtype(GeoDynamo.TransformWorkspace{Float64, CPU}, :arch) === CPU
end
```

- [ ] **Step 3: Run full verify**

```bash
julia --project -e '
using GeoDynamo
include("test/gpu_backend.jl")
include("test/stability_regressions.jl")
println("All tests passed")
'
```

Expected: `All tests passed`

Also run the grep guard one final time:
```bash
grep -rn "_GPU_BACKEND\|GPUBackendState\|register_gpu_backend\|with_gpu_backend\|uses_gpu\b\|solver_gpu_device\b" src/
```
Expected: empty.
