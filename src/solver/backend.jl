"""
    SolverBackend

Immutable description of the backend resources used by a solver run.

It records the selected architecture, SHTnsKit transform configuration, radial
domains, and MPI rank/process metadata. `create_solver_backend(...)` is the
normal constructor used by the public solver initialization path.
"""
struct SolverBackend{A<:AbstractArchitecture}
    parameters::SolverParameters
    architecture::A
    shtns_config::SHTnsConfigType
    outer_core_domain::RadialDomainType
    inner_core_domain::Union{RadialDomainType, Nothing}
    rank::Int
    process_count::Int
end

"""
    SolverTopographyState{T}

Topography and Stefan-condition state owned by the rewritten solver.

The solver keeps this separate from the core field runtime so topographic
coupling can be enabled or updated without rebuilding the spectral fields.
"""
mutable struct SolverTopographyState{T<:AbstractFloat}
    config::TopographyAPI.TopographyCouplingConfig
    data::Union{TopographyAPI.TopographyData{T}, Nothing}
    stefan::Union{TopographyAPI.StefanState{T}, Nothing}
end

"""
    SolverTimestepState

Minimal mutable state needed by the solver timestep loop.

This tracks the current time, timestep, step counters, and simple convergence
status without exposing the larger legacy runtime state.
"""
mutable struct SolverTimestepState
    time::Float64
    dt::Float64
    step::Int
    iteration::Int
    error::Float64
    converged::Bool
    needs_ab2_bootstrap::Bool
end

"""
    SolverGradientWorkspace{T}

Scratch spectral fields used to build scalar gradients for nonlinear terms.
"""
struct SolverGradientWorkspace{T}
    ∇θ_spec::SpectralFieldType{T}
    ∇φ_spec::SpectralFieldType{T}
    ∇r_spec::SpectralFieldType{T}
    theta_full_real::Vector{T}
    theta_full_imag::Vector{T}
end

"""
    SolverTransformBuffers{T}

Typed scratch storage replacing `TransformWorkspace{T}.cache::Dict{Symbol,Any}`.
Each field corresponds to a key formerly stored in the Dict.
"""
mutable struct SolverTransformBuffers{T}
    vector_coeffs_1           :: Union{Matrix{ComplexF64}, Nothing}
    vector_coeffs_2           :: Union{Matrix{ComplexF64}, Nothing}
    vector_coeffs_gathered_1  :: Union{Matrix{ComplexF64}, Nothing}
    vector_coeffs_gathered_2  :: Union{Matrix{ComplexF64}, Nothing}
    pol_rad_coeffs            :: Union{Matrix{ComplexF64}, Nothing}
    vector_component_vt       :: Union{Matrix{Float64}, Nothing}
    vector_component_vp       :: Union{Matrix{Float64}, Nothing}
    generic_slice             :: Union{Matrix{Float64}, Nothing}
    generic_slice_gathered    :: Union{Matrix{Float64}, Nothing}
    coeffs_buffer             :: Union{Matrix{ComplexF64}, Nothing}
    coeffs_gathered           :: Union{Matrix{ComplexF64}, Nothing}
end

SolverTransformBuffers{T}() where T = SolverTransformBuffers{T}(
    nothing, nothing, nothing, nothing, nothing,
    nothing, nothing, nothing, nothing, nothing, nothing,
)

"""
    TransformWorkspace{T}

Scratch storage owned by one solver runtime for transform-side gather/scatter
work.

GPU-marked runs route these allocations through backend hooks so the same
solver code can use either CPU or backend-provided scratch storage.
"""
struct TransformWorkspace{T, A<:AbstractArchitecture}
    arch::A
    buffers::SolverTransformBuffers{T}
end

TransformWorkspace{T}(arch::A) where {T, A<:AbstractArchitecture} =
    TransformWorkspace{T,A}(arch, SolverTransformBuffers{T}())

"""
    SolverRuntime{T}

The concrete field and workspace objects stepped by the rewritten solver.

`SolverState` wraps this together with parameters, topography state, timestep
matrices, caches, and diagnostics.
"""
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

function Base.show(io::IO, ::MIME"text/plain", backend::SolverBackend)
    cfg = backend.shtns_config
    println(io, "GeoDynamo SolverBackend")
    println(io, "├─ transforms")
    _solver_print_row(io, "backend", "SHTnsKit + PencilArrays + PencilFFTs")
    _solver_print_row(io, "architecture", backend.architecture)
    _solver_print_row(io, "compute device", SHTnsKit.get_config_device(cfg.sht_config))
    _solver_print_row(io, "ℓmax / mmax", "$(cfg.lmax) / $(cfg.mmax)")
    _solver_print_row(io, "Nθ × Nφ", "$(cfg.nlat) × $(cfg.nlon)")
    _solver_print_row(io, "spectral modes", cfg.nlm)
    println(io, "├─ domains")
    _solver_print_row(io, "outer core Nᵣ", backend.outer_core_domain.N)
    _solver_print_row(io, "inner core Nᵣ", isnothing(backend.inner_core_domain) ? "none" : backend.inner_core_domain.N)
    println(io, "└─ parallel")
    _solver_print_row(io, "rank", backend.rank)
    _solver_print_row(io, "processes", backend.process_count)
end

function Base.show(io::IO, ::MIME"text/plain", topography::SolverTopographyState)
    println(io, "GeoDynamo SolverTopographyState")
    println(io, "├─ coupling")
    _solver_print_row(io, "enabled", _solver_yesno(topography.config.enabled))
    _solver_print_row(io, "ε", topography.config.epsilon)
    _solver_print_row(io, "velocity", _solver_yesno(topography.config.velocity_coupling))
    _solver_print_row(io, "magnetic", _solver_yesno(topography.config.magnetic_coupling))
    _solver_print_row(io, "thermal", _solver_yesno(topography.config.thermal_coupling))
    println(io, "└─ boundaries")
    _solver_print_row(io, "ICB topography", topography.data === nothing || topography.data.icb === nothing ? "none" : "loaded")
    _solver_print_row(io, "OCB topography", topography.data === nothing || topography.data.cmb === nothing ? "none" : "loaded")
    _solver_print_row(io, "Stefan state", isnothing(topography.stefan) ? "inactive" : "ready")
end

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
    # Physical data arrays live on the GPU; transforms run on CPU.
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

function solver_architecture_from_symbol(architecture::Symbol)
    architecture === :cpu && return CPU()
    architecture === :gpu && return GPU(nothing)
    throw(ArgumentError("architecture = $(architecture) must be :cpu or :gpu"))
end

function solver_scale_radial_domain(domain::RadialDomainType, radius_scale::Real)
    scale = Float64(radius_scale)
    scale > 0 || throw(ArgumentError("inner-core radius scale must be positive, got $scale"))

    r = copy(domain.r)
    r[:, 4] .*= scale

    for p in axes(r, 2)
        p == 4 && continue
        power = p - 4
        for i in axes(r, 1)
            r_val = r[i, 4]
            r[i, p] = r_val == 0.0 && power < 0 ? 0.0 : r_val ^ power
        end
    end

    dr_matrices = [
        copy(matrix) ./ (scale ^ order)
        for (order, matrix) in enumerate(domain.dr_matrices)
    ]
    radial_laplacian = copy(domain.radial_laplacian) ./ (scale ^ 2)
    integration_weights = copy(domain.integration_weights) .* scale

    return RadialDomainType(
        domain.N,
        domain.local_range,
        r,
        dr_matrices,
        radial_laplacian,
        integration_weights,
    )
end

function solver_create_inner_core_domain(params::SolverParameters)
    unit_ball_domain = SOLVER_BALL_DOMAIN_BUILDER(
        params.nr_inner;
        radial_bandwidth=params.radial_bandwidth,
    )
    inner_core_radius = params.radius_ratio / (1.0 - params.radius_ratio)
    return solver_scale_radial_domain(unit_ball_domain, inner_core_radius)
end

function solver_create_radial_domains(params::SolverParameters)
    outer_core_domain =
        params.geometry === :ball ?
        SOLVER_BALL_DOMAIN_BUILDER(params.nr; radial_bandwidth=params.radial_bandwidth) :
        SOLVER_SHELL_DOMAIN_BUILDER(
            params.nr;
            radius_ratio=params.radius_ratio,
            radial_bandwidth=params.radial_bandwidth,
        )

    inner_core_domain =
        params.geometry === :shell ?
        solver_create_inner_core_domain(params) :
        nothing

    return outer_core_domain, inner_core_domain
end

"""
    create_solver_backend(params)

Validate the requested architecture and construct the backend description for a
solver run.

This is the main low-level builder behind `initialize_solver_state(...)` and is
useful for advanced workflows that want to inspect or customize backend setup
before allocating fields.
"""
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

function create_solver_backend(params::SolverParameters)
    arch = solver_architecture_from_symbol(params.architecture)
    return create_solver_backend(arch, params)
end

@inline solver_create_velocity_fields(::Type{T}, cfg, outer, pencils, params) where T =
    SOLVER_VELOCITY_FIELD_BUILDER(T, cfg, outer, pencils, pencils.spec; params)

@inline solver_create_magnetic_fields(::Type{T}, cfg, outer, inner, pencils) where T =
    SOLVER_MAGNETIC_FIELD_BUILDER(T, cfg, outer, inner, pencils, pencils.spec)

@inline solver_create_temperature_field(::Type{T}, cfg, outer, pencils) where T =
    SOLVER_TEMPERATURE_FIELD_BUILDER(T, cfg, outer, pencils, pencils.spec)

@inline solver_create_composition_field(::Type{T}, cfg, outer, pencils) where T =
    SOLVER_COMPOSITION_FIELD_BUILDER(T, cfg, outer, pencils, pencils.spec)

"""
    create_solver_fields(T, backend)

Allocate the spectral/physical field containers required by the solver runtime
for element type `T`.
"""
function create_solver_fields(::Type{T}, backend::SolverBackend{<:AbstractArchitecture}) where T
    cfg = backend.shtns_config
    outer = backend.outer_core_domain
    inner = isnothing(backend.inner_core_domain) ? outer : backend.inner_core_domain
    pencils = cfg.pencils

    velocity = solver_create_velocity_fields(T, cfg, outer, pencils, backend.parameters)
    # Allocate the magnetic storage unconditionally for now. The active-field
    # view decides whether it participates, but keeping the buffers present lets
    # the shared constructors and runtime layout stay uniform.
    magnetic = solver_create_magnetic_fields(T, cfg, outer, inner, pencils)
    temperature = solver_create_temperature_field(T, cfg, outer, pencils)
    composition =
        backend.parameters.include_composition ?
        solver_create_composition_field(T, cfg, outer, pencils) :
        nothing

    return velocity, magnetic, temperature, composition
end

function solver_build_velocity_implicit_matrices(cfg, domain, E, dt, velocity_bc_code)
    return (
        tor=SOLVER_VELOCITY_TOROIDAL_MATRIX_BUILDER(
            cfg, domain, E, dt; velocity_bc_code=velocity_bc_code, mass_coeff=E,
        ),
        pol=SOLVER_VELOCITY_POLOIDAL_MATRIX_BUILDER(
            cfg, domain, E, dt; velocity_bc_code=velocity_bc_code, mass_coeff=E,
        ),
    )
end

function solver_build_magnetic_implicit_matrices(cfg, domain, dt)
    return (
        tor=SOLVER_MAGNETIC_TOROIDAL_MATRIX_BUILDER(cfg, domain, 1.0, dt),
        pol=SOLVER_MAGNETIC_POLOIDAL_MATRIX_BUILDER(cfg, domain, 1.0, dt),
    )
end

@inline solver_build_temperature_implicit_matrix(cfg, domain, diffusivity, dt, temperature_bc_code) =
    SOLVER_TEMPERATURE_MATRIX_BUILDER(cfg, domain, diffusivity, dt; temperature_bc_code=temperature_bc_code)

@inline solver_build_composition_implicit_matrix(cfg, domain, diffusivity, dt, composition_bc_code) =
    SOLVER_COMPOSITION_MATRIX_BUILDER(cfg, domain, diffusivity, dt; composition_bc_code=composition_bc_code)

"""
    create_solver_implicit_matrices(T, backend)

Precompute the linear solve operators used by the solver timestep schemes for
element type `T`.

The returned matrices are later wrapped into the solver-owned matrix store so
the timestep loop can reuse them without rebuilding operators on each step.
"""
function create_solver_implicit_matrices(::Type{T}, backend::SolverBackend{<:AbstractArchitecture}) where T
    params = backend.parameters
    cfg = backend.shtns_config
    outer = backend.outer_core_domain

    dt = params.timestep
    E = params.Ek
    Pm = params.Pm
    Pr = params.Pr
    Sc = params.Sc
    velocity_bc_code = _velocity_bc_code(params.velocity_bcs)
    temperature_bc_code = _thermal_bc_code(params.temperature_bcs)
    composition_bc_code = _composition_bc_code(params.composition_bcs)

    # Materialize every linear solve operator once so the timestep loop only
    # selects from prebuilt matrices instead of re-deriving them per field.
    matrices = Dict{Symbol, OldImplicitMatrices{T}}()
    velocity = solver_build_velocity_implicit_matrices(cfg, outer, E, dt, velocity_bc_code)
    magnetic = solver_build_magnetic_implicit_matrices(cfg, outer, dt)
    matrices[:velocity_tor] = velocity.tor
    matrices[:velocity_pol] = velocity.pol
    matrices[:magnetic_tor] = magnetic.tor
    matrices[:magnetic_pol] = magnetic.pol
    matrices[:temperature] = solver_build_temperature_implicit_matrix(cfg, outer, Pm / Pr, dt, temperature_bc_code)

    if params.include_composition
        matrices[:composition] = solver_build_composition_implicit_matrix(cfg, outer, Pm / Sc, dt, composition_bc_code)
    end

    return matrices
end

@inline solver_create_gradient_field(::Type{T}, cfg, domain, pencil_spec) where T =
    SOLVER_SPECTRAL_FIELD_BUILDER(T, cfg, domain, pencil_spec)

function create_solver_gradient_workspace(::Type{T}, backend::SolverBackend{<:AbstractArchitecture}) where T
    cfg = backend.shtns_config
    domain = backend.outer_core_domain
    pencil_spec = cfg.pencils.spec
    return SolverGradientWorkspace{T}(
        solver_create_gradient_field(T, cfg, domain, pencil_spec),
        solver_create_gradient_field(T, cfg, domain, pencil_spec),
        solver_create_gradient_field(T, cfg, domain, pencil_spec),
        zeros(T, cfg.nlm),
        zeros(T, cfg.nlm),
    )
end

function create_transform_workspace(::Type{T}, backend::SolverBackend{A}) where {T, A}
    return TransformWorkspace{T}(backend.architecture)
end

function create_solver_timestep_state(backend::SolverBackend{<:AbstractArchitecture})
    return SolverTimestepState(
        backend.parameters.start_time,
        backend.parameters.timestep,
        0,
        0,
        Inf,
        false,
        true,
    )
end

function solver_load_field_bc_file!(field, filename, format, config, ::Type{T}, label, rank) where T
    try
        coefficients = SOLVER_LOAD_SPECTRAL_BC(
            filename,
            config;
            format=format,
            T=T,
        )
        SOLVER_STORE_BC_IN_FIELD!(field, coefficients)
        if rank == 0
            @info "Loaded $(label) BCs from $(filename) (format=$(format))"
        end
    catch e
        @warn "Failed to load $(label) BC file '$(filename)': $e"
    end
    return field
end

function load_solver_file_bcs!(runtime::SolverRuntime{T,<:AbstractArchitecture}, params::SolverParameters, rank::Int) where T
    if !isempty(params.temperature_bc_file)
        solver_load_field_bc_file!(
            runtime.temperature,
            params.temperature_bc_file,
            params.temperature_bc_format,
            runtime.shtns_config,
            T,
            "temperature",
            rank,
        )
    end

    if !isempty(params.composition_bc_file) && runtime.composition !== nothing
        solver_load_field_bc_file!(
            runtime.composition,
            params.composition_bc_file,
            params.composition_bc_format,
            runtime.shtns_config,
            T,
            "composition",
            rank,
        )
    end

    return runtime
end

@inline solver_scalar_boundary_type(::FixedTemperature) = Int(DIRICHLET)
@inline solver_scalar_boundary_type(::FixedFlux) = Int(NEUMANN)
@inline solver_scalar_boundary_value(bc) = bc.value

function solver_apply_scalar_boundary_parameters!(field, boundary_conditions::BoundaryConditions)
    T = eltype(field.boundary_values)
    fill!(field.bc_type_inner, solver_scalar_boundary_type(boundary_conditions.inner))
    fill!(field.bc_type_outer, solver_scalar_boundary_type(boundary_conditions.outer))
    fill!(field.boundary_values, zero(T))

    mean_mode = get_mode_index(field.config, 0, 0)
    if mean_mode > 0
        field.boundary_values[1, mean_mode] = T(solver_scalar_boundary_value(boundary_conditions.inner))
        field.boundary_values[2, mean_mode] = T(solver_scalar_boundary_value(boundary_conditions.outer))
    end

    return field
end

function create_solver_runtime(::Type{T}, backend::SolverBackend{A};
                               auto_optimize::Bool=false,
                               adaptive_threading::Bool=false) where {T, A}
    solver_backend_ensure_mpi!()

    velocity, magnetic, temperature, composition = create_solver_fields(T, backend)
    solver_apply_scalar_boundary_parameters!(temperature, backend.parameters.temperature_bcs)
    if composition !== nothing
        solver_apply_scalar_boundary_parameters!(composition, backend.parameters.composition_bcs)
    end
    gradient_workspace = create_solver_gradient_workspace(T, backend)
    transform_workspace = create_transform_workspace(T, backend)
    timestep_state = create_solver_timestep_state(backend)
    backend.shtns_config._buffers.solver_transform_workspace = transform_workspace
    # Store arch in SHTnsBuffers so SHT dispatch functions can retrieve it (Task 3)
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
        # Ball geometry has no distinct inner-core domain, but downstream
        # magnetic kernels expect both slots to be populated consistently.
        isnothing(backend.inner_core_domain) ? backend.outer_core_domain : backend.inner_core_domain,
        timestep_state,
    )

    load_solver_file_bcs!(runtime, backend.parameters, backend.rank)
    return runtime
end
