"""
    SolverBackend

Immutable description of the backend resources used by a solver run.

It records the selected architecture, SHTnsKit transform configuration, radial
domains, and MPI rank/process metadata. `create_solver_backend(...)` is the
normal constructor used by the public solver initialization path.
"""
struct SolverBackend{A <: AbstractArchitecture, C <: SHTnsConfigType}
    parameters::SolverParameters
    architecture::A
    shtns_config::C
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
mutable struct SolverTopographyState{T <: AbstractFloat}
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
# `S` pins the concrete spectral-field type. Declaring the gradient fields as the
# bare `SpectralFieldType{T}` (= `SHTnsSpecField{T}`, whose C/DR/DI params are
# still free) leaves them non-concrete, so every `ws.∇φ_spec` access in the
# gradient hot loop boxes and the per-element writes dispatch dynamically
# (~8 KB/call). Parametrising on `S` makes the fields concrete and allocation-free.
struct SolverGradientWorkspace{T, S <: SpectralFieldType{T}}
    ∇θ_spec::S
    ∇φ_spec::S
    ∇r_spec::S
    # Cross-rank gather scratch, shaped (nlm, nr_local). The θ-recurrence couples
    # (l,m) with (l±1,m), which may live on other ranks, so the full spectrum is
    # summed in. Holding every radial level at once lets the gather use a single
    # collective per component instead of one per radial level.
    theta_full_real::Matrix{T}
    theta_full_imag::Matrix{T}
    # Precomputed (l±1, m) -> global storage index for the θ-gradient recurrence.
    # Built once so the per-mode hot loop avoids hashing the full mode arrays.
    theta_lm_plus::Vector{Int}
    theta_lm_minus::Vector{Int}
end

"""
    SolverTransformBuffers{T}

Typed scratch storage replacing `TransformWorkspace{T}.cache::Dict{Symbol,Any}`.
Each field corresponds to a key formerly stored in the Dict.
"""
mutable struct SolverTransformBuffers{T}
    vector_coeffs_1::Union{Matrix{ComplexF64}, Nothing}
    vector_coeffs_2::Union{Matrix{ComplexF64}, Nothing}
    vector_coeffs_gathered_1::Union{Matrix{ComplexF64}, Nothing}
    vector_coeffs_gathered_2::Union{Matrix{ComplexF64}, Nothing}
    pol_rad_coeffs::Union{Matrix{ComplexF64}, Nothing}
    vector_component_vt::Union{Matrix{Float64}, Nothing}
    vector_component_vp::Union{Matrix{Float64}, Nothing}
    generic_slice::Union{Matrix{Float64}, Nothing}
    generic_slice_gathered::Union{Matrix{Float64}, Nothing}
    coeffs_buffer::Union{Matrix{ComplexF64}, Nothing}
    coeffs_gathered::Union{Matrix{ComplexF64}, Nothing}
    # Batched gather scratch: all radial levels stacked (…, nr_local) so the
    # cross-rank scalar-transform gather uses one collective instead of one per
    # level. coeffs_buffer_batched is (lmax+1, mmax+1, nr) for synthesis;
    # slice_buffer_batched is (nlat, nlon, nr) for analysis.
    coeffs_buffer_batched::Union{Array{ComplexF64, 3}, Nothing}
    slice_buffer_batched::Union{Array{Float64, 3}, Nothing}
end

function SolverTransformBuffers{T}() where {T}
    SolverTransformBuffers{T}(
        nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing
    )
end

"""
    TransformWorkspace{T}

Scratch storage owned by one solver runtime for transform-side gather/scatter
work.

GPU-marked runs route these allocations through backend hooks so the same
solver code can use either CPU or backend-provided scratch storage.
"""
struct TransformWorkspace{T, A <: AbstractArchitecture} <: AbstractTransformWorkspace
    arch::A
    buffers::SolverTransformBuffers{T}
end

function TransformWorkspace{T}(arch::A) where {T, A <: AbstractArchitecture}
    TransformWorkspace{T, A}(arch, SolverTransformBuffers{T}())
end

"""
    SolverRuntime{T}

The concrete field and workspace objects stepped by the rewritten solver.

`SolverState` wraps this together with parameters, topography state, timestep
matrices, caches, and diagnostics.
"""
struct SolverRuntime{
    T,
    A <: AbstractArchitecture,
    C <: SHTnsConfigType,
    V <: VelocityFieldsType{T},
    M <: MagneticFieldsType{T},
    Temp <: TemperatureFieldType{T},
    Comp <: Union{CompositionFieldType{T}, Nothing},
    GW <: SolverGradientWorkspace{T},
    TW <: TransformWorkspace{T, A}
}
    velocity::V
    magnetic::M
    temperature::Temp
    composition::Comp
    gradient_workspace::GW
    transform_workspace::TW
    shtns_config::C
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
    _solver_print_row(io, "lmax / mmax", "$(cfg.lmax) / $(cfg.mmax)")
    _solver_print_row(io, "Nθ × Nφ", "$(cfg.nlat) × $(cfg.nlon)")
    _solver_print_row(io, "spectral modes", cfg.nlm)
    println(io, "├─ domains")
    _solver_print_row(io, "outer core Nᵣ", backend.outer_core_domain.N)
    _solver_print_row(io, "inner core Nᵣ",
        isnothing(backend.inner_core_domain) ? "none" : backend.inner_core_domain.N)
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
    _solver_print_row(io, "ICB topography",
        topography.data === nothing || topography.data.icb === nothing ? "none" : "loaded")
    _solver_print_row(io, "OCB topography",
        topography.data === nothing || topography.data.cmb === nothing ? "none" : "loaded")
    _solver_print_row(io, "Stefan state", isnothing(topography.stefan) ? "inactive" :
                                          "ready")
end

function create_shtns_config(::CPU, params::SolverParameters)
    return SOLVER_SHTNS_CONFIG_BUILDER(
        lmax = params.lmax,
        mmax = params.mmax,
        nlat = params.nlat,
        nlon = params.nlon,
        nr = params.nr,
        optimize_decomp = true,
        device = :cpu
    )
end

function create_shtns_config(::GPU, params::SolverParameters)
    # Non-CUDA GPU: SHTnsKit has no native support; use CPU SHTns config.
    # Physical data arrays live on the GPU; transforms run on CPU.
    return SOLVER_SHTNS_CONFIG_BUILDER(
        lmax = params.lmax,
        mmax = params.mmax,
        nlat = params.nlat,
        nlon = params.nlon,
        nr = params.nr,
        optimize_decomp = true,
        device = :cpu
    )
end

function architecture_from_symbol(architecture::Symbol)
    architecture === :cpu && return CPU()
    architecture === :gpu && return GPU(nothing)
    throw(ArgumentError("architecture = $(architecture) must be :cpu or :gpu"))
end

function scale_radial_domain(domain::RadialDomainType, radius_scale::Real)
    scale = Float64(radius_scale)
    scale > 0 ||
        throw(ArgumentError("inner-core radius scale must be positive, got $scale"))

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

    dr_matrices = [copy(matrix) ./ (scale ^ order)
                   for (order, matrix) in enumerate(domain.dr_matrices)]
    radial_laplacian = copy(domain.radial_laplacian) ./ (scale ^ 2)
    integration_weights = copy(domain.integration_weights) .* scale

    return RadialDomainType(
        domain.N,
        domain.local_range,
        r,
        dr_matrices,
        radial_laplacian,
        integration_weights
    )
end

function create_inner_core_domain(params::SolverParameters)
    unit_ball_domain = SOLVER_BALL_DOMAIN_BUILDER(
        params.nr_inner;
        radial_bandwidth = params.radial_bandwidth
    )
    inner_core_radius = params.radius_ratio / (1.0 - params.radius_ratio)
    return scale_radial_domain(unit_ball_domain, inner_core_radius)
end

function create_radial_domains(params::SolverParameters)
    outer_core_domain = params.geometry === :ball ?
                        SOLVER_BALL_DOMAIN_BUILDER(params.nr; radial_bandwidth = params.radial_bandwidth) :
                        SOLVER_SHELL_DOMAIN_BUILDER(
        params.nr;
        radius_ratio = params.radius_ratio,
        radial_bandwidth = params.radial_bandwidth
    )

    inner_core_domain = params.geometry === :shell ?
                        create_inner_core_domain(params) :
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
    cfg = create_shtns_config(arch, params)
    outer_core_domain, inner_core_domain = create_radial_domains(params)
    return SolverBackend(
        params,
        arch,
        cfg,
        outer_core_domain,
        inner_core_domain,
        solver_backend_rank(),
        solver_backend_process_count()
    )
end

function create_solver_backend(params::SolverParameters)
    arch = architecture_from_symbol(params.architecture)
    return create_solver_backend(arch, params)
end

@inline solver_create_velocity_fields(::Type{T},
    cfg,
    outer,
    pencils,
    params) where {T} = SOLVER_VELOCITY_FIELD_BUILDER(
    T, cfg, outer, pencils, pencils.spec; params)

@inline solver_create_magnetic_fields(::Type{T},
    cfg,
    outer,
    inner,
    pencils) where {T} = SOLVER_MAGNETIC_FIELD_BUILDER(
    T, cfg, outer, inner, pencils, pencils.spec)

@inline solver_create_temperature_field(::Type{T},
    cfg,
    outer,
    pencils) where {T} = SOLVER_TEMPERATURE_FIELD_BUILDER(
    T, cfg, outer, pencils, pencils.spec)

@inline solver_create_composition_field(::Type{T},
    cfg,
    outer,
    pencils) where {T} = SOLVER_COMPOSITION_FIELD_BUILDER(
    T, cfg, outer, pencils, pencils.spec)

"""
    create_solver_fields(T, backend)

Allocate the spectral/physical field containers required by the solver runtime
for element type `T`.
"""
function create_solver_fields(::Type{T}, backend::SolverBackend{<:AbstractArchitecture}) where {T}
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
    composition = backend.parameters.include_composition ?
                  solver_create_composition_field(T, cfg, outer, pencils) :
                  nothing

    return velocity, magnetic, temperature, composition
end

function build_velocity_implicit_matrices(cfg, domain, E, dt, velocity_bc_code)
    return (
        tor = SOLVER_VELOCITY_TOROIDAL_MATRIX_BUILDER(
            cfg, domain, E, dt; velocity_bc_code = velocity_bc_code, mass_coeff = E
        ),
        pol = SOLVER_VELOCITY_POLOIDAL_MATRIX_BUILDER(
            cfg, domain, E, dt; velocity_bc_code = velocity_bc_code, mass_coeff = E
        )
    )
end

function build_magnetic_implicit_matrices(cfg, domain, dt)
    return (
        tor = SOLVER_MAGNETIC_TOROIDAL_MATRIX_BUILDER(cfg, domain, 1.0, dt),
        pol = SOLVER_MAGNETIC_POLOIDAL_MATRIX_BUILDER(cfg, domain, 1.0, dt)
    )
end

"""
    build_magnetic_implicit_matrices_conducting(T, cfg, domain, ic_domain, dt; theta)

Build the magnetic toroidal/poloidal implicit matrices for a conducting inner
core together with the inner-core ICB admittances.

For each component the inner-core diffusion operator and its ICB admittance `α_l`
are precomputed (`create_inner_core_admittance`), and the outer-core matrices are
built with the conducting Robin inner row `(∂/∂r − α_l) S = φ0` via the
`inner_alpha` kwarg. The magnetic diffusivity (`1.0`), `dt`, and `theta` MUST
match the values used to shift the outer-core system matrices so the CNAB2
history RHS is consistent across the ICB.

Returns `(tor, pol, admittance)` where `admittance` is a
`NamedTuple{(:tor,:pol)}` of `InnerCoreAdmittance{T}` objects.
"""
function build_magnetic_implicit_matrices_conducting(::Type{T}, cfg, domain, ic_domain, dt;
        theta::Float64 = 0.5) where {T}
    η = 1.0  # magnetic diffusivity (matches build_magnetic_implicit_matrices)
    uniq_l = filter(>(0), sort(unique(cfg.l_values)))

    adm_tor = create_inner_core_admittance(T, uniq_l, ic_domain, η, dt; theta = theta)
    adm_pol = create_inner_core_admittance(T, uniq_l, ic_domain, η, dt; theta = theta)

    alpha_tor = Dict{Int, T}(l => inner_core_alpha(adm_tor, l) for l in uniq_l)
    alpha_pol = Dict{Int, T}(l => inner_core_alpha(adm_pol, l) for l in uniq_l)

    tor = SOLVER_MAGNETIC_TOROIDAL_MATRIX_BUILDER(cfg, domain, η, dt;
        theta = theta, T = T, inner_alpha = alpha_tor)
    pol = SOLVER_MAGNETIC_POLOIDAL_MATRIX_BUILDER(cfg, domain, η, dt;
        theta = theta, T = T, inner_alpha = alpha_pol)

    return (tor = tor, pol = pol, admittance = (tor = adm_tor, pol = adm_pol))
end

@inline solver_build_temperature_implicit_matrix(cfg,
    domain,
    diffusivity,
    dt,
    temperature_bc_code) = SOLVER_TEMPERATURE_MATRIX_BUILDER(
    cfg, domain, diffusivity, dt; temperature_bc_code = temperature_bc_code)

@inline solver_build_composition_implicit_matrix(cfg,
    domain,
    diffusivity,
    dt,
    composition_bc_code) = SOLVER_COMPOSITION_MATRIX_BUILDER(
    cfg, domain, diffusivity, dt; composition_bc_code = composition_bc_code)

# Shared core for both the eager (construction-time) and rebuild (dt-change)
# implicit-matrix paths. `dt` is the authoritative timestep — callers pass it
# explicitly so the rebuild path can override the (frozen) backend timestep.
function _build_implicit_matrices_dict(
        ::Type{T}, cfg, outer, ic_domain, p::SolverParameters, dt::Float64
) where {T}
    matrices = Dict{Symbol, OldImplicitMatrices{T}}()
    velocity = build_velocity_implicit_matrices(
        cfg, outer, p.Ek, dt, _velocity_bc_code(p.velocity_bcs))
    matrices[:velocity_tor] = velocity.tor
    matrices[:velocity_pol] = velocity.pol

    magnetic_ic_admittance = nothing
    if p.magnetic_inner_bc === :conducting_inner_core
        # Conducting inner core: build the ICB admittances and outer-core
        # matrices with the conducting Robin inner row. Requires the inner-core
        # ball domain (present for shell geometry, enforced by parameter checks).
        ic_domain === nothing && error(
            "magnetic_inner_bc=:conducting_inner_core requires an inner-core domain " *
            "(geometry=:shell); got inner_core_domain === nothing")
        magnetic = build_magnetic_implicit_matrices_conducting(T, cfg, outer, ic_domain, dt)
        magnetic_ic_admittance = magnetic.admittance
    else
        magnetic = build_magnetic_implicit_matrices(cfg, outer, dt)
    end
    matrices[:magnetic_tor] = magnetic.tor
    matrices[:magnetic_pol] = magnetic.pol

    matrices[:temperature] = solver_build_temperature_implicit_matrix(
        cfg, outer, p.Pm / p.Pr, dt, _thermal_bc_code(p.temperature_bcs))
    if p.include_composition
        matrices[:composition] = solver_build_composition_implicit_matrix(
            cfg, outer, p.Pm / p.Sc, dt, _composition_bc_code(p.composition_bcs))
    end
    return matrices, magnetic_ic_admittance
end

"""
    create_solver_implicit_matrices(T, backend)

Precompute the linear solve operators used by the solver timestep schemes for
element type `T`.

The returned matrices are later wrapped into the solver-owned matrix store so
the timestep loop can reuse them without rebuilding operators on each step.

Returns `(matrices, magnetic_ic_admittance)`. `magnetic_ic_admittance` is a
`NamedTuple{(:tor,:pol)}` of `InnerCoreAdmittance` objects when the conducting
inner-core path is enabled (`params.magnetic_inner_bc === :conducting_inner_core`)
and `nothing` otherwise. The insulating default path is byte-for-byte unchanged.
"""
function create_solver_implicit_matrices(::Type{T}, backend::SolverBackend{<:AbstractArchitecture}) where {T}
    p = backend.parameters
    return _build_implicit_matrices_dict(
        T, backend.shtns_config, backend.outer_core_domain,
        backend.inner_core_domain, p, Float64(p.timestep))
end

@inline solver_create_gradient_field(::Type{T}, cfg, domain,
    pencil_spec) where {T} = SOLVER_SPECTRAL_FIELD_BUILDER(T, cfg, domain, pencil_spec)

# Precompute, for every spectral mode (l, m), the storage index of its
# (l+1, m) and (l-1, m) neighbors used by the θ-gradient recurrence. Neighbors
# outside the truncation map to 0. Building this table once removes the
# per-call mode-array hashing that `mode_index` would otherwise incur in the
# gradient hot loop (called per mode, per radial level, every timestep).
function build_theta_gradient_neighbors(cfg::SHTnsConfigType)
    nlm = cfg.nlm
    lvals = cfg.l_values
    mvals = cfg.m_values
    index_of = Dict{Tuple{Int, Int}, Int}()
    sizehint!(index_of, nlm)
    @inbounds for i in 1:nlm
        index_of[(lvals[i], mvals[i])] = i
    end
    lm_plus = zeros(Int, nlm)
    lm_minus = zeros(Int, nlm)
    @inbounds for i in 1:nlm
        l = lvals[i]
        m = mvals[i]
        lm_plus[i] = get(index_of, (l + 1, m), 0)
        lm_minus[i] = get(index_of, (l - 1, m), 0)
    end
    return lm_plus, lm_minus
end

function create_solver_gradient_workspace(::Type{T}, backend::SolverBackend{<:AbstractArchitecture}) where {T}
    cfg = backend.shtns_config
    domain = backend.outer_core_domain
    pencil_spec = cfg.pencils.spec
    theta_lm_plus, theta_lm_minus = build_theta_gradient_neighbors(cfg)
    nr_local = length(local_range(pencil_spec, 3))
    # All-inferred constructor so both T and the concrete spectral type S are
    # taken from the arguments (`SolverGradientWorkspace{T}(...)` would be a
    # partial parametric application with no matching constructor).
    return SolverGradientWorkspace(
        solver_create_gradient_field(T, cfg, domain, pencil_spec),
        solver_create_gradient_field(T, cfg, domain, pencil_spec),
        solver_create_gradient_field(T, cfg, domain, pencil_spec),
        zeros(T, cfg.nlm, nr_local),
        zeros(T, cfg.nlm, nr_local),
        theta_lm_plus,
        theta_lm_minus
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
        true
    )
end

function load_field_bc_file!(
        field, filename, format, config, ::Type{T}, label, rank) where {T}
    try
        coefficients = SOLVER_LOAD_SPECTRAL_BC(
            filename,
            config;
            format = format,
            T = T
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

function load_solver_file_bcs!(runtime::SolverRuntime{T, <:AbstractArchitecture},
        params::SolverParameters, rank::Int) where {T}
    if !isempty(params.temperature_bc_file)
        load_field_bc_file!(
            runtime.temperature,
            params.temperature_bc_file,
            params.temperature_bc_format,
            runtime.shtns_config,
            T,
            "temperature",
            rank
        )
    end

    if !isempty(params.composition_bc_file) && runtime.composition !== nothing
        load_field_bc_file!(
            runtime.composition,
            params.composition_bc_file,
            params.composition_bc_format,
            runtime.shtns_config,
            T,
            "composition",
            rank
        )
    end

    return runtime
end

@inline solver_scalar_boundary_type(::FixedTemperature) = Int(DIRICHLET)
@inline solver_scalar_boundary_type(::FixedFlux) = Int(NEUMANN)
@inline solver_scalar_boundary_value(bc) = bc.value

"""
    apply_scalar_boundary_parameters!(field, boundary_conditions)

Install scalar boundary values from `SolverParameters` into the field-owned
boundary storage used by CNAB2, EAB2, and ERK2.

Parameter-specified scalar BCs are spatially uniform, so only the `(l,m)=(0,0)`
mode receives a nonzero endpoint value. File-based spectral BCs can later
replace this with per-mode real and imaginary values.

The transform is orthonormal (`Y_0^0 = 1/√(4π)`), so a uniform physical value
`v` maps to the `(0,0)` spectral coefficient `v·√(4π)`; the boundary endpoints
are scaled accordingly (same convention as `bcs/topography/topography_data.jl`).
"""
function apply_scalar_boundary_parameters!(field, boundary_conditions::BoundaryConditions)
    T = eltype(field.boundary_values)
    fill!(field.bc_type_inner, solver_scalar_boundary_type(boundary_conditions.inner))
    fill!(field.bc_type_outer, solver_scalar_boundary_type(boundary_conditions.outer))
    fill!(field.boundary_values, zero(T))

    mean_mode = get_mode_index(field.config, 0, 0)
    if mean_mode > 0
        sqrt_4pi = sqrt(4 * convert(T, π))
        field.boundary_values[1, mean_mode] = sqrt_4pi *
                                              T(solver_scalar_boundary_value(boundary_conditions.inner))
        field.boundary_values[2, mean_mode] = sqrt_4pi *
                                              T(solver_scalar_boundary_value(boundary_conditions.outer))
    end

    return field
end

"""
    create_solver_runtime(T, backend; auto_optimize=false, adaptive_threading=false)

Allocate the solver-owned field objects, shared workspaces, timestep state, and
runtime domain references for a new run.

Scalar parameter BCs are installed before optional file BCs are loaded so a
user-supplied spectral boundary file cleanly overrides the homogeneous or
mean-mode defaults.
"""
function create_solver_runtime(::Type{T}, backend::SolverBackend{A};
        auto_optimize::Bool = false,
        adaptive_threading::Bool = false) where {T, A}
    backend_ensure_mpi!()

    velocity, magnetic, temperature, composition = create_solver_fields(T, backend)
    apply_scalar_boundary_parameters!(temperature, backend.parameters.temperature_bcs)
    if composition !== nothing
        apply_scalar_boundary_parameters!(composition, backend.parameters.composition_bcs)
    end
    gradient_workspace = create_solver_gradient_workspace(T, backend)
    transform_workspace = create_transform_workspace(T, backend)
    timestep_state = create_solver_timestep_state(backend)
    backend.shtns_config._buffers.solver_transform_workspace = transform_workspace
    # Store arch in SHTnsBuffers so SHT dispatch functions can retrieve it (Task 3)
    backend.shtns_config._buffers.transform_device = backend.architecture

    runtime = SolverRuntime(
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
        isnothing(backend.inner_core_domain) ? backend.outer_core_domain :
        backend.inner_core_domain,
        timestep_state
    )

    load_solver_file_bcs!(runtime, backend.parameters, backend.rank)
    return runtime
end
