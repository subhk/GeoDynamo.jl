"""
    SolverFields{T}

Convenient grouped view of the fields that actively participate in a solver
run.

This separates "allocated in the runtime" from "enabled by the current solver
parameters", which keeps the timestep code simpler when optional magnetic or
compositional physics are disabled.
"""
struct SolverFields{T}
    velocity::VelocityFieldsType{T}
    temperature::TemperatureFieldType{T}
    magnetic::Union{MagneticFieldsType{T}, Nothing}
    composition::Union{CompositionFieldType{T}, Nothing}
end

mutable struct SolverEnergyTracker
    kinetic_energy::Vector{Float64}
    magnetic_energy::Vector{Float64}
    thermal_energy::Vector{Float64}
    compositional_energy::Vector{Float64}
    total_energy::Vector{Float64}
    timestamps::Vector{Int}
    enable_tracking::Bool
end

mutable struct SolverSolenoidalMonitor
    velocity_div_l2::Vector{Float64}
    velocity_div_linf::Vector{Float64}
    magnetic_div_l2::Vector{Float64}
    magnetic_div_linf::Vector{Float64}
    timestamps::Vector{Int}
    enable_monitoring::Bool
end

struct BandedOperator{T}
    data::Matrix{T}
    bandwidth::Int
    size::Int
end

struct BandedFactorization{T}
    lu::Matrix{T}
    bandwidth::Int
    size::Int
end

struct EAB2CacheEntry{T}
    ν::Float64
    nr::Int
    map::Dict{Int, Tuple{BandedOperator{T}, BandedFactorization{T}}}
end

struct ERK2InfluenceOp{T}
    Gre::Matrix{T}
    invG::Matrix{T}
    l::Int
end

struct ERK2InfluenceCacheEntry{T}
    matrices::Dict{Int, ERK2InfluenceOp{T}}
    diffusivity::Float64
    dt::Float64
    theta::Float64
    velocity_bc_code::Int
    lmax::Int
    mmax::Int
    nlat::Int
    nlon::Int
    nr::Int
    domain_hash::UInt
end

struct ERK2StageCache{T}
    dt::Float64
    diffusivity::Float64
    nr::Int
    l_values::Vector{Int}
    E_half::Vector{Matrix{T}}
    E_full::Vector{Matrix{T}}
    phi1_half::Vector{Matrix{T}}
    phi1_full::Vector{Matrix{T}}
    phi2_full::Vector{Matrix{T}}
    use_krylov::Bool
    krylov_m::Int
    krylov_tol::Float64
    mpi_consistent::Bool
end

struct SolverRadialWork{T}
    u_real_global::Vector{T}
    u_imag_global::Vector{T}
    linear_real::Vector{T}
    linear_imag::Vector{T}
    tmp_real::Vector{T}
    tmp_imag::Vector{T}
end

# Reusable radial profiles for one field/operator family. The vectors are
# global in radius because each implicit or exponential radial solve is posed
# on the full radial stencil even when a rank owns only a local pencil slab.
SolverRadialWork{T}(nr::Int) where T = SolverRadialWork{T}(
    zeros(T, nr),
    zeros(T, nr),
    zeros(T, nr),
    zeros(T, nr),
    Vector{T}(undef, nr),
    Vector{T}(undef, nr),
)

mutable struct TimestepCaches{T}
    # EAB2 exponential integrator caches.
    etd_velocity_toroidal   :: Union{EAB2CacheEntry{T}, Nothing}
    etd_velocity_poloidal   :: Union{EAB2CacheEntry{T}, Nothing}
    etd_magnetic_toroidal   :: Union{EAB2CacheEntry{T}, Nothing}
    etd_magnetic_poloidal   :: Union{EAB2CacheEntry{T}, Nothing}
    etd_temperature         :: Union{EAB2CacheEntry{T}, Nothing}
    etd_composition         :: Union{EAB2CacheEntry{T}, Nothing}
    # ERK2 stage caches used by explicit and CNAB2 updates.
    erk2_velocity_toroidal  :: Union{ERK2StageCache{T}, Nothing}
    erk2_velocity_poloidal  :: Union{ERK2StageCache{T}, Nothing}
    erk2_magnetic_toroidal  :: Union{ERK2StageCache{T}, Nothing}
    erk2_magnetic_poloidal  :: Union{ERK2StageCache{T}, Nothing}
    erk2_temperature        :: Union{ERK2StageCache{T}, Nothing}
    erk2_composition        :: Union{ERK2StageCache{T}, Nothing}
    # ERK2 velocity-poloidal influence matrices
    erk2_influence_velocity_poloidal :: Union{ERK2InfluenceCacheEntry{T}, Nothing}
    # Field-keyed scratch profiles shared by CNAB2/EAB2 solves. Keeping them in
    # the timestep cache avoids reallocating full radial work vectors every step.
    radial_work::Dict{Symbol, SolverRadialWork{T}}
end

TimestepCaches{T}() where T = TimestepCaches{T}(
    nothing, nothing, nothing, nothing, nothing, nothing,
    nothing, nothing, nothing, nothing, nothing, nothing,
    nothing,
    Dict{Symbol, SolverRadialWork{T}}(),
)

struct ImplicitMatrixSet{T}
    system_matrices::Vector{BandedOperator{T}}
    factorizations::Vector{BandedFactorization{T}}
    linear_matrices::Vector{BandedOperator{T}}
    l_values::Vector{Int}
    lookup::Dict{Int, Int}
    theta::Float64
end

const SolverBandedMatrix = BandedOperator
const SolverBandedLU = BandedFactorization
const SolverEAB2ALUCacheEntry = EAB2CacheEntry
const SolverERK2InfluenceMatrix = ERK2InfluenceOp
const SolverERK2Cache = ERK2StageCache
const SolverImplicitMatrices = ImplicitMatrixSet

"""
    SolverState{T}

Top-level state object for the rewritten GeoDynamo solver.

It combines:

- `SolverParameters`
- backend and runtime objects
- active field views
- topography / Stefan state
- implicit matrices and timestep caches
- energy and solenoidal diagnostics

`initialize_simulation(Float64, params)` returns this type for the new solver
path.
"""
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

function Base.show(io::IO, ::MIME"text/plain", state::SolverState)
    println(io, "GeoDynamo SolverState")
    println(io, "├─ model")
    _solver_print_row(io, "architecture", state.backend.architecture)
    _solver_print_row(io, "geometry", state.parameters.geometry)
    _solver_print_row(io, "time", state.time)
    _solver_print_row(io, "step", state.step)
    _solver_print_row(io, "initialized", _solver_yesno(state.is_initialized))
    println(io, "├─ active fields")
    _solver_print_row(io, "velocity", "yes")
    _solver_print_row(io, "temperature", "yes")
    _solver_print_row(io, "magnetic", isnothing(state.fields.magnetic) ? "no" : "yes")
    _solver_print_row(io, "composition", isnothing(state.fields.composition) ? "no" : "yes")
    println(io, "└─ boundaries")
    _solver_print_row(io, "topography", _solver_yesno(state.topography.config.enabled))
    _solver_print_row(io, "Stefan ICB", isnothing(state.topography.stefan) ? "no" : "yes")
end

BandedOperator(A::OldBandedMatrix{T}) where {T} =
    BandedOperator{T}(copy(A.data), A.bandwidth, A.size)

BandedFactorization(A::OldBandedLU{T}) where {T} =
    BandedFactorization{T}(copy(A.lu), A.bandwidth, A.size)

function ImplicitMatrixSet(matrices::OldImplicitMatrices{T}) where {T}
    return ImplicitMatrixSet{T}(
        BandedOperator.(matrices.system_matrices),
        BandedFactorization.(matrices.factorizations),
        BandedOperator.(matrices.linear_matrices),
        copy(matrices.l_values),
        Dict{Int, Int}(matrices.lookup),
        matrices.theta,
    )
end

function create_solver_implicit_matrix_store(
    matrices_by_name::Dict{Symbol, OldImplicitMatrices{T}},
) where {T}
    store = Dict{Symbol, ImplicitMatrixSet{T}}()
    for (name, matrices) in matrices_by_name
        store[name] = ImplicitMatrixSet(matrices)
    end
    return store
end

function _collect_solver_fields(runtime::SolverRuntime{T,<:AbstractArchitecture}, params::SolverParameters) where T
    magnetic = params.include_magnetic_field ? runtime.magnetic : nothing
    composition = params.include_composition ? runtime.composition : nothing

    return SolverFields{T}(runtime.velocity, runtime.temperature, magnetic, composition)
end

function _synchronize_solver_views!(state::SolverState{T,<:AbstractArchitecture}) where T
    state.fields = _collect_solver_fields(state.runtime, state.parameters)
    state.time = state.runtime.timestep_state.time
    state.step = state.runtime.timestep_state.step
    return state
end

function solver_sync_output_physical_scalars!(state::SolverState{T,<:AbstractArchitecture}) where T
    solver_scalar_spectral_to_physical!(
        state.fields.temperature.spectral,
        state.fields.temperature.temperature,
    )

    if state.fields.composition !== nothing
        solver_scalar_spectral_to_physical!(
            state.fields.composition.spectral,
            state.fields.composition.composition,
        )
    end

    return state
end

"""
    extract_all_fields(state::SolverState)

Return a copy-based dictionary snapshot of the main solver fields.

This is primarily used by restart/output tooling and tests that need a stable
container representation independent of the in-memory field types.
"""
function GeoDynamo.extract_all_fields(state::SolverState{T,<:AbstractArchitecture}) where {T}
    solver_sync_output_physical_scalars!(state)

    fields = Dict{String, Any}()

    fields["velocity_toroidal"] = Dict(
        "real" => copy(parent(state.fields.velocity.𝒯.data_real)),
        "imag" => copy(parent(state.fields.velocity.𝒯.data_imag)),
    )

    fields["velocity_poloidal"] = Dict(
        "real" => copy(parent(state.fields.velocity.𝒫.data_real)),
        "imag" => copy(parent(state.fields.velocity.𝒫.data_imag)),
    )

    magnetic = state.fields.magnetic
    if magnetic !== nothing
        fields["magnetic_toroidal"] = Dict(
            "real" => copy(parent(magnetic.𝒯.data_real)),
            "imag" => copy(parent(magnetic.𝒯.data_imag)),
        )
        fields["magnetic_poloidal"] = Dict(
            "real" => copy(parent(magnetic.𝒫.data_real)),
            "imag" => copy(parent(magnetic.𝒫.data_imag)),
        )
    else
        fields["magnetic_toroidal"] = Dict(
            "real" => copy(parent(state.runtime.magnetic.𝒯.data_real)),
            "imag" => copy(parent(state.runtime.magnetic.𝒯.data_imag)),
        )
        fields["magnetic_poloidal"] = Dict(
            "real" => copy(parent(state.runtime.magnetic.𝒫.data_real)),
            "imag" => copy(parent(state.runtime.magnetic.𝒫.data_imag)),
        )
    end

    fields["temperature"] = copy(parent(state.fields.temperature.temperature.data))
    fields["temperature_spectral"] = Dict(
        "real" => copy(parent(state.fields.temperature.spectral.data_real)),
        "imag" => copy(parent(state.fields.temperature.spectral.data_imag)),
    )

    if state.fields.composition !== nothing
        fields["composition"] = copy(parent(state.fields.composition.composition.data))
        fields["composition_spectral"] = Dict(
            "real" => copy(parent(state.fields.composition.spectral.data_real)),
            "imag" => copy(parent(state.fields.composition.spectral.data_imag)),
        )
    end

    return fields
end

function _copy_restart_array!(destination::AbstractArray, source, name::AbstractString)
    size(destination) == size(source) || throw(DimensionMismatch(
        "Restart field $name has size $(size(source)); expected $(size(destination)).",
    ))
    copyto!(destination, source)
    return destination
end

function _restore_restart_spectral_pair!(field, data, name::AbstractString)
    haskey(data, "real") && haskey(data, "imag") || throw(ArgumentError(
        "Restart spectral field $name must contain real and imag arrays.",
    ))
    _copy_restart_array!(parent(field.data_real), data["real"], "$(name)_real")
    _copy_restart_array!(parent(field.data_imag), data["imag"], "$(name)_imag")
    return field
end

function restore_fields_from_restart!(
    state::SolverState{T,<:AbstractArchitecture},
    restart_data::Dict{String,Any},
) where T
    if haskey(restart_data, "velocity_toroidal")
        _restore_restart_spectral_pair!(
            state.fields.velocity.𝒯,
            restart_data["velocity_toroidal"],
            "velocity_toroidal",
        )
    end
    if haskey(restart_data, "velocity_poloidal")
        _restore_restart_spectral_pair!(
            state.fields.velocity.𝒫,
            restart_data["velocity_poloidal"],
            "velocity_poloidal",
        )
    end

    magnetic = state.fields.magnetic === nothing ? state.runtime.magnetic : state.fields.magnetic
    if haskey(restart_data, "magnetic_toroidal")
        _restore_restart_spectral_pair!(
            magnetic.𝒯,
            restart_data["magnetic_toroidal"],
            "magnetic_toroidal",
        )
    end
    if haskey(restart_data, "magnetic_poloidal")
        _restore_restart_spectral_pair!(
            magnetic.𝒫,
            restart_data["magnetic_poloidal"],
            "magnetic_poloidal",
        )
    end

    if haskey(restart_data, "temperature")
        _copy_restart_array!(
            parent(state.fields.temperature.temperature.data),
            restart_data["temperature"],
            "temperature",
        )
    end
    if haskey(restart_data, "temperature_spectral")
        _restore_restart_spectral_pair!(
            state.fields.temperature.spectral,
            restart_data["temperature_spectral"],
            "temperature_spectral",
        )
    end

    if state.fields.composition !== nothing
        if haskey(restart_data, "composition")
            _copy_restart_array!(
                parent(state.fields.composition.composition.data),
                restart_data["composition"],
                "composition",
            )
        end
        if haskey(restart_data, "composition_spectral")
            _restore_restart_spectral_pair!(
                state.fields.composition.spectral,
                restart_data["composition_spectral"],
                "composition_spectral",
            )
        end
    end

    _synchronize_solver_views!(state)
    state.is_initialized = true
    return state
end

"""
    create_enhanced_metadata(state::SolverState, time, step)

Create lightweight metadata describing a solver snapshot for outputs and
restart-style tooling.
"""
GeoDynamo.create_enhanced_metadata(state::SolverState, time, step) = Dict(
    "current_time" => time,
    "current_step" => step,
    "geometry" => state.parameters.geometry,
)
