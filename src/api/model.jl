"""
    GeodynamoModel(grid; T=Float64, Ek=1e-4, Pr=1.0, Pm=1.0, Sc=1.0, Ra=1e6,
                   velocity_bcs, temperature_bcs, composition_bcs,
                   boundary_conditions=nothing,
                   include_magnetic=false, include_composition=false,
                   initial_conditions=nothing, kwargs...)

Physical model built on a [`SphericalShellGrid`](@ref) or
[`SphericalBallGrid`](@ref). It bundles the solver `state`, the `grid`, and a
[`Clock`](@ref), and is the object handed to a [`Simulation`](@ref).

# Arguments
- `grid`: a [`SphericalShellGrid`](@ref) or [`SphericalBallGrid`](@ref).
- `T`: floating-point element type (`Float64` by default).
- `Ek`, `Pr`, `Pm`, `Sc`, `Ra`: Ekman, Prandtl, magnetic-Prandtl, Schmidt, and
  Rayleigh numbers.
- `velocity_bcs`, `temperature_bcs`, `composition_bcs`: per-field inner/outer
  `BoundaryConditions` (convenience kwargs, aliases for the NamedTuple form).
- `boundary_conditions`: Oceananigans-style `NamedTuple` of BCs, e.g.
  `(temperature = FieldBoundaryConditions(inner=…, outer=…),)`.  Keys must be a
  subset of `(:velocity, :temperature, :composition)`.  Passing the same field
  through both `boundary_conditions` and a per-field kwarg is an error.
- `include_magnetic`, `include_composition`: enable the magnetic / compositional
  equations.
- `initial_conditions`: initial-condition spec, or `nothing` to start from rest.

Topography- and Stefan-coupling keywords (`topography_enabled`,
`topography_epsilon`, `stefan_enabled`, …) are also accepted; see the
configuration guide.

# Example
```julia
grid  = SphericalShellGrid(lmax = 31, nr = 64)
model = GeodynamoModel(grid; Ra = 1e6, include_magnetic = true)
# Oceananigans-style boundary conditions:
t_bcs = FieldBoundaryConditions(inner = ValueBoundaryCondition(1.0),
                                outer = ValueBoundaryCondition(0.0))
model = GeodynamoModel(grid; Ra = 1e6, boundary_conditions = (temperature = t_bcs,))
```
"""
struct GeodynamoModel{T, A <: AbstractArchitecture, G}
    state::SolverState{T, A}
    grid::G
    clock::Clock{T}
end

# ────────────────────────────────────────────────────────────────────────────────
# Internal constructor helper — avoids duplicating the SolverParameters build
# ────────────────────────────────────────────────────────────────────────────────

function _build_geodynamo_model(
        grid, T::Type, arch_sym::Symbol, geometry::Symbol, radius_ratio::Float64,
        nr_inner::Int, Ek, Pr, Pm, Sc, Ra,
        velocity_bcs, temperature_bcs, composition_bcs,
        include_magnetic, include_composition, initial_conditions,
        topography_enabled, topography_epsilon, topography_degree,
        include_topography_velocity, include_topography_magnetic,
        include_topography_thermal, include_topography_slope_terms,
        include_topography_shift_terms, stefan_enabled, stefan_number,
        inner_core_conductivity_ratio, latent_heat,
        icb_topography_file, ocb_topography_file,
        magnetic_inner_bc::Symbol
)
    params = SolverParameters(
        architecture = arch_sym,
        geometry = geometry,
        lmax = grid.lmax,
        mmax = grid.mmax,
        nlat = grid.nlat,
        nlon = grid.nlon,
        nr = grid.nr,
        nr_inner = nr_inner,
        radius_ratio = radius_ratio,
        Ek = Ek,
        Pr = Pr,
        Pm = Pm,
        Sc = Sc,
        Ra = Ra,
        include_magnetic = include_magnetic,
        include_composition = include_composition,
        velocity_bcs = velocity_bcs,
        temperature_bcs = temperature_bcs,
        composition_bcs = composition_bcs,
        topography_enabled = topography_enabled,
        topography_epsilon = topography_epsilon,
        topography_degree = topography_degree,
        include_topography_velocity = include_topography_velocity,
        include_topography_magnetic = include_topography_magnetic,
        include_topography_thermal = include_topography_thermal,
        include_topography_slope_terms = include_topography_slope_terms,
        include_topography_shift_terms = include_topography_shift_terms,
        stefan_enabled = stefan_enabled,
        stefan_number = stefan_number,
        inner_core_conductivity_ratio = inner_core_conductivity_ratio,
        latent_heat = latent_heat,
        icb_topography_file = icb_topography_file,
        ocb_topography_file = ocb_topography_file,
        magnetic_inner_bc = magnetic_inner_bc
    )
    # Pass the grid's concrete architecture object so a real backend (e.g.
    # `GPU(CUDABackend())`) is preserved end-to-end instead of being flattened
    # to `arch_sym` and rebuilt lossily as `GPU(nothing)`.
    state = initialize_solver_state(T; params = params, arch = grid.arch)
    clock = Clock{T}(T(state.time), state.step, 0, zero(T))
    model = GeodynamoModel{T, typeof(state.backend.architecture), typeof(grid)}(state, grid, clock)
    if !isnothing(initial_conditions)
        for (field_sym, ic) in pairs(initial_conditions)
            set_initial_condition!(model, field_sym, ic)
        end
    end
    return model
end

# ────────────────────────────────────────────────────────────────────────────────
# BC resolver — merges the Oceananigans-style `boundary_conditions` NamedTuple
# with the per-field kwargs.  A field given through BOTH paths is ambiguous →
# error.  Unspecified fields fall back to `defaults`.
# ────────────────────────────────────────────────────────────────────────────────

function _resolve_bcs(boundary_conditions, velocity_bcs, temperature_bcs,
        composition_bcs, defaults)
    boundary_conditions === nothing &&
        return (something(velocity_bcs, defaults.velocity),
                something(temperature_bcs, defaults.temperature),
                something(composition_bcs, defaults.composition))
    valid = (:velocity, :temperature, :composition)
    for k in keys(boundary_conditions)
        k in valid || throw(ArgumentError(
            "GeodynamoModel: unknown boundary_conditions field :$k " *
            "(valid: :velocity, :temperature, :composition)"))
    end
    conflict(name, kwarg) = kwarg !== nothing && haskey(boundary_conditions, name) &&
        throw(ArgumentError(
            "GeodynamoModel: $(name) BCs given both via boundary_conditions " *
            "NamedTuple and the $(name)_bcs kwarg — pass one"))
    conflict(:velocity, velocity_bcs)
    conflict(:temperature, temperature_bcs)
    conflict(:composition, composition_bcs)
    vel  = haskey(boundary_conditions, :velocity)    ? boundary_conditions.velocity    :
           something(velocity_bcs,    defaults.velocity)
    temp = haskey(boundary_conditions, :temperature) ? boundary_conditions.temperature :
           something(temperature_bcs, defaults.temperature)
    comp = haskey(boundary_conditions, :composition) ? boundary_conditions.composition :
           something(composition_bcs, defaults.composition)
    return (vel, temp, comp)
end

# ────────────────────────────────────────────────────────────────────────────────
# Public constructors
# ────────────────────────────────────────────────────────────────────────────────

function GeodynamoModel(grid::SphericalShellGrid;
        T::Type = Float64,
        Ek::Real = 1e-4,
        Pr::Real = 1.0,
        Pm::Real = 1.0,
        Sc::Real = 1.0,
        Ra::Real = 1e6,
        velocity_bcs = nothing,
        temperature_bcs = nothing,
        composition_bcs = nothing,
        boundary_conditions = nothing,
        include_magnetic::Bool = false,
        include_composition::Bool = false,
        initial_conditions = nothing,
        topography_enabled::Bool = false,
        topography_epsilon::Real = 0.01,
        topography_degree::Int = -1,
        include_topography_velocity::Bool = true,
        include_topography_magnetic::Bool = true,
        include_topography_thermal::Bool = true,
        include_topography_slope_terms::Bool = true,
        include_topography_shift_terms::Bool = true,
        stefan_enabled::Bool = false,
        stefan_number::Real = 1.0,
        inner_core_conductivity_ratio::Real = 1.0,
        latent_heat::Real = 1.0,
        icb_topography_file::AbstractString = "",
        ocb_topography_file::AbstractString = "",
        magnetic_inner_bc::Symbol = :insulating
)
    velocity_bcs, temperature_bcs, composition_bcs = _resolve_bcs(
        boundary_conditions, velocity_bcs, temperature_bcs, composition_bcs,
        (velocity    = BoundaryConditions(inner = NoSlip(),       outer = NoSlip()),
         temperature = BoundaryConditions(inner = FixedFlux(1.0), outer = FixedTemperature(0.0)),
         composition = BoundaryConditions(inner = FixedFlux(0.0), outer = FixedTemperature(0.0))))
    arch_sym = grid.arch isa CPU ? :cpu : :gpu
    return _build_geodynamo_model(grid, T, arch_sym, :shell,
        grid.r_inner / grid.r_outer,
        grid.nr_inner, Float64(Ek), Float64(Pr), Float64(Pm), Float64(Sc), Float64(Ra),
        velocity_bcs, temperature_bcs, composition_bcs,
        include_magnetic, include_composition, initial_conditions,
        topography_enabled, Float64(topography_epsilon), topography_degree,
        include_topography_velocity, include_topography_magnetic,
        include_topography_thermal, include_topography_slope_terms,
        include_topography_shift_terms, stefan_enabled, Float64(stefan_number),
        Float64(inner_core_conductivity_ratio), Float64(latent_heat),
        String(icb_topography_file), String(ocb_topography_file),
        magnetic_inner_bc)
end

# ────────────────────────────────────────────────────────────────────────────────
# Oceananigans-style field access: model.velocity / model.temperature /
# model.magnetic / model.composition forward to the solver state's fields
# (`nothing` when the field is disabled).
# ────────────────────────────────────────────────────────────────────────────────
const _MODEL_FIELD_PROPS = (:velocity, :temperature, :magnetic, :composition)

function Base.getproperty(m::GeodynamoModel, name::Symbol)
    name in _MODEL_FIELD_PROPS &&
        return getproperty(getfield(m, :state).fields, name)
    return getfield(m, name)
end
Base.propertynames(m::GeodynamoModel) = (fieldnames(GeodynamoModel)..., _MODEL_FIELD_PROPS...)

function GeodynamoModel(grid::SphericalBallGrid;
        T::Type = Float64,
        Ek::Real = 1e-4,
        Pr::Real = 1.0,
        Pm::Real = 1.0,
        Sc::Real = 1.0,
        Ra::Real = 1e6,
        velocity_bcs = nothing,
        temperature_bcs = nothing,
        composition_bcs = nothing,
        boundary_conditions = nothing,
        include_magnetic::Bool = false,
        include_composition::Bool = false,
        initial_conditions = nothing,
        topography_enabled::Bool = false,
        topography_epsilon::Real = 0.01,
        topography_degree::Int = -1,
        include_topography_velocity::Bool = true,
        include_topography_magnetic::Bool = true,
        include_topography_thermal::Bool = true,
        include_topography_slope_terms::Bool = true,
        include_topography_shift_terms::Bool = true,
        stefan_enabled::Bool = false,
        stefan_number::Real = 1.0,
        inner_core_conductivity_ratio::Real = 1.0,
        latent_heat::Real = 1.0,
        icb_topography_file::AbstractString = "",
        ocb_topography_file::AbstractString = "",
        magnetic_inner_bc::Symbol = :insulating
)
    velocity_bcs, temperature_bcs, composition_bcs = _resolve_bcs(
        boundary_conditions, velocity_bcs, temperature_bcs, composition_bcs,
        (velocity    = BoundaryConditions(inner = NoSlip(),       outer = NoSlip()),
         temperature = BoundaryConditions(inner = FixedFlux(1.0), outer = FixedTemperature(0.0)),
         composition = BoundaryConditions(inner = FixedFlux(0.0), outer = FixedTemperature(0.0))))
    arch_sym = grid.arch isa CPU ? :cpu : :gpu
    return _build_geodynamo_model(grid, T, arch_sym, :ball,
        0.0,
        0, Float64(Ek), Float64(Pr), Float64(Pm), Float64(Sc), Float64(Ra),
        velocity_bcs, temperature_bcs, composition_bcs,
        include_magnetic, include_composition, initial_conditions,
        topography_enabled, Float64(topography_epsilon), topography_degree,
        include_topography_velocity, include_topography_magnetic,
        include_topography_thermal, include_topography_slope_terms,
        include_topography_shift_terms, stefan_enabled, Float64(stefan_number),
        Float64(inner_core_conductivity_ratio), Float64(latent_heat),
        String(icb_topography_file), String(ocb_topography_file),
        magnetic_inner_bc)
end
