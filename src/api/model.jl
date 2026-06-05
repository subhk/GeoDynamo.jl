"""
    GeodynamoModel(grid; T=Float64, Ek=1e-4, Pr=1.0, Pm=1.0, Sc=1.0, Ra=1e6,
                   velocity_bcs, temperature_bcs, composition_bcs,
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
  `BoundaryConditions`.
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
# Public constructors
# ────────────────────────────────────────────────────────────────────────────────

function GeodynamoModel(grid::SphericalShellGrid;
        T::Type = Float64,
        Ek::Real = 1e-4,
        Pr::Real = 1.0,
        Pm::Real = 1.0,
        Sc::Real = 1.0,
        Ra::Real = 1e6,
        velocity_bcs = BoundaryConditions(inner = NoSlip(), outer = NoSlip()),
        temperature_bcs = BoundaryConditions(inner = FixedFlux(1.0), outer = FixedTemperature(0.0)),
        composition_bcs = BoundaryConditions(inner = FixedFlux(0.0), outer = FixedTemperature(0.0)),
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

function GeodynamoModel(grid::SphericalBallGrid;
        T::Type = Float64,
        Ek::Real = 1e-4,
        Pr::Real = 1.0,
        Pm::Real = 1.0,
        Sc::Real = 1.0,
        Ra::Real = 1e6,
        velocity_bcs = BoundaryConditions(inner = NoSlip(), outer = NoSlip()),
        temperature_bcs = BoundaryConditions(inner = FixedFlux(1.0), outer = FixedTemperature(0.0)),
        composition_bcs = BoundaryConditions(inner = FixedFlux(0.0), outer = FixedTemperature(0.0)),
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
