# Whitespace-insensitive source matching: SciML auto-formatting (spacing,
# line wraps) must not break these static source-contract checks.
_sc_wsn(s) = replace(s, r"\s+" => "")
_sc_occ(pat::AbstractString, src) = occursin(_sc_wsn(pat), _sc_wsn(src))
_sc_occ(pat::Regex, src) = occursin(pat, replace(src, r"\s+" => " "))  # collapse ws so wrapped calls still match

using Test

const TEMPERATURE_BC_STATIC_ROOT = normpath(joinpath(@__DIR__, ".."))

function _temperature_bc_static_source(path_parts...)
    return read(joinpath(TEMPERATURE_BC_STATIC_ROOT, path_parts...), String)
end

function _temperature_bc_static_function_body(source::String, signature::String)
    start = findfirst(signature, source)
    start === nothing && error("Could not find function signature: $signature")
    next_function = findnext("\nfunction ", source, last(start) + 1)
    return next_function === nothing ? source[first(start):end] :
           source[first(start):(first(next_function) - 1)]
end

@testset "Temperature boundary-condition static contract" begin
    api = _temperature_bc_static_source("src", "api", "boundary_conditions.jl")
    scalar_bc = _temperature_bc_static_source("src", "bcs", "scalar_bc.jl")
    thermal_bc = _temperature_bc_static_source("src", "bcs", "thermal_bc.jl")
    temperature_solver = _temperature_bc_static_source("src", "physics", "temperature", "solver.jl")
    backend = _temperature_bc_static_source("src", "solver", "backend.jl")
    numerics = _temperature_bc_static_source("src", "solver", "numerics.jl")
    imex = _temperature_bc_static_source("src", "timestep", "imex.jl")
    erk2 = _temperature_bc_static_source("src", "timestep", "erk2.jl")

    @test _sc_occ("_thermal_bc_code(::BoundaryConditions{<:FixedTemperature, <:FixedTemperature}) = 1", api)
    @test _sc_occ("_thermal_bc_code(::BoundaryConditions{<:FixedTemperature, <:FixedFlux})        = 2", api)
    @test _sc_occ("_thermal_bc_code(::BoundaryConditions{<:FixedFlux,        <:FixedTemperature}) = 3", api)
    @test _sc_occ("_thermal_bc_code(::BoundaryConditions{<:FixedFlux,        <:FixedFlux})        = 4", api)

    temperature_matrices = _temperature_bc_static_function_body(
        thermal_bc,
        "function create_temperature_matrices("
    )
    @test _sc_occ("create_scalar_matrices(", temperature_matrices)
    @test _sc_occ("scalar_bc_code=temperature_bc_code", temperature_matrices)

    scalar_rows = _temperature_bc_static_function_body(
        scalar_bc,
        "function _apply_scalar_boundary_rows!("
    )
    @test _sc_occ("system_data[bw + 1, 1] = one(T)", scalar_rows)
    @test _sc_occ("system_data[bw + 1, N] = one(T)", scalar_rows)
    @test _sc_occ("= d1_data", scalar_rows)
    @test _sc_occ("scalar_bc_code == 4 && l == 0", scalar_rows)

    legacy_temperature_solve = _temperature_bc_static_function_body(
        thermal_bc,
        "function solve_temperature_implicit_step!("
    )
    @test _sc_occ("solve_scalar_implicit_step!(", legacy_temperature_solve)
    @test !_sc_occ("get_local_range(solution.pencil, 1)", legacy_temperature_solve)

    shared_scalar_solve = _temperature_bc_static_function_body(
        scalar_bc,
        "function solve_scalar_implicit_step!("
    )
    @test _sc_occ("local_spectral_mode_indices(solution.config)", shared_scalar_solve)

    get_bc_vectors = _temperature_bc_static_function_body(
        numerics,
        "function get_bc_vectors(field)"
    )
    @test _sc_occ("field.boundary_values", get_bc_vectors)
    @test _sc_occ("inner_real=view(field.boundary_values, 1, :)", get_bc_vectors)
    @test _sc_occ("inner_imag=nothing", get_bc_vectors)

    runtime_create = _temperature_bc_static_function_body(
        backend,
        "function create_solver_runtime("
    )
    @test _sc_occ("apply_scalar_boundary_parameters!(temperature", runtime_create)

    scalar_solve = _temperature_bc_static_function_body(
        imex,
        "function _solver_solve_scalar_implicit_step!("
    )
    @test _sc_occ("Union{AbstractVector{T}, Nothing}", scalar_solve)
    @test _sc_occ("local_spectral_mode_indices(solution.config)", scalar_solve)

    state = _temperature_bc_static_source("src", "solver", "state.jl")
    boundary_spec = _temperature_bc_static_function_body(
        state,
        "struct SolverERK2BoundarySpec"
    )
    @test _sc_occ("inner_mode_values_imag", boundary_spec)
    @test _sc_occ("outer_mode_values_imag", boundary_spec)
    @test _sc_occ("AbstractVector{T}", boundary_spec)

    prepare_erk2 = _temperature_bc_static_function_body(
        erk2,
        "function prepare_solver_erk2_field!("
    )
    finalize_erk2 = _temperature_bc_static_function_body(
        erk2,
        "function finalize_solver_erk2_field!("
    )
    @test _sc_occ("bc_spec.inner_mode_values_imag", prepare_erk2)
    @test _sc_occ("bc_spec.outer_mode_values_imag", prepare_erk2)
    @test _sc_occ("bc_spec.inner_mode_values_imag", finalize_erk2)
    @test _sc_occ("bc_spec.outer_mode_values_imag", finalize_erk2)

    temperature_update = _temperature_bc_static_function_body(
        temperature_solver,
        "function apply_temperature_implicit_update!("
    )
    @test _sc_occ("build_solver_erk2_scalar_bc", temperature_update)
    @test _sc_occ("with_boundary_mode_values", temperature_update)
    @test _sc_occ("bc_spec=bc_spec", temperature_update)
    @test _sc_occ("_timestepper_krylov_dimension(timestepper, state.parameters)", temperature_update)
    @test !_sc_occ("_timestepper_krylov_dimension(state.parameters.timestepper)", temperature_update)

    integrate_erk2 = _temperature_bc_static_function_body(
        erk2,
        "function integrate_solver_erk2_step!("
    )
    @test _sc_occ("temp_bc_values = get_bc_vectors(state.fields.temperature)", integrate_erk2)
    @test _sc_occ("temp_bc = with_boundary_mode_values(", integrate_erk2)
end
