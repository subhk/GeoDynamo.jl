using Test

const TEMPERATURE_BC_STATIC_ROOT = normpath(joinpath(@__DIR__, ".."))

function _temperature_bc_static_source(path_parts...)
    return read(joinpath(TEMPERATURE_BC_STATIC_ROOT, path_parts...), String)
end

function _temperature_bc_static_function_body(source::String, signature::String)
    start = findfirst(signature, source)
    start === nothing && error("Could not find function signature: $signature")
    next_function = findnext("\nfunction ", source, last(start) + 1)
    return next_function === nothing ? source[first(start):end] : source[first(start):first(next_function)-1]
end

@testset "Temperature boundary-condition static contract" begin
    api = _temperature_bc_static_source("src", "api", "boundary_conditions.jl")
    thermal_bc = _temperature_bc_static_source("src", "bcs", "thermal_bc.jl")
    temperature_solver = _temperature_bc_static_source("src", "physics", "temperature", "solver.jl")
    backend = _temperature_bc_static_source("src", "solver", "backend.jl")
    numerics = _temperature_bc_static_source("src", "solver", "numerics.jl")
    imex = _temperature_bc_static_source("src", "timestep", "imex.jl")
    erk2 = _temperature_bc_static_source("src", "timestep", "erk2.jl")

    @test occursin("_thermal_bc_code(::BoundaryConditions{<:FixedTemperature, <:FixedTemperature}) = 1", api)
    @test occursin("_thermal_bc_code(::BoundaryConditions{<:FixedTemperature, <:FixedFlux})        = 2", api)
    @test occursin("_thermal_bc_code(::BoundaryConditions{<:FixedFlux,        <:FixedTemperature}) = 3", api)
    @test occursin("_thermal_bc_code(::BoundaryConditions{<:FixedFlux,        <:FixedFlux})        = 4", api)

    matrices = _temperature_bc_static_function_body(
        thermal_bc,
        "function create_temperature_matrices(",
    )
    @test occursin("system_data[bw + 1, 1] = one(T)", matrices)
    @test occursin("system_data[bw + 1, N] = one(T)", matrices)
    @test occursin("= d1_matrix.data", matrices)
    @test occursin("temperature_bc_code == 4 && l == 0", matrices)

    legacy_temperature_solve = _temperature_bc_static_function_body(
        thermal_bc,
        "function solve_temperature_implicit_step!(",
    )
    @test occursin("local_spectral_mode_indices(solution.config)", legacy_temperature_solve)
    @test !occursin("get_local_range(solution.pencil, 1)", legacy_temperature_solve)

    get_bc_vectors = _temperature_bc_static_function_body(
        numerics,
        "function get_bc_vectors(field)",
    )
    @test occursin("field.boundary_values", get_bc_vectors)
    @test occursin("inner_real=view(field.boundary_values, 1, :)", get_bc_vectors)
    @test occursin("inner_imag=nothing", get_bc_vectors)

    runtime_create = _temperature_bc_static_function_body(
        backend,
        "function create_solver_runtime(",
    )
    @test occursin("solver_apply_scalar_boundary_parameters!(temperature", runtime_create)

    scalar_solve = _temperature_bc_static_function_body(
        imex,
        "function _solver_solve_scalar_implicit_step!(",
    )
    @test occursin("Union{AbstractVector{T}, Nothing}", scalar_solve)
    @test occursin("local_spectral_mode_indices(solution.config)", scalar_solve)

    boundary_spec = _temperature_bc_static_function_body(
        erk2,
        "struct SolverERK2BoundarySpec",
    )
    @test occursin("inner_mode_values_imag", boundary_spec)
    @test occursin("outer_mode_values_imag", boundary_spec)
    @test occursin("AbstractVector{T}", boundary_spec)

    prepare_erk2 = _temperature_bc_static_function_body(
        erk2,
        "function prepare_solver_erk2_field!(",
    )
    finalize_erk2 = _temperature_bc_static_function_body(
        erk2,
        "function finalize_solver_erk2_field!(",
    )
    @test occursin("bc_spec.inner_mode_values_imag", prepare_erk2)
    @test occursin("bc_spec.outer_mode_values_imag", prepare_erk2)
    @test occursin("bc_spec.inner_mode_values_imag", finalize_erk2)
    @test occursin("bc_spec.outer_mode_values_imag", finalize_erk2)

    temperature_update = _temperature_bc_static_function_body(
        temperature_solver,
        "function apply_temperature_implicit_update!(",
    )
    @test occursin("build_solver_erk2_scalar_bc", temperature_update)
    @test occursin("solver_with_boundary_mode_values", temperature_update)
    @test occursin("bc_spec=bc_spec", temperature_update)
    @test occursin("_timestepper_krylov_dimension(timestepper, state.parameters)", temperature_update)
    @test !occursin("_timestepper_krylov_dimension(state.parameters.timestepper)", temperature_update)

    integrate_erk2 = _temperature_bc_static_function_body(
        erk2,
        "function integrate_solver_erk2_step!(",
    )
    @test occursin("temp_bc_values = get_bc_vectors(state.fields.temperature)", integrate_erk2)
    @test occursin("temp_bc = solver_with_boundary_mode_values(temp_bc", integrate_erk2)
end
