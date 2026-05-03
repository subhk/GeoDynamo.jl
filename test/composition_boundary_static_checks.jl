using Test

const COMPOSITION_BC_STATIC_ROOT = normpath(joinpath(@__DIR__, ".."))

function _composition_bc_static_source(path_parts...)
    return read(joinpath(COMPOSITION_BC_STATIC_ROOT, path_parts...), String)
end

function _composition_bc_static_function_body(source::String, signature::String)
    start = findfirst(signature, source)
    start === nothing && error("Could not find function signature: $signature")
    next_function = findnext("\nfunction ", source, last(start) + 1)
    return next_function === nothing ? source[first(start):end] : source[first(start):first(next_function)-1]
end

@testset "Composition boundary-condition static contract" begin
    api = _composition_bc_static_source("src", "api", "boundary_conditions.jl")
    composition_bc = _composition_bc_static_source("src", "bcs", "compositional_bc.jl")
    composition_solver = _composition_bc_static_source("src", "physics", "composition", "solver.jl")
    backend = _composition_bc_static_source("src", "solver", "backend.jl")
    numerics = _composition_bc_static_source("src", "solver", "numerics.jl")
    imex = _composition_bc_static_source("src", "timestep", "imex.jl")
    erk2 = _composition_bc_static_source("src", "timestep", "erk2.jl")

    @test occursin("_composition_bc_code(bc) = _thermal_bc_code(bc)", api)

    matrices = _composition_bc_static_function_body(
        composition_bc,
        "function create_composition_matrices(",
    )
    @test occursin("system_data[bw + 1, 1] = one(T)", matrices)
    @test occursin("system_data[bw + 1, N] = one(T)", matrices)
    @test occursin("= d1_matrix.data", matrices)
    @test occursin("composition_bc_code == 4 && l == 0", matrices)

    legacy_composition_solve = _composition_bc_static_function_body(
        composition_bc,
        "function solve_composition_implicit_step!(",
    )
    @test occursin("Union{AbstractVector{T},Nothing}", legacy_composition_solve)
    @test occursin("local_spectral_mode_indices(solution.config)", legacy_composition_solve)
    @test !occursin("get_local_range(solution.pencil, 1)", legacy_composition_solve)

    get_bc_vectors = _composition_bc_static_function_body(
        numerics,
        "function get_bc_vectors(field)",
    )
    @test occursin("inner_real=view(field.boundary_values, 1, :)", get_bc_vectors)
    @test occursin("inner_imag=nothing", get_bc_vectors)

    runtime_create = _composition_bc_static_function_body(
        backend,
        "function create_solver_runtime(",
    )
    @test occursin("solver_apply_scalar_boundary_parameters!(composition", runtime_create)

    scalar_solve = _composition_bc_static_function_body(
        imex,
        "function _solver_solve_scalar_implicit_step!(",
    )
    @test occursin("Union{AbstractVector{T}, Nothing}", scalar_solve)
    @test occursin("local_spectral_mode_indices(solution.config)", scalar_solve)

    composition_update = _composition_bc_static_function_body(
        composition_solver,
        "function apply_composition_implicit_update!(",
    )
    @test occursin("build_solver_erk2_scalar_bc", composition_update)
    @test occursin("solver_with_boundary_mode_values", composition_update)
    @test occursin("bc_spec=bc_spec", composition_update)
    @test occursin("_timestepper_krylov_dimension(timestepper, state.parameters)", composition_update)
    @test !occursin("_timestepper_krylov_dimension(state.parameters.timestepper)", composition_update)

    integrate_erk2 = _composition_bc_static_function_body(
        erk2,
        "function integrate_solver_erk2_step!(",
    )
    @test occursin("comp_bc_values = get_bc_vectors(state.fields.composition)", integrate_erk2)
    @test occursin("comp_bc = solver_with_boundary_mode_values(", integrate_erk2)
end
