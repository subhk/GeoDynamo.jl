using Test

const COMPOSITION_BC_STATIC_ROOT = normpath(joinpath(@__DIR__, ".."))

function __composition_bc_static_source(path_parts...)
    return read(joinpath(COMPOSITION_BC_STATIC_ROOT, path_parts...), String)
end

function __composition_bc_static_function_body(source::String, signature::String)
    start = findfirst(signature, source)
    start === nothing && error("Could not find function signature: $signature")
    next_function = findnext("\nfunction ", source, last(start) + 1)
    return next_function === nothing ? source[first(start):end] : source[first(start):first(next_function)-1]
end

@testset "Composition boundary-condition static contract" begin
    api = __composition_bc_static_source("src", "api", "boundary_conditions.jl")
    scalar_bc = __composition_bc_static_source("src", "bcs", "scalar_bc.jl")
    composition_bc = __composition_bc_static_source("src", "bcs", "compositional_bc.jl")
    composition_solver = __composition_bc_static_source("src", "physics", "composition", "solver.jl")
    backend = __composition_bc_static_source("src", "solver", "backend.jl")
    numerics = __composition_bc_static_source("src", "solver", "numerics.jl")
    imex = __composition_bc_static_source("src", "timestep", "imex.jl")
    erk2 = __composition_bc_static_source("src", "timestep", "erk2.jl")

    @test occursin("__composition_bc_code(bc) = __thermal_bc_code(bc)", api)

    composition_matrices = __composition_bc_static_function_body(
        composition_bc,
        "function create_composition_matrices(",
    )
    @test occursin("create_scalar_matrices(", composition_matrices)
    @test occursin("scalar_bc_code=composition_bc_code", composition_matrices)

    scalar_rows = __composition_bc_static_function_body(
        scalar_bc,
        "function __apply_scalar_boundary_rows!(",
    )
    @test occursin("system_data[bw + 1, 1] = one(T)", scalar_rows)
    @test occursin("system_data[bw + 1, N] = one(T)", scalar_rows)
    @test occursin("= d1_data", scalar_rows)
    @test occursin("scalar_bc_code == 4 && l == 0", scalar_rows)

    legacy_composition_solve = __composition_bc_static_function_body(
        composition_bc,
        "function solve_composition_implicit_step!(",
    )
    @test occursin("Union{AbstractVector{T},Nothing}", legacy_composition_solve)
    @test occursin("solve_scalar_implicit_step!(", legacy_composition_solve)
    @test !occursin("get_local_range(solution.pencil, 1)", legacy_composition_solve)

    shared_scalar_solve = __composition_bc_static_function_body(
        scalar_bc,
        "function solve_scalar_implicit_step!(",
    )
    @test occursin("local_spectral_mode_indices(solution.config)", shared_scalar_solve)

    get_bc_vectors = __composition_bc_static_function_body(
        numerics,
        "function get_bc_vectors(field)",
    )
    @test occursin("inner_real=view(field.boundary_values, 1, :)", get_bc_vectors)
    @test occursin("inner_imag=nothing", get_bc_vectors)

    runtime_create = __composition_bc_static_function_body(
        backend,
        "function create_solver_runtime(",
    )
    @test occursin("apply_scalar_boundary_parameters!(composition", runtime_create)

    scalar_solve = __composition_bc_static_function_body(
        imex,
        "function __solver_solve_scalar_implicit_step!(",
    )
    @test occursin("Union{AbstractVector{T}, Nothing}", scalar_solve)
    @test occursin("local_spectral_mode_indices(solution.config)", scalar_solve)

    composition_update = __composition_bc_static_function_body(
        composition_solver,
        "function apply_composition_implicit_update!(",
    )
    @test occursin("build_solver_erk2_scalar_bc", composition_update)
    @test occursin("with_boundary_mode_values", composition_update)
    @test occursin("bc_spec=bc_spec", composition_update)
    @test occursin("__timestepper_krylov_dimension(timestepper, state.parameters)", composition_update)
    @test !occursin("__timestepper_krylov_dimension(state.parameters.timestepper)", composition_update)

    integrate_erk2 = __composition_bc_static_function_body(
        erk2,
        "function integrate_solver_erk2_step!(",
    )
    @test occursin("comp_bc_values = get_bc_vectors(state.fields.composition)", integrate_erk2)
    @test occursin("comp_bc = with_boundary_mode_values(", integrate_erk2)
end
