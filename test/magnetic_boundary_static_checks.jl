using Test

const MAGNETIC_BC_STATIC_ROOT = normpath(joinpath(@__DIR__, ".."))

function _magnetic_bc_static_source(path_parts...)
    return read(joinpath(MAGNETIC_BC_STATIC_ROOT, path_parts...), String)
end

function _magnetic_bc_static_function_body(source::String, signature::String)
    start = findfirst(signature, source)
    start === nothing && error("Could not find function signature: $signature")
    next_function = findnext("\nfunction ", source, last(start) + 1)
    return next_function === nothing ? source[first(start):end] : source[first(start):first(next_function)-1]
end

@testset "Magnetic boundary-condition static contract" begin
    magnetic_bc = _magnetic_bc_static_source("src", "bcs", "magnetic_bc.jl")
    magnetic_solver = _magnetic_bc_static_source("src", "physics", "magnetic", "solver.jl")
    erk2 = _magnetic_bc_static_source("src", "timestep", "erk2.jl")

    tor_matrices = _magnetic_bc_static_function_body(
        magnetic_bc,
        "function create_magnetic_toroidal_matrices(",
    )
    pol_matrices = _magnetic_bc_static_function_body(
        magnetic_bc,
        "function create_magnetic_poloidal_matrices(",
    )
    @test occursin("system_data[bw + 1, 1] = one(T)", tor_matrices)
    @test occursin("system_data[bw + 1, N] = one(T)", tor_matrices)
    @test occursin("system_data[bw + 1, 1] -= T(l * domain.r[1, 3])", pol_matrices)
    @test occursin("system_data[bw + 1, N] += T((l + 1) * domain.r[N, 3])", pol_matrices)

    legacy_magnetic_solve = _magnetic_bc_static_function_body(
        magnetic_bc,
        "function solve_magnetic_implicit_step!(",
    )
    toroidal_inner_bc = _magnetic_bc_static_function_body(
        magnetic_solver,
        "function _magnetic_toroidal_inner_bc_increment(",
    )
    @test occursin("local_spectral_mode_indices(solution.config)", legacy_magnetic_solve)
    @test occursin("local_spectral_mode_indices(magnetic.𝒯.config)", toroidal_inner_bc)
    @test !occursin("get_local_range(solution.pencil, 1)", legacy_magnetic_solve)
    @test !occursin("local_range(magnetic.𝒯.pencil, 1)", toroidal_inner_bc)

    insulating_inner = _magnetic_bc_static_function_body(
        erk2,
        "function solver_create_insulating_inner_bc(",
    )
    insulating_outer = _magnetic_bc_static_function_body(
        erk2,
        "function solver_create_insulating_outer_bc(",
    )
    @test occursin("-one(T)", insulating_inner)
    @test occursin("one(T)", insulating_outer)
    @test occursin("r_inv", insulating_outer)

    magnetic_tor_update = _magnetic_bc_static_function_body(
        magnetic_solver,
        "function apply_magnetic_toroidal_implicit_update!(",
    )
    magnetic_pol_update = _magnetic_bc_static_function_body(
        magnetic_solver,
        "function apply_magnetic_poloidal_implicit_update!(",
    )
    @test occursin("build_solver_erk2_magnetic_tor_bc", magnetic_tor_update)
    @test occursin("build_solver_erk2_magnetic_pol_bc", magnetic_pol_update)
    @test occursin("bc_spec=bc_spec", magnetic_tor_update)
    @test occursin("bc_spec=bc_spec", magnetic_pol_update)
    @test occursin("_timestepper_krylov_dimension(timestepper, state.parameters)", magnetic_tor_update)
    @test occursin("_timestepper_krylov_dimension(timestepper, state.parameters)", magnetic_pol_update)
    @test !occursin("_timestepper_krylov_dimension(state.parameters.timestepper)", magnetic_tor_update)
    @test !occursin("_timestepper_krylov_dimension(state.parameters.timestepper)", magnetic_pol_update)

    integrate_erk2 = _magnetic_bc_static_function_body(
        erk2,
        "function integrate_solver_erk2_step!(",
    )
    @test occursin("build_solver_erk2_magnetic_tor_bc", integrate_erk2)
    @test occursin("build_solver_erk2_magnetic_pol_bc", integrate_erk2)
    @test occursin("bc_spec=mag_tor_bc", integrate_erk2)
    @test occursin("bc_spec=mag_pol_bc", integrate_erk2)
end
