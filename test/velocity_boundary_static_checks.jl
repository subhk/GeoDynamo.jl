# Whitespace-insensitive source matching: SciML auto-formatting (spacing,
# line wraps) must not break these static source-contract checks.
_sc_wsn(s) = replace(s, r"\s+" => "")
_sc_occ(pat::AbstractString, src) = occursin(_sc_wsn(pat), _sc_wsn(src))
_sc_occ(pat::Regex, src) = occursin(pat, replace(src, r"\s+" => " "))  # collapse ws so wrapped calls still match

using Test

const VELOCITY_BC_STATIC_ROOT = normpath(joinpath(@__DIR__, ".."))

function _velocity_bc_static_source(path_parts...)
    return read(joinpath(VELOCITY_BC_STATIC_ROOT, path_parts...), String)
end

function _velocity_bc_static_function_body(source::String, signature::String)
    start = findfirst(signature, source)
    start === nothing && error("Could not find function signature: $signature")
    next_function = findnext("\nfunction ", source, last(start) + 1)
    return next_function === nothing ? source[first(start):end] :
           source[first(start):(first(next_function) - 1)]
end

@testset "Velocity boundary-condition static contract" begin
    api = _velocity_bc_static_source("src", "api", "boundary_conditions.jl")
    velocity_bc = _velocity_bc_static_source("src", "bcs", "velocity_bc.jl")
    velocity_solver = _velocity_bc_static_source("src", "physics", "velocity", "solver.jl")
    imex = _velocity_bc_static_source("src", "timestep", "imex.jl")
    erk2 = join([_velocity_bc_static_source("src", "timestep", "erk2", f) for f in
                 ("common.jl", "boundary.jl", "cache.jl", "influence.jl", "integrate.jl")], "\n")

    @test _sc_occ("_velocity_bc_code(::BoundaryConditions{NoSlip,     NoSlip})     = 1", api)
    @test _sc_occ("_velocity_bc_code(::BoundaryConditions{NoSlip,     StressFree}) = 2", api)
    @test _sc_occ("_velocity_bc_code(::BoundaryConditions{StressFree, NoSlip})     = 3", api)
    @test _sc_occ("_velocity_bc_code(::BoundaryConditions{StressFree, StressFree}) = 4", api)

    tor_matrices = _velocity_bc_static_function_body(
        velocity_bc,
        "function create_velocity_toroidal_matrices("
    )
    pol_matrices = _velocity_bc_static_function_body(
        velocity_bc,
        "function create_velocity_poloidal_matrices("
    )
    @test _sc_occ("system_data[bw + 1, 1] = one(T)", tor_matrices)
    @test _sc_occ("system_data[bw + 1, N] = one(T)", tor_matrices)
    @test _sc_occ("system_data[bw + 1, 1] -= T(domain.r[1, 3])", tor_matrices)
    @test _sc_occ("system_data[bw + 1, N] -= T(domain.r[N, 3])", tor_matrices)
    @test _sc_occ("= d1_matrix.data", pol_matrices)
    @test _sc_occ("= d2_matrix.data", pol_matrices)

    legacy_velocity_solve = _velocity_bc_static_function_body(
        velocity_bc,
        "function solve_velocity_implicit_step!("
    )
    velocity_solve = _velocity_bc_static_function_body(
        imex,
        "function solver_solve_velocity_implicit_step!("
    )
    influence_correction = _velocity_bc_static_function_body(
        erk2,
        "function apply_solver_velocity_poloidal_influence_correction!("
    )
    @test _sc_occ("local_spectral_mode_indices(solution.config)", legacy_velocity_solve)
    @test _sc_occ("local_spectral_mode_indices(solution.config)", velocity_solve)
    @test _sc_occ("local_spectral_mode_indices(config)", influence_correction)
    @test !_sc_occ("get_local_range(solution.pencil, 1)", legacy_velocity_solve)
    @test !_sc_occ("local_range(solution.pencil, 1)", velocity_solve)
    @test !_sc_occ("local_range(field.pencil, 1)", influence_correction)

    eab2_update = _velocity_bc_static_function_body(
        imex,
        "function solver_eab2_update_krylov_cached!("
    )
    prepare_erk2 = _velocity_bc_static_function_body(
        erk2,
        "function prepare_solver_erk2_field!("
    )
    finalize_erk2 = _velocity_bc_static_function_body(
        erk2,
        "function finalize_solver_erk2_field!("
    )
    @test _sc_occ("bc_spec=nothing", eab2_update)
    @test _sc_occ("solver_enforce_erk2_bc!", eab2_update)
    @test !_sc_occ("local_range(u.pencil, 1)", eab2_update)
    @test !_sc_occ("local_range(u.pencil, 1)", prepare_erk2)
    @test !_sc_occ("local_range(u.pencil, 1)", finalize_erk2)
    @test _sc_occ("bc_spec.inner_mode_values_imag", prepare_erk2)
    @test _sc_occ("bc_spec.inner_mode_values_imag", finalize_erk2)
    @test _sc_occ("theta = _timestepper_implicit_theta(params.timestepper, params)", erk2)

    poloidal_update = _velocity_bc_static_function_body(
        velocity_solver,
        "function apply_velocity_poloidal_implicit_update!("
    )
    toroidal_update = _velocity_bc_static_function_body(
        velocity_solver,
        "function apply_velocity_toroidal_implicit_update!("
    )
    no_penetration = _velocity_bc_static_function_body(
        velocity_solver,
        "function apply_velocity_poloidal_no_penetration!("
    )
    @test _sc_occ("bc_spec=bc_spec", toroidal_update)
    @test _sc_occ("bc_spec=bc_spec", poloidal_update)
    @test _sc_occ("_timestepper_krylov_dimension(timestepper, state.parameters)", toroidal_update)
    @test _sc_occ("_timestepper_krylov_dimension(timestepper, state.parameters)", poloidal_update)
    @test _sc_occ("apply_velocity_poloidal_no_penetration!(state, velocity_bc)", poloidal_update)
    @test _sc_occ("get_solver_erk2_influence_matrices!", no_penetration)
    @test _sc_occ("apply_solver_velocity_poloidal_influence_correction!", no_penetration)
end
