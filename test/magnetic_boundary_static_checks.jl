# Whitespace-insensitive source matching: SciML auto-formatting (spacing,
# line wraps) must not break these static source-contract checks.
_sc_wsn(s) = replace(s, r"\s+" => "")
_sc_occ(pat::AbstractString, src) = occursin(_sc_wsn(pat), _sc_wsn(src))
_sc_occ(pat::Regex, src) = occursin(pat, replace(src, r"\s+" => " "))  # collapse ws so wrapped calls still match

using Test

const MAGNETIC_BC_STATIC_ROOT = normpath(joinpath(@__DIR__, ".."))

function _magnetic_bc_static_source(path_parts...)
    return read(joinpath(MAGNETIC_BC_STATIC_ROOT, path_parts...), String)
end

function _magnetic_bc_static_function_body(source::String, signature::String)
    start = findfirst(signature, source)
    start === nothing && error("Could not find function signature: $signature")
    next_function = findnext("\nfunction ", source, last(start) + 1)
    return next_function === nothing ? source[first(start):end] :
           source[first(start):(first(next_function) - 1)]
end

@testset "Magnetic boundary-condition static contract" begin
    magnetic_bc = _magnetic_bc_static_source("src", "bcs", "magnetic_bc.jl")
    magnetic_solver = _magnetic_bc_static_source("src", "physics", "magnetic", "solver.jl")
    erk2 = join([_magnetic_bc_static_source("src", "timestep", "erk2", f) for f in
                 ("common.jl", "boundary.jl", "cache.jl", "influence.jl", "integrate.jl")], "\n")

    tor_matrices = _magnetic_bc_static_function_body(
        magnetic_bc,
        "function create_magnetic_toroidal_matrices("
    )
    pol_matrices = _magnetic_bc_static_function_body(
        magnetic_bc,
        "function create_magnetic_poloidal_matrices("
    )
    @test _sc_occ("system_data[bw + 1, 1] = one(T)", tor_matrices)
    @test _sc_occ("system_data[bw + 1, N] = one(T)", tor_matrices)
    @test _sc_occ("system_data[bw + 1, 1] -= T(l * domain.r[1, 3])", pol_matrices)
    @test _sc_occ("system_data[bw + 1, N] += T((l + 1) * domain.r[N, 3])", pol_matrices)

    legacy_magnetic_solve = _magnetic_bc_static_function_body(
        magnetic_bc,
        "function solve_magnetic_implicit_step!("
    )
    toroidal_inner_bc = _magnetic_bc_static_function_body(
        magnetic_solver,
        "function _magnetic_toroidal_inner_bc_increment("
    )
    @test _sc_occ("local_spectral_mode_indices(solution.config)", legacy_magnetic_solve)
    @test _sc_occ("local_spectral_mode_indices(magnetic.toroidal.config)", toroidal_inner_bc)
    @test !_sc_occ("get_local_range(solution.pencil, 1)", legacy_magnetic_solve)
    @test !_sc_occ("local_range(magnetic.toroidal.pencil, 1)", toroidal_inner_bc)

    insulating_inner = _magnetic_bc_static_function_body(
        erk2,
        "function solver_create_insulating_inner_bc("
    )
    insulating_outer = _magnetic_bc_static_function_body(
        erk2,
        "function solver_create_insulating_outer_bc("
    )
    @test _sc_occ("-one(T)", insulating_inner)
    @test _sc_occ("one(T)", insulating_outer)
    @test _sc_occ("r_inv", insulating_outer)

    magnetic_tor_update = _magnetic_bc_static_function_body(
        magnetic_solver,
        "function apply_magnetic_toroidal_implicit_update!("
    )
    magnetic_pol_update = _magnetic_bc_static_function_body(
        magnetic_solver,
        "function apply_magnetic_poloidal_implicit_update!("
    )
    @test _sc_occ("build_solver_erk2_magnetic_tor_bc", magnetic_tor_update)
    @test _sc_occ("build_solver_erk2_magnetic_pol_bc", magnetic_pol_update)
    @test _sc_occ("bc_spec=bc_spec", magnetic_tor_update)
    @test _sc_occ("bc_spec=bc_spec", magnetic_pol_update)
    @test _sc_occ("_timestepper_krylov_dimension(timestepper, state.parameters)", magnetic_tor_update)
    @test _sc_occ("_timestepper_krylov_dimension(timestepper, state.parameters)", magnetic_pol_update)
    @test !_sc_occ("_timestepper_krylov_dimension(state.parameters.timestepper)", magnetic_tor_update)
    @test !_sc_occ("_timestepper_krylov_dimension(state.parameters.timestepper)", magnetic_pol_update)

    integrate_erk2 = _magnetic_bc_static_function_body(
        erk2,
        "function integrate_solver_erk2_step!("
    )
    @test _sc_occ("build_solver_erk2_magnetic_tor_bc", integrate_erk2)
    @test _sc_occ("build_solver_erk2_magnetic_pol_bc", integrate_erk2)
    @test _sc_occ("bc_spec=mag_tor_bc", integrate_erk2)
    @test _sc_occ("bc_spec=mag_pol_bc", integrate_erk2)
end
