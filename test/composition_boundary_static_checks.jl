# Whitespace-insensitive source matching: SciML auto-formatting (spacing,
# line wraps) must not break these static source-contract checks.
_sc_wsn(s) = replace(s, r"\s+" => "")
_sc_occ(pat::AbstractString, src) = occursin(_sc_wsn(pat), _sc_wsn(src))
_sc_occ(pat::Regex, src) = occursin(pat, replace(src, r"\s+" => " "))  # collapse ws so wrapped calls still match

using Test

const COMPOSITION_BC_STATIC_ROOT = normpath(joinpath(@__DIR__, ".."))

function _composition_bc_static_source(path_parts...)
    return read(joinpath(COMPOSITION_BC_STATIC_ROOT, path_parts...), String)
end

function _composition_bc_static_function_body(source::String, signature::String)
    start = findfirst(signature, source)
    start === nothing && error("Could not find function signature: $signature")
    next_function = findnext("\nfunction ", source, last(start) + 1)
    return next_function === nothing ? source[first(start):end] :
           source[first(start):(first(next_function) - 1)]
end

@testset "Composition boundary-condition static contract" begin
    api = _composition_bc_static_source("src", "api", "boundary_conditions.jl")
    scalar_bc = _composition_bc_static_source("src", "bcs", "scalar_bc.jl")
    composition_bc = _composition_bc_static_source("src", "bcs", "compositional_bc.jl")
    composition_solver = _composition_bc_static_source("src", "physics", "composition", "solver.jl")
    backend = _composition_bc_static_source("src", "solver", "backend.jl")
    numerics = _composition_bc_static_source("src", "solver", "numerics.jl")
    imex = _composition_bc_static_source("src", "timestep", "imex.jl")
    erk2 = join([_composition_bc_static_source("src", "timestep", "erk2", f) for f in
                 ("common.jl", "boundary.jl", "cache.jl", "influence.jl", "integrate.jl")], "\n")

    @test _sc_occ("_composition_bc_code(bc) = _thermal_bc_code(bc)", api)

    composition_matrices = _composition_bc_static_function_body(
        composition_bc,
        "function create_composition_matrices("
    )
    @test _sc_occ("create_scalar_matrices(", composition_matrices)
    @test _sc_occ("scalar_bc_code=composition_bc_code", composition_matrices)

    scalar_rows = _composition_bc_static_function_body(
        scalar_bc,
        "function _apply_scalar_boundary_rows!("
    )
    @test _sc_occ("system_data[bw + 1, 1] = one(T)", scalar_rows)
    @test _sc_occ("system_data[bw + 1, N] = one(T)", scalar_rows)
    @test _sc_occ("= d1_data", scalar_rows)
    # Double-Neumann (code 4) must NOT special-case l == 0 with a Dirichlet pin:
    # the time-stepping operator is non-singular under pure-Neumann rows, and the
    # former pin imposed the inner FLUX datum as a prescribed VALUE on the mean
    # mode. The special-case was removed; assert it stays removed.
    @test !_sc_occ("scalar_bc_code == 4 && l == 0", scalar_rows)

    legacy_composition_solve = _composition_bc_static_function_body(
        composition_bc,
        "function solve_composition_implicit_step!("
    )
    @test _sc_occ("Union{AbstractVector{T},Nothing}", legacy_composition_solve)
    @test _sc_occ("solve_scalar_implicit_step!(", legacy_composition_solve)
    @test !_sc_occ("get_local_range(solution.pencil, 1)", legacy_composition_solve)

    shared_scalar_solve = _composition_bc_static_function_body(
        scalar_bc,
        "function solve_scalar_implicit_step!("
    )
    @test _sc_occ("local_spectral_mode_indices(solution.config)", shared_scalar_solve)

    get_bc_vectors = _composition_bc_static_function_body(
        numerics,
        "function get_bc_vectors(field)"
    )
    # The live-array choice (field rows vs a loaded spectral BC cache) is delegated to
    # `bcs.active_boundary_arrays`, the single place both this reader and the
    # topography couplings' writer consult. See temperature_boundary_static_checks.jl.
    @test _sc_occ("bcs.active_boundary_arrays(field)", get_bc_vectors)
    @test _sc_occ("view(bc_real, 1, :)", get_bc_vectors)
    # Uniform, type-stable `_BCVectors` return shape (absent slots positional `nothing`).
    @test _sc_occ("_BCVectors(", get_bc_vectors)

    runtime_create = _composition_bc_static_function_body(
        backend,
        "function create_solver_runtime("
    )
    @test _sc_occ("apply_scalar_boundary_parameters!(composition", runtime_create)

    scalar_solve = _composition_bc_static_function_body(
        imex,
        "function _solver_solve_scalar_implicit_step!("
    )
    @test _sc_occ("Union{AbstractVector{T}, Nothing}", scalar_solve)
    @test _sc_occ("local_spectral_mode_indices(solution.config)", scalar_solve)

    composition_update = _composition_bc_static_function_body(
        composition_solver,
        "function apply_composition_implicit_update!("
    )
    # apply_composition_implicit_update! is now a thin shim over the shared
    # _apply_scalar_implicit_update! (src/physics/scalar_field_solver_common.jl).
    # The EAB2 boundary-spec wiring is guarded once in
    # scalar_boundary_shared_static_checks.jl; here we pin the nil guard plus
    # delegation with the composition-specific arguments.
    @test _sc_occ("composition === nothing && return state", composition_update)
    @test _sc_occ("_apply_scalar_implicit_update!(", composition_update)
    @test _sc_occ(":composition", composition_update)
    @test _sc_occ("state.parameters.Pm / state.parameters.Sc", composition_update)
    @test _sc_occ("_composition_bc_code(state.parameters.composition_bcs)", composition_update)
    @test _sc_occ("solver_solve_composition_implicit_step!", composition_update)
    @test _sc_occ("state.timestep_caches.etd_composition", composition_update)

    integrate_erk2 = _composition_bc_static_function_body(
        erk2,
        "function integrate_solver_erk2_step!("
    )
    @test _sc_occ("comp_bc_values = get_bc_vectors(state.fields.composition)", integrate_erk2)
    @test _sc_occ("comp_bc = with_boundary_mode_values(", integrate_erk2)
end
