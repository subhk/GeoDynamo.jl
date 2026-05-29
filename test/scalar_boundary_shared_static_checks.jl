using Test

const SCALAR_BC_SHARED_ROOT = normpath(joinpath(@__DIR__, ".."))

function _scalar_bc_shared_source(path_parts...)
    return read(joinpath(SCALAR_BC_SHARED_ROOT, path_parts...), String)
end

function _scalar_bc_shared_function_body(source::String, signature::String)
    start = findfirst(signature, source)
    start === nothing && error("Could not find function signature: $signature")
    next_function = findnext("\nfunction ", source, last(start) + 1)
    return next_function === nothing ? source[first(start):end] : source[first(start):first(next_function)-1]
end

@testset "Shared scalar boundary-condition implementation" begin
    geodynamo = _scalar_bc_shared_source("src", "GeoDynamo.jl")
    scalar_bc_path = joinpath(SCALAR_BC_SHARED_ROOT, "src", "bcs", "scalar_bc.jl")
    @test isfile(scalar_bc_path)
    scalar_bc = isfile(scalar_bc_path) ? read(scalar_bc_path, String) : ""
    thermal_bc = _scalar_bc_shared_source("src", "bcs", "thermal_bc.jl")
    composition_bc = _scalar_bc_shared_source("src", "bcs", "compositional_bc.jl")

    @test occursin("include(\"bcs/scalar_bc.jl\")", geodynamo)
    @test occursin("function create_scalar_matrices(", scalar_bc)
    @test occursin("function solve_scalar_implicit_step!(", scalar_bc)
    @test occursin("function set_scalar_rhs_bc!(", scalar_bc)

    temperature_matrices = _scalar_bc_shared_function_body(
        thermal_bc,
        "function create_temperature_matrices(",
    )
    composition_matrices = _scalar_bc_shared_function_body(
        composition_bc,
        "function create_composition_matrices(",
    )
    @test occursin("create_scalar_matrices(", temperature_matrices)
    @test occursin("create_scalar_matrices(", composition_matrices)
    @test !occursin("unique_l = unique(config.l_values)", temperature_matrices)
    @test !occursin("unique_l = unique(config.l_values)", composition_matrices)

    temperature_solve = _scalar_bc_shared_function_body(
        thermal_bc,
        "function solve_temperature_implicit_step!(",
    )
    composition_solve = _scalar_bc_shared_function_body(
        composition_bc,
        "function solve_composition_implicit_step!(",
    )
    @test occursin("solve_scalar_implicit_step!(", temperature_solve)
    @test occursin("solve_scalar_implicit_step!(", composition_solve)
    @test !occursin("Vector{T}(undef, nr)", temperature_solve)
    @test !occursin("Vector{T}(undef, nr)", composition_solve)
end
