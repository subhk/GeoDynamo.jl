using Test

const PRECOMPILE_SYNTAX_REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
precompile_source_path(parts...) = joinpath(PRECOMPILE_SYNTAX_REPO_ROOT, parts...)

@testset "Precompile syntax" begin
    erk2_source = read(precompile_source_path("src", "timestep", "erk2.jl"), String)
    netcdf_demo_source = read(precompile_source_path("examples", "netcdf_boundary_demo.jl"), String)
    hybrid_demo_source = read(precompile_source_path("examples", "hybrid_boundary_demo.jl"), String)

    @test !occursin(r"(?m)^const GeoDynamo\.", erk2_source)
    @test !occursin("Base.include(@__MODULE__", netcdf_demo_source)
    @test !occursin("Base.include(@__MODULE__", hybrid_demo_source)
end
