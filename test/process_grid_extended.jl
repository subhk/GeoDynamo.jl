using Test

@testset "parse_proc_grid edge cases" begin
    @test GeoDynamo.parse_proc_grid("4x2", 8) == (4, 2)
    @test GeoDynamo.parse_proc_grid("1x4", 4) == (1, 4)
    @test GeoDynamo.parse_proc_grid(nothing, 1) == (1, 1)
    @test_throws ErrorException GeoDynamo.parse_proc_grid("4x2", 6)    # product != nprocs
    @test_throws ErrorException GeoDynamo.parse_proc_grid(nothing, 4)  # unset spec, np>1
    @test_throws ErrorException GeoDynamo.parse_proc_grid("4", 4)      # malformed (no 'x')
    @test_throws Exception GeoDynamo.parse_proc_grid("axb", 4)         # malformed (non-numeric -> ArgumentError)
end

@testset "read_proc_grid env handling" begin
    old = get(ENV, "GEODYNAMO_PROC_GRID", nothing)
    try
        delete!(ENV, "GEODYNAMO_PROC_GRID")
        @test GeoDynamo.read_proc_grid(1) == (1, 1)
        ENV["GEODYNAMO_PROC_GRID"] = "2x2"
        @test GeoDynamo.read_proc_grid(4) == (2, 2)
    finally
        old === nothing ? delete!(ENV, "GEODYNAMO_PROC_GRID") : (ENV["GEODYNAMO_PROC_GRID"] = old)
    end
end
