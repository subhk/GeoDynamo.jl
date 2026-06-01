using Test, GeoDynamo
@testset "GEODYNAMO_PROC_GRID parsing" begin
    @test GeoDynamo.parse_proc_grid("4x2", 8) == (4, 2)        # (θ_ranks, r_ranks)
    @test GeoDynamo.parse_proc_grid("8x1", 8) == (8, 1)
    @test_throws ErrorException GeoDynamo.parse_proc_grid("4x2", 6)   # product != nprocs
    @test GeoDynamo.parse_proc_grid(nothing, 1) == (1, 1)            # np==1 trivial
    @test_throws ErrorException GeoDynamo.parse_proc_grid(nothing, 4) # np>1 requires the var
end
