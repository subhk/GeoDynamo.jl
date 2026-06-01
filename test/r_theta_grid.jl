using Test, GeoDynamo
@testset "GEODYNAMO_PROC_GRID parsing" begin
    @test GeoDynamo.parse_proc_grid("4x2", 8) == (4, 2)        # (θ_ranks, r_ranks)
    @test GeoDynamo.parse_proc_grid("8x1", 8) == (8, 1)
    @test_throws ErrorException GeoDynamo.parse_proc_grid("4x2", 6)   # product != nprocs
    @test GeoDynamo.parse_proc_grid(nothing, 1) == (1, 1)            # np==1 trivial
    @test_throws ErrorException GeoDynamo.parse_proc_grid(nothing, 4) # np>1 requires the var
end

using MPI, PencilArrays
MPI.Initialized() || MPI.Init()
@testset "2D r×θ config (single rank)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=6)
    rloc = PencilArrays.range_local(cfg.pencils.r)
    @test length(rloc[2]) == 20                 # φ local (full)
    @test :theta_phys in keys(cfg.pencils)      # prototype present
    @test cfg.pencils.θ_comm !== nothing        # θ-subcomm exposed
    @test cfg.pencils.r_comm !== nothing        # r-subcomm exposed
end
