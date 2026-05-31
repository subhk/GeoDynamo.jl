using Test, GeoDynamo, MPI, PencilArrays
# Owns MPI lifecycle only if it had to initialize (so later appended @testsets and
# the suite runner don't double-init / prematurely finalize). Mirrors the
# FINALIZE_* guard pattern in test/mpi_parallel_invariants.jl.
const FINALIZE_MPI_THETA_DIST = !MPI.Initialized()
FINALIZE_MPI_THETA_DIST && MPI.Init()

@testset "1D-theta layout + prototype pencil" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=4)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    rloc = PencilArrays.range_local(cfg.pencils.r)
    @test length(rloc[2]) == 20          # φ FULLY LOCAL (1:nlon)
    @test length(rloc[3]) == 4           # r LOCAL (1:nr)
    # θ is distributed: under np>1 the local θ-slab is a strict subset of 1:nlat.
    # (On np=1 it is the full 1:12, so the split-match below is trivially true —
    #  the non-trivial multi-rank check runs under `mpiexec -n 2+` in Task 6.)
    @test nprocs == 1 || length(rloc[1]) < 12
    p2 = cfg.pencils.theta_phys
    p2loc = PencilArrays.range_local(p2)
    @test length(p2loc[2]) == 20         # φ full on the 2D prototype
    @test p2loc[1] == rloc[1]            # θ-split MATCHES pencils.r (critical)
end

# ============================================================================
# Tasks 3/5 (scalar + vector transform roundtrips): APPEND new @testset blocks
# ABOVE this line — the MPI.Finalize() below must remain the last statement.
# ============================================================================

FINALIZE_MPI_THETA_DIST && MPI.Finalize()
