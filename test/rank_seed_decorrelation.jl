# Regression test: random initial conditions must not be correlated across MPI
# ranks. Each rank fills a DISJOINT subset of spectral modes by stepping a shared
# rand() stream, so a common seed gave the k-th owned mode the SAME random value
# on every rank (spurious cross-rank correlation). `rank_seed` offsets the seed
# per rank to decorrelate, while leaving rank 0 (single-rank runs) unchanged.

using Test
using Random

@testset "rank_seed per-rank decorrelation" begin
    # nothing passes through (caller skips seeding -> per-process default RNG)
    @test GeoDynamo.rank_seed(nothing, 0) === nothing
    @test GeoDynamo.rank_seed(nothing, 3) === nothing

    # rank 0 (single-rank) preserves the seed exactly -> no behavior change
    @test GeoDynamo.rank_seed(42, 0) == 42

    # distinct ranks get distinct effective seeds
    @test GeoDynamo.rank_seed(42, 0) != GeoDynamo.rank_seed(42, 1)
    @test GeoDynamo.rank_seed(42, 1) != GeoDynamo.rank_seed(42, 2)

    # behavioral: the per-rank streams actually differ
    Random.seed!(GeoDynamo.rank_seed(42, 0)); a = rand()
    Random.seed!(GeoDynamo.rank_seed(42, 1)); b = rand()
    @test a != b
end
