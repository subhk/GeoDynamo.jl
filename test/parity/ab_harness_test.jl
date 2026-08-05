using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

include(joinpath(@__DIR__, "state_digest.jl"))
include(joinpath(@__DIR__, "fixtures.jl"))
include(joinpath(@__DIR__, "ab_harness.jl"))
using .ParityDigest
using .ParityFixtures
using .ParityAB

@testset "A/B harness" begin
    cases = ParityFixtures.PARITY_MATRIX_DEFAULT[1:2]

    @testset "identical builders agree" begin
        # Both sides build the same way, so this must pass. This is the shape
        # every clean-break sub-project will use: two builders, one assertion.
        build(case) = ParityFixtures.build_state(case; seed = 11)
        ParityAB.assert_ab_parity(build, build; cases = cases)
    end

    @testset "a divergent builder is caught" begin
        build_a(case) = ParityFixtures.build_state(case; seed = 11)
        build_b(case) = ParityFixtures.build_state(case; seed = 12)
        results = ParityAB.compare_ab(build_a, build_b; cases = cases)
        @test all(r -> !r.ok, results)
        @test all(r -> !isempty(r.message), results)
    end
end
