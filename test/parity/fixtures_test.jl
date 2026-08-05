using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

include(joinpath(@__DIR__, "state_digest.jl"))
include(joinpath(@__DIR__, "fixtures.jl"))
using .ParityDigest
using .ParityFixtures

@testset "parity fixtures" begin
    case = ParityFixtures.PARITY_MATRIX_DEFAULT[1]

    @testset "perturbation survives to the evolved state" begin
        # This is the whole point of the fixture. solver_step! regenerates the IC
        # on its first call unless is_initialized is already true
        # (src/solver/mainloop.jl:92), so a naive perturb-then-step silently
        # discards the seed. Two different seeds MUST produce different states.
        a = ParityFixtures.evolve!(ParityFixtures.build_state(case; seed = 11))
        b = ParityFixtures.evolve!(ParityFixtures.build_state(case; seed = 12))
        ok, _ = ParityDigest.digests_equal(
            ParityDigest.digest_state(a), ParityDigest.digest_state(b))
        @test !ok
    end

    @testset "same seed is reproducible bit-for-bit" begin
        a = ParityFixtures.evolve!(ParityFixtures.build_state(case; seed = 11))
        b = ParityFixtures.evolve!(ParityFixtures.build_state(case; seed = 11))
        ok, msg = ParityDigest.digests_equal(
            ParityDigest.digest_state(a), ParityDigest.digest_state(b))
        @test ok
        @test isempty(msg)
    end

    @testset "digest captures the fields that matter" begin
        st = ParityFixtures.evolve!(ParityFixtures.build_state(
            ParityFixtures.ParityCase("CNAB2", GeoDynamo.CNAB2(), 1, 1, true, true)))
        names = [f.name for f in ParityDigest.digest_state(st).fields]
        @test any(n -> occursin("velocity.toroidal.data_real", n), names)
        @test any(n -> occursin("velocity.poloidal.data_real", n), names)
        @test any(n -> occursin("temperature.spectral.data_real", n), names)
        @test any(n -> occursin("magnetic.toroidal.data_real", n), names)
        @test any(n -> occursin("composition.spectral.data_real", n), names)
        # CNAB2 history must be captured; corrupting it stays invisible for one step.
        @test any(n -> occursin("prev_nl_toroidal", n), names)
        # Wall-clock timers must NOT be captured or every run fails spuriously.
        @test !any(n -> occursin("computation_time", n), names)
        @test !any(n -> occursin("transform_time", n), names)

        # Exact field count for this canonical case (CNAB2/scalar1/wall1/mag/comp),
        # AFTER build_state's initialize_solver_fields!+perturb and evolve!'s
        # 4 solver_step! calls. A change in this number means the walk
        # changed shape and must be explained, not silently updated: the
        # walker can silently drop fields it classifies as known-skipped
        # leaves, and an exact count is the cheapest tripwire for that
        # regressing unnoticed. Measured stable at 191 across two
        # independent runs of this exact case (see task-2-report.md). This
        # is NOT the 176 Task 1 measured on a built-but-not-stepped state —
        # a stepped state additionally reaches lazily-built scratch such as
        # VelocityWorkspace (fields.velocity.velocity_workspace), which the
        # walker now recurses into element-by-element instead of skipping.
        @test length(names) == 191
    end

    @testset "matrices are well formed" begin
        @test length(ParityFixtures.PARITY_MATRIX_FULL) == 192
        d = ParityFixtures.PARITY_MATRIX_DEFAULT
        @test 8 <= length(d) <= 24
        # every level of every factor appears at least once
        @test sort(unique(c.timestepper_name for c in d)) == ["CNAB2", "ERK2", "RK3"]
        @test sort(unique(c.scalar_code for c in d)) == [1, 2, 3, 4]
        @test sort(unique(c.wall_code for c in d)) == [1, 2, 3, 4]
        @test sort(unique(c.magnetic for c in d)) == [false, true]
        @test sort(unique(c.composition for c in d)) == [false, true]
    end
end
