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
        # Per-thread scratch must NOT be captured either. VelocityWorkspace is
        # sized from Threads.nthreads(), so capturing it makes both the count
        # below and the digest itself thread-dependent. Re-including it would
        # reintroduce a spurious-difference path at nthreads > 1.
        @test !any(n -> occursin("velocity_workspace", n), names)

        # Exact field count for this canonical case (CNAB2/scalar1/wall1/mag/comp),
        # AFTER build_state's initialize_solver_fields!+perturb and evolve!'s
        # 4 solver_step! calls. A change in this number means the walk
        # changed shape and must be explained, not silently updated: the
        # walker can silently drop fields it classifies as known-skipped
        # leaves, and an exact count is the cheapest tripwire for that
        # regressing unnoticed.
        #
        # 176 is THREAD-INDEPENDENT — verified at 1, 2 and 4 threads on both
        # Julia 1.11.1 and 1.12.6. It was briefly 191, which held only at one
        # thread: the walk then included fields.velocity.velocity_workspace,
        # 15 per-thread scratch buffers that grew to 60 entries at 2 threads
        # and 120 at 4, so CI (which does not run single-threaded) measured 206
        # and this assertion failed. velocity_workspace is now in SKIP_FIELDS —
        # see the rationale there. If this number moves again, find out WHY
        # before touching it.
        # 176 -> 178 on 2026-08-18: `SHTnsTemperatureField` and
        # `SHTnsCompositionField` each gained a `boundary_values_imag` row, so that
        # `get_bc_vectors` can hand the scalar solve the `bc_*_imag` vectors it
        # already accepts (the imaginary half of every m > 0 scalar boundary
        # correction previously had nowhere to go). Two new leaves, one per field.
        @test length(names) == 178
    end

    @testset "matrices are well formed" begin
        @test length(ParityFixtures.PARITY_MATRIX_FULL) == 192
        d = ParityFixtures.PARITY_MATRIX_DEFAULT
        @test 8 <= length(d) <= 24
        # Every level of every factor appears at least once. Necessary but NOT
        # sufficient for the pairwise claim below — marginal coverage like
        # this is exactly what let a prior 12-case version of
        # PARITY_MATRIX_DEFAULT silently miss the scalar_code x wall_code
        # anti-diagonal (1/4, 2/3, 3/2, 4/1) while still passing every one of
        # these five assertions.
        @test sort(unique(c.timestepper_name for c in d)) == ["CNAB2", "ERK2", "RK3"]
        @test sort(unique(c.scalar_code for c in d)) == [1, 2, 3, 4]
        @test sort(unique(c.wall_code for c in d)) == [1, 2, 3, 4]
        @test sort(unique(c.magnetic for c in d)) == [false, true]
        @test sort(unique(c.composition for c in d)) == [false, true]

        # Real pairwise coverage: for EVERY one of the 10 unordered factor
        # pairs, every combination of their levels must appear at least once
        # in PARITY_MATRIX_DEFAULT. This is the actual covering-array
        # property fixtures.jl's PARITY_MATRIX_DEFAULT docstring claims, and
        # is what the marginal checks above cannot catch.
        factor_levels = (
            timestepper_name = ["CNAB2", "ERK2", "RK3"],
            scalar_code = [1, 2, 3, 4],
            wall_code = [1, 2, 3, 4],
            magnetic = [false, true],
            composition = [false, true],
        )
        fnames = collect(keys(factor_levels))
        for i in 1:length(fnames), j in (i + 1):length(fnames)
            fi, fj = fnames[i], fnames[j]
            observed = Set((getfield(c, fi), getfield(c, fj)) for c in d)
            expected = Set(
                (li, lj) for li in factor_levels[fi], lj in factor_levels[fj])
            @test expected ⊆ observed
        end
    end

    @testset "every default-matrix case digests without throwing" begin
        # Digesting index 1 alone is not sufficient coverage for a harness
        # whose whole job is to digest arbitrary configurations: it was
        # previously pinned to a CNAB2 case, so no test here ever exercised
        # digest_state on an ERK2 or RK3 case, and digest_state crashed for
        # BOTH of them (SolverParameters.timestepper reachable at
        # fields.velocity.parameters.timestepper, and
        # ExponentialRungeKutta2/RungeKutta3 are fieldless marker structs
        # the walker could not classify) — two-thirds of this very matrix.
        # Fixed by skipping :parameters entirely (state_digest.jl) rather
        # than allow-listing the timestepper marker types, since
        # SolverParameters is configuration, not evolved state. Build,
        # evolve, and digest EVERY case here so this class of gap cannot
        # hide behind a single index again. Slow (16 builds x 4 steps) is
        # accepted — that is the point.
        for case in ParityFixtures.PARITY_MATRIX_DEFAULT
            st = ParityFixtures.evolve!(ParityFixtures.build_state(case))
            digest = ParityDigest.digest_state(st)
            @test !isempty(digest.fields)
        end
    end
end
