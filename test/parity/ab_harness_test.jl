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

"""
    _count_test_failures(ts) -> Int

Recursively count `Test.Fail`/`Test.Error` results inside a `Test.AbstractTestSet`,
descending into nested testsets. `assert_ab_parity` nests one `@testset` per case,
so a probe testset's direct `.results` holds child testsets, not bare `Test.Fail`
entries — a non-recursive count would silently read 0 regardless of what happened
inside.
"""
function _count_test_failures(ts::Test.AbstractTestSet)
    n = 0
    for r in ts.results
        if r isa Test.Fail || r isa Test.Error
            n += 1
        elseif r isa Test.AbstractTestSet
            n += _count_test_failures(r)
        end
    end
    return n
end

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

    @testset "assert_ab_parity itself fails on divergence" begin
        # compare_ab is the non-asserting form; the test above never exercises the
        # `@test r.ok` line inside assert_ab_parity that actually converts a digest
        # mismatch into a test failure. A regression there (e.g. `@test r.ok`
        # silently weakened to `@test true`) would leave every other test in this
        # file green. Capture assert_ab_parity's own testset output in a probe
        # testset and assert it recorded at least one failure.
        build_a(case) = ParityFixtures.build_state(case; seed = 11)
        build_b(case) = ParityFixtures.build_state(case; seed = 12)

        ts = Test.DefaultTestSet("probe")
        Test.push_testset(ts)
        try
            ParityAB.assert_ab_parity(build_a, build_b; cases = cases)
        finally
            Test.pop_testset()
        end

        nfail = _count_test_failures(ts)
        @test nfail > 0
    end

    @testset "a custom step function is actually used, not silently ignored" begin
        # Mechanism B's brief promises "two constructor functions and two step
        # functions". Both `legacy_step` and `new_step` default to
        # ParityFixtures.evolve!, so a regression could silently keep calling
        # the default on both sides regardless of what a caller passes.
        build(case) = ParityFixtures.build_state(case; seed = 11)

        @testset "each side's custom step is invoked once per case" begin
            legacy_calls = Ref(0)
            new_calls = Ref(0)
            legacy_step(state; nsteps) = (legacy_calls[] += 1; ParityFixtures.evolve!(state; nsteps = nsteps))
            new_step(state; nsteps) = (new_calls[] += 1; ParityFixtures.evolve!(state; nsteps = nsteps))
            ParityAB.assert_ab_parity(build, build; cases = cases,
                legacy_step = legacy_step, new_step = new_step)
            @test legacy_calls[] == length(cases)
            @test new_calls[] == length(cases)
        end

        @testset "a custom new_step that skips evolution is caught, not defaulted away" begin
            # If legacy_step/new_step were silently ignored in favor of the
            # ParityFixtures.evolve! default, both sides would evolve
            # identically from the same seed and this would wrongly pass.
            noop_step(state; nsteps) = state
            results = ParityAB.compare_ab(build, build; cases = cases,
                legacy_step = ParityFixtures.evolve!, new_step = noop_step)
            @test all(r -> !r.ok, results)
            # Confirms it is the step counter/clock that diverged (thanks to
            # the state.time/state.step check), not merely "some field or
            # other" — direct evidence new_step actually ran instead of
            # evolve!.
            @test all(r -> occursin("state.step differs", r.message) ||
                    occursin("state.time differs", r.message), results)
        end
    end

    @testset "empty cases is rejected, not silently green" begin
        # A caller-supplied `cases` filtered down to nothing must not report a
        # green, zero-comparison testset — that "proves" parity while checking
        # nothing. Both entry points must guard against it explicitly.
        build(case) = ParityFixtures.build_state(case; seed = 11)
        empty_cases = ParityFixtures.ParityCase[]
        @test_throws Exception ParityAB.compare_ab(build, build; cases = empty_cases)
        @test_throws Exception ParityAB.assert_ab_parity(build, build; cases = empty_cases)
    end
end
