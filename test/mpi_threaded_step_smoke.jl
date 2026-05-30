using Test
using MPI

const FINALIZE_MPI_THREADED_SMOKE = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") ==
                                    "true"

# Regression guard for the multi-rank + multi-thread CNAB2 deadlock.
#
# Before the fix, `_solver_can_thread_implicit_updates` allowed threaded field
# updates at any rank count. With ≥2 MPI ranks each running ≥2 Julia threads, the
# `@spawn`'d field solves reached MPI collectives in nondeterministic per-rank
# order, mismatched, and deadlocked (observed: hours of busy-wait). The fix
# restricts threaded field updates to single-rank runs.
#
# This test only bites under `mpiexec -n≥2 … julia -t≥2` (BOTH are required —
# the deadlock needs concurrent threads issuing collectives across ranks). Run it
# MPIEXEC_TIMEOUT-bounded via `test/run_mpi_threaded_smoke.sh`: if the guard
# regresses, the first multi-rank threaded step hangs, MPIEXEC_TIMEOUT kills the
# job, and CI sees a non-zero exit. Single-rank or single-thread runs skip.
@testset "multi-rank threaded step does not deadlock" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping multi-rank threaded smoke test"
        return
    end
    MPI.Initialized() || MPI.Init()

    nprocs = GeoDynamo.get_nprocs()
    nthreads = Threads.nthreads()

    if nprocs < 2 || nthreads < 2
        @test_skip "needs ≥2 MPI ranks AND ≥2 Julia threads (have nprocs=$nprocs, nthreads=$nthreads)"
    else
        # Small, gently forced state so a few steps stay finite (this guards the
        # threading deadlock, not physics). Magnetic + composition on so all six
        # @spawn'd field updates run.
        params = GeoDynamo.SolverParameters(
            nr = 16, lmax = 8, mmax = 8, nlat = 16, nlon = 32,
            timestep = 1e-6, timestepper = GeoDynamo.CNAB2(),
            Ra = 1e2, RaC = 1e2,
            include_magnetic = true,
            include_composition = true
        )
        state = GeoDynamo.initialize_simulation(params)
        GeoDynamo.initialize_solver_fields!(state)

        # The real entry point. If the gate regresses, the first multi-rank
        # threaded implicit dispatch deadlocks here and MPIEXEC_TIMEOUT kills the
        # job. Reaching the asserts means every rank completed the step in sync.
        for _ in 1:3
            GeoDynamo.advance_solver_step!(state)
            MPI.Barrier(GeoDynamo.get_comm())
        end

        @test isfinite(state.time)
        @test state.step >= 3
    end
end
