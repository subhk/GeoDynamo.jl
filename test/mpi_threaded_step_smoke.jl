using Test
using MPI

const FINALIZE_MPI_THREADED_SMOKE = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") ==
                                    "true"

# Regression guard for the multi-rank + multi-thread CNAB2 threading gate.
#
# CNAB2 field-update kernels are radial-only banded solves that issue NO MPI
# collective on their base (single-field) path, so they are threaded at any rank
# count. The one exception is the magnetic update under a *conducting inner core*
# or a `CONTINUITY_MAG` inner boundary: those branches call an `Allreduce` inside
# BOTH the toroidal and poloidal `@spawn`'d tasks, whose per-rank ordering can
# diverge and deadlock (observed once: hours of busy-wait). The gate therefore
# threads collective-free CNAB2 multi-rank runs but keeps the conducting/
# CONTINUITY_MAG configs — and every ExponentialAdamsBashforth2 multi-rank run —
# on the sequential path.
#
# This test only bites under `mpiexec -n≥2 … julia -t≥2` (BOTH are required — the
# deadlock needs concurrent threads issuing collectives across ranks). Run it
# MPIEXEC_TIMEOUT-bounded via `test/run_mpi_threaded_smoke.sh`: if the gate
# regresses (threads a collective magnetic config, or a future kernel starts
# issuing a collective on the threaded path), the step hangs, MPIEXEC_TIMEOUT
# kills the job, and CI sees a non-zero exit. Single-rank or single-thread runs
# skip.
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
        # Small, gently forced states so a few steps stay finite (this guards the
        # threading gate, not physics). Magnetic + composition on so all six
        # @spawn'd field updates run.
        #
        # Two magnetic configs, both must complete without deadlock:
        #   :insulating           → collective-free ⇒ THREADED multi-rank path
        #   :conducting_inner_core → Allreduce in magnetic tasks ⇒ SEQUENTIAL path
        base_kwargs = (;
            nr = 16, nr_inner = 4, lmax = 8, mmax = 8, nlat = 16, nlon = 32,
            timestep = 1e-6, timestepper = GeoDynamo.CNAB2(),
            Ra = 1e2, RaC = 1e2,
            include_magnetic = true,
            include_composition = true,
        )

        for inner_bc in (:insulating, :conducting_inner_core)
            params = GeoDynamo.SolverParameters(; base_kwargs...,
                magnetic_inner_bc = inner_bc)
            state = GeoDynamo.initialize_simulation(params)
            GeoDynamo.initialize_solver_fields!(state)

            # The real entry point. If the gate regresses, the first multi-rank
            # implicit dispatch deadlocks here and MPIEXEC_TIMEOUT kills the job.
            # Reaching the asserts means every rank completed the step in sync.
            for _ in 1:3
                GeoDynamo.solver_step!(state)
                MPI.Barrier(GeoDynamo.get_comm())
            end

            @test isfinite(state.time)
            @test state.step >= 3
        end
    end
end
