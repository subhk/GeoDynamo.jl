# ================================================================================
# Multi-rank CONTROL-PLANE invariants
# ================================================================================
#
# The existing MPI gates cover the NUMERICAL core very well: the r×θ
# process-grid equivalence scripts compare 4x1 / 1x4 / 2x2 against the 1x1 serial
# reference and report maxdiff = 0.0 (bit-exact) for every tensor. What none of
# them can reach is the CONTROL plane — the decisions taken *around* the step:
# stop conditions, health checks, output scheduling. Those gates run a fixed step
# count, attach no NaN, set no wall-time limit and use explicit writes.
#
# That blind spot is exactly where four separate defects lived, all the same
# shape — a decision taken from RANK-LOCAL data and then used to gate a
# COLLECTIVE, so the offending ranks left `run!` (or entered `write_fields!`)
# while the others blocked forever in the next collective:
#
#   * `nan_checker` / `HealthCheck` stopping from a rank-local field scan
#   * `wall_time_limit_exceeded` comparing a rank-local `time()`
#   * `WallTimeInterval` writers gating the collective `write_fields!`
#   * the threaded-update denylist missing poloidal CONTINUITY_MAG / topography
#
# The fixes are `_any_rank_flag` (Allreduce MAX) and `_collective_wtime`
# (Bcast of rank 0's elapsed) in api/schedules.jl, plus the collective-side guard
# in solver/numerics.jl. This file pins them where they actually matter.
#
# The failure being tested for is a HANG, not an exception, so the runner bounds
# the job with MPIEXEC_TIMEOUT: a regression kills the launcher (non-zero exit)
# instead of blocking forever. Needs >= 2 ranks to bite; at one rank the
# reductions are no-ops and the file degrades to asserting the helpers' local
# semantics, which is still a valid (if weaker) regression.
# ================================================================================

using Test
using MPI
using GeoDynamo

@testset "MPI control-plane invariants" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping control-plane invariants"
        return
    end
    MPI.Initialized() || MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
        lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
    mkmodel(; kw...) = GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4,
        include_magnetic = false, include_composition = false, kw...)

    # ── the wall clock every rank sees must be the SAME wall clock ─────────────
    @testset "_collective_wtime agrees on every rank" begin
        sim = GeoDynamo.Simulation(mkmodel(); Δt = 1e-4, stop_iteration = 1)
        # Stagger the per-rank start so a rank-LOCAL read would disagree by 4 s.
        sim._wall_start = time() - (rank == 0 ? 5.0 : 9.0)
        w = GeoDynamo._collective_wtime(sim)
        allw = MPI.Allgather(w, comm)
        @test all(==(allw[1]), allw)
        # before run! there is no start time, so it is defined as zero everywhere
        sim._wall_start = 0.0
        @test GeoDynamo._collective_wtime(sim) == 0.0
    end

    # ── a flag set on ONE rank must be seen by ALL ranks ──────────────────────
    @testset "_any_rank_flag reduces across ranks" begin
        # only the last rank sets it; every rank must observe true
        g = GeoDynamo._any_rank_flag(rank == nranks - 1)
        @test all(MPI.Allgather(g, comm))
        @test GeoDynamo._any_rank_flag(false) == false
    end

    # ── a NaN must stop every rank on the same iteration, not hang ────────────
    @testset "single-rank NaN stops all ranks together" begin
        model = mkmodel()
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, stop_iteration = 50)
        GeoDynamo.add_callback!(sim, GeoDynamo.nan_checker;
            schedule = GeoDynamo.IterationInterval(1), name = :nan_fast)
        # Initialize BEFORE injecting: a model built without ICs leaves
        # is_initialized false, so the first solver_step! would run
        # initialize_solver_fields! straight over the injected value.
        GeoDynamo.initialize_fields!(model.state)
        if rank == nranks - 1
            spec = parent(model.state.fields.temperature.spectral.data_real)
            length(spec) > 0 && (spec[1] = NaN)
        end
        MPI.Barrier(comm)
        GeoDynamo.run!(sim)
        # Reaching here on every rank is the assertion: pre-fix, only the ranks
        # that owned the NaN left run! and the rest blocked in the next collective.
        MPI.Barrier(comm)
        @test sim.running == false
        iters = MPI.Allgather(model.clock.iteration, comm)
        @test all(==(iters[1]), iters)
        @test iters[1] < 50            # stopped by the NaN, not by stop_iteration
    end

    # ── a public callback may stop from rank-local state; run! must stay collective ─
    @testset "single-rank user callback stops all ranks together" begin
        model = mkmodel()
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, stop_iteration = 50)
        GeoDynamo.add_callback!(sim,
            s -> (rank == nranks - 1 && (s.running = false));
            schedule = GeoDynamo.IterationInterval(1), name = :rank_local_stop)

        GeoDynamo.run!(sim)
        MPI.Barrier(comm)

        @test sim.running == false
        iters = MPI.Allgather(model.clock.iteration, comm)
        @test all(==(iters[1]), iters)
        @test iters[1] == 1
    end

    # ── a WallTimeInterval writer gates a COLLECTIVE; it must not desync ──────
    @testset "WallTimeInterval writer does not desync the write gate" begin
        # One shared directory, broadcast: a per-rank mktempdir() would have each
        # rank write a different path and the collective NetCDF open fails EACCES.
        dir = MPI.bcast(rank == 0 ? mktempdir() : "", 0, comm)
        model = mkmodel()
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, stop_iteration = 4,
            output_writers = (snap = GeoDynamo.FieldWriter(dir;
                schedule = GeoDynamo.WallTimeInterval(1e-9),   # fires every step
                fields = [:temperature]),))
        GeoDynamo.run!(sim)
        MPI.Barrier(comm)
        @test model.clock.iteration == 4
    end

    # ── the threaded-update collective guard must not fire on a clean config ──
    @testset "threaded update guard stays quiet on a supported config" begin
        model = GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4,
            include_magnetic = true, include_composition = false)
        # insulating, no topography ⇒ no in-kernel collective ⇒ threading allowed
        @test GeoDynamo._solver_magnetic_config_has_collective(model.state) == false
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, stop_iteration = 3)
        GeoDynamo.run!(sim)
        MPI.Barrier(comm)
        @test model.clock.iteration == 3
        @test GeoDynamo._in_threaded_implicit_update() == false
    end

    # ── the path handed to a COLLECTIVE open must be rank 0's choice ──────────
    @testset "restart file selection is rank-0 authoritative" begin
        # Deliberately give each rank a DIFFERENT directory — which is what node-local
        # scratch, or an NFS mount with a stale attribute cache, looks like from inside
        # the rank-local `readdir` in `find_restart_files`. A rank-local pick returns
        # each rank's own file and the collective NCDataset open then targets different
        # paths on different ranks: an MPI-IO hang, or two checkpoints silently mixed.
        dir = mktempdir()
        touch(joinpath(dir, "geodynamo_shell_restart_$(rank + 1).nc"))
        chosen = GeoDynamo._restart_path_for_all_ranks(dir, -1.0)
        n = parse(Int, match(r"_(\d+)\.nc$", basename(chosen)).captures[1])
        ns = MPI.Allgather(n, comm)
        @test all(==(ns[1]), ns)
        @test ns[1] == 1                      # rank 0's file, on every rank

        # a missing checkpoint must raise on EVERY rank: if only the ranks with an
        # empty listing raise, the others walk into the collective open alone
        empty_dir = mktempdir()
        raised = 0
        try
            GeoDynamo._restart_path_for_all_ranks(empty_dir, -1.0)
        catch
            raised = 1
        end
        flags = MPI.Allgather(raised, comm)
        @test all(==(1), flags)
        MPI.Barrier(comm)
    end
end
