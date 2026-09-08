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
# The fixes are `any_rank` (Allreduce MAX) and `_collective_wtime`
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

struct _InjectedNetCDFWriteFailure end
Base.setindex!(::_InjectedNetCDFWriteFailure, _value, _indices...) =
    error("injected root-only NetCDF payload failure")

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
    @testset "any_rank reduces across ranks" begin
        # only the last rank sets it; every rank must observe true
        g = GeoDynamo.any_rank(rank == nranks - 1)
        @test all(MPI.Allgather(g, comm))
        @test GeoDynamo.any_rank(false) == false
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
        # This is the only testset here that drives a real writer, so it needs the
        # repo's parallel-NetCDF probe: the Windows JLLs ship without MPI-IO and every
        # collective open there fails with NetCDF -114. Collective, so every rank
        # probes and every rank takes the same branch.
        probe_err = GeoDynamo.parallel_netcdf_probe(comm)
        if probe_err !== nothing
            @warn "Parallel NetCDF unavailable; skipping WallTimeInterval write gate" error = probe_err
        else
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

    # ── callback REGISTRIES must match before any callback can enter MPI ───────
    @testset "rank-local callback registration fails collectively" begin
        # Callback implementations are allowed to contain collectives
        # (EnergyDiagnostics, SolenoidalMonitor, and HealthCheck all do). A callback
        # present only on rank 0 can therefore enter an Allreduce while peers skip to
        # the next solver collective. Reject structural registry mismatches at run!
        # entry, on every rank, before any callback fires.
        if nranks == 1
            @test_skip "callback-registry asymmetry requires at least two ranks"
        else
            model = mkmodel()
            sim = GeoDynamo.Simulation(model; Δt = 1e-4, stop_iteration = 3)
            if rank == 0
                GeoDynamo.add_callback!(sim, s -> nothing;
                    schedule = GeoDynamo.IterationInterval(1), name = :rank0_only)
            end
            MPI.Barrier(comm)

            raised = 0
            message_ok = 0
            try
                GeoDynamo.run!(sim)
            catch err
                raised = 1
                message_ok = occursin(
                    "callback registry", lowercase(sprint(showerror, err))) ? 1 : 0
            end
            @test all(==(1), MPI.Allgather(raised, comm))
            @test all(==(1), MPI.Allgather(message_ok, comm))
            MPI.Barrier(comm)
        end
    end

    # ── output scan failures must also abort collectively ─────────────────────
    @testset "unreadable output directory fails on every rank" begin
        dir = MPI.bcast(rank == 0 ? mktempdir() : "", 0, comm)
        if rank == 0
            touch(joinpath(dir, "geodynamo_shell_hist_1.nc"))
            chmod(dir, 0o300)
        end
        MPI.Barrier(comm)
        unreadable = MPI.bcast(rank == 0 ? !isreadable(dir) : false, 0, comm)

        if unreadable
            raised = 0
            try
                GeoDynamo._existing_writer_count(dir, :hist, :shell)
            catch
                raised = 1
            end
            @test all(==(1), MPI.Allgather(raised, comm))
        end

        MPI.Barrier(comm)
        rank == 0 && chmod(dir, 0o700)
        MPI.Barrier(comm)
    end

    # ── a capability verdict used to gate a collective must be unanimous ───────
    @testset "parallel-NetCDF verdict is reduced, not rank-local" begin
        # `parallel_netcdf_available` is documented as the degrade-or-skip form, i.e.
        # it is meant to be a branch predicate around COLLECTIVE NetCDF writes. The
        # probe it wraps can genuinely succeed on some ranks and fail on others — its
        # `tempname()` lands on node-local `/tmp`, so in a multi-node job the ranks
        # off rank 0's node cannot see the path the collective create targets. A
        # rank-local verdict then sends one group past the write while the rest enter
        # `NCDataset(comm, ...)` and block.
        @test GeoDynamo.all_ranks(true, comm) == true
        if nranks > 1
            # At one rank the reduction is a no-op by construction and `rank == 0`
            # is simply true everywhere, so the split case needs >= 2 ranks to bite.
            split = GeoDynamo.all_ranks(rank == 0, comm)
            @test all(==(false), MPI.Allgather(split, comm))
        end

        avail = GeoDynamo.parallel_netcdf_available(comm)
        @test all(==(avail), MPI.Allgather(avail, comm))
        MPI.Barrier(comm)
    end

    # ── a restart file only SOME ranks can see must raise on all of them ───────
    @testset "restart file invisible on one rank raises on every rank" begin
        # The pre-existing guard checks `isfile` on rank 0 and broadcasts, so it
        # catches "missing everywhere". It does not catch the node-local-scratch case
        # it was written for: a rank that cannot see rank 0's pick only warns, and
        # then fails alone inside the collective `NCDataset` open while the ranks that
        # can see the file block inside it — the very hang the guard exists to stop.
        dir = mktempdir()                       # distinct per rank
        path = joinpath(dir, "geodynamo_shell_restart_1.nc")
        rank == 0 && touch(path)                # visible on rank 0 ONLY
        if nranks > 1
            # Needs >= 2 ranks: at one rank "visible on rank 0 only" is the same as
            # "visible everywhere", so there is nothing asymmetric to catch.
            raised = 0
            try
                GeoDynamo._require_restart_file_everywhere(path, comm)
            catch
                raised = 1
            end
            flags = MPI.Allgather(raised, comm)
            @test all(==(1), flags)
        end

        # a file NO rank can see must raise everywhere, at any rank count
        gone = 0
        try
            GeoDynamo._require_restart_file_everywhere(joinpath(dir, "absent.nc"), comm)
        catch
            gone = 1
        end
        @test all(==(1), MPI.Allgather(gone, comm))

        # and a file every rank can see must raise on none of them
        shared = joinpath(dir, "seen_by_all.nc")
        touch(shared)
        raised2 = 0
        try
            GeoDynamo._require_restart_file_everywhere(shared, comm)
        catch
            raised2 = 1
        end
        @test all(==(0), MPI.Allgather(raised2, comm))
        MPI.Barrier(comm)
    end

    # ── the OUTPUT-WRITER registry gates collectives exactly like the callbacks ─
    @testset "asymmetric output-writer registry aborts on every rank" begin
        # `_restore_output_writer_trackers!` enters three collectives per
        # FieldWriter/CheckpointWriter and none for the generic fallback, so
        # `output_writers = rank == 0 ? (...) : NamedTuple()` made the number of
        # collectives inside the `Simulation` constructor a function of the LOCAL
        # registry — a hang, with no error, on the restart path. Same shape as the
        # four defects this file already pins; the callback registry had a guard for
        # it and the writer registry did not.
        sched = GeoDynamo.IterationInterval(1)
        asymmetric = rank == 0 ?
                     GeoDynamo._to_ordered(
            (snap = GeoDynamo.FieldWriter("out"; schedule = sched),), :writer) :
                     GeoDynamo._to_ordered(NamedTuple(), :writer)
        raised = 0
        why = ""
        try
            GeoDynamo._freeze_output_writers!(GeoDynamo.CollectiveRegistry{:writer}(asymmetric))
        catch e
            raised = 1
            why = sprint(showerror, e)
        end
        # every rank must learn about it — including the ranks whose own registry is
        # the "normal" one, because they are the ones that would block in the scan
        @test all(==(nranks > 1 ? 1 : 0), MPI.Allgather(raised, comm))
        # ... and for the stated reason: a missing validator would also "raise" here
        nranks > 1 && @test occursin("output-writer registry mismatch", why)

        # and the registry every rank agrees on must raise on none of them
        symmetric = GeoDynamo._to_ordered(
            (snap = GeoDynamo.FieldWriter("out"; schedule = sched),), :writer)
        clean = 0
        try
            GeoDynamo._freeze_output_writers!(GeoDynamo.CollectiveRegistry{:writer}(symmetric))
        catch
            clean = 1
        end
        @test all(==(0), MPI.Allgather(clean, comm))
        MPI.Barrier(comm)
    end

    # ── ... and the callback guard has to say how to fix it ────────────────────
    @testset "asymmetric callback registry names the remedy" begin
        sim = GeoDynamo.Simulation(mkmodel(); Δt = 1e-4, stop_iteration = 1)
        if rank == 0
            GeoDynamo.add_callback!(sim, s -> nothing;
                schedule = GeoDynamo.IterationInterval(1), name = :rank0_only)
        end
        msg = ""
        try
            GeoDynamo.run!(sim)
        catch e
            msg = sprint(showerror, e)
        end
        @test all(==(nranks > 1 ? 1 : 0),
            MPI.Allgather(isempty(msg) ? 0 : 1, comm))
        if nranks > 1
            # the abort has to point at the working alternative, because the pattern
            # it rejects is one the `_stop_needs_reduce` comment used to advertise
            @test occursin("every rank", msg)
            @test occursin("inside its body", msg)
        end
        MPI.Barrier(comm)
    end

    # ── a process grid that empties an axis must ABORT, on every rank ─────────
    @testset "empty-axis process grid aborts on every rank" begin
        # 4x1 on lmax=mmax=1 gives the spectral pencil an EMPTY m-slot range on half
        # the ranks (`axes_local=(1:2, 1:0, 1:8)`), and `run!` then never returns —
        # reproduced as a 240 s launcher timeout with no error, while the same grid
        # completes at 1 and 2 ranks. `validate_proc_grid` named the condition and only
        # warned, on rank 0. It has to refuse, and refuse EVERYWHERE: a rank-0-only
        # throw leaves the other ranks blocking in the next collective.
        raised = 0
        why = ""
        try
            GeoDynamo.validate_proc_grid(4, 1; nlat = 4, nr = 8, lmax = 1, mmax = 1)
        catch e
            raised = 1
            why = sprint(showerror, e)
        end
        @test all(==(1), MPI.Allgather(raised, comm))
        @test occursin("empty local range", why)

        # ... and the grids the numerical gates run stay legal on every rank, including
        # 2x2, whose high-m/low-l corner rank owns zero MODES (but a non-empty slot
        # range) and still reproduces serial bit-exactly.
        rejected = 0
        try
            GeoDynamo.validate_proc_grid(2, 2; nlat = 10, nr = 8, lmax = 4, mmax = 4)
        catch
            rejected = 1
        end
        @test all(==(0), MPI.Allgather(rejected, comm))
        MPI.Barrier(comm)
    end

    # ── hand-stepping must apply the same registry guard as run! ──────────────
    @testset "manual time_step! validates callback and writer registries" begin
        if nranks == 1
            @test_skip "registry asymmetry requires at least two ranks"
        else
            callback_sim = GeoDynamo.Simulation(
                mkmodel(); Δt = 1e-4, stop_iteration = 10)
            if rank == 0
                # This callback is intentionally harmless and non-collective. Before
                # the guard existed, manual stepping therefore returned normally
                # instead of exposing the registry mismatch (a collective callback
                # would hang and make a poor RED regression).
                GeoDynamo.add_callback!(callback_sim, _ -> nothing;
                    schedule = GeoDynamo.IterationInterval(1000), name = :rank0_only)
            end

            callback_raised = 0
            callback_message = ""
            try
                GeoDynamo.time_step!(callback_sim)
            catch err
                callback_raised = 1
                callback_message = sprint(showerror, err)
            end
            @test all(==(1), MPI.Allgather(callback_raised, comm))
            @test all(==(1), MPI.Allgather(
                occursin("callback registry mismatch", lowercase(callback_message)) ? 1 : 0,
                comm))
            # Validation belongs before the numerical step: rejecting bad control
            # structure must not advance either the fields or the public clock.
            @test all(==(0), MPI.Allgather(callback_sim.model.clock.iteration, comm))
            MPI.Barrier(comm)

            writers = rank == 0 ?
                      (snap = GeoDynamo.FieldWriter("never-fired";
                           schedule = GeoDynamo.IterationInterval(1000),
                           fields = [:temperature]),) : NamedTuple()
            # Writers are fixed at construction, so the asymmetry is rejected by the
            # constructor itself — before any collective tracker restore can run.
            writer_raised = 0
            writer_message = ""
            try
                GeoDynamo.Simulation(mkmodel(); Δt = 1e-4,
                    stop_iteration = 10, output_writers = writers)
            catch err
                writer_raised = 1
                writer_message = sprint(showerror, err)
            end
            @test all(==(1), MPI.Allgather(writer_raised, comm))
            @test all(==(1), MPI.Allgather(
                occursin("output-writer registry mismatch", lowercase(writer_message)) ? 1 : 0,
                comm))
            MPI.Barrier(comm)
        end
    end

    # ── rank-0 filesystem errors must be broadcast before peer barriers ───────
    @testset "history filesystem failures abort on every rank" begin
        guards_present =
            isdefined(GeoDynamo, :_ensure_output_directory_collectively!) &&
            isdefined(GeoDynamo, :_write_grid_file_collectively!)
        @test guards_present

        # Do not enter the known-deadlocking pre-fix path during RED. The failed
        # existence assertion above is the RED signal; once the guards exist these
        # calls exercise their real write_fields! integration.
        if guards_present
            base = GeoDynamo.default_config(Float64)
            make_config(dir, prefix; include_grid) = GeoDynamo.OutputConfig(
                base.output_space, dir, prefix,
                base.include_metadata, include_grid, base.include_diagnostics,
                base.output_precision, base.spectral_lmax_output, base.overwrite_files,
                base.output_interval, Inf, base.max_output_time, base.time_tolerance)
            metadata = Dict{String, Any}("current_time" => 0.0, "current_step" => 0)

            shared = MPI.bcast(rank == 0 ? mktempdir() : "", 0, comm)
            blocker = joinpath(shared, "not_a_directory")
            rank == 0 && touch(blocker)
            MPI.Barrier(comm)
            bad_dir = joinpath(blocker, "history")
            dir_config = make_config(bad_dir, "dir_failure"; include_grid = false)
            dir_tracker = GeoDynamo.create_time_tracker(dir_config, 0.0)
            dir_raised = 0
            dir_message = ""
            try
                GeoDynamo.write_fields!(Dict{String, Any}(), dir_tracker, metadata,
                    dir_config)
            catch err
                dir_raised = 1
                dir_message = sprint(showerror, err)
            end
            @test all(==(1), MPI.Allgather(dir_raised, comm))
            @test all(==(1), MPI.Allgather(
                occursin("output directory", lowercase(dir_message)) ? 1 : 0, comm))
            MPI.Barrier(comm)

            grid_dir = MPI.bcast(rank == 0 ? mktempdir() : "", 0, comm)
            grid_prefix = "grid_failure"
            grid_target = joinpath(grid_dir, "$(grid_prefix)_shell_grid.nc")
            rank == 0 && mkpath(grid_target) # NCDataset cannot create a file over a directory
            MPI.Barrier(comm)
            grid_config = make_config(grid_dir, grid_prefix; include_grid = true)
            grid_tracker = GeoDynamo.create_time_tracker(grid_config, 0.0)
            grid_raised = 0
            grid_message = ""
            try
                GeoDynamo.write_fields!(Dict{String, Any}(), grid_tracker, metadata,
                    grid_config)
            catch err
                grid_raised = 1
                grid_message = sprint(showerror, err)
            end
            @test all(==(1), MPI.Allgather(grid_raised, comm))
            @test all(==(1), MPI.Allgather(
                occursin("grid file", lowercase(grid_message)) ? 1 : 0, comm))
            @test all(==(false), MPI.Allgather(grid_tracker.grid_file_written, comm))
            MPI.Barrier(comm)
        end
    end

    @testset "initial-condition save failure aborts on every rank" begin
        guard_present = isdefined(
            GeoDynamo.InitialConditions, :_save_initial_conditions_file_collectively!)
        @test guard_present

        if guard_present
            model = mkmodel()
            GeoDynamo.initialize_fields!(model.state)
            shared = MPI.bcast(rank == 0 ? mktempdir() : "", 0, comm)
            blocker = joinpath(shared, "not_a_directory")
            rank == 0 && touch(blocker)
            MPI.Barrier(comm)
            target = joinpath(blocker, "temperature.nc")

            raised = 0
            message = ""
            try
                GeoDynamo.save_initial_conditions(
                    model.state.fields.temperature, :temperature, target)
            catch err
                raised = 1
                message = sprint(showerror, err)
            end
            @test all(==(1), MPI.Allgather(raised, comm))
            @test all(==(1), MPI.Allgather(
                occursin("initial-condition file", lowercase(message)) ? 1 : 0, comm))
            MPI.Barrier(comm)
        end
    end

    @testset "NetCDF root-only setup failures are collective" begin
        # Inject the failure instead of relying on chmod: Windows does not apply
        # POSIX directory unlink permissions, and privileged POSIX users may also
        # delete through a mode-0500 directory.
        shared = MPI.bcast(rank == 0 ? mktempdir() : "", 0, comm)
        output_file = joinpath(shared, "existing_output.nc")
        probe_file = joinpath(shared, "existing_probe.nc")
        if rank == 0
            touch(output_file)
            touch(probe_file)
        end
        MPI.Barrier(comm)

        create_raised = 0
        create_message = ""
        try
            GeoDynamo.create_parallel_netcdf(
                output_file, GeoDynamo.default_config(), GeoDynamo.FieldInfo(),
                Dict{String, Any}(), comm;
                _remove_file = _ -> error("injected output removal failure"))
        catch err
            create_raised = 1
            create_message = sprint(showerror, err)
        end
        @test all(==(1), MPI.Allgather(create_raised, comm))
        @test all(==(1), MPI.Allgather(
            occursin("injected output removal failure", create_message) ? 1 : 0, comm))
        MPI.Barrier(comm)

        probe_called = Ref(false)
        name_threw = 0
        name_failure = nothing
        try
            name_failure = GeoDynamo.parallel_netcdf_probe(comm;
                _make_tempfile = () -> error("injected probe filename failure"),
                _probe_dataset = (_comm, _path) -> (probe_called[] = true))
        catch err
            name_threw = 1
            name_failure = err
        end
        name_message = sprint(showerror, name_failure)
        @test all(==(0), MPI.Allgather(name_threw, comm))
        @test all(==(1), MPI.Allgather(
            occursin("injected probe filename failure", name_message) ? 1 : 0, comm))
        @test all(==(0), MPI.Allgather(probe_called[] ? 1 : 0, comm))
        MPI.Barrier(comm)

        cleanup_threw = 0
        cleanup_failure = nothing
        try
            cleanup_failure = GeoDynamo.parallel_netcdf_probe(comm;
                _make_tempfile = () -> probe_file,
                _probe_dataset = (_comm, _path) -> nothing,
                _remove_file = _ -> error("injected probe cleanup failure"))
        catch err
            cleanup_threw = 1
            cleanup_failure = err
        end
        cleanup_message = sprint(showerror, cleanup_failure)
        @test all(==(0), MPI.Allgather(cleanup_threw, comm))
        @test all(==(1), MPI.Allgather(
            occursin("injected probe cleanup failure", cleanup_message) ? 1 : 0,
            comm))
        MPI.Barrier(comm)

        primary_threw = 0
        primary_failure = nothing
        try
            primary_failure = GeoDynamo.parallel_netcdf_probe(comm;
                _make_tempfile = () -> probe_file,
                _probe_dataset = (_comm, _path) -> error("injected primary probe failure"),
                _remove_file = _ -> error("injected secondary cleanup failure"))
        catch err
            primary_threw = 1
            primary_failure = err
        end
        primary_message = sprint(showerror, primary_failure)
        @test all(==(0), MPI.Allgather(primary_threw, comm))
        @test all(==(1), MPI.Allgather(
            occursin("injected primary probe failure", primary_message) ? 1 : 0, comm))
        @test all(==(0), MPI.Allgather(
            occursin("injected secondary cleanup failure", primary_message) ? 1 : 0,
            comm))
        MPI.Barrier(comm)
    end

    @testset "root-only NetCDF payload failures abort on every rank" begin
        sink = _InjectedNetCDFWriteFailure()
        config = GeoDynamo.default_config()
        field_info = GeoDynamo.FieldInfo(
            1, 0, 0, 0,
            [0.0], Float64[], Float64[], Int[], Int[],
            false, NamedTuple(), false, nothing,
            Dict{Symbol, UnitRange{Int}}(), Int[])

        operations = (
            ("coordinate data", () -> GeoDynamo.write_coordinate_data!(
                Dict{String, Any}("theta" => sink), field_info, config)),
            ("restart payload", () -> GeoDynamo.write_restart_field_data!(
                Dict{String, Any}("temperature_internal_sources" => sink),
                Dict{String, Any}("temperature_internal_sources" => [1.0]),
                config, GeoDynamo.FieldInfo())),
            ("time data", () -> GeoDynamo.write_time_data!(
                Dict{String, Any}("time" => sink, "step" => sink), 0.0, 0, config)),
            ("diagnostic data", () -> GeoDynamo.write_diagnostics!(
                Dict{String, Any}("diag_energy" => sink),
                Dict("energy" => 1.0), config)),
        )

        for (expected_text, operation) in operations
            raised = 0
            message = ""
            try
                operation()
            catch err
                raised = 1
                message = sprint(showerror, err)
            end
            @test all(==(1), MPI.Allgather(raised, comm))
            @test all(==(1), MPI.Allgather(
                occursin(expected_text, lowercase(message)) ? 1 : 0, comm))
            MPI.Barrier(comm)
        end
    end

    @testset "restart tracker failures abort on every rank" begin
        helper_present = isdefined(
            GeoDynamo, :_write_restart_tracker_data_collectively!)
        @test helper_present
        if helper_present
            sink = _InjectedNetCDFWriteFailure()
            ds = Dict{String, Any}(
                "last_output_time" => sink,
                "output_count" => sink,
                "restart_count" => sink,
                "grid_file_written" => sink,
            )
            raised = 0
            message = ""
            try
                GeoDynamo._write_restart_tracker_data_collectively!(
                    ds, 0.0, 0, 1, false, GeoDynamo.default_config(), comm)
            catch err
                raised = 1
                message = sprint(showerror, err)
            end
            @test all(==(1), MPI.Allgather(raised, comm))
            @test all(==(1), MPI.Allgather(
                occursin("restart tracker", lowercase(message)) ? 1 : 0, comm))
            MPI.Barrier(comm)
        end
    end

    @testset "history size reporting is nonfatal" begin
        helper_present = isdefined(GeoDynamo, :_report_parallel_write_complete!)
        @test helper_present
        if helper_present
            size_called = Ref(false)
            raised = 0
            try
                GeoDynamo._report_parallel_write_complete!(
                    "unused.nc", 0.1, comm,
                    _ -> true,
                    _ -> begin
                        size_called[] = true
                        error("injected history status failure")
                    end)
            catch
                raised = 1
            end
            @test all(==(0), MPI.Allgather(raised, comm))
            @test sum(MPI.Allgather(size_called[] ? 1 : 0, comm)) == 1
            MPI.Barrier(comm)
        end
    end

    # ── and the symmetric default must still SKIP the per-step reduction ───────
    @testset "default registry leaves the per-step stop reduction disarmed" begin
        sim = GeoDynamo.Simulation(mkmodel(); Δt = 1e-4, stop_iteration = 2)
        GeoDynamo.run!(sim)
        MPI.Barrier(comm)
        decisions = MPI.Allgather(sim._stop_needs_reduce, comm)
        @test all(==(decisions[1]), decisions)
        @test decisions[1] == false      # every built-in stop is rank-symmetric
    end

    # ── registries freeze on the first step; asymmetric registration is loud ──
    @testset "callback registry freezes on first time_step! on every rank" begin
        sim = GeoDynamo.Simulation(mkmodel(); Δt = 1e-4, stop_iteration = 5)
        GeoDynamo.time_step!(sim)
        frozen = GeoDynamo.isfrozen(sim.callbacks) ? 1 : 0
        reduce = sim._stop_needs_reduce ? 1 : 0
        @test all(==(1), MPI.Allgather(frozen, comm))
        @test all(==(0), MPI.Allgather(reduce, comm))   # default registry: no per-step Allreduce
        raised = try
            GeoDynamo.add_callback!(sim, s -> nothing;
                schedule = GeoDynamo.IterationInterval(1), name = :late)
            0
        catch err
            err isa ArgumentError && occursin("frozen", sprint(showerror, err)) ? 1 : 0
        end
        @test all(==(1), MPI.Allgather(raised, comm))
        # steps keep working on the frozen registry
        GeoDynamo.time_step!(sim)
        @test all(==(2), MPI.Allgather(sim.model.clock.iteration, comm))
        MPI.Barrier(comm)
    end

    @testset "asymmetric add_callback! before the first step aborts everywhere" begin
        if nranks == 1
            @test_skip "needs two ranks"
        else
            sim = GeoDynamo.Simulation(mkmodel(); Δt = 1e-4, stop_iteration = 5)
            rank == nranks - 1 && GeoDynamo.add_callback!(sim, s -> nothing;
                schedule = GeoDynamo.IterationInterval(1), name = :last_rank_only)
            raised = try
                GeoDynamo.time_step!(sim)
                0
            catch err
                occursin("callback registry mismatch", lowercase(sprint(showerror, err))) ? 1 : 0
            end
            @test all(==(1), MPI.Allgather(raised, comm))
            @test all(==(0), MPI.Allgather(sim.model.clock.iteration, comm))
            MPI.Barrier(comm)
        end
    end

    @testset "direct registry mutation is rejected on every rank" begin
        sim = GeoDynamo.Simulation(mkmodel(); Δt = 1e-4, stop_iteration = 1)
        raised = try
            sim.callbacks[:direct] = GeoDynamo.Callback(s -> nothing, GeoDynamo.IterationInterval(1))
            0
        catch err
            err isa ArgumentError ? 1 : 0
        end
        @test all(==(1), MPI.Allgather(raised, comm))
        MPI.Barrier(comm)
    end
end
