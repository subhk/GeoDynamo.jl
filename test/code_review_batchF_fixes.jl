# ================================================================================
# Regression tests for batch F — the findings from the src/physics and src/io
# high-effort review passes.
# ================================================================================
#
#   F1 solver/numerics.jl:213  the threaded-update guard was a PROCESS-global Ref, so
#                              a second Simulation stepped concurrently in the same
#                              process had its perfectly ordered reductions rejected,
#                              and the `finally` disarmed the guard as soon as the
#                              first spawned task rethrew — while its siblings were
#                              still running unfetched. Now scoped to the task that
#                              arms it.
#   F2 solver/numerics.jl:194  `mpi_barrier!` sat three lines above the guard and was
#                              the one collective in that file left unwrapped;
#                              MPI.Barrier deadlocks from a spawned task exactly like
#                              Allreduce.
#   F3 io/restart.jl:107       the restart file was chosen by a rank-LOCAL readdir and
#                              handed straight to the COLLECTIVE NCDataset open, so
#                              ranks with different directory views open different
#                              files (or one rank errors alone and the rest hang).
#                              Rank 0 now selects and broadcasts.
#   F4 io/restart.jl:32        the persisted `output_count` came from whichever tracker
#                              wrote the checkpoint. CheckpointWriter uses its own
#                              private tracker (output_interval = Inf), which is
#                              permanently at 0, so resuming from one of its files and
#                              writing history clobbered hist_1.
# ================================================================================

using Test
using MPI
using GeoDynamo

@testset "Max-effort review batch F" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping batch F fix tests"
        return
    end
    MPI.Initialized() || MPI.Init()


    # ── F1: the guard belongs to the task that armed it, not to the process ──
    @testset "F1 threaded-update guard is task-scoped" begin
        @test GeoDynamo._in_threaded_implicit_update() == false

        armed_here, armed_in_child = GeoDynamo._with_threaded_update_guard() do
            # a task spawned from inside the region does NOT inherit the flag, which
            # is what keeps an unrelated concurrent solver from being rejected
            (GeoDynamo._in_threaded_implicit_update(),
                fetch(Threads.@spawn GeoDynamo._in_threaded_implicit_update()))
        end
        @test armed_here == true
        @test armed_in_child == false

        # restored on the way out, including when the body throws
        @test GeoDynamo._in_threaded_implicit_update() == false
        @test_throws ErrorException GeoDynamo._with_threaded_update_guard(
            () -> error("boom"))
        @test GeoDynamo._in_threaded_implicit_update() == false

        # a sibling task is unaffected while this one is armed — the concurrent
        # Simulation scenario the process-global Ref got wrong
        sibling_ok = GeoDynamo._with_threaded_update_guard() do
            fetch(Threads.@spawn begin
                GeoDynamo.allreduce_sum(1.0)
                true
            end)
        end
        @test sibling_ok == true

        # and the reduction helpers still refuse to run inside the armed task
        GeoDynamo._with_threaded_update_guard() do
            @test_throws ErrorException GeoDynamo.allreduce_sum(1.0)
        end
    end

    # ── F2: Barrier deadlocks from a spawned task exactly like Allreduce ──
    @testset "F2 mpi_barrier! is covered by the guard" begin
        GeoDynamo._with_threaded_update_guard() do
            err = try
                GeoDynamo.mpi_barrier!()
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("mpi_barrier!", sprint(showerror, err))
        end
        # …and is a plain barrier outside the region
        @test GeoDynamo.mpi_barrier!() === nothing
    end

    # ── F3: the collectively-opened restart path must be rank 0's choice ──
    @testset "F3 restart file selection is rank-0 authoritative" begin
        if MPI.Initialized() && MPI.Comm_size(GeoDynamo.output_comm()) > 1
            # Give every rank a DIFFERENT directory. A rank-local selection returns
            # each rank's own file; a broadcast selection returns rank 0's on all of
            # them. Only the latter is safe to hand to a collective NCDataset open.
            comm = GeoDynamo.output_comm()
            rank = MPI.Comm_rank(comm)
            mktempdir() do dir
                touch(joinpath(dir, "geodynamo_shell_restart_$(rank + 1).nc"))
                chosen = GeoDynamo._restart_path_for_all_ranks(dir, -1.0)
                n = parse(Int, match(r"_(\d+)\.nc$", basename(chosen)).captures[1])
                gathered = MPI.Allgather(Int[n], comm)
                @test all(==(gathered[1]), gathered)
                @test gathered[1] == 1          # rank 0's file, on every rank
            end
        else
            # At one rank the broadcast is the identity, but the entry point must
            # still resolve the same file the local scan would have picked.
            mktempdir() do dir
                touch(joinpath(dir, "geodynamo_shell_restart_1.nc"))
                chosen = GeoDynamo._restart_path_for_all_ranks(dir, -1.0)
                @test basename(chosen) == "geodynamo_shell_restart_1.nc"
            end
        end

        # an empty directory must raise on EVERY rank (a rank-local `error` in front
        # of a collective open is the deadlock this fix exists to remove)
        mktempdir() do dir
            @test_throws ErrorException GeoDynamo._restart_path_for_all_ranks(dir, -1.0)
        end

        # …and so must a MISSING one. `find_restart_files` opens with a bare `readdir`,
        # which throws SystemError; on rank 0 alone that leaves the other ranks blocked
        # in the broadcast forever, so the scan has to be total.
        missing_dir = joinpath(mktempdir(), "definitely_not_here")
        @test !isdir(missing_dir)
        err = try
            GeoDynamo._restart_path_for_all_ranks(missing_dir, -1.0)
            nothing
        catch e
            e
        end
        @test err isa ErrorException            # NOT SystemError: reached the unanimous path
        @test occursin("restart files", sprint(showerror, err))
    end

    # ── F5: one parallel-NetCDF probe, usable as a predicate ─────────────────
    @testset "F5 parallel NetCDF capability is probeable, not just fatal" begin
        comm = GeoDynamo.output_comm()
        probe_err = GeoDynamo.parallel_netcdf_probe(comm)
        @test probe_err === nothing || probe_err isa Exception
        @test GeoDynamo.parallel_netcdf_available(comm) == (probe_err === nothing)

        # the fail-loud wrapper agrees with the predicate — it exists so callers can
        # choose to abort, not so the capability can only be discovered by crashing
        if probe_err === nothing
            @test GeoDynamo.check_parallel_netcdf_support(comm) === nothing
        else
            @test_throws ErrorException GeoDynamo.check_parallel_netcdf_support(comm)
        end
    end

    # ── F4b: the scan pattern must survive a regex-hostile filename prefix ────
    @testset "F4b output-count scan escapes the configured prefix" begin
        mktempdir() do dir
            # a second capture group would make `only(matched.captures)` throw, and an
            # unbalanced bracket would throw in the Regex constructor — both from inside
            # a checkpoint write
            for prefix in ("run(1)", "run[", "a.b", "x+y")
                touch(joinpath(dir, "$(prefix)_shell_hist_4.nc"))
                @test GeoDynamo._scan_output_count(dir, prefix, :shell, "hist") == 4
            end
            # and the escaping must not make the pattern match too much
            @test GeoDynamo._scan_output_count(dir, "a.b", :shell, "hist") == 4
            @test GeoDynamo._scan_output_count(dir, "axb", :shell, "hist") == 0
        end
    end

    # ── F4: the persisted output count must describe the run, not the writer ──
    @testset "F4 persisted output_count is taken from the output directory" begin
        mktempdir() do dir
            for n in 1:9
                touch(joinpath(dir, "geodynamo_shell_hist_$(n).nc"))
            end
            # the local scan the two counters now share
            @test GeoDynamo._scan_output_count(dir, "geodynamo", :shell, "hist") == 9
            @test GeoDynamo._scan_output_count(dir, "geodynamo", :shell, "restart") == 0
            @test GeoDynamo._scan_output_count(dir, "geodynamo", :ball, "hist") == 0

            base = GeoDynamo.default_config()
            config = GeoDynamo.OutputConfig(
                base.output_space, dir, base.filename_prefix,
                base.include_metadata, base.include_grid, base.include_diagnostics,
                base.output_precision, base.spectral_lmax_output, base.overwrite_files,
                base.output_interval, base.restart_interval, base.max_output_time,
                base.time_tolerance)

            # CheckpointWriter's private tracker: output_interval = Inf keeps it at 0
            # forever, so trusting it persisted 0 and the resume clobbered hist_1
            private = GeoDynamo.TimeTracker(-Inf, 1.0, 0, 3, Inf, 2.0, false)
            @test GeoDynamo._persisted_output_count(private, false, config, :shell) == 9

            # the write_fields! path already knew the answer; it must not regress
            live = GeoDynamo.TimeTracker(1.0, 1.0, 9, 3, 2.0, 2.0, false)
            @test GeoDynamo._persisted_output_count(live, true, config, :shell) == 10
        end
    end
end
