# ================================================================================
# Regression tests for batch E — the findings from the high-effort review of the
# control-plane branch delta (git diff main...test/mpi-control-plane-invariants).
# ================================================================================
#
#   E1 api/output_writers.jl:105  a restored `grid_file_written` describes the
#                                 directory the CHECKPOINT was written to, not the
#                                 one a resumed run writes to. Trusting it made a
#                                 resume into a fresh FieldWriter(new_dir) skip
#                                 write_grid_file! forever, leaving new_dir with
#                                 history files and no grid file. The counters are
#                                 already guarded this way by _existing_writer_count.
#   E2 timestep/driver.jl:212     `SolverState.topography` is typed
#                                 SolverTopographyState{T}, never a Union with
#                                 Nothing, so the `topo !== nothing` guard was dead.
#   E3 api/callbacks.jl:285       the stop-flag Allreduce fired on EVERY step even
#                                 when every registered callback already leaves
#                                 `sim.running` rank-identical (the built-ins reduce
#                                 internally; the stop conditions read only the
#                                 clock). Skipping it must stay fail-safe: an
#                                 unrecognised callback still forces the reduction.
# ================================================================================

using Test
using MPI
using GeoDynamo

# `_run_callbacks!` only reads the clock, the callback registry and the stop flag,
# so a mock stands in for a full Simulation (which would need a solver state).
struct _BatchEClock
    time::Float64
    iteration::Int
end
struct _BatchEModel
    clock::_BatchEClock
end
mutable struct _BatchESim
    model::_BatchEModel
    callbacks::Any
    running::Bool
    _wall_start::Float64
    stop_time::Float64
    stop_iteration::Int
    wall_time_limit::Float64
end

# `nan_checker` is the one default that needs a real model (it scans the fields via
# _health_check), so the behaviour test below drops it; the three stop conditions
# read only the clock and the mock's own limits.
function _batchE_sim()
    callbacks = GeoDynamo._default_callbacks()
    delete!(callbacks, :nan_checker)
    return _BatchESim(_BatchEModel(_BatchEClock(0.5, 3)), callbacks, true, 0.0,
        Inf, typemax(Int), Inf)
end

@testset "Max-effort review batch E" begin

    # ── E1: the grid-file flag must come from the writer's OWN directory ──
    @testset "E1 restored grid_file_written is recomputed from the target path" begin
        # restored says "already written" — but it was written to another directory
        mktempdir() do dir
            restored = GeoDynamo.TimeTracker(1.0, 1.0, 7, 3, 2.0, 2.0, true)
            writer = GeoDynamo.FieldWriter(dir; schedule = GeoDynamo.IterationInterval(1))
            tracker = GeoDynamo._restore_output_writer_tracker!(writer, restored, :shell)
            @test tracker.grid_file_written == false
            # the counters must still be restored — this fix must not undo #109/#110
            @test tracker.output_count == 7
        end

        # restored says "not written", but this directory already has one
        mktempdir() do dir
            touch(joinpath(dir, "geodynamo_shell_grid.nc"))
            restored = GeoDynamo.TimeTracker(1.0, 1.0, 7, 3, 2.0, 2.0, false)
            writer = GeoDynamo.FieldWriter(dir; schedule = GeoDynamo.IterationInterval(1))
            tracker = GeoDynamo._restore_output_writer_tracker!(writer, restored, :shell)
            @test tracker.grid_file_written == true
        end

        # the merge branch (writer already holds a tracker) must agree with the
        # fresh-copy branch: both read the directory, neither trusts `restored`
        mktempdir() do dir
            restored = GeoDynamo.TimeTracker(1.0, 1.0, 7, 3, 2.0, 2.0, true)
            writer = GeoDynamo.FieldWriter(dir; schedule = GeoDynamo.IterationInterval(1))
            writer._tracker[] = GeoDynamo.TimeTracker(0.0, 0.0, 0, 0, 0.0, 0.0, false)
            tracker = GeoDynamo._restore_output_writer_tracker!(writer, restored, :shell)
            @test tracker.grid_file_written == false
        end

        # geometry is part of the filename — a :ball resume must not be satisfied
        # by a :shell grid file left in the same directory
        mktempdir() do dir
            touch(joinpath(dir, "geodynamo_shell_grid.nc"))
            restored = GeoDynamo.TimeTracker(1.0, 1.0, 7, 3, 2.0, 2.0, false)
            writer = GeoDynamo.FieldWriter(dir; schedule = GeoDynamo.IterationInterval(1))
            tracker = GeoDynamo._restore_output_writer_tracker!(writer, restored, :ball)
            @test tracker.grid_file_written == false
        end

        # CheckpointWriter takes the same treatment (include_grid = false today, so
        # this only pins that the two restore methods stay in step)
        mktempdir() do dir
            restored = GeoDynamo.TimeTracker(1.0, 1.0, 7, 3, 2.0, 2.0, true)
            writer = GeoDynamo.CheckpointWriter(dir; schedule = GeoDynamo.IterationInterval(1))
            tracker = GeoDynamo._restore_output_writer_tracker!(writer, restored, :shell)
            @test tracker.grid_file_written == false
            @test tracker.restart_count == 3
        end
    end

    # ── E2: the dead nothing-check rested on this field never being optional ──
    @testset "E2 SolverState.topography is not optional" begin
        ft = fieldtype(GeoDynamo.SolverState, :topography)
        @test !(Nothing <: ft)
    end

    # ── E3: skip the per-step stop-flag reduction only when it cannot matter ──
    @testset "E3 stop-flag reduction is skipped only for rank-symmetric callbacks" begin
        every = GeoDynamo.IterationInterval(1)
        defaults = GeoDynamo._default_callbacks()

        # the four built-ins: three clock-only stop conditions + nan_checker, which
        # reduces internally via _any_rank_flag
        @test GeoDynamo._callbacks_may_stop_rank_locally(defaults) == false

        # built-in callback types never touch sim.running (HealthCheck throws instead)
        for cb in (GeoDynamo.HealthCheck(schedule = every),
            GeoDynamo.EnergyDiagnostics(schedule = every),
            GeoDynamo.SolenoidalMonitor(schedule = every),
            GeoDynamo.SimulationProgress(schedule = every))
            registry = copy(defaults)
            registry[:extra] = cb
            @test GeoDynamo._callbacks_may_stop_rank_locally(registry) == false
        end

        # an unrecognised user callback is the conservative case: it may stop the run
        # from rank-local state, so the reduction must NOT be skipped
        user = copy(defaults)
        user[:user] = GeoDynamo.Callback(sim -> nothing, every)
        @test GeoDynamo._callbacks_may_stop_rank_locally(user) == true

        # ClockOnlyCallback is the documented opt-out: the clock is rank-symmetric
        clock_only = copy(defaults)
        clock_only[:user] = GeoDynamo.Callback(
            GeoDynamo.ClockOnlyCallback(sim -> nothing), every)
        @test GeoDynamo._callbacks_may_stop_rank_locally(clock_only) == false

        # …and an unrecognised NON-Callback entry must also stay conservative
        unknown = copy(defaults)
        unknown[:unknown] = 42
        @test GeoDynamo._callbacks_may_stop_rank_locally(unknown) == true
    end

    # ── E3 (behaviour): skipping the reduction must not skip the stop itself ──
    @testset "E3 a user callback can still stop the run" begin
        sim = _batchE_sim()
        sim.callbacks[:stopper] = GeoDynamo.Callback(
            s -> (s.running = false), GeoDynamo.IterationInterval(1))
        @test GeoDynamo._callbacks_may_stop_rank_locally(sim.callbacks) == true
        GeoDynamo._run_callbacks!(sim)
        @test sim.running == false

        # and a run whose callbacks are all rank-symmetric keeps going with the
        # reduction skipped
        quiet = _batchE_sim()
        @test GeoDynamo._callbacks_may_stop_rank_locally(quiet.callbacks) == false
        GeoDynamo._run_callbacks!(quiet)
        @test quiet.running == true
    end
end
