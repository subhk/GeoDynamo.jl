# ================================================================================
# Review 2026-09-04 — writer-registry collectives, restart bookkeeping,
#                     topography rebase, Gaunt cross-cache contention
# ================================================================================
#
# Serial regressions for the 2026-09-04 branch review of
# `test/mpi-control-plane-invariants`. The multi-rank half (the writer-registry
# validator actually raising on EVERY rank) lives in
# `test/mpi_control_plane_invariants.jl`, because that defect is a HANG rather
# than a failure and needs >= 2 ranks to bite.
# ================================================================================

using Test
using GeoDynamo

const topo0904 = GeoDynamo.bcs.topography

@testset "Review 2026-09-04 fixes" begin

    # ── R1: the output-writer registry gates collectives, so it must be checked ──
    @testset "R1 output-writer registry has a structural signature + validator" begin
        # `_restore_output_writer_trackers!` fires three collectives per
        # FieldWriter/CheckpointWriter (`_existing_writer_count` -> Bcast! + bcast,
        # `_existing_grid_file` -> Bcast!) and NONE for the generic fallback. So the
        # number of collectives entered inside the `Simulation` constructor is a
        # function of THIS rank's `output_writers`, and
        # `output_writers = rank == 0 ? (w = FieldWriter(dir; ...),) : NamedTuple()`
        # hangs the constructor. The callback registry already had
        # `_freeze_callbacks!` for exactly this; the writer registry did not.
        sched = GeoDynamo.IterationInterval(5)
        a = GeoDynamo._to_ordered((snap = GeoDynamo.FieldWriter("out"; schedule = sched),),
            :writer)
        b = GeoDynamo._to_ordered((snap = GeoDynamo.FieldWriter("out"; schedule = sched),),
            :writer)
        empty_reg = GeoDynamo._to_ordered(NamedTuple(), :writer)

        sig = GeoDynamo._output_writer_registry_signature
        # structurally identical registries agree
        @test sig(a) == sig(b)
        # ... and every way of differing disagrees
        @test sig(a) != sig(empty_reg)
        @test sig(a) != sig(GeoDynamo._to_ordered(
            (snap = GeoDynamo.FieldWriter("other"; schedule = sched),), :writer))
        @test sig(a) != sig(GeoDynamo._to_ordered(
            (other = GeoDynamo.FieldWriter("out"; schedule = sched),), :writer))
        @test sig(a) != sig(GeoDynamo._to_ordered(
            (snap = GeoDynamo.FieldWriter("out";
                schedule = GeoDynamo.IterationInterval(6)),), :writer))
        @test sig(a) != sig(GeoDynamo._to_ordered(
            (snap = GeoDynamo.CheckpointWriter("out"; schedule = sched),), :writer))

        # at one rank the validator is a no-op rather than an error
        @test GeoDynamo.isfrozen(GeoDynamo._freeze_output_writers!(
            GeoDynamo.CollectiveRegistry{:writer}(a)))
        @test GeoDynamo.isfrozen(GeoDynamo._freeze_output_writers!(
            GeoDynamo.CollectiveRegistry{:writer}(empty_reg)))
    end

    @testset "R1 registries are frozen at both collective entry points" begin
        # Writers: validated + frozen in the constructor, BEFORE the collective
        # tracker restore. Callbacks: frozen at run! entry and at the top of the
        # public time_step!. Neither call is reachable from a serial test, so pin
        # the source: a freeze nothing calls fixes nothing.
        root = pkgdir(GeoDynamo)
        sim_src = read(joinpath(root, "src", "api", "simulation.jl"), String)
        ctor_freeze = findfirst("_freeze_output_writers!(output_writer_items)", sim_src)
        restore_call = findfirst("_restore_output_writer_trackers!(", sim_src)
        @test ctor_freeze !== nothing && restore_call !== nothing
        @test first(ctor_freeze) < first(restore_call)
        runbang = match(r"function run!\(sim::Simulation\)\n(?s:.*?)\nend\n", sim_src)
        @test runbang !== nothing && occursin("_freeze_callbacks!(sim)", runbang.match)
        stepbang = match(r"function time_step!\(sim::Simulation\)\n(?s:.*?)\nend\n", sim_src)
        @test stepbang !== nothing && occursin("_freeze_callbacks!(sim)", stepbang.match)
    end

    # ── R2: the callback guard and the _stop_needs_reduce comment must agree ────
    @testset "R2 registry mismatch is explained, and the comment matches the guard" begin
        # `_freeze_callbacks!` aborts on EVERY asymmetric registry, while
        # the `_stop_needs_reduce` field comment advertised
        # `rank == 0 && add_callback!(...)` as ordinary usage. Two halves of the same
        # change disagreeing is how a user learns the rule from a stack trace.
        root = pkgdir(GeoDynamo)
        sim_src = read(joinpath(root, "src", "api", "simulation.jl"), String)
        # the stale claim must be gone
        @test !occursin("is ordinary usage", sim_src)
        # and the comment must point at the guard that is actually enforced
        @test occursin("_freeze_callbacks!", sim_src)
        # the error has to say what to do instead of just what went wrong
        msg = GeoDynamo._CALLBACK_REGISTRY_MISMATCH
        @test startswith(msg, "MPI callback registry mismatch")
        @test occursin("every rank", msg)
        @test occursin("inside", msg)
    end

    # ── R3: a cross-Gaunt cache HIT must not take the process-global lock ───────
    @testset "R3 get_cross_gaunt serves cache hits without the lock" begin
        cache = topo0904.GauntTensorCache{Float64}(3, 2)
        # (l1,m1,l2,m2,L,M) = (1,0,1,0,1,0) passes every selection rule:
        # m1 == m2 + M, |l2-L| <= l1 <= l2+L, and l1+l2+L is odd.
        key = (1, 0, 1, 0, 1, 0)
        warm = topo0904.get_cross_gaunt(cache, key...)

        # Hold the lock from ANOTHER task. A reader that still needs it to serve a
        # warmed key blocks here; a lock-free hit path returns immediately. Task
        # (not thread) based, so this bites at nthreads() == 1 too.
        held = Channel{Nothing}(1)
        release = Channel{Nothing}(1)
        holder = Threads.@spawn begin
            lock(topo0904._GAUNT_CROSS_LOCK)
            put!(held, nothing)
            take!(release)
            unlock(topo0904._GAUNT_CROSS_LOCK)
        end
        take!(held)

        got = Ref{Any}(nothing)
        finished = Ref(false)
        reader = Threads.@spawn begin
            got[] = topo0904.get_cross_gaunt(cache, key...)
            finished[] = true
        end
        served = timedwait(() -> finished[], 10.0)

        put!(release, nothing)
        wait(holder)
        wait(reader)

        @test served === :ok
        @test got[] == warm
        # the whole selection-rule fast path must stay lock-free as well
        @test topo0904.get_cross_gaunt(cache, 1, 0, 1, 0, 2, 0) == 0.0
    end

    @testset "R3 lazy cross-Gaunt batches publish their final entries" begin
        numerical_zero = (1, 0, 1, 0, 1, 0)
        nonzero = (2, 1, 1, 1, 2, 0)
        cache = topo0904.GauntTensorCache{Float64}(4, 4)

        # The first insert is auto-published (threshold 1), while the second leaves
        # the cache at size 2, below the next doubling threshold of 3. A completed
        # direct-access batch therefore needs an explicit final publication.
        @test topo0904.get_cross_gaunt(cache, numerical_zero...) == 0.0
        expected = topo0904.get_cross_gaunt(cache, nonzero...)
        @test abs(expected) > 1e-14
        @test !haskey(cache.G_cross_view, nonzero)

        flush = isdefined(topo0904, :flush_cross_gaunt_view!) ?
                getfield(topo0904, :flush_cross_gaunt_view!) : nothing
        @test flush !== nothing
        flush === nothing || flush(cache)
        @test haskey(cache.G_cross_view, nonzero)
        published = cache.G_cross_view
        if flush !== nothing
            @test flush(cache) === cache
        end
        @test cache.G_cross_view === published

        # A hit on that second key must now complete while the process-global write
        # lock is held by another task.
        held = Channel{Nothing}(1)
        release = Channel{Nothing}(1)
        done = Channel{Float64}(1)
        holder = Threads.@spawn begin
            lock(topo0904._GAUNT_CROSS_LOCK)
            put!(held, nothing)
            take!(release)
            unlock(topo0904._GAUNT_CROSS_LOCK)
        end
        take!(held)
        reader = Threads.@spawn put!(done, topo0904.get_cross_gaunt(cache, nonzero...))
        served = timedwait(() -> isready(done), 2.0)
        put!(release, nothing)
        wait(holder)
        wait(reader)

        @test served === :ok
        @test take!(done) == expected

        # Replacing a remembered zero does not increase the authoritative cache's
        # size. The flush must still publish the changed bounded contents.
        bounded = topo0904.GauntTensorCache{Float64}(4, 4;
            cross_zero_cache_limit = 1)
        second_zero = (2, 0, 1, 0, 2, 0)
        @test topo0904.get_cross_gaunt(bounded, numerical_zero...) == 0.0
        @test topo0904.get_cross_gaunt(bounded, second_zero...) == 0.0
        @test length(bounded.G_cross_zero) == 1
        @test haskey(bounded.G_cross_view, numerical_zero)
        @test !haskey(bounded.G_cross_view, second_zero)
        flush === nothing || flush(bounded)
        @test length(bounded.G_cross_zero) == 1
        @test !haskey(bounded.G_cross_view, numerical_zero)
        @test haskey(bounded.G_cross_view, second_zero)

        # The solver-facing topography dispatcher is the batch boundary: it must
        # publish once after all enabled correction families have completed.
        production_cache = topo0904.GauntTensorCache{Float64}(4, 4)
        topo0904.get_cross_gaunt(production_cache, numerical_zero...)
        topo0904.get_cross_gaunt(production_cache, nonzero...)
        @test !haskey(production_cache.G_cross_view, nonzero)
        topography = topo0904.TopographyData{Float64}()
        topography.gaunt_cache = production_cache
        config = topo0904.TopographyCouplingConfig(enabled = true,
            velocity_coupling = false, magnetic_coupling = false,
            thermal_coupling = false)
        topo0904.apply_all_topography_corrections!(NamedTuple(), topography;
            config = config)
        @test haskey(production_cache.G_cross_view, nonzero)
        published_after_pass = production_cache.G_cross_view
        topo0904.apply_all_topography_corrections!(NamedTuple(), topography;
            config = config)
        @test production_cache.G_cross_view === published_after_pass
    end

    # ── R4: the persisted history count must describe the HISTORY directory ─────
    @testset "R4 _persisted_output_count scans the history directory" begin
        # `_run_output_writer!(::CheckpointWriter, ...)` builds its config with
        # `output_dir = ow.path` — the CHECKPOINT directory. Scanning that for
        # `geodynamo_<geom>_hist_N.nc` finds nothing, so the persisted `output_count`
        # stays 0 and the docstring's stated fix never fires.
        mktempdir() do root
            hist = joinpath(root, "output"); mkpath(hist)
            ckpt = joinpath(root, "checkpoints"); mkpath(ckpt)
            for n in 1:9
                touch(joinpath(hist, "geodynamo_shell_hist_$(n).nc"))
            end

            base = GeoDynamo.default_config()
            mkconfig(dir) = GeoDynamo.OutputConfig(
                base.output_space, dir, base.filename_prefix,
                base.include_metadata, base.include_grid, base.include_diagnostics,
                base.output_precision, base.spectral_lmax_output, base.overwrite_files,
                base.output_interval, base.restart_interval, base.max_output_time,
                base.time_tolerance)

            private = GeoDynamo.TimeTracker(-Inf, 1.0, 0, 3, Inf, 2.0, false)
            # the checkpoint writer's own directory holds no history files at all
            @test GeoDynamo._persisted_output_count(
                private, false, mkconfig(ckpt), :shell) == 0
            # pointed at the history directory it recovers the real count
            @test GeoDynamo._persisted_output_count(private, false, mkconfig(ckpt), :shell;
                history_dirs = (hist,)) == 9
            # the legacy single-directory path must not regress
            @test GeoDynamo._persisted_output_count(
                private, false, mkconfig(hist), :shell) == 9
            # and the tracker still wins when it is ahead
            live = GeoDynamo.TimeTracker(1.0, 1.0, 9, 3, 2.0, 2.0, false)
            @test GeoDynamo._persisted_output_count(live, true, mkconfig(ckpt), :shell;
                history_dirs = (hist,)) == 10

            # the checkpoint writer learns the history directories from the registry
            sched = GeoDynamo.IterationInterval(1)
            writers = GeoDynamo._to_ordered((
                    snap = GeoDynamo.FieldWriter(hist; schedule = sched),
                    chk = GeoDynamo.CheckpointWriter(ckpt; schedule = sched),
                ), :writer)
            @test GeoDynamo._history_output_dirs(writers) == (hist,)
            # no FieldWriter at all: fall back to the caller's own directory
            only_ckpt = GeoDynamo._to_ordered(
                (chk = GeoDynamo.CheckpointWriter(ckpt; schedule = sched),), :writer)
            @test GeoDynamo._history_output_dirs(only_ckpt) == ()
        end
    end

    # ── R5: rolling a topography correction back must be element-wise ───────────
    @testset "R5 reset_boundary_to_base! rolls back through a NaN" begin
        # `bv == entry.applied` is FALSE whenever an element is NaN, so the rollback
        # was skipped and the CORRECTED array became the new base — the compounding
        # the base/snapshot mechanism exists to prevent.
        bv = [1.0 2.0; 3.0 4.0]
        topo0904.reset_boundary_to_base!(bv)      # base = [1 2; 3 4]
        bv[1, 1] = NaN                            # the correction itself wrote NaN
        topo0904.mark_boundary_applied!(bv)
        topo0904.reset_boundary_to_base!(bv)
        @test bv == [1.0 2.0; 3.0 4.0]
    end

    @testset "R5 reset_boundary_to_base! rebases only what another owner wrote" begin
        # `apply_temperature_boundaries!` / `apply_composition_boundaries!` write only
        # `1:min(length(coeffs), nlm)` of the REAL array and never the imaginary one,
        # so a partial overwrite is the shape that actually occurs. Exact whole-array
        # equality turns that into "adopt the corrected array wholesale".
        bv = [1.0 2.0; 3.0 4.0]
        topo0904.reset_boundary_to_base!(bv)      # base = [1 2; 3 4]
        bv .+= 0.5                                # topography correction
        topo0904.mark_boundary_applied!(bv)
        bv[2, 1] = 99.0                           # another owner writes ONE element
        topo0904.reset_boundary_to_base!(bv)
        @test bv == [1.0 2.0; 99.0 4.0]

        # and a second correction pass must not compound on top of the new base
        bv .+= 0.5
        topo0904.mark_boundary_applied!(bv)
        topo0904.reset_boundary_to_base!(bv)
        @test bv == [1.0 2.0; 99.0 4.0]
    end
end
