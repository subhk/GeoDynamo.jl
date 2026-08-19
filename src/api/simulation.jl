# ================================================================================
# Simulation — top-level driver for the Oceananigans-style API
# ================================================================================

"""
    mutable struct Simulation{M,C,O}

Holds a `GeodynamoModel` together with time-stepping controls, callbacks, and
output writers.  Create with `Simulation(model; Δt, ...)` and advance with
`run!(sim)`.
"""
mutable struct Simulation{M, C, O}
    model::M
    dt::Float64
    stop_time::Float64
    stop_iteration::Int
    wall_time_limit::Float64
    running::Bool
    gpu::Bool
    callbacks::C
    output_writers::O
    _wall_start::Float64
    _gpu_state::Any         # cached device-state bundle (built lazily)
    _gpu_erk2::Any          # cached ExponentialRungeKutta2 operator pack
    _gpu_dt::Float64        # dt baked into _gpu_state (rebuild on change)
    _gpu_sync::Symbol       # :every (host state synced per step) or :output (lazy)
    _gpu_dirty::Bool        # device state ahead of the host mirror
    # Host-side copies of the internal-source profiles baked into `_gpu_state`.
    # Kept here, not read back from the bundle: `gpu_to_device` moves every array
    # in the bundle, so on CUDA the baked vector lives on the device and diffing
    # it against the host field would mean scalar indexing.
    _gpu_src::NTuple{2, Union{Vector{Float64}, Nothing}}
    # Host-side copies of the scalar boundary ENDPOINT values baked into `_gpu_state`
    # (temperature, composition), for the same reason as `_gpu_src`: the bundle copies
    # them at pack time and is rebuilt only on a Δt change, so a programmatic BC write
    # mid-run would otherwise keep being integrated against the old wall values.
    _gpu_bc::NTuple{2, Union{Matrix{Float64}, Nothing}}
    # Whether `_run_callbacks!` must reduce the `running` flag across ranks each
    # step. Decided ONCE, collectively, at `run!` entry — never from the rank's own
    # callback registry, which is not guaranteed to be identical on every rank
    # (`rank == 0 && add_callback!(...)` is ordinary usage, and `IterationInterval`
    # is pure, so an asymmetric registry desynchronises nothing by itself). Gating
    # the reduction on the local registry instead makes the rank holding the odd
    # callback enter an `Allreduce` alone. Starts `true` so a hand-stepped
    # simulation that never enters `run!` reduces rather than risking the split.
    _stop_needs_reduce::Bool
end

_to_ordered(::Nothing, prefix::Symbol) = OrderedDict{Symbol, Any}()
function _to_ordered(items, prefix::Symbol)
    d = OrderedDict{Symbol, Any}()
    if items isa AbstractDict
        for (k, v) in items
            d[Symbol(k)] = v
        end
    elseif items isa NamedTuple
        for k in keys(items)
            d[k] = items[k]
        end
    else
        seq = items isa Tuple ? items :
              items isa AbstractVector ? items : (items,)
        for (i, v) in enumerate(seq)
            d[Symbol(prefix, i)] = v
        end
    end
    return d
end

# ── Default callbacks (Oceananigans semantics): stop conditions live in the
# callback table, not the run! loop. Each sets sim.running = false. ──────────

function stop_time_exceeded(sim)
    if sim.model.clock.time >= sim.stop_time - 1e-15
        @info "Simulation is stopping after reaching stop time ($(prettysummary(sim.stop_time)))."
        sim.running = false
    end
    return nothing
end

function stop_iteration_exceeded(sim)
    if sim.model.clock.iteration >= sim.stop_iteration
        @info "Simulation is stopping after reaching stop iteration $(prettysummary(sim.stop_iteration))."
        sim.running = false
    end
    return nothing
end

function wall_time_limit_exceeded(sim)
    # `_collective_wtime` (api/schedules.jl), not a bare local `time()`: ranks that
    # straddle the limit at an iteration boundary would otherwise leave run! at
    # different iterations and the survivors hang in the next collective — turning
    # the clean wall-limited shutdown (which exists so the final state gets saved)
    # into a scheduler SIGKILL.
    if sim._wall_start > 0 && _collective_wtime(sim) >= sim.wall_time_limit
        @info "Simulation is stopping after exceeding the wall time limit ($(prettytime(sim.wall_time_limit)))."
        sim.running = false
    end
    return nothing
end

function nan_checker(sim)
    r = _health_check(sim.model)
    # `_health_check` scans only THIS rank's modes/radial slab, so the stop
    # decision has to be reduced (as `check_simulation_state_for_nan` in
    # core/simulation_health.jl already does). Without it, only the ranks that own
    # the blown-up modes leave run! and the rest hang in the next collective.
    if _any_rank_flag(r.has_issue)
        if r.has_issue
            @warn "NaN/Inf found in fields $(r.fields) at iteration $(sim.model.clock.iteration); stopping simulation."
        else
            @warn "NaN/Inf found on another rank at iteration $(sim.model.clock.iteration); stopping simulation."
        end
        sim.running = false
    end
    return nothing
end

function _default_callbacks()
    OrderedDict{Symbol, Any}(
        :stop_time_exceeded      => Callback(stop_time_exceeded,      IterationInterval(1)),
        :stop_iteration_exceeded => Callback(stop_iteration_exceeded, IterationInterval(1)),
        :wall_time_limit_exceeded => Callback(wall_time_limit_exceeded, IterationInterval(1)),
        :nan_checker             => Callback(nan_checker,             IterationInterval(100)),
    )
end

"""
    Simulation(model::GeodynamoModel;
               Δt, stop_time=Inf, stop_iteration=typemax(Int),
               timestepper, timestep_scheme, implicit_theta,
               etd_krylov_dimension, krylov_tolerance,
               callbacks=[], output_writers=[],
               gpu=:auto, restart_from="")

Construct a `Simulation`.

The supported timestepper objects are `CNAB2()`,
`ExponentialRungeKutta2()`, and `RungeKutta3()`. Experimental descriptor types
such as `ExponentialAdamsBashforth2`, `ETD`, and `ThetaMethod` are retained for
compatibility but rejected by solver parameter validation.

A positive timestep is required: pass it as `Δt` (canonical, Oceananigans convention) or `dt` (alias);
passing both, neither, or a non-positive value throws an `ArgumentError`.
`stop_time` accepts any `Real` and is converted to `Float64`.

`gpu` selects the dense device-state stepping path (`gpu_solver_step!` with a
cached bundle): `:auto` (default) uses it when the model lives on a GPU
architecture, `true` forces it (also valid on the CPU/Array backend), `false`
disables it. The path supports CNAB2, ExponentialRungeKutta2, and RungeKutta3
(insulating magnetic) — other
configurations warn once and use the standard CPU stepping. The first step
(and the first after a `Δt` change) runs on the CPU to bootstrap the CNAB2
history — the device state is mirrored back to the host first, so a `Δt` change
never discards device progress.

`gpu_sync` controls how often the host `SolverState` mirror is refreshed:
`:every` (default) syncs after every device step; `:output` syncs lazily, only on
iterations where an output writer or a field-reading callback actually fires (and
unconditionally at the end of `run!`). Both give identical results; `:output` just
skips device→host copies nothing would have read.

Schedules that can be evaluated without side effects — `IterationInterval` — are
pre-queried exactly, so a writer on `IterationInterval(500)` costs one sync per 500
steps rather than one per step. `TimeInterval`, `WallTimeInterval`, and
`SpecifiedTimes` advance internal bookkeeping when queried, so they cannot be tested
in advance and conservatively force a sync every step; use `IterationInterval`
schedules to get the benefit of `:output`.

If `restart_from` is a non-empty string it is treated as a restart directory
passed to `read_restart!` (via the legacy `TimeTracker`-based interface).
The underlying `read_restart!` requires a `TimeTracker`, `OutputConfig`, and an
initialized MPI environment.  Restart is **fail-loud**: if MPI is not
initialized, or the restart read fails, construction throws rather than silently
starting from the initial state (which would mask data loss).
"""
function Simulation(model::GeodynamoModel;
        dt::Union{Real, Nothing} = nothing,
        Δt::Union{Real, Nothing} = nothing,
        stop_time::Real = Inf,
        stop_iteration::Int = typemax(Int),
        wall_time_limit::Real = Inf,
        timestepper = nothing,
        timestep_scheme::Union{Symbol, Nothing} = nothing,
        implicit_theta::Union{Real, Nothing} = nothing,
        etd_krylov_dimension::Union{Int, Nothing} = nothing,
        krylov_tolerance::Union{Real, Nothing} = nothing,
        courant::Union{Real, Nothing} = nothing,
        callbacks = (),
        output_writers = (),
        gpu = :auto,
        gpu_sync::Symbol = :every,
        restart_from::String = "")
    # Resolve the timestep: `Δt` is canonical (Oceananigans convention); `dt` is the ASCII alias.
    if dt !== nothing && Δt !== nothing
        throw(ArgumentError("Simulation: pass either `Δt` or `dt`, not both"))
    end
    dt_in = dt !== nothing ? dt : Δt
    dt_in === nothing &&
        throw(ArgumentError("Simulation: a timestep is required (pass `Δt=` or `dt=`)"))
    dt_in > 0 ||
        throw(ArgumentError("Simulation: dt = $dt_in must be positive"))
    stop_time_f = Float64(stop_time)
    restored_output_tracker = nothing

    if !isempty(restart_from)
        if MPI.Initialized()
            config = default_config()
            tracker = create_time_tracker(config)
            try
                restart_data,
                metadata = read_restart!(tracker, restart_from,
                    model.state.time, config,
                    model.state.runtime.shtns_config.pencils;
                    shtns_config = model.state.runtime.shtns_config)
                restore_fields_from_restart!(model.state, restart_data)
                restart_time = Float64(get(metadata, "current_time", model.state.time))
                restart_step = Int(get(metadata, "current_step", model.state.step))
                reset_solver_clock!(model.state; time = restart_time, step = restart_step)
                restored_output_tracker = tracker
                @info "Simulation: loaded restart from $restart_from" time=model.state.time
            catch e
                # Fail loud: the caller explicitly asked to restart, so silently
                # starting from the initial state would hide data loss.
                throw(ErrorException(
                    "Simulation: failed to read restart from \"$restart_from\": $e"))
            end
        else
            throw(ErrorException(
                "Simulation: restart_from=\"$restart_from\" requires MPI to be " *
                "initialized (the restart reader is collective); call MPI.Init() first"))
        end
    end

    dt_f = Float64(dt_in)

    # Propagate dt, stop_time, and stop_iteration into the solver's SolverParameters so
    # that solver_step! uses the timestep the caller requested.
    p = model.state.parameters
    old_timestep = model.state.parameters.timestep
    timestep_options = _resolve_timestepper(
        timestepper,
        timestep_scheme,
        implicit_theta,
        etd_krylov_dimension,
        krylov_tolerance,
        p
    )
    new_params = SolverParameters(;
        (f => getfield(p, f) for f in fieldnames(SolverParameters))...,
        timestep = dt_f,
        end_time = stop_time_f,
        stop_iteration = stop_iteration,
        timestepper = timestep_options.timestepper,
        courant = Float64(something(courant, p.courant))
    )
    # Validate the run controls the caller just set (courant, stop_iteration,
    # stop_time/end_time) instead of silently accepting invalid values that
    # would only surface later. Use the non-printing checker so a normal
    # construction does not spam validation warnings. The model's own params
    # were already valid.
    control_errors, _ = _parameter_errors_warnings(new_params)
    isempty(control_errors) || throw(ArgumentError(
        "Simulation: invalid run controls: " * join(control_errors, "; ")))
    model.state.parameters = new_params
    if dt_f != old_timestep
        rebuild_solver_implicit_matrices!(model.state, dt_f)
        model.state.runtime.timestep_state.dt = dt_f
    end

    callback_items = merge(_default_callbacks(), _to_ordered(callbacks, :callback))
    output_writer_items = _to_ordered(output_writers, :writer)
    if restored_output_tracker !== nothing
        _restore_output_writer_trackers!(
            output_writer_items, restored_output_tracker, model.state.parameters.geometry)
    end

    gpu_resolved = _resolve_gpu_stepping(gpu, model, timestep_options.timestepper)
    gpu_sync in (:every, :output) || throw(ArgumentError(
        "Simulation: gpu_sync must be :every or :output (got $gpu_sync)"))

    sync_clock!(model.clock, model.state)
    return Simulation{typeof(model), typeof(callback_items), typeof(output_writer_items)}(
        model, dt_f, stop_time_f, stop_iteration, Float64(wall_time_limit),
        false,
        gpu_resolved,
        callback_items,
        output_writer_items,
        0.0,
        nothing,
        nothing,
        0.0,
        gpu_sync,
        false,
        (nothing, nothing),
        (nothing, nothing),
        true
    )
end

# Resolve the `gpu` Simulation kwarg: `:auto` enables the dense device-state
# stepping path when the model lives on a GPU architecture; `true` forces it
# (also useful on the CPU/Array backend); `false` disables it. The device path
# supports CNAB2, ExponentialRungeKutta2, and RungeKutta3 (insulating magnetic) — anything else warns and
# falls back to the standard CPU stepping.
function _resolve_gpu_stepping(gpu, model, timestepper)
    resolved = if gpu === :auto
        model.state.backend.architecture isa GPU
    elseif gpu isa Bool
        gpu
    else
        throw(ArgumentError("Simulation: gpu must be true, false, or :auto (got $gpu)"))
    end
    resolved || return false
    if !(timestepper isa CNAB2 || timestepper isa ExponentialRungeKutta2 || timestepper isa RungeKutta3)
        @warn "Simulation: the GPU stepping path supports CNAB2, ExponentialRungeKutta2, and RungeKutta3 only; " *
              "using the CPU path" timestepper
        return false
    end
    nprocs = get_nprocs()
    if nprocs > 1
        @warn "Simulation: the GPU stepping path is single-rank only (the device bundle " *
              "is a whole-domain dense copy with no pencil awareness); using the CPU path" nprocs
        return false
    end
    p = model.state.parameters
    # The remaining `build_gpu_solver_state` scope limits (device_state.jl:368/373/378)
    # are hard `error()`s, so they must be pre-screened here too. Under the default
    # gpu = :auto on a CUDA machine the user never asked for the device path, and
    # every one of these configurations runs fine on the CPU — so decline the GPU
    # with a warning, exactly like the three limits below, instead of letting the
    # run die at iteration 1 inside `_gpu_time_step!`.
    if p.geometry !== :shell
        @warn "Simulation: the GPU stepping path supports only :shell geometry " *
              "(the device kernels hard-code the shell poloidal recovery and BC rows); " *
              "using the CPU path" geometry = p.geometry
        return false
    end
    if model.state.topography.config.enabled
        @warn "Simulation: topography coupling is enabled but the GPU step path does " *
              "not apply it (no GPU port of apply_solver_topography!); using the CPU path"
        return false
    end
    let st = model.state
        tdep = String[]
        for (name, field) in (("temperature", st.fields.temperature),
            ("velocity", st.fields.velocity),
            ("magnetic", st.fields.magnetic),
            ("composition", st.fields.composition))
            _gpu_field_has_time_dependent_bc(field) && push!(tdep, name)
        end
        if !isempty(tdep)
            @warn "Simulation: the GPU stepping path bakes boundary endpoint values at " *
                  "pack time, so time-dependent boundary data would be silently frozen; " *
                  "using the CPU path" fields = tdep
            return false
        end
    end
    if p.include_magnetic && p.magnetic_inner_bc !== :insulating
        # A conducting inner core IS supported on the device, but only under CNAB2:
        # gpu_magnetic_field_step! takes the packed admittance via its `ic` argument,
        # whereas the ERK2 and RungeKutta3 device steps run their own magnetic update
        # with no inner-core hook.
        if p.magnetic_inner_bc === :conducting_inner_core && timestepper isa CNAB2
            return true
        end
        @warn "Simulation: the GPU stepping path supports an insulating magnetic inner " *
              "core, or a conducting one under CNAB2; using the CPU path" magnetic_inner_bc = p.magnetic_inner_bc timestepper
        return false
    end
    return true
end

# ================================================================================
# time_step!
# ================================================================================

"""
    time_step!(model::GeodynamoModel, dt)

Advance the model by one step with timestep `dt`, then sync the clock.
"""
function time_step!(model::GeodynamoModel, dt::Real)
    dt > 0 || throw(ArgumentError("time_step!: dt = $dt must be positive"))
    state = model.state
    dt_f = Float64(dt)
    if dt_f != state.parameters.timestep
        p = state.parameters
        state.parameters = SolverParameters(;
            (f => getfield(p, f) for f in fieldnames(SolverParameters))...,
            timestep = dt_f
        )
        rebuild_solver_implicit_matrices!(state, dt_f)
        state.runtime.timestep_state.dt = dt_f
    end
    solver_step!(state)
    sync_clock!(model.clock, state)
    model.clock.last_dt = dt_f
    return model
end

"""
    time_step!(sim::Simulation)

Advance the simulation by one step at `sim.dt`, firing callbacks and writers.
"""
function time_step!(sim::Simulation)
    if sim.gpu
        _gpu_time_step!(sim)
    else
        time_step!(sim.model, sim.dt)
    end
    _run_callbacks!(sim)
    _run_output_writers!(sim)
    return sim
end

# One step on the dense device-state path (gpu_solver_step!) with the bundle
# cached on the Simulation. The FIRST step (and the first step after a dt
# change) runs on the CPU instead: it bootstraps the CNAB2 history exactly as
# the reference path does, after which the device bundle is built from the
# warmed state (the packed implicit matrices bake dt in, hence the rebuild).
# After every device step the spectral state is synced back to the host
# SolverState so callbacks, writers, diagnostics, and the clock stay live.
function _gpu_time_step!(sim::Simulation)
    model = sim.model
    state = model.state
    if sim._gpu_state === nothing || sim._gpu_dt != sim.dt
        # The bootstrap step below runs on the HOST state, so under
        # gpu_sync = :output the device must be mirrored back first — otherwise
        # every step since the last sync is overwritten and the fields silently
        # rewind while the clock keeps counting.
        _gpu_sync_host!(sim)
        time_step!(model, sim.dt)               # CPU bootstrap step
        gst = build_gpu_solver_state(state)
        erk = state.parameters.timestepper isa ExponentialRungeKutta2 ? build_gpu_erk2_state(state) : nothing
        arch = state.backend.architecture
        if !(arch isa CPU)
            gst = gpu_to_device(gst, arch)
            erk === nothing || (erk = gpu_to_device(erk, arch))
        end
        sim._gpu_state = gst
        sim._gpu_erk2 = erk
        sim._gpu_dt = sim.dt
        sim._gpu_src = _gpu_source_snapshot(state)
        sim._gpu_bc = _gpu_bc_snapshot(state)
        return sim
    end
    # The bundle bakes boundary endpoint values and the internal-source profiles
    # at BUILD time and is only rebuilt on a Δt change, so the scope limits have
    # to be re-checked here too — otherwise attaching time-dependent boundary data
    # (or calling set_internal_heating!) mid-run is silently ignored while the
    # builder docstring promises a loud rejection.
    _gpu_assert_bundle_current(sim)
    # Match the CPU convention: solver_step! (solver/mainloop.jl) bumps
    # timestep_state.step BEFORE the physics, so anything reading it during a step
    # sees the step being computed. Nothing on the device path reads it today —
    # this keeps a future CPU-side hook inside the device loop from inheriting an
    # off-by-one. `reset_solver_clock!` below sets the authoritative pair.
    state.runtime.timestep_state.step = state.step + 1
    # Three-way device-step dispatch; each gpu_*_solver_step! call is itself a
    # dispatch barrier over the ::Any bundle. The host re-sync is deferred to
    # the gpu_sync block below (PR #77 lazy mirror) — no unconditional sync here.
    if state.parameters.timestepper isa ExponentialRungeKutta2
        gpu_erk2_solver_step!(sim._gpu_state, sim._gpu_erk2)
    elseif state.parameters.timestepper isa RungeKutta3
        gpu_cb3_solver_step!(sim._gpu_state)
    else
        gpu_solver_step!(sim._gpu_state)
    end
    # Advance BOTH clocks. `runtime.timestep_state` is what get_current_simulation_time
    # (bcs/integration.jl) and the ERK2 diagnostics read; leaving it at the bootstrap
    # value makes them see a frozen time for the whole device run.
    reset_solver_clock!(state; time = state.time + sim.dt, step = state.step + 1)
    sync_clock!(model.clock, state)        # counters are host-side — no device read
    model.clock.last_dt = sim.dt
    if sim._gpu_sync === :every || _gpu_host_read_pending(sim)
        sync_gpu_state_to_cpu!(state, sim._gpu_state)
        sim._gpu_dirty = false
    else
        sim._gpu_dirty = true
    end
    return sim
end

# Whether `schedule` fires at `iteration`, answered WITHOUT consuming the firing.
# `IterationInterval` is the only pure `should_fire` (schedules.jl: `step % interval`);
# TimeInterval / WallTimeInterval / SpecifiedTimes all mutate their fire bookkeeping,
# so pre-querying them would swallow the real firing. Those answer "yes" and the
# caller stays conservative.
_gpu_schedule_may_fire(s::IterationInterval, iteration::Int) =
    s.interval > 0 && iteration % s.interval == 0
_gpu_schedule_may_fire(::Any, ::Int) = true             # stateful/unrecognised: assume it reads

# Under gpu_sync = :output the host mirror is refreshed only when something will
# actually read it this iteration: an output writer or a callback that FIRES now.
# Attachment alone is not a read — a writer on IterationInterval(1000) must not
# force a device→host copy on the other 999 steps, or `:output` would collapse
# into `:every` for every run that writes output at all.
#
# The three stop conditions read the clock, never the fields, so they are skipped
# whatever their schedule. Everything else (including nan_checker, which does read
# the fields via _health_check) is judged by its own registered schedule — never by
# function identity, so re-registering nan_checker at a different interval keeps
# working.
"""
    ClockOnlyCallback(func)

Marks `func` as reading only the clock, never the field data.

Under `gpu_sync = :output` the host mirror is refreshed on any iteration where something
will read it, so a callback that merely inspects `clock.time`/`clock.iteration` — a custom
stop condition, a progress print — would otherwise force a device→host copy every step and
collapse `:output` into `:every`. Wrap it to opt out:

    add_callback!(sim, ClockOnlyCallback(my_stop); schedule = IterationInterval(1))

Calling a `ClockOnlyCallback` forwards to `func`, so it behaves exactly like the bare
function everywhere else.
"""
struct ClockOnlyCallback{F}
    func::F
end

(c::ClockOnlyCallback)(sim) = c.func(sim)

"""
    _gpu_clock_only(func) -> Bool

Whether `func` reads only the clock. `false` by default: an unrecognised callback is
assumed to read the fields, which is the conservative answer (an unnecessary sync, never a
stale read). The three built-in stop conditions dispatch to `true`, and anything wrapped
in [`ClockOnlyCallback`](@ref) does too — that wrapper is what makes the exclusion
extensible instead of a closed identity test.
"""
_gpu_clock_only(::Any) = false
_gpu_clock_only(::typeof(stop_time_exceeded)) = true
_gpu_clock_only(::typeof(stop_iteration_exceeded)) = true
_gpu_clock_only(::typeof(wall_time_limit_exceeded)) = true
_gpu_clock_only(::ClockOnlyCallback) = true

# Which stop conditions leave `sim.running` rank-identical, so that
# `_run_callbacks!` can skip its reconciling Allreduce (api/callbacks.jl).
# The three built-ins decide from rank-consistent inputs: `clock.time` and
# `clock.iteration` advance in lockstep, and `wall_time_limit_exceeded` reads
# `_collective_wtime`, which broadcasts rank 0's elapsed time. `nan_checker`
# scans only this rank's slab but already reduces the verdict with
# `_any_rank_flag` before assigning.
#
# `ClockOnlyCallback` is deliberately NOT listed. Its contract (see its docstring) is
# "reads only the clock, never the field data" — a GPU-sync property, not an MPI one, and
# the two are not the same claim. A custom stop condition built on the WALL clock,
# `ClockOnlyCallback(s -> (time() - t0 > budget) && (s.running = false))`, satisfies the
# documented contract exactly while deciding from rank-local data; declaring it symmetric
# here would skip the reconciling reduction and let ranks leave `run!` on different
# iterations — the deadlock this whole trait exists to prevent. Anything not listed pays
# one Allreduce per step, which is the cheap side of that trade.
_running_flag_rank_symmetric(::typeof(stop_time_exceeded)) = true
_running_flag_rank_symmetric(::typeof(stop_iteration_exceeded)) = true
_running_flag_rank_symmetric(::typeof(wall_time_limit_exceeded)) = true
_running_flag_rank_symmetric(::typeof(nan_checker)) = true

function _gpu_host_read_pending(sim::Simulation)
    iteration = sim.model.clock.iteration
    for ow in values(sim.output_writers)
        sched = hasproperty(ow, :schedule) ? ow.schedule : nothing
        _gpu_schedule_may_fire(sched, iteration) && return true
    end
    for cb in values(sim.callbacks)
        # Clock-only callbacks read the clock, never the fields, so they never force a
        # mirror. Decided by the `_gpu_clock_only` TRAIT, not by function identity:
        # identity is the mechanism the comment above says was abandoned, and it cannot
        # see through a closure, a partial, or a user's own stop condition — each of
        # which silently collapsed `:output` back into `:every`.
        if cb isa Callback && _gpu_clock_only(cb.func)
            continue
        end
        # Every registered callback type (Callback, EnergyDiagnostics,
        # SolenoidalMonitor, SimulationProgress, HealthCheck) carries `.schedule`;
        # anything else falls back to the conservative answer rather than throwing
        # here — _run_callbacks! reports the unknown type with a better message.
        sched = hasproperty(cb, :schedule) ? cb.schedule : nothing
        _gpu_schedule_may_fire(sched, iteration) && return true
    end
    return false
end

# Bring the host SolverState up to date with the device bundle (no-op when
# already synced or when not on the GPU path).
function _gpu_sync_host!(sim::Simulation)
    if sim.gpu && sim._gpu_dirty && sim._gpu_state !== nothing
        sync_gpu_state_to_cpu!(sim.model.state, sim._gpu_state)
        sim._gpu_dirty = false
    end
    return sim
end

"""
    sync_gpu_host!(sim::Simulation) -> sim

Bring the host `SolverState` up to date with the device bundle.

Only does work under `gpu_sync = :output`, which deliberately leaves the host
mirror behind the device on steps where nothing reads it. `run!` flushes the
mirror when it returns, so this is for code driving [`time_step!`](@ref)
directly and wanting to read fields, energies, or diagnostics mid-run.

A no-op — not an error — when the mirror is already current, when no device
bundle has been built yet, or on the CPU path.
"""
sync_gpu_host!(sim::Simulation) = _gpu_sync_host!(sim)

# Host-side copies of the internal-source profiles baked into the device bundle,
# in the order (temperature, composition). `nothing` for a disabled field or one
# with no source configured (which is what `sbundle` packs as `internal_source`).
function _gpu_source_snapshot(st)
    snap(field) = begin
        field === nothing && return nothing
        hasproperty(field, :internal_sources) || return nothing
        src = field.internal_sources
        all(iszero, src) ? nothing : collect(Float64, src)
    end
    return (snap(st.fields.temperature), snap(st.fields.composition))
end

# Host-side copies of the scalar boundary endpoint values baked into the device bundle,
# in the order (temperature, composition). `sbundle` (gpu/device_state.jl) copies
# `get_bc_vectors(field)` — i.e. `field.boundary_values` — at pack time, and
# `gpu_implicit_solve_field!` re-plants those copies on every device step, whereas the
# CPU path re-reads the live view each step. Only the scalar fields are snapshotted
# because only `sbundle` bakes BC values; `vbundle` (velocity/magnetic) packs zeros.
function _gpu_bc_snapshot(st)
    snap(field) = begin
        field === nothing && return nothing
        hasproperty(field, :boundary_values) || return nothing
        Matrix{Float64}(field.boundary_values)
    end
    return (snap(st.fields.temperature), snap(st.fields.composition))
end

# Whether a field's boundary endpoint values still match what was baked.
function _gpu_bc_current(field, baked)
    field === nothing && return true
    hasproperty(field, :boundary_values) || return true
    baked === nothing && return true
    cur = field.boundary_values
    size(cur) == size(baked) || return false
    @inbounds for i in eachindex(baked)
        cur[i] == baked[i] || return false
    end
    return true
end

# Whether a field's internal-source profile still matches what was baked. `baked
# === nothing` means the bundle packed no source, so the field must still be all
# zero for the device step to reproduce the CPU one.
function _gpu_source_current(field, baked)
    field === nothing && return true
    hasproperty(field, :internal_sources) || return true
    cur = field.internal_sources
    baked === nothing && return all(iszero, cur)
    length(cur) == length(baked) || return false
    @inbounds for i in eachindex(baked)
        cur[i] == baked[i] || return false
    end
    return true
end

# Re-assert the build-time scope limits against the CURRENT host state. Cheap:
# four property lookups plus at most two nr-length scans per step.
function _gpu_assert_bundle_current(sim::Simulation)
    state = sim.model.state
    _gpu_assert_static_bcs(state)
    baked_t, baked_c = sim._gpu_src
    _gpu_source_current(state.fields.temperature, baked_t) || error(
        "GPU solver path: the temperature internal-source profile changed after the " *
        "device bundle was built (e.g. set_internal_heating!). The bundle bakes the " *
        "profile at pack time and is only rebuilt on a Δt change, so the device would " *
        "keep integrating the old source. Rebuild the Simulation, or change the source " *
        "before the first step.")
    _gpu_source_current(state.fields.composition, baked_c) || error(
        "GPU solver path: the composition internal-source profile changed after the " *
        "device bundle was built. The bundle bakes the profile at pack time and is only " *
        "rebuilt on a Δt change, so the device would keep integrating the old source. " *
        "Rebuild the Simulation, or change the source before the first step.")
    # `_gpu_assert_static_bcs` only rejects a time-dependent BoundaryConditionSet; a
    # direct write to `boundary_values` (the programmatic BC surface, or set_*_bcs!)
    # leaves the set static, so the endpoint VALUES have to be diffed as well. Without
    # this the device kept re-planting the wall values baked at build time while the
    # reported parameters and output metadata showed the new ones.
    bc_t, bc_c = sim._gpu_bc
    _gpu_bc_current(state.fields.temperature, bc_t) || error(
        "GPU solver path: the temperature boundary endpoint values changed after the " *
        "device bundle was built. The bundle bakes `boundary_values` at pack time and is " *
        "only rebuilt on a Δt change, so the device would keep applying the old wall " *
        "values. Rebuild the Simulation, or change the boundary values before the first step.")
    _gpu_bc_current(state.fields.composition, bc_c) || error(
        "GPU solver path: the composition boundary endpoint values changed after the " *
        "device bundle was built. The bundle bakes `boundary_values` at pack time and is " *
        "only rebuilt on a Δt change, so the device would keep applying the old wall " *
        "values. Rebuild the Simulation, or change the boundary values before the first step.")
    return nothing
end

# ================================================================================
# run!
# ================================================================================

"""
    run!(sim::Simulation)

Advance the simulation until a stop criterion fires.  Stop conditions
(`stop_time`, `stop_iteration`, `wall_time_limit`, NaN detection) are enforced
by the default callbacks registered at construction; any callback may halt the
run by setting `sim.running = false`.  Callbacks (including stop conditions) fire
after each step.
"""
function run!(sim::Simulation)
    sim._wall_start = time()
    sim.running = true
    # Decide the per-step stop reduction here, where EVERY rank arrives, rather
    # than inside `_run_callbacks!` from the rank's own registry. The reduction
    # itself is what must be unanimous: one rank holding an unrecognised callback
    # arms it for all of them, and the default registry (every built-in stop is
    # rank-symmetric) disarms it for all of them, so a default run still pays no
    # per-step `Allreduce`. Callbacks registered after this line do not re-arm it;
    # `add_callback!` mid-`run!` is not supported for exactly this reason.
    sim._stop_needs_reduce = _any_rank_flag(_callbacks_may_stop_rank_locally(sim.callbacks))
    # A simulation already past its stop criteria must not take a step
    # (e.g. a second run! after completion). Check the stop conditions
    # directly rather than via _run_callbacks! so user callbacks do not
    # fire an extra time before the first step.
    stop_time_exceeded(sim)
    stop_iteration_exceeded(sim)
    wall_time_limit_exceeded(sim)
    while sim.running
        time_step!(sim)
    end
    _gpu_sync_host!(sim)        # lazy gpu_sync = :output: final state to host
    return sim
end

"""
    add_callback!(sim, func; schedule, name=auto)

Register `func(sim)` to fire on `schedule`. Returns `sim`.
"""
function add_callback!(sim::Simulation, func; schedule,
        name::Symbol = Symbol(:callback, length(sim.callbacks) + 1))
    sim.callbacks[name] = Callback(func; schedule = schedule)
    return sim
end

# ================================================================================
# Oceananigans-canonical `Δt` property
# (api-layer exception to the ASCII policy, approved 2026-06-11).
# The struct field stays ASCII `dt`; `Δt` is a virtual alias.
# ================================================================================

function Base.getproperty(sim::Simulation, name::Symbol)
    name === :Δt && return getfield(sim, :dt)
    return getfield(sim, name)
end

function Base.setproperty!(sim::Simulation, name::Symbol, x)
    name === :Δt && return setfield!(sim, :dt, Float64(x))
    return setfield!(sim, name, x)
end

Base.propertynames(sim::Simulation) = (fieldnames(Simulation)..., :Δt)
