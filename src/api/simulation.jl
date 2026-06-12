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
    if sim._wall_start > 0 && (time() - sim._wall_start) >= sim.wall_time_limit
        @info "Simulation is stopping after exceeding the wall time limit ($(prettytime(sim.wall_time_limit)))."
        sim.running = false
    end
    return nothing
end

function nan_checker(sim)
    r = _health_check(sim.model)
    if r.has_issue
        @warn "NaN/Inf found in fields $(r.fields) at iteration $(sim.model.clock.iteration); stopping simulation."
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
history; the host state is re-synced after every device step so callbacks,
output writers, and the clock stay live.

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
        false
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
    p = model.state.parameters
    if p.include_magnetic && p.magnetic_inner_bc !== :insulating
        @warn "Simulation: the GPU stepping path supports only an insulating magnetic " *
              "inner core; using the CPU path" magnetic_inner_bc = p.magnetic_inner_bc
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
        return sim
    end
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
    state.step += 1
    state.time += sim.dt
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

# Under gpu_sync = :output the host mirror is refreshed only when something
# will actually read it this iteration: any output writer, any callback other
# than the three clock-only stop defaults, or the nan_checker's interval.
# (Schedule state mutates on should_fire, so firing cannot be pre-queried —
# this is a conservative over-approximation.)
function _gpu_host_read_pending(sim::Simulation)
    isempty(sim.output_writers) || return true
    sim.model.clock.iteration % 100 == 0 && return true     # default nan_checker
    for cb in values(sim.callbacks)
        if !(cb isa Callback && (cb.func === stop_time_exceeded ||
             cb.func === stop_iteration_exceeded ||
             cb.func === wall_time_limit_exceeded ||
             cb.func === nan_checker))
            return true
        end
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
