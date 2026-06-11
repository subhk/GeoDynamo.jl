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
    callbacks::C
    output_writers::O
    _wall_start::Float64
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
               restart_from="")

Construct a `Simulation`.

A positive timestep is required: pass it as `Δt` (canonical, Oceananigans convention) or `dt` (alias);
passing both, neither, or a non-positive value throws an `ArgumentError`.
`stop_time` accepts any `Real` and is converted to `Float64`.

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

    sync_clock!(model.clock, model.state)
    return Simulation{typeof(model), typeof(callback_items), typeof(output_writer_items)}(
        model, dt_f, stop_time_f, stop_iteration, Float64(wall_time_limit),
        false,
        callback_items,
        output_writer_items,
        0.0
    )
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
    time_step!(sim.model, sim.dt)
    _run_callbacks!(sim)
    _run_output_writers!(sim)
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
    while sim.running
        time_step!(sim)
    end
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
