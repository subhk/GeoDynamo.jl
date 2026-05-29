# ================================================================================
# Simulation — top-level driver for the Oceananigans-style API
# ================================================================================

"""
    mutable struct Simulation{M,C,O}

Holds a `GeodynamoModel` together with time-stepping controls, callbacks, and
output writers.  Create with `Simulation(model; dt, ...)` and advance with
`run!(sim)`.
"""
mutable struct Simulation{M, C, O}
    model::M
    dt::Float64
    stop_time::Float64
    stop_iteration::Int
    wall_time_limit::Float64
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

"""
    Simulation(model::GeodynamoModel;
               dt, stop_time=Inf, stop_iteration=typemax(Int),
               timestepper, timestep_scheme, implicit_theta,
               etd_krylov_dimension, krylov_tolerance,
               callbacks=[], output_writers=[],
               restart_from="")

Construct a `Simulation`.

If `restart_from` is a non-empty string it is treated as a restart directory
passed to `read_restart!` (via the legacy `TimeTracker`-based interface).
Because the underlying `read_restart!` in `io/output_writer.jl` requires a
`TimeTracker`, `OutputConfig`, and MPI, this feature is only active when MPI is
initialized.  A warning is emitted when `restart_from` is set but MPI is not
available.
"""
function Simulation(model::GeodynamoModel;
        dt::Real,
        stop_time::Float64 = Inf,
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
                @warn "Simulation: read_restart! failed — starting from initial state" exception=e
            end
        else
            @warn "Simulation: restart_from set but MPI is not initialized; ignoring restart"
        end
    end

    dt_f = Float64(dt)

    # Propagate dt, stop_time, and stop_iteration into the solver's SolverParameters so
    # that advance_solver_step! uses the timestep the caller requested.
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
    model.state.parameters = SolverParameters(;
        (f => getfield(p, f) for f in fieldnames(SolverParameters))...,
        timestep = dt_f,
        end_time = stop_time,
        stop_iteration = stop_iteration,
        timestepper = timestep_options.timestepper,
        courant = Float64(something(courant, p.courant))
    )
    if dt_f != old_timestep
        rebuild_solver_implicit_matrices!(model.state, dt_f)
        model.state.runtime.timestep_state.dt = dt_f
    end

    callback_items = _to_ordered(callbacks, :callback)
    output_writer_items = _to_ordered(output_writers, :writer)

    sync_clock!(model.clock, model.state)
    return Simulation{typeof(model), typeof(callback_items), typeof(output_writer_items)}(
        model, dt_f, stop_time, stop_iteration, Float64(wall_time_limit),
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
    advance_solver_step!(state)
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

Advance the simulation until `model.clock.time >= sim.stop_time` or
`model.clock.iteration >= sim.stop_iteration`, whichever comes first.

Each iteration calls `time_step!(sim)` which advances physics, syncs the clock,
and fires scheduled callbacks and output writers.
"""
function run!(sim::Simulation)
    sim._wall_start = time()
    clock = sim.model.clock
    while clock.time < sim.stop_time &&
              clock.iteration < sim.stop_iteration &&
              (time() - sim._wall_start) < sim.wall_time_limit
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
