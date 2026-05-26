# ================================================================================
# Simulation — top-level driver for the Oceananigans-style API
# ================================================================================

"""
    mutable struct Simulation{M,C,O}

Holds a `GeodynamoModel` together with time-stepping controls, callbacks, and
output writers.  Create with `Simulation(model; Δt, ...)` and advance with
`run!(sim)`.
"""
mutable struct Simulation{M,C,O}
    model           :: M
    Δt              :: Float64
    stop_time       :: Float64
    stop_iteration  :: Int
    wall_time_limit :: Float64
    callbacks       :: C
    output_writers  :: O
    _wall_start     :: Float64
end

_schedule_items_tuple(::Nothing) = ()
_schedule_items_tuple(items::Tuple) = items
_schedule_items_tuple(items::AbstractVector) = Tuple(items)
_schedule_items_tuple(item) = (item,)

"""
    Simulation(model::GeodynamoModel;
               Δt, stop_time=Inf, stop_iteration=typemax(Int),
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
        Δt             :: Real,
        stop_time      :: Float64 = Inf,
        stop_iteration :: Int     = typemax(Int),
        wall_time_limit :: Real   = Inf,
        timestepper = nothing,
        timestep_scheme :: Union{Symbol, Nothing} = nothing,
        implicit_theta  :: Union{Real, Nothing} = nothing,
        etd_krylov_dimension :: Union{Int, Nothing} = nothing,
        krylov_tolerance :: Union{Real, Nothing} = nothing,
        courant        :: Union{Real, Nothing} = nothing,
        callbacks                 = (),
        output_writers            = (),
        restart_from   :: String  = "")

    if !isempty(restart_from)
        if MPI.Initialized()
            config  = default_config()
            tracker = create_time_tracker(config)
            try
                restart_data, metadata = read_restart!(tracker, restart_from,
                                                       model.state.time, config,
                                                       model.state.runtime.shtns_config.pencils;
                                                       shtns_config=model.state.runtime.shtns_config)
                restore_fields_from_restart!(model.state, restart_data)
                restart_time = Float64(get(metadata, "current_time", model.state.time))
                restart_step = Int(get(metadata, "current_step", model.state.step))
                reset_solver_clock!(model.state; time=restart_time, step=restart_step)
                @info "Simulation: loaded restart from $restart_from" time=model.state.time
            catch e
                @warn "Simulation: read_restart! failed — starting from initial state" exception=e
            end
        else
            @warn "Simulation: restart_from set but MPI is not initialized; ignoring restart"
        end
    end

    Δt_f = Float64(Δt)

    # Propagate Δt, stop_time, and stop_iteration into the solver's SolverParameters so
    # that advance_solver_step! uses the timestep the caller requested.
    p = model.state.parameters
    old_timestep = model.state.parameters.timestep
    timestep_options = _resolve_timestepper(
        timestepper,
        timestep_scheme,
        implicit_theta,
        etd_krylov_dimension,
        krylov_tolerance,
        p,
    )
    model.state.parameters = SolverParameters(;
        (f => getfield(p, f) for f in fieldnames(SolverParameters))...,
        timestep  = Δt_f,
        end_time  = stop_time,
        stop_iteration = stop_iteration,
        timestepper = timestep_options.timestepper,
        courant = Float64(something(courant, p.courant)),
    )
    if Δt_f != old_timestep
        rebuild_solver_implicit_matrices!(model.state, Δt_f)
        model.state.runtime.timestep_state.dt = Δt_f
    end

    callback_items = _schedule_items_tuple(callbacks)
    output_writer_items = _schedule_items_tuple(output_writers)

    sync_clock!(model.clock, model.state)
    return Simulation{typeof(model), typeof(callback_items), typeof(output_writer_items)}(
        model, Δt_f, stop_time, stop_iteration, Float64(wall_time_limit),
        callback_items,
        output_writer_items,
        0.0,
    )
end

# ================================================================================
# time_step!
# ================================================================================

"""
    time_step!(model::GeodynamoModel, Δt)

Advance the model by one step with timestep `Δt`, then sync the clock.
"""
function time_step!(model::GeodynamoModel, Δt::Real)
    state = model.state
    Δt_f = Float64(Δt)
    if Δt_f != state.parameters.timestep
        p = state.parameters
        state.parameters = SolverParameters(;
            (f => getfield(p, f) for f in fieldnames(SolverParameters))...,
            timestep = Δt_f,
        )
        rebuild_solver_implicit_matrices!(state, Δt_f)
        state.runtime.timestep_state.dt = Δt_f
    end
    advance_solver_step!(state)
    sync_clock!(model.clock, state)
    model.clock.last_Δt = Δt_f
    return model
end

"""
    time_step!(sim::Simulation)

Advance the simulation by one step at `sim.Δt`, firing callbacks and writers.
"""
function time_step!(sim::Simulation)
    time_step!(sim.model, sim.Δt)
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

# ================================================================================
# show
# ================================================================================

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation)
    c = sim.model.clock
    println(io, "Simulation")
    println(io, "  model: $(typeof(sim.model))")
    println(io, "  Δt=$(sim.Δt), stop_time=$(sim.stop_time), stop_iteration=$(sim.stop_iteration)")
    print(io,   "  step=$(c.iteration), time=$(c.time)")
end
