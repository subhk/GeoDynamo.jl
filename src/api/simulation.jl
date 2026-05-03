# ================================================================================
# Simulation — top-level driver for the Oceananigans-style API
# ================================================================================

"""
    mutable struct Simulation{M}

Holds a `GeodynamoModel` together with time-stepping controls, callbacks, and
output writers.  Create with `Simulation(model; Δt, ...)` and advance with
`run!(sim)`.
"""
mutable struct Simulation{M}
    model          :: M
    Δt             :: Float64
    stop_time      :: Float64
    max_steps      :: Int
    callbacks      :: Vector{Any}
    output_writers :: Vector{Any}
    step           :: Int
    time           :: Float64
    _wall_start    :: Float64
end

"""
    Simulation(model::GeodynamoModel;
               Δt, stop_time=Inf, max_steps=typemax(Int),
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
        max_steps      :: Int     = typemax(Int),
        timestepper = nothing,
        timestep_scheme :: Union{Symbol, Nothing} = nothing,
        implicit_theta  :: Union{Real, Nothing} = nothing,
        etd_krylov_dimension :: Union{Int, Nothing} = nothing,
        krylov_tolerance :: Union{Real, Nothing} = nothing,
        courant        :: Union{Real, Nothing} = nothing,
        callbacks                 = Any[],
        output_writers            = Any[],
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

    # Propagate Δt, stop_time, and max_steps into the solver's SolverParameters so
    # that advance_solver_step! uses the timestep the caller requested.
    p = model.state.parameters
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
        max_steps = max_steps,
        timestep_scheme = timestep_options.timestep_scheme,
        implicit_theta = timestep_options.implicit_theta,
        etd_krylov_dimension = timestep_options.etd_krylov_dimension,
        krylov_tolerance = timestep_options.krylov_tolerance,
        courant = Float64(something(courant, p.courant)),
    )

    return Simulation{typeof(model)}(
        model, Δt_f, stop_time, max_steps,
        collect(Any, callbacks),
        collect(Any, output_writers),
        model.state.step,
        model.state.time,
        0.0,
    )
end

# ================================================================================
# run!
# ================================================================================

"""
    run!(sim::Simulation)

Advance the simulation until `sim.time >= sim.stop_time` or
`sim.step >= sim.max_steps`, whichever comes first.

Each iteration:
1. Calls `advance_solver_step!(state)` to advance physics by one timestep.
2. Syncs `sim.step` / `sim.time` from the updated solver state.
3. Fires scheduled callbacks via `_run_callbacks!(sim)`.
4. Fires scheduled output writers via `_run_output_writers!(sim)`.
"""
function run!(sim::Simulation)
    sim._wall_start = time()
    state = sim.model.state

    while sim.time < sim.stop_time && sim.step < sim.max_steps
        advance_solver_step!(state)
        sim.step = state.step
        sim.time = state.time
        _run_callbacks!(sim)
        _run_output_writers!(sim)
    end

    return sim
end

# ================================================================================
# show
# ================================================================================

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation)
    println(io, "Simulation")
    println(io, "  model: $(typeof(sim.model))")
    println(io, "  Δt=$(sim.Δt), stop_time=$(sim.stop_time)")
    print(io,   "  step=$(sim.step), time=$(sim.time)")
end
