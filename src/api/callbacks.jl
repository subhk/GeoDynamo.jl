# ================================================================================
# Callback System
# ================================================================================
#
# Provides a composable, schedule-driven callback system for use during
# simulation time-stepping.  Each callback type carries its own schedule
# (an `AbstractSchedule` from schedules.jl) and is fired only when
# `should_fire` returns true for the current `_ScheduleContext`.
#
# Public types
# ------------
#   Callback           — wraps an arbitrary callable
#   EnergyDiagnostics  — logs kinetic energy
#   SolenoidalMonitor  — placeholder for divergence checks
#   SimulationProgress — logs simulation time / step count
#   HealthCheck        — placeholder for NaN/Inf detection
#
# Internal helpers
# ----------------
#   _callback_schedule(cb)      — uniform schedule accessor
#   _fire_callback!(cb, sim)    — dispatch per callback type
#   _run_callbacks!(sim)        — iterate sim.callbacks and fire as needed
#
# ================================================================================

# ================================================================================
# Struct definitions
# ================================================================================

"""
    Callback(f; schedule)
    Callback(f, schedule)

Wraps an arbitrary callable `f` with a firing schedule.  When the callback
fires, `f(sim)` is called with the current `Simulation` object.
"""
struct Callback{F, S <: AbstractSchedule}
    func::F
    schedule::S
end
Callback(f; schedule) = Callback(f, schedule)

"""
    EnergyDiagnostics(; schedule)

Logs kinetic energy at each scheduled interval.  Calls
`compute_kinetic_energy` when the model exposes velocity fields and a
radial domain; otherwise emits a placeholder log message.
"""
struct EnergyDiagnostics{S <: AbstractSchedule}
    schedule::S
end
EnergyDiagnostics(; schedule) = EnergyDiagnostics(schedule)

"""
    SolenoidalMonitor(; schedule, threshold=1e-10)

Checks divergence of the velocity field against `threshold` at each
scheduled interval.  Currently a placeholder — logs a note and returns.
"""
struct SolenoidalMonitor{S <: AbstractSchedule}
    schedule::S
    threshold::Float64
end
SolenoidalMonitor(; schedule, threshold = 1e-10) = SolenoidalMonitor(schedule, threshold)

"""
    SimulationProgress(; schedule)

Logs simulation time and step count at each scheduled interval.
"""
struct SimulationProgress{S <: AbstractSchedule}
    schedule::S
end
SimulationProgress(; schedule) = SimulationProgress(schedule)

"""
    HealthCheck(; schedule, abort=true)

Checks for NaN/Inf in simulation fields at each scheduled interval.
Currently a placeholder — NaN detection wiring is deferred.
If `abort=true` the simulation should be halted upon detecting a NaN.
"""
struct HealthCheck{S <: AbstractSchedule}
    schedule::S
    abort::Bool
end
HealthCheck(; schedule, abort = true) = HealthCheck(schedule, abort)

# ================================================================================
# Schedule accessor
# ================================================================================

"""
    _callback_schedule(cb)

Returns the `AbstractSchedule` associated with any callback object.
"""
_callback_schedule(cb::Callback) = cb.schedule
_callback_schedule(cb::EnergyDiagnostics) = cb.schedule
_callback_schedule(cb::SolenoidalMonitor) = cb.schedule
_callback_schedule(cb::SimulationProgress) = cb.schedule
_callback_schedule(cb::HealthCheck) = cb.schedule
function _callback_schedule(cb)
    error("Unknown callback type $(typeof(cb)). " *
          "Expected one of: Callback, EnergyDiagnostics, SolenoidalMonitor, " *
          "SimulationProgress, HealthCheck.")
end

# ================================================================================
# _fire_callback! implementations
# ================================================================================

"""
    _fire_callback!(cb::Callback, sim)

Calls `cb.func(sim)`.
"""
function _fire_callback!(cb::Callback, sim)
    cb.func(sim)
    return nothing
end

"""
    _fire_callback!(cb::SimulationProgress, sim)

Logs the current simulation time and step number.
"""
function _fire_callback!(cb::SimulationProgress, sim)
    @info "Simulation progress" step=sim.model.clock.iteration time=sim.model.clock.time dt=sim.dt
    return nothing
end

"""
    _fire_callback!(cb::EnergyDiagnostics, sim)

Calls `compute_kinetic_energy` if the model exposes `velocity_fields` and
`domain`; otherwise emits a placeholder info message.
"""
function _fire_callback!(cb::EnergyDiagnostics, sim)
    model = sim.model
    if hasproperty(model, :velocity_fields) && hasproperty(model, :domain)
        ke = compute_kinetic_energy(model.velocity_fields, model.domain)
        @info "EnergyDiagnostics" step=sim.model.clock.iteration time=sim.model.clock.time kinetic_energy=ke
    else
        @info "EnergyDiagnostics (placeholder)" step=sim.model.clock.iteration time=sim.model.clock.time kinetic_energy="unavailable"
    end
    return nothing
end

"""
    _fire_callback!(cb::SolenoidalMonitor, sim)

Placeholder — divergence-checking requires spectral field access that is
not yet wired through the public API.  Logs a note and returns.
"""
function _fire_callback!(cb::SolenoidalMonitor, sim)
    @info "SolenoidalMonitor (placeholder)" step=sim.model.clock.iteration time=sim.model.clock.time threshold=cb.threshold
    return nothing
end

"""
    _fire_callback!(cb::HealthCheck, sim)

Placeholder — NaN/Inf detection requires per-field iteration that is not
yet wired through the public API.  Logs a note and returns.
"""
function _fire_callback!(cb::HealthCheck, sim)
    @info "HealthCheck (placeholder)" step=sim.model.clock.iteration time=sim.model.clock.time abort=cb.abort
    return nothing
end

# ================================================================================
# _run_callbacks!
# ================================================================================

"""
    _run_callbacks!(sim)

Iterates over `sim.callbacks`, builds a `_ScheduleContext` from the current
simulation state, and fires each callback whose schedule returns `true` from
`should_fire`.
"""
function _run_callbacks!(sim)
    wtime = sim._wall_start > 0.0 ? time() - sim._wall_start : 0.0
    ctx = _ScheduleContext(sim.model.clock.time, sim.model.clock.iteration, wtime)
    for cb in values(sim.callbacks)
        if should_fire(_callback_schedule(cb), ctx)
            _fire_callback!(cb, sim)
        end
    end
    return nothing
end
