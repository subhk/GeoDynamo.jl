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
#   EnergyDiagnostics  — logs kinetic (and magnetic) energy
#   SolenoidalMonitor  — runs the solver divergence diagnostic, threshold-warns
#   SimulationProgress — logs simulation time / step count
#   HealthCheck        — scans fields for NaN/Inf, optionally aborts
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

Logs the kinetic energy (and magnetic energy, when a magnetic field is present)
at each scheduled interval, computed from the model's spectral coefficients.
"""
struct EnergyDiagnostics{S <: AbstractSchedule}
    schedule::S
end
EnergyDiagnostics(; schedule) = EnergyDiagnostics(schedule)

"""
    SolenoidalMonitor(; schedule, threshold=1e-10)

Runs the solver's solenoidal-constraint diagnostic at each scheduled interval,
logging the velocity (and magnetic) divergence norms and warning when the L∞
divergence exceeds `threshold`.
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

Scans all simulation fields for NaN/Inf at each scheduled interval.  When a
non-finite value is found a warning is logged and, if `abort=true`, the
simulation is halted by throwing.
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
    _energy_diagnostics(model) -> (kinetic, magnetic)

Compute the global kinetic energy (and magnetic energy, if the model carries a
magnetic field) from the current solver state's spectral coefficients.  Returns
a `NamedTuple` of `Float64` energies; `magnetic` is `0.0` when no magnetic field
is present.
"""
function _energy_diagnostics(model)
    state = model.state
    domain = state.backend.outer_core_domain
    kinetic = compute_kinetic_energy(state.fields.velocity, domain)
    magnetic = state.fields.magnetic === nothing ? 0.0 :
               compute_magnetic_energy(state.fields.magnetic, domain)
    return (kinetic = kinetic, magnetic = magnetic)
end

"""
    _fire_callback!(cb::EnergyDiagnostics, sim)

Compute and log the current kinetic (and magnetic) energy of the model from its
spectral toroidal-poloidal coefficients.
"""
function _fire_callback!(cb::EnergyDiagnostics, sim)
    e = _energy_diagnostics(sim.model)
    @info "EnergyDiagnostics" step=sim.model.clock.iteration time=sim.model.clock.time kinetic_energy=e.kinetic magnetic_energy=e.magnetic
    return nothing
end

"""
    _solenoidal_diagnostics(model) -> (velocity_l2, velocity_linf, magnetic_l2, magnetic_linf)

Run the solver's solenoidal-constraint diagnostic against the current state and
return the freshly recorded divergence norms for the velocity (and magnetic)
fields.  This records one sample into `model.state.solenoidal_monitor`.

NOTE: the divergence norms come from `compute_divergence_spectral`, which is
currently a deferred solver stub returning zero; the toroidal-poloidal
representation is solenoidal by construction so this metric is ~0 until that
primitive is implemented.  The callback itself is fully wired: it runs the real
diagnostic pipeline and threshold-checks the recorded values.
"""
function _solenoidal_diagnostics(model)
    state = model.state
    check_solenoidal_constraint!(state)
    mon = state.solenoidal_monitor
    last_or_zero(v) = isempty(v) ? 0.0 : v[end]
    return (velocity_l2 = last_or_zero(mon.velocity_div_l2),
        velocity_linf = last_or_zero(mon.velocity_div_linf),
        magnetic_l2 = last_or_zero(mon.magnetic_div_l2),
        magnetic_linf = last_or_zero(mon.magnetic_div_linf))
end

"""
    _fire_callback!(cb::SolenoidalMonitor, sim)

Run the solver's solenoidal-constraint diagnostic and log the divergence norms,
warning when the velocity (or magnetic) L∞ divergence exceeds `cb.threshold`.
"""
function _fire_callback!(cb::SolenoidalMonitor, sim)
    d = _solenoidal_diagnostics(sim.model)
    @info "SolenoidalMonitor" step=sim.model.clock.iteration time=sim.model.clock.time velocity_div_linf=d.velocity_linf magnetic_div_linf=d.magnetic_linf threshold=cb.threshold
    if d.velocity_linf > cb.threshold
        @warn "SolenoidalMonitor: velocity divergence exceeds threshold" linf=d.velocity_linf threshold=cb.threshold
    end
    if d.magnetic_linf > cb.threshold
        @warn "SolenoidalMonitor: magnetic divergence exceeds threshold" linf=d.magnetic_linf threshold=cb.threshold
    end
    return nothing
end

"""
    _health_check(model) -> (has_issue, fields)

Scan every spectral field of the model (velocity, temperature, and magnetic /
composition when present) for NaN/Inf values.  Returns a `NamedTuple` with
`has_issue::Bool` and `fields::Vector{String}` naming the offending components.
"""
function _health_check(model)
    state = model.state
    issues = String[]
    scan!(field, name) = begin
        any(!isfinite, parent(field.data_real)) && push!(issues, "$(name)_real")
        any(!isfinite, parent(field.data_imag)) && push!(issues, "$(name)_imag")
        return nothing
    end

    scan!(state.fields.velocity.toroidal, "velocity_toroidal")
    scan!(state.fields.velocity.poloidal, "velocity_poloidal")
    scan!(state.fields.temperature.spectral, "temperature")
    if state.fields.magnetic !== nothing
        scan!(state.fields.magnetic.toroidal, "magnetic_toroidal")
        scan!(state.fields.magnetic.poloidal, "magnetic_poloidal")
    end
    if state.fields.composition !== nothing
        scan!(state.fields.composition.spectral, "composition")
    end

    return (has_issue = !isempty(issues), fields = issues)
end

"""
    _fire_callback!(cb::HealthCheck, sim)

Scan all fields for NaN/Inf.  If any are found, log a warning; when `cb.abort`
is true, throw to halt the simulation.
"""
function _fire_callback!(cb::HealthCheck, sim)
    r = _health_check(sim.model)
    # `_health_check` is rank-LOCAL, so `abort` has to be a collective decision:
    # calling error() on only the offending ranks leaves the others in the next
    # collective, which hangs instead of aborting.
    if _any_rank_flag(r.has_issue)
        @warn "HealthCheck: non-finite values detected" step=sim.model.clock.iteration time=sim.model.clock.time fields=r.fields
        if cb.abort
            error("HealthCheck: non-finite values detected in fields $(r.fields) " *
                  "at step $(sim.model.clock.iteration); aborting simulation" *
                  (r.has_issue ? "" : " (detected on another rank)"))
        end
    else
        @info "HealthCheck: all fields finite" step=sim.model.clock.iteration time=sim.model.clock.time
    end
    return nothing
end

# ================================================================================
# _run_callbacks!
# ================================================================================

"""
    _running_flag_rank_symmetric(cb) -> Bool

Whether `cb` can only ever leave `sim.running` IDENTICAL on every rank.

`false` by default: an unrecognised callback is assumed to decide from rank-local
state, which is the conservative answer — a redundant reduction, never a diverged
stop. The built-in callback TYPES qualify because none of them assigns
`sim.running` at all (`HealthCheck` throws instead). The built-in stop FUNCTIONS
qualify for their own reasons and are registered beside their definitions in
api/simulation.jl.

There is deliberately no user-facing opt-in wrapper: rank-symmetry of a stop decision
is not a property a caller can assert by wrapping, and getting it wrong reintroduces the
deadlock. Anything unrecognised simply pays one Allreduce per step.
"""
_running_flag_rank_symmetric(::Any) = false
_running_flag_rank_symmetric(cb::Callback) = _running_flag_rank_symmetric(cb.func)
_running_flag_rank_symmetric(::SimulationProgress) = true
_running_flag_rank_symmetric(::EnergyDiagnostics) = true
_running_flag_rank_symmetric(::SolenoidalMonitor) = true
_running_flag_rank_symmetric(::HealthCheck) = true

"""
    _callbacks_may_stop_rank_locally(callbacks) -> Bool

Whether ANY registered callback could set `sim.running = false` on some ranks and
not others — i.e. whether the reconciling reduction in `_run_callbacks!` can change
the answer at all.
"""
_callbacks_may_stop_rank_locally(callbacks) =
    any(cb -> !_running_flag_rank_symmetric(cb), values(callbacks))

"""
    _run_callbacks!(sim)

Iterates over `sim.callbacks`, builds a `_ScheduleContext` from the current
simulation state, and fires each callback whose schedule returns `true` from
`should_fire`.
"""
function _run_callbacks!(sim)
    # Rank-consistent elapsed time: a WallTimeInterval callback must fire on the
    # same iteration on every rank (see `_collective_wtime`).
    wtime = _collective_wtime(sim)
    ctx = _ScheduleContext(sim.model.clock.time, sim.model.clock.iteration, wtime)
    for cb in values(sim.callbacks)
        if should_fire(_callback_schedule(cb), ctx)
            _fire_callback!(cb, sim)
        end
    end
    # `run!` explicitly allows any callback to stop the simulation by assigning
    # `sim.running = false`. A user callback may base that decision on rank-local
    # state, so reconcile the flag before writers run or another solver step can
    # enter a collective. Built-in rank-local health callbacks use the same
    # reduction internally; this final guard covers the public callback contract.
    #
    # Skipped when every registered callback already leaves the flag rank-identical,
    # so a default run does not pay an Allreduce per step for a value that cannot
    # differ. The skip is decided from the callback REGISTRY, which the surrounding
    # design already requires to be identical on every rank — `should_fire` mutates
    # per-schedule fire bookkeeping, so an asymmetric registry desynchronises firing
    # long before it reaches this line. Unrecognised entries answer "may stop", so a
    # new callback type reduces until it is explicitly declared symmetric.
    if _callbacks_may_stop_rank_locally(sim.callbacks) && _any_rank_flag(!sim.running)
        sim.running = false
    end
    return nothing
end
