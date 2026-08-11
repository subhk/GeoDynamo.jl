"""
    reset_solver_clock!(state; time, step)

Synchronize the public solver clock and the runtime `TimestepState`.

Use this when initializing or rewinding a solver state so diagnostics,
callbacks, and field views all see the same time/step pair.
"""
function reset_solver_clock!(state::SolverState; time::Float64, step::Int)
    state.runtime.timestep_state.time = time
    state.runtime.timestep_state.step = step
    state.time = time
    state.step = step
    return state
end

"""
    _solver_uses_two_step_history(timestepper)

Return `true` for timestep schemes that need previous nonlinear terms.
"""
function _solver_uses_two_step_history(timestepper)
    timestepper isa CNAB2 ||
        timestepper isa ExponentialAdamsBashforth2 ||
        timestepper isa ExponentialRungeKutta2
end

"""
    _sync_solver_history!(previous, current)

Copy spectral field storage from `current` into the matching `previous` buffer.

This helper works on the real and imaginary parent arrays so it covers both
local PencilArray views and ordinary arrays.
"""
function _sync_solver_history!(previous, current)
    copyto!(parent(previous.data_real), parent(current.data_real))
    copyto!(parent(previous.data_imag), parent(current.data_imag))
    return previous
end

"""
    _sync_solver_nonlinear_histories!(state, magnetic_enabled)

Copy all active nonlinear terms into their previous-step buffers.

Temperature and velocity are always present; magnetic and composition histories
are updated only when those fields are enabled in the solver state.
"""
function _sync_solver_nonlinear_histories!(
        state::SolverState,
        magnetic_enabled::Bool
)
    _sync_solver_history!(state.fields.temperature.prev_nonlinear, state.fields.temperature.nonlinear)
    _sync_solver_history!(state.fields.velocity.prev_nl_toroidal, state.fields.velocity.nl_toroidal)
    _sync_solver_history!(state.fields.velocity.prev_nl_poloidal, state.fields.velocity.nl_poloidal)

    if magnetic_enabled && state.fields.magnetic !== nothing
        _sync_solver_history!(state.fields.magnetic.prev_nl_toroidal, state.fields.magnetic.nl_toroidal)
        _sync_solver_history!(state.fields.magnetic.prev_nl_poloidal, state.fields.magnetic.nl_poloidal)
    end

    if state.fields.composition !== nothing
        _sync_solver_history!(state.fields.composition.prev_nonlinear, state.fields.composition.nonlinear)
    end
end

"""
    timestepper isa RungeKutta3 && return integrate_solver_cb3_step!(state)

    bootstrap_solver_history!(state, timestepper, magnetic_enabled)

Seed previous nonlinear histories for two-step schemes on their first step.

CNAB2, ExponentialAdamsBashforth2, and ExponentialRungeKutta2 all need a previous nonlinear term. At startup there is
no true previous step, so the current nonlinear term is copied once to avoid a
special-case branch in each update kernel.
"""
function bootstrap_solver_history!(
        state::SolverState,
        timestepper,
        magnetic_enabled::Bool
)
    # Two-step schemes need a synthetic "previous" nonlinear state on their
    # first step so they can reuse AB2-style formulas without a special branch.
    if _solver_uses_two_step_history(timestepper) &&
       state.runtime.timestep_state.needs_ab2_bootstrap
        _sync_solver_nonlinear_histories!(state, magnetic_enabled)
        state.runtime.timestep_state.needs_ab2_bootstrap = false
    end

    return state
end

"""
    roll_solver_histories!(state, timestepper, magnetic_enabled)

Advance previous nonlinear histories at the end of a completed timestep.
"""
function roll_solver_histories!(
        state::SolverState,
        timestepper,
        magnetic_enabled::Bool
)
    _solver_uses_two_step_history(timestepper) || return state
    _sync_solver_nonlinear_histories!(state, magnetic_enabled)
    return state
end

"""
    initialize_solver_fields!(state)

Initialize all enabled field families and mark the solver state ready to step.

This is the solver-local implementation behind `GeoDynamo.initialize_fields!`.
"""
function initialize_solver_fields!(state::SolverState{T, <:AbstractArchitecture}) where {T}
    Random.seed!(42 + state.backend.rank)

    initialize_temperature_field!(state)
    initialize_velocity_field!(state)
    initialize_magnetic_field!(state)
    initialize_composition_field!(state)

    reset_solver_clock!(state; time = state.parameters.start_time, step = 0)
    _synchronize_solver_views!(state)
    state.is_initialized = true
    return state
end

"""
    GeoDynamo.initialize_fields!(state)

Public entry point for solver field initialization.

This forwards to `initialize_solver_fields!` so external callers can initialize
all enabled fields without depending on the solver-local helper name.
"""
function GeoDynamo.initialize_fields!(state::SolverState{
        T, <:AbstractArchitecture}) where {T}
    initialize_solver_fields!(state)
end

"""
    _solver_can_thread_implicit_updates(timestepper)

Return whether field implicit updates can run on Julia threads for `timestepper`,
based on the scheme and the rank topology alone.

`ExponentialAdamsBashforth2` update kernels issue MPI collectives directly, so at
more than one rank the `@spawn`'d field solves let the per-rank collective
ordering diverge and deadlock. `CNAB2` field kernels, by contrast, are radial-only
banded solves that issue NO collective in their base (single-field) path, so they
may be threaded at any rank count. Single-rank runs may thread every scheme (the
collectives are then no-ops and ordering is irrelevant).

NOTE: a *conducting inner core* or a `CONTINUITY_MAG` magnetic inner boundary adds
an `Allreduce` inside the CNAB2 magnetic toroidal/poloidal update tasks; that
config-dependent hazard is excluded separately at the call site via
`_solver_multirank_magnetic_collective`, since it needs the solver state.
"""
@inline function _solver_can_thread_implicit_updates(timestepper)
    return Threads.nthreads() > 1 && (mpi_comm_size() == 1 || timestepper isa CNAB2)
end

"""
    _solver_multirank_magnetic_collective(state)

Return `true` when a multi-rank run would issue an MPI collective from inside a
spawned magnetic field-update task, so the caller must keep the field solves
sequential.

The conducting-inner-core CNAB2 branches call `_magnetic_conducting_history_flux`
(an `Allreduce`) in BOTH the toroidal and poloidal spawned tasks, and the
`CONTINUITY_MAG` toroidal branch calls `_magnetic_toroidal_inner_bc_increment`
(also an `Allreduce`). Two spawned tasks each issuing a collective can interleave
differently across ranks and deadlock, so these configs stay on the sequential
path. Single-rank runs never reach here (the collectives are no-ops there).
"""
@inline function _solver_multirank_magnetic_collective(state::SolverState)
    mpi_comm_size() == 1 && return false
    return _solver_magnetic_config_has_collective(state)
end

"""
    _solver_magnetic_config_has_collective(state)

Whether this magnetic CONFIGURATION reaches an MPI collective from inside a spawned
field-update task, independent of rank count (the rank gate lives in
`_solver_multirank_magnetic_collective`, so this part stays testable at one rank).

Three sources, all of which keep the field solves sequential:
  * a conducting inner core — `_magnetic_conducting_history_flux` reduces in BOTH the
    toroidal and poloidal tasks;
  * a `CONTINUITY_MAG` inner boundary on EITHER component — the toroidal branch calls
    `_magnetic_toroidal_inner_bc_increment`, which reduces. Only `toroidal` used to be
    inspected here, so a poloidal-side entry slipped through;
  * topography magnetic coupling — `bcs/topography/magnetic_coupling.jl` reduces its own
    boundary quantities, and nothing in this predicate used to mention it.

This list is still enumerated by hand, which is why the collective side now carries
`_assert_no_collective_in_threaded_update` (solver/numerics.jl): anything this predicate
fails to anticipate raises there instead of deadlocking silently.
"""
@inline function _solver_magnetic_config_has_collective(state::SolverState)
    magnetic = state.fields.magnetic
    magnetic === nothing && return false
    state.magnetic_ic_admittance !== nothing && return true
    continuity = Int(CONTINUITY_MAG)
    any(==(continuity), magnetic.toroidal.bc_type_inner) && return true
    any(==(continuity), magnetic.poloidal.bc_type_inner) && return true
    topo = state.topography
    if topo !== nothing && topo.config.enabled && topo.config.magnetic_coupling
        return true
    end
    return false
end

"""
    _prepare_solver_eab2_caches!(state)

Ensure all active ExponentialAdamsBashforth2 exponential-action caches match the current parameters.
"""
function _prepare_solver_eab2_caches!(state::SolverState{
        T, <:AbstractArchitecture}) where {T}
    runtime = state.runtime
    params = state.parameters

    _ensure_etd_cache!(
        state.timestep_caches,
        :etd_temperature,
        params.Pm / params.Pr,
        T,
        runtime.outer_core_domain
    )
    _ensure_etd_cache!(
        state.timestep_caches,
        :etd_velocity_toroidal,
        params.Ek,
        T,
        runtime.outer_core_domain
    )
    _ensure_etd_cache!(
        state.timestep_caches,
        :etd_velocity_poloidal,
        params.Ek,
        T,
        runtime.outer_core_domain
    )

    if state.fields.magnetic !== nothing
        _ensure_etd_cache!(
            state.timestep_caches,
            :etd_magnetic_toroidal,
            1.0,
            T,
            runtime.outer_core_domain
        )
        _ensure_etd_cache!(
            state.timestep_caches,
            :etd_magnetic_poloidal,
            1.0,
            T,
            runtime.outer_core_domain
        )
    end

    if state.fields.composition !== nothing
        _ensure_etd_cache!(
            state.timestep_caches,
            :etd_composition,
            params.Pm / params.Sc,
            T,
            runtime.outer_core_domain
        )
    end

    return state
end

"""
    _apply_solver_implicit_updates_sequential!(state)

Apply every active field's implicit update on the current task.
"""
function _apply_solver_implicit_updates_sequential!(state::SolverState)
    apply_temperature_implicit_update!(state)
    apply_velocity_toroidal_implicit_update!(state)
    apply_velocity_poloidal_implicit_update!(state)
    apply_magnetic_toroidal_implicit_update!(state)
    apply_magnetic_poloidal_implicit_update!(state)
    apply_composition_implicit_update!(state)
    return state
end

"""
    _apply_solver_implicit_updates_threaded!(state)

Apply independent field implicit updates concurrently with Julia tasks.

This is used only for schemes whose update kernels do not require ordered MPI
collectives across fields.
"""
function _apply_solver_implicit_updates_threaded!(state::SolverState)
    tasks = Task[]
    sizehint!(tasks, 6)

    # Arm the collective-side guard for the duration of the spawned region, but only
    # where ordering divergence is actually possible: at one rank the collectives are
    # no-ops, threading every scheme is allowed, and arming it would turn a harmless
    # reduction into a spurious error.
    guard = mpi_comm_size() > 1
    guard && (_IN_THREADED_IMPLICIT_UPDATE[] = true)
    try
        push!(tasks, Threads.@spawn apply_temperature_implicit_update!(state))
        push!(tasks, Threads.@spawn apply_velocity_toroidal_implicit_update!(state))
        push!(tasks, Threads.@spawn apply_velocity_poloidal_implicit_update!(state))

        if state.fields.magnetic !== nothing
            push!(tasks, Threads.@spawn apply_magnetic_toroidal_implicit_update!(state))
            push!(tasks, Threads.@spawn apply_magnetic_poloidal_implicit_update!(state))
        end

        if state.fields.composition !== nothing
            push!(tasks, Threads.@spawn apply_composition_implicit_update!(state))
        end

        foreach(fetch, tasks)
    finally
        guard && (_IN_THREADED_IMPLICIT_UPDATE[] = false)
    end
    return state
end

"""
    apply_solver_implicit_step!(state)

Run one configured implicit/IMEX timestep update for all active fields.

The dispatcher handles first-step history bootstrap, prepares ExponentialAdamsBashforth2 caches when
needed, delegates ExponentialRungeKutta2 to the staged integrator, and rolls nonlinear histories
after a successful update.
"""
function apply_solver_implicit_step!(state::SolverState)
    timestepper = state.parameters.timestepper
    magnetic_enabled = state.parameters.include_magnetic &&
                       state.fields.magnetic !== nothing

    if timestepper isa RungeKutta3
        return integrate_solver_cb3_step!(state)
    end

    bootstrap_solver_history!(state, timestepper, magnetic_enabled)

    if timestepper isa ExponentialAdamsBashforth2
        _prepare_solver_eab2_caches!(state)
    end

    if timestepper isa ExponentialRungeKutta2
        integrate_solver_erk2_step!(state)
    else
        if _solver_can_thread_implicit_updates(timestepper) &&
           !_solver_multirank_magnetic_collective(state)
            _apply_solver_implicit_updates_threaded!(state)
        else
            # Sequential path when threading is unsafe: ExponentialAdamsBashforth2
            # issues MPI collectives during the update at multi-rank, and CNAB2
            # with a conducting inner core / CONTINUITY_MAG magnetic BC issues an
            # Allreduce inside the spawned magnetic tasks. Both would let per-rank
            # collective ordering diverge and deadlock.
            _apply_solver_implicit_updates_sequential!(state)
        end
    end

    roll_solver_histories!(state, timestepper, magnetic_enabled)
    return state
end

"""
    finalize_solver_step!(state, step)

Advance the runtime clock after a timestep and refresh solver views.
"""
function finalize_solver_step!(state::SolverState, step::Int)
    state.runtime.timestep_state.time = state.time + state.parameters.timestep
    state.runtime.timestep_state.step = step
    _synchronize_solver_views!(state)
    return state
end

"""
    check_solver_health!(state)

Run periodic health checks on the runtime state.

Currently this checks for NaNs every ten solver steps.
"""
function check_solver_health!(state::SolverState)
    state.step % 10 == 0 || return state
    check_runtime_for_nan(state)
    return state
end
