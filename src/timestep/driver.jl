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
_solver_uses_two_step_history(timestepper) =
    timestepper isa CNAB2 || timestepper isa EAB2 || timestepper isa ERK2

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
    magnetic_enabled::Bool,
)
    _sync_solver_history!(state.fields.temperature.prev_nonlinear, state.fields.temperature.nonlinear)
    _sync_solver_history!(state.fields.velocity.prev_nlᵀ, state.fields.velocity.nlᵀ)
    _sync_solver_history!(state.fields.velocity.prev_nlᴾ, state.fields.velocity.nlᴾ)

    if magnetic_enabled && state.fields.magnetic !== nothing
        _sync_solver_history!(state.fields.magnetic.prev_nlᵀ, state.fields.magnetic.nlᵀ)
        _sync_solver_history!(state.fields.magnetic.prev_nlᴾ, state.fields.magnetic.nlᴾ)
    end

    if state.fields.composition !== nothing
        _sync_solver_history!(state.fields.composition.prev_nonlinear, state.fields.composition.nonlinear)
    end
end

"""
    bootstrap_solver_history!(state, timestepper, magnetic_enabled)

Seed previous nonlinear histories for two-step schemes on their first step.

CNAB2, EAB2, and ERK2 all need a previous nonlinear term. At startup there is
no true previous step, so the current nonlinear term is copied once to avoid a
special-case branch in each update kernel.
"""
function bootstrap_solver_history!(
    state::SolverState,
    timestepper,
    magnetic_enabled::Bool,
)
    # Two-step schemes need a synthetic "previous" nonlinear state on their
    # first step so they can reuse AB2-style formulas without a special branch.
    if _solver_uses_two_step_history(timestepper) && state.runtime.timestep_state.needs_ab2_bootstrap
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
    magnetic_enabled::Bool,
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
function initialize_solver_fields!(state::SolverState{T,<:AbstractArchitecture}) where T
    Random.seed!(42 + state.backend.rank)

    initialize_temperature_field!(state)
    initialize_velocity_field!(state)
    initialize_magnetic_field!(state)
    initialize_composition_field!(state)

    reset_solver_clock!(state; time=state.parameters.start_time, step=0)
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
GeoDynamo.initialize_fields!(state::SolverState{T,<:AbstractArchitecture}) where {T} =
    initialize_solver_fields!(state)

"""
    _solver_can_thread_implicit_updates(timestepper)

Return whether field implicit updates can run on Julia threads.

Field update kernels reach MPI collectives (EAB2 directly; others via influence
and transpose paths). With more than one rank, issuing those collectives from
multiple Julia threads lets the per-rank ordering diverge and the collectives
mismatch, which deadlocks. So restrict threaded field updates to single-rank
runs regardless of scheme; multi-rank runs use the sequential path.
"""
@inline function _solver_can_thread_implicit_updates(timestepper)
    return mpi_comm_size() == 1
end

"""
    _prepare_solver_eab2_caches!(state)

Ensure all active EAB2 exponential-action caches match the current parameters.
"""
function _prepare_solver_eab2_caches!(state::SolverState{T,<:AbstractArchitecture}) where T
    runtime = state.runtime
    params = state.parameters

    _ensure_etd_cache!(
        state.timestep_caches,
        :etd_temperature,
        params.Pm / params.Pr,
        T,
        runtime.𝒟ᵒᶜ,
    )
    _ensure_etd_cache!(
        state.timestep_caches,
        :etd_velocity_toroidal,
        params.Ek,
        T,
        runtime.𝒟ᵒᶜ,
    )
    _ensure_etd_cache!(
        state.timestep_caches,
        :etd_velocity_poloidal,
        params.Ek,
        T,
        runtime.𝒟ᵒᶜ,
    )

    if state.fields.magnetic !== nothing
        _ensure_etd_cache!(
            state.timestep_caches,
            :etd_magnetic_toroidal,
            1.0,
            T,
            runtime.𝒟ᵒᶜ,
        )
        _ensure_etd_cache!(
            state.timestep_caches,
            :etd_magnetic_poloidal,
            1.0,
            T,
            runtime.𝒟ᵒᶜ,
        )
    end

    if state.fields.composition !== nothing
        _ensure_etd_cache!(
            state.timestep_caches,
            :etd_composition,
            params.Pm / params.Sc,
            T,
            runtime.𝒟ᵒᶜ,
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
    return state
end

"""
    apply_solver_implicit_step!(state)

Run one configured implicit/IMEX timestep update for all active fields.

The dispatcher handles first-step history bootstrap, prepares EAB2 caches when
needed, delegates ERK2 to the staged integrator, and rolls nonlinear histories
after a successful update.
"""
function apply_solver_implicit_step!(state::SolverState)
    timestepper = state.parameters.timestepper
    magnetic_enabled = state.parameters.include_magnetic_field && state.fields.magnetic !== nothing

    bootstrap_solver_history!(state, timestepper, magnetic_enabled)

    if timestepper isa EAB2
        _prepare_solver_eab2_caches!(state)
    end

    if timestepper isa ERK2
        integrate_solver_erk2_step!(state)
    else
        if _solver_can_thread_implicit_updates(timestepper)
            _apply_solver_implicit_updates_threaded!(state)
        else
            # EAB2 issues MPI collectives during the update, so multi-rank runs
            # must keep those field solves on one thread to avoid deadlocks.
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
