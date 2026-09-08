"""
    initialize_solver_state([T=Float64]; params=create_solver_parameters())

Construct a fully configured `SolverState{T}` for the rewritten solver path.

This synchronizes the public `SolverParameters` into the shared backend layer,
builds the backend/runtime objects, activates topography, and prepares the
field views and timestep caches used by the main loop.
"""
function initialize_solver_state(::Type{T} = Float64;
        params::SolverParameters = create_solver_parameters(),
        arch::Union{Nothing, AbstractArchitecture} = nothing) where {T}

    # Reject invalid parameters up front rather than building a runtime around
    # them (negative timestep, courant<=0, transform-invalid grid, ...). Use the
    # non-printing checker so valid builds stay quiet.
    param_errors, _ = _parameter_errors_warnings(params)
    isempty(param_errors) || throw(ArgumentError(
        "initialize_solver_state: invalid parameters: " * join(param_errors, "; ")))

    # Keep the package-level parameter view in sync while the new solver stack
    # still shares a few kernels and file-loading paths with the older runtime.
    apply_solver_parameters!(params)
    # When the caller hands us a concrete architecture object (e.g. the grid's
    # `GPU(CUDABackend())`), thread it through verbatim so the live backend is
    # preserved rather than rebuilt — lossily — from `params.architecture`.
    backend = isnothing(arch) ? create_solver_backend(params) :
              create_solver_backend(arch, params)
    implicit_matrices, magnetic_ic_admittance = create_solver_implicit_matrices(T, backend)
    runtime = create_solver_runtime(
        T,
        backend;
        auto_optimize = false,
        adaptive_threading = false
    )

    # Topography state stays separate from the core runtime so coupling can be
    # enabled, disabled, or swapped without rebuilding the spectral fields.
    topography = create_solver_topography_state(T, params)
    activate_solver_topography!(topography)
    fields = _collect_solver_fields(runtime, params)

    return SolverState(
        params,
        backend,
        fields,
        topography,
        runtime,
        create_solver_implicit_matrix_store(implicit_matrices),
        TimestepCaches{T}(solver_operator_key(params)),
        create_solver_energy_tracker(),
        create_solver_solenoidal_monitor(),
        magnetic_ic_admittance,
        false
    )
end

"""
    initialize_simulation(T, params::SolverParameters)

Public entry point for creating a rewritten solver state with numeric element
type `T`.
"""
function GeoDynamo.initialize_simulation(::Type{T}, params::SolverParameters) where {T}
    initialize_solver_state(T; params = params)
end

"""
    initialize_simulation(params::SolverParameters)

Convenience wrapper for `initialize_simulation(Float64, params)`.
"""
function GeoDynamo.initialize_simulation(params::SolverParameters)
    initialize_solver_state(Float64; params = params)
end

"""
    solver_step!(state)

Advance the rewritten solver by one timestep.

The step order is:

1. compute nonlinear terms
2. apply topographic corrections
3. apply the IMEX/ERK2 timestep update
4. finalize time/step bookkeeping and diagnostics
"""
function solver_step!(state::SolverState{T, <:AbstractArchitecture}) where {T}
    prepare_solver_host_update!(state)
    ensure_solver_operators!(state)
    state.is_initialized || initialize_solver_fields!(state)

    next_step = state.step + 1

    # Step 1: compute explicit nonlinear terms from the current fields.
    compute_solver_nonlinear_terms!(state)

    # Step 2: apply boundary topography corrections before the implicit solve.
    apply_solver_topography!(state)

    # Step 3: advance one IMEX step using the existing kernels.
    apply_solver_implicit_step!(state)

    # Step 4: update time, solver views, and phase change bookkeeping.
    finalize_solver_step!(state, next_step)
    if state.parameters.timestepper isa ExponentialRungeKutta2
        run_diagnostics!(state; interval = 100)
    end
    update_solver_icb_phase_change!(state)
    check_solver_health!(state)

    return state
end

# Back-compat alias for the pre-rename public name.
const advance_solver_step! = solver_step!

"""
    rebuild_solver_implicit_matrices!(state, dt)

Rebuild implicit matrices and discard derived operators using the current
parameters and timestep `dt`. Grid resources and scratch buffers are reused.
"""
function rebuild_solver_implicit_matrices!(
        state::SolverState{
            T, <:AbstractArchitecture}, dt::Real) where {T}
    prepare_solver_host_update!(state)
    backend = state.backend
    # Physical parameters come from the LIVE `state.parameters`, not
    # `backend.parameters`: `SolverBackend` is immutable and its snapshot is frozen at
    # construction, while `Simulation`/`time_step!` replace `state.parameters` (the only
    # two `.parameters =` sites in src/ are both on the state). Reading the frozen copy
    # silently reverted theta (from `p.timestepper`), Ek, Pm/Pr, Pm/Sc, the
    # velocity/thermal/composition BC codes and `magnetic_inner_bc` to their
    # construction-time values — so `Simulation(model; implicit_theta=1.0)` left the
    # toroidal/scalar systems at Crank-Nicolson while the poloidal W-split (which reads
    # `state.parameters`) used theta=1.0: two different implicit weights in one scheme.
    # Only the grid/domain objects still come from the backend, and `dt` stays
    # authoritative from the caller (backend.parameters.timestep is frozen too).
    matrices,
    magnetic_ic_admittance = _build_implicit_matrices_dict(
        T, backend.shtns_config, backend.outer_core_domain,
        backend.inner_core_domain, state.parameters, Float64(dt))
    store = create_solver_implicit_matrix_store(matrices)
    key = solver_operator_key(state.parameters, dt)
    caches = fresh_solver_operator_caches(state.timestep_caches, key)
    old_key = state.timestep_caches.operator_key
    # Preserve variable-step history when only dt changes. A different operator
    # or scheme requires a fresh bootstrap of its nonlinear history.
    if old_key !== nothing && solver_operator_key(state.parameters, old_key.dt) != old_key
        state.runtime.timestep_state.needs_ab2_bootstrap = true
    end
    state.implicit_matrices = store
    state.magnetic_ic_admittance = magnetic_ic_admittance
    state.timestep_caches = caches
    state.runtime.timestep_state.dt = dt
    return state
end

"""Ensure all cached timestep operators match the requested run controls."""
function ensure_solver_operators!(state::SolverState, dt::Real=state.parameters.timestep)
    key = solver_operator_key(state.parameters, dt)
    key == state.timestep_caches.operator_key || rebuild_solver_implicit_matrices!(state, dt)
    return state
end

"""
    run_solver!(state)

Run the rewritten solver until `state.parameters.end_time` or
`state.parameters.stop_iteration` is reached.
"""
function run_solver!(state::SolverState{T, <:AbstractArchitecture}) where {T}
    state.is_initialized || initialize_solver_fields!(state)

    while state.time < state.parameters.end_time &&
        state.step < state.parameters.stop_iteration
        solver_step!(state)
    end

    return state
end

"""
    run_simulation!(state::SolverState; restart_file="", restart_dir="", restart_time=0.0)

Public package-facing loop driver for `SolverState`.

This intentionally rejects the legacy restart keywords on the rewritten solver
path, because restart handling is configured directly on the new state/runtime
objects before calling `run_solver!`.
"""
function GeoDynamo.run_simulation!(
        state::SolverState{T, <:AbstractArchitecture};
        restart_file::String = "",
        restart_dir::String = "",
        restart_time::Float64 = 0.0
) where {T}
    if !isempty(restart_file) || !isempty(restart_dir) || restart_time != 0.0
        throw(ArgumentError(
            "SolverState uses the rewritten solver loop and does not support legacy restart keywords in `run_simulation!`. " *
            "Use `run_solver!` directly after setting up restart state on the solver path.",
        ))
    end
    return run_solver!(state)
end
