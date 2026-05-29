"""
    initialize_solver_state([T=Float64]; params=create_solver_parameters())

Construct a fully configured `SolverState{T}` for the rewritten solver path.

This synchronizes the public `SolverParameters` into the shared backend layer,
builds the backend/runtime objects, activates topography, and prepares the
field views and timestep caches used by the main loop.
"""
function initialize_solver_state(::Type{T} = Float64;
        params::SolverParameters = create_solver_parameters()) where {T}

    # Keep the package-level parameter view in sync while the new solver stack
    # still shares a few kernels and file-loading paths with the older runtime.
    apply_solver_parameters!(params)
    backend = create_solver_backend(params)
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
        TimestepCaches{T}(),
        create_solver_energy_tracker(),
        create_solver_solenoidal_monitor(),
        magnetic_ic_admittance,
        params.start_time,
        0,
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
    advance_solver_step!(state)

Advance the rewritten solver by one timestep.

The step order is:

1. compute nonlinear terms
2. apply topographic corrections
3. apply the IMEX/ERK2 timestep update
4. finalize time/step bookkeeping and diagnostics
"""
function advance_solver_step!(state::SolverState{T, <:AbstractArchitecture}) where {T}
    state.is_initialized || initialize_solver_fields!(state)

    next_step = state.step + 1
    state.runtime.timestep_state.step = next_step

    # Step 1: compute explicit nonlinear terms from the current fields.
    compute_solver_nonlinear_terms!(state)

    # Step 2: apply boundary topography corrections before the implicit solve.
    apply_solver_topography!(state)

    # Step 3: advance one IMEX step using the existing kernels.
    apply_solver_implicit_step!(state)

    # Step 4: update time, solver views, and phase change bookkeeping.
    finalize_solver_step!(state, next_step)
    update_solver_icb_phase_change!(state)
    check_solver_health!(state)

    return state
end

"""
    rebuild_solver_implicit_matrices!(state, dt)

Rebuild the pre-factored implicit matrix store for a new timestep `dt`.
The implicit operators bake `dt` in at factorization time, so changing the
timestep requires re-factorization. Reuses the backend's grid/config and all
non-timestep physical parameters; only `dt` changes.
"""
function rebuild_solver_implicit_matrices!(
        state::SolverState{
            T, <:AbstractArchitecture}, dt::Real) where {T}
    backend = state.backend
    # NOTE: backend.parameters.timestep is frozen at construction and intentionally
    # ignored here — `dt` is authoritative. Only non-timestep params are read.
    # The conducting-inner-core ICB admittance also depends on dt, so refresh it too.
    matrices,
    magnetic_ic_admittance = _build_implicit_matrices_dict(
        T, backend.shtns_config, backend.outer_core_domain,
        backend.inner_core_domain, backend.parameters, Float64(dt))
    state.implicit_matrices = create_solver_implicit_matrix_store(matrices)
    state.magnetic_ic_admittance = magnetic_ic_admittance
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
        advance_solver_step!(state)
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
