# ================================================================================
# Shared scalar-field solver driver logic (temperature + composition)
# ================================================================================
#
# Temperature and composition obey the same scalar transport equation
#   ∂C/∂t + u·∇C = κ ∇²C  (+ internal sources, temperature only)
# and share the same struct interface (AbstractScalarField), the same core
# implicit solve (_solver_solve_scalar_implicit_step!), and the same boundary
# code (_thermal_bc_code). These two helpers hold the field-agnostic driver
# logic; the public per-field functions in {temperature,composition}/solver.jl
# are thin shims that supply the few field-specific values.
#
# Included from src/solver.jl BEFORE the temperature/composition solver files so
# both can call into it; all referenced solver/timestep symbols (SolverState,
# EAB2CacheEntry, CNAB2/EAB2, get_radial_work!, solver_build_rhs_cnab2!,
# with_boundary_mode_values, build_solver_erk2_scalar_bc,
# solver_eab2_update_krylov_cached!) are defined by this point.

"""
    _solver_compute_scalar_nonlinear!(𝔽, vel_fields, outer_core_domain, ws;
                                      add_internal_sources)

Field-agnostic scalar nonlinear assembly: zero work arrays, compute gradients,
transform field+gradients to physical space, form advection (and, for
temperature, internal sources), then transform the nonlinear term back to
spectral. `add_internal_sources` is the ONLY behavioral difference between
temperature (`true`) and composition (`false`).
"""
function _solver_compute_scalar_nonlinear!(
        𝔽::AbstractScalarField{T},
        vel_fields,
        outer_core_domain::RadialDomainType,
        ws::SolverGradientWorkspace{T};
        add_internal_sources::Bool,
) where {T}
    t_start = timing_enabled() ? mpi_wtime() : 0.0

    solver_zero_scalar_work_arrays!(𝔽)
    zero_gradient_workspace!(ws)

    if timing_enabled()
        t_spectral = mpi_wtime()
    end
    compute_all_gradients_spectral!(𝔽, outer_core_domain, ws)
    if timing_enabled()
        𝔽.spectral_time[] += mpi_wtime() - t_spectral
    end

    if timing_enabled()
        t_transform = mpi_wtime()
    end
    transform_field_and_gradients_to_physical!(𝔽, ws, outer_core_domain)
    if timing_enabled()
        𝔽.transform_time[] += mpi_wtime() - t_transform
    end

    if vel_fields !== nothing
        solver_compute_scalar_advection_local!(𝔽, vel_fields)
    end
    if add_internal_sources
        solver_add_internal_sources_local!(𝔽, outer_core_domain)
    end

    if timing_enabled()
        t_transform = mpi_wtime()
    end
    scalar_nonlinear_to_spectral!(
        𝔽.advection_physical,
        𝔽.nonlinear,
    )
    if timing_enabled()
        𝔽.transform_time[] += mpi_wtime() - t_transform
        𝔽.computation_time[] += mpi_wtime() - t_start
    end
    return 𝔽
end

"""
    _apply_scalar_implicit_update!(state, 𝔽, key, diffusivity, bc_code,
                                   solve_step!, etd_entry)

Field-agnostic CNAB2/EAB2/default implicit update for a scalar field. The shim
supplies the field-specific cache/matrix `key` (`:temperature` / `:composition`),
the `diffusivity` (`Pm/Pr` / `Pm/Sc`), the boundary `bc_code`, the per-field
solve wrapper `solve_step!`, and the EAB2 cache `etd_entry`
(`state.timestep_caches.etd_temperature` / `etd_composition`).
"""
function _apply_scalar_implicit_update!(
        state::SolverState{T, <:AbstractArchitecture},
        𝔽,
        key::Symbol,
        diffusivity::T,
        bc_code::Int,
        solve_step!::G,
        etd_entry::Union{EAB2CacheEntry{T}, Nothing},
) where {T, G}
    runtime = state.runtime
    bc = get_bc_vectors(𝔽)
    timestepper = state.parameters.timestepper
    dt = state.parameters.timestep

    if timestepper isa CNAB2
        matrices = state.implicit_matrices[key]
        radial_work = get_radial_work!(
            state.timestep_caches,
            key,
            matrices.system_matrices[1].size,
        )
        solver_build_rhs_cnab2!(
            𝔽.work_spectral,
            𝔽.spectral,
            𝔽.nonlinear,
            𝔽.prev_nonlinear,
            dt,
            matrices;
            work = radial_work,
        )
        solve_step!(
            𝔽.spectral,
            𝔽.work_spectral,
            matrices;
            bc_inner = bc.inner_real,
            bc_outer = bc.outer_real,
            bc_inner_imag = bc.inner_imag,
            bc_outer_imag = bc.outer_imag,
            work = radial_work,
        )
    elseif timestepper isa ExponentialAdamsBashforth2
        alu_map = (etd_entry::EAB2CacheEntry{T}).map
        radial_work = get_radial_work!(
            state.timestep_caches,
            key,
            runtime.outer_core_domain.N,
        )
        scalar_bc = build_solver_erk2_scalar_bc(T, runtime.outer_core_domain, bc_code;
            inner_regularity = state.parameters.geometry === :ball)
        bc_spec = with_boundary_mode_values(
            scalar_bc,
            bc.inner_real,
            bc.outer_real,
            bc.inner_imag,
            bc.outer_imag,
        )
        solver_eab2_update_krylov_cached!(
            𝔽.spectral,
            𝔽.nonlinear,
            𝔽.prev_nonlinear,
            alu_map,
            runtime.outer_core_domain,
            diffusivity,
            runtime.shtns_config,
            dt;
            m = _timestepper_krylov_dimension(timestepper, state.parameters),
            tol = _timestepper_krylov_tolerance(timestepper, state.parameters),
            bc_spec = bc_spec,
            krylov_work = radial_work,
        )
    else
        matrices = state.implicit_matrices[key]
        radial_work = get_radial_work!(
            state.timestep_caches,
            key,
            matrices.system_matrices[1].size,
        )
        solve_step!(
            𝔽.spectral,
            𝔽.nonlinear,
            matrices;
            bc_inner = bc.inner_real,
            bc_outer = bc.outer_real,
            bc_inner_imag = bc.inner_imag,
            bc_outer_imag = bc.outer_imag,
            work = radial_work,
        )
    end

    return state
end

# Map per-boundary DIRICHLET/NEUMANN ints to the scalar_bc_code used by
# _apply_scalar_boundary_rows! (1=DD, 2=DN, 3=ND, 4=NN).
@inline function _scalar_bc_code_from_types(inner_type::Int, outer_type::Int)
    di = Int(DIRICHLET)
    inner_d = inner_type == di
    outer_d = outer_type == di
    return inner_d ? (outer_d ? 1 : 2) : (outer_d ? 3 : 4)
end

# Resolve a source spec to a per-radial-node vector S(r). `default` is the
# geometry-aware fallback used when `source === nothing`.
function _resolve_source(source, domain, default::Real)
    nr = domain.N
    if source === nothing
        return fill(Float64(default), nr)
    elseif source isa Function
        return Float64[source(domain.r[k, 4]) for k in 1:nr]
    else
        return fill(Float64(source), nr)
    end
end
