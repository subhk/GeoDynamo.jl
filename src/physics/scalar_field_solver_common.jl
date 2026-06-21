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
transform field+gradients to physical space, form advection (and, when
`add_internal_sources`, the radial internal-source profile), then transform the
nonlinear term back to spectral. Both temperature and composition pass
`add_internal_sources = true` so their respective source fields
(`internal_heating` / `compositional_source`) enter the evolution.
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

# Set a scalar field's l=0 conductive (0,0)-coefficient profile + sustain its
# internal source, BC- and source-aware. `diffusivity` is the field's implicit
# diffusivity (Pm/Pr for temperature, Pm/Sc for composition). `source` is the
# user spec (nothing/Real/Function); `default_H` the geometry-aware fallback.
# Writes field.internal_sources AND the (0,0) spectral coefficient. Assumes the
# field's boundary_values/bc_type_* are already populated. This is the single
# shared driver for both the default-IC path (initialize_{temperature,
# composition}_field!) and the public AnalyticIC(:conductive) path.
function apply_scalar_conductive_l0!(field, domain, geometry::Symbol, source,
        diffusivity::Real, default_H::Real)
    T = eltype(parent(field.spectral.data_real))
    cfg = field.spectral.config
    m00 = get_mode_index(cfg, 0, 0)
    in_t = m00 > 0 ? field.bc_type_inner[m00] : Int(DIRICHLET)
    out_t = m00 > 0 ? field.bc_type_outer[m00] : Int(DIRICHLET)
    in_v = m00 > 0 ? field.boundary_values[1, m00] : zero(T)
    out_v = m00 > 0 ? field.boundary_values[2, m00] : zero(T)
    H_vec = _resolve_source(source, domain, default_H)
    copyto!(field.internal_sources, T.(H_vec))
    s4pi = sqrt(4 * Float64(π))
    S_coeff = (H_vec ./ Float64(diffusivity)) .* s4pi
    cond_c = conductive_profile_solve(; domain = domain,
        bc_code = _scalar_bc_code_from_types(in_t, out_t),
        inner_value = in_v, outer_value = out_v,
        source = S_coeff, inner_regularity = geometry === :ball)
    slot = local_spectral_storage_slot(cfg, m00)
    if slot !== nothing
        r_range = get_local_range(field.spectral.pencil, 3)
        real_data = parent(field.spectral.data_real)
        for (lr, gr) in enumerate(r_range)
            lr <= size(real_data, 3) || continue
            set_local_spectral_value!(real_data, slot, lr, T(cond_c[gr]))
        end
    end
    return cond_c
end

"""
    conductive_profile_solve(; domain, bc_code, inner_value, outer_value,
                             source, inner_regularity) -> Vector{Float64}

Steady l=0 (0,0)-coefficient profile c(r) solving ∇²₀ c = −source with the same
boundary rows the implicit scalar matrices use (⇒ the IC is a discrete
equilibrium). `inner_value`/`outer_value` are the boundary coefficients (Dirichlet
value, or Neumann flux). `source` is a length-N vector (RHS of ∇²c = −S). The
function is linear and unit-agnostic; callers pre-scale by √(4π) as needed.
`inner_regularity=true` selects the ball-centre row (Θ′(r₁)=0 for l=0).
"""
function conductive_profile_solve(; domain, bc_code::Int,
        inner_value::Real, outer_value::Real,
        source::AbstractVector, inner_regularity::Bool = false)
    T = Float64
    N = domain.N
    bw = radial_bandwidth(domain)
    lap = create_radial_laplacian(domain)
    d1 = create_derivative_matrix(T, 1, domain)
    sys = T.(copy(lap.data))
    _zero_scalar_boundary_rows!(sys, bw, N)
    _apply_scalar_boundary_rows!(sys, T.(d1.data), bc_code, 0, bw, N,
        inner_regularity, Float64(domain.r[1, 3]))
    if bc_code == 4
        # Double-Neumann (NN): the STEADY bare Laplacian has a constant null
        # space (both walls Neumann ⇒ the mean level is unconstrained), so anchor
        # the otherwise-free gauge by pinning the inner mean value (rhs[1] =
        # inner_value below). This is a property of the singular steady solve
        # ONLY — the implicit time-stepping operator (mass/dt) I − θκL is already
        # non-singular under pure-Neumann rows, so its builder must NOT pin l=0
        # (see _apply_scalar_boundary_rows!).
        @inbounds for j in 1:(1 + bw)
            sys[bw + 1 + 1 - j, j] = zero(T)
        end
        sys[bw + 1, 1] = one(T)
    end
    A = BandedMatrix{T}(sys, bw, N)
    lu = factorize_banded(A)
    rhs = Vector{T}(undef, N)
    @inbounds for i in 1:N
        rhs[i] = -T(source[i])
    end
    rhs[1] = T(inner_value)
    rhs[N] = T(outer_value)
    solve_banded!(rhs, lu, rhs)   # in-place banded solve (x === b aliasing safe); rhs ← c(r)
    return rhs
end
