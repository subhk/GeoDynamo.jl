function initialize_velocity_field!(state::SolverState{T, <:AbstractArchitecture}) where {T}
    velocity = state.fields.velocity
    fill!(parent(velocity.toroidal.data_real), zero(T))
    fill!(parent(velocity.toroidal.data_imag), zero(T))
    fill!(parent(velocity.poloidal.data_real), zero(T))
    fill!(parent(velocity.poloidal.data_imag), zero(T))
    return state
end

function prepare_velocity_fields!(velocity_fields, domain)
    # Nonlinear velocity terms use the current velocity and vorticity in
    # physical space, so refresh both before accumulating body forces.
    reset_velocity_work_arrays!(velocity_fields)
    refresh_velocity_physical_fields!(velocity_fields, domain)
    refresh_vorticity_physical_fields!(velocity_fields, domain)
    return velocity_fields
end

function accumulate_velocity_nonlinear_terms!(
        velocity_fields,
        temperature_field,
        composition_field,
        magnetic_field,
        domain,
        params::SolverParameters
)
    return compute_velocity_body_forces!(
        velocity_fields,
        temperature_field,
        composition_field,
        magnetic_field,
        domain,
        params
    )
end

function finish_velocity_nonlinear!(velocity_fields; geometry::Symbol)
    if geometry === :ball
        return solver_ball_vector_analysis!(
            velocity_fields.advection_physical,
            velocity_fields.nl_toroidal,
            velocity_fields.nl_poloidal
        )
    end
    # Stage-4B momentum projections. Toroidal: r̂·∇× of momentum gives
    # Ek(∂t − Δ_l)T = T_F — the raw toroidal sphtor scalar of the force is
    # exactly the RHS (structure unchanged). Poloidal: r̂·∇×∇× gives
    # Ek(∂t − D_pol)W = N_W with W = D_pol·P and
    #   N_W = ∂_r(r·S_F) − Q_F
    # (λ/r² factors cancel in the W-equation; Q_F carries buoyancy — the
    # radial force component the legacy path silently discarded).
    vector_physical_to_spectral!(
        velocity_fields.advection_physical,
        velocity_fields.nl_toroidal,
        velocity_fields.nl_poloidal;
        raw_spheroidal = true
    )
    scalar_physical_to_spectral!(
        velocity_fields.advection_physical.r_component,
        velocity_fields.work_pol           # Q_F scratch
    )
    _poloidal_force_projection!(velocity_fields)
    return velocity_fields.nl_toroidal, velocity_fields.nl_poloidal
end

# nl_poloidal holds S_F on entry, work_pol holds Q_F; combine in place:
# nl_poloidal ← N_W = ∂_r(r·S_F) − Q_F  (per mode over the r-local axis).
function _poloidal_force_projection!(velocity_fields)
    cfg = velocity_fields.nl_poloidal.config
    domain = velocity_fields.domain
    T = eltype(parent(velocity_fields.nl_poloidal.data_real))
    D1 = create_derivative_matrix(T, 1, domain)
    nr = domain.N
    rS = Vector{T}(undef, nr)
    drS = Vector{T}(undef, nr)
    r_range = local_range(velocity_fields.nl_poloidal.pencil, 3)
    length(r_range) == nr || error(
        "poloidal force projection requires the radial axis fully local " *
        "(got $(length(r_range)) of $nr levels); r-distributed support is a follow-up")

    for (s_arr, q_arr) in (
        (parent(velocity_fields.nl_poloidal.data_real),
         parent(velocity_fields.work_pol.data_real)),
        (parent(velocity_fields.nl_poloidal.data_imag),
         parent(velocity_fields.work_pol.data_imag)),
    )
        @inbounds for lm in 1:cfg.nlm
            slot = local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            for r_idx in 1:nr
                rS[r_idx] = T(domain.r[r_idx, 4]) *
                            local_spectral_value(s_arr, slot, r_idx)
            end
            mul!(drS, D1, rS)
            for r_idx in 1:nr
                q = local_spectral_value(q_arr, slot, r_idx)
                set_local_spectral_value!(s_arr, slot, r_idx, drS[r_idx] - q)
            end
        end
    end
    return velocity_fields
end

function apply_velocity_toroidal_implicit_update!(state::SolverState{
        T, <:AbstractArchitecture}) where {T}
    velocity = state.fields.velocity
    runtime = state.runtime
    timestepper = state.parameters.timestepper
    dt = state.parameters.timestep
    E = state.parameters.Ek
    velocity_bc = _velocity_bc_code(state.parameters.velocity_bcs)

    if timestepper isa CNAB2
        matrices = state.implicit_matrices[:velocity_tor]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :velocity_toroidal,
            matrices.system_matrices[1].size
        )
        solver_build_rhs_cnab2!(
            velocity.work_tor,
            velocity.toroidal,
            velocity.nl_toroidal,
            velocity.prev_nl_toroidal,
            dt,
            matrices;
            mass_coeff = E,
            work = radial_work
        )
        solver_solve_velocity_implicit_step!(
            velocity.toroidal,
            velocity.work_tor,
            matrices,
            :toroidal;
            velocity_bc_code = velocity_bc,
            domain = runtime.outer_core_domain,
            work = radial_work
        )
    elseif timestepper isa EAB2
        alu_map = (state.timestep_caches.etd_velocity_toroidal::EAB2CacheEntry{T}).map
        radial_work = get_radial_work!(
            state.timestep_caches,
            :velocity_toroidal,
            runtime.outer_core_domain.N
        )
        bc_spec = build_solver_erk2_velocity_tor_bc(
            T,
            runtime.outer_core_domain,
            velocity_bc;
            config = runtime.shtns_config,
            rot_omega = 0.0
        )
        solver_eab2_update_krylov_cached!(
            velocity.toroidal,
            velocity.nl_toroidal,
            velocity.prev_nl_toroidal,
            alu_map,
            runtime.outer_core_domain,
            E,
            runtime.shtns_config,
            dt;
            m = _timestepper_krylov_dimension(timestepper, state.parameters),
            tol = _timestepper_krylov_tolerance(timestepper, state.parameters),
            mass_coeff = E,
            bc_spec = bc_spec,
            krylov_work = radial_work
        )
    else
        matrices = state.implicit_matrices[:velocity_tor]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :velocity_toroidal,
            matrices.system_matrices[1].size
        )
        solver_solve_velocity_implicit_step!(
            velocity.toroidal,
            velocity.nl_toroidal,
            matrices,
            :toroidal;
            velocity_bc_code = velocity_bc,
            domain = runtime.outer_core_domain,
            work = radial_work
        )
    end

    return state
end

function apply_velocity_poloidal_no_penetration!(
        state::SolverState{T, <:AbstractArchitecture},
        velocity_bc_code::Int
) where {T}
    params = state.parameters
    runtime = state.runtime
    theta = _timestepper_implicit_theta(params.timestepper, params)
    effective_diffusivity = one(params.Ek)
    influence = get_solver_erk2_influence_matrices!(
        state.timestep_caches,
        :velocity_poloidal,
        T,
        runtime.shtns_config,
        runtime.outer_core_domain,
        effective_diffusivity,
        params.timestep,
        velocity_bc_code;
        theta = theta
    )
    apply_solver_velocity_poloidal_influence_correction!(
        state.fields.velocity.poloidal,
        influence,
        runtime.shtns_config
    )
    return state
end

function apply_velocity_poloidal_implicit_update!(state::SolverState{
        T, <:AbstractArchitecture}) where {T}
    velocity = state.fields.velocity
    runtime = state.runtime
    timestepper = state.parameters.timestepper
    dt = state.parameters.timestep
    E = state.parameters.Ek
    velocity_bc = _velocity_bc_code(state.parameters.velocity_bcs)

    if timestepper isa CNAB2
        # Stage-4B pressure-free double-curl form (W-split):
        #   W := D_pol·P;  Ek(∂t − D_pol)W = N_W (AB2 on nl_poloidal = N_W);
        #   D_pol·P⁺ = W⁺ with P=0 walls; no-slip P′=0 via influence corrections.
        split = _get_or_build_poloidal_split!(state, velocity_bc)
        _apply_poloidal_wsplit_cnab2!(velocity, split, runtime.outer_core_domain, dt)
    else
        # Stage-4B: nl_poloidal now carries the W-equation RHS (N_W); the
        # exponential/theta paths still expect the legacy projection and are
        # gated until ported.
        error(_VEL_POL_STAGE4B_MSG)
    end

    return state
end

const _VEL_POL_STAGE4B_MSG =
    "velocity dynamics under the solenoidal convention are CNAB2-only until " *
    "the exponential (ERK2/EAB2) and theta-method poloidal paths are ported " *
    "to the W-split; see docs/superpowers/plans/2026-06-10-double-curl-stage4b-poloidal-momentum.md"

# Lazily build (and cache on the state) the Stage-4B poloidal split operators.
function _get_or_build_poloidal_split!(state::SolverState{T, <:AbstractArchitecture},
        velocity_bc::Int) where {T}
    caches = state.timestep_caches
    split = caches.poloidal_split
    split === nothing || return split::PoloidalSplitMatrices{T}
    split = create_velocity_poloidal_split_matrices(
        state.runtime.shtns_config,
        state.runtime.outer_core_domain,
        state.parameters.Ek,
        state.parameters.timestep;
        velocity_bc_code = velocity_bc,
        theta = _timestepper_implicit_theta(state.parameters.timestepper, state.parameters),
        T = T)
    caches.poloidal_split = split
    return split
end

# One CNAB2 step of the W-split per (l,m) mode (r-local layout).
function _apply_poloidal_wsplit_cnab2!(velocity, split::PoloidalSplitMatrices{T},
        domain, dt::Float64) where {T}
    cfg = velocity.poloidal.config
    nr = domain.N
    r_range = local_range(velocity.poloidal.pencil, 3)
    length(r_range) == nr || error(
        "poloidal W-split requires the radial axis fully local " *
        "(got $(length(r_range)) of $nr levels); r-distributed support is a follow-up")

    inv_dt = T(split.mass_coeff / dt)
    one_m_theta = T(1.0 - split.theta)

    P = Vector{T}(undef, nr); W = Vector{T}(undef, nr)
    LW = Vector{T}(undef, nr); rhs = Vector{T}(undef, nr)
    Wp = Vector{T}(undef, nr); Pp = Vector{T}(undef, nr)

    for (p_arr, n_arr, pn_arr) in (
        (parent(velocity.poloidal.data_real),
         parent(velocity.nl_poloidal.data_real),
         parent(velocity.prev_nl_poloidal.data_real)),
        (parent(velocity.poloidal.data_imag),
         parent(velocity.nl_poloidal.data_imag),
         parent(velocity.prev_nl_poloidal.data_imag)),
    )
        @inbounds for lm in 1:cfg.nlm
            slot = local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]
            if l == 0
                for r_idx in 1:nr
                    set_local_spectral_value!(p_arr, slot, r_idx, zero(T))
                end
                continue
            end
            idx = split.lookup[l]

            for r_idx in 1:nr
                P[r_idx] = local_spectral_value(p_arr, slot, r_idx)
            end
            mul!(W, split.dpol_op[idx], P)        # W = D_pol·P
            mul!(LW, split.w_linear[idx], W)      # Ek·D_pol·W
            for r_idx in 1:nr
                rhs[r_idx] = inv_dt * W[r_idx] + one_m_theta * LW[r_idx] +
                             T(1.5) * local_spectral_value(n_arr, slot, r_idx) -
                             T(0.5) * local_spectral_value(pn_arr, slot, r_idx)
            end
            solve_banded!(Wp, split.w_factor[idx], rhs)

            # Dirichlet P-recovery (P = 0 rows ⇒ zero those RHS entries)
            Wp[1] = zero(T); Wp[nr] = zero(T)
            solve_banded!(Pp, split.p_factor[idx], Wp)

            # No-slip influence correction: zero endpoint P′ via the cached
            # Green responses.
            rho1 = dot(split.d1_row_inner, Pp)
            rho2 = dot(split.d1_row_outer, Pp)
            M = split.influence[idx]
            det = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]
            a1 = (-rho1 * M[2, 2] + rho2 * M[1, 2]) / det
            a2 = (-rho2 * M[1, 1] + rho1 * M[2, 1]) / det
            h1 = split.h1[idx]; h2 = split.h2[idx]
            for r_idx in 1:nr
                set_local_spectral_value!(p_arr, slot, r_idx,
                    Pp[r_idx] + a1 * h1[r_idx] + a2 * h2[r_idx])
            end
        end
    end
    return velocity
end

function queue_velocity_implicit_updates!(
        operations::Vector{Function},
        state::SolverState{T, <:AbstractArchitecture}
) where {T}
    push!(operations, () -> apply_velocity_toroidal_implicit_update!(state))
    push!(operations, () -> apply_velocity_poloidal_implicit_update!(state))
    return operations
end
