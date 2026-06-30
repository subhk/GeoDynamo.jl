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

function finish_velocity_nonlinear!(velocity_fields)
    # geometry-blind: the ball grid has no r=0 node (off-center grid)
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
    # Reuse the cached first-derivative operator (built from this same
    # outer-core domain in the field constructor) instead of rebuilding the
    # identical banded matrix every CNAB2 step.
    ∂r = velocity_fields.∂r
    nr = domain.N
    # Reuse the field's cached workspace scratch instead of allocating two
    # nr-vectors every velocity nonlinear pass. The mode loop is serial, so the
    # thread-1 buffers are safe.
    ws = _get_or_build_velocity_workspace!(velocity_fields, nr)
    rS  = ws.force_proj_rS[1]
    drS = ws.force_proj_drS[1]
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
            mul!(drS, ∂r, rS)
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
            bc_inner = view(velocity.toroidal.boundary_values, 1, :),
            bc_outer = view(velocity.toroidal.boundary_values, 2, :),
            work = radial_work
        )
    elseif timestepper isa ExponentialAdamsBashforth2
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
            rot_omega = 0.0,
            inner_regularity = state.parameters.geometry === :ball
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
            bc_inner = view(velocity.toroidal.boundary_values, 1, :),
            bc_outer = view(velocity.toroidal.boundary_values, 2, :),
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
        # Stage-4B: nl_poloidal carries the W-equation RHS (N_W); the exponential
        # (ERK2/EAB2) and theta-method poloidal paths are gated until ported to
        # the W-split. NOTE: EAB2 layers 1 (φ2 order) and 2 (singular operator)
        # are fixed; the remaining layer 3 is the exponential W-split P-recovery
        # with a LIFT-based φ₁-column influence (the homogeneous-Dirichlet ETD
        # operator decouples the walls, so the influence source must be the
        # boundary-coupling columns, not unit wall vectors). See
        # docs/superpowers/plans/2026-06-10-double-curl-stage4b-poloidal-momentum.md.
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
        ball = state.parameters.geometry === :ball,
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

    P, W, LW, rhs, Wp, Pp = split.work   # cached per-step radial scratch

    # Topography impermeability correction modifies the P wall VALUE (real-only,
    # stored in poloidal.boundary_values). Injected as the Dirichlet RHS of the
    # P-recovery (Wp[1]/Wp[nr]); zero when topography is disabled ⇒ unchanged.
    pol_bv = velocity.poloidal.boundary_values

    for (is_real, p_arr, n_arr, pn_arr) in (
        (true,
         parent(velocity.poloidal.data_real),
         parent(velocity.nl_poloidal.data_real),
         parent(velocity.prev_nl_poloidal.data_real)),
        (false,
         parent(velocity.poloidal.data_imag),
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

            # Inner residual: ball evaluates the W-regularity Robin row on the
            # W solution (pre-zeroing); shell evaluates the inner wall row on P.
            # rho1w is a short-lived temporary consumed by the rho1 selection
            # below; ball must read Wp before the wall-zeroing that follows,
            # while shell reads Pp post-recovery (hence the two-variable idiom).
            rho1w = split.ball ?
                    dot(split.d1_row_inner, Wp) -
                    T((l + 1) * split.reg_r_inv) * Wp[1] : zero(T)

            # Dirichlet P-recovery: the wall RHS entries are the imposed P values.
            # Base impermeability is P=0; the topography correction (real part of
            # poloidal.boundary_values) shifts the wall value. Ball: row 1 is the
            # regularity Robin (homogeneous RHS) — never inject there.
            Wp[1] = (is_real && !split.ball) ? pol_bv[1, lm] : zero(T)
            Wp[nr] = is_real ? pol_bv[2, lm] : zero(T)
            solve_banded!(Pp, split.p_factor[idx], Wp)

            # Influence correction: zero the remaining endpoint residuals via
            # the cached Green responses.
            rho1 = split.ball ? rho1w : dot(split.d1_row_inner, Pp)
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

# ============================================================================
# Stage-4B ERK2 W-split support (see docs/superpowers/plans/2026-06-11-erk2-
# wsplit-port.md). The ERK2 stage machinery advances V := Ek·W = Ek·D_pol·P
# (so ∂t V = D_pol·V + N_W with the cache built on D_pol, diffusivity 1, and
# nl_poloidal unscaled). P is recovered from V with Dirichlet walls plus
# influence corrections whose Green responses are the φ1 columns:
# g_i = c·φ1(cA)·e_i (c = dt/2 at the stage, dt at the finalize).
# ============================================================================

# V := Ek·D_pol·P written into velocity.work_pol (the V host during the step).
function _erk2_poloidal_to_V!(velocity, split::PoloidalSplitMatrices{T},
        Ek::Float64) where {T}
    cfg = velocity.poloidal.config
    nr = length(split.d1_row_inner)
    P = Vector{T}(undef, nr); W = Vector{T}(undef, nr)
    for (p_arr, v_arr) in (
        (parent(velocity.poloidal.data_real), parent(velocity.work_pol.data_real)),
        (parent(velocity.poloidal.data_imag), parent(velocity.work_pol.data_imag)),
    )
        @inbounds for lm in 1:cfg.nlm
            slot = local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]
            if l == 0
                for r_idx in 1:nr
                    set_local_spectral_value!(v_arr, slot, r_idx, zero(T))
                end
                continue
            end
            idx = split.lookup[l]
            for r_idx in 1:nr
                P[r_idx] = local_spectral_value(p_arr, slot, r_idx)
            end
            mul!(W, split.dpol_op[idx], P)
            for r_idx in 1:nr
                set_local_spectral_value!(v_arr, slot, r_idx, T(Ek) * W[r_idx])
            end
        end
    end
    return velocity
end

# P ← recover(V) with Dirichlet walls + φ1-column influence corrections.
# `half` selects the stage (dt/2, phi1_half) vs finalize (dt, phi1_full)
# Green responses. cache_lookup maps l → the cache's per-l index.
# Ball: the output is P = Pt + Σaᵢhᵢ with hᵢ = p_factor⁻¹R(gᵢ), so the
# implied recovery RHS is Wv + Σaᵢgᵢ. Row 1 enforces the W-regularity
# Robin functional on that composed object: ρ₁ on Wv, M[1,i] on the RAW
# gᵢ — no Ek factor (verified: residual on the corrected field is 0 to
# machine precision; an invEk here leaves (1−Ek)·ρ₁ uncorrected).
function _erk2_poloidal_recover!(velocity, split::PoloidalSplitMatrices{T},
        cache, cache_lookup, dt::Float64, Ek::Float64, half::Bool) where {T}
    cfg = velocity.poloidal.config
    nr = length(split.d1_row_inner)
    c = half ? dt / 2 : dt
    phis = half ? cache.phi1_half : cache.phi1_full
    Wv = Vector{T}(undef, nr); Pt = Vector{T}(undef, nr)
    g = Vector{T}(undef, nr)
    h1 = Vector{T}(undef, nr); h2 = Vector{T}(undef, nr)
    invEk = 1.0 / Ek
    for (p_arr, v_arr) in (
        (parent(velocity.poloidal.data_real), parent(velocity.work_pol.data_real)),
        (parent(velocity.poloidal.data_imag), parent(velocity.work_pol.data_imag)),
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
            cidx = get(cache_lookup, l, 0)
            cidx == 0 && error("missing ERK2 cache entry for l=$l in poloidal recovery")
            for r_idx in 1:nr
                Wv[r_idx] = invEk * local_spectral_value(v_arr, slot, r_idx)
            end
            # Ball: inner W-regularity residual, read off Wv BEFORE the
            # wall-zeroing below consumes it.
            rho1w = split.ball ?
                    dot(split.d1_row_inner, Wv) -
                    T((l + 1) * split.reg_r_inv) * Wv[1] : zero(T)
            Wv[1] = zero(T); Wv[nr] = zero(T)
            solve_banded!(Pt, split.p_factor[idx], Wv)

            # Green responses through the SAME recovery (R zeroes the walls).
            # Ball: row 1 applies ρ to the RAW g columns (no Ek factor) —
            # the same aᵢ multiplies both gᵢ (row 1) and hᵢ (output).
            phi = phis[cidx]
            for r_idx in 1:nr
                g[r_idx] = c * phi[r_idx, 1]
            end
            m11b = split.ball ?
                   dot(split.d1_row_inner, g) -
                   T((l + 1) * split.reg_r_inv) * g[1] : zero(T)
            g[1] = zero(T); g[nr] = zero(T)
            solve_banded!(h1, split.p_factor[idx], g)
            for r_idx in 1:nr
                g[r_idx] = c * phi[r_idx, nr]
            end
            m12b = split.ball ?
                   dot(split.d1_row_inner, g) -
                   T((l + 1) * split.reg_r_inv) * g[1] : zero(T)
            g[1] = zero(T); g[nr] = zero(T)
            solve_banded!(h2, split.p_factor[idx], g)

            if split.ball
                m11 = m11b; m12 = m12b
                r1 = rho1w
            else
                m11 = dot(split.d1_row_inner, h1)
                m12 = dot(split.d1_row_inner, h2)
                r1 = dot(split.d1_row_inner, Pt)
            end
            m21 = dot(split.d1_row_outer, h1); m22 = dot(split.d1_row_outer, h2)
            r2 = dot(split.d1_row_outer, Pt)
            det = m11 * m22 - m12 * m21
            a1 = (-r1 * m22 + r2 * m12) / det
            a2 = (-r2 * m11 + r1 * m21) / det
            for r_idx in 1:nr
                set_local_spectral_value!(p_arr, slot, r_idx,
                    Pt[r_idx] + a1 * h1[r_idx] + a2 * h2[r_idx])
            end
        end
    end
    return velocity
end
