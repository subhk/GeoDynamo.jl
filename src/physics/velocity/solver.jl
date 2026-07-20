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

function finish_velocity_nonlinear!(
        velocity_fields, temperature_field, composition_field, params)
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
    # Buoyancy is exactly linear in the scalar fields and purely radial, so it
    # contributes ONLY to Q_F (raw_spheroidal analysis above consumes just v_θ,v_φ,
    # so S_F never sees it). The physical force factor·r·scalar(θ,φ) analyses to
    # factor·r·scalar_spec (factor·r is constant per radial level), so inject it in
    # spectral space directly instead of synthesizing T/C to the grid every step.
    _add_spectral_buoyancy!(
        velocity_fields, temperature_field, composition_field, params)
    _poloidal_force_projection!(velocity_fields)
    return velocity_fields.nl_toroidal, velocity_fields.nl_poloidal
end

# Add the buoyancy force to Q_F (velocity_fields.work_pol) directly in spectral
# space — buoyancy is linear in the scalar, so this needs no transform:
#   Q_F[l,m,r] += factor · r · scalar_spec[l,m,r]
# with factor = (Pm/Pr)·Ra (temperature) and (Pm/Sc)·RaC (composition), and the
# radial weight r = domain.r[r,4]. This reproduces the former physical-space
# buoyancy (a scalar synthesis + force + radial analysis) with the same prefactors.
function _add_spectral_buoyancy!(
        velocity_fields, temperature_field, composition_field, params)
    if temperature_field !== nothing
        _accumulate_spectral_buoyancy!(
            velocity_fields, temperature_field.spectral,
            (params.Pm / params.Pr) * params.Ra)
    end
    if composition_field !== nothing
        _accumulate_spectral_buoyancy!(
            velocity_fields, composition_field.spectral,
            (params.Pm / params.Sc) * params.RaC)
    end
    return velocity_fields
end

function _accumulate_spectral_buoyancy!(velocity_fields, scalar_spec, factor)
    iszero(factor) && return velocity_fields
    cfg = velocity_fields.work_pol.config
    domain = velocity_fields.domain
    T = eltype(parent(velocity_fields.work_pol.data_real))
    nr = domain.N
    r_range = local_range(velocity_fields.work_pol.pencil, 3)
    length(r_range) == nr || error(
        "spectral buoyancy injection requires the radial axis fully local " *
        "(got $(length(r_range)) of $nr levels); r-distributed support is a follow-up")
    for (q_arr, s_arr) in (
        (parent(velocity_fields.work_pol.data_real), parent(scalar_spec.data_real)),
        (parent(velocity_fields.work_pol.data_imag), parent(scalar_spec.data_imag)),
    )
        @inbounds for lm in 1:cfg.nlm
            slot = local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            for r_idx in 1:nr
                fr = factor * domain.r[r_idx, 4]
                q = local_spectral_value(q_arr, slot, r_idx)
                set_local_spectral_value!(q_arr, slot, r_idx,
                    q + T(fr * local_spectral_value(s_arr, slot, r_idx)))
            end
        end
    end
    return velocity_fields
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
            bc_inner_imag = view(velocity.toroidal.boundary_values_imag, 1, :),
            bc_outer_imag = view(velocity.toroidal.boundary_values_imag, 2, :),
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
            bc_inner_imag = view(velocity.toroidal.boundary_values_imag, 1, :),
            bc_outer_imag = view(velocity.toroidal.boundary_values_imag, 2, :),
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

    # Topography impermeability correction modifies the complex P wall VALUE.
    # Injected as the Dirichlet RHS of the
    # P-recovery (Wp[1]/Wp[nr]); zero when topography is disabled ⇒ unchanged.
    pol_bv_real = velocity.poloidal.boundary_values
    pol_bv_imag = velocity.poloidal.boundary_values_imag

    for (pol_bv, p_arr, n_arr, pn_arr) in (
        (pol_bv_real,
         parent(velocity.poloidal.data_real),
         parent(velocity.nl_poloidal.data_real),
         parent(velocity.prev_nl_poloidal.data_real)),
        (pol_bv_imag,
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
            # Base impermeability is P=0; the topography correction shifts the
            # wall value. Ball: row 1 is the
            # regularity Robin (homogeneous RHS) — never inject there.
            Wp[1] = !split.ball ? pol_bv[1, lm] : zero(T)
            Wp[nr] = pol_bv[2, lm]
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
    P, W = split.work[1], split.work[2]
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

# Lazily build (and cache on the state) the ERK2 P-recovery Green responses.
# The φ1-column responses hᵢ and the 2×2 influence matrix depend only on the
# degree l and on the stage half (dt/2 vs dt) — not on m and not on the
# real/imag half of the spectral field — so they are built once per
# (split, ERK2 cache, dt) instead of once per mode per half.
function _get_or_build_erk2_poloidal_green!(caches::TimestepCaches{T},
        split::PoloidalSplitMatrices{T}, cache::ERK2StageCache{T},
        dt::Float64) where {T}
    green = caches.erk2_poloidal_green
    if green !== nothing && green.dt == dt &&
       green.split === split && green.cache === cache
        return green::ERK2PoloidalGreenCache{T}
    end

    nr = length(split.d1_row_inner)
    nl = length(cache.l_values)
    h1_half = [zeros(T, nr) for _ in 1:nl]
    h2_half = [zeros(T, nr) for _ in 1:nl]
    h1_full = [zeros(T, nr) for _ in 1:nl]
    h2_full = [zeros(T, nr) for _ in 1:nl]
    influence_half = [zeros(T, 2, 2) for _ in 1:nl]
    influence_full = [zeros(T, 2, 2) for _ in 1:nl]
    det_half = zeros(T, nl)
    det_full = zeros(T, nl)
    g = Vector{T}(undef, nr)

    for (cidx, l) in enumerate(cache.l_values)
        l == 0 && continue                       # l = 0 carries no poloidal flow
        idx = get(split.lookup, l, 0)
        idx == 0 && continue
        for (c, phi, h1, h2, M, det) in (
            (dt / 2, cache.phi1_half[cidx], h1_half[cidx], h2_half[cidx],
             influence_half[cidx], det_half),
            (dt, cache.phi1_full[cidx], h1_full[cidx], h2_full[cidx],
             influence_full[cidx], det_full),
        )
            # Green responses through the SAME recovery (R zeroes the walls).
            # Ball: row 1 applies ρ to the RAW g columns (no Ek factor) —
            # the same aᵢ multiplies both gᵢ (row 1) and hᵢ (output).
            @inbounds for r_idx in 1:nr
                g[r_idx] = c * phi[r_idx, 1]
            end
            m11b = split.ball ?
                   dot(split.d1_row_inner, g) -
                   T((l + 1) * split.reg_r_inv) * g[1] : zero(T)
            g[1] = zero(T); g[nr] = zero(T)
            solve_banded!(h1, split.p_factor[idx], g)
            @inbounds for r_idx in 1:nr
                g[r_idx] = c * phi[r_idx, nr]
            end
            m12b = split.ball ?
                   dot(split.d1_row_inner, g) -
                   T((l + 1) * split.reg_r_inv) * g[1] : zero(T)
            g[1] = zero(T); g[nr] = zero(T)
            solve_banded!(h2, split.p_factor[idx], g)

            if split.ball
                M[1, 1] = m11b; M[1, 2] = m12b
            else
                M[1, 1] = dot(split.d1_row_inner, h1)
                M[1, 2] = dot(split.d1_row_inner, h2)
            end
            M[2, 1] = dot(split.d1_row_outer, h1)
            M[2, 2] = dot(split.d1_row_outer, h2)
            det[cidx] = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]
        end
    end

    green = ERK2PoloidalGreenCache{T}(dt, h1_half, h2_half, h1_full, h2_full,
        influence_half, influence_full, det_half, det_full, split, cache)
    caches.erk2_poloidal_green = green
    return green
end

# P ← recover(V) with Dirichlet walls + φ1-column influence corrections.
# `half` selects the stage (dt/2, phi1_half) vs finalize (dt, phi1_full)
# Green responses, precomputed per degree in `green`. cache_lookup maps
# l → the cache's per-l index.
# Ball: the output is P = Pt + Σaᵢhᵢ with hᵢ = p_factor⁻¹R(gᵢ), so the
# implied recovery RHS is Wv + Σaᵢgᵢ. Row 1 enforces the W-regularity
# Robin functional on that composed object: ρ₁ on Wv, M[1,i] on the RAW
# gᵢ — no Ek factor (verified: residual on the corrected field is 0 to
# machine precision; an invEk here leaves (1−Ek)·ρ₁ uncorrected).
function _erk2_poloidal_recover!(velocity, split::PoloidalSplitMatrices{T},
        green::ERK2PoloidalGreenCache{T}, cache_lookup, Ek::Float64,
        half::Bool) where {T}
    cfg = velocity.poloidal.config
    nr = length(split.d1_row_inner)
    h1s = half ? green.h1_half : green.h1_full
    h2s = half ? green.h2_half : green.h2_full
    Ms = half ? green.influence_half : green.influence_full
    dets = half ? green.det_half : green.det_full
    Wv, Pt = split.work[1], split.work[2]
    invEk = 1.0 / Ek
    pol_bv_real = velocity.poloidal.boundary_values
    pol_bv_imag = velocity.poloidal.boundary_values_imag
    for (pol_bv, p_arr, v_arr) in (
        (pol_bv_real, parent(velocity.poloidal.data_real),
            parent(velocity.work_pol.data_real)),
        (pol_bv_imag, parent(velocity.poloidal.data_imag),
            parent(velocity.work_pol.data_imag)),
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
            Wv[1] = !split.ball ? pol_bv[1, lm] : zero(T)
            Wv[nr] = pol_bv[2, lm]
            solve_banded!(Pt, split.p_factor[idx], Wv)

            h1 = h1s[cidx]; h2 = h2s[cidx]; M = Ms[cidx]
            m11 = M[1, 1]; m12 = M[1, 2]; m21 = M[2, 1]; m22 = M[2, 2]
            r1 = split.ball ? rho1w : dot(split.d1_row_inner, Pt)
            r2 = dot(split.d1_row_outer, Pt)
            det = dets[cidx]
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
