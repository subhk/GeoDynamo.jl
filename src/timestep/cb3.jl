# ================================================================================
# Cavaglieri-Bewley / Williamson 2N-storage IMEX-RK3
# ================================================================================

const CB3_GAMMA = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)
const CB3_ZETA = (0.0, -17.0 / 60.0, -5.0 / 12.0)
# Companion Crank-Nicolson coefficients for the implicit (diffusion) operator L
# of the Spalart-Moser-Rogers / Cavaglieri-Bewley IMEX-RK3. Substage k is
#   (I − β_k·dt·L) φ^k = φ^{k-1} + dt[γ_k N^{k-1} + ζ_k N^{k-2} + α_k·L·φ^{k-1}]
# with α explicit and β implicit. Per stage α_k + β_k = γ_k + ζ_k and the totals
# Σα = Σβ = 1/2 (Σ(α+β) = 1), so the linear operator advances by exactly dt and
# the scheme is 2nd-order consistent for both the nonlinear and the diffusion
# terms. (Treating L fully implicitly with the explicit weights γ, as a prior
# version did, gave Σ = 1.7 and over-integrated diffusion ~70% per step.)
const CB3_ALPHA = (29.0 / 96.0, -3.0 / 40.0, 1.0 / 6.0)
const CB3_BETA = (37.0 / 160.0, 5.0 / 24.0, 1.0 / 6.0)

"""
    solver_build_rhs_cb3_stage!(rhs, u, n, nprev, matrices, dt, gamma, zeta, alpha; mass_coeff, work)

Build the SMR/Cavaglieri-Bewley IMEX-RK3 substage RHS

    rhs = (mass_coeff/dt)·u + γ·N + ζ·N_prev + α·(L·u)

consumed by the implicit solve against the system matrix `(mass_coeff/dt) I −
β·L` (built per stage with `theta = β`). `L` is the diffusivity-scaled linear
operator carried in `matrices.linear_matrices`; the explicit `α·L·u` term is the
companion Crank-Nicolson half and mirrors the CNAB2 `(1−θ)·L·u` carry-over in
`solver_build_rhs_cnab2!`. Per-mode and radius-coupled only, so assembly needs no
inter-rank communication (each mode owns its full radial profile locally).
"""
function solver_build_rhs_cb3_stage!(
        rhs::SpectralFieldType{T},
        u::SpectralFieldType{T},
        n::SpectralFieldType{T},
        nprev::SpectralFieldType{T},
        matrices::ImplicitMatrixSet{T},
        dt::Float64,
        gamma::Float64,
        zeta::Float64,
        alpha::Float64;
        mass_coeff::Float64 = 1.0,
        work::Union{SolverRadialWork{T}, Nothing} = nothing,
) where {T}
    rhs_real = parent(rhs.data_real)
    rhs_imag = parent(rhs.data_imag)
    u_real = parent(u.data_real)
    u_imag = parent(u.data_imag)
    n_real = parent(n.data_real)
    n_imag = parent(n.data_imag)
    p_real = parent(nprev.data_real)
    p_imag = parent(nprev.data_imag)

    r_range = local_range(u.pencil, 3)
    inv_stage_dt = T(mass_coeff / dt)
    γT = T(gamma)
    ζT = T(zeta)
    αT = T(alpha)
    add_linear = !iszero(alpha)

    nr_global = add_linear ? matrices.system_matrices[1].size : 0
    work_ok = work !== nothing && length(work.u_real_global) == nr_global
    u_real_global = add_linear ? (work_ok ? work.u_real_global : zeros(T, nr_global)) : T[]
    u_imag_global = add_linear ? (work_ok ? work.u_imag_global : zeros(T, nr_global)) : T[]
    linear_real = add_linear ? (work_ok ? work.linear_real : zeros(T, nr_global)) : T[]
    linear_imag = add_linear ? (work_ok ? work.linear_imag : zeros(T, nr_global)) : T[]

    @inbounds for lm_idx in 1:u.nlm
        slot = local_spectral_storage_slot(u.config, lm_idx)
        slot === nothing && continue

        l = u.config.l_values[lm_idx]

        if add_linear
            matrix_idx = get(matrices.lookup, l, nothing)
            matrix_idx === nothing && error("Missing implicit matrix for l=$l")
            fill!(u_real_global, zero(T))
            fill!(u_imag_global, zero(T))
            gather_local_radial_profile!(
                u_real_global, u_imag_global, u_real, u_imag, slot, r_range)
            fill!(linear_real, zero(T))
            fill!(linear_imag, zero(T))
            apply_banded_full!(linear_real, matrices.linear_matrices[matrix_idx], u_real_global)
            apply_banded_full!(linear_imag, matrices.linear_matrices[matrix_idx], u_imag_global)
        end

        for r_idx in r_range
            local_r = r_idx - first(r_range) + 1
            local_r <= size(rhs_real, 3) || continue
            vr = inv_stage_dt * local_spectral_value(u_real, slot, local_r) +
                 γT * local_spectral_value(n_real, slot, local_r) +
                 ζT * local_spectral_value(p_real, slot, local_r)
            vi = inv_stage_dt * local_spectral_value(u_imag, slot, local_r) +
                 γT * local_spectral_value(n_imag, slot, local_r) +
                 ζT * local_spectral_value(p_imag, slot, local_r)
            if add_linear
                vr += αT * linear_real[r_idx]
                vi += αT * linear_imag[r_idx]
            end
            set_local_spectral_value!(rhs_real, slot, local_r, vr)
            set_local_spectral_value!(rhs_imag, slot, local_r, vi)
        end
    end
    return rhs
end

function _cb3_stage_matrices(state::SolverState{T, <:AbstractArchitecture},
        beta::Float64) where {T}
    # System matrix (mass/dt) I − β·L with the FULL step dt: the implicit
    # diffusion weight is the companion CN coefficient β (not the explicit γ),
    # and matrices.linear_matrices carries the bare L for the explicit α·L term.
    matrices, magnetic_ic_admittance = _build_implicit_matrices_dict(
        T,
        state.backend.shtns_config,
        state.backend.outer_core_domain,
        state.backend.inner_core_domain,
        state.parameters,
        state.parameters.timestep;
        theta = beta,
    )
    magnetic_ic_admittance === nothing || throw(ArgumentError(
        "RungeKutta3() does not yet support magnetic_inner_bc=:conducting_inner_core"))
    return create_solver_implicit_matrix_store(matrices)
end

# RK3's three substages use distinct γ, so the (γ·dt)-shifted implicit operators and
# poloidal W-split differ per stage. They depend only on (γ, dt, parameters, geometry);
# parameters/geometry are fixed per run, so we cache per stage and invalidate when dt
# changes. This replaces a full rebuild + LU-refactorization of every operator on every
# substage (3×/step) with a build-once-per-(stage, dt).
function _cb3_invalidate_caches_if_dt_changed!(state::SolverState)
    caches = state.timestep_caches
    if caches.cb3_built_dt != state.parameters.timestep
        fill!(caches.cb3_stage_matrices, nothing)
        fill!(caches.cb3_poloidal_split, nothing)
        caches.cb3_built_dt = state.parameters.timestep
    end
    return nothing
end

function _get_or_build_cb3_stage_matrices!(state::SolverState{T, <:AbstractArchitecture},
        stage::Int, beta::Float64) where {T}
    caches = state.timestep_caches
    cached = caches.cb3_stage_matrices[stage]
    cached === nothing || return cached::Dict{Symbol, ImplicitMatrixSet{T}}
    built = _cb3_stage_matrices(state, beta)
    caches.cb3_stage_matrices[stage] = built
    return built::Dict{Symbol, ImplicitMatrixSet{T}}
end

function _get_or_build_cb3_poloidal_split!(state::SolverState{T, <:AbstractArchitecture},
        stage::Int, beta::Float64, velocity_bc::Int) where {T}
    caches = state.timestep_caches
    cached = caches.cb3_poloidal_split[stage]
    cached === nothing || return cached::PoloidalSplitMatrices{T}
    # W-advance system (Ek/dt) I − β·(Ek·D_pol) with the FULL step dt; split.w_linear
    # carries Ek·D_pol for the explicit α·(Ek·D_pol)·W term.
    split = create_velocity_poloidal_split_matrices(
        state.runtime.shtns_config,
        state.runtime.outer_core_domain,
        state.parameters.Ek,
        state.parameters.timestep;
        velocity_bc_code = velocity_bc,
        theta = beta,
        T = T,
        ball = state.parameters.geometry === :ball,
    )
    caches.cb3_poloidal_split[stage] = split
    return split
end

function _cb3_apply_scalar_stage!(state, field, key::Symbol, solve_step!, matrices,
        gamma::Float64, zeta::Float64, alpha::Float64; mass_coeff::Float64 = 1.0)
    mset = matrices[key]
    radial_work = get_radial_work!(
        state.timestep_caches,
        key,
        mset.system_matrices[1].size,
    )
    solver_build_rhs_cb3_stage!(
        field.work_spectral,
        field.spectral,
        field.nonlinear,
        field.prev_nonlinear,
        mset,
        state.parameters.timestep,
        gamma,
        zeta,
        alpha;
        mass_coeff,
        work = radial_work,
    )
    bc = get_bc_vectors(field)
    solve_step!(
        field.spectral,
        field.work_spectral,
        mset;
        bc_inner = bc.inner_real,
        bc_outer = bc.outer_real,
        bc_inner_imag = bc.inner_imag,
        bc_outer_imag = bc.outer_imag,
        work = radial_work,
    )
    return state
end

function _cb3_apply_velocity_toroidal_stage!(state::SolverState{T, <:AbstractArchitecture},
        matrices, gamma::Float64, zeta::Float64, alpha::Float64) where {T}
    velocity = state.fields.velocity
    mset = matrices[:velocity_tor]
    radial_work = get_radial_work!(
        state.timestep_caches,
        :velocity_toroidal,
        mset.system_matrices[1].size,
    )
    solver_build_rhs_cb3_stage!(
        velocity.work_tor,
        velocity.toroidal,
        velocity.nl_toroidal,
        velocity.prev_nl_toroidal,
        mset,
        state.parameters.timestep,
        gamma,
        zeta,
        alpha;
        mass_coeff = state.parameters.Ek,
        work = radial_work,
    )
    solver_solve_velocity_implicit_step!(
        velocity.toroidal,
        velocity.work_tor,
        mset,
        :toroidal;
        velocity_bc_code = _velocity_bc_code(state.parameters.velocity_bcs),
        domain = state.runtime.outer_core_domain,
        bc_inner = view(velocity.toroidal.boundary_values, 1, :),
        bc_outer = view(velocity.toroidal.boundary_values, 2, :),
        bc_inner_imag = view(velocity.toroidal.boundary_values_imag, 1, :),
        bc_outer_imag = view(velocity.toroidal.boundary_values_imag, 2, :),
        work = radial_work,
    )
    return state
end

function _cb3_apply_poloidal_wsplit_stage!(state::SolverState{T, <:AbstractArchitecture},
        stage::Int, gamma::Float64, zeta::Float64, alpha::Float64, beta::Float64) where {T}
    velocity = state.fields.velocity
    velocity_bc = _velocity_bc_code(state.parameters.velocity_bcs)
    split = _get_or_build_cb3_poloidal_split!(state, stage, beta, velocity_bc)
    domain = state.runtime.outer_core_domain
    cfg = velocity.poloidal.config
    nr = domain.N
    r_range = local_range(velocity.poloidal.pencil, 3)
    length(r_range) == nr || error(
        "RungeKutta3 poloidal W-split requires the radial axis fully local " *
        "(got $(length(r_range)) of $nr levels)")

    inv_stage_dt = T(state.parameters.Ek / state.parameters.timestep)
    γT = T(gamma)
    ζT = T(zeta)
    αT = T(alpha)

    P, W, LW, rhs, Wp, Pp = split.work   # cached per-step radial scratch

    # Topography impermeability correction modifies the complex P wall value.
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
            mul!(LW, split.w_linear[idx], W)      # Ek·D_pol·W (explicit CN term)
            for r_idx in 1:nr
                rhs[r_idx] = inv_stage_dt * W[r_idx] + αT * LW[r_idx] +
                             γT * local_spectral_value(n_arr, slot, r_idx) +
                             ζT * local_spectral_value(pn_arr, slot, r_idx)
            end
            solve_banded!(Wp, split.w_factor[idx], rhs)
            rho1w = split.ball ?
                    dot(split.d1_row_inner, Wp) -
                    T((l + 1) * split.reg_r_inv) * Wp[1] : zero(T)
            # Dirichlet P-recovery wall RHS = imposed P value (base 0 + topography
            # complex correction). Ball inner row is regularity ⇒ never inject there.
            Wp[1] = !split.ball ? pol_bv[1, lm] : zero(T)
            Wp[nr] = pol_bv[2, lm]
            solve_banded!(Pp, split.p_factor[idx], Wp)

            rho1 = split.ball ? rho1w : dot(split.d1_row_inner, Pp)
            rho2 = dot(split.d1_row_outer, Pp)
            M = split.influence[idx]
            det = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]
            a1 = (-rho1 * M[2, 2] + rho2 * M[1, 2]) / det
            a2 = (-rho2 * M[1, 1] + rho1 * M[2, 1]) / det
            h1 = split.h1[idx]
            h2 = split.h2[idx]
            for r_idx in 1:nr
                set_local_spectral_value!(
                    p_arr, slot, r_idx, Pp[r_idx] + a1 * h1[r_idx] + a2 * h2[r_idx])
            end
        end
    end
    return state
end

function _cb3_apply_magnetic_stage!(state::SolverState{T, <:AbstractArchitecture},
        matrices, gamma::Float64, zeta::Float64, alpha::Float64) where {T}
    magnetic = state.fields.magnetic
    magnetic === nothing && return state
    state.magnetic_ic_admittance === nothing || throw(ArgumentError(
        "RungeKutta3() does not yet support magnetic_inner_bc=:conducting_inner_core"))

    for (key, workkey, solution, nonlinear, prev, work, component) in (
        (:magnetic_tor, :magnetic_toroidal, magnetic.toroidal, magnetic.nl_toroidal,
         magnetic.prev_nl_toroidal, magnetic.work_tor, :toroidal),
        (:magnetic_pol, :magnetic_poloidal, magnetic.poloidal, magnetic.nl_poloidal,
         magnetic.prev_nl_poloidal, magnetic.work_pol, :poloidal),
    )
        mset = matrices[key]
        radial_work = get_radial_work!(
            state.timestep_caches,
            workkey,
            mset.system_matrices[1].size,
        )
        solver_build_rhs_cb3_stage!(
            work,
            solution,
            nonlinear,
            prev,
            mset,
            state.parameters.timestep,
            gamma,
            zeta,
            alpha;
            mass_coeff = 1.0,
            work = radial_work,
        )
        solver_solve_magnetic_implicit_step!(
            solution,
            work,
            mset,
            component;
            _topo_mag_bc(solution)...,
            work = radial_work,
        )
    end
    return state
end

function _cb3_apply_stage!(state::SolverState, stage::Int, gamma::Float64, zeta::Float64,
        alpha::Float64, beta::Float64)
    matrices = _get_or_build_cb3_stage_matrices!(state, stage, beta)

    _cb3_apply_velocity_toroidal_stage!(state, matrices, gamma, zeta, alpha)
    _cb3_apply_poloidal_wsplit_stage!(state, stage, gamma, zeta, alpha, beta)
    _cb3_apply_magnetic_stage!(state, matrices, gamma, zeta, alpha)

    _cb3_apply_scalar_stage!(
        state,
        state.fields.temperature,
        :temperature,
        solver_solve_temperature_implicit_step!,
        matrices,
        gamma,
        zeta,
        alpha,
    )
    if state.fields.composition !== nothing
        _cb3_apply_scalar_stage!(
            state,
            state.fields.composition,
            :composition,
            solver_solve_composition_implicit_step!,
            matrices,
            gamma,
            zeta,
            alpha,
        )
    end
    return state
end

"""
    integrate_solver_cb3_step!(state)

Advance one complete Cavaglieri-Bewley/Williamson 2N-storage IMEX-RK3 step.
`solver_step!` computes the first nonlinear pass and topography correction
before entering this function; substages 2 and 3 recompute them here.
"""
function integrate_solver_cb3_step!(state::SolverState)
    _cb3_invalidate_caches_if_dt_changed!(state)
    for stage in 1:3
        stage == 1 || begin
            compute_solver_nonlinear_terms!(state)
            apply_solver_topography!(state)
        end
        _cb3_apply_stage!(state, stage, CB3_GAMMA[stage], CB3_ZETA[stage],
            CB3_ALPHA[stage], CB3_BETA[stage])
        _sync_solver_nonlinear_histories!(
            state,
            state.parameters.include_magnetic && state.fields.magnetic !== nothing,
        )
    end
    return state
end
