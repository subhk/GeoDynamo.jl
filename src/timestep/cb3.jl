# ================================================================================
# Cavaglieri-Bewley / Williamson 2N-storage IMEX-RK3
# ================================================================================

const CB3_GAMMA = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)
const CB3_ZETA = (0.0, -17.0 / 60.0, -5.0 / 12.0)

"""
    solver_build_rhs_cb3_stage!(rhs, u, n, nprev, dt, gamma, zeta; mass_coeff)

Build the linearly implicit low-storage RK3 substage RHS. The substage equation
is scaled by `1 / gamma` so it can reuse the existing shifted matrix form
`(mass_coeff / (gamma*dt)) I - L`.
"""
function solver_build_rhs_cb3_stage!(
        rhs::SpectralFieldType{T},
        u::SpectralFieldType{T},
        n::SpectralFieldType{T},
        nprev::SpectralFieldType{T},
        dt::Float64,
        gamma::Float64,
        zeta::Float64;
        mass_coeff::Float64 = 1.0
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
    inv_stage_dt = T(mass_coeff / (gamma * dt))
    zeta_over_gamma = T(zeta / gamma)

    @inbounds for lm_idx in 1:u.nlm
        slot = local_spectral_storage_slot(u.config, lm_idx)
        slot === nothing && continue
        for r_idx in r_range
            local_r = r_idx - first(r_range) + 1
            local_r <= size(rhs_real, 3) || continue
            set_local_spectral_value!(
                rhs_real, slot, local_r,
                inv_stage_dt * local_spectral_value(u_real, slot, local_r) +
                local_spectral_value(n_real, slot, local_r) +
                zeta_over_gamma * local_spectral_value(p_real, slot, local_r),
            )
            set_local_spectral_value!(
                rhs_imag, slot, local_r,
                inv_stage_dt * local_spectral_value(u_imag, slot, local_r) +
                local_spectral_value(n_imag, slot, local_r) +
                zeta_over_gamma * local_spectral_value(p_imag, slot, local_r),
            )
        end
    end
    return rhs
end

function _cb3_stage_matrices(state::SolverState{T, <:AbstractArchitecture},
        gamma::Float64) where {T}
    matrices, magnetic_ic_admittance = _build_implicit_matrices_dict(
        T,
        state.backend.shtns_config,
        state.backend.outer_core_domain,
        state.backend.inner_core_domain,
        state.parameters,
        gamma * state.parameters.timestep;
        theta = 1.0,
    )
    magnetic_ic_admittance === nothing || throw(ArgumentError(
        "RungeKutta3() does not yet support magnetic_inner_bc=:conducting_inner_core"))
    return create_solver_implicit_matrix_store(matrices)
end

function _cb3_apply_scalar_stage!(state, field, key::Symbol, solve_step!, matrices,
        gamma::Float64, zeta::Float64; mass_coeff::Float64 = 1.0)
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
        state.parameters.timestep,
        gamma,
        zeta;
        mass_coeff,
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
        matrices, gamma::Float64, zeta::Float64) where {T}
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
        state.parameters.timestep,
        gamma,
        zeta;
        mass_coeff = state.parameters.Ek,
    )
    solver_solve_velocity_implicit_step!(
        velocity.toroidal,
        velocity.work_tor,
        mset,
        :toroidal;
        velocity_bc_code = _velocity_bc_code(state.parameters.velocity_bcs),
        domain = state.runtime.outer_core_domain,
        work = radial_work,
    )
    return state
end

function _cb3_apply_poloidal_wsplit_stage!(state::SolverState{T, <:AbstractArchitecture},
        gamma::Float64, zeta::Float64) where {T}
    velocity = state.fields.velocity
    velocity_bc = _velocity_bc_code(state.parameters.velocity_bcs)
    split = create_velocity_poloidal_split_matrices(
        state.runtime.shtns_config,
        state.runtime.outer_core_domain,
        state.parameters.Ek,
        gamma * state.parameters.timestep;
        velocity_bc_code = velocity_bc,
        theta = 1.0,
        T = T,
    )
    domain = state.runtime.outer_core_domain
    cfg = velocity.poloidal.config
    nr = domain.N
    r_range = local_range(velocity.poloidal.pencil, 3)
    length(r_range) == nr || error(
        "RungeKutta3 poloidal W-split requires the radial axis fully local " *
        "(got $(length(r_range)) of $nr levels)")

    inv_stage_dt = T(state.parameters.Ek / (gamma * state.parameters.timestep))
    zeta_over_gamma = T(zeta / gamma)

    P = Vector{T}(undef, nr)
    W = Vector{T}(undef, nr)
    rhs = Vector{T}(undef, nr)
    Wp = Vector{T}(undef, nr)
    Pp = Vector{T}(undef, nr)

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
            mul!(W, split.dpol_op[idx], P)
            for r_idx in 1:nr
                rhs[r_idx] = inv_stage_dt * W[r_idx] +
                             local_spectral_value(n_arr, slot, r_idx) +
                             zeta_over_gamma * local_spectral_value(pn_arr, slot, r_idx)
            end
            solve_banded!(Wp, split.w_factor[idx], rhs)
            Wp[1] = zero(T)
            Wp[nr] = zero(T)
            solve_banded!(Pp, split.p_factor[idx], Wp)

            rho1 = dot(split.d1_row_inner, Pp)
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
        matrices, gamma::Float64, zeta::Float64) where {T}
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
            state.parameters.timestep,
            gamma,
            zeta;
            mass_coeff = 1.0,
        )
        solver_solve_magnetic_implicit_step!(
            solution,
            work,
            mset,
            component;
            work = radial_work,
        )
    end
    return state
end

function _cb3_apply_stage!(state::SolverState, gamma::Float64, zeta::Float64)
    matrices = _cb3_stage_matrices(state, gamma)

    _cb3_apply_velocity_toroidal_stage!(state, matrices, gamma, zeta)
    _cb3_apply_poloidal_wsplit_stage!(state, gamma, zeta)
    _cb3_apply_magnetic_stage!(state, matrices, gamma, zeta)

    _cb3_apply_scalar_stage!(
        state,
        state.fields.temperature,
        :temperature,
        solver_solve_temperature_implicit_step!,
        matrices,
        gamma,
        zeta,
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
    for stage in 1:3
        stage == 1 || begin
            compute_solver_nonlinear_terms!(state)
            apply_solver_topography!(state)
        end
        _cb3_apply_stage!(state, CB3_GAMMA[stage], CB3_ZETA[stage])
        _sync_solver_nonlinear_histories!(
            state,
            state.parameters.include_magnetic && state.fields.magnetic !== nothing,
        )
    end
    return state
end
