# =============================================================================
# GPU Cavaglieri-Bewley / Williamson 2N-storage IMEX-RK3
# =============================================================================

function gpu_build_rhs_cb3_stage!(rr, ri, ur, ui, nr_, ni_, pr, pi_, Lur, Lui,
        inv_stage_dt, gamma, zeta, alpha)
    T = eltype(rr)
    a = T(inv_stage_dt)
    g = T(gamma)
    z = T(zeta)
    al = T(alpha)
    # rhs = (mass/dt)·u + γ·N + ζ·N_prev + α·(L·u)  (SMR/Cavaglieri-Bewley)
    @. rr = a * ur + g * nr_ + z * pr + al * Lur
    @. ri = a * ui + g * ni_ + z * pi_ + al * Lui
    return nothing
end

function _gpu_cb3_solve_field!(bundle, lin, lu, nr_, ni_, inv_stage_dt,
        gamma, zeta, alpha, bw::Int; apply_bc::Bool = true)
    # Explicit companion-CN term α·(L·u) over the current field.
    Lur = similar(bundle.spec_r)
    Lui = similar(bundle.spec_i)
    gpu_batched_banded_matvec_perl!(Lur, bundle.spec_r, lin, bw)
    gpu_batched_banded_matvec_perl!(Lui, bundle.spec_i, lin, bw)
    rr = similar(bundle.spec_r)
    ri = similar(bundle.spec_i)
    gpu_build_rhs_cb3_stage!(
        rr, ri,
        bundle.spec_r, bundle.spec_i,
        nr_, ni_,
        bundle.prev_nl_r, bundle.prev_nl_i,
        Lur, Lui,
        inv_stage_dt, gamma, zeta, alpha,
    )
    if apply_bc
        # `get(::NamedTuple, k, default)` evaluates `default` EAGERLY even when
        # the key exists — the old form allocated four discarded slice+`zero`
        # temporaries every call. `haskey` + one shared zero buffer is non-eager
        # (zbuf is materialized once and only when no key is missing it goes
        # unused; behavior is identical: missing keys still see zeros).
        zbuf = zero(@view bundle.spec_r[:, :, 1])
        gpu_implicit_solve_field!(
            rr, ri, lu,
            haskey(bundle, :bc_in_r) ? bundle.bc_in_r : zbuf,
            haskey(bundle, :bc_in_i) ? bundle.bc_in_i : zbuf,
            haskey(bundle, :bc_out_r) ? bundle.bc_out_r : zbuf,
            haskey(bundle, :bc_out_i) ? bundle.bc_out_i : zbuf,
            bw,
        )
    else
        nr = size(rr, 3)
        rr[:, :, 1] .= zero(eltype(rr)); ri[:, :, 1] .= zero(eltype(ri))
        rr[:, :, nr] .= zero(eltype(rr)); ri[:, :, nr] .= zero(eltype(ri))
        gpu_batched_banded_solve!(rr, rr, lu, bw)
        gpu_batched_banded_solve!(ri, ri, lu, bw)
    end
    bundle.spec_r .= rr
    bundle.spec_i .= ri
    return bundle
end

function _gpu_cb3_recover_poloidal!(P, W, ws, bw::Int)
    T = eltype(P)
    nr = size(P, 3)
    Wp = copy(W)
    Wp[:, :, 1] .= zero(T)
    Wp[:, :, nr] .= zero(T)
    Pt = similar(P)
    gpu_batched_banded_solve!(Pt, Wp, ws.plu, bw)
    rho1 = dropdims(sum(Pt .* reshape(ws.d1_inner, 1, 1, :); dims = 3); dims = 3)
    rho2 = dropdims(sum(Pt .* reshape(ws.d1_outer, 1, 1, :); dims = 3); dims = 3)
    M11 = reshape(@view(ws.M[1, 1, :]), :, 1); M12 = reshape(@view(ws.M[1, 2, :]), :, 1)
    M21 = reshape(@view(ws.M[2, 1, :]), :, 1); M22 = reshape(@view(ws.M[2, 2, :]), :, 1)
    det = @. M11 * M22 - M12 * M21
    det = @. det + T(det == 0)
    a1 = @. (-rho1 * M22 + rho2 * M12) / det
    a2 = @. (-rho2 * M11 + rho1 * M21) / det
    h1 = reshape(ws.h1, size(ws.h1, 1), 1, nr)
    h2 = reshape(ws.h2, size(ws.h2, 1), 1, nr)
    @. P = Pt + a1 * h1 + a2 * h2
    P[1, :, :] .= zero(T)
    return P
end

function _gpu_cb3_poloidal_stage!(pol, nl_r, nl_i, stage, inv_stage_dt,
        gamma, zeta, alpha, bw::Int)
    ws = stage.wsplit
    W_r = similar(pol.spec_r)
    W_i = similar(pol.spec_i)
    gpu_batched_banded_matvec_perl!(W_r, pol.spec_r, ws.dpol, bw)   # W = D_pol·P
    gpu_batched_banded_matvec_perl!(W_i, pol.spec_i, ws.dpol, bw)

    # Explicit companion-CN term α·(Ek·D_pol)·W.
    LW_r = similar(pol.spec_r)
    LW_i = similar(pol.spec_i)
    gpu_batched_banded_matvec_perl!(LW_r, W_r, ws.wlin, bw)
    gpu_batched_banded_matvec_perl!(LW_i, W_i, ws.wlin, bw)

    rr = similar(pol.spec_r)
    ri = similar(pol.spec_i)
    T = eltype(rr)
    a = T(inv_stage_dt)
    g = T(gamma)
    z = T(zeta)
    al = T(alpha)
    @. rr = a * W_r + al * LW_r + g * nl_r + z * pol.prev_nl_r
    @. ri = a * W_i + al * LW_i + g * nl_i + z * pol.prev_nl_i
    gpu_batched_banded_solve!(rr, rr, ws.wlu, bw)
    gpu_batched_banded_solve!(ri, ri, ws.wlu, bw)
    _gpu_cb3_recover_poloidal!(pol.spec_r, rr, ws, bw)
    _gpu_cb3_recover_poloidal!(pol.spec_i, ri, ws, bw)
    return pol
end

function _gpu_cb3_roll_stage_nl!(state, nl)
    v = state.velocity
    v.tor.prev_nl_r .= nl.vt_r; v.tor.prev_nl_i .= nl.vt_i
    v.pol.prev_nl_r .= nl.vp_r; v.pol.prev_nl_i .= nl.vp_i
    state.temperature.prev_nl_r .= nl.t_r
    state.temperature.prev_nl_i .= nl.t_i
    if state.magnetic !== nothing
        state.magnetic.tor.prev_nl_r .= nl.mt_r
        state.magnetic.tor.prev_nl_i .= nl.mt_i
        state.magnetic.pol.prev_nl_r .= nl.mp_r
        state.magnetic.pol.prev_nl_i .= nl.mp_i
    end
    if state.composition !== nothing
        state.composition.prev_nl_r .= nl.c_r
        state.composition.prev_nl_i .= nl.c_i
    end
    return state
end

"""
    gpu_cb3_solver_step!(state) -> nothing

Advance the dense GPU solver state by one RungeKutta3 IMEX-RK3 step. `state` must come
from `build_gpu_solver_state(cpu_state)` where `cpu_state.parameters.timestepper
isa RungeKutta3`.
"""
function gpu_cb3_solver_step!(state)
    state.cb3 === nothing && error("gpu_cb3_solver_step!: state was not built for RungeKutta3()")
    bw = state.bw
    dt = state.inv_dt_mag == 0 ? one(eltype(state.velocity.tor.spec_r)) :
         inv(Float64(state.inv_dt_mag))

    for stage_index in 1:3
        gamma = CB3_GAMMA[stage_index]
        zeta = CB3_ZETA[stage_index]
        alpha = CB3_ALPHA[stage_index]
        stage = state.cb3[stage_index]

        has_mag = state.magnetic !== nothing
        has_comp = state.composition !== nothing
        work = get(state, :work, nothing)
        # Pooled nl buffers, distinct per stage (the stage's solves consume them
        # before the next stage; a unique tag keeps the pool keys non-aliasing).
        nl = _gpu_erk2_nl_arrays(state.velocity.tor.spec_r, has_mag, has_comp,
            work, Symbol(:cb3_nl, stage_index))
        _gpu_erk2_nonlinear_pass!(state, nl)

        _gpu_cb3_solve_field!(
            state.velocity.tor,
            stage.velocity_tor_lin,
            stage.velocity_tor_lu,
            nl.vt_r,
            nl.vt_i,
            state.nlops_vel.E / dt,
            gamma,
            zeta,
            alpha,
            bw,
        )
        _gpu_cb3_poloidal_stage!(
            state.velocity.pol,
            nl.vp_r,
            nl.vp_i,
            stage,
            state.nlops_vel.E / dt,
            gamma,
            zeta,
            alpha,
            bw,
        )
        if has_mag
            _gpu_cb3_solve_field!(
                state.magnetic.tor,
                stage.magnetic_tor_lin,
                stage.magnetic_tor_lu,
                nl.mt_r,
                nl.mt_i,
                1.0 / dt,
                gamma,
                zeta,
                alpha,
                bw;
                apply_bc = false,
            )
            _gpu_cb3_solve_field!(
                state.magnetic.pol,
                stage.magnetic_pol_lin,
                stage.magnetic_pol_lu,
                nl.mp_r,
                nl.mp_i,
                1.0 / dt,
                gamma,
                zeta,
                alpha,
                bw;
                apply_bc = false,
            )
        end
        _gpu_cb3_solve_field!(
            state.temperature,
            stage.temperature_lin,
            stage.temperature_lu,
            nl.t_r,
            nl.t_i,
            state.inv_dt_temp,
            gamma,
            zeta,
            alpha,
            bw,
        )
        if has_comp
            _gpu_cb3_solve_field!(
                state.composition,
                stage.composition_lin,
                stage.composition_lu,
                nl.c_r,
                nl.c_i,
                state.inv_dt_comp,
                gamma,
                zeta,
                alpha,
                bw,
            )
        end
        _gpu_cb3_roll_stage_nl!(state, nl)
    end
    return nothing
end
