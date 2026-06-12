# =============================================================================
# GPU ERK2 — staged exponential RK2 step on the dense device state, mirroring
# the CPU integrate_solver_erk2_step! exactly:
#
#   prepare:   linear = E_full·u₀;  k1 = φ1_full·n₀
#              stage  = E_half·u₀ + (dt/2)·φ1_half·n₀   (+ BC enforcement)
#   stage:     u ← stage;  recompute every nonlinear at the staged fields
#   finalize:  u⁺ = linear + dt·k1 + 2dt·φ2_full·(n_stage − n₀)  (+ BC)
#
# Velocity poloidal runs in the Stage-4B W-split: the stage machinery advances
# V := Ek·D_pol·P (cache built on D_pol, diffusivity 1, N_W unscaled); P is
# recovered from V with Dirichlet walls + φ1-column influence corrections at
# the stage (dt/2 Greens) and the finalize (dt Greens), all packed as
# constants by build_gpu_erk2_state.
#
# The two nonlinear passes replicate the CPU compute_solver_nonlinear_terms!
# semantics: fresh T/C synthesis from the CURRENT spectral state before the
# velocity force assembly; B/J buffers refreshed from the current magnetic
# spectral state by each pass's magnetic section (so velocity's Lorentz input
# is lagged by exactly one pass, as on the CPU).
# =============================================================================

# Per-(l,m) dense matvec: Y[li,mi,:] = M3[:,:,li] · X[li,mi,:].
@kernel function _dense_matvec_perl_kernel!(Y, @Const(X), @Const(M3), nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(Y)
    @inbounds for i in 1:nr
        s = zero(T)
        for j in 1:nr
            s += M3[i, j, li] * X[li, mi, j]
        end
        Y[li, mi, i] = s
    end
end

"""
    gpu_batched_dense_matvec_perl!(Y, X, M3) -> Y

Apply the per-degree DENSE radial operator stack `M3 :: (nr, nr, nl)` to every
mode's radial profile: `Y[l,m,:] = M3[:,:,l+1] · X[l,m,:]`.  `Y` must not
alias `X`.
"""
function gpu_batched_dense_matvec_perl!(Y, X, M3)
    nl, nm, nr = size(Y)
    backend = KernelAbstractions.get_backend(Y)
    _dense_matvec_perl_kernel!(backend)(Y, X, M3, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)
    return Y
end

# ERK2 endpoint enforcement for one side (mirrors solver_enforce_erk2_bc!).
# kind 0: endpoint ← 0. kind 1 (Dirichlet): endpoint ← val. kind 2 (stencil):
# solve the boundary stencil row for the endpoint, with optional l-correction
# and the l0-Dirichlet special case. Empty (l < m) modes are skipped.
@kernel function _erk2_bc_kernel!(X, b::Int, kind::Int32, @Const(stencil),
        r_inv::Float64, l_sign::Float64, use_l_corr::Bool, fixed_corr::Float64,
        l0_dir::Bool, @Const(val), ptol::Float64, nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(X)
    l = li - 1
    m = mi - 1
    @inbounds if l >= m
        v = val[li, mi]
        if kind == Int32(0)
            X[li, mi, b] = zero(T)
        elseif kind == Int32(1) || (l0_dir && l == 0)
            X[li, mi, b] = v
        else
            self = stencil[b] + fixed_corr
            if use_l_corr
                self += l_sign * T(l) * r_inv
            end
            od = zero(T)
            for j in 1:nr
                j == b || (od += stencil[j] * X[li, mi, j])
            end
            X[li, mi, b] = abs(self) > ptol ? (v - od) / self : v
        end
    end
end

# Apply both endpoints; `imag` selects the imaginary-part value tables.
function gpu_erk2_enforce_bcs!(X, bc, nr::Int; imag::Bool = false)
    nl, nm, _ = size(X)
    backend = KernelAbstractions.get_backend(X)
    for (side, b) in ((bc.inner, 1), (bc.outer, nr))
        _erk2_bc_kernel!(backend)(X, b, side.kind, side.stencil,
            side.r_inv, side.l_sign, side.use_l_correction, side.fixed_correction,
            side.l0_dirichlet, imag ? side.val_i : side.val_r,
            pivot_tol(eltype(X)), nr; ndrange = (nl, nm))
    end
    KernelAbstractions.synchronize(backend)
    return X
end

# Low-storage prepare + stage for one split-complex component. Returns
# (acc, stage) where acc = E_f·u₀ + dt·M1·n₀ already carries the FULL n₀
# contribution to the finalize (M1 = φ1_f − 2φ2, precomputed in the pack), so
# neither n₀ nor separate linear/k1 buffers need to survive the stage pass.
# `acc` persists until the finalize → pooled under the per-field tag; sphi is
# transient.
function _gpu_erk2_prepare(u, n, fld, dt, bc, nr, ws, tag::Symbol; imag::Bool)
    T = eltype(u)
    acc = gpu_scratch!(ws, Symbol(tag, :_acc), u)
    gpu_batched_dense_matvec_perl!(acc, u, fld.Ef)
    sphi = gpu_scratch!(ws, Symbol(tag, :_sp), u)
    gpu_batched_dense_matvec_perl!(sphi, n, fld.M1)
    @. acc = acc + T(dt) * sphi
    stage = gpu_scratch!(ws, Symbol(tag, :_st), u)
    gpu_batched_dense_matvec_perl!(stage, u, fld.Eh)
    gpu_batched_dense_matvec_perl!(sphi, n, fld.p1h)
    @. stage = stage + T(dt / 2) * sphi
    gpu_erk2_enforce_bcs!(stage, bc, nr; imag)
    return acc, stage
end

# finalize one component into `dst`:  u⁺ = acc + 2dt·φ2·n_stage.
function _gpu_erk2_finalize!(dst, acc, nstage, fld, dt, bc, nr, ws, tag::Symbol; imag::Bool)
    T = eltype(dst)
    corr = gpu_scratch!(ws, Symbol(tag, :_co), dst)
    gpu_batched_dense_matvec_perl!(corr, nstage, fld.p2f)
    @. dst = acc + T(2 * dt) * corr
    gpu_erk2_enforce_bcs!(dst, bc, nr; imag)
    return dst
end

# P ← recover(V): V/Ek with zeroed walls through the Dirichlet P-solve, then
# the φ1-column influence correction with the packed per-degree Greens
# (`half` picks the dt/2 vs dt set). Mirrors _erk2_poloidal_recover!.
function _gpu_erk2_recover_P!(P, V, wsplit, rec, Ek, bw::Int, ws, tag::Symbol; half::Bool)
    T = eltype(P)
    nr = size(P, 3)
    Wv = gpu_scratch!(ws, Symbol(tag, :_Wv), P)
    @. Wv = V / T(Ek)
    Wv[:, :, 1] .= zero(T); Wv[:, :, nr] .= zero(T)
    Pt = gpu_scratch!(ws, Symbol(tag, :_Pt), P)
    gpu_batched_banded_solve!(Pt, Wv, wsplit.plu, bw)
    nl_, nm_ = size(P, 1), size(P, 2)
    rho1 = gpu_scratch!(ws, Symbol(tag, :_r1), P, (nl_, nm_))
    rho2 = gpu_scratch!(ws, Symbol(tag, :_r2), P, (nl_, nm_))
    dprod = gpu_scratch!(ws, Symbol(tag, :_dp), P)
    dprod .= Pt .* reshape(wsplit.d1_inner, 1, 1, :)
    sum!(reshape(rho1, nl_, nm_, 1), dprod)
    dprod .= Pt .* reshape(wsplit.d1_outer, 1, 1, :)
    sum!(reshape(rho2, nl_, nm_, 1), dprod)
    M = half ? rec.Mh : rec.Mf
    h1 = half ? rec.h1h : rec.h1f
    h2 = half ? rec.h2h : rec.h2f
    M11 = reshape(@view(M[1, 1, :]), :, 1); M12 = reshape(@view(M[1, 2, :]), :, 1)
    M21 = reshape(@view(M[2, 1, :]), :, 1); M22 = reshape(@view(M[2, 2, :]), :, 1)
    a1 = gpu_scratch!(ws, Symbol(tag, :_a1), P, (nl_, nm_))
    a2 = gpu_scratch!(ws, Symbol(tag, :_a2), P, (nl_, nm_))
    @. a1 = (-rho1 * M22 + rho2 * M12) / (M11 * M22 - M12 * M21 + T((M11 * M22 - M12 * M21) == 0))
    @. a2 = (-rho2 * M11 + rho1 * M21) / (M11 * M22 - M12 * M21 + T((M11 * M22 - M12 * M21) == 0))
    h1r = reshape(h1, size(h1, 1), 1, nr)
    h2r = reshape(h2, size(h2, 1), 1, nr)
    @. P = Pt + a1 * h1r + a2 * h2r   # Pt local var (pooled buffer) — safe
    P[1, :, :] .= zero(T)
    return P
end

# One full nonlinear pass on the CURRENT spectral state of `state`, mirroring
# compute_solver_nonlinear_terms!: fresh T/C → velocity nl (with the CURRENT
# lagged B/J buffers) → refresh B/J buffers from the current magnetic spec →
# magnetic nl → scalar nls. Writes the per-field nl arrays into `out`
# (NamedTuple of preallocated arrays) and updates state.B_*/J_*/T_phys/C_phys.
function _gpu_erk2_nonlinear_pass!(state, out)
    cfg = state.config; lmax = state.lmax; bw = state.bw
    v = state.velocity
    arch = arch_of(v.tor.spec_r)
    nr = size(v.tor.spec_r, 3)
    spec(a, b) = GPUSpectralField{eltype(a), typeof(a)}(cfg, size(a, 1), size(a, 2), size(a, 3), a, b)
    ws = get(state, :work, nothing)
    Tel = eltype(v.tor.spec_r)
    ph(k::Symbol) = gpu_scratch_phys!(ws, k, Tel, arch, cfg, nr)

    # fresh scalars (CPU refreshes them before the velocity force assembly)
    Tn = ph(:e_T); gpu_scalar_spectral_to_physical!(Tn, spec(state.temperature.spec_r, state.temperature.spec_i), cfg; ws, tag = :e_Ts)
    Cn = state.composition === nothing ? nothing :
        (c = ph(:e_C); gpu_scalar_spectral_to_physical!(c, spec(state.composition.spec_r, state.composition.spec_i), cfg; ws, tag = :e_Cs); c)

    # velocity nonlinear with the LAGGED B/J buffers
    gpu_velocity_nonlinear!(out.vt_r, out.vt_i, out.vp_r, out.vp_i,
        v.tor.spec_r, v.tor.spec_i, v.pol.spec_r, v.pol.spec_i, cfg,
        state.nlops_vel.d1, state.nlops_vel.d2, state.nlops_vel.lfac,
        state.nlops_vel.rinv, state.nlops_vel.rinv2, state.nlops_vel.rscale,
        state.nlops_vel.sinθ, state.nlops_vel.cosθ, state.nlops_vel.E, lmax, bw;
        ws = ws, tag = :e_vnl,
        T_phys = Tn.data, thermal_factor = state.thermal_factor, r_vec = state.r_vec,
        C_phys = Cn === nothing ? nothing : Cn.data,
        comp_factor = state.composition === nothing ? zero(eltype(v.tor.spec_r)) : state.comp_factor,
        J_r = state.J_r, J_θ = state.J_θ, J_φ = state.J_φ,
        B_r = state.B_r, B_θ = state.B_θ, B_φ = state.B_φ,
        lorentz_coeff = state.lorentz_coeff)

    # shared u for induction + scalar advection (current velocity spec)
    u = ph(:e_ur); uθ = ph(:e_ut); uφ = ph(:e_up)
    gpu_vector_spectral_to_physical!(u, uθ, uφ, spec(v.tor.spec_r, v.tor.spec_i),
        spec(v.pol.spec_r, v.pol.spec_i), cfg,
        state.nlops_vel.lfac, state.nlops_vel.rscale, state.nlops_vel.d1, state.nlops_vel.rinv, bw;
        ws, tag = :e_us)

    if state.magnetic !== nothing
        m = state.magnetic
        # refresh B/J buffers from the current magnetic spec (CPU magnetic pass)
        br = ph(:e_Br); bθ = ph(:e_Bt); bφ = ph(:e_Bp)
        gpu_vector_spectral_to_physical!(br, bθ, bφ, spec(m.tor.spec_r, m.tor.spec_i),
            spec(m.pol.spec_r, m.pol.spec_i), cfg,
            state.nlops_mag.lfac, state.nlops_mag.rscale, state.nlops_mag.d1, state.nlops_mag.rinv, bw;
            ws, tag = :e_Bs)
        jtr = gpu_scratch!(ws, :e_jtr, m.tor.spec_r); jti = gpu_scratch!(ws, :e_jti, m.tor.spec_i)
        jpr = gpu_scratch!(ws, :e_jpr, m.pol.spec_r); jpi = gpu_scratch!(ws, :e_jpi, m.pol.spec_i)
        gpu_spectral_curl!(jtr, jti, jpr, jpi, m.tor.spec_r, m.tor.spec_i,
            m.pol.spec_r, m.pol.spec_i,
            state.nlops_mag.d1, state.nlops_mag.d2, state.nlops_mag.lfac,
            state.nlops_mag.rinv, state.nlops_mag.rinv2, state.nlops_mag.r, bw; ws, tag = :e_jc)
        jr = ph(:e_Jr); jθ = ph(:e_Jt); jφ = ph(:e_Jp)
        gpu_vector_spectral_to_physical!(jr, jθ, jφ, spec(jtr, jti), spec(jpr, jpi), cfg,
            state.nlops_mag.lfac, state.nlops_mag.rscale, state.nlops_mag.d1, state.nlops_mag.rinv, bw;
            ws, tag = :e_Js)
        state.B_r .= br.data; state.B_θ .= bθ.data; state.B_φ .= bφ.data
        state.J_r .= jr.data; state.J_θ .= jθ.data; state.J_φ .= jφ.data

        gpu_magnetic_nonlinear!(out.mt_r, out.mt_i, out.mp_r, out.mp_i,
            m.tor.spec_r, m.tor.spec_i, m.pol.spec_r, m.pol.spec_i,
            u.data, uθ.data, uφ.data, cfg,
            state.nlops_mag.d1, state.nlops_mag.d2, state.nlops_mag.lfac,
            state.nlops_mag.rinv, state.nlops_mag.rinv2, state.nlops_mag.rscale, lmax, bw;
            ws, tag = :e_mnl)
    end

    gpu_scalar_nonlinear!(out.t_r, out.t_i, state.temperature.spec_r, state.temperature.spec_i,
        u.data, uθ.data, uφ.data, cfg, state.d1, state.mvals, state.rinv, lmax, bw;
        ws, tag = :e_tnl)
    if state.composition !== nothing
        gpu_scalar_nonlinear!(out.c_r, out.c_i, state.composition.spec_r, state.composition.spec_i,
            u.data, uθ.data, uφ.data, cfg, state.d1, state.mvals, state.rinv, lmax, bw;
            ws, tag = :e_cnl)
    end

    state.T_phys .= Tn.data
    Cn !== nothing && (state.C_phys .= Cn.data)
    return out
end

_gpu_erk2_nl_arrays(template, has_mag::Bool, has_comp::Bool, ws, tag::Symbol) = (;
    vt_r = gpu_scratch!(ws, Symbol(tag, :_vtr), template),
    vt_i = gpu_scratch!(ws, Symbol(tag, :_vti), template),
    vp_r = gpu_scratch!(ws, Symbol(tag, :_vpr), template),
    vp_i = gpu_scratch!(ws, Symbol(tag, :_vpi), template),
    mt_r = has_mag ? gpu_scratch!(ws, Symbol(tag, :_mtr), template) : nothing,
    mt_i = has_mag ? gpu_scratch!(ws, Symbol(tag, :_mti), template) : nothing,
    mp_r = has_mag ? gpu_scratch!(ws, Symbol(tag, :_mpr), template) : nothing,
    mp_i = has_mag ? gpu_scratch!(ws, Symbol(tag, :_mpi), template) : nothing,
    t_r = gpu_scratch!(ws, Symbol(tag, :_tr), template),
    t_i = gpu_scratch!(ws, Symbol(tag, :_ti), template),
    c_r = has_comp ? gpu_scratch!(ws, Symbol(tag, :_cr), template) : nothing,
    c_i = has_comp ? gpu_scratch!(ws, Symbol(tag, :_ci), template) : nothing)

"""
    gpu_erk2_solver_step!(state, erk) -> nothing

Advance every field one staged exponential RK2 step on the dense device state
`state` (from [`build_gpu_solver_state`](@ref)) with the ERK2 operator pack
`erk` (from [`build_gpu_erk2_state`](@ref)). Mirrors the CPU
`compute_solver_nonlinear_terms!` + `integrate_solver_erk2_step!` sequence,
including the velocity-poloidal W-split V-advance with φ1-column P-recovery
at the stage and the finalize. The lagged physical buffers evolve exactly as
on the CPU (each nonlinear pass refreshes them).
"""
function gpu_erk2_solver_step!(state, erk)
    cfg = state.config; bw = state.bw
    v = state.velocity
    T = eltype(v.tor.spec_r)
    nr = size(v.tor.spec_r, 3)
    dt = erk.dt
    Ek = state.nlops_vel.E
    has_mag = state.magnetic !== nothing && erk.magnetic_tor !== nothing
    has_comp = state.composition !== nothing && erk.composition !== nothing
    wsplit = state.wsplit
    work = get(state, :work, nothing)

    # --- entry nonlinear pass at u₀ (n₀ for every field) ---
    n0 = _gpu_erk2_nl_arrays(v.tor.spec_r, has_mag, has_comp, work, :e_n0)
    _gpu_erk2_nonlinear_pass!(state, n0)

    # --- V₀ = Ek·D_pol·P₀ (the stage machinery advances V, not P) ---
    V_r = gpu_scratch!(work, :e_Vr, v.pol.spec_r)
    V_i = gpu_scratch!(work, :e_Vi, v.pol.spec_i)
    gpu_batched_banded_matvec_perl!(V_r, v.pol.spec_r, wsplit.dpol, bw)
    gpu_batched_banded_matvec_perl!(V_i, v.pol.spec_i, wsplit.dpol, bw)
    @. V_r = T(Ek) * V_r
    @. V_i = T(Ek) * V_i
    V_r[1, :, :] .= zero(T); V_i[1, :, :] .= zero(T)

    # --- prepare every field (linear/k1/stage from u₀, n₀) ---
    fields = Any[
        (erk.temperature, state.temperature.spec_r, state.temperature.spec_i, n0.t_r, n0.t_i),
        (erk.velocity_tor, v.tor.spec_r, v.tor.spec_i, n0.vt_r, n0.vt_i),
        (erk.velocity_pol, V_r, V_i, n0.vp_r, n0.vp_i),
    ]
    has_mag && push!(fields,
        (erk.magnetic_tor, state.magnetic.tor.spec_r, state.magnetic.tor.spec_i, n0.mt_r, n0.mt_i),
        (erk.magnetic_pol, state.magnetic.pol.spec_r, state.magnetic.pol.spec_i, n0.mp_r, n0.mp_i))
    has_comp && push!(fields,
        (erk.composition, state.composition.spec_r, state.composition.spec_i, n0.c_r, n0.c_i))

    prepared = Any[]
    for (fi, (fld, ur, ui, nr_, ni_)) in enumerate(fields)
        ftag = Symbol(:e_f, fi)
        acc_r, st_r = _gpu_erk2_prepare(ur, nr_, fld, dt, fld.bc, nr, work,
            Symbol(ftag, :r); imag = false)
        acc_i, st_i = _gpu_erk2_prepare(ui, ni_, fld, dt, fld.bc, nr, work,
            Symbol(ftag, :i); imag = true)
        push!(prepared, (; fld, ur, ui, acc_r, st_r, acc_i, st_i))
    end

    # --- stage: write provisional fields; recover the stage P from stage V ---
    for p in prepared
        p.ur .= p.st_r
        p.ui .= p.st_i
    end
    _gpu_erk2_recover_P!(v.pol.spec_r, V_r, wsplit, erk.recovery, Ek, bw, work, :e_rcr; half = true)
    _gpu_erk2_recover_P!(v.pol.spec_i, V_i, wsplit, erk.recovery, Ek, bw, work, :e_rci; half = true)

    # --- stage nonlinear pass at the provisional fields ---
    nstage = _gpu_erk2_nl_arrays(v.tor.spec_r, has_mag, has_comp, work, :e_ns)
    _gpu_erk2_nonlinear_pass!(state, nstage)
    stage_nls = Any[(nstage.t_r, nstage.t_i), (nstage.vt_r, nstage.vt_i), (nstage.vp_r, nstage.vp_i)]
    has_mag && push!(stage_nls, (nstage.mt_r, nstage.mt_i), (nstage.mp_r, nstage.mp_i))
    has_comp && push!(stage_nls, (nstage.c_r, nstage.c_i))

    # --- finalize every field; recover the final P from V⁺ ---
    for (fi, (p, (ns_r, ns_i))) in enumerate(zip(prepared, stage_nls))
        ftag = Symbol(:e_f, fi)
        _gpu_erk2_finalize!(p.ur, p.acc_r, ns_r, p.fld, dt, p.fld.bc, nr,
            work, Symbol(ftag, :r); imag = false)
        _gpu_erk2_finalize!(p.ui, p.acc_i, ns_i, p.fld, dt, p.fld.bc, nr,
            work, Symbol(ftag, :i); imag = true)
    end
    _gpu_erk2_recover_P!(v.pol.spec_r, V_r, wsplit, erk.recovery, Ek, bw, work, :e_rcr; half = false)
    _gpu_erk2_recover_P!(v.pol.spec_i, V_i, wsplit, erk.recovery, Ek, bw, work, :e_rci; half = false)
    return nothing
end
