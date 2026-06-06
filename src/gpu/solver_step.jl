# =============================================================================
# GPU Phase 5n — full multi-field CNAB2 timestep orchestrator, composing the
# per-field steps in the CPU order velocity → magnetic → temperature →
# composition (compute_solver_nonlinear_terms! nonlinear.jl:1010 + the implicit
# updates + roll_solver_histories!).  No new kernels.  Velocity runs first and
# synthesizes the shared physical velocity u, reused by every other field.
#
# THE LAG (matches CPU exactly): velocity's buoyancy + Lorentz read the physical
# T/C/B/J synthesized during the PREVIOUS step (CPU never refreshes them before
# velocity).  So this step (a) runs velocity with the persistent LAGGED buffers
# `state.T_phys/C_phys/B_*/J_*`, and (b) re-synthesizes those buffers from the OLD
# spectral state (before the implicit overwrites it) for the NEXT step's velocity,
# rolling them at the end.  u itself is NOT lagged (fresh synth of current velocity).
#
# `state` (NamedTuple) holds the per-field bundles, shared operators, coupling
# factors, and the persistent physical buffers.  magnetic/composition optional
# (nothing → skipped).  Per-field rollover (nl→prev_nl) happens inside each step.
# Runs on Array + CuArray.  (Per-call scratch — Phase-6 may cache.  The velocity
# step recomputes u internally; redundant with the shared u but identical since
# both synth the same OLD velocity.)  Device-state builder + GPU≈CPU gate = 5n2.
# =============================================================================

"""
    gpu_solver_step!(state) -> nothing

Advance every field one CNAB2 step on the GPU, in the CPU order
velocity → magnetic → temperature → composition, with the shared physical velocity
and the one-step physical-field lag for velocity's buoyancy/Lorentz coupling.  See
the file header for the `state` bundle layout and the lag semantics.  `state.magnetic`
/ `state.composition` may be `nothing` to skip those fields.  All arrays on the same backend.
"""
function gpu_solver_step!(state)
    cfg = state.config; lmax = state.lmax; bw = state.bw; linw = state.linear_weight
    v = state.velocity
    arch = arch_of(v.tor.spec_r)
    nr = size(v.tor.spec_r, 3)
    spec(a, b) = GPUSpectralField{eltype(a), typeof(a)}(cfg, size(a, 1), size(a, 2), size(a, 3), a, b)
    ph() = allocate_gpu_physical_field(eltype(v.tor.spec_r), arch, cfg, nr)

    # --- (1) shared physical velocity u from the OLD velocity spectral (fresh) ---
    u = ph(); uθ = ph(); uφ = ph()
    gpu_vector_spectral_to_physical!(u, uθ, uφ, spec(v.tor.spec_r, v.tor.spec_i),
        spec(v.pol.spec_r, v.pol.spec_i), cfg, state.nlops_vel.lfac, state.nlops_vel.rscale)

    # --- (2) current-step physical buffers from OLD spectral (for the NEXT step's velocity lag) ---
    Tn = ph(); gpu_scalar_spectral_to_physical!(Tn, spec(state.temperature.spec_r, state.temperature.spec_i), cfg)
    Cn = state.composition === nothing ? nothing :
        (c = ph(); gpu_scalar_spectral_to_physical!(c, spec(state.composition.spec_r, state.composition.spec_i), cfg); c)
    Bn_r = Bn_θ = Bn_φ = Jn_r = Jn_θ = Jn_φ = nothing
    if state.magnetic !== nothing
        m = state.magnetic
        br = ph(); bθ = ph(); bφ = ph()
        gpu_vector_spectral_to_physical!(br, bθ, bφ, spec(m.tor.spec_r, m.tor.spec_i),
            spec(m.pol.spec_r, m.pol.spec_i), cfg, state.nlops_vel.lfac, state.nlops_vel.rscale)
        jtr = similar(m.tor.spec_r); jti = similar(m.tor.spec_i); jpr = similar(m.pol.spec_r); jpi = similar(m.pol.spec_i)
        gpu_spectral_curl!(jtr, jti, jpr, jpi, m.tor.spec_r, m.tor.spec_i, m.pol.spec_r, m.pol.spec_i,
            state.nlops_mag.d1, state.nlops_mag.d2, state.nlops_mag.lfac, state.nlops_mag.rinv, state.nlops_mag.rinv2, bw)
        jr = ph(); jθ = ph(); jφ = ph()
        gpu_vector_spectral_to_physical!(jr, jθ, jφ, spec(jtr, jti), spec(jpr, jpi), cfg,
            state.nlops_vel.lfac, state.nlops_vel.rscale)
        Bn_r = br.data; Bn_θ = bθ.data; Bn_φ = bφ.data; Jn_r = jr.data; Jn_θ = jθ.data; Jn_φ = jφ.data
    end

    # --- (3) velocity step with the LAGGED physical buffers (previous step's synthesis) ---
    gpu_velocity_field_step!(v.tor, v.pol, cfg, state.nlops_vel, state.influence,
        state.inv_dt_vel, linw, lmax, bw;
        T_phys = state.T_phys, thermal_factor = state.thermal_factor, r_vec = state.r_vec,
        C_phys = state.C_phys, comp_factor = state.comp_factor,
        J_r = state.J_r, J_θ = state.J_θ, J_φ = state.J_φ,
        B_r = state.B_r, B_θ = state.B_θ, B_φ = state.B_φ, lorentz_coeff = state.lorentz_coeff)

    # --- (4) magnetic step (if present) with the shared u ---
    if state.magnetic !== nothing
        m = state.magnetic
        gpu_magnetic_field_step!(m.tor, m.pol, u.data, uθ.data, uφ.data, cfg, state.nlops_mag,
            state.inv_dt_mag, linw, lmax, bw)
    end

    # --- (5) temperature step with the shared u ---
    t = state.temperature
    gpu_scalar_field_step!(t.spec_r, t.spec_i, t.prev_nl_r, t.prev_nl_i, u.data, uθ.data, uφ.data, cfg,
        state.d1, state.mvals, state.rinv, t.lin, t.lu, t.bc_in_r, t.bc_in_i, t.bc_out_r, t.bc_out_i,
        state.inv_dt_temp, linw, lmax, bw)

    # --- (6) composition step (if present) with the shared u ---
    if state.composition !== nothing
        c = state.composition
        gpu_scalar_field_step!(c.spec_r, c.spec_i, c.prev_nl_r, c.prev_nl_i, u.data, uθ.data, uφ.data, cfg,
            state.d1, state.mvals, state.rinv, c.lin, c.lu, c.bc_in_r, c.bc_in_i, c.bc_out_r, c.bc_out_i,
            state.inv_dt_comp, linw, lmax, bw)
    end

    # --- (7) roll the persistent physical buffers (current synthesis → lagged) for the next step ---
    state.T_phys .= Tn.data
    Cn !== nothing && (state.C_phys .= Cn.data)
    if state.magnetic !== nothing
        state.B_r .= Bn_r; state.B_θ .= Bn_θ; state.B_φ .= Bn_φ
        state.J_r .= Jn_r; state.J_θ .= Jn_θ; state.J_φ .= Jn_φ
    end
    return nothing
end
