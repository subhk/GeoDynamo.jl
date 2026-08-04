# =============================================================================
# GPU Phase 5n — full multi-field CNAB2 timestep orchestrator, composing the
# per-field steps in the CPU order velocity → magnetic → temperature →
# composition (compute_solver_nonlinear_terms! nonlinear.jl:1010 + the implicit
# updates + roll_solver_histories!).  No new kernels.  Velocity runs first and
# synthesizes the shared physical velocity u, reused by every other field.
#
# BUFFER FRESHNESS (matches CPU exactly): compute_solver_nonlinear_terms!
# refreshes BOTH the scalar physical fields (buoyancy) AND the magnetic
# field/current (Lorentz) from the CURRENT spectral state BEFORE the velocity
# force assembly, so buoyancy reads FRESH T/C and the Lorentz force reads FRESH
# B/J (`Bn_*`/`Jn_*`) synthesized at the start of this step — NOT one-step
# lagged.  `state.B_*`/`J_*` are still rolled at the end for any external reader.
# u itself is NOT lagged (fresh synth of current velocity).
#
# `state` (NamedTuple) holds the per-field bundles, shared operators, coupling
# factors, and the persistent physical buffers.  magnetic/composition optional
# (nothing → skipped).  Per-field rollover (nl→prev_nl) happens inside each step.
# Runs on Array + CuArray.  (Scratch pooled via GPUWorkspace (`state.work`).  The velocity
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
    ws = get(state, :work, nothing)
    T = eltype(v.tor.spec_r)
    ph(k::Symbol) = gpu_scratch_phys!(ws, k, T, arch, cfg, nr)

    # --- (1) shared physical velocity u from the OLD velocity spectral (fresh) ---
    u = ph(:st_ur); uθ = ph(:st_ut); uφ = ph(:st_up)
    gpu_vector_spectral_to_physical!(u, uθ, uφ, spec(v.tor.spec_r, v.tor.spec_i),
        spec(v.pol.spec_r, v.pol.spec_i), cfg, state.nlops_vel.lfac, state.nlops_vel.rscale,
        state.nlops_vel.d1, state.nlops_vel.rinv, bw)

    # --- (2) physical buffers from the OLD (start-of-step) spectral state:
    #         T/C feed THIS step's buoyancy (fresh, CPU semantics);
    #         B/J feed the NEXT step's Lorentz (one-step lag, CPU semantics) ---
    Tn = ph(:st_T); gpu_scalar_spectral_to_physical!(Tn, spec(state.temperature.spec_r, state.temperature.spec_i), cfg; ws, tag = :st_Ts)
    Cn = state.composition === nothing ? nothing :
        (c = ph(:st_C); gpu_scalar_spectral_to_physical!(c, spec(state.composition.spec_r, state.composition.spec_i), cfg; ws, tag = :st_Cs); c)
    Bn_r = Bn_θ = Bn_φ = Jn_r = Jn_θ = Jn_φ = nothing
    if state.magnetic !== nothing
        m = state.magnetic
        br = ph(:st_Br); bθ = ph(:st_Bt); bφ = ph(:st_Bp)
        gpu_vector_spectral_to_physical!(br, bθ, bφ, spec(m.tor.spec_r, m.tor.spec_i),
            spec(m.pol.spec_r, m.pol.spec_i), cfg, state.nlops_mag.lfac, state.nlops_mag.rscale,
            state.nlops_mag.d1, state.nlops_mag.rinv, bw)
        jtr = gpu_scratch!(ws, :st_jtr, m.tor.spec_r); jti = gpu_scratch!(ws, :st_jti, m.tor.spec_i)
        jpr = gpu_scratch!(ws, :st_jpr, m.pol.spec_r); jpi = gpu_scratch!(ws, :st_jpi, m.pol.spec_i)
        gpu_spectral_curl!(jtr, jti, jpr, jpi, m.tor.spec_r, m.tor.spec_i, m.pol.spec_r, m.pol.spec_i,
            state.nlops_mag.d1, state.nlops_mag.d2, state.nlops_mag.lfac, state.nlops_mag.rinv, state.nlops_mag.rinv2,
            state.nlops_mag.r, bw)
        jr = ph(:st_Jr); jθ = ph(:st_Jt); jφ = ph(:st_Jp)
        gpu_vector_spectral_to_physical!(jr, jθ, jφ, spec(jtr, jti), spec(jpr, jpi), cfg,
            state.nlops_mag.lfac, state.nlops_mag.rscale, state.nlops_mag.d1, state.nlops_mag.rinv, bw)
        Bn_r = br.data; Bn_θ = bθ.data; Bn_φ = bφ.data; Jn_r = jr.data; Jn_θ = jθ.data; Jn_φ = jφ.data
    end

    # --- (3) velocity step: FRESH T/C and FRESH B/J, both synthesized above from
    #         the current spectral state (matches the CPU fix that refreshes the
    #         magnetic field before velocity, removing the Lorentz one-step lag) ---
    gpu_velocity_field_step!(v.tor, v.pol, cfg, state.nlops_vel, state.influence,
        state.inv_dt_vel, linw, lmax, bw;
        wsplit = get(state, :wsplit, nothing), ws = ws,
        T_phys = Tn.data, thermal_factor = state.thermal_factor, r_vec = state.r_vec,
        C_phys = Cn === nothing ? nothing : Cn.data,
        comp_factor = state.composition === nothing ? zero(eltype(v.tor.spec_r)) : state.comp_factor,
        J_r = Jn_r, J_θ = Jn_θ, J_φ = Jn_φ,
        B_r = Bn_r, B_θ = Bn_θ, B_φ = Bn_φ, lorentz_coeff = state.lorentz_coeff)

    # --- (4) magnetic step (if present) with the shared u ---
    if state.magnetic !== nothing
        m = state.magnetic
        # `ic` is nothing for an insulating inner core (the default) and the packed
        # admittance + inner-core spectra when conducting, in which case
        # gpu_magnetic_field_step! runs the φ0 history-flux inner boundary and
        # advances the inner-core field in place.
        gpu_magnetic_field_step!(m.tor, m.pol, u.data, uθ.data, uφ.data, cfg, state.nlops_mag,
            state.inv_dt_mag, linw, lmax, bw; ic = state.ic, ws = ws)
    end

    # --- (5) temperature step with the shared u ---
    t = state.temperature
    gpu_scalar_field_step!(t.spec_r, t.spec_i, t.prev_nl_r, t.prev_nl_i, u.data, uθ.data, uφ.data, cfg,
        state.d1, state.mvals, state.rinv, t.lin, t.lu, t.bc_in_r, t.bc_in_i, t.bc_out_r, t.bc_out_i,
        state.inv_dt_temp, linw, lmax, bw; ws = ws, tag = :st_temp,
        internal_source = t.internal_source)

    # --- (6) composition step (if present) with the shared u ---
    if state.composition !== nothing
        c = state.composition
        gpu_scalar_field_step!(c.spec_r, c.spec_i, c.prev_nl_r, c.prev_nl_i, u.data, uθ.data, uφ.data, cfg,
            state.d1, state.mvals, state.rinv, c.lin, c.lu, c.bc_in_r, c.bc_in_i, c.bc_out_r, c.bc_out_i,
            state.inv_dt_comp, linw, lmax, bw; ws = ws, tag = :st_comp,
            internal_source = c.internal_source)
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
