# =============================================================================
# GPU Phase 5m — one magnetic field CNAB2 timestep (toroidal + poloidal),
# composing the verified pieces: magnetic nonlinear (5h, induction ∇×(u×B),
# returns BOTH nl_tor and nl_pol) → CNAB2 RHS (5c, mass_coeff=1) → implicit
# solve (5d) for tor then pol → field update + nl_prev rollover.  Mirrors
# apply_magnetic_toroidal/poloidal_implicit_update!.  No new kernels.
#
# INSULATING path (ic=nothing, default): the toroidal inner boundary optionally
# takes the CONTINUITY_MAG increment −nl_pol[ICB] + prev_nl_pol[ICB]; the
# poloidal is homogeneous.
#
# CONDUCTING path (ic given, Phase 5m2): the inner boundary RHS for both
# toroidal and poloidal is the conducting history flux φ0 (5l kernels).  After
# the outer-core solve and field update, the inner-core spectral state is
# reconstructed via gpu_reconstruct_inner_core! (5l) using the new outer-core
# ICB value.  The CONTINUITY_MAG flag is superseded and ignored when ic is given.
#
# Runs on Array + CuArray.  (Per-call scratch — Phase-6 may cache.)
#
# Bundles:  tor/pol :: (; spec_r, spec_i, prev_nl_r, prev_nl_i, lin, lu)
#           nlops   :: (; d1, d2, lfac, rinv, rinv2, rscale)
#           ic      :: (; tor_adm, pol_adm, tor_ic_r, tor_ic_i, pol_ic_r, pol_ic_i)
# =============================================================================

"""
    gpu_magnetic_field_step!(tor, pol, u_r, u_θ, u_φ, config, nlops, inv_dt, linear_weight,
                             lmax, bw; continuity_mag=false, ic=nothing) -> nothing

Advance the magnetic field one CNAB2 step.  `tor`/`pol` are NamedTuple bundles
`(; spec_r, spec_i, prev_nl_r, prev_nl_i, lin, lu)`; on exit `*.spec_*` is the
updated field and `*.prev_nl_*` holds THIS step's nonlinear term.  `u_*` is the
physical velocity (supplied — from the velocity step).  `nlops` carries the
magnetic nonlinear/curl operators.  `inv_dt = 1/dt` and `lin` carry the magnetic
mass coefficient (η is in `lin`); `linear_weight = 1−θ`.

**Insulating inner core (default, `ic = nothing`):**
`continuity_mag=true` applies the `CONTINUITY_MAG` toroidal inner-boundary
coupling: the toroidal inner RHS row is set to `−nl_pol[ICB] + prev_nl_pol[ICB]`
(ICB = radial index 1), computed from the just-formed poloidal nonlinear and the
OLD poloidal history.  Otherwise the toroidal inner row is 0.  The poloidal is
fully homogeneous.

**Conducting inner core (`ic` keyword given):**
`ic` must be a NamedTuple `(; tor_adm, pol_adm, tor_ic_r, tor_ic_i, pol_ic_r,
pol_ic_i)` where `*_adm` are `gpu_pack_inner_core` bundles and `*_ic_*` are
`(nl, nm, Nic)` inner-core spectral state arrays.  When `ic` is supplied:
- The inner BC for BOTH toroidal and poloidal is the conducting history flux `φ0`
  from `gpu_inner_core_history_flux!` (5l), computed from the OLD inner-core
  state BEFORE the outer-core solve.  This is NOT incremental (no prev subtraction,
  unlike the CONTINUITY_MAG path); `continuity_mag` is ignored.
- Outer BC = 0 (same as insulating).
- After the outer-core field update, the inner-core profile is reconstructed via
  `gpu_reconstruct_inner_core!` (5l) using `g = spec[:,:,1]` (the NEW outer-core
  ICB value, materialized — not a view), into fresh scratch, then written back into
  the `ic.*_ic_*` arrays in place.

ORDERING INVARIANT: φ0 reads OLD ic state; both `build_rhs` calls and the
CONTINUITY_MAG BC all read OLD outer-core state (`*.spec_*`, `pol.prev_nl_*`);
the field/history and ic state are overwritten ONLY after every such read.
All arrays on the same backend.
"""
function gpu_magnetic_field_step!(tor, pol, u_r, u_θ, u_φ, config, nlops,
        inv_dt, linear_weight, lmax::Int, bw::Int; continuity_mag::Bool = false, ic = nothing)
    nl, nm, _ = size(tor.spec_r)
    # 1. magnetic nonlinear (5h): nl_tor/nl_pol from the OLD B (tor/pol spec).
    nlt_r = similar(tor.spec_r); nlt_i = similar(tor.spec_i)   # Phase-6: workspace
    nlp_r = similar(pol.spec_r); nlp_i = similar(pol.spec_i)
    gpu_magnetic_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i,
        tor.spec_r, tor.spec_i, pol.spec_r, pol.spec_i, u_r, u_θ, u_φ, config,
        nlops.d1, nlops.d2, nlops.lfac, nlops.rinv, nlops.rinv2, nlops.rscale, lmax, bw)

    # 2. inner-boundary RHS rows for tor/pol. Three cases:
    #    - conducting (ic given): inner = φ0 history flux (5l) from the OLD inner-core state.
    #    - insulating + continuity_mag: tor inner = −nl_pol[ICB]+prev_nl_pol[ICB]; pol inner = 0.
    #    - insulating plain: both 0.  Outer = 0 always.
    z = similar(tor.spec_r, nl, nm); fill!(z, zero(eltype(tor.spec_r)))
    bcin_tor_r = similar(z); bcin_tor_i = similar(z)
    bcin_pol_r = similar(z); bcin_pol_i = similar(z)
    if ic !== nothing
        # conducting: φ0 supersedes the CONTINUITY_MAG increment (continuity_mag ignored).
        gpu_inner_core_history_flux!(bcin_tor_r, bcin_tor_i, ic.tor_ic_r, ic.tor_ic_i, ic.tor_adm)
        gpu_inner_core_history_flux!(bcin_pol_r, bcin_pol_i, ic.pol_ic_r, ic.pol_ic_i, ic.pol_adm)
    elseif continuity_mag
        @views bcin_tor_r .= .-nlp_r[:, :, 1] .+ pol.prev_nl_r[:, :, 1]
        @views bcin_tor_i .= .-nlp_i[:, :, 1] .+ pol.prev_nl_i[:, :, 1]
        fill!(bcin_pol_r, zero(eltype(bcin_pol_r))); fill!(bcin_pol_i, zero(eltype(bcin_pol_i)))
    else
        fill!(bcin_tor_r, zero(eltype(bcin_tor_r))); fill!(bcin_tor_i, zero(eltype(bcin_tor_i)))
        fill!(bcin_pol_r, zero(eltype(bcin_pol_r))); fill!(bcin_pol_i, zero(eltype(bcin_pol_i)))
    end

    # 3. toroidal CNAB2 RHS (5c) from OLD tor spec, then implicit solve (5d).
    rt_r = similar(tor.spec_r); rt_i = similar(tor.spec_i)     # rt ≠ tor.spec — build_rhs reads tor.spec
    gpu_build_rhs_cnab2!(rt_r, rt_i, tor.spec_r, tor.spec_i, nlt_r, nlt_i,
        tor.prev_nl_r, tor.prev_nl_i, tor.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rt_r, rt_i, tor.lu, bcin_tor_r, bcin_tor_i, z, z, bw)

    # 4. poloidal CNAB2 RHS (5c) from OLD pol spec, implicit solve (5d).
    rp_r = similar(pol.spec_r); rp_i = similar(pol.spec_i)     # rp ≠ pol.spec — build_rhs reads pol.spec
    gpu_build_rhs_cnab2!(rp_r, rp_i, pol.spec_r, pol.spec_i, nlp_r, nlp_i,
        pol.prev_nl_r, pol.prev_nl_i, pol.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rp_r, rp_i, pol.lu, bcin_pol_r, bcin_pol_i, z, z, bw)

    # 5. update the fields (AFTER every read of old spec / old pol.prev_nl — ORDERING INVARIANT).
    tor.spec_r .= rt_r; tor.spec_i .= rt_i
    pol.spec_r .= rp_r; pol.spec_i .= rp_i

    # 5b. conducting: reconstruct the inner-core profile from the NEW outer-core ICB value
    #     (g = spec[:,:,1], materialized). Reconstruct reads the OLD ic state (5l requires
    #     S_new ≠ S_old → scratch), then writes the new profile back into the ic state.
    if ic !== nothing
        gt_r = tor.spec_r[:, :, 1]; gt_i = tor.spec_i[:, :, 1]
        gp_r = pol.spec_r[:, :, 1]; gp_i = pol.spec_i[:, :, 1]
        nic_tr = similar(ic.tor_ic_r); nic_ti = similar(ic.tor_ic_i)
        nic_pr = similar(ic.pol_ic_r); nic_pi = similar(ic.pol_ic_i)
        gpu_reconstruct_inner_core!(nic_tr, nic_ti, ic.tor_ic_r, ic.tor_ic_i, gt_r, gt_i, ic.tor_adm)
        gpu_reconstruct_inner_core!(nic_pr, nic_pi, ic.pol_ic_r, ic.pol_ic_i, gp_r, gp_i, ic.pol_adm)
        ic.tor_ic_r .= nic_tr; ic.tor_ic_i .= nic_ti
        ic.pol_ic_r .= nic_pr; ic.pol_ic_i .= nic_pi
    end

    # 6. roll histories: prev_nl ← this step's nl (captured at step 1).
    tor.prev_nl_r .= nlt_r; tor.prev_nl_i .= nlt_i
    pol.prev_nl_r .= nlp_r; pol.prev_nl_i .= nlp_i
    return nothing
end
