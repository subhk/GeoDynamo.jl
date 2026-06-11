# =============================================================================
# GPU Phase 5k — one velocity field CNAB2 timestep (toroidal + poloidal),
# composing the verified pieces: velocity nonlinear (5i, returns BOTH nl_tor and
# nl_pol) → CNAB2 RHS (5c, per component) → implicit solve (5d, per component) →
# poloidal influence-matrix correction (5j) → field update + nl_prev rollover.
# Mirrors apply_velocity_toroidal_implicit_update! + apply_velocity_poloidal_
# implicit_update! + the CNAB2 history rollover.  No new kernels — pure
# composition.  Runs on Array + CuArray.  (Per-call scratch — Phase-6 may cache.)
#
# Per-component state is grouped into NamedTuple bundles `tor`/`pol` (mutated in
# place through their array fields) to keep the argument list legible:
#   tor/pol :: (; spec_r, spec_i, prev_nl_r, prev_nl_i, lin, lu,
#                 bc_in_r, bc_in_i, bc_out_r, bc_out_i)
#   nlops   :: (; d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E)
#   influence :: (; Gre_b, invG_b)
# =============================================================================

"""
    gpu_velocity_field_step!(tor, pol, config, nlops, influence, inv_dt, linear_weight,
                             lmax, bw; <coupling kwargs>) -> nothing

Advance the velocity field one CNAB2 step.  `tor`/`pol` are NamedTuple bundles of
the toroidal/poloidal arrays (see the file header); on entry `*.spec_*` is the
field and `*.prev_nl_*` the previous nonlinear term, on exit `*.spec_*` is the
updated field and `*.prev_nl_*` holds THIS step's nonlinear term (rolled over).
`nlops` carries the nonlinear/curl operators, `influence` the poloidal 2×2
correction operators.  `inv_dt = E/dt` and the per-l `lin` operators must already
carry the mass coefficient `E` (so 5c reproduces the velocity RHS); `linear_weight
= 1−θ`.  Coupling kwargs (thermal/compositional buoyancy, Lorentz) are forwarded
to [`gpu_velocity_nonlinear!`](@ref); omit them for the velocity-only step.  All
arrays on the same backend.

ROTATING INNER CORE: the toroidal `bc_in_*`/`bc_out_*` rows are applied verbatim by
the implicit solve; for a rotating inner core the caller must set the `l=1, m=0` inner
BC slot to the prescribed rotation value (`rot_omega·r_inner`, incremental) BEFORE
calling — this function does not assemble it.

ORDERING INVARIANT (as in `gpu_scalar_field_step!`): the nonlinear and BOTH
`build_rhs` calls read the OLD `*.spec_*`; the spec is overwritten with the
solution ONLY after every such read.  Do not move the field-update copies earlier.
"""
function gpu_velocity_field_step!(tor, pol, config, nlops, influence,
        inv_dt, linear_weight, lmax::Int, bw::Int;
        T_phys = nothing, thermal_factor = zero(eltype(tor.spec_r)), r_vec = nothing,
        C_phys = nothing, comp_factor = zero(eltype(tor.spec_r)),
        J_r = nothing, J_θ = nothing, J_φ = nothing,
        B_r = nothing, B_θ = nothing, B_φ = nothing, lorentz_coeff = zero(eltype(tor.spec_r)))

    # 1. velocity nonlinear (5i): nl_tor / nl_pol captured from the OLD tor/pol spec.
    nlt_r = similar(tor.spec_r)
    nlt_i = similar(tor.spec_i)   # Phase-6: workspace
    nlp_r = similar(pol.spec_r)
    nlp_i = similar(pol.spec_i)

    gpu_velocity_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i,
        tor.spec_r, tor.spec_i, pol.spec_r, pol.spec_i, config,
        nlops.d1, nlops.d2, nlops.lfac, nlops.rinv, nlops.rinv2, nlops.rscale,
        nlops.sinθ, nlops.cosθ, nlops.E, lmax, bw;
        T_phys = T_phys, thermal_factor = thermal_factor, r_vec = r_vec,
        C_phys = C_phys, comp_factor = comp_factor,
        J_r = J_r, J_θ = J_θ, J_φ = J_φ, B_r = B_r, B_θ = B_θ, B_φ = B_φ,
        lorentz_coeff = lorentz_coeff)

    # 2. toroidal CNAB2 RHS (5c) from OLD tor spec, then implicit solve (5d).
    rt_r = similar(tor.spec_r); rt_i = similar(tor.spec_i)     # Phase-6: workspace
    # rt ≠ tor.spec is REQUIRED — build_rhs reads tor.spec as input (ORDERING INVARIANT).
    gpu_build_rhs_cnab2!(rt_r, rt_i, tor.spec_r, tor.spec_i, nlt_r, nlt_i,
        tor.prev_nl_r, tor.prev_nl_i, tor.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rt_r, rt_i, tor.lu,
        tor.bc_in_r, tor.bc_in_i, tor.bc_out_r, tor.bc_out_i, bw)

    # 3. poloidal CNAB2 RHS (5c) from OLD pol spec, implicit solve (5d), then the
    #    2×2 influence correction (5j) on the poloidal solution.
    rp_r = similar(pol.spec_r); rp_i = similar(pol.spec_i)     # Phase-6: workspace

    # rp ≠ pol.spec is REQUIRED — build_rhs reads pol.spec as input (ORDERING INVARIANT).
    gpu_build_rhs_cnab2!(rp_r, rp_i, pol.spec_r, pol.spec_i, nlp_r, nlp_i,
        pol.prev_nl_r, pol.prev_nl_i, pol.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rp_r, rp_i, pol.lu,
        pol.bc_in_r, pol.bc_in_i, pol.bc_out_r, pol.bc_out_i, bw)
    gpu_velocity_poloidal_influence_correction!(rp_r, rp_i, influence.Gre_b, influence.invG_b)

    # 4. update the fields (AFTER every read of the old spec — ORDERING INVARIANT).
    tor.spec_r .= rt_r
    tor.spec_i .= rt_i
    pol.spec_r .= rp_r
    pol.spec_i .= rp_i

    # 5. roll the histories: prev_nl ← this step's nl (captured at step 1).
    tor.prev_nl_r .= nlt_r
    tor.prev_nl_i .= nlt_i
    pol.prev_nl_r .= nlp_r
    pol.prev_nl_i .= nlp_i

    return nothing
end
