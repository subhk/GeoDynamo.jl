# =============================================================================
# GPU Phase 5m — one magnetic field CNAB2 timestep (toroidal + poloidal,
# INSULATING inner core), composing the verified pieces: magnetic nonlinear
# (5h, induction ∇×(u×B), returns BOTH nl_tor and nl_pol) → CNAB2 RHS (5c,
# mass_coeff=1) → implicit solve (5d) for tor then pol → field update + nl_prev
# rollover.  Mirrors apply_magnetic_toroidal/poloidal_implicit_update! (insulating
# CNAB2 branch).  No new kernels.  The toroidal inner boundary optionally takes
# the CONTINUITY_MAG increment −nl_pol[ICB] + prev_nl_pol[ICB]; the poloidal is
# homogeneous.  The conducting-inner-core path is Phase 5m2.  Runs on Array +
# CuArray.  (Per-call scratch — Phase-6 may cache.)
#
# Bundles:  tor/pol :: (; spec_r, spec_i, prev_nl_r, prev_nl_i, lin, lu)
#           nlops   :: (; d1, d2, lfac, rinv, rinv2, rscale)
# =============================================================================

"""
    gpu_magnetic_field_step!(tor, pol, u_r, u_θ, u_φ, config, nlops, inv_dt, linear_weight,
                             lmax, bw; continuity_mag=false) -> nothing

Advance the magnetic field one CNAB2 step (insulating inner core).  `tor`/`pol`
are NamedTuple bundles `(; spec_r, spec_i, prev_nl_r, prev_nl_i, lin, lu)`; on exit
`*.spec_*` is the updated field and `*.prev_nl_*` holds THIS step's nonlinear term.
`u_*` is the physical velocity (supplied — from the velocity step).  `nlops` carries
the magnetic nonlinear/curl operators.  `inv_dt = 1/dt` and `lin` carry the magnetic
mass coefficient (η is in `lin`); `linear_weight = 1−θ`.

`continuity_mag=true` applies the `CONTINUITY_MAG` toroidal inner-boundary coupling:
the toroidal inner RHS row is set to `−nl_pol[ICB] + prev_nl_pol[ICB]` (ICB = radial
index 1), computed from the just-formed poloidal nonlinear and the OLD poloidal
history.  Otherwise the toroidal inner row is 0.  The poloidal is fully homogeneous.

ORDERING INVARIANT (as in `gpu_velocity_field_step!`): the nonlinear, both
`build_rhs` calls, and the `CONTINUITY_MAG` BC all read OLD state (`*.spec_*`,
`pol.prev_nl_*`); the field/history are overwritten ONLY after every such read.
All arrays on the same backend.
"""
function gpu_magnetic_field_step!(tor, pol, u_r, u_θ, u_φ, config, nlops,
        inv_dt, linear_weight, lmax::Int, bw::Int; continuity_mag::Bool = false)
    nl, nm, _ = size(tor.spec_r)
    # 1. magnetic nonlinear (5h): nl_tor/nl_pol from the OLD B (tor/pol spec).
    nlt_r = similar(tor.spec_r); nlt_i = similar(tor.spec_i)   # Phase-6: workspace
    nlp_r = similar(pol.spec_r); nlp_i = similar(pol.spec_i)
    gpu_magnetic_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i,
        tor.spec_r, tor.spec_i, pol.spec_r, pol.spec_i, u_r, u_θ, u_φ, config,
        nlops.d1, nlops.d2, nlops.lfac, nlops.rinv, nlops.rinv2, nlops.rscale, lmax, bw)

    # 2. toroidal BC rows. Inner = CONTINUITY_MAG increment −nl_pol[ICB]+prev_nl_pol[ICB]
    #    (computed from nl_pol + OLD pol.prev_nl, both read before any overwrite) or 0;
    #    outer = 0. bcin/z are (nl,nm) on the same backend.
    z = similar(tor.spec_r, nl, nm); fill!(z, zero(eltype(tor.spec_r)))
    bcin_r = similar(z); bcin_i = similar(z)
    if continuity_mag
        @views bcin_r .= .-nlp_r[:, :, 1] .+ pol.prev_nl_r[:, :, 1]
        @views bcin_i .= .-nlp_i[:, :, 1] .+ pol.prev_nl_i[:, :, 1]
    else
        fill!(bcin_r, zero(eltype(bcin_r))); fill!(bcin_i, zero(eltype(bcin_i)))
    end

    # 3. toroidal CNAB2 RHS (5c) from OLD tor spec, then implicit solve (5d).
    rt_r = similar(tor.spec_r); rt_i = similar(tor.spec_i)     # rt ≠ tor.spec — build_rhs reads tor.spec
    gpu_build_rhs_cnab2!(rt_r, rt_i, tor.spec_r, tor.spec_i, nlt_r, nlt_i,
        tor.prev_nl_r, tor.prev_nl_i, tor.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rt_r, rt_i, tor.lu, bcin_r, bcin_i, z, z, bw)

    # 4. poloidal CNAB2 RHS (5c) from OLD pol spec, homogeneous solve (5d).
    rp_r = similar(pol.spec_r); rp_i = similar(pol.spec_i)     # rp ≠ pol.spec — build_rhs reads pol.spec
    gpu_build_rhs_cnab2!(rp_r, rp_i, pol.spec_r, pol.spec_i, nlp_r, nlp_i,
        pol.prev_nl_r, pol.prev_nl_i, pol.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rp_r, rp_i, pol.lu, z, z, z, z, bw)

    # 5. update the fields (AFTER every read of old spec / old pol.prev_nl — ORDERING INVARIANT).
    tor.spec_r .= rt_r; tor.spec_i .= rt_i
    pol.spec_r .= rp_r; pol.spec_i .= rp_i
    # 6. roll histories: prev_nl ← this step's nl (captured at step 1).
    tor.prev_nl_r .= nlt_r; tor.prev_nl_i .= nlt_i
    pol.prev_nl_r .= nlp_r; pol.prev_nl_i .= nlp_i
    return nothing
end
