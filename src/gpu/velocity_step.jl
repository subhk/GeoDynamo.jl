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
        wsplit = nothing,
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

    # 3. poloidal update. With a W-split operator pack (Stage-4B pressure-free
    #    double-curl form, matching the CPU _apply_poloidal_wsplit_cnab2!):
    #    W = D_pol·P advances by CNAB2 with N_W as forcing, then P is recovered
    #    from W with Dirichlet walls + the endpoint influence corrections.
    #    Without a pack (wsplit = nothing): the legacy projection-form CNAB2 +
    #    2×2 influence correction (kernel-wiring tests; NOT CPU-parity).
    rp_r = similar(pol.spec_r); rp_i = similar(pol.spec_i)     # Phase-6: workspace

    # rp ≠ pol.spec is REQUIRED — build_rhs reads pol.spec as input (ORDERING INVARIANT).
    gpu_build_rhs_cnab2!(rp_r, rp_i, pol.spec_r, pol.spec_i, nlp_r, nlp_i,
        pol.prev_nl_r, pol.prev_nl_i, pol.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rp_r, rp_i, pol.lu,
        pol.bc_in_r, pol.bc_in_i, pol.bc_out_r, pol.bc_out_i, bw)
    gpu_velocity_poloidal_influence_correction!(rp_r, rp_i, influence.Gre_b, influence.invG_b)
  
    if wsplit !== nothing
        gpu_poloidal_wsplit_advance!(rp_r, pol.spec_r, nlp_r, pol.prev_nl_r,
            wsplit, inv_dt, linear_weight, bw)
        gpu_poloidal_wsplit_advance!(rp_i, pol.spec_i, nlp_i, pol.prev_nl_i,
            wsplit, inv_dt, linear_weight, bw)
    else
        # rp ≠ pol.spec is REQUIRED — build_rhs reads pol.spec as input (ORDERING INVARIANT).
        gpu_build_rhs_cnab2!(rp_r, rp_i, pol.spec_r, pol.spec_i, nlp_r, nlp_i,
            pol.prev_nl_r, pol.prev_nl_i, pol.lin, inv_dt, linear_weight, bw)
        gpu_implicit_solve_field!(rp_r, rp_i, pol.lu,
            pol.bc_in_r, pol.bc_in_i, pol.bc_out_r, pol.bc_out_i, bw)
        gpu_velocity_poloidal_influence_correction!(rp_r, rp_i, influence.Gre_b, influence.invG_b)
    end

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

"""
    gpu_poloidal_wsplit_advance!(P_new, P, nl, prev_nl, ws, inv_dt, linear_weight, bw) -> P_new

One CNAB2 step of the Stage-4B poloidal W-split for ONE split-complex component
(call once for real, once for imag), batched over all `(l,m)` modes.  Mirrors
the CPU `_apply_poloidal_wsplit_cnab2!` (physics/velocity/solver.jl):

1. `W = D_pol·P`, `LW = Ek·D_pol·W`
2. `rhs = (Ek/dt)·W + (1−θ)·LW + 1.5·nl − 0.5·prev_nl`   (`nl` carries N_W)
3. `W⁺ = A_W⁻¹·rhs` (banded LU per degree)
4. Dirichlet P-recovery: zero `W⁺` endpoint rows, `P⁺ = A_P⁻¹·W⁺`
5. endpoint-residual influence correction:
   `ρᵢ = d1_rowᵢ·P⁺`, solve the per-degree 2×2 `M·a = −ρ`,
   `P_new = P⁺ + a₁·h₁ + a₂·h₂`; the `l = 0` plane carries no poloidal flow.

`ws` is the device-state `wsplit` operator pack (see `_build_wsplit_pack`).
`inv_dt = Ek/dt` (the same mass-scaled value the toroidal CNAB2 uses) and
`linear_weight = 1−θ`.  `P_new` must not alias `P`/`nl`/`prev_nl`.
"""
function gpu_poloidal_wsplit_advance!(P_new, P, nlv, prev_nl, ws,
        inv_dt, linear_weight, bw::Int)
    T = eltype(P)
    nr = size(P, 3)
    W = similar(P)
    gpu_batched_banded_matvec_perl!(W, P, ws.dpol, bw)
    LW = similar(P)
    gpu_batched_banded_matvec_perl!(LW, W, ws.wlin, bw)
    rhs = similar(P)
    @. rhs = inv_dt * W + linear_weight * LW + T(1.5) * nlv - T(0.5) * prev_nl
    Wp = similar(P)
    gpu_batched_banded_solve!(Wp, rhs, ws.wlu, bw)
    Wp[:, :, 1] .= zero(T)
    Wp[:, :, nr] .= zero(T)
    Pp = similar(P)
    gpu_batched_banded_solve!(Pp, Wp, ws.plu, bw)
    # endpoint first-derivative residuals, per (l,m)
    rho1 = dropdims(sum(Pp .* reshape(ws.d1_inner, 1, 1, :); dims = 3); dims = 3)
    rho2 = dropdims(sum(Pp .* reshape(ws.d1_outer, 1, 1, :); dims = 3); dims = 3)
    M11 = reshape(ws.M[1, 1, :], :, 1); M12 = reshape(ws.M[1, 2, :], :, 1)
    M21 = reshape(ws.M[2, 1, :], :, 1); M22 = reshape(ws.M[2, 2, :], :, 1)
    det = @. M11 * M22 - M12 * M21
    det = @. det + T(det == 0)             # unfilled degree slots → harmless 1
    a1 = @. (-rho1 * M22 + rho2 * M12) / det
    a2 = @. (-rho2 * M11 + rho1 * M21) / det
    h1 = reshape(ws.h1, size(ws.h1, 1), 1, nr)   # (nl, 1, nr), zero-copy
    h2 = reshape(ws.h2, size(ws.h2, 1), 1, nr)
    @. P_new = Pp + a1 * h1 + a2 * h2
    P_new[1, :, :] .= zero(T)              # l = 0 carries no poloidal flow
    return P_new
end
