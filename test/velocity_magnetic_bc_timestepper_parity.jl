using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

include(joinpath(@__DIR__, "parity", "fixtures.jl"))
using .ParityFixtures

# =============================================================================
# End-to-end velocity and magnetic boundary-condition enforcement, per timestepper.
#
# Sibling of scalar_bc_timestepper_parity.jl, same contract and same reason for
# existing: velocity_boundary_numerical.jl and magnetic_boundary_numerical.jl
# build the implicit matrices and check their boundary ROWS in isolation, which
# is structurally blind to a timestepper whose update path bypasses or
# post-multiplies those rows. That is precisely how the ERK2 magnetic Robin
# (2026-07-20), ERK2 scalar Neumann (2026-07-21) and ERK2 l=0 NN (2026-08-02)
# bugs all survived a green suite.
#
# The invariant under test is cross-implementation: EVERY timestepper must
# enforce the SAME discrete boundary condition on the EVOLVED field. Any one of
# those three bugs would have failed this file on the day it was introduced.
# =============================================================================

const _VM_TS = [
    ("CNAB2", GeoDynamo.CNAB2()),
    ("ERK2",  GeoDynamo.ExponentialRungeKutta2()),
    ("RK3",   GeoDynamo.RungeKutta3()),
]

# velocity_bc_code: 1 = NS/NS, 2 = NS inner / SF outer, 3 = SF inner / NS outer, 4 = SF/SF
#
# All four wall codes x three timesteppers, with the magnetic assertions folded
# into the same states.
#
# This file was briefly cut to a single wall code: every SolverState used to
# allocate four MPI communicators that were never freed, and the suite sat near
# MPICH's 2048 ceiling, so a full-matrix version exhausted it and took 21
# unrelated downstream files down with it. That is fixed at the source —
# create_pencil_decomposition_shtnskit now memoizes per grid — so all twelve
# states here share one decomposition and the full matrix fits.
const _VM_WALLS = [
    (1, "code1 NS/NS", GeoDynamo.NoSlip(),     GeoDynamo.NoSlip()),
    (2, "code2 NS/SF", GeoDynamo.NoSlip(),     GeoDynamo.StressFree()),
    (3, "code3 SF/NS", GeoDynamo.StressFree(), GeoDynamo.NoSlip()),
    (4, "code4 SF/SF", GeoDynamo.StressFree(), GeoDynamo.StressFree()),
]

# State construction now comes from the shared parity fixture. The previous local
# builder perturbed the fields BEFORE is_initialized was set, so solver_step!
# regenerated the IC on its first call and discarded the perturbation entirely —
# every seed produced the same trajectory. ParityFixtures.build_state calls
# initialize_solver_fields! first, so the perturbation now survives and these
# assertions run against a genuinely richer state.
#
# `wall_code` is taken directly from the caller's loop variable (see _VM_WALLS)
# rather than recovered by identity comparison against ParityFixtures.WALL_BCS
# values — a fresh `BoundaryConditions` at each call site would never
# `===`-match, so that lookup would silently fall back to a default and narrow
# coverage without failing anything. scalar_code is fixed at 1 (DD), matching
# this file's previous omission of temperature_bcs (SolverParameters' own
# default).
function _vm_state(timestepper, wall_code::Int; magnetic = false, seed = 11)
    case = ParityFixtures.ParityCase(
        string(typeof(timestepper).name.name), timestepper,
        1, wall_code, magnetic, false)
    return ParityFixtures.build_state(case; seed = seed)
end

# Worst relative residual of `resid(p, dp, ddp, l, r)` over every l >= 1 mode
# carrying amplitude, plus the number of modes examined. A sweep that scans
# nothing also reports 0.0, so callers assert the count.
function _vm_residual(st, spec, resid; derivative_scale::Bool = false)
    cfg = st.backend.shtns_config
    dom = st.runtime.outer_core_domain
    nr = dom.N
    r = Float64[dom.r[k, 4] for k in 1:nr]
    d1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    d2 = GeoDynamo.create_derivative_matrix(Float64, 2, dom)
    sr, si = GeoDynamo.cpu_spectral_to_dense(spec, cfg, nr, Float64)
    worst = 0.0
    nseen = 0
    for part in (sr, si)
        for mi in 1:size(part, 2), li in max(mi, 2):size(part, 1)
            p = Float64[part[li, mi, k] for k in 1:nr]
            amp = maximum(abs, p)
            amp > 1e-14 || continue
            nseen += 1
            dp = d1 * p
            ddp = d2 * p
            res = resid(p, dp, ddp, li - 1, r)
            scale = derivative_scale ? max(maximum(abs, dp), amp / r[1]) : amp
            worst = max(worst, abs(res) / max(scale, 1e-300))
        end
    end
    return worst, nseen
end

# --- velocity discrete conditions (src/bcs/velocity_bc.jl:17-23) -------------
#   toroidal no-slip     : T = 0
#   toroidal stress-free : dT/dr - T/r = 0
#   poloidal (both)      : P = 0, imposed by the influence correction
#   poloidal no-slip     : dP/dr = 0
#   poloidal stress-free : d2P/dr2 - (2/r) dP/dr = 0
_vm_tor(noslip, inner) =
    if inner
        noslip ? ((p, dp, ddp, l, r) -> p[1]) : ((p, dp, ddp, l, r) -> dp[1] - p[1] / r[1])
    else
        noslip ? ((p, dp, ddp, l, r) -> p[end]) : ((p, dp, ddp, l, r) -> dp[end] - p[end] / r[end])
    end
_vm_pol(noslip, inner) =
    if inner
        noslip ? ((p, dp, ddp, l, r) -> dp[1]) : ((p, dp, ddp, l, r) -> ddp[1] - 2dp[1] / r[1])
    else
        noslip ? ((p, dp, ddp, l, r) -> dp[end]) : ((p, dp, ddp, l, r) -> ddp[end] - 2dp[end] / r[end])
    end
_vm_pol_value(inner) = inner ? ((p, dp, ddp, l, r) -> p[1]) : ((p, dp, ddp, l, r) -> p[end])

# --- magnetic insulating (potential-field matching) -------------------------
#   inner: P ~ r^{l+1}  =>  dP/dr - (l+1)/r P = 0
#   outer: P ~ r^{-l}   =>  dP/dr + l/r     P = 0
#   toroidal: B_T = 0 at both walls
_vm_mag_pol(inner) = inner ?
    ((p, dp, ddp, l, r) -> dp[1] - (l + 1) / r[1] * p[1]) :
    ((p, dp, ddp, l, r) -> dp[end] + l / r[end] * p[end])
_vm_mag_tor(inner) = inner ? ((p, dp, ddp, l, r) -> p[1]) : ((p, dp, ddp, l, r) -> p[end])

@testset "Velocity/magnetic BC enforcement on the evolved state, per timestepper" begin
    NSTEPS = 15
    # Boundary rows are enforced exactly (identity / derivative stencils), so the
    # residual sits at round-off. The second-derivative poloidal rows amplify it
    # most, hence the looser bound there; measured worst over this whole matrix
    # is 4.2e-14, so 1e-11 keeps ~200x headroom while still being a real gate.
    tol = 1e-11

    @testset "velocity + magnetic — $tsname" for (tsname, ts) in _VM_TS
        @testset "$label" for (code, label, inner_bc, outer_bc) in _VM_WALLS
            # One state per (timestepper, wall type), magnetic enabled, reused for
            # both the velocity and the magnetic assertions — see _VM_WALLS on why
            # the state count is kept down.
            st = _vm_state(ts, code; magnetic = true)
            for _ in 1:NSTEPS
                GeoDynamo.solver_step!(st)
            end
            tor = st.fields.velocity.toroidal
            pol = st.fields.velocity.poloidal
            ns_in = (code == 1 || code == 2)
            ns_out = (code == 1 || code == 3)

            ri, ni = _vm_residual(st, tor, _vm_tor(ns_in, true);   derivative_scale = !ns_in)
            ro, no = _vm_residual(st, tor, _vm_tor(ns_out, false); derivative_scale = !ns_out)
            @test ni > 0
            @test no > 0
            @test ri < tol
            @test ro < tol

            # P = 0 on both walls, whatever the wall type (influence correction)
            pvi, npi = _vm_residual(st, pol, _vm_pol_value(true))
            pvo, npo = _vm_residual(st, pol, _vm_pol_value(false))
            @test npi > 0
            @test npo > 0
            @test pvi < tol
            @test pvo < tol

            # ...plus the second poloidal row implied by the wall type
            p2i, _ = _vm_residual(st, pol, _vm_pol(ns_in, true);   derivative_scale = true)
            p2o, _ = _vm_residual(st, pol, _vm_pol(ns_out, false); derivative_scale = true)
            @test p2i < tol
            @test p2o < tol

            # --- magnetic insulating, from the same state --------------------
            mtor = st.fields.magnetic.toroidal
            mpol = st.fields.magnetic.poloidal

            ti, nti = _vm_residual(st, mtor, _vm_mag_tor(true))
            to, nto = _vm_residual(st, mtor, _vm_mag_tor(false))
            @test nti > 0
            @test nto > 0
            @test ti < tol
            @test to < tol

            # The Robin rows are the ones ERK2 only lag-enforced before PR #103.
            mpi_, nmpi = _vm_residual(st, mpol, _vm_mag_pol(true);  derivative_scale = true)
            mpo_, nmpo = _vm_residual(st, mpol, _vm_mag_pol(false); derivative_scale = true)
            @test nmpi > 0
            @test nmpo > 0
            @test mpi_ < tol
            @test mpo_ < tol
        end
    end
end
