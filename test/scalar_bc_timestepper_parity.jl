using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

include(joinpath(@__DIR__, "parity", "fixtures.jl"))
using .ParityFixtures

# =============================================================================
# End-to-end scalar boundary-condition enforcement, per timestepper.
#
# `temperature_boundary_numerical.jl` and `composition_boundary_numerical.jl`
# solve the implicit matrix in ISOLATION and check its boundary rows. That
# cannot see a timestepper whose update path bypasses those rows — the ERK2
# failure mode found in the 2026-07-20/21 audits, where correct rows were
# exponentiated away. This file steps the REAL solver and measures the discrete
# boundary residual on the EVOLVED field, for every live timestepper.
#
# The l = 0 mean mode is checked separately because it is where the NN gauge
# question lives. The rule is stated in scalar_field_solver_common.jl: the
# Dirichlet pin belongs to the SINGULAR STEADY conductive solve only; the
# time-stepping operator (mass/dt)I − θκL is non-singular under pure-Neumann
# rows, so no time-stepping builder may pin l = 0. All timesteppers must
# therefore enforce the same inner Neumann flux on the mean mode.
# =============================================================================

const _SBP_VAL = GeoDynamo.ValueBoundaryCondition
const _SBP_FLX = GeoDynamo.FluxBoundaryCondition

# code => (name, BoundaryConditions, inner_is_dirichlet, outer_is_dirichlet)
const _SBP_CODES = [
    (1, "DD", GeoDynamo.BoundaryConditions(inner = _SBP_VAL(1.0), outer = _SBP_VAL(0.0)), true, true),
    (2, "DN", GeoDynamo.BoundaryConditions(inner = _SBP_VAL(1.0), outer = _SBP_FLX(0.0)), true, false),
    (3, "ND", GeoDynamo.BoundaryConditions(inner = _SBP_FLX(1.0), outer = _SBP_VAL(0.0)), false, true),
    (4, "NN", GeoDynamo.BoundaryConditions(inner = _SBP_FLX(1.0), outer = _SBP_FLX(0.0)), false, false),
]

const _SBP_TS = [
    ("CNAB2", GeoDynamo.CNAB2()),
    ("ERK2",  GeoDynamo.ExponentialRungeKutta2()),
    ("RK3",   GeoDynamo.RungeKutta3()),
]

# State construction now comes from the shared parity fixture. The previous local
# builder perturbed the fields BEFORE is_initialized was set, so solver_step!
# regenerated the IC on its first call and discarded the perturbation entirely —
# every seed produced the same trajectory. ParityFixtures.build_state calls
# initialize_solver_fields! first, so the perturbation now survives and these
# assertions run against a genuinely richer state.
#
# `code` is the scalar BC code (1=DD, 2=DN, 3=ND, 4=NN), taken directly from the
# caller's loop variable rather than recovered by identity comparison against
# ParityFixtures.SCALAR_BCS values — a fresh `BoundaryConditions` at each call
# site would never `===`-match, so that lookup would silently fall back to a
# default and narrow coverage without failing anything.
# wall_code is fixed at 1 (NoSlip/NoSlip), matching this file's previous
# omission of velocity_bcs (SolverParameters' own default, parameters.jl:54).
function _sbp_state(timestepper, code::Int; composition_bcs_code = nothing, seed = 11)
    case = ParityFixtures.ParityCase(
        string(typeof(timestepper).name.name), timestepper,
        code, 1, false, composition_bcs_code !== nothing)
    return ParityFixtures.build_state(case; seed = seed)
end

# Worst relative boundary residual over every l ≥ 1 mode carrying amplitude,
# plus the number of modes actually examined (a sweep that scans nothing also
# reports zero, so the count is asserted by the caller).
function _sbp_residual_lge1(st, spec, dirichlet::Bool, side::Symbol)
    cfg = st.backend.shtns_config
    dom = st.runtime.outer_core_domain
    nr = dom.N
    d1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    sr, si = GeoDynamo.cpu_spectral_to_dense(spec, cfg, nr, Float64)
    b = side === :inner ? 1 : nr
    worst = 0.0
    nseen = 0
    for part in (sr, si)
        for mi in 1:size(part, 2), li in max(mi, 2):size(part, 1)
            p = Float64[part[li, mi, k] for k in 1:nr]
            amp = maximum(abs, p)
            amp > 1e-14 || continue
            nseen += 1
            dp = d1 * p
            # boundary values are zero for l ≥ 1, so the target is 0 either way
            res = dirichlet ? abs(p[b]) : abs(dp[b])
            scale = dirichlet ? amp : max(maximum(abs, dp), amp / Float64(dom.r[1, 4]))
            worst = max(worst, res / max(scale, 1e-300))
        end
    end
    return worst, nseen
end

# l = 0 mean mode: absolute residual against the stored endpoint target.
function _sbp_residual_l0(st, field, spec, dirichlet::Bool, side::Symbol)
    cfg = st.backend.shtns_config
    dom = st.runtime.outer_core_domain
    nr = dom.N
    d1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    sr, _ = GeoDynamo.cpu_spectral_to_dense(spec, cfg, nr, Float64)
    p = Float64[sr[1, 1, k] for k in 1:nr]
    dp = d1 * p
    lm00 = GeoDynamo.get_mode_index(cfg, 0, 0)
    target = lm00 > 0 ? field.boundary_values[side === :inner ? 1 : 2, lm00] : 0.0
    b = side === :inner ? 1 : nr
    got = dirichlet ? p[b] : dp[b]
    return abs(got - target), target, got
end

@testset "Scalar BC enforcement on the evolved state, per timestepper" begin
    NSTEPS = 15
    # Boundary rows are enforced exactly (identity / first-derivative stencil),
    # so residuals should sit at round-off. The Neumann stencil amplifies
    # round-off by ~1/h, hence the looser derivative bound.
    rel_tol = 1e-11

    @testset "$tsname" for (tsname, ts) in _SBP_TS
        @testset "$cname (code $code)" for (code, cname, bcs_, din, dout) in _SBP_CODES
            st = _sbp_state(ts, code)
            for _ in 1:NSTEPS
                GeoDynamo.solver_step!(st)
            end
            field = st.fields.temperature
            spec = field.spectral

            ri, ni = _sbp_residual_lge1(st, spec, din, :inner)
            ro, no = _sbp_residual_lge1(st, spec, dout, :outer)
            # guard against a vacuous sweep reporting a clean zero
            @test ni > 0
            @test no > 0
            @test ri < rel_tol
            @test ro < rel_tol

            # --- l = 0 mean mode -------------------------------------------
            # Every timestepper must honour the SAME endpoint condition here.
            # Under NN the inner side is Neumann like any other degree: the
            # gauge pin belongs to the steady conductive solve, not to the
            # time-stepping operator. Pinning it would (a) diverge from CNAB2 /
            # RK3 and (b) write the prescribed FLUX into the field VALUE.
            l0i, tgt_i, got_i = _sbp_residual_l0(st, field, spec, din, :inner)
            l0o, tgt_o, got_o = _sbp_residual_l0(st, field, spec, dout, :outer)
            @test l0i < 1e-9
            @test l0o < 1e-9
            if code == 4
                # explicit: the inner mean-mode FLUX carries the prescribed
                # value, and the field VALUE there is not pinned to it
                @test isapprox(got_i, tgt_i; atol = 1e-9)
                @test tgt_i > 1.0                      # fixture really is nonzero
                sr, _ = GeoDynamo.cpu_spectral_to_dense(spec, st.backend.shtns_config,
                    st.runtime.outer_core_domain.N, Float64)
                @test !isapprox(sr[1, 1, 1], tgt_i; atol = 1e-6)
            end
        end
    end

    @testset "composition shares the scalar builder (ERK2, NN)" begin
        # temperature and composition go through the same
        # build_solver_erk2_scalar_bc, so the l = 0 NN path must hold for both.
        # code 4 = NN (see _SBP_CODES above).
        st = _sbp_state(GeoDynamo.ExponentialRungeKutta2(), 4; composition_bcs_code = 4)
        for _ in 1:NSTEPS
            GeoDynamo.solver_step!(st)
        end
        field = st.fields.composition
        l0i, tgt_i, got_i = _sbp_residual_l0(st, field, field.spectral, false, :inner)
        @test tgt_i > 1.0
        @test l0i < 1e-9
        @test isapprox(got_i, tgt_i; atol = 1e-9)
    end
end
