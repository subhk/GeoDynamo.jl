using Test
using GeoDynamo
using MPI
using Random
using LinearAlgebra

MPI.Initialized() || MPI.Init()

# =============================================================================
# Observed temporal order of accuracy, end to end through the real solver.
#
# Nothing else in the suite measures this. `erk2_*.jl` and `cb3_*.jl` check
# kernels, caches and matrix functions in isolation; the parity files check that
# two schemes agree and that boundary residuals are small. All of that is
# consistent with a scheme that converges at the wrong RATE — which is what an
# order study, and only an order study, detects.
#
# METHOD: self-convergence. Integrate the same initial state to the same physical
# time with dt, dt/2, dt/4; with u(dt) = u* + C*dt^p the successive differences
# fall by 2^p, so p = log2(|u(dt)-u(dt/2)| / |u(dt/2)-u(dt/4)|).
#
# TWO PROBE PITFALLS, both of which produce a WRONG order-1 verdict for every
# scheme, and both of which cost real time to find:
#
#  1. `initialize_solver_fields!` seeds l = 1..4 with `rand(T)` INSIDE the radial
#     loop (physics/temperature/solver.jl:80), i.e. a radially WHITE profile whose
#     stiffest radial modes have |lambda|*dt = O(1) at every dt worth testing. That
#     alone reduces the observed order to ~1. Adding a smooth perturbation on top
#     does NOT fix it — the noisy part still dominates the norm. The l >= 1 content
#     has to be REPLACED, which is what `_smooth_lge1!` does.
#  2. A perturbation that does not vanish at the walls violates the Dirichlet /
#     no-slip rows the implicit solve enforces and reduces the order for reasons
#     unrelated to the scheme. `sin^2` vanishes with its first derivative at both
#     walls, so Dirichlet, zero-Neumann and no-slip all hold.
#
# The probe's ability to resolve order 2 is not assumed — CNAB2 is asserted at
# >= 1.8 under exactly the same grid, state, norm and dt ladder, so a scheme that
# reports 1.0 here is being compared against a working control.
# =============================================================================

const _TO_VAL = GeoDynamo.ValueBoundaryCondition

function _smooth_lge1!(field, cfg, nr, seed)
    rng = MersenneTwister(seed)
    pr = parent(field.data_real)
    pim = parent(field.data_imag)
    amps = Dict{Tuple{Int, Int}, Tuple{Float64, Float64}}()
    for lm in 1:cfg.nlm
        amps[(cfg.l_values[lm], cfg.m_values[lm])] =
            (1e-3 * (rand(rng) - 0.5), 1e-3 * (rand(rng) - 0.5))
    end
    for lm in 1:cfg.nlm
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        l = cfg.l_values[lm]
        m = cfg.m_values[lm]
        l >= 1 || continue
        ar, ai = amps[(l, m)]
        for k in 1:nr
            w = sin(pi * (k - 1) / (nr - 1))^2
            GeoDynamo.set_local_spectral_value!(pr, slot, k, ar * w)
            GeoDynamo.set_local_spectral_value!(pim, slot, k, m > 0 ? ai * w : 0.0)
        end
    end
    return field
end

function _to_build(ts, dt; zero_velocity::Bool = false, bcs = nothing, seed::Int = 7)
    st = GeoDynamo.initialize_solver_state(Float64;
        params = GeoDynamo.SolverParameters(;
            geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24,
            nr = 16, nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
            Ek = 1e-3, Ra = zero_velocity ? 1e-8 : 1e3, Pm = 1.0, Pr = 1.0, timestep = dt,
            include_magnetic = false, include_composition = false, timestepper = ts,
            temperature_bcs = bcs === nothing ? GeoDynamo.BoundaryConditions(
                inner = _TO_VAL(1.0), outer = _TO_VAL(0.0)) : bcs,
            velocity_bcs = GeoDynamo.BoundaryConditions(
                inner = GeoDynamo.NoSlip(), outer = GeoDynamo.NoSlip())))
    GeoDynamo.initialize_solver_fields!(st)
    cfg = st.backend.shtns_config
    nr = st.runtime.outer_core_domain.N
    for (i, f) in enumerate((st.fields.temperature.spectral,
        st.fields.velocity.toroidal, st.fields.velocity.poloidal))
        _smooth_lge1!(f, cfg, nr, seed + i)
    end
    if zero_velocity
        # Kills the advecting flow, so temperature is an exactly LINEAR diffusion
        # problem. A scheme that loses order HERE cannot be blamed on the coupling.
        # Zeroing the velocity is NOT sufficient on its own: with Ra = 1e3 buoyancy
        # regenerates a flow on the very first step and the problem is nonlinear
        # again — which is why `Ra` is dropped to 1e-8 above whenever this is set.
        # Getting that wrong makes ERK2 look first order on a "linear" problem when
        # it is in fact exact there.
        for f in (st.fields.velocity.toroidal, st.fields.velocity.poloidal)
            fill!(parent(f.data_real), 0.0)
            fill!(parent(f.data_imag), 0.0)
        end
    end
    return st
end

"Integrate to `T_end` and return the l >= 1 spectral content of `field_name`."
function _to_final(ts, dt, T_end, field_name::Symbol; kw...)
    st = _to_build(ts, dt; kw...)
    nsteps = round(Int, T_end / dt)
    @assert isapprox(nsteps * dt, T_end; rtol = 1e-12)
    for _ in 1:nsteps
        GeoDynamo.solver_step!(st)
    end
    cfg = st.backend.shtns_config
    nr = st.runtime.outer_core_domain.N
    f = field_name === :temperature ? st.fields.temperature.spectral :
        field_name === :velocity_toroidal ? st.fields.velocity.toroidal :
        st.fields.velocity.poloidal
    a, b = GeoDynamo.cpu_spectral_to_dense(f, cfg, nr, Float64)
    return vcat(vec(a[2:end, :, :]), vec(b[2:end, :, :]))
end

const _TO_T_END = 4e-4
const _TO_DTS = (4e-5, 2e-5, 1e-5)

"Observed self-convergence order of `field_name` under `ts`."
function observed_order(ts, field_name::Symbol; kw...)
    us = [_to_final(ts, dt, _TO_T_END, field_name; kw...) for dt in _TO_DTS]
    e1 = norm(us[1] - us[2], Inf)
    e2 = norm(us[2] - us[3], Inf)
    (e1 > 0 && e2 > 0) || return NaN
    return log2(e1 / e2)
end


"Dense form of a `BandedOperator`, obtained by applying it to unit vectors."
function _to_dense(op)
    n = op.size
    M = zeros(Float64, n, n)
    y = zeros(Float64, n)
    e = zeros(Float64, n)
    for j in 1:n
        fill!(e, 0.0); e[j] = 1.0; fill!(y, 0.0)
        GeoDynamo.apply_banded_full!(y, op, e)
        M[:, j] .= y
    end
    return M
end

"""
    _to_matrix_order(bcs, u0; project) -> Float64

Observed order of the CNAB2 linear update run on the solver's OWN matrices with the
nonlinear term identically zero. `project` first solves the 2x2 endpoint block so `u0`
satisfies the discrete constraint rows exactly.
"""
function _to_matrix_order(bcs, u0; project::Bool)
    nr = length(u0)
    function march(dt)
        st = GeoDynamo.initialize_solver_state(Float64;
            params = GeoDynamo.SolverParameters(;
                geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24,
                nr = nr, nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
                Ek = 1e-3, Ra = 1e-8, Pm = 1.0, Pr = 1.0, timestep = dt,
                include_magnetic = false, include_composition = false,
                timestepper = GeoDynamo.CNAB2(), temperature_bcs = bcs,
                velocity_bcs = GeoDynamo.BoundaryConditions(
                    inner = GeoDynamo.NoSlip(), outer = GeoDynamo.NoSlip())))
        GeoDynamo.initialize_solver_fields!(st)
        ms = st.implicit_matrices[:temperature]
        idx = ms.lookup[1]
        sys = _to_dense(ms.system_matrices[idx])
        lin = _to_dense(ms.linear_matrices[idx])
        u = copy(u0)
        if project
            C = sys[[1, nr], :]
            u[[1, nr]] = -(C[:, [1, nr]] \ (C[:, 2:(nr - 1)] * u0[2:(nr - 1)]))
        end
        for _ in 1:round(Int, _TO_T_END / dt)
            rhs = (1 / dt) .* u .+ (1 - ms.theta) .* (lin * u)
            rhs[1] = 0.0
            rhs[nr] = 0.0
            u = sys \ rhs
        end
        return u
    end
    us = [march(dt) for dt in _TO_DTS]
    e1 = norm(us[1] - us[2], Inf)
    e2 = norm(us[2] - us[3], Inf)
    return (e1 > 0 && e2 > 0) ? log2(e1 / e2) : NaN
end

@testset "timestepper temporal order of accuracy" begin

    # ── CNAB2: the control, and a real assertion ──────────────────────────────
    # Crank-Nicolson diffusion + Adams-Bashforth-2 advection is second order, and
    # measuring that here is what proves the probe resolves order 2 at all.
    @testset "CNAB2 is second order" begin
        for f in (:temperature, :velocity_toroidal, :velocity_poloidal)
            p = observed_order(GeoDynamo.CNAB2(), f)
            @test p >= 1.8
            @test p <= 2.3
        end
    end

    # ── ERK2 on an exactly linear problem: EXACT ──────────────────────────────
    # With no flow and no buoyancy the nonlinear term is constant in time, so the
    # exponential-midpoint correction 2*dt*phi2*(N(u*) - N(u)) vanishes and
    # u+ = E*u + dt*phi1*N is the exact solution. Measuring roundoff here is the
    # statement that BOTH the propagator AND the Dirichlet boundary elimination
    # (`solver_erk2_constrained_propagators`) are right.
    @testset "ERK2 is exact on a linear Dirichlet problem" begin
        us = [_to_final(GeoDynamo.ExponentialRungeKutta2(), dt, _TO_T_END,
            :temperature; zero_velocity = true) for dt in _TO_DTS]
        scale = norm(us[end], Inf)
        @test scale > 1e-6                                   # the state is not trivially zero
        @test norm(us[1] - us[2], Inf) < 1e-14 * scale
        @test norm(us[2] - us[3], Inf) < 1e-14 * scale
    end

    # ── CNAB2 and RK3 keep their order on the same linear problem ─────────────
    @testset "CNAB2 and RK3 stay >= second order on a linear Dirichlet problem" begin
        for ts in (GeoDynamo.CNAB2(), GeoDynamo.RungeKutta3())
            @test observed_order(ts, :temperature; zero_velocity = true) >= 1.8
        end
    end

    # ── A Neumann wall does NOT cost an order — but an inconsistent IC fakes one ─
    # Driven straight through the solver from a grid-sampled `sin^2`, swapping the two
    # Dirichlet walls for zero-flux walls appears to drop CNAB2 from 2.00 to 1.02. It
    # does not. `sin^2` has zero slope at the walls in the CONTINUOUS sense but sits
    # O(h^p) off the one-sided DIFFERENCE stencil the Neumann row actually imposes, so
    # the run starts off the constraint manifold and pays an initial layer. Projecting
    # the endpoints onto the discrete constraint rows restores 2.00.
    #
    # This is asserted at the matrix level — the solver's own `system_matrices` and
    # `linear_matrices`, N == 0, no solver plumbing — because that is what isolates the
    # scheme from every other per-step effect. Both directions are pinned: a consistent
    # IC must give order 2, and an inconsistent one must NOT, so nobody re-derives the
    # false verdict from the cheap experiment.
    @testset "Neumann rows are second order given a constraint-consistent IC" begin
        nr = 16
        u0 = [sin(pi * x)^2 for x in range(0, 1; length = nr)]
        for (name, bcs) in (
                ("DD", GeoDynamo.BoundaryConditions(
                    inner = _TO_VAL(0.0), outer = _TO_VAL(0.0))),
                ("NN", GeoDynamo.BoundaryConditions(
                    inner = GeoDynamo.FluxBoundaryCondition(0.0),
                    outer = GeoDynamo.FluxBoundaryCondition(0.0))))
            p = _to_matrix_order(bcs, u0; project = true)
            @test p >= 1.8
            @test p <= 2.3
        end
        # ... and the trap itself, so the pitfall stays documented by a test
        nn = GeoDynamo.BoundaryConditions(
            inner = GeoDynamo.FluxBoundaryCondition(0.0),
            outer = GeoDynamo.FluxBoundaryCondition(0.0))
        @test _to_matrix_order(nn, u0; project = false) < 1.5
        dd = GeoDynamo.BoundaryConditions(inner = _TO_VAL(0.0), outer = _TO_VAL(0.0))
        @test _to_matrix_order(dd, u0; project = false) >= 1.8   # Dirichlet IC is consistent
    end

    # ── ERK2 and RK3 lose order through the NONLINEAR COUPLING ────────────────
    # Both are exact/second order on the linear problem above, and CNAB2 is a clean
    # second order on this very state, so the loss is in the coupled path: the
    # exponential-midpoint stage correction for ERK2, and the explicit stage
    # coupling for RK3 (nominally a THIRD-order IMEX pair).
    @testset "ERK2 and RK3 lose order through the nonlinear coupling" begin
        for ts in (GeoDynamo.ExponentialRungeKutta2(), GeoDynamo.RungeKutta3())
            for f in (:temperature, :velocity_toroidal, :velocity_poloidal)
                p = observed_order(ts, f)
                @test_broken p >= 1.8
                @test p >= 0.7  # pin current behaviour
            end
        end
    end
end
