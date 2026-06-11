using Test
using MPI
using LinearAlgebra
using Random
using GeoDynamo

MPI.Initialized() || MPI.Init()

# ===========================================================================
# Stage-4B gates: the poloidal momentum W-split.
#
# (c) CONVECTIVE ONSET — the acceptance test for the entire double-curl fix:
#     buoyancy (a purely radial force) must project onto the poloidal momentum
#     equation. Historically nl_poloidal was the raw spheroidal scalar of the
#     force, which is identically zero for a radial force: velocity at rest
#     stayed at rest forever and convection could not start.
# ===========================================================================

function _pm_build(; Ra = 1e6, dt = 1e-4, stepper = GeoDynamo.CNAB2())
    params = GeoDynamo.SolverParameters(
        architecture = :cpu, geometry = :shell,
        nr = 16, nr_inner = 4, lmax = 8, mmax = 8, nlat = 20, nlon = 40,
        Ra = Ra, Ek = 1e-2, Pr = 1.0, Pm = 1.0, Sc = 1.0,
        timestep = dt, start_time = 0.0, end_time = 1.0, stop_iteration = 100000,
        include_magnetic = false, include_composition = false,
        timestepper = stepper, topography_enabled = false, stefan_enabled = false,
    )
    return GeoDynamo.initialize_simulation(Float64, params)
end

_pm_ke(state) = begin
    v = state.fields.velocity
    norm(parent(v.toroidal.data_real))^2 + norm(parent(v.toroidal.data_imag))^2 +
    norm(parent(v.poloidal.data_real))^2 + norm(parent(v.poloidal.data_imag))^2
end

@testset "convective onset: buoyancy projects and drives flow from rest" begin
    state = _pm_build(; Ra = 1e4)
    @test _pm_ke(state) == 0.0   # starts exactly at rest

    GeoDynamo.solver_step!(state)
    # THE assert that kills the original bug: with |T| > 0 and Ra > 0 the
    # radial buoyancy force must project onto the poloidal nonlinear term at
    # the very first step.
    nlpol = norm(parent(state.fields.velocity.nl_poloidal.data_real)) +
            norm(parent(state.fields.velocity.nl_poloidal.data_imag))
    @info "step-1 buoyancy projection" nlpol
    @test nlpol > 1e-12

    # and the flow must leave rest and keep growing while buoyancy dominates
    ke1 = _pm_ke(state)
    @test ke1 > 0.0
    for _ in 1:30
        GeoDynamo.solver_step!(state)
    end
    ke31 = _pm_ke(state)
    @info "kinetic energy" ke1 ke31
    @test ke31 > ke1
    @test all(isfinite, parent(state.fields.velocity.poloidal.data_real))
end

@testset "subcritical: imposed flow decays, no spurious growth" begin
    state = _pm_build(; Ra = 1e-6)
    GeoDynamo.initialize_solver_fields!(state)   # consume one-shot init BEFORE seeding
    # seed a small smooth poloidal flow (l=2,m=0), BC-compatible (vanishes at walls)
    vel = state.fields.velocity
    cfg = vel.poloidal.config
    dom = state.backend.outer_core_domain
    lm = findfirst(i -> cfg.l_values[i] == 2 && cfg.m_values[i] == 0, 1:cfg.nlm)
    slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
    ri = dom.r[1, 4]; ro = dom.r[dom.N, 4]
    for r_idx in 1:dom.N
        x = (dom.r[r_idx, 4] - ri) / (ro - ri)
        GeoDynamo.set_local_spectral_value!(parent(vel.poloidal.data_real), slot, r_idx,
            1e-4 * sinpi(x)^2)
    end
    ke0 = _pm_ke(state)
    @test ke0 > 0.0
    # Rotating-frame transient (non-normal) amplification peaks within ~200
    # steps at this Ek, then the flow decays viscously — assert boundedness
    # and eventual decay, not instantaneous monotonicity.
    ke_peak = ke0
    ke_end = ke0
    for k in 1:300
        GeoDynamo.solver_step!(state)
        ke_end = _pm_ke(state)
        ke_peak = max(ke_peak, ke_end)
    end
    @info "subcritical transient + decay" ke0 ke_peak ke_end
    @test ke_peak < 100 * ke0            # bounded: no instability
    @test ke_end < 0.7 * ke_peak         # decaying past the transient
    @test all(isfinite, parent(vel.poloidal.data_real))
end

@testset "stress-free walls: W-split runs, wall conditions hold" begin
    params = GeoDynamo.SolverParameters(
        architecture = :cpu, geometry = :shell,
        nr = 16, nr_inner = 4, lmax = 8, mmax = 8, nlat = 20, nlon = 40,
        Ra = 1e4, Ek = 1e-2, Pr = 1.0, Pm = 1.0, Sc = 1.0,
        timestep = 1e-4, start_time = 0.0, end_time = 1.0, stop_iteration = 100000,
        include_magnetic = false, include_composition = false,
        timestepper = GeoDynamo.CNAB2(), topography_enabled = false, stefan_enabled = false,
        velocity_bcs = GeoDynamo.BoundaryConditions(
            inner = GeoDynamo.StressFree(), outer = GeoDynamo.StressFree()),
    )
    state = GeoDynamo.initialize_simulation(Float64, params)
    for _ in 1:20
        GeoDynamo.solver_step!(state)
    end
    vel = state.fields.velocity
    P = parent(vel.poloidal.data_real)
    @test all(isfinite, P)
    @test _pm_ke(state) > 0.0    # buoyancy drives flow under stress-free too

    # wall conditions on a representative active mode: P(±) = 0 and the
    # stress-free residual P″ − (2/r)P′ ≈ 0 at both walls
    cfg = vel.poloidal.config
    dom = state.backend.outer_core_domain
    D1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    D2 = GeoDynamo.create_derivative_matrix(Float64, 2, dom)
    nr = dom.N
    checked = 0
    for lm in 1:cfg.nlm
        l = cfg.l_values[lm]
        l == 0 && continue
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        prof = [GeoDynamo.local_spectral_value(P, slot, r) for r in 1:nr]
        scale = maximum(abs, prof)
        scale < 1e-12 && continue
        dp = D1 * prof; d2p = D2 * prof
        @test abs(prof[1]) < 1e-8 * scale
        @test abs(prof[nr]) < 1e-8 * scale
        @test abs(d2p[1] - 2 / dom.r[1, 4] * dp[1]) < 1e-6 * maximum(abs, d2p)
        @test abs(d2p[nr] - 2 / dom.r[nr, 4] * dp[nr]) < 1e-6 * maximum(abs, d2p)
        checked += 1
        checked >= 3 && break
    end
    @test checked >= 3
end

@testset "ERK2 W-split: consistent with CNAB2, wall conditions hold" begin
    function run10(stepper)
        params = GeoDynamo.SolverParameters(
            architecture = :cpu, geometry = :shell,
            nr = 16, nr_inner = 4, lmax = 8, mmax = 8, nlat = 20, nlon = 40,
            Ra = 1e4, Ek = 1e-2, Pr = 1.0, Pm = 1.0, Sc = 1.0,
            timestep = 5e-5, start_time = 0.0, end_time = 1.0, stop_iteration = 100000,
            include_magnetic = false, include_composition = false,
            timestepper = stepper, topography_enabled = false, stefan_enabled = false)
        state = GeoDynamo.initialize_simulation(Float64, params)
        for _ in 1:10
            GeoDynamo.solver_step!(state)
        end
        return state
    end
    sC = run10(GeoDynamo.CNAB2())
    sE = run10(GeoDynamo.ERK2())
    # both schemes are 2nd order on the same equations: trajectories agree to
    # scheme-difference accuracy (not bit-exact)
    pC = parent(sC.fields.velocity.poloidal.data_real)
    pE = parent(sE.fields.velocity.poloidal.data_real)
    @test all(isfinite, pE)
    relΔ = norm(pE - pC) / max(norm(pC), eps())
    @info "ERK2 vs CNAB2 poloidal after 10 steps" relΔ
    @test relΔ < 0.05

    # ERK2 wall conditions (no-slip): P=0 and P′=0 on active modes
    cfg = sE.fields.velocity.poloidal.config
    dom = sE.backend.outer_core_domain
    D1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    nr = dom.N
    checked = 0
    for lm in 1:cfg.nlm
        l = cfg.l_values[lm]; l == 0 && continue
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        prof = [GeoDynamo.local_spectral_value(pE, slot, r) for r in 1:nr]
        scale = maximum(abs, prof); scale < 1e-12 && continue
        dp = D1 * prof
        @test abs(prof[1]) < 1e-8 * scale
        @test abs(prof[nr]) < 1e-8 * scale
        @test abs(dp[1]) < 1e-5 * maximum(abs, dp)
        @test abs(dp[nr]) < 1e-5 * maximum(abs, dp)
        checked += 1
        checked >= 3 && break
    end
    @test checked >= 3
end
