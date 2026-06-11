using Test
using MPI
using LinearAlgebra
using GeoDynamo

MPI.Initialized() || MPI.Init()

# ===========================================================================
# Ball solver physics: the full solver step in :ball geometry must use the
# SAME verified nonlinear projection paths as the shell (Stage-4/4B double
# curl). Historically the ball routed through a legacy potential-style
# analysis that discarded the radial force component (Q): buoyancy never
# entered ball momentum and velocity from rest stayed exactly zero — the
# same dropped-Q defect the shell had before the double-curl fix.
# With the off-center radial grid (no r=0 node) every 1/r entry is finite,
# so the shell projection paths apply verbatim.
# ===========================================================================

function _ball_test_params(; Ra, timestepper = GeoDynamo.CNAB2(),
        include_magnetic = false)
    return GeoDynamo.SolverParameters(
        architecture = :cpu, geometry = :ball, radius_ratio = 0.0,
        nr = 16, lmax = 8, mmax = 8, nlat = 18, nlon = 36,
        Ra = Ra, Ek = 1e-2, Pr = 1.0, Pm = 1.0, Sc = 1.0,
        timestep = 1e-5, start_time = 0.0, end_time = 1.0,
        stop_iteration = 100000,
        include_magnetic = include_magnetic, include_composition = false,
        timestepper = timestepper,
        topography_enabled = false, stefan_enabled = false,
    )
end

# Inject a smooth temperature perturbation in the (l, m) spectral mode
# (same seeding idiom as poloidal_momentum_split.jl).
function _seed_temperature_mode!(state, l, m, amp)
    temp = state.fields.temperature
    cfg = temp.spectral.config
    dom = state.backend.outer_core_domain
    lm = findfirst(i -> cfg.l_values[i] == l && cfg.m_values[i] == m, 1:cfg.nlm)
    lm === nothing && error("mode (l=$l, m=$m) not in config")
    slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
    slot === nothing && return state
    ri = dom.r[1, 4]
    ro = dom.r[dom.N, 4]
    for r_idx in 1:dom.N
        x = (dom.r[r_idx, 4] - ri) / (ro - ri)
        GeoDynamo.set_local_spectral_value!(
            parent(temp.spectral.data_real), slot, r_idx, amp * sinpi(x)^2)
    end
    return state
end

@testset "ball CNAB2 stepping: finite + buoyancy alive" begin
    params = _ball_test_params(; Ra = 1e4)
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_solver_fields!(state)   # consume one-shot init BEFORE seeding
    _seed_temperature_mode!(state, 2, 2, 1e-3)
    for i in 1:10
        GeoDynamo.solver_step!(state)
    end
    vel = state.fields.velocity
    pol = parent(vel.poloidal.data_real)
    @test all(isfinite, pol)
    @test maximum(abs, pol) > 0          # buoyancy entered momentum in the ball
    @test all(isfinite, parent(vel.toroidal.data_real))
    @test all(isfinite, parent(state.fields.temperature.spectral.data_real))
    nl = parent(vel.nl_poloidal.data_real)
    @test maximum(abs, nl) > 1e-14       # N_W assembled (projection path live)
end

# ERK2 on the ball uses the same regularity rows (descriptors) and the mixed
# 2x2 influence recovery as CNAB2; both schemes are 2nd order on the same
# equations, so short trajectories must agree to scheme-difference accuracy.
@testset "ball ERK2 vs CNAB2 consistency" begin
    p_cn = _ball_test_params(; Ra = 1e4)
    p_rk = _ball_test_params(; Ra = 1e4, timestepper = GeoDynamo.ERK2())
    s_cn = GeoDynamo.initialize_simulation(Float64, p_cn)
    s_rk = GeoDynamo.initialize_simulation(Float64, p_rk)
    for s in (s_cn, s_rk)
        GeoDynamo.initialize_solver_fields!(s)
        _seed_temperature_mode!(s, 2, 2, 1e-3)
    end
    for i in 1:20
        GeoDynamo.solver_step!(s_cn)
        GeoDynamo.solver_step!(s_rk)
    end
    a = parent(s_cn.fields.velocity.poloidal.data_real)
    b = parent(s_rk.fields.velocity.poloidal.data_real)
    denom = max(maximum(abs, a), maximum(abs, b), 1e-30)
    relΔ = maximum(abs, a .- b) / denom
    @info "ball ERK2 vs CNAB2 poloidal after 20 steps" relΔ
    @test relΔ < 0.05
    @test all(isfinite, b)
end
