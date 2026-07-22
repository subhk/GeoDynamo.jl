using Test
using LinearAlgebra
using MPI

# ================================================================================
# Velocity ERK2 boundary-condition behaviour at PRODUCTION parameters
# ================================================================================
#
# The velocity TOROIDAL ERK2 cache eliminates its endpoint constraints from the
# generator (create_solver_erk2_cache), so its wall is embedded in the propagated
# operator and it is unconditionally stable — including the stress-free l = 1
# rigid-rotation marginal mode, via the homogeneous (forcing-dropped) path. The
# first testset pins that at ν = 1, where the pre-fix generic cache blew up.
#
# The velocity POLOIDAL ERK2 path uses a different mechanism — the influence-
# matrix W-split recovery, whose V-propagation runs on natural (un-embedded) rows
# at diffusivity 1. It is correct and stable at the small dt/h² of production runs
# but retains a dt/h² stability ceiling (documented on the builder). The second
# testset pins ERK2/CNAB2 agreement over many steps from a natural IC at
# production dt, so the regime real simulations use cannot silently regress.
# ================================================================================

_sph_j0(x) = sin(x) / x
_sph_j1(x) = sin(x) / x^2 - cos(x) / x
_sph_y0(x) = -cos(x) / x
_sph_y1(x) = -cos(x) / x^2 - sin(x) / x
function _sj(l::Int, x::Float64)
    l == 0 && return _sph_j0(x); l == 1 && return _sph_j1(x)
    fm, f = _sph_j0(x), _sph_j1(x)
    for n in 1:(l - 1); fm, f = f, (2n + 1) / x * f - fm; end
    return f
end
function _sy(l::Int, x::Float64)
    l == 0 && return _sph_y0(x); l == 1 && return _sph_y1(x)
    fm, f = _sph_y0(x), _sph_y1(x)
    for n in 1:(l - 1); fm, f = f, (2n + 1) / x * f - fm; end
    return f
end
# stress-free toroidal row (∂ᵣ − 1/r)t = 0 in argument space: (l−1)fₗ − x f_{l+1}
_sf_j(l::Int, x::Float64) = (l - 1) * _sj(l, x) - x * _sj(l + 1, x)
_sf_y(l::Int, x::Float64) = (l - 1) * _sy(l, x) - x * _sy(l + 1, x)

_vt_disc(code, k, l, ri, ro) = code == 1 ?
    (_sj(l, k * ri) * _sy(l, k * ro) - _sy(l, k * ri) * _sj(l, k * ro)) :
    (_sf_j(l, k * ri) * _sf_y(l, k * ro) - _sf_y(l, k * ri) * _sf_j(l, k * ro))
function _vt_profile(code, k, l, ro, rr)
    A, B = code == 1 ? (_sy(l, k * ro), -_sj(l, k * ro)) :
           (_sf_y(l, k * ro), -_sf_j(l, k * ro))
    return [A * _sj(l, k * r) + B * _sy(l, k * r) for r in rr]
end
function _first_root(f, kmin, kmax; nscan = 30000)
    ks = range(kmin, kmax; length = nscan); prev = f(ks[1])
    for i in 2:nscan
        cur = f(ks[i])
        if prev * cur < 0
            a, b = ks[i - 1], ks[i]
            for _ in 1:100; m = 0.5 * (a + b); (f(a) * f(m) <= 0) ? (b = m) : (a = m); end
            return 0.5 * (a + b)
        end
        prev = cur
    end
    error("no root in [$kmin, $kmax]")
end

@testset "velocity toroidal ERK2 free decay (unconditional)" begin
    # The toroidal cache eliminates its endpoint constraints (homogeneous walls
    # drop the forcing term), so the propagator embeds the wall and is stable at
    # ANY diffusive step size — not just the small ν·dt of production. ν = 1 is a
    # deliberately hard stress: before the fix the un-embedded generator's
    # projected step had spectral radius up to 84 (no-slip) and the stress-free
    # l = 1 rigid-rotation mode blew up to ~1e14. It must now reproduce the
    # analytic shell free-decay rate exactly, l = 1 included.
    if !MPI.Initialized(); MPI.Init(); end
    nr = 48
    dt = 2e-4
    nsteps = 400
    ratio = 0.35
    nu = 1.0
    cfg = GeoDynamo.create_shtnskit_config(lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = nr)
    dom = GeoDynamo.create_radial_domain(nr; radius_ratio = ratio)
    rr = dom.r[1:nr, 4]; ri, ro = rr[1], rr[nr]
    Solver = GeoDynamo.Solver

    function erk2_rate(cache, spec, l, profile)
        idx = findfirst(==(l), cache.l_values); E = cache.E_full[idx]
        x = copy(profile); mid = nr ÷ 2; nhalf = nsteps ÷ 2; vh = 0.0
        for s in 1:nsteps
            x = E * x
            Solver.solver_enforce_erk2_bc!(x, spec.inner, 1, l, nr)
            Solver.solver_enforce_erk2_bc!(x, spec.outer, nr, l, nr)
            s == nhalf && (vh = x[mid])
        end
        return log(vh / x[mid]) / ((nsteps - nhalf) * dt)
    end

    @testset "$name (code $code)" for (code, name) in ((1, "no-slip"), (4, "stress-free"))
        spec = Solver.build_solver_erk2_velocity_tor_bc(Float64, dom, code; config = cfg, rot_omega = 0.0)
        cache = Solver.create_solver_erk2_cache(Float64, cfg, dom, nu, dt; bc_spec = spec)
        for l in 1:3
            k = _first_root(kk -> _vt_disc(code, kk, l, ri, ro), 0.3, 12.0)
            rate = erk2_rate(cache, spec, l, _vt_profile(code, k, l, ro, rr))
            @test isapprox(rate, nu * k^2; rtol = 1e-3)
        end
    end
end

@testset "velocity ERK2 vs CNAB2 stability from a natural IC (production dt)" begin
    if !MPI.Initialized(); MPI.Init(); end

    function run_steps(ts, nsteps)
        p = GeoDynamo.SolverParameters(
            architecture = :cpu, geometry = :shell, nr = 32, lmax = 4, mmax = 4,
            nlat = 12, nlon = 24, include_magnetic = false, include_composition = false,
            Ra = 1e3, Ek = 1e-2, Pr = 1.0, Pm = 1.0,
            timestep = 1e-5, timestepper = ts, stop_iteration = 10^9, end_time = 1e9)
        st = GeoDynamo.initialize_solver_state(Float64; params = p)
        GeoDynamo.initialize_fields!(st)
        dom = st.backend.outer_core_domain
        ke = Float64[]
        for _ in 1:nsteps
            GeoDynamo.solver_step!(st)
            push!(ke, GeoDynamo.compute_kinetic_energy(st.fields.velocity, dom))
        end
        return ke
    end

    nsteps = 60
    ke_cnab2 = run_steps(GeoDynamo.CNAB2(), nsteps)
    ke_erk2 = run_steps(GeoDynamo.ExponentialRungeKutta2(), nsteps)

    # ERK2 must stay finite and track CNAB2 — no boundary-driven blow-up at
    # production dt. (Both grow at Ra=1e3; the point is that ERK2 does not run
    # away relative to the matrix-embedded reference.)
    @test all(isfinite, ke_erk2)
    @test all(isfinite, ke_cnab2)
    @test ke_erk2[end] > 0
    @test isapprox(ke_erk2[end], ke_cnab2[end]; rtol = 0.1)
end
