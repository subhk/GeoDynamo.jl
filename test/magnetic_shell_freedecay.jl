using Test
using LinearAlgebra
using MPI

# ================================================================================
# Analytic free-decay anchor for the SHELL magnetic boundary conditions
# ================================================================================
#
# test/ball_bessel_decay.jl anchors the BALL geometry (center-regularity inner
# row + insulating outer row) on the classic full-sphere dipole rate sigma = pi^2.
# It does NOT transfer to the shell: the ball path replaces the inner row with the
# regularity row, so the shell insulating inner row had no analytic check at all
# (src/bcs/magnetic_bc.jl notes "no single-row analytic discriminator exists for
# the shell inner row alone").
#
# A two-row discriminator does exist. Under the code's B_r = l(l+1)P/r^2
# convention the poloidal diffusion operator is D_pol = d2/dr2 - l(l+1)/r^2, so
# D_pol P = -k^2 P has the general solution
#
#     P(r) = r * [A j_l(kr) + B y_l(kr)]
#
# and the insulating rows reduce, via x f_l'(x) = l f_l(x) - x f_{l+1}(x) and
# x f_l'(x) + (l+1) f_l(x) = x f_{l-1}(x), to
#
#     inner  (d/dr - (l+1)/r) P = 0  ->  A j_{l+1}(k ri) + B y_{l+1}(k ri) = 0
#     outer  (d/dr + l/r)     P = 0  ->  A j_{l-1}(k ro) + B y_{l-1}(k ro) = 0
#
# giving the eigencondition
#
#     j_{l+1}(k ri) y_{l-1}(k ro) - y_{l+1}(k ri) j_{l-1}(k ro) = 0,  sigma = eta k^2.
#
# Sanity: as ri -> 0, y_{l+1} -> -inf dominates and the condition collapses to
# j_{l-1}(k ro) = 0, i.e. the full-sphere result the ball test already pins
# (l=1: k ro = pi, sigma = pi^2).
#
# The toroidal scalar obeys the full radial Laplacian with T = 0 at both walls,
# so T = A j_l(kr) + B y_l(kr) and
#
#     j_l(k ri) y_l(k ro) - y_l(k ri) j_l(k ro) = 0.
#
# These pin BOTH shell rows of BOTH components simultaneously: a wrong sign or a
# wrong (l+1)-vs-l coefficient moves the eigenvalue well outside tolerance.
# ================================================================================

# Spherical Bessel functions of both kinds (no SpecialFunctions dependency).
# Upward recurrence f_{l+1} = (2l+1)/x f_l - f_{l-1} is stable for y_l at all x
# and for j_l while x >~ l, which holds for every (k, l) exercised here.
_sph_j0(x) = sin(x) / x
_sph_j1(x) = sin(x) / x^2 - cos(x) / x
_sph_y0(x) = -cos(x) / x
_sph_y1(x) = -cos(x) / x^2 - sin(x) / x

function _sph_j(l::Int, x::Float64)
    l == 0 && return _sph_j0(x)
    l == 1 && return _sph_j1(x)
    fm, f = _sph_j0(x), _sph_j1(x)
    for n in 1:(l - 1)
        fm, f = f, (2n + 1) / x * f - fm
    end
    return f
end

function _sph_y(l::Int, x::Float64)
    l == 0 && return _sph_y0(x)
    l == 1 && return _sph_y1(x)
    fm, f = _sph_y0(x), _sph_y1(x)
    for n in 1:(l - 1)
        fm, f = f, (2n + 1) / x * f - fm
    end
    return f
end

_pol_discriminant(k, l, ri, ro) =
    _sph_j(l + 1, k * ri) * _sph_y(l - 1, k * ro) -
    _sph_y(l + 1, k * ri) * _sph_j(l - 1, k * ro)

_tor_discriminant(k, l, ri, ro) =
    _sph_j(l, k * ri) * _sph_y(l, k * ro) - _sph_y(l, k * ri) * _sph_j(l, k * ro)

# Smallest positive root => slowest-decaying (fundamental) free-decay mode.
function _first_root(f, kmin, kmax; nscan = 20000)
    ks = range(kmin, kmax; length = nscan)
    prev = f(ks[1])
    for i in 2:nscan
        cur = f(ks[i])
        if prev * cur < 0
            a, b = ks[i - 1], ks[i]
            for _ in 1:100
                m = 0.5 * (a + b)
                (f(a) * f(m) <= 0) ? (b = m) : (a = m)
            end
            return 0.5 * (a + b)
        end
        prev = cur
    end
    error("no sign change of the free-decay discriminant in [$kmin, $kmax]")
end

# Fundamental eigenprofile normalised by the outer row (A, B chosen so the outer
# condition holds identically); it is an eigenvector of the continuous problem,
# so any mismatch in the measured rate is a property of the discrete BC rows.
function _pol_profile(k, l, ro, rr)
    A = _sph_y(l - 1, k * ro)
    B = -_sph_j(l - 1, k * ro)
    return [r * (A * _sph_j(l, k * r) + B * _sph_y(l, k * r)) for r in rr]
end

function _tor_profile(k, l, ro, rr)
    A = _sph_y(l, k * ro)
    B = -_sph_j(l, k * ro)
    return [A * _sph_j(l, k * r) + B * _sph_y(l, k * r) for r in rr]
end

# Crank-Nicolson a pure-diffusion radial profile through the BC-embedded banded
# matrices and read the decay rate off the second half (first half absorbs the
# transient from the boundary-row replacement).
function _cnab2_decay_rate(mats, l::Int, profile::Vector{Float64}; dt, nsteps)
    nr = length(profile)
    idx = mats.lookup[l]
    A = mats.factorizations[idx]
    L = mats.linear_matrices[idx]
    x = copy(profile)
    rhs = similar(x)
    Lx = similar(x)
    out = similar(x)
    inv_dt = 1 / dt
    mid = nr ÷ 2
    nhalf = nsteps ÷ 2
    vh = 0.0
    for s in 1:nsteps
        mul!(Lx, L, x)
        @. rhs = inv_dt * x + 0.5 * Lx
        rhs[1] = 0.0
        rhs[nr] = 0.0
        GeoDynamo.solve_banded!(out, A, rhs)
        copyto!(x, out)
        s == nhalf && (vh = x[mid])
    end
    return log(vh / x[mid]) / ((nsteps - nhalf) * dt)
end

@testset "magnetic shell free decay (analytic anchor)" begin
    if !MPI.Initialized()
        MPI.Init()
    end

    nr = 64
    dt = 2e-4
    nsteps = 400
    ratio = 0.35
    cfg = GeoDynamo.create_shtnskit_config(lmax = 4, mmax = 4,
        nlat = 12, nlon = 24, nr = nr)
    dom = GeoDynamo.create_radial_domain(nr; radius_ratio = ratio)
    rr = dom.r[1:nr, 4]
    ri, ro = rr[1], rr[nr]

    pol = GeoDynamo.create_magnetic_poloidal_matrices(cfg, dom, 1.0, dt)
    tor = GeoDynamo.create_magnetic_toroidal_matrices(cfg, dom, 1.0, dt)

    @testset "poloidal insulating rows, l=$l" for l in 1:3
        k = _first_root(kk -> _pol_discriminant(kk, l, ri, ro), 0.05, 12.0)
        rate = _cnab2_decay_rate(pol, l, _pol_profile(k, l, ro, rr); dt, nsteps)
        # Both rows are exact on the discrete grid, so the only error is the
        # finite-difference truncation of D_pol; 1e-4 is ~100x the observed gap.
        @test isapprox(rate, k^2; rtol = 1e-4)
    end

    @testset "toroidal insulating rows, l=$l" for l in 1:3
        k = _first_root(kk -> _tor_discriminant(kk, l, ri, ro), 0.05, 12.0)
        rate = _cnab2_decay_rate(tor, l, _tor_profile(k, l, ro, rr); dt, nsteps)
        @test isapprox(rate, k^2; rtol = 1e-4)
    end

    @testset "insulating rows annihilate the vacuum continuations" begin
        bw = GeoDynamo.radial_bandwidth(dom)
        for l in 1:3
            S = pol.system_matrices[pol.lookup[l]]
            # Interior vacuum P ~ r^{l+1} must be killed by the inner row, and
            # exterior vacuum P ~ r^{-l} by the outer row.
            inner_profile = [r^(l + 1) for r in rr]
            outer_profile = [r^(-l) for r in rr]
            row1 = sum(S.data[bw + 1 + 1 - j, j] * inner_profile[j] for j in 1:(1 + bw))
            rowN = sum(S.data[bw + 1 + nr - j, j] * outer_profile[j] for j in (nr - bw):nr)
            @test abs(row1) < 1e-10 * maximum(abs, inner_profile)
            @test abs(rowN) < 1e-9 * maximum(abs, outer_profile)
        end
    end

    @testset "no spurious growing modes" begin
        # BC rows are stamped into the system matrix, so an incorrect row can
        # inject a mode with Re(sigma) < 0. Recover the full discrete spectrum
        # from the backward-Euler step map M = (1/dt) X^{-1} P and check signs.
        dt_be = 1e-3
        polbe = GeoDynamo.create_magnetic_poloidal_matrices(cfg, dom, 1.0, dt_be;
            theta = 1.0)
        torbe = GeoDynamo.create_magnetic_toroidal_matrices(cfg, dom, 1.0, dt_be;
            theta = 1.0)
        for (mats, disc) in ((polbe, _pol_discriminant), (torbe, _tor_discriminant))
            for l in 1:3
                A = mats.factorizations[mats.lookup[l]]
                M = zeros(nr, nr)
                e = zeros(nr)
                out = zeros(nr)
                for j in 2:(nr - 1)
                    fill!(e, 0.0)
                    e[j] = 1 / dt_be
                    GeoDynamo.solve_banded!(out, A, e)
                    M[:, j] .= out
                end
                mu = eigvals(M)
                sigmas = sort([(1 / m - 1) / dt_be for m in mu if abs(m) > 1e-13])
                @test all(>(0), sigmas)
                # Slowest discrete mode == fundamental analytic root.
                k = _first_root(kk -> disc(kk, l, ri, ro), 0.05, 12.0)
                @test isapprox(sigmas[1], k^2; rtol = 1e-6)
            end
        end
    end
end

@testset "magnetic shell free decay - ERK2 propagators" begin
    # The ERK2 caches stamp their own boundary rows (src/timestep/erk2/cache.jl),
    # historically pinned only by source-text matching. Run the same analytic
    # anchor through the exponential propagators actually used by the staged
    # integrator: with no nonlinear term, one ERK2 step is exactly
    # E_full * u followed by the endpoint projection.
    nr = 64
    dt = 2e-4
    nsteps = 400
    ratio = 0.35
    cfg = GeoDynamo.create_shtnskit_config(lmax = 4, mmax = 4,
        nlat = 12, nlon = 24, nr = nr)
    dom = GeoDynamo.create_radial_domain(nr; radius_ratio = ratio)
    rr = dom.r[1:nr, 4]
    ri, ro = rr[1], rr[nr]

    Solver = GeoDynamo.Solver
    pol_spec = Solver.build_solver_erk2_magnetic_pol_bc(Float64, dom)
    tor_spec = Solver.build_solver_erk2_magnetic_tor_bc(Float64, dom)
    pol_cache = GeoDynamo.create_solver_erk2_magnetic_poloidal_cache(
        Float64, cfg, dom, 1.0, dt; bc_spec = pol_spec)
    tor_cache = GeoDynamo.create_solver_erk2_magnetic_toroidal_cache(
        Float64, cfg, dom, 1.0, dt; bc_spec = tor_spec)

    function erk2_decay_rate(cache, spec, l, profile)
        idx = findfirst(==(l), cache.l_values)
        E = cache.E_full[idx]
        x = copy(profile)
        mid = nr ÷ 2
        nhalf = nsteps ÷ 2
        vh = 0.0
        for s in 1:nsteps
            x = E * x
            Solver.solver_enforce_erk2_bc!(x, spec.inner, 1, l, nr)
            Solver.solver_enforce_erk2_bc!(x, spec.outer, nr, l, nr)
            s == nhalf && (vh = x[mid])
        end
        return log(vh / x[mid]) / ((nsteps - nhalf) * dt)
    end

    @testset "poloidal, l=$l" for l in 1:3
        k = _first_root(kk -> _pol_discriminant(kk, l, ri, ro), 0.05, 12.0)
        rate = erk2_decay_rate(pol_cache, pol_spec, l, _pol_profile(k, l, ro, rr))
        # Robin rows: only the boundary-DOF elimination makes this exact. A
        # lagged endpoint projection gives errors of tens of percent that GROW
        # with radial resolution, so the tolerance must stay tight.
        @test isapprox(rate, k^2; rtol = 1e-3)
    end

    @testset "toroidal, l=$l" for l in 1:3
        k = _first_root(kk -> _tor_discriminant(kk, l, ri, ro), 0.05, 12.0)
        rate = erk2_decay_rate(tor_cache, tor_spec, l, _tor_profile(k, l, ro, rr))
        @test isapprox(rate, k^2; rtol = 1e-3)
    end

    @testset "poloidal BC residual stays at round-off, l=$l" for l in 1:3
        k = _first_root(kk -> _pol_discriminant(kk, l, ri, ro), 0.05, 12.0)
        x = _pol_profile(k, l, ro, rr)
        idx = findfirst(==(l), pol_cache.l_values)
        E = pol_cache.E_full[idx]
        for _ in 1:50
            x = E * x
        end
        d1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
        bw = GeoDynamo.radial_bandwidth(dom)
        scale = maximum(abs, x)
        res_in = sum(d1.data[bw + 1 + 1 - j, j] * x[j] for j in 1:(1 + bw)) -
                 (l + 1) / ri * x[1]
        res_out = sum(d1.data[bw + 1 + nr - j, j] * x[j] for j in (nr - bw):nr) +
                  l / ro * x[nr]
        # The propagator itself must preserve the constraint, without help from
        # the endpoint projection.
        @test abs(res_in) < 1e-8 * scale
        @test abs(res_out) < 1e-8 * scale
    end
end
