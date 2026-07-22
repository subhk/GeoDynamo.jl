using Test
using LinearAlgebra
using MPI

# ================================================================================
# Analytic free-decay anchor for the SHELL scalar (temperature / composition) BCs
# ================================================================================
#
# test/ball_bessel_decay.jl anchors the BALL scalar path (center regularity inner +
# Dirichlet outer) at l=0,1. Nothing anchored the SHELL scalar rows, nor the
# Neumann codes at all — yet inner-Neumann (code 3) is the DEFAULT for both
# temperature and composition.
#
# Scalar diffusion d/dt Θ = κ ∇²Θ with Θ = R(r) Y_lm gives the radial eigenproblem
# R″ + (2/r)R′ − l(l+1)/r² R = −k² R, so R(r) = A j_l(kr) + B y_l(kr) and the
# decay rate is σ = κ k². The four scalar_bc_code endpoint pairs reduce to:
#
#   code 1 (DD):  j_l(k ri) y_l(k ro)   − y_l(k ri) j_l(k ro)   = 0
#   code 2 (DN):  j_l(k ri) y_l′(k ro)  − y_l(k ri) j_l′(k ro)  = 0
#   code 3 (ND):  j_l′(k ri) y_l(k ro)  − y_l′(k ri) j_l(k ro)  = 0
#   code 4 (NN):  j_l′(k ri) y_l′(k ro) − y_l′(k ri) j_l′(k ro) = 0
#
# where Dirichlet ⇒ f=0 and Neumann ⇒ f′=0 (radial derivative, so the k factor
# drops out of the row=0 condition), and ′ denotes d/d(argument). These pin BOTH
# endpoint rows of BOTH the inner/outer choice simultaneously — a wrong stencil,
# sign, or an ERK2 propagator that only lag-enforces a Neumann row lands well
# outside tolerance.
# ================================================================================

_sph_j0(x) = sin(x) / x
_sph_j1(x) = sin(x) / x^2 - cos(x) / x
_sph_y0(x) = -cos(x) / x
_sph_y1(x) = -cos(x) / x^2 - sin(x) / x

function _sj(l::Int, x::Float64)
    l == 0 && return _sph_j0(x)
    l == 1 && return _sph_j1(x)
    fm, f = _sph_j0(x), _sph_j1(x)
    for n in 1:(l - 1)
        fm, f = f, (2n + 1) / x * f - fm
    end
    return f
end

function _sy(l::Int, x::Float64)
    l == 0 && return _sph_y0(x)
    l == 1 && return _sph_y1(x)
    fm, f = _sph_y0(x), _sph_y1(x)
    for n in 1:(l - 1)
        fm, f = f, (2n + 1) / x * f - fm
    end
    return f
end

# d/dx f_l(x) = f_{l-1}(x) - (l+1)/x f_l(x);  l=0 special-case f_0′ = -f_1.
_djx(l::Int, x::Float64) = l == 0 ? -_sph_j1(x) : _sj(l - 1, x) - (l + 1) / x * _sj(l, x)
_dyx(l::Int, x::Float64) = l == 0 ? -_sph_y1(x) : _sy(l - 1, x) - (l + 1) / x * _sy(l, x)

# `true` marks a Dirichlet end (f=0); `false` a Neumann end (f′=0).
_inner_dirichlet(code) = code == 1 || code == 2
_outer_dirichlet(code) = code == 1 || code == 3

function _scalar_discriminant(code, k, l, ri, ro)
    fi_in, fi_out = _inner_dirichlet(code) ?
                    (_sj(l, k * ri), _sy(l, k * ri)) : (_djx(l, k * ri), _dyx(l, k * ri))
    fo_in, fo_out = _outer_dirichlet(code) ?
                    (_sj(l, k * ro), _sy(l, k * ro)) : (_djx(l, k * ro), _dyx(l, k * ro))
    return fi_in * fo_out - fi_out * fo_in
end

# Eigenprofile normalised by the OUTER row so it is an exact eigenvector.
function _scalar_profile(code, k, l, ro, rr)
    A, B = _outer_dirichlet(code) ?
           (_sy(l, k * ro), -_sj(l, k * ro)) : (_dyx(l, k * ro), -_djx(l, k * ro))
    return [A * _sj(l, k * r) + B * _sy(l, k * r) for r in rr]
end

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
    error("no sign change of the scalar free-decay discriminant in [$kmin, $kmax]")
end

const _CODE_NAME = Dict(1 => "DD", 2 => "DN", 3 => "ND", 4 => "NN")

@testset "scalar shell free decay (analytic anchor)" begin
    if !MPI.Initialized()
        MPI.Init()
    end

    nr = 64
    dt = 2e-4
    nsteps = 400
    ratio = 0.35
    kappa = 1.0
    cfg = GeoDynamo.create_shtnskit_config(lmax = 4, mmax = 4,
        nlat = 12, nlon = 24, nr = nr)
    dom = GeoDynamo.create_radial_domain(nr; radius_ratio = ratio)
    rr = dom.r[1:nr, 4]
    ri, ro = rr[1], rr[nr]
    Solver = GeoDynamo.Solver

    function cnab2_rate(mats, l, profile)
        idx = mats.lookup[l]
        A = mats.factorizations[idx]
        L = mats.linear_matrices[idx]
        x = copy(profile)
        rhs = similar(x)
        Lx = similar(x)
        out = similar(x)
        mid = nr ÷ 2
        nhalf = nsteps ÷ 2
        vh = 0.0
        for s in 1:nsteps
            mul!(Lx, L, x)
            @. rhs = (1 / dt) * x + 0.5 * Lx
            rhs[1] = 0.0
            rhs[nr] = 0.0
            GeoDynamo.solve_banded!(out, A, rhs)
            copyto!(x, out)
            s == nhalf && (vh = x[mid])
        end
        return log(vh / x[mid]) / ((nsteps - nhalf) * dt)
    end

    function erk2_rate(cache, spec, l, profile)
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

    @testset "code $code ($(_CODE_NAME[code]))" for code in 1:4
        mats = GeoDynamo.create_scalar_matrices(cfg, dom, kappa, dt; scalar_bc_code = code)
        spec = Solver.build_solver_erk2_scalar_bc(Float64, dom, code)
        cache = GeoDynamo.create_solver_erk2_scalar_cache(
            Float64, cfg, dom, kappa, dt, code)
        for l in 1:3
            k = _first_root(kk -> _scalar_discriminant(code, kk, l, ri, ro), 0.05, 12.0)
            profile = _scalar_profile(code, k, l, ro, rr)
            target = kappa * k^2
            # CNAB2 embeds the rows in the banded matrix; only finite-difference
            # truncation of the operator separates it from the analytic rate.
            @test isapprox(cnab2_rate(mats, l, profile), target; rtol = 1e-4)
            # ERK2 must match too. A Neumann row that is only enforced by the
            # trailing endpoint projection (not eliminated from the generator)
            # gives errors of tens of percent, so the tolerance stays tight.
            @test isapprox(erk2_rate(cache, spec, l, profile), target; rtol = 1e-3)
        end
    end

    @testset "ERK2 propagator preserves the Neumann constraint (code $code)" for code in 2:4
        spec = Solver.build_solver_erk2_scalar_bc(Float64, dom, code)
        cache = GeoDynamo.create_solver_erk2_scalar_cache(
            Float64, cfg, dom, kappa, dt, code)
        for l in 1:3
            idx = findfirst(==(l), cache.l_values)
            row_in = Solver.solver_erk2_constraint_row(Float64, spec.inner, 1, l, nr)
            row_out = Solver.solver_erk2_constraint_row(Float64, spec.outer, nr, l, nr)
            v = [sin(2.3 * i / nr) + 0.5 * cos(5.1 * i / nr) for i in 1:nr]
            for M in (cache.E_half[idx], cache.E_full[idx],
                cache.phi1_half[idx], cache.phi1_full[idx], cache.phi2_full[idx])
                y = M * v
                scale = max(maximum(abs, y), 1e-300)
                @test abs(dot(row_in, y)) < 1e-8 * scale * maximum(abs, row_in)
                @test abs(dot(row_out, y)) < 1e-8 * scale * maximum(abs, row_out)
            end
        end
    end
end
