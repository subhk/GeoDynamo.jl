using Test
using MPI

# Numerical contract for the matrix-embedded temperature boundary conditions.
#
# The companion `temperature_boundary_static_checks.jl` only string-matches the
# source. This file builds the real production operators, solves a radial
# profile per spherical-harmonic degree, and asserts the solved field actually
# *satisfies* each condition (codes 1–4 = DD / DN / ND / NN):
#
#   Dirichlet (FixedTemperature) : T        = value   (identity row)
#   Neumann   (FixedFlux)         : ∂T/∂r    = value   (first-derivative row)
#
# Temperature shares the scalar operator/BC embedding with composition
# (src/bcs/scalar_bc.jl). Boundary derivatives are evaluated with the same
# one-sided stencil the solver embeds, so each test verifies the solve enforces
# exactly the discrete condition specified.
#
# Special case: with Neumann at BOTH boundaries (code 4) the l=0 mean mode has a
# null-space gauge freedom (temperature defined up to an additive constant), so
# the solver pins the inner row to Dirichlet for l=0 only. The mean-mode block
# below checks that pin is in place and that the outer flux is still honored.

function __tempbc_banded_row(bm, i::Int)
    N = bm.size
    bw = bm.bandwidth
    row = zeros(eltype(bm.data), N)
    for j in max(1, i - bw):min(N, i + bw)
        band_row = bw + 1 + i - j
        if 1 <= band_row <= 2 * bw + 1
            row[j] = bm.data[band_row, j]
        end
    end
    return row
end

@testset "Temperature boundary-condition numerical satisfaction" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping temperature BC numerical tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    lmax = 4
    mmax = 4
    nlat = max(lmax + 2, 10)
    nlon = max(2lmax + 1, 16)
    nr = 24

    cfg = GeoDynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = GeoDynamo.create_radial_domain(nr)
    N = dom.N
    dt = 1.0e-3
    θ = 0.5
    diffusivity = 1.0     # BC rows (identity / ∂r) are diffusivity-independent

    d1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)

    solve_mode(mats, ll, rhs) = begin
        lu = mats.factorizations[mats.lookup[ll]]
        out = similar(rhs)
        GeoDynamo.solve_banded!(out, lu, copy(rhs))
        out
    end
    mat(code) = GeoDynamo.create_temperature_matrices(cfg, dom, diffusivity, dt;
                    temperature_bc_code=code, theta=θ)

    # Smooth interior RHS; endpoints carry the prescribed boundary values.
    interior = Float64[0.3 * sin(1.7 * dom.r[i, 4]) for i in 1:N]
    rhs_with(inner, outer) = (b = copy(interior); b[1] = inner; b[N] = outer; b)

    val_atol = 1.0e-9    # Dirichlet value residuals enforced ~exactly
    der_atol = 1.0e-6    # Neumann derivative residuals amplify round-off by ~1/h

    @testset "DD (code 1) ⇒ T = value at both boundaries" begin
        T = solve_mode(mat(1), 2, rhs_with(0.8, -0.3))
        @test isapprox(T[1], 0.8;  atol=val_atol)
        @test isapprox(T[N], -0.3; atol=val_atol)
    end

    @testset "NN (code 4), l ≥ 1 ⇒ ∂T/∂r = value at both boundaries" begin
        T = solve_mode(mat(4), 2, rhs_with(0.5, -0.2))
        dT = d1 * T
        @test isapprox(dT[1], 0.5;  atol=der_atol)
        @test isapprox(dT[N], -0.2; atol=der_atol)
    end

    @testset "DN (code 2) ⇒ T(inner)=value, ∂T/∂r(outer)=value" begin
        T = solve_mode(mat(2), 2, rhs_with(0.8, -0.2))
        dT = d1 * T
        @test isapprox(T[1], 0.8;   atol=val_atol)     # Dirichlet inner
        @test isapprox(dT[N], -0.2; atol=der_atol)     # Neumann outer
    end

    @testset "ND (code 3) ⇒ ∂T/∂r(inner)=value, T(outer)=value" begin
        T = solve_mode(mat(3), 2, rhs_with(0.5, -0.3))
        dT = d1 * T
        @test isapprox(dT[1], 0.5;  atol=der_atol)     # Neumann inner
        @test isapprox(T[N], -0.3; atol=val_atol)      # Dirichlet outer
    end

    @testset "NN (code 4), l=0 mean mode ⇒ inner pinned to Dirichlet (gauge fix)" begin
        mats = mat(4)
        e1 = zeros(N); e1[1] = 1.0
        # Structural: l=0 inner row is the identity (pinned); l≥1 inner stays Neumann.
        @test __tempbc_banded_row(mats.system_matrices[mats.lookup[0]], 1) ≈ e1 atol=1e-14
        @test !(__tempbc_banded_row(mats.system_matrices[mats.lookup[2]], 1) ≈ e1)
        # Behavioral: the solve is well-posed (no singular pivot) and honors the
        # pinned inner value plus the prescribed outer flux.
        T = solve_mode(mats, 0, rhs_with(0.8, -0.2))
        dT = d1 * T
        @test isapprox(T[1], 0.8;   atol=val_atol)     # pinned reference value
        @test isapprox(dT[N], -0.2; atol=der_atol)     # outer flux still honored
    end

    @testset "thermal BC code mapping matches BoundaryConditions types" begin
        D = GeoDynamo.FixedTemperature(0.0)
        F = GeoDynamo.FixedFlux(0.0)
        @test GeoDynamo.__thermal_bc_code(GeoDynamo.BoundaryConditions(inner=D, outer=D)) == 1
        @test GeoDynamo.__thermal_bc_code(GeoDynamo.BoundaryConditions(inner=D, outer=F)) == 2
        @test GeoDynamo.__thermal_bc_code(GeoDynamo.BoundaryConditions(inner=F, outer=D)) == 3
        @test GeoDynamo.__thermal_bc_code(GeoDynamo.BoundaryConditions(inner=F, outer=F)) == 4
    end
end
