using Test
using MPI

# Numerical contract for the matrix-embedded velocity boundary conditions.
#
# The companion `velocity_boundary_static_checks.jl` only string-matches the
# source to confirm the BC code *exists*. This file builds the real production
# operators, solves a radial profile per spherical-harmonic degree, and asserts
# the solved field actually *satisfies* each condition:
#
#   Toroidal  no-slip      : T = 0
#   Toroidal  stress-free  : ∂T/∂r − T/r = 0
#   Poloidal  no-slip      : P = 0  (no-penetration, via influence matrix)
#                            ∂P/∂r = 0
#   Poloidal  stress-free  : P = 0  (no-penetration, via influence matrix)
#                            ∂²P/∂r² = 0
#
# Unlike the magnetic case (whose insulating BC annihilates a known potential
# field r^l / r^{-(l+1)}), the velocity conditions have no simple analytic
# matching profile, so we verify by solving and re-evaluating the embedded
# boundary stencil on the solution. The poloidal scalar carries two conditions
# per boundary: the derivative row is embedded in the implicit matrix and the
# no-penetration value (P = 0) is imposed afterward by the influence-matrix
# correction — both must hold simultaneously after correction.
@testset "Velocity boundary-condition numerical satisfaction" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping velocity BC numerical tests"
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

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    dom = GeoDynamo.create_radial_domain(nr)
    N = dom.N
    dt = 1.0e-3
    θ = 0.5
    l = 2                        # any 1 ≤ l ≤ lmax present in the config

    d1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    d2 = GeoDynamo.create_derivative_matrix(Float64, 2, dom)
    rinv_in = dom.r[1, 3]        # 1/r at inner boundary  (r columns: 2=1/r², 3=1/r, 4=r)
    rinv_out = dom.r[N, 3]       # 1/r at outer boundary

    # Smooth, asymmetric interior RHS with homogeneous (zero) BC rows.
    base_rhs = Float64[sin(2.3 * dom.r[i, 4]) + 0.4 * dom.r[i, 4] for i in 1:N]
    make_rhs() = (b = copy(base_rhs); b[1] = 0.0; b[N] = 0.0; b)

    bc_ns_ns = GeoDynamo.BoundaryConditions(inner = GeoDynamo.NoSlip(), outer = GeoDynamo.NoSlip())
    bc_ns_sf = GeoDynamo.BoundaryConditions(inner = GeoDynamo.NoSlip(), outer = GeoDynamo.StressFree())
    bc_sf_ns = GeoDynamo.BoundaryConditions(inner = GeoDynamo.StressFree(), outer = GeoDynamo.NoSlip())
    bc_sf_sf = GeoDynamo.BoundaryConditions(inner = GeoDynamo.StressFree(), outer = GeoDynamo.StressFree())

    solve_mode(mats, ll, rhs) = begin
        lu = mats.factorizations[mats.lookup[ll]]
        out = similar(rhs)
        GeoDynamo.solve_banded!(out, lu, copy(rhs))
        out
    end
    tor(code) = GeoDynamo.create_velocity_toroidal_matrices(cfg, dom, 1.0, dt;
        velocity_bc_code = code, mass_coeff = 1.0, theta = θ)
    pol(code) = GeoDynamo.create_velocity_poloidal_matrices(cfg, dom, 1.0, dt;
        velocity_bc_code = code, mass_coeff = 1.0, theta = θ)

    val_atol = 1.0e-9    # boundary value residuals are enforced ~exactly
    der_atol = 1.0e-6    # boundary derivative residuals amplify round-off by ~1/h

    @testset "toroidal no-slip ⇒ T = 0 at both boundaries" begin
        T = solve_mode(tor(1), l, make_rhs())
        @test isapprox(T[1], 0.0; atol = val_atol)
        @test isapprox(T[N], 0.0; atol = val_atol)
    end

    @testset "toroidal stress-free ⇒ ∂T/∂r − T/r = 0 at both boundaries" begin
        T = solve_mode(tor(4), l, make_rhs())
        dT = d1 * T
        @test isapprox(dT[1] - T[1] * rinv_in, 0.0; atol = der_atol)
        @test isapprox(dT[N] - T[N] * rinv_out, 0.0; atol = der_atol)
    end

    @testset "toroidal mixed (no-slip inner / stress-free outer)" begin
        T = solve_mode(tor(2), l, make_rhs())
        dT = d1 * T
        @test isapprox(T[1], 0.0; atol = val_atol)                        # no-slip inner
        @test isapprox(dT[N] - T[N] * rinv_out, 0.0; atol = der_atol)     # stress-free outer
    end

    @testset "poloidal no-slip ⇒ P = 0 (influence) and ∂P/∂r = 0" begin
        P = solve_mode(pol(1), l, make_rhs())
        dP = d1 * P
        @test isapprox(dP[1], 0.0; atol = der_atol)        # derivative BC from main solve
        @test isapprox(dP[N], 0.0; atol = der_atol)
        @test abs(P[1]) + abs(P[N]) > 1.0e-8             # no-penetration NOT yet imposed
        infl = GeoDynamo.create_velocity_poloidal_influence_matrices(
            Float64, cfg, dom, 1.0, dt, bc_ns_ns; theta = θ)
        GeoDynamo.apply_influence_matrix_correction!(P, infl[l])
        dP = d1 * P
        @test isapprox(P[1], 0.0; atol = val_atol)         # no-penetration now imposed
        @test isapprox(P[N], 0.0; atol = val_atol)
        @test isapprox(dP[1], 0.0; atol = der_atol)        # derivative BC preserved
        @test isapprox(dP[N], 0.0; atol = der_atol)
    end

    @testset "poloidal stress-free ⇒ P = 0 (influence) and ∂²P/∂r² = 0" begin
        P = solve_mode(pol(4), l, make_rhs())
        d2P = d2 * P
        @test isapprox(d2P[1], 0.0; atol = der_atol)
        @test isapprox(d2P[N], 0.0; atol = der_atol)
        infl = GeoDynamo.create_velocity_poloidal_influence_matrices(
            Float64, cfg, dom, 1.0, dt, bc_sf_sf; theta = θ)
        GeoDynamo.apply_influence_matrix_correction!(P, infl[l])
        d2P = d2 * P
        @test isapprox(P[1], 0.0; atol = val_atol)
        @test isapprox(P[N], 0.0; atol = val_atol)
        @test isapprox(d2P[1], 0.0; atol = der_atol)
        @test isapprox(d2P[N], 0.0; atol = der_atol)
    end

    @testset "production scaling (main diffusivity = Ek, influence = 1) keeps BCs exact" begin
        Ek = 1.0e-3
        mats = GeoDynamo.create_velocity_poloidal_matrices(cfg, dom, Ek, dt;
            velocity_bc_code = 4, mass_coeff = Ek, theta = θ)
        P = solve_mode(mats, l, make_rhs())
        infl = GeoDynamo.create_velocity_poloidal_influence_matrices(
            Float64, cfg, dom, 1.0, dt, bc_sf_sf; theta = θ)
        GeoDynamo.apply_influence_matrix_correction!(P, infl[l])
        d2P = d2 * P
        @test isapprox(P[1], 0.0; atol = val_atol)
        @test isapprox(P[N], 0.0; atol = val_atol)
        @test isapprox(d2P[1], 0.0; atol = 1.0e-5)
        @test isapprox(d2P[N], 0.0; atol = 1.0e-5)
    end

    @testset "rotating inner core: toroidal l=1 no-slip enforces T[inner] = Ω·r_inner" begin
        Ω = 0.7
        rhs = make_rhs()
        rhs[1] = Ω * dom.r[1, 4]     # prescribed inner value (no-slip identity row)
        rhs[N] = 0.0
        T = solve_mode(tor(1), 1, rhs)
        @test isapprox(T[1], Ω * dom.r[1, 4]; atol = val_atol)
        @test isapprox(T[N], 0.0; atol = val_atol)
    end

    @testset "BC code mapping matches BoundaryConditions types" begin
        @test GeoDynamo._velocity_bc_code(bc_ns_ns) == 1
        @test GeoDynamo._velocity_bc_code(bc_ns_sf) == 2
        @test GeoDynamo._velocity_bc_code(bc_sf_ns) == 3
        @test GeoDynamo._velocity_bc_code(bc_sf_sf) == 4
    end
end
