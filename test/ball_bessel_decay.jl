using Test
using LinearAlgebra
using MPI

const Ball = GeoDynamo.GeoDynamoBall

# Spherical Bessel closed forms + first zeros (no SpecialFunctions dep).
sph_j0(x) = x == 0 ? 1.0 : sin(x) / x
sph_j1(x) = x == 0 ? 0.0 : sin(x) / x^2 - cos(x) / x
const ALPHA_J0 = Float64(pi)            # first zero of j0
const ALPHA_J1 = 4.493409457909064      # first zero of j1 (tbl value)

# CN-step a pure-diffusion radial profile through banded matrices `mats`
# (homogeneous BC rows) and return the measured decay rate from the second
# half of the run (first half discards the row-replacement transient).
function measured_decay_rate(mats, dom, l::Int, theta::Vector{Float64}; dt, nsteps)
    nr = dom.N
    idx = mats.lookup[l]
    A = mats.factorizations[idx]
    L = mats.linear_matrices[idx]
    rhs = similar(theta); Lf = similar(theta); out = similar(theta)
    inv_dt = 1 / dt
    mid = nr ÷ 2
    nhalf = nsteps ÷ 2
    vh = 0.0
    for s in 1:nsteps
        mul!(Lf, L, theta)
        @. rhs = inv_dt * theta + 0.5 * Lf
        rhs[1] = 0.0; rhs[nr] = 0.0          # homogeneous BC rows
        GeoDynamo.solve_banded!(out, A, rhs)
        copyto!(theta, out)
        s == nhalf && (vh = theta[mid])
    end
    return log(vh / theta[mid]) / ((nsteps - nhalf) * dt)
end

@testset "ball scalar Bessel decay (analytic anchor)" begin
    if !MPI.Initialized()
        MPI.Init()
    end

    nr = 48; dt = 2e-4; nsteps = 200
    cfg = GeoDynamo.create_shtnskit_config(lmax = 4, mmax = 4,
        nlat = 12, nlon = 24, nr = nr)
    dom = Ball.create_ball_radial_domain(nr)
    mats = GeoDynamo.create_scalar_matrices(cfg, dom, 1.0, dt;
        scalar_bc_code = 1, inner_regularity = true)
    rr = dom.r[1:nr, 4]

    # l=0: Θ = j0(πr), σ = π²  (inner row reduces to Θ′(r₁)=0 automatically
    # since β = l = 0; outer Dirichlet via scalar_bc_code=1).
    σ0 = measured_decay_rate(mats, dom, 0, [sph_j0(ALPHA_J0 * r) for r in rr];
        dt, nsteps)
    # Analytic target: π² ≈ 9.8696.  O(r₁²) Robin truncation at nr=48 gives
    # ~0.2% error; rtol=5e-3 is the acceptance criterion.
    @test isapprox(σ0, ALPHA_J0^2; rtol = 5e-3)

    # l=1: Θ = j1(α₁r), σ = α₁²  (inner row stamps Θ′(r₁) = Θ(r₁)/r₁)
    σ1 = measured_decay_rate(mats, dom, 1, [sph_j1(ALPHA_J1 * r) for r in rr];
        dt, nsteps)
    # Analytic target: α₁² ≈ 20.1907.  Same O(r₁²) Robin truncation error.
    @test isapprox(σ1, ALPHA_J1^2; rtol = 5e-3)
end
