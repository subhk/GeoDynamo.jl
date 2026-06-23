using Test, MPI, LinearAlgebra
using GeoDynamo

MPI.Initialized() || MPI.Init()

# ===========================================================================
# Bug: the conducting-inner-core POLOIDAL admittance was built with the full
# scalar Laplacian ∇²_l = d²/dr² + (2/r)d/dr − l(l+1)/r², but the OUTER-core
# magnetic poloidal operator (create_magnetic_poloidal_matrices) uses
# D_pol = d²/dr² − l(l+1)/r²  (NO 2/r term). The ICB admittance must be built
# from the SAME operator as the outer core it is matched to, else α_pol is
# inconsistent. The toroidal admittance correctly keeps the full Laplacian.
#
# Probe: apply the stored linear operator L = η·(operator) to f(r) = r. On
# interior nodes the radial derivative matrices are exact on a linear function
# (D¹r = 1, D²r = 0), so:
#     D_pol · r = −l(l+1)/r           (no 2/r contribution)
#     ∇²_l  · r = 2/r − l(l+1)/r      (2/r contribution present)
# ===========================================================================

function _ic_domain(; N = 8, radius_ratio = 0.35)
    unit_ball = GeoDynamo.create_ball_radial_domain(N; radial_bandwidth = 4)
    icr = radius_ratio / (1.0 - radius_ratio)
    return GeoDynamo.scale_radial_domain(unit_ball, icr)
end

_rvec(dom, N) = Float64[dom.r[n, 4] for n in 1:N]

@testset "inner-core poloidal admittance uses D_pol (no 2/r term)" begin
    icdom = _ic_domain()
    N = icdom.N
    η = 1.0; dt = 1e-3; θ = 0.5; l = 2; λ = Float64(l * (l + 1))

    adm = GeoDynamo.create_inner_core_admittance(
        Float64, [l], icdom, η, dt; theta = θ, poloidal = true)
    L = adm.lin[adm.lookup[l]]
    r = _rvec(icdom, N)
    Lr = L * r
    for n in 2:(N - 1)
        @test isapprox(Lr[n], -η * λ / r[n]; rtol = 1e-7, atol = 1e-7)
    end
end

@testset "inner-core toroidal admittance keeps the full Laplacian (with 2/r)" begin
    icdom = _ic_domain()
    N = icdom.N
    η = 1.0; dt = 1e-3; θ = 0.5; l = 2; λ = Float64(l * (l + 1))

    adm = GeoDynamo.create_inner_core_admittance(Float64, [l], icdom, η, dt; theta = θ)
    L = adm.lin[adm.lookup[l]]
    r = _rvec(icdom, N)
    Lr = L * r
    for n in 2:(N - 1)
        @test isapprox(Lr[n], η * (2 / r[n] - λ / r[n]); rtol = 1e-7, atol = 1e-7)
    end
end

@testset "conducting build wires poloidal→D_pol, toroidal→∇²_l" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16)
    dom = GeoDynamo.create_radial_domain(16)
    icdom = _ic_domain()
    N = icdom.N; l = 2; λ = Float64(l * (l + 1)); η = 1.0

    res = GeoDynamo.build_magnetic_implicit_matrices_conducting(
        Float64, cfg, dom, icdom, 1e-3; theta = 0.5)
    r = _rvec(icdom, N)
    Lp = res.admittance.pol.lin[res.admittance.pol.lookup[l]] * r
    Lt = res.admittance.tor.lin[res.admittance.tor.lookup[l]] * r
    for n in 2:(N - 1)
        @test isapprox(Lp[n], -η * λ / r[n]; rtol = 1e-7, atol = 1e-7)             # poloidal = D_pol
        @test isapprox(Lt[n], η * (2 / r[n] - λ / r[n]); rtol = 1e-7, atol = 1e-7) # toroidal = ∇²_l
    end
end
