using Test
using GeoDynamo

const Topo = GeoDynamo.bcs.topography

struct DummyConfig
    lmax::Int
    mmax::Int
    nlm::Int
end

struct DummySpectralField{T}
    config::DummyConfig
    nlm::Int
    boundary_values::Matrix{T}
end

@testset "Stefan slope coupling sign" begin
    lmax = 1
    mmax = 1
    ri = 0.35
    eps = 0.02

    # Topography: single mode h_{1,0}
    topo = Topo.TopographyField(lmax, mmax, ri, Topo.INNER_BOUNDARY)
    coeffs = zeros(ComplexF64, topo.nlm)
    idx_h = Topo.lm_to_index(1, 0, lmax)
    h_amp = 0.3
    coeffs[idx_h] = h_amp
    Topo.set_topography_coefficients!(topo, coeffs)
    topo_data = Topo.TopographyData{Float64}(topo, nothing, nothing, eps, false)

    # Stefan state with distinct conductivities
    state = Topo.StefanState(lmax=lmax, ri=ri, k_ic=2.0, k_oc=1.0, rho=1.0, L=1.0)
    state.topography = topo
    state.heat_flux_ic .= 0.0
    state.heat_flux_oc .= 0.0

    # Temperature boundary values: only (l=1,m=0) mode is nonzero at ICB
    nlm = topo.nlm
    cfg = DummyConfig(lmax, mmax, nlm)
    bv_oc = zeros(Float64, 2, nlm)
    bv_ic = zeros(Float64, 2, nlm)
    theta_oc = 2.0
    theta_ic = 1.0
    bv_oc[1, idx_h] = theta_oc
    bv_ic[1, idx_h] = theta_ic
    temp_oc = DummySpectralField(cfg, nlm, bv_oc)
    temp_ic = DummySpectralField(cfg, nlm, bv_ic)

    # Gaunt cache with the single required entry populated analytically
    gaunt = Topo.GauntTensorCache(lmax, lmax)
    l = 1
    m = 0
    lp = 1
    mp = 0
    L = 1
    M = 0
    G = Topo.gaunt_on_the_fly(l, m, lp, mp, L, M; use_wigner=true)
    G_grad = Topo.gradient_gaunt_from_basic(l, lp, L, G)
    gaunt.G[(l, m, lp, mp, L, M)] = G
    gaunt.G_∇[(l, m, lp, mp, L, M)] = G_grad

    config = Topo.TopographyCouplingConfig(
        enabled=true,
        stefan_enabled=true,
        include_slope_terms=true,
        include_shift_terms=false,
        epsilon=eps
    )

    flux = Topo.compute_stefan_flux_with_topography(
        state, temp_ic, temp_oc, topo_data, gaunt, config
    )

    expected = eps * h_amp * G_grad / ri^2 * (state.k_oc * theta_oc - state.k_ic * theta_ic)
    idx_flux = Topo.lm_to_index(l, 0, lmax)
    @test flux[idx_flux] ≈ expected atol=1e-12 rtol=1e-12
end
