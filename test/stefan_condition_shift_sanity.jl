using Test
using GeoDynamo
using MPI

const Topo = GeoDynamo.bcs.topography

struct ShiftDummyConfig
    lmax::Int
    mmax::Int
    nlm::Int
    __buffers::GeoDynamo.SHTnsBuffers
end

# Build a dummy config whose spectral-slot lookup is pre-populated for the
# simple (nlm, 1, nr) layout, so the buffer-backed `local_spectral_storage_slot`
# resolves without needing full pencil metadata.
function ShiftDummyConfig(lmax::Int, mmax::Int, nlm::Int)
    buffers = GeoDynamo.SHTnsBuffers()
    buffers.local_spectral_slot_lookup = [CartesianIndex(i, 1) for i in 1:nlm]
    return ShiftDummyConfig(lmax, mmax, nlm, buffers)
end

struct ShiftDummyPencil
    axes_local::NTuple{3, UnitRange{Int}}
end

struct ShiftDummySpectralField{T}
    config::ShiftDummyConfig
    nlm::Int
    data_real::Array{T, 3}
    data_imag::Array{T, 3}
    pencil::ShiftDummyPencil
end

struct ShiftDummyTemperatureField{T}
    spectral::ShiftDummySpectralField{T}
    ∂r::GeoDynamo.BandedMatrix{T}
    ∂²r::GeoDynamo.BandedMatrix{T}
    domain::GeoDynamo.RadialDomain
end

@testset "Stefan shift coupling sign" begin
    if !MPI.Initialized()
        MPI.Init()
    end

    lmax = 1
    mmax = 1
    ri = 0.35
    eps = 0.02

    # Topography: single mode h_{1,0}
    topo = Topo.TopographyField(lmax, mmax, ri, Topo.INNER_BOUNDARY)
    coeffs = zeros(ComplexF64, topo.nlm)
    idx_h = Topo.lm_to_index(1, 0, lmax)
    h_amp = 0.25
    coeffs[idx_h] = h_amp
    Topo.set_topography_coefficients!(topo, coeffs)
    topo_data = Topo.TopographyData{Float64}(topo, nothing, nothing, eps, false)

    # Stefan state with distinct conductivities
    state = Topo.StefanState(lmax=lmax, ri=ri, k_ic=2.0, k_oc=1.0, rho=1.0, L=1.0)
    state.topography = topo
    state.heat_flux_ic .= 0.0
    state.heat_flux_oc .= 0.0

    nlm = topo.nlm
    cfg = ShiftDummyConfig(lmax, mmax, nlm)
    nr = 2

    # Minimal radial domain (only N is used by the cache)
    r = zeros(Float64, nr, 7)
    dr_mats = [zeros(1, nr) for _ in 1:3]
    radial_lap = zeros(1, nr)
    weights = zeros(nr)
    domain = GeoDynamo.RadialDomain(nr, 1:nr, r, dr_mats, radial_lap, weights)

    # Identity d2, zero d1
    ∂r = GeoDynamo.BandedMatrix(zeros(1, nr), 0, nr)
    ∂²r = GeoDynamo.BandedMatrix(ones(1, nr), 0, nr)

    # Spectral data: constant across radius for (l=1,m=0)
    data_real_oc = zeros(Float64, nlm, 1, nr)
    data_imag_oc = zeros(Float64, nlm, 1, nr)
    data_real_ic = zeros(Float64, nlm, 1, nr)
    data_imag_ic = zeros(Float64, nlm, 1, nr)
    val_oc = 2.0
    val_ic = 1.0
    data_real_oc[idx_h, 1, :] .= val_oc
    data_real_ic[idx_h, 1, :] .= val_ic

    pencil = ShiftDummyPencil((1:nlm, 1:1, 1:nr))
    spec_oc = ShiftDummySpectralField(cfg, nlm, data_real_oc, data_imag_oc, pencil)
    spec_ic = ShiftDummySpectralField(cfg, nlm, data_real_ic, data_imag_ic, pencil)

    temp_oc = ShiftDummyTemperatureField(spec_oc, ∂r, ∂²r, domain)
    temp_ic = ShiftDummyTemperatureField(spec_ic, ∂r, ∂²r, domain)

    # Gaunt cache with required entry
    gaunt = Topo.GauntTensorCache(lmax, lmax)
    l = 1
    m = 0
    lp = 1
    mp = 0
    L = 1
    M = 0
    G = Topo.gaunt_on_the_fly(l, m, lp, mp, L, M; use_wigner=true)
    gaunt.G[(l, m, lp, mp, L, M)] = G

    config = Topo.TopographyCouplingConfig(
        enabled=true,
        stefan_enabled=true,
        include_slope_terms=false,
        include_shift_terms=true,
        epsilon=eps
    )

    flux = Topo.compute_stefan_flux_with_topography(
        state, temp_ic, temp_oc, topo_data, gaunt, config
    )

    expected = eps * h_amp * G * (state.k_ic * val_ic - state.k_oc * val_oc)
    idx_flux = Topo.lm_to_index(l, 0, lmax)
    @test flux[idx_flux] ≈ expected atol=1e-12 rtol=1e-12
end
