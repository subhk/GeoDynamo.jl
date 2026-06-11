using Test
using Random
const Ball = GeoDynamo.GeoDynamoBall

# Ball transforms are the SAME geometry-blind transforms the shell uses: the
# off-center radial grid has no r=0 node, so no centre-plane zeroing wrapper
# exists anymore (regularity lives in the implicit-matrix Robin rows).

@testset "Ball geometry transforms (geometry-blind)" begin

    # Small config for quick test
    lmax = 6;
    mmax = 6
    nlat = max(lmax + 2, 12)
    nlon = max(2lmax + 1, 24)
    nr = 6

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    dom = Ball.create_ball_radial_domain(nr)

    # Scalar: random physical -> analysis -> finite spectra, live content
    spec = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    phys = GeoDynamo.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)

    randn!(parent(phys.data))

    GeoDynamo.shtnskit_physical_to_spectral!(phys, spec)

    sreal = parent(spec.data_real);
    simag = parent(spec.data_imag)
    lm_range = GeoDynamo.local_spectral_mode_indices(cfg)

    @test !isempty(lm_range)
    @test all(isfinite, sreal)
    @test all(isfinite, simag)
    @test maximum(abs, sreal) > 0.0

    # Scalar roundtrip: spectral -> physical stays finite and nonzero
    phys_back = GeoDynamo.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)
    GeoDynamo.shtnskit_spectral_to_physical!(spec, phys_back)
    @test all(isfinite, parent(phys_back.data))
    @test maximum(abs, parent(phys_back.data)) > 0.0

    # Vector: random physical -> analysis -> finite toroidal/poloidal spectra
    tor = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    pol = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    vec = GeoDynamo.create_shtns_vector_field(Float64, cfg, dom, (
        cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))

    randn!(parent(vec.r_component.data))
    randn!(parent(vec.θ_component.data))
    randn!(parent(vec.φ_component.data))

    GeoDynamo.shtnskit_vector_analysis!(vec, tor, pol)
    for sf in (tor, pol)
        @test all(isfinite, parent(sf.data_real))
        @test all(isfinite, parent(sf.data_imag))
    end
    @test max(maximum(abs, parent(tor.data_real)),
        maximum(abs, parent(pol.data_real))) > 0.0
end
