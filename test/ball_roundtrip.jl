using Test
using GeoDynamo
const Ball = GeoDynamo.GeoDynamoBall

@testset "Ball geometry regularity and roundtrip" begin
    # Small config for quick test
    lmax = 6; mmax = 6
    nlat = max(lmax + 2, 12)
    nlon = max(2lmax + 1, 24)
    nr   = 6

    cfg = GeoDynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = Ball.create_ball_radial_domain(nr)

    # Scalar: random physical -> analysis with regularity -> check inner r plane zero for l>0
    spec = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    phys = GeoDynamo.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)
    parent(phys.data) .= randn.(Float64)
    Ball.ball_physical_to_spectral!(phys, spec)

    sreal = parent(spec.data_real); simag = parent(spec.data_imag)
    lm_range = GeoDynamo.range_local(cfg.pencils.spec, 1)
    @test !isempty(lm_range)
    if size(sreal, 3) >= 1
        for (k, lm_idx) in enumerate(lm_range)
            l = cfg.l_values[lm_idx]
            if l > 0
                @test sreal[k, 1, 1] ≈ 0.0 atol=1e-12
                @test simag[k, 1, 1] ≈ 0.0 atol=1e-12
            end
        end
    end

    # Vector: random physical -> analysis with regularity -> check inner plane zero for l≥1
    tor = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    pol = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    vec = GeoDynamo.create_shtns_vector_field(Float64, cfg, dom, (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))
    parent(vec.r_component.data)      .= randn.(Float64)
    parent(vec.θ_component.data)      .= randn.(Float64)
    parent(vec.φ_component.data)      .= randn.(Float64)

    Ball.ball_vector_analysis!(vec, tor, pol)
    treal = parent(tor.data_real); timag = parent(tor.data_imag)
    preal = parent(pol.data_real); pimag = parent(pol.data_imag)
    if size(treal, 3) >= 1
        for (k, lm_idx) in enumerate(lm_range)
            l = cfg.l_values[lm_idx]
            if l >= 1
                @test treal[k, 1, 1] ≈ 0.0 atol=1e-12
                @test timag[k, 1, 1] ≈ 0.0 atol=1e-12
                @test preal[k, 1, 1] ≈ 0.0 atol=1e-12
                @test pimag[k, 1, 1] ≈ 0.0 atol=1e-12
            end
        end
    end
end
