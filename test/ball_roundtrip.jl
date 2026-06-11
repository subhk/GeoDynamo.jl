using Test
using Random
const Ball = GeoDynamo.GeoDynamoBall

@testset "Ball geometry regularity and roundtrip" begin

    # Small config for quick test
    lmax = 6;
    mmax = 6
    nlat = max(lmax + 2, 12)
    nlon = max(2lmax + 1, 24)
    nr = 6

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    dom = Ball.create_ball_radial_domain(nr)

    # Scalar: random physical -> analysis with regularity -> check inner r plane zero for l>0
    spec = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    phys = GeoDynamo.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)

    randn!(parent(phys.data))

    Ball.ball_physical_to_spectral!(phys, spec)

    sreal = parent(spec.data_real);
    simag = parent(spec.data_imag)
    lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)

    @test !isempty(lm_range)
    # Only check inner boundary regularity if this rank owns global r=1
    if 1 in r_range
        r_local_idx = 1 - first(r_range) + 1
        for lm_idx in lm_range
            l = cfg.l_values[lm_idx]
            if l > 0
                slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
                slot === nothing && continue
                @test sreal[slot[1], slot[2], r_local_idx] ≈ 0.0 atol=1e-12
                @test simag[slot[1], slot[2], r_local_idx] ≈ 0.0 atol=1e-12
            end
        end
    end

    # Vector: random physical -> analysis with regularity -> check inner plane zero for l≥1
    tor = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    pol = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    vec = GeoDynamo.create_shtns_vector_field(Float64, cfg, dom, (
        cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))

    randn!(parent(vec.r_component.data))
    randn!(parent(vec.θ_component.data))
    randn!(parent(vec.φ_component.data))

    Ball.ball_vector_analysis!(vec, tor, pol)
    treal = parent(tor.data_real);
    timag = parent(tor.data_imag)
    preal = parent(pol.data_real);
    pimag = parent(pol.data_imag)

    if 1 in r_range
        r_local_idx = 1 - first(r_range) + 1
        for lm_idx in lm_range
            l = cfg.l_values[lm_idx]
            if l >= 1
                slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
                slot === nothing && continue
                @test treal[slot[1], slot[2], r_local_idx] ≈ 0.0 atol=1e-12
                @test timag[slot[1], slot[2], r_local_idx] ≈ 0.0 atol=1e-12
                @test preal[slot[1], slot[2], r_local_idx] ≈ 0.0 atol=1e-12
                @test pimag[slot[1], slot[2], r_local_idx] ≈ 0.0 atol=1e-12
            end
        end
    end
end
