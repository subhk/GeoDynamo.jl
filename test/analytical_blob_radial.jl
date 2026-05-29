using Test
using MPI
using GeoDynamo
const G = GeoDynamo

# Blob presets are now radial shells: l=0 only (no angular structure), with the
# (0,0) physical profile a background plus a Gaussian bump, stored ×√(4π).

function __physical_from_mean_coeff(cfg, dom, coeff::Float64)
    spec = G.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    phys = G.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.phi)
    fill!(parent(spec.data_real), 0.0)
    fill!(parent(spec.data_imag), 0.0)
    m00 = G.get_mode_index(cfg, 0, 0)
    slot = G.local_spectral_storage_slot(cfg, m00)
    for r in 1:dom.N
        G.set_local_spectral_value!(parent(spec.data_real), slot, r, coeff)
    end
    G.shtnskit_spectral_to_physical!(spec, phys)
    return parent(phys.data)[1]
end

function __assert_no_lgt0(field)
    cfg = field.config
    sr = parent(field.spectral.data_real)
    si = parent(field.spectral.data_imag)
    for lm in G.local_spectral_mode_indices(cfg)
        lm <= cfg.nlm || continue
        cfg.l_values[lm] == 0 && continue
        s = G.local_spectral_storage_slot(cfg, lm)
        s === nothing && continue
        for r in 1:size(sr, 3)
            @test G.local_spectral_value(sr, s, r) == 0.0
            @test G.local_spectral_value(si, s, r) == 0.0
        end
    end
end

@testset "Analytical blob presets are radial shells (l=0 only)" begin
    MPI.Finalized() && (@warn "MPI finalized; skipping"; return)
    MPI.Initialized() || MPI.Init()

    lmax = 4; mmax = 4; nlat = 10; nlon = 16; nr = 8
    cfg = G.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = G.create_radial_domain(nr)
    m00 = G.get_mode_index(cfg, 0, 0)
    slot00 = G.local_spectral_storage_slot(cfg, m00)

    @testset "temp :hot_blob — pure l=0, Gaussian-bump physical profile" begin
        temp = G.create_shtns_temperature_field(Float64, cfg, dom)
        G.set_analytical_initial_conditions!(temp, :temperature, :hot_blob,
                                             amplitude=1.0, r_center=0.5, blob_width=0.2)
        __assert_no_lgt0(temp)
        c1 = G.local_spectral_value(parent(temp.spectral.data_real), slot00, 1)  # r_frac=0
        expected = 0.5 * (1.0 - 0.0) + 1.0 * exp(-0.5 * ((0.0 - 0.5) / 0.2)^2)
        @test isapprox(__physical_from_mean_coeff(cfg, dom, c1), expected; atol=1e-9)
    end

    @testset "comp :blob — pure l=0, Gaussian-bump physical profile" begin
        comp = G.create_shtns_composition_field(Float64, cfg, dom)
        G.set_analytical_initial_conditions!(comp, :composition, :blob,
                                             r_center=0.3, blob_width=0.2, blob_composition=0.8)
        __assert_no_lgt0(comp)
        slot = G.local_spectral_storage_slot(comp.config, m00)
        c1 = G.local_spectral_value(parent(comp.spectral.data_real), slot, 1)  # r_frac=0
        expected = 0.1 + (0.8 - 0.1) * exp(-0.5 * ((0.0 - 0.3) / 0.2)^2)
        @test isapprox(__physical_from_mean_coeff(cfg, dom, c1), expected; atol=1e-9)
    end
end
