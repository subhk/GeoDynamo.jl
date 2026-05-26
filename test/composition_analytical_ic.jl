using Test
using MPI
using GeoDynamo
const G = GeoDynamo

# Regression: analytical composition presets must store their (0,0) physical mean
# as value·√(4π) (orthonormal SH), like the temperature presets and the solver's
# background composition IC.

function _physical_from_mean_coeff(cfg, dom, coeff::Float64)
    spec = G.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    phys = G.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.phi)
    fill!(parent(spec.data_real), 0.0)
    fill!(parent(spec.data_imag), 0.0)
    m00 = G.get_mode_index(cfg, 0, 0)
    slot = G.local_spectral_storage_slot(cfg, m00)
    @assert slot !== nothing
    for r in 1:dom.N
        G.set_local_spectral_value!(parent(spec.data_real), slot, r, coeff)
    end
    G.shtnskit_spectral_to_physical!(spec, phys)
    return parent(phys.data)[1]
end

@testset "Analytical composition IC √(4π) normalization" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping"
        return
    end
    MPI.Initialized() || MPI.Init()

    lmax = 4; mmax = 4
    nlat = max(lmax + 2, 10); nlon = max(2lmax + 1, 16); nr = 8
    cfg = G.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = G.create_radial_domain(nr)
    m00 = G.get_mode_index(cfg, 0, 0)
    slot = G.local_spectral_storage_slot(cfg, m00)

    @testset ":stratified (0,0) reconstructs to physical bottom composition" begin
        comp = G.create_shtns_composition_field(Float64, cfg, dom)
        G.set_analytical_initial_conditions!(comp, :composition, :stratified,
                                             bottom_composition=0.3, top_composition=0.1)
        c_inner = G.local_spectral_value(parent(comp.spectral.data_real), slot, 1)  # r_frac=0 ⇒ 0.3
        @test isapprox(_physical_from_mean_coeff(cfg, dom, c_inner), 0.3; atol=1e-9)
    end

    @testset ":blob (0,0) background reconstructs to physical 0.1" begin
        comp = G.create_shtns_composition_field(Float64, cfg, dom)
        G.set_analytical_initial_conditions!(comp, :composition, :blob,
                                             r_center=0.3, blob_width=0.2, blob_composition=0.8)
        # local_r=1 ⇒ r_frac=0, |0 - 0.3| = 0.3 > 0.2 ⇒ outside blob ⇒ background 0.1
        c_bg = G.local_spectral_value(parent(comp.spectral.data_real), slot, 1)
        @test isapprox(_physical_from_mean_coeff(cfg, dom, c_bg), 0.1; atol=1e-9)
    end
end
