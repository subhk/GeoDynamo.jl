using Test
using MPI
using GeoDynamo
const G = GeoDynamo

# (1) Nusselt number must be 1 for a pure conductive profile. compute_surface_flux
#     integrates ∂T/∂r over solid angle (∮·dΩ); the heat current through a sphere
#     of radius r needs the r² area factor. (2) Analytical :conductive IC must use
#     the orthonormal √(4π) normalization for its (0,0) physical mean, like the
#     solver's default conductive IC.

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

@testset "Nusselt and analytical IC normalization" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping"
        return
    end
    MPI.Initialized() || MPI.Init()

    lmax = 4; mmax = 4
    nlat = max(lmax + 2, 10); nlon = max(2lmax + 1, 16); nr = 16
    cfg = G.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = G.create_radial_domain(nr)

    @testset "Nu = 1 for a pure conductive radial gradient" begin
        temp = G.create_shtns_temperature_field(Float64, cfg, dom)
        ri = dom.r[1, 4]; ro = dom.r[nr, 4]
        # Conductive T(r) = ri*ro/(ro-ri)*(1/r - 1/ro) ⇒ dT/dr = -ri*ro/(ro-ri)/r².
        gr = parent(temp.gradient.r_component.data)
        θr, φr, rr = G.range_local(cfg.pencils.r)
        for (local_r, global_r) in enumerate(rr)
            rval = dom.r[global_r, 4]
            gr[:, :, local_r] .= -ri * ro / (ro - ri) / rval^2
        end
        Nu = G.compute_nusselt_number(temp, dom)
        @test isapprox(Nu, 1.0; atol=1e-6)
    end

    @testset "Analytical :conductive (0,0) reconstructs to physical amplitude" begin
        temp = G.create_shtns_temperature_field(Float64, cfg, dom)
        G.set_analytical_initial_conditions!(temp, :temperature, :conductive, amplitude=1.0)
        m00 = G.get_mode_index(cfg, 0, 0)
        slot = G.local_spectral_storage_slot(cfg, m00)
        sr = parent(temp.spectral.data_real)
        # local_r = 1 ⇒ r_frac = 0 ⇒ amplitude*(1 - 0) = 1.0 physical.
        coeff_inner = G.local_spectral_value(sr, slot, 1)
        @test isapprox(_physical_from_mean_coeff(cfg, dom, coeff_inner), 1.0; atol=1e-9)
    end

    @testset "Random/hot_blob (0,0) background means reconstruct physically" begin
        m00 = G.get_mode_index(cfg, 0, 0)
        slot = G.local_spectral_storage_slot(cfg, m00)

        # generate_random_initial_conditions! base conductive: r_frac=0 ⇒ ~1.0
        tr = G.create_shtns_temperature_field(Float64, cfg, dom)
        G.generate_random_initial_conditions!(tr, :temperature, amplitude=0.2, modes_range=1:3, seed=7)
        cr = G.local_spectral_value(parent(tr.spectral.data_real), slot, 1)
        @test isapprox(_physical_from_mean_coeff(cfg, dom, cr), 1.0; atol=5e-2)  # ±seed noise

        # hot_blob l=0 at r_frac=0: value = 0.5·(1-r_frac) + amplitude·Gaussian bump.
        # The bump is NOT compact, so r_frac=0 carries base 0.5 plus the Gaussian
        # tail amplitude·exp(-½(r_center/blob_width)²); reconstructing it confirms
        # the √(4π) normalization is applied.
        amp = 1.0; r_center = 0.5; blob_width = 0.2
        th = G.create_shtns_temperature_field(Float64, cfg, dom)
        G.set_analytical_initial_conditions!(th, :temperature, :hot_blob, amplitude=amp, r_center=r_center, blob_width=blob_width)
        ch = G.local_spectral_value(parent(th.spectral.data_real), slot, 1)
        expected = 0.5 * (1.0 - 0.0) + amp * exp(-0.5 * ((0.0 - r_center) / blob_width)^2)
        @test isapprox(_physical_from_mean_coeff(cfg, dom, ch), expected; atol=1e-9)
    end
end
