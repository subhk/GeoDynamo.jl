using Test
using MPI
using GeoDynamo
const G = GeoDynamo

# The conductive (0,0) background of the composition IC must match the prescribed
# composition boundary values (no jump at the boundaries), instead of an
# unconditional 0.5 that conflicted with the default 0/0 Dirichlet BC.

function _physical_from_mean_coeff(cfg, dom, coeff::Float64)
    spec = G.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    phys = G.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.phi)
    fill!(parent(spec.data_real), 0.0);
    fill!(parent(spec.data_imag), 0.0)
    m00 = G.get_mode_index(cfg, 0, 0)
    slot = G.local_spectral_storage_slot(cfg, m00)
    for r in 1:dom.N
        G.set_local_spectral_value!(parent(spec.data_real), slot, r, coeff)
    end
    G.shtnskit_spectral_to_physical!(spec, phys)
    return parent(phys.data)[1]
end

@testset "Composition IC background matches its boundary conditions" begin
    MPI.Finalized() && (@warn "MPI finalized; skipping"; return)
    MPI.Initialized() || MPI.Init()

    @testset "non-trivial Dirichlet (0.8/0.2) ⇒ IC reconstructs to those values" begin
        params = G.SolverParameters(
            geometry = :shell, nr = 8, lmax = 4, mmax = 4, nlat = 10, nlon = 16,
            include_composition = true, include_magnetic = false,
            composition_bcs = G.BoundaryConditions(inner = G.FixedTemperature(0.8),
                outer = G.FixedTemperature(0.2)))
        state = G.initialize_simulation(params)
        G.initialize_composition_field!(state)
        comp = state.fields.composition
        cfg = comp.config;
        dom = state.backend.outer_core_domain;
        N = dom.N
        m00 = G.get_mode_index(cfg, 0, 0);
        slot = G.local_spectral_storage_slot(cfg, m00)
        sr = parent(comp.spectral.data_real)
        inner_c = G.local_spectral_value(sr, slot, 1)
        outer_c = G.local_spectral_value(sr, slot, N)
        @test isapprox(_physical_from_mean_coeff(cfg, dom, inner_c), 0.8; atol = 1e-9)
        @test isapprox(_physical_from_mean_coeff(cfg, dom, outer_c), 0.2; atol = 1e-9)
    end

    @testset "default 0/0 BC ⇒ background is 0 (no spurious 0.5)" begin
        params = G.SolverParameters(
            geometry = :shell, nr = 8, lmax = 4, mmax = 4, nlat = 10, nlon = 16,
            include_composition = true, include_magnetic = false)
        state = G.initialize_simulation(params)
        G.initialize_composition_field!(state)
        comp = state.fields.composition
        cfg = comp.config;
        N = state.backend.outer_core_domain.N
        m00 = G.get_mode_index(cfg, 0, 0);
        slot = G.local_spectral_storage_slot(cfg, m00)
        sr = parent(comp.spectral.data_real)
        for r in 1:N
            @test G.local_spectral_value(sr, slot, r) == 0.0
        end
    end
end
