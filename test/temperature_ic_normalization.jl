using Test
using MPI
using GeoDynamo
const G = GeoDynamo

# Regression test: the conductive temperature initial condition must use the
# same orthonormal √(4π) normalization as the boundary conditions, so the
# initial field is physically T=1 at the inner shell boundary and T=0 at the
# outer boundary (and therefore starts ON the FixedTemperature(1)/(0) BC instead
# of √(4π) away from it).

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

@testset "Conductive temperature IC carries orthonormal √(4π) normalization" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping"
        return
    end
    MPI.Initialized() || MPI.Init()

    params = G.SolverParameters(
        geometry = :shell,
        nr = 8, lmax = 4, mmax = 4, nlat = 10, nlon = 16,
        include_composition = true,
        include_magnetic = false,
        # temperature_bcs default = FixedTemperature(1.0) / FixedTemperature(0.0)
        # initialize_composition_field! seeds a BC-consistent (0,0) background, so
        # pin a uniform physical mean of 0.5 via the composition boundaries (the
        # default is 0/0, which would seed 0 — see src/physics/composition/solver.jl).
        composition_bcs = G.BoundaryConditions(
            inner = G.FixedTemperature(0.5), outer = G.FixedTemperature(0.5))
    )

    state = G.initialize_simulation(params)
    G.initialize_temperature_field!(state)

    temp = state.fields.temperature
    cfg = temp.config
    dom = state.backend.outer_core_domain
    N = dom.N

    m00 = G.get_mode_index(cfg, 0, 0)
    slot = G.local_spectral_storage_slot(cfg, m00)
    sr = parent(temp.spectral.data_real)

    inner_coeff = G.local_spectral_value(sr, slot, 1)
    outer_coeff = G.local_spectral_value(sr, slot, N)

    phys_inner = _physical_from_mean_coeff(cfg, dom, inner_coeff)
    phys_outer = _physical_from_mean_coeff(cfg, dom, outer_coeff)

    # Standard nondimensional shell conduction: physical T(inner)=1, T(outer)=0.
    @test isapprox(phys_inner, 1.0; atol = 1e-9)
    @test isapprox(phys_outer, 0.0; atol = 1e-9)

    # IC must agree with the FixedTemperature BC endpoints it is solved against.
    bc = G.get_bc_vectors(temp)
    @test isapprox(inner_coeff, bc.inner_real[m00]; atol = 1e-9)
    @test isapprox(outer_coeff, bc.outer_real[m00]; atol = 1e-9)

    # Composition shares the scalar path; its uniform background of physical
    # mean 0.5 must likewise reconstruct to 0.5 after synthesis.
    G.initialize_composition_field!(state)
    comp = state.fields.composition
    @test comp !== nothing
    cspec = parent(comp.spectral.data_real)
    cslot = G.local_spectral_storage_slot(comp.config, m00)
    comp_coeff = G.local_spectral_value(cspec, cslot, 1)
    @test isapprox(_physical_from_mean_coeff(comp.config, dom, comp_coeff), 0.5; atol = 1e-9)
end
