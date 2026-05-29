using Test
using MPI
using GeoDynamo
const G = GeoDynamo

# Regression test for the orthonormal √(4π) normalization of scalar boundary
# values (temperature / composition).
#
# The SH transform is orthonormal (Y_0^0 = 1/√(4π)), so a spatially uniform
# physical boundary value `v` must be stored as the (0,0) spectral coefficient
# v·√(4π). `apply_scalar_boundary_parameters!` is responsible for that
# conversion. This test asserts a `FixedTemperature(v)` BC reconstructs to the
# physical value `v` after synthesis through the real GeoDynamo transform.
#
# Companion `temperature_boundary_numerical.jl` tests the matrix-embedding
# *mechanism* (it injects raw endpoint values straight into the radial solve and
# so does not exercise the physical→coefficient conversion checked here).

# Physical value of a field whose only nonzero coefficient is the (0,0) mode set
# to `coeff` (uniform in radius). The synthesized field is spherically symmetric,
# so every grid point holds the same value.
function __physical_from_mean_coeff(cfg, dom, coeff::Float64)
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

@testset "Scalar BC values carry orthonormal √(4π) normalization" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping"
        return
    end
    MPI.Initialized() || MPI.Init()

    lmax = 6; mmax = 6
    nlat = max(lmax + 2, 12); nlon = max(2lmax + 1, 24); nr = 6
    cfg = G.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = G.create_radial_domain(nr)

    temp = G.create_shtns_temperature_field(Float64, cfg, dom)
    m00 = G.get_mode_index(cfg, 0, 0)

    @testset "FixedTemperature(v) ⇒ physical boundary value v" begin
        G.apply_scalar_boundary_parameters!(temp,
            G.BoundaryConditions(inner=G.FixedTemperature(1.0),
                                 outer=G.FixedTemperature(0.0)))
        phys_inner = __physical_from_mean_coeff(cfg, dom, temp.boundary_values[1, m00])
        phys_outer = __physical_from_mean_coeff(cfg, dom, temp.boundary_values[2, m00])
        @test isapprox(phys_inner, 1.0; atol=1e-10)
        @test isapprox(phys_outer, 0.0; atol=1e-10)
    end

    @testset "FixedTemperature ⇒ physical ΔT across shell preserved" begin
        G.apply_scalar_boundary_parameters!(temp,
            G.BoundaryConditions(inner=G.FixedTemperature(2.5),
                                 outer=G.FixedTemperature(0.5)))
        phys_inner = __physical_from_mean_coeff(cfg, dom, temp.boundary_values[1, m00])
        phys_outer = __physical_from_mean_coeff(cfg, dom, temp.boundary_values[2, m00])
        @test isapprox(phys_inner - phys_outer, 2.0; atol=1e-10)
    end

    @testset "FixedFlux(q) ⇒ physical mean gradient q (same √4π scaling)" begin
        # The Neumann boundary row enforces (d1·T_coeff) = boundary_value, and the
        # (0,0) physical gradient is that coefficient gradient / √(4π); so a
        # prescribed physical gradient q must be stored as q·√(4π), identical to
        # the Dirichlet case. Verify the stored value reconstructs to q.
        G.apply_scalar_boundary_parameters!(temp,
            G.BoundaryConditions(inner=G.FixedFlux(-1.0),
                                 outer=G.FixedFlux(0.5)))
        @test isapprox(__physical_from_mean_coeff(cfg, dom, temp.boundary_values[1, m00]), -1.0; atol=1e-10)
        @test isapprox(__physical_from_mean_coeff(cfg, dom, temp.boundary_values[2, m00]), 0.5; atol=1e-10)
    end
end
