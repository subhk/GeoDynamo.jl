using Test
using MPI

const FINALIZE_MPI_FIELD_BEHAVIOR = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") ==
                                    "true"

@testset "Composition And Magnetic Field Behavior" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping composition/magnetic field tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    lmax = 4
    mmax = 4
    nlat = max(lmax + 2, 10)
    nlon = max(2lmax + 1, 16)
    nr = 6

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    shell = GeoDynamo.create_radial_domain(nr)

    @testset "composition defaults and initial-condition helpers" begin
        comp = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)

        @test size(comp.boundary_values) == (2, cfg.nlm)
        @test all(comp.boundary_values .== 0.0)
        @test all(comp.bc_type_inner .== Int(GeoDynamo.NEUMANN))
        @test all(comp.bc_type_outer .== Int(GeoDynamo.NEUMANN))
        @test all(comp.internal_sources .== 0.0)

        GeoDynamo.set_composition_boundary_conditions!(comp, :fixed, :no_flux, 2.5, -1.0)
        @test all(comp.bc_type_inner .== Int(GeoDynamo.DIRICHLET))
        @test all(comp.bc_type_outer .== Int(GeoDynamo.NEUMANN))
        @test all(comp.boundary_values[1, :] .== 2.5)

        # :linear should overwrite any stale higher modes and imaginary part.
        fill!(parent(comp.spectral.data_real), 5.0)
        fill!(parent(comp.spectral.data_imag), -3.0)
        GeoDynamo.set_composition_ic!(comp, :linear, shell)

        spec_real = parent(comp.spectral.data_real)
        spec_imag = parent(comp.spectral.data_imag)
        @test all(spec_imag .== 0.0)
        @test all(spec_real[2:end, :, :] .== 0.0)
    end

    @testset "magnetic work arrays and diagnostics stay consistent" begin
        mag = GeoDynamo.create_shtns_magnetic_fields(Float64, cfg, shell, shell)

        @test GeoDynamo.compute_magnetic_energy(mag, shell) ≈ 0.0 atol=1e-15
        @test GeoDynamo.compute_ohmic_dissipation(mag) ≈ 0.0 atol=1e-15
        @test GeoDynamo.compute_magnetic_helicity(mag) ≈ 0.0 atol=1e-15

        fill!(parent(mag.work_tor.data_real), 1.0)
        fill!(parent(mag.work_tor.data_imag), -1.0)
        fill!(parent(mag.work_pol.data_real), 2.0)
        fill!(parent(mag.work_pol.data_imag), -2.0)
        fill!(parent(mag.work_physical.r_component.data), 3.0)
        fill!(parent(mag.work_physical.θ_component.data), 4.0)
        fill!(parent(mag.work_physical.φ_component.data), 5.0)
        fill!(parent(mag.induction_physical.r_component.data), 6.0)
        fill!(parent(mag.induction_physical.θ_component.data), 7.0)
        fill!(parent(mag.induction_physical.φ_component.data), 8.0)

        GeoDynamo.zero_magnetic_work_arrays!(mag)

        @test all(parent(mag.work_tor.data_real) .== 0.0)
        @test all(parent(mag.work_tor.data_imag) .== 0.0)
        @test all(parent(mag.work_pol.data_real) .== 0.0)
        @test all(parent(mag.work_pol.data_imag) .== 0.0)
        @test all(parent(mag.work_physical.r_component.data) .== 0.0)
        @test all(parent(mag.work_physical.θ_component.data) .== 0.0)
        @test all(parent(mag.work_physical.φ_component.data) .== 0.0)
        @test all(parent(mag.induction_physical.r_component.data) .== 0.0)
        @test all(parent(mag.induction_physical.θ_component.data) .== 0.0)
        @test all(parent(mag.induction_physical.φ_component.data) .== 0.0)
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_FIELD_BEHAVIOR && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
