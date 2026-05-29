using Test
using MPI

const FINALIZE_MPI_THERM = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Temperature Field" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping thermal tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    lmax = 4;
    mmax = 4
    nlat = max(lmax + 2, 10)
    nlon = max(2lmax + 1, 16)
    nr = 6

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    dom = GeoDynamo.create_radial_domain(nr)
    temp = GeoDynamo.create_shtns_temperature_field(Float64, cfg, dom)

    @testset "Initial state is zero" begin
        @test all(parent(temp.spectral.data_real) .== 0.0)
        @test all(parent(temp.spectral.data_imag) .== 0.0)
        @test all(parent(temp.temperature.data) .== 0.0)
    end

    @testset "l_factors type stability" begin
        @test eltype(temp.l_factors) == Float64
        temp32 = GeoDynamo.create_shtns_temperature_field(Float32, cfg, dom)
        @test eltype(temp32.l_factors) == Float32
    end

    @testset "Internal sources default to zero" begin
        @test all(temp.internal_sources .== 0.0)
        @test length(temp.internal_sources) == nr
    end

    @testset "Boundary values shape" begin
        @test size(temp.boundary_values) == (2, cfg.nlm)
        # Row 1 = inner, Row 2 = outer
        @test all(temp.boundary_values .== 0.0)
    end

    @testset "BC type arrays" begin
        @test length(temp.bc_type_inner) == cfg.nlm
        @test length(temp.bc_type_outer) == cfg.nlm
        @test all(temp.bc_type_inner .== Int(GeoDynamo.DIRICHLET))
        @test all(temp.bc_type_outer .== Int(GeoDynamo.DIRICHLET))
    end

    @testset "Thermal energy of zero field is zero" begin
        energy = GeoDynamo.compute_thermal_energy(temp)
        @test energy ≈ 0.0 atol=1e-15
    end

    @testset "Thermal energy of nonzero field is positive" begin
        # Set l=0,m=0 mode to nonzero at all radial points
        spec_real = parent(temp.spectral.data_real)
        slot = GeoDynamo.local_spectral_storage_slot(cfg, 1)
        if slot !== nothing
            spec_real[slot[1], slot[2], :] .= 1.0
        end
        energy = GeoDynamo.compute_thermal_energy(temp)
        @test energy > 0.0
        # Reset
        fill!(spec_real, 0.0)
    end

    @testset "Derivative matrices dimensions" begin
        @test temp.∂r.size == nr
        @test temp.∂²r.size == nr
        @test temp.∂r.bandwidth > 0
    end

    @testset "Performance counters initialized to zero" begin
        @test temp.computation_time[] == 0.0
        @test temp.transform_time[] == 0.0
        @test temp.comm_time[] == 0.0
        @test temp.spectral_time[] == 0.0
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_THERM && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
