using Test
using MPI

const FINALIZE_MPI_OW = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Output Writer" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping outputs writer tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    # ------------------------------------------------------------------
    # Test 1: OutputConfig construction from parameters
    # ------------------------------------------------------------------
    @testset "OutputConfig from parameters" begin
        config = GeoDynamo.output_config_from_parameters(
            output_precision = :float32,
            output_interval  = 50.0,
        )
        @test config !== nothing
        @test config.output_precision === Float32
        @test config.output_interval == 50.0
    end

    # ------------------------------------------------------------------
    # Test 2: Time tracker creation and scheduling
    # ------------------------------------------------------------------
    @testset "TimeTracker scheduling" begin
        config = GeoDynamo.output_config_from_parameters(output_interval=10.0)
        tracker = GeoDynamo.create_time_tracker(config)
        @test tracker !== nothing
    end

    # ------------------------------------------------------------------
    # Test 3: Energy tracker reset
    # ------------------------------------------------------------------
    @testset "EnergyTracker reset" begin
        GeoDynamo.reset_energy_tracker!()
        tracker = GeoDynamo.ENERGY_TRACKER
        @test isempty(tracker.kinetic_energy)
        @test isempty(tracker.magnetic_energy)
        @test isempty(tracker.thermal_energy)
        @test isempty(tracker.total_energy)
        @test isempty(tracker.timestamps)
        @test tracker.enable_tracking == true
    end

    # ------------------------------------------------------------------
    # Test 4: Compute field energy on synthetic data
    # ------------------------------------------------------------------
    @testset "compute_field_energy" begin
        # Zero field → zero energy
        data_zero = zeros(Float64, 4, 4, 4)
        E0 = GeoDynamo.compute_field_energy(data_zero)
        @test E0 ≈ 0.0 atol=1e-15

        # Uniform field → known energy
        data_ones = ones(Float64, 4, 4, 4)
        E1 = GeoDynamo.compute_field_energy(data_ones)
        # E = 0.5 * sum(|f|²) = 0.5 * 64 = 32 (local, single rank)
        # With MPI, Allreduce sums across ranks, so expected = 32 * nprocs
        nprocs = MPI.Comm_size(GeoDynamo.get_comm())
        @test E1 ≈ 32.0 * nprocs atol=1e-10
    end

    # ------------------------------------------------------------------
    # Test 5: Compute vector energy
    # ------------------------------------------------------------------
    @testset "compute_vector_energy" begin
        n = 3
        vr = ones(Float64, n, n, n)
        vt = zeros(Float64, n, n, n)
        vp = zeros(Float64, n, n, n)
        E = GeoDynamo.compute_vector_energy(vr, vt, vp)
        nprocs = MPI.Comm_size(GeoDynamo.get_comm())
        # E = 0.5 * 27 * nprocs
        @test E ≈ 0.5 * 27.0 * nprocs atol=1e-10

        # Zero vectors → zero energy
        E0 = GeoDynamo.compute_vector_energy(
            zeros(Float64, n, n, n),
            zeros(Float64, n, n, n),
            zeros(Float64, n, n, n))
        @test E0 ≈ 0.0 atol=1e-15
    end

    # ------------------------------------------------------------------
    # Test 6: Enhanced metadata creation
    # ------------------------------------------------------------------
    @testset "enhanced metadata" begin
        tiny_params = GeoDynamo.SolverParameters(
            architecture=:cpu,
            geometry=:shell,
            radial_points_outer=16,
            radial_points_inner=4,
            spherical_degree=4,
            spherical_order=4,
            latitude_points=12,
            longitude_points=16,
            thermal_rayleigh=1e4,
            ekman_number=1e-2,
            timestep=1e-4,
            end_time=1e-3,
            max_steps=10,
            include_magnetic_field=false,
            include_composition=false,
            topography_enabled=false,
            stefan_enabled=false,
        )
        state = GeoDynamo.initialize_simulation(Float64, tiny_params)
        GeoDynamo.initialize_fields!(state)

        metadata = GeoDynamo.create_enhanced_metadata(state, 0.001, 10)
        @test metadata["current_time"] ≈ 0.001
        @test metadata["current_step"] == 10
        @test metadata["geometry"] === :shell
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_OW && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
