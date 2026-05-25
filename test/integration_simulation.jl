using Test
using MPI

const FINALIZE_MPI_INTEG = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Integration: Simulation Initialization and Stepping" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping integration tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    # Use a tiny grid to keep the test fast (~seconds, not minutes)
    tiny_params = GeoDynamo.SolverParameters(
        architecture = :cpu,
        geometry = :shell,
        nr = 16,
        nr_inner = 4,
        lmax = 4,
        mmax = 4,
        nlat = 12,
        nlon = 16,
        Ra = 1e4,
        Ek = 1e-2,
        Pr = 1.0,
        Pm = 1.0,
        timestep = 1e-4,
        start_time = 0.0,
        end_time = 1e-3,
        max_steps = 10,
        include_magnetic_field = false,
        include_composition = false,
        timestepper = GeoDynamo.CNAB2(),
        topography_enabled = false,
        stefan_enabled = false,
    )
    cfl_limit = 0.1 / (
        tiny_params.lmax^2 * max(
            1.0,
            tiny_params.Pm / tiny_params.Pr,
            tiny_params.Pm / tiny_params.Sc,
            tiny_params.Ek,
        )
    )
    @test tiny_params.timestep < cfl_limit

    # ------------------------------------------------------------------
    # Test 1: Simulation initialization succeeds
    # ------------------------------------------------------------------
    @testset "initialize_simulation" begin
        state = GeoDynamo.initialize_simulation(Float64, tiny_params)

        @test state !== nothing
        @test state.parameters.geometry === :shell
        @test state.backend.shtns_config.lmax == 4
        @test state.runtime.𝒟ᵒᶜ.N == 16

        # Fields should exist and be finite after initialization
        GeoDynamo.initialize_fields!(state)

        temp_real = parent(state.fields.temperature.spectral.data_real)
        @test all(isfinite, temp_real)

        vel_tor = parent(state.fields.velocity.𝒯.data_real)
        @test all(isfinite, vel_tor)
        # Velocity starts at zero
        @test all(vel_tor .== 0.0)

        # Temperature should have a non-trivial conductive profile (l=0,m=0)
        @test !all(temp_real .== 0.0)
    end

    # ------------------------------------------------------------------
    # Test 2: Single nonlinear + implicit step stays finite
    # ------------------------------------------------------------------
    @testset "single timestep stays finite" begin
        state = GeoDynamo.initialize_simulation(Float64, tiny_params)
        GeoDynamo.initialize_fields!(state)

        GeoDynamo.advance_solver_step!(state)

        # All fields should remain finite
        @test all(isfinite, parent(state.fields.temperature.spectral.data_real))
        @test all(isfinite, parent(state.fields.temperature.spectral.data_imag))
        @test all(isfinite, parent(state.fields.velocity.𝒯.data_real))
        @test all(isfinite, parent(state.fields.velocity.𝒯.data_imag))
        @test all(isfinite, parent(state.fields.velocity.𝒫.data_real))
        @test all(isfinite, parent(state.fields.velocity.𝒫.data_imag))
    end

    # ------------------------------------------------------------------
    # Test 3: Multiple timesteps — energy stays bounded
    # ------------------------------------------------------------------
    @testset "multi-step energy bounded" begin
        state = GeoDynamo.initialize_simulation(Float64, tiny_params)
        GeoDynamo.initialize_fields!(state)

        n_steps = 5

        # Record initial thermal energy
        E0 = GeoDynamo.compute_thermal_energy(state.fields.temperature)
        @test isfinite(E0)
        @test E0 >= 0.0

        for step in 1:n_steps
            GeoDynamo.advance_solver_step!(state)
            @test state.step == step
        end

        # Final energy should be finite and not blow up
        Ef = GeoDynamo.compute_thermal_energy(state.fields.temperature)
        @test isfinite(Ef)
        @test Ef >= 0.0
        # Energy should not have grown by more than 100× in 5 steps with moderate parameters
        @test Ef < max(E0, 1e-10) * 100

        # All fields still finite
        @test all(isfinite, parent(state.fields.temperature.spectral.data_real))
        @test all(isfinite, parent(state.fields.velocity.𝒯.data_real))
        @test all(isfinite, parent(state.fields.velocity.𝒫.data_real))
    end

    # ------------------------------------------------------------------
    # Test 4: Field extraction for restart roundtrip
    # ------------------------------------------------------------------
    @testset "field extraction" begin
        state = GeoDynamo.initialize_simulation(Float64, tiny_params)
        GeoDynamo.initialize_fields!(state)

        fields = GeoDynamo.extract_all_fields(state)

        @test haskey(fields, "velocity_toroidal")
        @test haskey(fields, "velocity_poloidal")
        @test haskey(fields, "magnetic_toroidal")
        @test haskey(fields, "temperature_spectral")
        @test !haskey(fields, "composition_spectral")  # composition disabled

        # Verify the extracted data matches the state
        @test fields["temperature_spectral"]["real"] ≈ parent(state.fields.temperature.spectral.data_real)
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_INTEG && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
