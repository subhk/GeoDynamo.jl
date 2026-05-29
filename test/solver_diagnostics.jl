using Test
using MPI

const FINALIZE_MPI_DIAGNOSTICS = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Solver Diagnostics" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping solver diagnostics tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    @testset "tracker constructors and trimming keep newest history" begin
        tracker = GeoDynamo.create_solver_energy_tracker()
        @test isempty(tracker.total_energy)
        @test isempty(tracker.timestamps)
        @test tracker.enable_tracking == true

        n = GeoDynamo.SOLVER_MAX_TRACKER_HISTORY + 2
        tracker.kinetic_energy = collect(1.0:n)
        tracker.magnetic_energy = collect(11.0:(10.0 + n))
        tracker.thermal_energy = collect(21.0:(20.0 + n))
        tracker.compositional_energy = collect(31.0:(30.0 + n))
        tracker.total_energy = collect(41.0:(40.0 + n))
        tracker.timestamps = collect(1:n)

        GeoDynamo.trim_energy_tracker!(tracker)

        kept = GeoDynamo.SOLVER_MAX_TRACKER_HISTORY ÷ 2
        expected_first = n - kept + 1
        @test length(tracker.total_energy) == kept
        @test first(tracker.timestamps) == expected_first
        @test last(tracker.timestamps) == n

        monitor = GeoDynamo.create_solver_solenoidal_monitor()
        @test isempty(monitor.velocity_div_l2)
        @test isempty(monitor.timestamps)
        @test monitor.enable_monitoring == true

        monitor.velocity_div_l2 = collect(1.0:n)
        monitor.velocity_div_linf = collect(2.0:(n + 1.0))
        monitor.magnetic_div_l2 = collect(3.0:(n + 2.0))
        monitor.magnetic_div_linf = collect(4.0:(n + 3.0))
        monitor.timestamps = collect(1:n)

        GeoDynamo.trim_solenoidal_monitor!(monitor)

        @test length(monitor.velocity_div_l2) == kept
        @test first(monitor.timestamps) == expected_first
        @test last(monitor.timestamps) == n
    end

    @testset "NaN helpers respect cadence and count both NaN and Inf" begin
        config = GeoDynamo.NaNDetectionConfig(true, 2, false, false)
        field = reshape([0.0, NaN, Inf, 1.0], 2, 2)

        @test GeoDynamo.check_field_for_nan(field, "field", config, 1) ==
              (false, false, 0, 0)
        @test GeoDynamo.check_field_for_nan(field, "field", config, 2) == (true, true, 1, 1)
    end

    @testset "runtime NaN check uses current runtime timestep" begin
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
            stop_iteration = 10,
            include_magnetic_field = false,
            include_composition = false,
            timestepper = GeoDynamo.CNAB2(),
            topography_enabled = false,
            stefan_enabled = false
        )

        state = GeoDynamo.initialize_simulation(Float64, tiny_params)
        GeoDynamo.initialize_fields!(state)

        # Diagnostics elsewhere already use runtime.timestep_state.step. This
        # check should follow the same source of truth even if state.step has not
        # been synchronized yet.
        state.step = 1
        state.runtime.timestep_state.step = 2
        parent(state.fields.velocity.𝒯.data_real)[1, 1, 1] = NaN

        config = GeoDynamo.NaNDetectionConfig(true, 2, false, false)
        @test GeoDynamo.check_runtime_for_nan(state; config = config) == true
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_DIAGNOSTICS && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
