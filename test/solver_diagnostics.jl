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
            include_magnetic = false,
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
        parent(state.fields.velocity.toroidal.data_real)[1, 1, 1] = NaN

        config = GeoDynamo.NaNDetectionConfig(true, 2, false, false)
        @test GeoDynamo.check_runtime_for_nan(state; config = config) == true
    end

    @testset "energy norms match analytic values" begin
        # field_energy = 0.5 * Σ x^2 ; vector_energy sums the three components.
        @test GeoDynamo.field_energy(ones(Float64, 2, 3, 4)) == 0.5 * 24
        @test GeoDynamo.field_energy(zeros(Float64, 3, 3, 3)) == 0.0
        vr = ones(Float64, 2, 2, 2)
        vt = fill(2.0, 2, 2, 2)
        vp = fill(3.0, 2, 2, 2)
        # 0.5 * (8*1 + 8*4 + 8*9)
        @test GeoDynamo.vector_energy(vr, vt, vp) == 0.5 * (8 + 32 + 72)
    end

    @testset "energy + solenoidal diagnostics on a live solver state" begin
        params = GeoDynamo.SolverParameters(
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
            include_magnetic = true,
            include_composition = true,
            timestepper = GeoDynamo.CNAB2(),
            topography_enabled = false,
            stefan_enabled = false
        )
        state = GeoDynamo.initialize_simulation(Float64, params)
        GeoDynamo.initialize_fields!(state)

        tracker = state.energy_tracker
        monitor = state.solenoidal_monitor

        @testset "compute_total_energy! appends finite, additive samples" begin
            n0 = length(tracker.total_energy)
            GeoDynamo.compute_total_energy!(state)
            @test length(tracker.total_energy) == n0 + 1
            @test isfinite(tracker.total_energy[end])
            @test tracker.total_energy[end] >= 0
            # total is the sum of the four component energies that were recorded
            @test tracker.total_energy[end] ≈ tracker.kinetic_energy[end] +
                  tracker.magnetic_energy[end] +
                  tracker.thermal_energy[end] +
                  tracker.compositional_energy[end]
            # magnetic + composition are enabled, so their slots are populated
            @test tracker.magnetic_energy[end] >= 0
            @test tracker.compositional_energy[end] >= 0

            # disabling tracking is a no-op
            tracker.enable_tracking = false
            n1 = length(tracker.total_energy)
            @test GeoDynamo.compute_total_energy!(state) === nothing
            @test length(tracker.total_energy) == n1
            tracker.enable_tracking = true
        end

        @testset "compute_divergence_spectral returns (0.0, 0.0) for zero-initialized fields" begin
            # The velocity field is initialized to zero by initialize_fields!, so the
            # real divergence is exactly zero.  This exercises the live code path
            # (synthesize V, apply banded D1 + sphtor angular derivatives) rather than
            # characterizing a stub.  See test/solenoidal_transform_pair.jl for the
            # gate test that validates the non-zero case.
            div = GeoDynamo.compute_divergence_spectral(
                state.fields.velocity.toroidal,
                state.fields.velocity.poloidal,
                state.backend.outer_core_domain
            )
            @test div == (0.0, 0.0)
        end

        @testset "check_solenoidal_constraint! records divergence samples" begin
            m0 = length(monitor.velocity_div_l2)
            GeoDynamo.check_solenoidal_constraint!(state)
            @test length(monitor.velocity_div_l2) == m0 + 1
            # velocity is zero after initialize_fields! → divergence is exactly 0.0
            @test monitor.velocity_div_l2[end] == 0.0
            # magnetic IC has a non-zero r^2(1-r) profile → the real diagnostic now
            # returns a non-negative finite norm (was 0.0 only with the old stub).
            @test isfinite(monitor.magnetic_div_l2[end])
            @test monitor.magnetic_div_l2[end] >= 0.0

            monitor.enable_monitoring = false
            m1 = length(monitor.velocity_div_l2)
            @test GeoDynamo.check_solenoidal_constraint!(state) === nothing
            @test length(monitor.velocity_div_l2) == m1
            monitor.enable_monitoring = true
        end

        @testset "run_diagnostics! honours the interval gate" begin
            ne = length(tracker.total_energy)
            nm = length(monitor.velocity_div_l2)
            state.runtime.timestep_state.step = 4
            GeoDynamo.run_diagnostics!(state; interval = 2)  # 4 % 2 == 0 -> runs
            @test length(tracker.total_energy) == ne + 1
            @test length(monitor.velocity_div_l2) == nm + 1

            ne2 = length(tracker.total_energy)
            state.runtime.timestep_state.step = 5
            GeoDynamo.run_diagnostics!(state; interval = 2)  # 5 % 2 != 0 -> skipped
            @test length(tracker.total_energy) == ne2
        end

        @testset "report_* exercise reporting + warning branches" begin
            # early-return when the step is not on the reporting interval
            @test GeoDynamo.report_energy_conservation(state, 1; interval = 100) === nothing
            @test GeoDynamo.report_solenoidal_constraint(state, 1; interval = 100) === nothing

            # force the >1% drift warning branch
            tracker.total_energy = [1.0, 2.0]
            tracker.kinetic_energy = [0.5, 1.0]
            tracker.magnetic_energy = [0.0, 0.0]
            tracker.thermal_energy = [0.5, 1.0]
            tracker.compositional_energy = [0.0, 0.0]
            tracker.timestamps = [1, 2]
            @test GeoDynamo.report_energy_conservation(state, 100; interval = 100) === nothing

            # force the solenoidal-violation warning branch
            monitor.velocity_div_l2 = [1.0]
            monitor.velocity_div_linf = [1.0]
            monitor.magnetic_div_l2 = [1.0]
            monitor.magnetic_div_linf = [1.0]
            monitor.timestamps = [1]
            @test GeoDynamo.report_solenoidal_constraint(state, 100; interval = 100) === nothing
        end
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_DIAGNOSTICS && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
