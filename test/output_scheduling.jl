using Test

@testset "Output Scheduling" begin
    @testset "OutputConfig defaults" begin
        config = GeoDynamo.default_config()
        @test config.output_interval == 0.1
        @test config.restart_interval == 1.0
        @test config.output_precision == Float64
        @test config.time_tolerance == 1e-10
        @test config.max_output_time == Inf
    end

    @testset "resolve_output_precision" begin
        @test GeoDynamo.resolve_output_precision(:float32) === Float32
        @test GeoDynamo.resolve_output_precision(:float64) === Float64
        # Unknown symbol should warn and default to Float64
        @test (@test_logs (:warn,) GeoDynamo.resolve_output_precision(:float16)) === Float64
    end

    @testset "with_output_precision" begin
        config = GeoDynamo.default_config()
        config32 = GeoDynamo.with_output_precision(config, Float32)
        @test config32.output_precision === Float32
        @test config32.output_interval == config.output_interval
        @test config32.output_dir == config.output_dir
    end

    @testset "TimeTracker creation" begin
        config = GeoDynamo.default_config()
        tracker = GeoDynamo.create_time_tracker(config, 0.0)
        @test tracker.output_count == 0
        @test tracker.restart_count == 0
        @test tracker.grid_file_written == false
        @test tracker.next_output_time == 0.0
    end

    @testset "should_output_now basic logic" begin
        config = GeoDynamo.default_config()  # interval = 0.1
        tracker = GeoDynamo.create_time_tracker(config, 0.0)

        # At t=0, should output (last_output_time = -0.1, so time_since = 0.1 >= 0.1)
        @test GeoDynamo.should_output_now(tracker, 0.0, config) == true

        # Update tracker to record output at t=0
        GeoDynamo.update_tracker!(tracker, 0.0, config, true, false)
        @test tracker.output_count == 1
        @test tracker.last_output_time == 0.0

        # At t=0.05, should NOT output
        @test GeoDynamo.should_output_now(tracker, 0.05, config) == false

        # At t=0.1, should output
        @test GeoDynamo.should_output_now(tracker, 0.1, config) == true
    end

    @testset "should_restart_now basic logic" begin
        config = GeoDynamo.default_config()  # restart_interval = 1.0
        tracker = GeoDynamo.create_time_tracker(config, 0.0)

        # At t=0, should restart (last_restart_time = -1.0)
        @test GeoDynamo.should_restart_now(tracker, 0.0, config) == true

        GeoDynamo.update_tracker!(tracker, 0.0, config, false, true)
        @test tracker.restart_count == 1

        # At t=0.5, should NOT restart
        @test GeoDynamo.should_restart_now(tracker, 0.5, config) == false

        # At t=1.0, should restart
        @test GeoDynamo.should_restart_now(tracker, 1.0, config) == true
    end

    @testset "max_output_time respected" begin
        config = GeoDynamo.OutputConfig(
            GeoDynamo.MIXED_FIELDS, "./output", "test", true, true, true,
            Float64, -1, true,
            0.1, 1.0,
            0.5,   # max_output_time = 0.5
            1e-10
        )
        tracker = GeoDynamo.create_time_tracker(config, 0.0)
        GeoDynamo.update_tracker!(tracker, 0.0, config, true, false)

        # At t=0.3, should output (within max_output_time)
        @test GeoDynamo.should_output_now(tracker, 0.3, config) == true

        # At t=0.6, should NOT output (past max_output_time)
        @test GeoDynamo.should_output_now(tracker, 0.6, config) == false
    end

    @testset "time_to_next_output" begin
        config = GeoDynamo.default_config()
        tracker = GeoDynamo.create_time_tracker(config, 0.0)
        GeoDynamo.update_tracker!(tracker, 0.0, config, true, true)

        dt = GeoDynamo.time_to_next_output(tracker, 0.0, config)
        @test dt ≈ config.output_interval  # next output at 0.1
    end

    @testset "update_tracker! increments counts" begin
        config = GeoDynamo.default_config()
        tracker = GeoDynamo.create_time_tracker(config, 0.0)

        GeoDynamo.update_tracker!(tracker, 0.0, config, true, true)
        @test tracker.output_count == 1
        @test tracker.restart_count == 1

        GeoDynamo.update_tracker!(tracker, 0.1, config, true, false)
        @test tracker.output_count == 2
        @test tracker.restart_count == 1

        GeoDynamo.update_tracker!(tracker, 1.0, config, false, true)
        @test tracker.output_count == 2
        @test tracker.restart_count == 2
    end
end
