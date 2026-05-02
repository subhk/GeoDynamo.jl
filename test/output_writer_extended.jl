using Test

@testset "Output Writer Extended" begin

    # ─── OutputSpace enum ─────────────────────────────────────────────────
    @testset "OutputSpace enum values are distinct" begin
        @test Int(GeoDynamo.MIXED_FIELDS) != Int(GeoDynamo.PHYSICAL_ONLY)
        @test Int(GeoDynamo.PHYSICAL_ONLY) != Int(GeoDynamo.SPECTRAL_ONLY)
        @test Int(GeoDynamo.MIXED_FIELDS) != Int(GeoDynamo.SPECTRAL_ONLY)
    end

    # ─── OutputConfig constructors ────────────────────────────────────────
    @testset "default_config with Float32 precision" begin
        config = GeoDynamo.default_config(Float32)
        @test config.output_precision === Float32
        @test config.output_interval == 0.1
        @test config.output_dir == "./output"
        @test config.filename_prefix == "geodynamo"
        @test config.include_metadata == true
        @test config.include_grid == true
        @test config.include_diagnostics == true
        @test config.spectral_lmax_output == -1
        @test config.overwrite_files == true
    end

    @testset "default_config with keyword precision" begin
        config = GeoDynamo.default_config(; precision=Float32)
        @test config.output_precision === Float32
    end

    @testset "OutputConfig full construction" begin
        config = GeoDynamo.OutputConfig(
            GeoDynamo.SPECTRAL_ONLY,
            "/tmp/test",
            "myprefix",
            false,
            false,
            false,
            Float32,
            16,
            false,
            0.5,
            2.0,
            10.0,
            1e-8
        )
        @test config.output_space == GeoDynamo.SPECTRAL_ONLY
        @test config.output_dir == "/tmp/test"
        @test config.filename_prefix == "myprefix"
        @test config.include_metadata == false
        @test config.include_grid == false
        @test config.include_diagnostics == false
        @test config.output_precision === Float32
        @test config.spectral_lmax_output == 16
        @test config.overwrite_files == false
        @test config.output_interval == 0.5
        @test config.restart_interval == 2.0
        @test config.max_output_time == 10.0
        @test config.time_tolerance == 1e-8
    end

    # ─── with_output_precision ────────────────────────────────────────────
    @testset "with_output_precision preserves other fields" begin
        base = GeoDynamo.OutputConfig(
            GeoDynamo.PHYSICAL_ONLY,
            "/custom/dir",
            "custom",
            false, false, true,
            Float64, 32, true,
            0.2, 0.5, 5.0, 1e-6
        )
        modified = GeoDynamo.with_output_precision(base, Float32)
        @test modified.output_precision === Float32
        @test modified.output_space == GeoDynamo.PHYSICAL_ONLY
        @test modified.output_dir == "/custom/dir"
        @test modified.filename_prefix == "custom"
        @test modified.spectral_lmax_output == 32
        @test modified.output_interval == 0.2
        @test modified.restart_interval == 0.5
        @test modified.max_output_time == 5.0
    end

    # ─── resolve_output_precision ─────────────────────────────────────────
    @testset "resolve_output_precision" begin
        @test GeoDynamo.resolve_output_precision(:float32) === Float32
        @test GeoDynamo.resolve_output_precision(:float64) === Float64
        @test (@test_logs (:warn,) GeoDynamo.resolve_output_precision(:bfloat16)) === Float64
    end

    # ─── FieldInfo ────────────────────────────────────────────────────────
    @testset "FieldInfo default constructor" begin
        fi = GeoDynamo.FieldInfo()
        @test fi.nlat == 0
        @test fi.nlon == 0
        @test fi.nr == 0
        @test fi.nlm == 0
        @test isempty(fi.theta)
        @test isempty(fi.phi)
        @test isempty(fi.r)
        @test isempty(fi.l_values)
        @test isempty(fi.m_values)
        @test fi.has_pencils == false
        @test fi.has_config == false
        @test fi.config === nothing
        @test isempty(fi.local_ranges)
    end

    # ─── TimeTracker advanced scenarios ───────────────────────────────────
    @testset "TimeTracker with custom start time" begin
        config = GeoDynamo.default_config()
        tracker = GeoDynamo.create_time_tracker(config, 5.0)
        @test tracker.next_output_time == 5.0
        @test tracker.next_restart_time == 5.0
        @test tracker.last_output_time == 5.0 - config.output_interval
        @test tracker.last_restart_time == 5.0 - config.restart_interval
    end

    @testset "multiple sequential outputs tracked correctly" begin
        config = GeoDynamo.OutputConfig(
            GeoDynamo.MIXED_FIELDS, "./out", "test", true, true, true,
            Float64, -1, true,
            0.1, 1.0, Inf, 1e-10
        )
        tracker = GeoDynamo.create_time_tracker(config, 0.0)

        # Output at t=0
        GeoDynamo.update_tracker!(tracker, 0.0, config, true, true)
        @test tracker.output_count == 1
        @test tracker.restart_count == 1
        @test tracker.next_output_time ≈ 0.1
        @test tracker.next_restart_time ≈ 1.0

        # Output at t=0.1
        GeoDynamo.update_tracker!(tracker, 0.1, config, true, false)
        @test tracker.output_count == 2
        @test tracker.restart_count == 1
        @test tracker.next_output_time ≈ 0.2

        # Output at t=0.2, restart at t=0.2 too
        GeoDynamo.update_tracker!(tracker, 0.2, config, true, true)
        @test tracker.output_count == 3
        @test tracker.restart_count == 2
    end

    @testset "time_to_next_output returns minimum" begin
        config = GeoDynamo.OutputConfig(
            GeoDynamo.MIXED_FIELDS, "./out", "test", true, true, true,
            Float64, -1, true,
            0.5, 2.0, Inf, 1e-10
        )
        tracker = GeoDynamo.create_time_tracker(config, 0.0)
        GeoDynamo.update_tracker!(tracker, 0.0, config, true, true)

        # At t=0, next output at 0.5, next restart at 2.0 → min is 0.5
        dt = GeoDynamo.time_to_next_output(tracker, 0.0, config)
        @test dt ≈ 0.5
    end

    # ─── should_output_now edge cases ─────────────────────────────────────
    @testset "should_output_now respects time_tolerance" begin
        config = GeoDynamo.OutputConfig(
            GeoDynamo.MIXED_FIELDS, "./out", "test", true, true, true,
            Float64, -1, true,
            1.0, 1.0, Inf, 0.01  # tolerance = 0.01
        )
        tracker = GeoDynamo.create_time_tracker(config, 0.0)
        GeoDynamo.update_tracker!(tracker, 0.0, config, true, false)

        # t=0.99 is within tolerance of 1.0 interval
        @test GeoDynamo.should_output_now(tracker, 0.99, config) == true

        # t=0.98 is NOT within tolerance
        @test GeoDynamo.should_output_now(tracker, 0.98, config) == false
    end
end
