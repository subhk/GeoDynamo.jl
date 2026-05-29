using Test

@testset "NaN/Inf Detection" begin
    @testset "NaNDetectionConfig defaults" begin
        config = GeoDynamo.DEFAULT_NAN_CONFIG
        @test config.enabled
        @test config.check_every_n_steps == 1
        @test config.abort_on_nan
        @test config.verbose
    end

    @testset "check_field_for_nan with clean data" begin
        config = GeoDynamo.NaNDetectionConfig(true, 1, false, false)
        data = rand(10, 10)
        has_nan, has_inf,
        nan_count, inf_count = GeoDynamo.check_field_for_nan(data, "test_field", config, 1)
        @test !has_nan
        @test !has_inf
        @test nan_count == 0
        @test inf_count == 0
    end

    @testset "check_field_for_nan detects NaN" begin
        config = GeoDynamo.NaNDetectionConfig(true, 1, false, false)
        data = rand(10, 10)
        data[3, 5] = NaN
        data[7, 2] = NaN
        has_nan, has_inf,
        nan_count, inf_count = GeoDynamo.check_field_for_nan(data, "test_field", config, 1)
        @test has_nan
        @test !has_inf
        @test nan_count == 2
        @test inf_count == 0
    end

    @testset "check_field_for_nan detects Inf" begin
        config = GeoDynamo.NaNDetectionConfig(true, 1, false, false)
        data = rand(10, 10)
        data[1, 1] = Inf
        data[5, 5] = -Inf
        has_nan, has_inf,
        nan_count, inf_count = GeoDynamo.check_field_for_nan(data, "test_field", config, 1)
        @test !has_nan
        @test has_inf
        @test nan_count == 0
        @test inf_count == 2
    end

    @testset "check_field_for_nan respects check_every_n_steps" begin
        config = GeoDynamo.NaNDetectionConfig(true, 5, false, false)
        data = fill(NaN, 3, 3)

        # Should skip on non-matching step
        has_nan, _, _, _ = GeoDynamo.check_field_for_nan(data, "test", config, 3)
        @test !has_nan

        # Should check on matching step
        has_nan, _, _, _ = GeoDynamo.check_field_for_nan(data, "test", config, 5)
        @test has_nan
    end

    @testset "check_field_for_nan respects enabled flag" begin
        config = GeoDynamo.NaNDetectionConfig(false, 1, false, false)
        data = fill(NaN, 3, 3)
        has_nan, _, _, _ = GeoDynamo.check_field_for_nan(data, "test", config, 1)
        @test !has_nan
    end
end
