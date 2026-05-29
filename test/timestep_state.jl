using Test

@testset "Timestep State" begin
    @testset "TimestepState construction" begin
        ts = GeoDynamo.TimestepState(0.0, 1e-4, 0, 0, 0.0, false, true)
        @test ts.time == 0.0
        @test ts.dt == 1e-4
        @test ts.step == 0
        @test ts.iteration == 0
        @test ts.error == 0.0
        @test ts.converged == false
        @test ts.needs_ab2_bootstrap == true
    end

    @testset "TimestepState is mutable" begin
        ts = GeoDynamo.TimestepState(0.0, 1e-4, 0, 0, 0.0, false, true)
        ts.time = 0.5
        ts.dt = 2e-4
        ts.step = 100
        ts.converged = true
        ts.needs_ab2_bootstrap = false

        @test ts.time == 0.5
        @test ts.dt == 2e-4
        @test ts.step == 100
        @test ts.converged == true
        @test ts.needs_ab2_bootstrap == false
    end

    @testset "NaNDetectionConfig construction" begin
        config = GeoDynamo.NaNDetectionConfig(true, 10, false, true)
        @test config.enabled == true
        @test config.check_every_n_steps == 10
        @test config.abort_on_nan == false
        @test config.verbose == true
    end

    @testset "check_field_for_nan: mixed NaN and Inf" begin
        config = GeoDynamo.NaNDetectionConfig(true, 1, false, false)
        data = zeros(5, 5)
        data[1, 1] = NaN
        data[2, 2] = Inf
        data[3, 3] = -Inf
        data[4, 4] = NaN

        has_nan, has_inf,
        nan_count, inf_count = GeoDynamo.check_field_for_nan(
            data, "mixed", config, 1)
        @test has_nan
        @test has_inf
        @test nan_count == 2
        @test inf_count == 2
    end

    @testset "check_field_for_nan: empty array" begin
        config = GeoDynamo.NaNDetectionConfig(true, 1, false, false)
        data = Float64[]
        has_nan, has_inf,
        nan_count, inf_count = GeoDynamo.check_field_for_nan(
            data, "empty", config, 1)
        @test !has_nan
        @test !has_inf
        @test nan_count == 0
        @test inf_count == 0
    end

    @testset "check_field_for_nan: 3D array" begin
        config = GeoDynamo.NaNDetectionConfig(true, 1, false, false)
        data = zeros(3, 4, 5)
        data[2, 3, 4] = NaN
        has_nan, has_inf,
        nan_count, inf_count = GeoDynamo.check_field_for_nan(
            data, "3d", config, 1)
        @test has_nan
        @test nan_count == 1
    end
end
