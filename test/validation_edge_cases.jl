using Test

@testset "Parameter Validation Edge Cases" begin
    @testset "Coarse grid warnings" begin
        params = GeoDynamoParameters(i_N=12)
        _, errors, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test isempty(errors)
        @test any(contains(w, "coarse") for w in warnings)
    end

    @testset "i_M out of range" begin
        params = GeoDynamoParameters(i_M=64, i_L=32)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "i_M") for e in errors)
    end

    @testset "Aliasing warnings for theta/phi resolution" begin
        # i_Th < 2*i_L
        params = GeoDynamoParameters(i_L=32, i_Th=32)
        _, _, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test any(contains(w, "aliasing") for w in warnings)

        # i_Ph < 2*i_M
        params = GeoDynamoParameters(i_M=32, i_Ph=32)
        _, _, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test any(contains(w, "aliasing") for w in warnings)
    end

    @testset "Extreme Rayleigh number warning" begin
        params = GeoDynamoParameters(d_Ra=1e11)
        _, _, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test any(contains(w, "d_Ra") for w in warnings)
    end

    @testset "Very small Ekman number warning" begin
        params = GeoDynamoParameters(d_E=1e-10)
        _, _, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test any(contains(w, "d_E") for w in warnings)
    end

    @testset "Large timestep warning" begin
        params = GeoDynamoParameters(d_timestep=10.0)
        _, _, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test any(contains(w, "d_timestep") for w in warnings)
    end

    @testset "Invalid output precision" begin
        params = GeoDynamoParameters(output_precision=:float16)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "output_precision") for e in errors)
    end

    @testset "Invalid Schmidt number" begin
        params = GeoDynamoParameters(d_Sc=-1.0)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "d_Sc") for e in errors)
    end

    @testset "Valid ball geometry passes" begin
        params = GeoDynamoParameters(geometry=:ball, d_rratio=0.0)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test is_valid
        @test isempty(errors)
    end

    @testset "Valid erk2 scheme passes" begin
        params = GeoDynamoParameters(ts_scheme=:erk2)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test is_valid
    end

    @testset "Valid etd scheme passes" begin
        params = GeoDynamoParameters(ts_scheme=:etd)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test is_valid
    end

    @testset "Multiple simultaneous errors" begin
        params = GeoDynamoParameters(
            d_E=-1.0, d_Ra=-100.0, d_Pr=-1.0, i_N=2,
            d_timestep=-0.01, geometry=:cube
        )
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test length(errors) >= 5
    end
end
