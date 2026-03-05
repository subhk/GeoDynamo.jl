using Test

@testset "Parameter Validation" begin
    @testset "Valid default parameters" begin
        params = GeoDynamoParameters()
        is_valid, errors, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test is_valid
        @test isempty(errors)
    end

    @testset "Invalid physical parameters" begin
        # Negative Ekman number
        params = GeoDynamoParameters(d_E=-1.0)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "d_E") for e in errors)

        # Negative Rayleigh number
        params = GeoDynamoParameters(d_Ra=-100.0)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "d_Ra") for e in errors)

        # Negative Prandtl number
        params = GeoDynamoParameters(d_Pr=-1.0)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "d_Pr") for e in errors)

        # Negative magnetic Prandtl number
        params = GeoDynamoParameters(d_Pm=-1.0)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "d_Pm") for e in errors)
    end

    @testset "Invalid grid parameters" begin
        params = GeoDynamoParameters(i_N=2)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "i_N") for e in errors)

        params = GeoDynamoParameters(i_L=0)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "i_L") for e in errors)
    end

    @testset "Invalid timestepping" begin
        params = GeoDynamoParameters(d_timestep=-0.001)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "d_timestep") for e in errors)

        params = GeoDynamoParameters(i_maxtstep=0)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "i_maxtstep") for e in errors)
    end

    @testset "End time must exceed start time" begin
        params = GeoDynamoParameters(d_time=1.0, d_t_end=0.5)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "d_t_end") for e in errors)
    end

    @testset "Ball geometry requires d_rratio == 0" begin
        params = GeoDynamoParameters(geometry=:ball, d_rratio=0.35)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "d_rratio") for e in errors)
    end

    @testset "Invalid geometry" begin
        params = GeoDynamoParameters(geometry=:cube)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "geometry") for e in errors)
    end

    @testset "Invalid timestepping scheme" begin
        params = GeoDynamoParameters(ts_scheme=:rk4)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "ts_scheme") for e in errors)
    end

    @testset "Strict mode throws on invalid params" begin
        params = GeoDynamoParameters(d_E=-1.0)
        @test_throws ErrorException GeoDynamo.validate_parameters(params; strict=true)
    end

    @testset "CFL warning for large timestep" begin
        params = GeoDynamoParameters(d_timestep=1.0, i_L=32, d_E=1e-4)
        _, _, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test any(contains(w, "CFL") for w in warnings)
    end
end
