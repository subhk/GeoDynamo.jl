using Test

@testset "Parameter Validation Edge Cases" begin
    @testset "Coarse grid warnings" begin
        params = GeoDynamo.SolverParameters(nr=12)
        _, errors, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test isempty(errors)
        @test any(contains(w, "coarse") for w in warnings)
    end

    @testset "mmax out of range" begin
        params = GeoDynamo.SolverParameters(mmax=64, lmax=32)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "mmax") for e in errors)
    end

    @testset "Aliasing warnings for theta/phi resolution" begin
        # nlat < 2*lmax
        params = GeoDynamo.SolverParameters(lmax=32, nlat=32)
        _, _, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test any(contains(w, "aliasing") for w in warnings)

        # nlon < 2*mmax
        params = GeoDynamo.SolverParameters(mmax=32, nlon=32)
        _, _, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test any(contains(w, "aliasing") for w in warnings)
    end

    @testset "Extreme Rayleigh number warning" begin
        params = GeoDynamo.SolverParameters(Ra=1e11)
        _, _, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test any(contains(w, "Ra") for w in warnings)
    end

    @testset "Very small Ekman number warning" begin
        params = GeoDynamo.SolverParameters(Ek=1e-10)
        _, _, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test any(contains(w, "Ek") for w in warnings)
    end

    @testset "Large timestep warning" begin
        params = GeoDynamo.SolverParameters(timestep=10.0)
        _, _, warnings = GeoDynamo.validate_parameters(params; strict=false)
        @test any(contains(w, "timestep") for w in warnings)
    end

    @testset "Invalid output precision" begin
        params = GeoDynamo.SolverParameters(output_precision=:float16)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "output_precision") for e in errors)
    end

    @testset "Invalid Schmidt number" begin
        params = GeoDynamo.SolverParameters(Sc=-1.0)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test any(contains(e, "Sc") for e in errors)
    end

    @testset "Valid ball geometry passes" begin
        params = GeoDynamo.SolverParameters(geometry=:ball, radius_ratio=0.0)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test is_valid
        @test isempty(errors)
    end

    @testset "Valid erk2 scheme passes" begin
        params = GeoDynamo.SolverParameters(timestepper=GeoDynamo.ERK2())
        is_valid, _, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test is_valid
    end

    @testset "Valid etd scheme passes" begin
        params = GeoDynamo.SolverParameters(timestepper=GeoDynamo.ETD())
        is_valid, _, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test is_valid
    end

    @testset "Multiple simultaneous errors" begin
        params = GeoDynamo.SolverParameters(
            Ek=-1.0, Ra=-100.0, Pr=-1.0, nr=2,
            timestep=-0.01, geometry=:cube,
        )
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict=false)
        @test !is_valid
        @test length(errors) >= 5
    end
end
