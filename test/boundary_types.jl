using Test

@testset "Boundary Types and Enums" begin
    @testset "BoundaryType enum values" begin
        @test isdefined(GeoDynamo, :DIRICHLET)
        @test isdefined(GeoDynamo, :NEUMANN)
        @test isdefined(GeoDynamo.bcs, :BoundaryType)
        # Ensure they have distinct integer values
        @test Int(GeoDynamo.DIRICHLET) != Int(GeoDynamo.NEUMANN)
    end

    @testset "FieldType enum" begin
        @test isdefined(GeoDynamo.bcs, :FieldType)
    end

    @testset "BoundaryLocation enum" begin
        @test isdefined(GeoDynamo.bcs, :BoundaryLocation)
    end

    @testset "BoundaryConditionSet type exists" begin
        @test isdefined(GeoDynamo.bcs, :BoundaryConditionSet)
    end

    @testset "OutputSpace enum" begin
        @test isdefined(GeoDynamo, :MIXED_FIELDS)
        @test isdefined(GeoDynamo, :PHYSICAL_ONLY)
        @test isdefined(GeoDynamo, :SPECTRAL_ONLY)
    end
end
