using Test
using GeoDynamo

const TEST_DIR = @__DIR__
const MPI_FINALIZE_KEY = "GEODYNAMO_TEST_MPI_FINALIZE"

@testset "GeoDynamo.jl" begin
    @testset "Package Loading" begin
        @test isdefined(GeoDynamo, :SHTnsKitConfig)
        @test isdefined(GeoDynamo, :SimulationState)
        @test isdefined(GeoDynamo, :GeoDynamoParameters)
        @test isdefined(GeoDynamo, :BoundaryConditions)
        @test isdefined(GeoDynamo, :InitialConditions)
        @test isdefined(GeoDynamo, :GeoDynamoShell)
        @test isdefined(GeoDynamo, :GeoDynamoBall)
    end

    @testset "Basic Types" begin
        @test isdefined(GeoDynamo, :SHTnsSpectralField)
        @test isdefined(GeoDynamo, :SHTnsPhysicalField)
        @test isdefined(GeoDynamo, :RadialDomain)

        # Test parameter system
        params = GeoDynamoParameters()
        @test params !== nothing
    end

    @testset "Submodules" begin
        @test isdefined(GeoDynamo.BoundaryConditions, :FieldType)
        @test isdefined(GeoDynamo.BoundaryConditions, :BoundaryLocation)
        @test isdefined(GeoDynamo.BoundaryConditions, :BoundaryType)
    end
end

additional_tests = (
    "erk2.jl",
    "shell_boundaries.jl",
    "shtnskit_roundtrip.jl",
    "ball_roundtrip.jl",
    "ball_finiteness.jl",
)

previous_setting = haskey(ENV, MPI_FINALIZE_KEY) ? ENV[MPI_FINALIZE_KEY] : nothing
ENV[MPI_FINALIZE_KEY] = "false"

try
    @testset "Extended GeoDynamo tests" begin
        for file in additional_tests
            include(joinpath(TEST_DIR, file))
        end
    end
finally
    if previous_setting === nothing
        delete!(ENV, MPI_FINALIZE_KEY)
    else
        ENV[MPI_FINALIZE_KEY] = previous_setting
    end
end

println("✓ GeoDynamo test suite completed")
