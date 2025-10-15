using Test
using Geodynamo

const TEST_DIR = @__DIR__
const MPI_FINALIZE_KEY = "GEODYNAMO_TEST_MPI_FINALIZE"

@testset "Geodynamo.jl" begin
    @testset "Package Loading" begin
        @test isdefined(Geodynamo, :SHTnsKitConfig)
        @test isdefined(Geodynamo, :SimulationState)
        @test isdefined(Geodynamo, :GeodynamoParameters)
        @test isdefined(Geodynamo, :BoundaryConditions)
        @test isdefined(Geodynamo, :InitialConditions)
        @test isdefined(Geodynamo, :GeodynamoShell)
        @test isdefined(Geodynamo, :GeodynamoBall)
    end

    @testset "Basic Types" begin
        @test isdefined(Geodynamo, :SHTnsSpectralField)
        @test isdefined(Geodynamo, :SHTnsPhysicalField)
        @test isdefined(Geodynamo, :RadialDomain)

        # Test parameter system
        params = GeodynamoParameters()
        @test params !== nothing
    end

    @testset "Submodules" begin
        @test isdefined(Geodynamo.BoundaryConditions, :FieldType)
        @test isdefined(Geodynamo.BoundaryConditions, :BoundaryLocation)
        @test isdefined(Geodynamo.BoundaryConditions, :BoundaryType)
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
    @testset "Extended Geodynamo tests" begin
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

println("✓ Geodynamo test suite completed")
