using Test
using Geodynamo

@testset "Geodynamo.jl" begin
    @testset "Package Loading" begin
        @test isdefined(Geodynamo, :SHTnsKitConfig)
        @test isdefined(Geodynamo, :SimulationState)  
        @test isdefined(Geodynamo, :GeodynamoParameters)
    end
    
    @testset "Basic Types" begin
        @test isdefined(Geodynamo, :SHTnsSpectralField)
        @test isdefined(Geodynamo, :SHTnsPhysicalField)
        @test isdefined(Geodynamo, :RadialDomain)
    end
end
