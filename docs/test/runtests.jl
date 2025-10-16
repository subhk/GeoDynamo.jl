using Test
using GeoDynamo

@testset "GeoDynamo.jl" begin
    @testset "Package Loading" begin
        @test isdefined(GeoDynamo, :SHTnsKitConfig)
        @test isdefined(GeoDynamo, :SimulationState)  
        @test isdefined(GeoDynamo, :GeoDynamoParameters)
    end
    
    @testset "Basic Types" begin
        @test isdefined(GeoDynamo, :SHTnsSpectralField)
        @test isdefined(GeoDynamo, :SHTnsPhysicalField)
        @test isdefined(GeoDynamo, :RadialDomain)
    end
end
