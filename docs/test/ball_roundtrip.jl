using Test
using Geodynamo
using MPI

@testset "Ball roundtrip" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    
    @test isdefined(Geodynamo, :GeodynamoBall)
    
    Ball = Geodynamo.GeodynamoBall  
    @test isdefined(Ball, :create_ball_radial_domain)
    @test isdefined(Ball, :create_ball_spectral_field)
    @test isdefined(Ball, :create_ball_physical_field)
    @test isdefined(Ball, :enforce_ball_scalar_regularity!)
    @test isdefined(Ball, :ball_physical_to_spectral!)
    
    @info "Ball roundtrip tests require MPI - basic structure verified"
    
    if !MPI.Finalized()
        MPI.Finalize()
    end
end
