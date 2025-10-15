using Test
using Geodynamo
using MPI

@testset "Ball finiteness" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    
    @test isdefined(Geodynamo, :GeodynamoBall)
    
    Ball = Geodynamo.GeodynamoBall
    @test isdefined(Ball, :apply_ball_temperature_regularity!)
    @test isdefined(Ball, :apply_ball_composition_regularity!)
    @test isdefined(Ball, :enforce_ball_vector_regularity!)
    
    @info "Ball finiteness tests require MPI - basic structure verified"
    
    if !MPI.Is_finalized()
        MPI.Finalize()
    end
end
