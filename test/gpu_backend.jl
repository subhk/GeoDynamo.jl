using Test
using GeoDynamo
import KernelAbstractions

@testset "GPU Architecture (Oceananigans style)" begin
    @testset "Architecture type hierarchy" begin
        @test CPU() isa AbstractArchitecture
        @test GPU{Nothing} <: AbstractArchitecture

        # Old global state is gone (not in this task, just prep for future tests)
        @test Base.isexported(GeoDynamo, :AbstractArchitecture)
        @test Base.isexported(GeoDynamo, :CPU)
        @test Base.isexported(GeoDynamo, :GPU)
        @test Base.isexported(GeoDynamo, :arch_zeros)
        @test Base.isexported(GeoDynamo, :on_architecture)
        @test Base.isexported(GeoDynamo, :get_backend)
    end

    @testset "arch_zeros on CPU" begin
        a = arch_zeros(CPU(), Float64, 3, 4)
        @test a == zeros(Float64, 3, 4)
        @test a isa Matrix{Float64}
    end

    @testset "on_architecture CPU returns Array" begin
        a = [1, 2, 3]
        @test on_architecture(CPU(), a) isa Array
        @test on_architecture(CPU(), a) == a
    end

    @testset "get_backend CPU" begin
        @test get_backend(CPU()) isa KernelAbstractions.CPU
    end

    @testset "Architecture symbol conversion" begin
        @test GeoDynamo.architecture_from_symbol(:cpu) isa CPU
        @test GeoDynamo.architecture_from_symbol(:gpu) isa GPU
        @test_throws ArgumentError GeoDynamo.architecture_from_symbol(:cuda)
    end

    @testset "GPU{B} parametric" begin
        @test GPU{Nothing} <: AbstractArchitecture
        g = GPU(nothing)
        @test g isa GPU
        @test g.backend === nothing
    end
end
