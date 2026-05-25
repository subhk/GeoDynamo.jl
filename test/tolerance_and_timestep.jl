using Test

@testset "Tolerance Constants" begin
    @testset "pivot_tol" begin
        @test GeoDynamo.pivot_tol(Float64) == eps(Float64) * 100.0
        @test GeoDynamo.pivot_tol(Float32) == eps(Float32) * Float32(100)
        @test GeoDynamo.pivot_tol(Float64) > 0
        @test GeoDynamo.pivot_tol(Float32) > GeoDynamo.pivot_tol(Float64)
    end

    @testset "series_tol" begin
        @test GeoDynamo.series_tol(Float64) == eps(Float64) * 100.0
        @test GeoDynamo.series_tol(Float32) == eps(Float32) * Float32(100)
        @test GeoDynamo.series_tol(Float64) > 0
    end

    @testset "rcond_fallback_tol" begin
        @test GeoDynamo.rcond_fallback_tol(Float64) == sqrt(eps(Float64))
        @test GeoDynamo.rcond_fallback_tol(Float32) == sqrt(eps(Float32))
        @test GeoDynamo.rcond_fallback_tol(Float64) > GeoDynamo.pivot_tol(Float64)
    end

    @testset "PIVOT_SINGULARITY_FACTOR and SERIES_CONVERGENCE_FACTOR" begin
        @test GeoDynamo.PIVOT_SINGULARITY_FACTOR == 100
        @test GeoDynamo.SERIES_CONVERGENCE_FACTOR == 100
    end

    @testset "solver/numerics.jl has no duplicate constants" begin
        @test !isdefined(GeoDynamo, :SOLVER_PIVOT_SINGULARITY_FACTOR)
        @test !isdefined(GeoDynamo, :SOLVER_SERIES_CONVERGENCE_FACTOR)
        @test !isdefined(GeoDynamo, :pivot_tolerance)
        @test !isdefined(GeoDynamo, :series_tolerance)
        @test !isdefined(GeoDynamo, :rcond_fallback_tolerance)
    end
end
