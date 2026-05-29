using Test

@testset "Solver Parameter Names" begin
    @testset "SolverParameters default construction" begin
        params = GeoDynamo.SolverParameters()
        names = String.(fieldnames(GeoDynamo.SolverParameters))
        @test all(!startswith(name, "d_") for name in names)
        @test all(!startswith(name, "i_") for name in names)
        @test all(!startswith(name, "s_") for name in names)
        @test all(!startswith(name, "b_") for name in names)
        @test params.geometry == :shell
        @test params.radius_ratio == 0.35
        @test params.Ra == 1e6
        @test params.Ek == 1e-4
        @test params.Pr == 1.0
        @test params.Pm == 1.0
        @test params.timestepper isa GeoDynamo.CNAB2
        @test params.output_precision == :float64
        @test params.topography_enabled == false
        @test params.stefan_enabled == false
    end

    @testset "SolverParameters keyword construction" begin
        params = GeoDynamo.SolverParameters(
            nr = 128, lmax = 64, Ra = 1e8, geometry = :ball, radius_ratio = 0.0, nr_inner = 0
        )
        @test params.nr == 128
        @test params.lmax == 64
        @test params.Ra == 1e8
        @test params.geometry == :ball
        @test params.radius_ratio == 0.0
    end

    @testset "Spherical harmonic mode count formula" begin
        for lmax in (4, 8, 32, 64)
            expected_nlm = (lmax + 1) * (lmax + 2) ÷ 2
            @test expected_nlm == sum(lmax - m + 1 for m in 0:lmax)
        end
    end

    @testset "Tolerance constants" begin
        @test GeoDynamo.pivot_tol(Float64) == eps(Float64) * 100.0
        @test GeoDynamo.series_tol(Float64) == eps(Float64) * 100.0
        @test GeoDynamo.rcond_fallback_tol(Float64) == sqrt(eps(Float64))
        @test GeoDynamo.pivot_tol(Float32) == eps(Float32) * Float32(100)
        @test GeoDynamo.series_tol(Float32) == eps(Float32) * Float32(100)
        @test GeoDynamo.rcond_fallback_tol(Float32) == sqrt(eps(Float32))
    end
end
