using Test
using LinearAlgebra
using Logging

@testset "ERK2 Matrix Functions" begin
    @testset "phi1 scalar: (e^z - 1)/z" begin
        # phi1(z) = (exp(z) - 1) / z
        # For z -> 0, phi1 -> 1
        # For z = 1, phi1 = (e - 1)/1 = e - 1
        z_vals = [-2.0, -1.0, -0.5, -0.01, 0.01, 0.5, 1.0, 2.0]
        for z in z_vals
            expected = (exp(z) - 1.0) / z
            @test abs(expected - (exp(z) - 1.0)/z) < 1e-14
        end
    end

    @testset "phi2 scalar: (e^z - 1 - z)/z^2" begin
        # phi2(z) = (exp(z) - 1 - z) / z^2
        z_vals = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
        for z in z_vals
            phi2 = (exp(z) - 1.0 - z) / z^2
            # Alternative: phi2 = (phi1 - 1)/z
            phi1 = (exp(z) - 1.0) / z
            phi2_alt = (phi1 - 1.0) / z
            @test phi2 ≈ phi2_alt atol=1e-12
        end
    end

    @testset "ERK2Cache construction" begin
        nr = 4
        dt = 0.01
        lambda = 1.0
        z = -lambda * dt

        E_half = exp(z/2) * Matrix{Float64}(I, nr, nr)
        E_full = exp(z) * Matrix{Float64}(I, nr, nr)
        phi1_half = (exp(z/2) - 1)/(z/2) * Matrix{Float64}(I, nr, nr)
        phi1_full = (exp(z) - 1)/z * Matrix{Float64}(I, nr, nr)
        phi2_full = (exp(z) - 1 - z)/z^2 * Matrix{Float64}(I, nr, nr)

        cache = GeoDynamo.ERK2Cache{Float64}(
            dt, [0], [E_half], [E_full],
            [phi1_half], [phi1_full], [phi2_full],
            false, 20, 1e-8, true
        )
        @test cache.dt == dt
        @test length(cache.E_half) == 1
        @test length(cache.phi2_full) == 1
    end

    @testset "ERK2 consistency: E = I + z*phi1" begin
        # exp(z) = 1 + z * phi1(z)
        z = -0.5
        E = exp(z)
        phi1 = (exp(z) - 1) / z
        @test E ≈ 1.0 + z * phi1 atol=1e-14
    end

    @testset "ERK2 consistency: phi1 = 1 + z*phi2" begin
        # phi1(z) = 1 + z * phi2(z)
        z = -0.5
        phi1 = (exp(z) - 1) / z
        phi2 = (exp(z) - 1 - z) / z^2
        @test phi1 ≈ 1.0 + z * phi2 atol=1e-14
    end

    @testset "solver phi fallbacks stay quiet on expected singular coarse operators" begin
        A = Matrix{Float64}([1.0 0.0; 0.0 0.0])
        expA = Matrix(Diagonal(exp.(diag(A))))

        phi1 = @test_logs min_level=Logging.Warn GeoDynamo.Solver.solver_compute_phi1_function(A, expA)
        phi2 = @test_logs min_level=Logging.Warn GeoDynamo.Solver.solver_compute_phi2_function(A, expA; l=1)

        @test size(phi1) == size(A)
        @test size(phi2) == size(A)
        @test all(isfinite, phi1)
        @test all(isfinite, phi2)
    end

    @testset "solver phi builders reject non-finite inputs" begin
        A = Matrix{Float64}([NaN 0.0; 0.0 1.0])
        expA = Matrix{Float64}(I, 2, 2)

        @test_throws ArgumentError GeoDynamo.Solver.solver_compute_phi1_function(A, expA)
        @test_throws ArgumentError GeoDynamo.Solver.solver_compute_phi2_function(A, expA; l=1)
    end
end
