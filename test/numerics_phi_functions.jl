using Test
using LinearAlgebra

# phi1(z) = (e^z - 1)/z ,  phi1(0) = 1
# phi2(z) = (e^z - 1 - z)/z^2 , phi2(0) = 1/2
# Both series operate on a matrix; a scalar multiple of I must reproduce the scalar value on the diagonal.
@testset "phi-function Taylor series (numerics.jl)" begin
    for z in (-0.30, -0.05, 0.10, 0.25)
        A = z * Matrix{Float64}(I, 3, 3)
        phi1_exact = (exp(z) - 1) / z
        phi2_exact = (exp(z) - 1 - z) / z^2
        @test GeoDynamo.phi1_series(A) ≈ phi1_exact * Matrix{Float64}(I, 3, 3) atol = 1e-12
        @test GeoDynamo.phi2_series(A) ≈ phi2_exact * Matrix{Float64}(I, 3, 3) atol = 1e-12
    end
    # Zero-matrix limits
    Z = zeros(3, 3)
    @test GeoDynamo.phi1_series(Z) ≈ Matrix{Float64}(I, 3, 3) atol = 1e-14
    @test GeoDynamo.phi2_series(Z) ≈ Matrix{Float64}(I, 3, 3) ./ 2 atol = 1e-14
end

@testset "rcond_estimate (numerics.jl)" begin
    I3 = Matrix{Float64}(I, 3, 3)
    # Well-conditioned: reciprocal condition number of the identity is 1.
    @test GeoDynamo.rcond_estimate(lu(I3), I3) ≈ 1.0 atol = 1e-10
    # anorm == 0 guard: zero-norm matrix returns one(T). (lu(I3) is an unused placeholder
    # because the guard returns before touching the factorization.)
    @test GeoDynamo.rcond_estimate(lu(I3), zeros(3, 3)) == 1.0
    # Ill-conditioned matrix -> small reciprocal condition estimate.
    B = [1.0 1.0; 1.0 1.0 + 1e-10]
    @test GeoDynamo.rcond_estimate(lu(B), B) < 1e-6
end
