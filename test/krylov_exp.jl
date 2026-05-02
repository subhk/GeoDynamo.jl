using Test
using LinearAlgebra

@testset "Krylov Exponential Action" begin
    @testset "exp_action_krylov: diagonal matrix" begin
        # For a diagonal A, exp(dt*A)*v should scale each component
        N = 5
        lambda = [-1.0, -2.0, -0.5, -3.0, -0.1]
        dt = 0.1
        v = ones(N)

        function Aop!(out, x)
            for i in 1:N
                out[i] = lambda[i] * x[i]
            end
            return nothing
        end

        result = GeoDynamo.exp_action_krylov(Aop!, v, dt; m=10, tol=1e-12)
        expected = [exp(dt * lambda[i]) for i in 1:N]
        @test result ≈ expected atol=1e-8
    end

    @testset "exp_action_krylov: zero vector input" begin
        N = 4
        Aop!(out, x) = (out .= x; nothing)
        v = zeros(N)
        result = GeoDynamo.exp_action_krylov(Aop!, v, 0.1)
        @test all(result .== 0.0)
    end

    @testset "exp_action_krylov: zero timestep" begin
        N = 4
        Aop!(out, x) = (out .= -x; nothing)
        v = [1.0, 2.0, 3.0, 4.0]
        result = GeoDynamo.exp_action_krylov(Aop!, v, 0.0)
        @test result ≈ v atol=1e-14
    end

    @testset "exp_action_krylov: small dense matrix" begin
        # Build a small negative-definite matrix and verify against dense exp
        A = [-2.0 0.5; 0.5 -3.0]
        dt = 0.05
        v = [1.0, 0.5]

        function Aop!(out, x)
            out[1] = A[1,1]*x[1] + A[1,2]*x[2]
            out[2] = A[2,1]*x[1] + A[2,2]*x[2]
            return nothing
        end

        result = GeoDynamo.exp_action_krylov(Aop!, v, dt; m=10, tol=1e-14)
        expected = exp(dt * A) * v
        @test result ≈ expected atol=1e-10
    end

    @testset "exp_action_krylov: identity operator" begin
        # exp(dt*I)*v = exp(dt)*v
        N = 3
        dt = 0.1
        v = [1.0, 2.0, 3.0]
        Aop!(out, x) = (out .= x; nothing)

        result = GeoDynamo.exp_action_krylov(Aop!, v, dt; m=10, tol=1e-14)
        @test result ≈ exp(dt) .* v atol=1e-10
    end

    @testset "solver_phi1_action_krylov: small dt fallback respects dt*A scale" begin
        λ = [1.0e9, -2.0e9]
        dt = 1.0e-9
        v = [1.0, -0.5]

        function Aop!(out, x)
            @inbounds for i in eachindex(λ)
                out[i] = λ[i] * x[i]
            end
            return nothing
        end

        operator = GeoDynamo.Solver.BandedOperator{Float64}(reshape(copy(λ), 1, :), 0, length(λ))
        lu = GeoDynamo.Solver.solver_factorize_banded(operator)

        result = GeoDynamo.Solver.solver_phi1_action_krylov(Aop!, lu, v, dt; m=10, tol=1e-14)
        expected = [((exp(dt * λi) - 1.0) / (dt * λi)) * vi for (λi, vi) in zip(λ, v)]

        @test result ≈ expected atol=1e-10
    end
end
