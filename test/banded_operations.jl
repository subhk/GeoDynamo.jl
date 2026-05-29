using Test
using LinearAlgebra

@testset "Banded Matrix Operations" begin
    @testset "banded_to_dense identity" begin
        N = 5;
        bw = 1
        data = zeros(Float64, 2*bw+1, N)
        for j in 1:N
            data[bw + 1, j] = 1.0
        end
        bm = GeoDynamo.BandedMatrix(data, bw, N)
        dense = GeoDynamo.banded_to_dense(bm)
        @test dense ≈ Matrix{Float64}(I, N, N)
    end

    @testset "banded_to_dense tridiagonal" begin
        N = 6;
        bw = 1
        A = diagm(0 => fill(4.0, N), 1 => fill(-1.0, N-1), -1 => fill(-1.0, N-1))

        data = zeros(Float64, 3, N)
        for j in 1:N
            for i in max(1, j - bw):min(N, j + bw)
                data[bw + 1 + i - j, j] = A[i, j]
            end
        end
        bm = GeoDynamo.BandedMatrix(data, bw, N)
        dense = GeoDynamo.banded_to_dense(bm)
        @test dense ≈ A
    end

    @testset "banded_to_dense pentadiagonal" begin
        N = 8;
        bw = 2
        # Build a pentadiagonal matrix
        A = zeros(N, N)
        for i in 1:N
            A[i, i] = 6.0
            if i > 1
                ;
                A[i, i - 1] = -2.0;
            end
            if i < N
                ;
                A[i, i + 1] = -2.0;
            end
            if i > 2
                ;
                A[i, i - 2] = 0.5;
            end
            if i < N-1
                ;
                A[i, i + 2] = 0.5;
            end
        end

        data = zeros(Float64, 2*bw+1, N)
        for j in 1:N
            for i in max(1, j - bw):min(N, j + bw)
                data[bw + 1 + i - j, j] = A[i, j]
            end
        end
        bm = GeoDynamo.BandedMatrix(data, bw, N)
        dense = GeoDynamo.banded_to_dense(bm)
        @test dense ≈ A
    end

    @testset "apply_banded_full! matches dense multiply" begin
        N = 6;
        bw = 2
        A = zeros(N, N)
        for i in 1:N
            A[i, i] = 3.0
            if i > 1
                ;
                A[i, i - 1] = -1.0;
            end
            if i < N
                ;
                A[i, i + 1] = -1.0;
            end
            if i > 2
                ;
                A[i, i - 2] = 0.25;
            end
            if i < N-1
                ;
                A[i, i + 2] = 0.25;
            end
        end

        data = zeros(Float64, 2*bw+1, N)
        for j in 1:N
            for i in max(1, j - bw):min(N, j + bw)
                data[bw + 1 + i - j, j] = A[i, j]
            end
        end
        bm = GeoDynamo.BandedMatrix(data, bw, N)

        v = randn(N)
        out = zeros(N)
        GeoDynamo.apply_banded_full!(out, bm, v)
        @test out ≈ A * v atol=1e-12
    end

    @testset "apply_banded_full! with zero vector" begin
        N = 5;
        bw = 1
        data = ones(Float64, 3, N)
        bm = GeoDynamo.BandedMatrix(data, bw, N)
        v = zeros(N)
        out = ones(N)  # should be zeroed
        GeoDynamo.apply_banded_full!(out, bm, v)
        @test all(out .== 0.0)
    end

    @testset "factorize_banded detects singular matrix" begin
        N = 4;
        bw = 1
        data = zeros(Float64, 3, N)
        # Zero diagonal → singular
        bm = GeoDynamo.BandedMatrix(data, bw, N)
        @test_throws ErrorException GeoDynamo.factorize_banded(bm)
    end

    @testset "BandedMatrix * operator consistency with apply_banded_full!" begin
        N = 5;
        bw = 1
        data = zeros(Float64, 3, N)
        for j in 1:N
            data[2, j] = 2.0
            if j > 1
                ;
                data[3, j] = -0.5;
            end
            if j < N
                ;
                data[1, j] = -0.5;
            end
        end
        bm = GeoDynamo.BandedMatrix(data, bw, N)
        v = randn(N)

        result_mul = bm * v
        result_apply = zeros(N)
        GeoDynamo.apply_banded_full!(result_apply, bm, v)
        @test result_mul ≈ result_apply atol=1e-12
    end

    @testset "apply_banded_full! also accepts solver banded operators" begin
        N = 5;
        bw = 1
        data = zeros(Float64, 3, N)
        for j in 1:N
            data[2, j] = 2.0
            if j > 1
                ;
                data[3, j] = -0.5;
            end
            if j < N
                ;
                data[1, j] = -0.5;
            end
        end
        A = zeros(N, N)
        for j in 1:N
            for i in max(1, j - bw):min(N, j + bw)
                A[i, j] = data[bw + 1 + i - j, j]
            end
        end

        op = GeoDynamo.Solver.BandedOperator(copy(data), bw, N)
        v = randn(N)
        out = zeros(N)
        GeoDynamo.apply_banded_full!(out, op, v)
        @test out ≈ A * v atol=1e-12
    end
end
