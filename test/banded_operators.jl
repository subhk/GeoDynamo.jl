using Test
using LinearAlgebra

@testset "Banded Operators" begin
    @testset "BandedMatrix construction" begin
        bm = GeoDynamo.BandedMatrix(zeros(3, 5), 1, 5)
        @test bm.bandwidth == 1
        @test bm.size == 5
        @test size(bm.data) == (3, 5)
    end

    @testset "BandedMatrix * vector (tridiagonal identity)" begin
        # Build a banded identity: bandwidth=1, diagonal=1
        N = 5
        bw = 1
        data = zeros(Float64, 2*bw+1, N)
        for j in 1:N
            data[bw + 1, j] = 1.0  # diagonal
        end
        bm = GeoDynamo.BandedMatrix(data, bw, N)
        v = collect(1.0:N)
        @test bm * v ≈ v
    end

    @testset "BandedMatrix * vector (tridiagonal)" begin
        # [-1, 2, -1] tridiagonal (second-order central difference stencil)
        N = 6
        bw = 1
        data = zeros(Float64, 3, N)
        for j in 1:N
            data[2, j] = 2.0   # diagonal
            if j > 1
                data[3, j - 1] = -1.0  # sub-diagonal stored at band_row = bw+1+i-j = 1+1+i-j
            end
        end
        # Sub-diagonal: band_row = bw+1+(j+1)-j = bw+2 = 3 → data[3, j]
        # Super-diagonal: band_row = bw+1+(j-1)-j = bw = 1 → data[1, j]
        data_correct = zeros(Float64, 3, N)
        for j in 1:N
            data_correct[2, j] = 2.0   # diagonal
        end
        for j in 2:N
            data_correct[3, j] = -1.0   # sub-diagonal: row i=j, col j-1...
        end
        for j in 1:(N - 1)
            data_correct[1, j] = -1.0   # super-diagonal
        end
        # Actually let's just build the dense matrix and verify
        A = diagm(0 => fill(2.0, N), 1 => fill(-1.0, N-1), -1 => fill(-1.0, N-1))
        # Build banded from dense
        bm = GeoDynamo.BandedMatrix(zeros(Float64, 3, N), 1, N)
        for j in 1:N
            for i in max(1, j - 1):min(N, j + 1)
                band_row = 1 + 1 + i - j  # bw+1+i-j
                bm.data[band_row, j] = A[i, j]
            end
        end
        v = randn(N)
        @test bm * v ≈ A * v atol=1e-12
    end

    @testset "Banded LU factorize and solve" begin
        N = 8
        bw = 2
        # Build a diagonally-dominant banded matrix
        data = zeros(Float64, 2*bw+1, N)
        for j in 1:N
            data[bw + 1, j] = 4.0  # diagonal
            for k in 1:bw
                if j+k <= N
                    data[bw + 1 - k, j] = -0.5/k  # super-diagonal
                end
                if j-k >= 1
                    data[bw + 1 + k, j] = -0.5/k  # sub-diagonal
                end
            end
        end
        bm = GeoDynamo.BandedMatrix(data, bw, N)
        lu = GeoDynamo.factorize_banded(bm)

        b = randn(N)
        x = zeros(N)
        GeoDynamo.solve_banded!(x, lu, b)

        # Build dense matrix to verify
        A = zeros(N, N)
        for j in 1:N
            for i in max(1, j - bw):min(N, j + bw)
                band_row = bw + 1 + i - j
                A[i, j] = data[band_row, j]
            end
        end
        x_ref = A \ b
        @test x ≈ x_ref atol=1e-10
    end

    @testset "solve_banded! supports x === b aliasing" begin
        N = 6
        bw = 1
        data = zeros(Float64, 3, N)
        for j in 1:N
            data[2, j] = 3.0
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
        lu = GeoDynamo.factorize_banded(bm)

        b = randn(N)
        x_copy = copy(b)
        x_separate = zeros(N)

        GeoDynamo.solve_banded!(x_separate, lu, b)
        GeoDynamo.solve_banded!(x_copy, lu, x_copy)  # in-place aliasing

        @test x_copy ≈ x_separate atol=1e-12
    end

    @testset "apply_∂r! matrix-vector product" begin
        N = 5
        bw = 1
        data = zeros(Float64, 3, N)
        for j in 1:N
            data[2, j] = 2.0
        end
        bm = GeoDynamo.BandedMatrix(data, bw, N)
        input = ones(N)
        output = zeros(N)
        GeoDynamo.apply_∂r!(output, bm, input)
        @test output ≈ fill(2.0, N)
    end
end
