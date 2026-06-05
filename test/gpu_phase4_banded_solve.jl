using Test
using GeoDynamo
using Random

# Build a non-singular banded matrix (diagonally dominant) in BandedLU storage.
function _rand_banded(::Type{T}, N, bw; seed) where {T}
    import_rng = MersenneTwister(seed)
    data = zeros(T, 2bw+1, N)
    for j in 1:N, i in max(1,j-bw):min(N,j+bw)
        data[bw+1+i-j, j] = (i == j) ? (T(2bw) + rand(import_rng, T)) : (rand(import_rng, T) - T(0.5))
    end
    return GeoDynamo.BandedMatrix{T}(data, bw, N)
end

@testset "GPU Phase 4 — Batched Banded Solve" begin
    @testset "pack banded LU [LOCAL]" begin
        N, bw, nl = 8, 2, 3
        lus = [GeoDynamo.factorize_banded(_rand_banded(Float64, N, bw; seed = 10 + l)) for l in 1:nl]
        packed = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
        @test size(packed) == (2bw + 1, N, nl)
        for l in 1:nl
            @test packed[:, :, l] == lus[l].lu
        end
    end

    @testset "batched solve == solve_banded! (multiple l, bw=2) [LOCAL]" begin
        N, bw, nl, nm = 10, 2, 4, 3
        mats = [_rand_banded(Float64, N, bw; seed = 100 + l) for l in 1:nl]
        lus = [GeoDynamo.factorize_banded(m) for m in mats]
        packed = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
        rng = MersenneTwister(7)
        B = rand(rng, Float64, nl, nm, N)
        X = zeros(Float64, nl, nm, N)
        GeoDynamo.gpu_batched_banded_solve!(X, B, packed, bw)
        # reference: per (l,m), solve_banded! with that l's factor on that column
        for l in 1:nl, m in 1:nm
            xref = zeros(Float64, N)
            GeoDynamo.solve_banded!(xref, lus[l], collect(B[l, m, :]))
            @test X[l, m, :] == xref
        end
    end

    @testset "in-place X===B + bandwidth 1 + single l [LOCAL]" begin
        # in-place aliasing (X === B) must match the out-of-place result
        N, bw, nl, nm = 9, 2, 2, 2
        lus = [GeoDynamo.factorize_banded(_rand_banded(Float64, N, bw; seed = 200 + l)) for l in 1:nl]
        packed = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
        B = rand(MersenneTwister(11), Float64, nl, nm, N)
        Xout = zeros(Float64, nl, nm, N)
        GeoDynamo.gpu_batched_banded_solve!(Xout, B, packed, bw)        # out-of-place
        Xin = copy(B)
        GeoDynamo.gpu_batched_banded_solve!(Xin, Xin, packed, bw)        # in-place
        @test Xin == Xout

        # bandwidth 1 (tridiagonal) still correct
        lus1 = [GeoDynamo.factorize_banded(_rand_banded(Float64, N, 1; seed = 300 + l)) for l in 1:nl]
        p1 = GeoDynamo.gpu_pack_banded_lu(lus1, CPU())
        B1 = rand(MersenneTwister(13), Float64, nl, nm, N); X1 = zeros(Float64, nl, nm, N)
        GeoDynamo.gpu_batched_banded_solve!(X1, B1, p1, 1)
        for l in 1:nl, m in 1:nm
            xref = zeros(Float64, N); GeoDynamo.solve_banded!(xref, lus1[l], collect(B1[l,m,:]))
            @test X1[l, m, :] == xref
        end
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-4 gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            N, bw, nl, nm = 12, 2, 4, 3
            lus = [GeoDynamo.factorize_banded(_rand_banded(Float64, N, bw; seed = 400 + l)) for l in 1:nl]
            B = rand(MersenneTwister(21), Float64, nl, nm, N)
            # CPU reference
            cpacked = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
            cX = zeros(Float64, nl, nm, N)
            GeoDynamo.gpu_batched_banded_solve!(cX, B, cpacked, bw)
            # GPU
            gpacked = GeoDynamo.gpu_pack_banded_lu(lus, GPU())
            gX = GeoDynamo.on_architecture(GPU(), zeros(Float64, nl, nm, N))
            gB = GeoDynamo.on_architecture(GPU(), B)
            GeoDynamo.gpu_batched_banded_solve!(gX, gB, gpacked, bw)
            @test gX isa CUDA.CuArray
            @test isapprox(Array(gX), cX; atol = 1e-12, rtol = 1e-10)
        end
    end
end
