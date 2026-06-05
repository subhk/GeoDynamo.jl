using Test
using GeoDynamo
using Random

# per-l banded matrices stacked into (2bw+1, nr, nl)
function _band(::Type{T}, N, bw; seed) where {T}
    rng = MersenneTwister(seed); d = zeros(T, 2bw+1, N)
    for j in 1:N, i in max(1,j-bw):min(N,j+bw); d[bw+1+i-j,j] = rand(rng,T)-T(0.5); end
    GeoDynamo.BandedMatrix{T}(d, bw, N)
end

@testset "GPU Phase 5c — CNAB2 RHS" begin
    @testset "per-l batched mat-vec == apply_banded_full! [LOCAL]" begin
        N, bw, nl, nm = 10, 2, 4, 3
        mats = [_band(Float64, N, bw; seed = 50 + l) for l in 1:nl]
        matb = zeros(Float64, 2bw+1, N, nl)
        for l in 1:nl; matb[:,:,l] .= mats[l].data; end
        X = rand(MersenneTwister(60), nl, nm, N)
        Y = zeros(nl, nm, N)
        GeoDynamo.gpu_batched_banded_matvec_perl!(Y, X, matb, bw)
        for l in 1:nl, m in 1:nm
            ref = zeros(N)
            GeoDynamo.apply_banded_full!(ref, mats[l], collect(X[l,m,:]))
            @test Y[l,m,:] == ref
        end
    end

    @testset "CNAB2 RHS == formula [LOCAL]" begin
        N, bw, nl, nm = 10, 2, 4, 3
        mats = [_band(Float64, N, bw; seed = 70 + l) for l in 1:nl]
        lin = zeros(Float64, 2bw+1, N, nl); for l in 1:nl; lin[:,:,l] .= mats[l].data; end
        rng = MersenneTwister(71)
        ur = rand(rng,nl,nm,N); ui = rand(rng,nl,nm,N)
        nr_ = rand(rng,nl,nm,N); ni_ = rand(rng,nl,nm,N)
        pr = rand(rng,nl,nm,N); pi_ = rand(rng,nl,nm,N)
        inv_dt = 1.0 / 0.01; lw = 0.5
        rr = zeros(nl,nm,N); ri = zeros(nl,nm,N)
        GeoDynamo.gpu_build_rhs_cnab2!(rr, ri, ur, ui, nr_, ni_, pr, pi_, lin, inv_dt, lw, bw)
        for l in 1:nl, m in 1:nm
            Lur = zeros(N); Lui = zeros(N)
            GeoDynamo.apply_banded_full!(Lur, mats[l], collect(ur[l,m,:]))
            GeoDynamo.apply_banded_full!(Lui, mats[l], collect(ui[l,m,:]))
            for r in 1:N
                @test rr[l,m,r] == inv_dt*ur[l,m,r] + 1.5*nr_[l,m,r] - 0.5*pr[l,m,r] + lw*Lur[r]
                @test ri[l,m,r] == inv_dt*ui[l,m,r] + 1.5*ni_[l,m,r] - 0.5*pi_[l,m,r] + lw*Lui[r]
            end
        end
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5c gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            N, bw, nl, nm = 12, 2, 4, 3
            mats = [_band(Float64, N, bw; seed = 80 + l) for l in 1:nl]
            lin = zeros(Float64, 2bw+1, N, nl); for l in 1:nl; lin[:,:,l] .= mats[l].data; end
            rng = MersenneTwister(81)
            ur=rand(rng,nl,nm,N); ui=rand(rng,nl,nm,N); nr_=rand(rng,nl,nm,N); ni_=rand(rng,nl,nm,N); pr=rand(rng,nl,nm,N); pi_=rand(rng,nl,nm,N)
            inv_dt = 1.0/0.01; lw = 0.5
            crr=zeros(nl,nm,N); cri=zeros(nl,nm,N)
            GeoDynamo.gpu_build_rhs_cnab2!(crr,cri, ur,ui, nr_,ni_, pr,pi_, lin, inv_dt, lw, bw)
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            grr=d(zeros(nl,nm,N)); gri=d(zeros(nl,nm,N))
            GeoDynamo.gpu_build_rhs_cnab2!(grr,gri, d(ur),d(ui), d(nr_),d(ni_), d(pr),d(pi_), d(lin), inv_dt, lw, bw)
            @test grr isa CUDA.CuArray
            @test isapprox(Array(grr), crr; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gri), cri; atol = 1e-12, rtol = 1e-10)
        end
    end
end
