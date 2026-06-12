using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using Random

# A banded matrix in (2bw+1, N) storage (NOT necessarily invertible — it's a derivative op).
function _rand_band_mat(::Type{T}, N, bw; seed) where {T}
    rng = MersenneTwister(seed)
    data = zeros(T, 2bw+1, N)
    for j in 1:N, i in max(1,j-bw):min(N,j+bw)
        data[bw+1+i-j, j] = rand(rng, T) - T(0.5)
    end
    return GeoDynamo.BandedMatrix{T}(data, bw, N)
end

@testset "GPU Phase 5a — Spectral Curl" begin
    @testset "batched banded mat-vec == apply_radial_derivative! [LOCAL]" begin
        N, bw, nl, nm = 10, 2, 3, 2
        mat = _rand_band_mat(Float64, N, bw; seed = 1)
        X = rand(MersenneTwister(2), Float64, nl, nm, N)
        Y = zeros(Float64, nl, nm, N)
        GeoDynamo.gpu_batched_banded_matvec!(Y, X, mat.data, bw)
        for l in 1:nl, m in 1:nm
            ref = zeros(Float64, N)
            GeoDynamo.apply_radial_derivative!(ref, mat, collect(X[l, m, :]))
            @test Y[l, m, :] == ref
        end
    end

    @testset "spectral curl == Stage-2 vorticity formula [LOCAL]" begin
        N, bw, nl, nm = 10, 2, 4, 3
        d1 = _rand_band_mat(Float64, N, bw; seed = 11)
        d2 = _rand_band_mat(Float64, N, bw; seed = 12)
        rng = MersenneTwister(13)
        str = rand(rng, nl, nm, N); sti = rand(rng, nl, nm, N)   # source toroidal (T)
        spr = rand(rng, nl, nm, N); spi = rand(rng, nl, nm, N)   # source poloidal (P)
        lfac = Float64[l * (l + 1) for l in 0:(nl - 1)]
        r_vec = [0.5 + 0.1k for k in 1:N]
        rinv = 1.0 ./ r_vec
        rinv2 = rinv .^ 2
        dtr = zeros(nl,nm,N); dti = zeros(nl,nm,N); dpr = zeros(nl,nm,N); dpi = zeros(nl,nm,N)
        GeoDynamo.gpu_spectral_curl!(dtr, dti, dpr, dpi, str, sti, spr, spi, d1.data, d2.data, lfac, rinv, rinv2, r_vec, bw)
        # independent reference per (l,m): d2·P via apply_radial_derivative!,
        # then the Stage-2 solenoidal curl formula:
        #   T_curl = (P'' - l(l+1)P/r^2)/r, P_curl = -r*T.
        for l in 1:nl, m in 1:nm
            d2Pr = zeros(N); d2Pi = zeros(N)
            GeoDynamo.apply_radial_derivative!(d2Pr, d2, collect(spr[l,m,:]))
            GeoDynamo.apply_radial_derivative!(d2Pi, d2, collect(spi[l,m,:]))
            for r in 1:N
                @test dtr[l,m,r] == rinv[r] * (d2Pr[r] - lfac[l] * rinv2[r] * spr[l,m,r])
                @test dti[l,m,r] == rinv[r] * (d2Pi[r] - lfac[l] * rinv2[r] * spi[l,m,r])
                @test dpr[l,m,r] == -r_vec[r] * str[l,m,r]
                @test dpi[l,m,r] == -r_vec[r] * sti[l,m,r]
            end
        end
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5a gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            N, bw, nl, nm = 12, 2, 4, 3
            d1 = _rand_band_mat(Float64, N, bw; seed = 21); d2 = _rand_band_mat(Float64, N, bw; seed = 22)
            rng = MersenneTwister(23)
            str = rand(rng, nl,nm,N); sti = rand(rng, nl,nm,N); spr = rand(rng, nl,nm,N); spi = rand(rng, nl,nm,N)
            lfac = Float64[l*(l+1) for l in 0:(nl-1)]; r_vec = [0.5+0.1k for k in 1:N]; rinv = 1.0 ./ r_vec; rinv2 = rinv.^2
            # CPU reference
            z() = zeros(Float64, nl,nm,N)
            cdtr,cdti,cdpr,cdpi = z(),z(),z(),z()
            GeoDynamo.gpu_spectral_curl!(cdtr,cdti,cdpr,cdpi, str,sti,spr,spi, d1.data,d2.data, lfac,rinv,rinv2,r_vec, bw)
            # GPU
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            gdtr,gdti,gdpr,gdpi = d(z()),d(z()),d(z()),d(z())
            GeoDynamo.gpu_spectral_curl!(gdtr,gdti,gdpr,gdpi, d(str),d(sti),d(spr),d(spi),
                                         d(d1.data),d(d2.data), d(lfac),d(rinv),d(rinv2),d(r_vec), bw)
            @test gdtr isa CUDA.CuArray
            @test isapprox(Array(gdtr), cdtr; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gdti), cdti; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gdpr), cdpr; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gdpi), cdpi; atol = 1e-12, rtol = 1e-10)
        end
    end
end
