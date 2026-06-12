using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
import SHTnsKit
using Random

function _phase3_rand_band(::Type{T}, N, bw; seed) where {T}
    rng = MersenneTwister(seed)
    data = zeros(T, 2bw + 1, N)
    for j in 1:N, i in max(1, j - bw):min(N, j + bw)
        data[bw + 1 + i - j, j] = rand(rng, T) - T(0.5)
    end
    return GeoDynamo.BandedMatrix{T}(data, bw, N)
end

@testset "GPU Phase 3 — Vector Transform" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 3)
    sht = cfg.sht_config
    nl, nm = cfg.lmax + 1, cfg.mmax + 1

    @testset "sphtor helper CPU method [LOCAL]" begin
        S = zeros(ComplexF64, nl, nm); T = zeros(ComplexF64, nl, nm)
        S[3, 1] = 1.0; T[4, 2] = 0.5 - 0.25im
        vt, vp = GeoDynamo._vector_synth_sphtor(sht, S, T)
        @test size(vt) == (cfg.nlat, cfg.nlon) && size(vp) == (cfg.nlat, cfg.nlon)
        rt, rp = SHTnsKit.synthesis_sphtor(sht, S, T; real_output = true)
        @test vt == rt && vp == rp
        S2, T2 = GeoDynamo._vector_anal_sphtor(sht, vt, vp)
        rS, rT = SHTnsKit.analysis_sphtor(sht, vt, vp)
        @test S2 == rS && T2 == rT
    end

    @testset "gpu_vr_scale! [LOCAL]" begin
        nr = 3
        pr = rand(Float64, nl, nm, nr); pi_ = rand(Float64, nl, nm, nr)
        lfac = Float64[l * (l + 1) for l in 0:cfg.lmax]      # length nl
        rscale = [1.0 / (0.5 + 0.1k)^2 for k in 1:nr]         # length nr (Stage-2 1/r^2)
        vr = zeros(Float64, nl, nm, nr); vi = zeros(Float64, nl, nm, nr)
        GeoDynamo.gpu_vr_scale!(vr, vi, pr, pi_, lfac, rscale)
        refr = similar(vr); refi = similar(vi)
        @inbounds for k in 1:nr, m in 1:nm, l in 1:nl
            f = lfac[l] * rscale[k]
            refr[l,m,k] = pr[l,m,k] * f
            refi[l,m,k] = pi_[l,m,k] * f
        end
        @test vr == refr && vi == refi
    end

    @testset "vector spectral_to_physical [LOCAL]" begin
        nr = 3
        bw = 1
        d1 = _phase3_rand_band(Float64, nr, bw; seed = 31)
        arch = CPU()
        tor = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr)
        pol = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr)
        vr = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
        vθ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
        vφ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
        for k in 1:nr
            pol.data_real[3,1,k] = Float64(k); tor.data_real[4,2,k] = 0.5; tor.data_imag[4,2,k] = -0.25
        end
        lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
        rinv = [1.0/(0.5 + 0.1k) for k in 1:nr]
        rinv2 = rinv .^ 2
        GeoDynamo.gpu_vector_spectral_to_physical!(
            vr, vθ, vφ, tor, pol, cfg, d1.data, lfac, rinv, rinv2, bw)

        S_r = similar(pol.data_real); S_i = similar(pol.data_imag)
        GeoDynamo.gpu_batched_banded_matvec!(S_r, pol.data_real, d1.data, bw)
        GeoDynamo.gpu_batched_banded_matvec!(S_i, pol.data_imag, d1.data, bw)
        ri = reshape(rinv, 1, 1, :)
        @. S_r = S_r * ri
        @. S_i = S_i * ri
        vr_r = similar(pol.data_real); vr_i = similar(pol.data_imag)
        GeoDynamo.gpu_vr_scale!(vr_r, vr_i, pol.data_real, pol.data_imag, lfac, rinv2)

        for k in 1:nr
            S_k = complex.(S_r[:, :, k], S_i[:, :, k])
            T_k = complex.(tor.data_real[:, :, k], tor.data_imag[:, :, k])
            rt, rp = SHTnsKit.synthesis_sphtor(cfg.sht_config, S_k, T_k; real_output = true)
            @test isapprox(vθ.data[:, :, k], rt; atol = 1e-12, rtol = 1e-10)
            @test isapprox(vφ.data[:, :, k], rp; atol = 1e-12, rtol = 1e-10)
            vr_k = complex.(vr_r[:, :, k], vr_i[:, :, k])
            @test isapprox(vr.data[:, :, k],
                SHTnsKit.synthesis(cfg.sht_config, vr_k; real_output = true);
                atol = 1e-12, rtol = 1e-10)
        end
    end

    @testset "vector physical_to_spectral + roundtrip [LOCAL]" begin
        nr = 3
        bw = 1
        d1 = _phase3_rand_band(Float64, nr, bw; seed = 41)
        arch = CPU()
        tor = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr)
        pol = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr)
        vr = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
        vθ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
        vφ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
        for k in 1:nr
            pol.data_real[3,1,k] = Float64(k); tor.data_real[4,2,k] = 0.5; tor.data_imag[4,2,k] = -0.25
        end
        tor0_r = copy(tor.data_real); tor0_i = copy(tor.data_imag)
        pol0_r = copy(pol.data_real); pol0_i = copy(pol.data_imag)
        lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
        rinv = [1.0/(0.5+0.1k) for k in 1:nr]
        rinv2 = rinv .^ 2
        GeoDynamo.gpu_vector_spectral_to_physical!(
            vr, vθ, vφ, tor, pol, cfg, d1.data, lfac, rinv, rinv2, bw)
        fill!(tor.data_real, 0.0); fill!(tor.data_imag, 0.0)
        fill!(pol.data_real, 0.0); fill!(pol.data_imag, 0.0)
        GeoDynamo.gpu_vector_physical_to_spectral!(
            tor, pol, vr, vθ, vφ, cfg, lfac, rinv2)
        @test isapprox(tor.data_real, tor0_r; atol = 1e-10, rtol = 1e-10)
        @test isapprox(tor.data_imag, tor0_i; atol = 1e-10, rtol = 1e-10)
        @test isapprox(pol.data_real, pol0_r; atol = 1e-10, rtol = 1e-10)
        @test isapprox(pol.data_imag, pol0_i; atol = 1e-10, rtol = 1e-10)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-3 gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            nr = 3
            bw = 1
            d1 = _phase3_rand_band(Float64, nr, bw; seed = 51)
            lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
            rinv = [1.0/(0.5+0.1k) for k in 1:nr]
            rinv2 = rinv .^ 2
            mk(arch) = (GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr),
                        GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr),
                        GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr),
                        GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr),
                        GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr))
            ctor, cpol, cvr, cvθ, cvφ = mk(CPU())
            for k in 1:nr
                cpol.data_real[3,1,k] = Float64(k); ctor.data_real[4,2,k] = 0.5; ctor.data_imag[4,2,k] = -0.25
            end
            GeoDynamo.gpu_vector_spectral_to_physical!(
                cvr, cvθ, cvφ, ctor, cpol, cfg, d1.data, lfac, rinv, rinv2, bw)  # CPU ref

            gtor, gpol, gvr, gvθ, gvφ = mk(GPU())
            d!(dst, src) = (copyto!(dst.data_real, src.data_real); copyto!(dst.data_imag, src.data_imag))
            d!(gtor, ctor); d!(gpol, cpol)
            gd1 = GeoDynamo.on_architecture(GPU(), d1.data)
            glfac = GeoDynamo.on_architecture(GPU(), lfac)
            grinv = GeoDynamo.on_architecture(GPU(), rinv)
            grinv2 = GeoDynamo.on_architecture(GPU(), rinv2)
            GeoDynamo.gpu_vector_spectral_to_physical!(
                gvr, gvθ, gvφ, gtor, gpol, cfg, gd1, glfac, grinv, grinv2, bw)  # GPU
            @test gvr.data isa CUDA.CuArray
            @test gvθ.data isa CUDA.CuArray
            @test gvφ.data isa CUDA.CuArray
            @test isapprox(Array(gvr.data), cvr.data; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gvθ.data), cvθ.data; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gvφ.data), cvφ.data; atol = 1e-12, rtol = 1e-10)
        end
    end
end
