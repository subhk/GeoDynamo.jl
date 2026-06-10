using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
import SHTnsKit

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
        rscale = [1.0 / (0.5 + 0.1k) for k in 1:nr]           # length nr (solver path 1/r)
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
        rscale = [1.0/(0.5 + 0.1k) for k in 1:nr]
        # Stage-2 gate: GPU vector transforms are not yet ported to the
        # solenoidal P convention and must refuse loudly rather than silently
        # produce old-convention fields. The old-convention parity asserts that
        # lived here return when the port lands (see the double-curl spec).
        @test_throws ErrorException GeoDynamo.gpu_vector_spectral_to_physical!(
            vr, vθ, vφ, tor, pol, cfg, lfac, rscale)
    end

    @testset "vector physical_to_spectral + roundtrip [LOCAL]" begin
        nr = 3
        arch = CPU()
        tor = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr)
        pol = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr)
        vr = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
        vθ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
        vφ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
        for k in 1:nr
            pol.data_real[3,1,k] = Float64(k); tor.data_real[4,2,k] = 0.5; tor.data_imag[4,2,k] = -0.25
        end
        lfac = Float64[l*(l+1) for l in 0:cfg.lmax]; rscale = [1.0/(0.5+0.1k) for k in 1:nr]
        # Stage-2 gate (see the synthesis testset above): both directions refuse
        # until the GPU port adopts the solenoidal P convention.
        @test_throws ErrorException GeoDynamo.gpu_vector_spectral_to_physical!(
            vr, vθ, vφ, tor, pol, cfg, lfac, rscale)
        @test_throws ErrorException GeoDynamo.gpu_vector_physical_to_spectral!(
            tor, pol, vθ, vφ, cfg)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-3 gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            nr = 3
            lfac = Float64[l*(l+1) for l in 0:cfg.lmax]; rscale = [1.0/(0.5+0.1k) for k in 1:nr]
            mk(arch) = (GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr),
                        GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr),
                        GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr),
                        GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr),
                        GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr))
            ctor, cpol, cvr, cvθ, cvφ = mk(CPU())
            for k in 1:nr
                cpol.data_real[3,1,k] = Float64(k); ctor.data_real[4,2,k] = 0.5; ctor.data_imag[4,2,k] = -0.25
            end
            GeoDynamo.gpu_vector_spectral_to_physical!(cvr, cvθ, cvφ, ctor, cpol, cfg, lfac, rscale)  # CPU ref

            gtor, gpol, gvr, gvθ, gvφ = mk(GPU())
            d!(dst, src) = (copyto!(dst.data_real, src.data_real); copyto!(dst.data_imag, src.data_imag))
            d!(gtor, ctor); d!(gpol, cpol)
            glfac = GeoDynamo.on_architecture(GPU(), lfac); grscale = GeoDynamo.on_architecture(GPU(), rscale)
            GeoDynamo.gpu_vector_spectral_to_physical!(gvr, gvθ, gvφ, gtor, gpol, cfg, glfac, grscale)  # GPU
            @test gvr.data isa CUDA.CuArray
            @test gvθ.data isa CUDA.CuArray
            @test gvφ.data isa CUDA.CuArray
            @test isapprox(Array(gvr.data), cvr.data; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gvθ.data), cvθ.data; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gvφ.data), cvφ.data; atol = 1e-12, rtol = 1e-10)
        end
    end
end
