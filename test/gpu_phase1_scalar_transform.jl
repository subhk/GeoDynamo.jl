using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
import SHTnsKit

@testset "GPU Phase 1 — Scalar Transform" begin
    nr = 3   # radial levels used throughout; SHTnsKitConfig does not store nr
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = nr)
    sht = cfg.sht_config

    @testset "transform helper CPU method [LOCAL]" begin
        alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        alm[3, 1] = 1.0 + 0.0im                       # a single (l=2,m=0) mode
        f = GeoDynamo._scalar_synth(sht, alm)         # -> (nlat,nlon) real
        @test size(f) == (cfg.nlat, cfg.nlon)
        @test eltype(f) <: Real
        # matches a direct SHTnsKit synthesis exactly (helper is a thin pass-through on CPU)
        @test f == SHTnsKit.synthesis(sht, alm; real_output = true)
        alm2 = GeoDynamo._scalar_anal(sht, f)         # back to coeffs
        @test size(alm2) == (cfg.lmax + 1, cfg.mmax + 1)
        @test alm2 == SHTnsKit.analysis(sht, f)
    end

    @testset "spectral_to_physical [LOCAL]" begin
        arch = CPU()                                            # Array-backed GPU fields → CPU transform path
        spec = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr)
        phys = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
        # band-limited spectral content per level
        for k in 1:nr
            spec.data_real[3, 1, k] = Float64(k)               # (l=2,m=0) real
            spec.data_real[4, 2, k] = 0.5                       # (l=3,m=1) real
            spec.data_imag[4, 2, k] = -0.25                     # (l=3,m=1) imag
        end
        GeoDynamo.gpu_scalar_spectral_to_physical!(phys, spec, cfg)
        @test size(phys.data) == (cfg.nlat, cfg.nlon, nr)
        # each level must equal a direct synthesis of that level's complex coeffs
        for k in 1:nr
            alm_k = complex.(spec.data_real[:, :, k], spec.data_imag[:, :, k])
            @test phys.data[:, :, k] == SHTnsKit.synthesis(cfg.sht_config, alm_k; real_output = true)
        end
    end

    @testset "physical_to_spectral + roundtrip [LOCAL]" begin
        arch = CPU()
        spec = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, nr)
        phys = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
        # band-limited start (so analysis∘synthesis is the identity to ~1e-12)
        for k in 1:nr
            spec.data_real[3, 1, k] = Float64(k)
            spec.data_real[4, 2, k] = 0.5
            spec.data_imag[4, 2, k] = -0.25
        end
        spec0_r = copy(spec.data_real); spec0_i = copy(spec.data_imag)

        GeoDynamo.gpu_scalar_spectral_to_physical!(phys, spec, cfg)
        # zero spec, then analyze physical back into it
        fill!(spec.data_real, 0.0); fill!(spec.data_imag, 0.0)
        GeoDynamo.gpu_scalar_physical_to_spectral!(spec, phys, cfg)

        @test size(spec.data_real) == (cfg.lmax + 1, cfg.mmax + 1, nr)
        # roundtrip: recovered coeffs ≈ original (only the populated band; high-l slots stay ~0)
        @test isapprox(spec.data_real, spec0_r; atol = 1e-10)
        @test isapprox(spec.data_imag, spec0_i; atol = 1e-10)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-1 gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            # Build identical content on CPU (Array) and GPU (CuArray) fields.
            cspec = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, CPU(), cfg, nr)
            cphys = GeoDynamo.allocate_gpu_physical_field(Float64, CPU(), cfg, nr)
            for k in 1:nr
                cspec.data_real[3, 1, k] = Float64(k)
                cspec.data_real[4, 2, k] = 0.5
                cspec.data_imag[4, 2, k] = -0.25
            end
            GeoDynamo.gpu_scalar_spectral_to_physical!(cphys, cspec, cfg)   # CPU reference

            gspec = GeoDynamo.field_to_device(GPU(), (copy(cspec.data_real), copy(cspec.data_imag)), cfg, nr)
            gphys = GeoDynamo.allocate_gpu_physical_field(Float64, GPU(), cfg, nr)
            GeoDynamo.gpu_scalar_spectral_to_physical!(gphys, gspec, cfg)   # GPU path
            @test gphys.data isa CUDA.CuArray
            # GPU ≈ CPU synthesis (reduction reorder → tolerance, not bitwise)
            @test isapprox(Array(gphys.data), cphys.data; atol = 1e-12, rtol = 1e-10)

            # analysis parity + GPU roundtrip
            gspec2 = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, GPU(), cfg, nr)
            GeoDynamo.gpu_scalar_physical_to_spectral!(gspec2, gphys, cfg)
            @test isapprox(Array(gspec2.data_real), cspec.data_real; atol = 1e-10)
            @test isapprox(Array(gspec2.data_imag), cspec.data_imag; atol = 1e-10)
        end
    end
end
