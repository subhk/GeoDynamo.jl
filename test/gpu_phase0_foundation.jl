using Test
using GeoDynamo
import KernelAbstractions

@testset "GPU Phase 0 — Foundation" begin
    @testset "gpu_functional gate [LOCAL]" begin
        @test GeoDynamo.gpu_functional() isa Bool
        # No CUDA GPU in CI / dev machine → false. On a GPU box this flips true.
        @test gpu_synchronize() === nothing
        # All Phase-0 symbols must be exported from the top-level module.
        for s in (:gpu_functional, :gpu_synchronize, :GPUPhysicalField, :GPUSpectralField,
                  :allocate_gpu_physical_field, :allocate_gpu_spectral_field,
                  :field_to_host, :field_to_device)
            @test Base.isexported(GeoDynamo, s)
        end
    end

    @testset "GPU() constructor [LOCAL/GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            # Without a functional GPU, GPU() must error clearly, not silently fall back.
            @test_throws ErrorException GPU()
        else
            a = GPU()
            @test a isa GPU
            @test get_backend(a) !== KernelAbstractions.CPU()
        end
    end

    @testset "GPU physical field allocation [GPU-BOX]" begin
        cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 16, nlon = 32, nr = 4)
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            f = GeoDynamo.allocate_gpu_physical_field(Float64, GPU(), cfg, 4)
            @test f isa GeoDynamo.GPUPhysicalField
            @test size(f.data) == (cfg.nlat, cfg.nlon, 4)
            @test f.data isa CUDA.CuArray          # CUDA in scope on the GPU box test run
            @test all(Array(f.data) .== 0)         # zero-initialised
        end
    end

    @testset "GPU spectral field allocation [GPU-BOX]" begin
        cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 16, nlon = 32, nr = 4)
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            f = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, GPU(), cfg, 4)
            @test f isa GeoDynamo.GPUSpectralField
            @test size(f.data_real) == (cfg.nlm, 4)
            @test size(f.data_imag) == (cfg.nlm, 4)
            @test f.data_real isa CUDA.CuArray
            @test all(Array(f.data_real) .== 0) && all(Array(f.data_imag) .== 0)
        end
    end

    @testset "host<->device field roundtrip (Phase-0 gate) [GPU-BOX]" begin
        cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 16, nlon = 32, nr = 4)
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            # physical
            host_phys = rand(Float64, cfg.nlat, cfg.nlon, 4)
            gf = GeoDynamo.field_to_device(GPU(), host_phys, cfg, 4)            # host -> device
            back = GeoDynamo.field_to_host(gf)                                   # device -> host
            @test back.data == host_phys                                        # BIT-IDENTICAL

            # spectral
            hr = rand(Float64, cfg.nlm, 4); hi = rand(Float64, cfg.nlm, 4)
            gs = GeoDynamo.field_to_device(GPU(), (hr, hi), cfg, 4)
            bs = GeoDynamo.field_to_host(gs)
            @test bs.data_real == hr && bs.data_imag == hi
        end
    end
end
