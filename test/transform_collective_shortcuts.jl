using Test
using MPI

const FINALIZE_MPI = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Transform Collective Shortcuts" begin
    if MPI.Finalized()
        @warn "MPI already finalized before transform collective shortcut tests; skipping"
        return
    end

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = GeoDynamo.get_comm()

    @test GeoDynamo.get_nprocs() == 1

    function build_cfg_and_domain()
        cfg = GeoDynamo.create_shtnskit_config(lmax=4, mmax=4, nlat=12, nlon=16, nr=4)
        dom = GeoDynamo.create_radial_domain(4)
        return cfg, dom
    end

    @testset "Physical-grid collective guard bypasses reducer when grid is local" begin
        @test isdefined(GeoDynamo, :physical_grid_is_local)
        @test isdefined(GeoDynamo, :maybe_allreduce_matrix!)

        full_axes = (Base.OneTo(12), Base.OneTo(16))
        @test GeoDynamo.physical_grid_is_local((12, 16), full_axes, 12, 16)
        @test GeoDynamo.physical_grid_is_local((12, 16), nothing, 12, 16)
        @test !GeoDynamo.physical_grid_is_local((6, 16), nothing, 12, 16)

        buffer = reshape(collect(1.0:12.0), 3, 4)
        result = GeoDynamo.maybe_allreduce_matrix!(copy(buffer), false, nothing; reducer=(args...)->error("collective should not run"))
        @test result == buffer
    end

    @testset "Scalar coefficient extraction skips gathered buffer when local" begin
        cfg, dom = build_cfg_and_domain()
        spec = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)

        real_data = parent(spec.data_real)
        imag_data = parent(spec.data_imag)
        for idx in eachindex(IndexLinear(), view(real_data, :, 1, 1))
            real_data[idx, 1, 1] = idx + 0.25
            imag_data[idx, 1, 1] = -idx - 0.5
        end

        expected = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        GeoDynamo.extract_coefficients_for_shtnskit!(expected, real_data, imag_data, 1, cfg)

        coeffs = GeoDynamo.extract_coefficients_for_shtnskit(real_data, imag_data, 1, cfg)
        @test coeffs == expected
        @test cfg.__buffers.coeffs_buffer !== nothing
        @test cfg.__buffers.coeffs_buffer_gathered === nothing  # gathered path not taken when local
    end

    @testset "Paired coefficient extraction skips gathered buffers when local" begin
        cfg, dom = build_cfg_and_domain()
        spec1 = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
        spec2 = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)

        real1 = parent(spec1.data_real)
        imag1 = parent(spec1.data_imag)
        real2 = parent(spec2.data_real)
        imag2 = parent(spec2.data_imag)

        for idx in eachindex(IndexLinear(), view(real1, :, 1, 1))
            real1[idx, 1, 1] = 2idx
            imag1[idx, 1, 1] = -2idx
            real2[idx, 1, 1] = 3idx + 0.75
            imag2[idx, 1, 1] = -3idx - 0.25
        end

        expected1 = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        expected2 = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        GeoDynamo.extract_coefficients_for_shtnskit!(expected1, real1, imag1, 1, cfg)
        GeoDynamo.extract_coefficients_for_shtnskit!(expected2, real2, imag2, 1, cfg)

        coeffs1, coeffs2 = GeoDynamo.extract_coefficients_pair_for_shtnskit(
            real1,
            imag1,
            real2,
            imag2,
            1,
            cfg,
        )

        @test coeffs1 == expected1
        @test coeffs2 == expected2
        @test cfg.__buffers.coeffs_buffer_pair1 !== nothing
        @test cfg.__buffers.coeffs_buffer_pair2 !== nothing
        @test cfg.__buffers.coeffs_gathered_pair1 === nothing  # gathered path not taken when local
        @test cfg.__buffers.coeffs_gathered_pair2 === nothing  # gathered path not taken when local
    end

    if MPI.Initialized()
        MPI.Barrier(comm)
        if FINALIZE_MPI && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
