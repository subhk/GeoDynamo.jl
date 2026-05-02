using Test
using MPI

const FINALIZE_MPI_FIELDS = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Field Construction" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping field tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    lmax = 4; mmax = 4
    nlat = max(lmax + 2, 10)
    nlon = max(2lmax + 1, 16)
    nr = 6

    cfg = GeoDynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = GeoDynamo.create_radial_domain(nr)

    @testset "SHTnsKitConfig properties" begin
        @test cfg.lmax == lmax
        @test cfg.mmax == mmax
        @test cfg.nlat >= lmax + 2
        @test cfg.nlon >= 2lmax + 1
        @test cfg.nlm > 0
        @test length(cfg.l_values) == cfg.nlm
        @test length(cfg.m_values) == cfg.nlm
        @test all(cfg.l_values .>= 0)
        @test all(cfg.m_values .>= 0)
        @test maximum(cfg.l_values) == lmax
        @test maximum(cfg.m_values) == mmax
    end

    @testset "Spectral field creation" begin
        spec = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
        @test spec.nlm == cfg.nlm
        # Initially zero
        @test all(parent(spec.data_real) .== 0.0)
        @test all(parent(spec.data_imag) .== 0.0)
        # Default BC type = DIRICHLET (1) for all modes
        @test all(spec.bc_type_inner .== 1)
        @test all(spec.bc_type_outer .== 1)
        # Boundary values initialized to zero
        @test all(spec.boundary_values .== 0.0)
    end

    @testset "Physical field creation" begin
        phys = GeoDynamo.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)
        @test phys.nlat == cfg.nlat
        @test phys.nlon == cfg.nlon
        @test all(parent(phys.data) .== 0.0)
    end

    @testset "Vector field creation" begin
        vec = GeoDynamo.create_shtns_vector_field(Float64, cfg, dom,
            (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))
        @test all(parent(vec.r_component.data) .== 0.0)
        @test all(parent(vec.θ_component.data) .== 0.0)
        @test all(parent(vec.φ_component.data) .== 0.0)
    end

    @testset "Float32 spectral field" begin
        spec32 = GeoDynamo.create_shtns_spectral_field(Float32, cfg, dom, cfg.pencils.spec)
        @test eltype(parent(spec32.data_real)) == Float32
        @test eltype(parent(spec32.data_imag)) == Float32
    end

    @testset "Temperature field creation" begin
        temp = GeoDynamo.create_shtns_temperature_field(Float64, cfg, dom)
        @test temp.config === cfg
        @test length(temp.internal_sources) == nr
        @test size(temp.boundary_values) == (2, cfg.nlm)
        @test length(temp.ℓ_factors) == cfg.nlm
        # ℓ_factors should be l*(l+1)
        for (i, l) in enumerate(cfg.l_values)
            @test temp.ℓ_factors[i] ≈ l * (l + 1)
        end
    end

    @testset "Velocity workspace creation" begin
        ws = GeoDynamo.create_velocity_workspace(Float64, nr, 2)
        @test length(ws.Pᴾ_profile_real) == 2  # nthreads=2
        @test length(ws.Pᴾ_profile_real[1]) == nr

        # Setting and getting workspace
        GeoDynamo.set_velocity_workspace!(ws)
        ws_back = GeoDynamo.get_velocity_workspace(Float64)
        @test ws_back === ws

        # Type mismatch returns nothing
        ws_f32 = GeoDynamo.get_velocity_workspace(Float32)
        @test ws_f32 === nothing

        # Reset to nothing
        GeoDynamo.set_velocity_workspace!(nothing)
        @test GeoDynamo.get_velocity_workspace(Float64) === nothing
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_FIELDS && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
