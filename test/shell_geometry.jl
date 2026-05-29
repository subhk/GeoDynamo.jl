using Test
using MPI

const Shell = GeoDynamo.GeoDynamoShell
const FINALIZE_MPI_SHELL = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Shell Geometry Module" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping shell geometry tests"
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

    @testset "Shell radial domain" begin
        dom = Shell.create_shell_radial_domain(nr)
        @test dom.N == nr
        # Shell has both inner and outer boundary > 0
        @test dom.r[1, 4] > 0.0
        @test dom.r[nr, 4] > dom.r[1, 4]
    end

    @testset "Solver inner-core domain" begin
        params = GeoDynamo.SolverParameters(
            geometry=:shell,
            nr=16,
            nr_inner=8,
            lmax=4,
            mmax=4,
            nlat=12,
            nlon=16,
            radius_ratio=0.35,
        )
        _, inner = GeoDynamo.create_radial_domains(params)
        @test inner !== nothing
        @test inner.r[1, 4] ≈ 0.0
        @test inner.r[end, 4] ≈ params.radius_ratio / (1 - params.radius_ratio)
    end

    @testset "Shell spectral field" begin
        dom = Shell.create_shell_radial_domain(nr)
        spec = Shell.create_shell_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
        @test spec.nlm == cfg.nlm
        @test all(parent(spec.data_real) .== 0.0)
    end

    @testset "Shell physical field" begin
        dom = Shell.create_shell_radial_domain(nr)
        phys = Shell.create_shell_physical_field(Float64, cfg, dom, cfg.pencils.r)
        @test phys.nlat == cfg.nlat
        @test phys.nlon == cfg.nlon
    end

    @testset "Shell vector field" begin
        dom = Shell.create_shell_radial_domain(nr)
        vec = Shell.create_shell_vector_field(Float64, cfg, dom,
            (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))
        @test all(parent(vec.r_component.data) .== 0.0)
    end

    @testset "Shell temperature field" begin
        temp = Shell.create_shell_temperature_field(Float64, cfg; nr=nr)
        @test temp !== nothing
        @test temp.config === cfg
        @test length(temp.l_factors) == cfg.nlm
    end

    @testset "Shell composition field" begin
        comp = Shell.create_shell_composition_field(Float64, cfg; nr=nr)
        @test comp !== nothing
        @test comp.config === cfg
    end

    @testset "Shell velocity fields" begin
        vel = Shell.create_shell_velocity_fields(Float64, cfg; nr=nr)
        @test vel !== nothing
        @test all(parent(vel.𝒯.data_real) .== 0.0)
        @test all(parent(vel.𝒫.data_real) .== 0.0)
    end

    @testset "Shell magnetic fields" begin
        mag = Shell.create_shell_magnetic_fields(Float64, cfg; nr_oc=nr, nr_ic=nr)
        @test mag !== nothing
        @test all(parent(mag.𝒯.data_real) .== 0.0)
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_SHELL && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
