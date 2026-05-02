using Test
using MPI

const FINALIZE_MPI_PROG = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Programmatic Boundaries" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping programmatic boundary tests"
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

    @testset "Ylm validation" begin
        ylm = GeoDynamo.bcs.Ylm(2, 1)
        @test ylm.l == 2
        @test ylm.m == 1

        # Negative l
        @test_throws ArgumentError GeoDynamo.bcs.Ylm(-1, 0)
        # |m| > l
        @test_throws ArgumentError GeoDynamo.bcs.Ylm(2, 3)
        @test_throws ArgumentError GeoDynamo.bcs.Ylm(2, -3)
    end

    @testset "_parse_boundary_spec" begin
        vals, bc_type = GeoDynamo.bcs._parse_boundary_spec((:uniform, 1.0), cfg)
        @test size(vals) == (cfg.nlat, cfg.nlon)
        @test all(vals .== 1.0)
        @test bc_type == GeoDynamo.bcs.DIRICHLET

        vals_n, bc_type_n = GeoDynamo.bcs._parse_boundary_spec((:neumann, -2.5), cfg)
        @test all(vals_n .== -2.5)
        @test bc_type_n == GeoDynamo.bcs.NEUMANN

        @test_throws ArgumentError GeoDynamo.bcs._parse_boundary_spec((:invalid, 1.0), cfg)
    end

    @testset "create_programmatic_temperature_boundaries" begin
        bset = GeoDynamo.bcs.create_programmatic_temperature_boundaries(
            (:uniform, 100.0), (:uniform, 250.0), cfg)
        @test bset !== nothing
        @test bset.inner_bc_type == GeoDynamo.bcs.DIRICHLET
        @test bset.outer_bc_type == GeoDynamo.bcs.DIRICHLET
    end

    @testset "create_programmatic_composition_boundaries" begin
        bset = GeoDynamo.bcs.create_programmatic_composition_boundaries(
            (:uniform, 0.0), (:neumann, 0.0), cfg)
        @test bset !== nothing
        @test bset.inner_bc_type == GeoDynamo.bcs.DIRICHLET
        @test bset.outer_bc_type == GeoDynamo.bcs.NEUMANN
    end

    @testset "Shell hybrid temperature boundaries (Tuple+Tuple)" begin
        Shell = GeoDynamo.GeoDynamoShell
        bset = Shell.create_shell_hybrid_temperature_boundaries(
            (:uniform, 1.0), (:uniform, 0.0), cfg)
        @test bset !== nothing
    end

    @testset "Ball hybrid temperature boundaries (Tuple+Tuple)" begin
        Ball = GeoDynamo.GeoDynamoBall
        bset = Ball.create_ball_hybrid_temperature_boundaries(
            (:uniform, 1.0), (:uniform, 0.0), cfg)
        @test bset !== nothing
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_PROG && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
