using Test

@testset "Oceananigans-style API" begin

    @testset "stop_iteration rename" begin
        p = GeoDynamo.SolverParameters(stop_iteration = 42)
        @test p.stop_iteration == 42
        @test :stop_iteration in fieldnames(GeoDynamo.SolverParameters)
        @test !(:max_steps in fieldnames(GeoDynamo.SolverParameters))
    end

    @testset "legacy max_steps param key is mapped with a warning" begin
        mktempdir() do dir
            path = joinpath(dir, "legacy.jl")
            write(path, "nr = 16\nlmax = 4\nmax_steps = 7\n")
            local params
            @test_logs (:warn, r"max_steps.*deprecated") match_mode=:any begin
                params = GeoDynamo.load_parameters_from_file(path)
            end
            @test params.stop_iteration == 7
        end
    end

    @testset "Clock attaches to model" begin
        using MPI
        if !MPI.Initialized(); MPI.Init(); end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax=4, mmax=4, nlat=12, nlon=16, nr=16, nr_inner=4)
        model = GeoDynamo.GeodynamoModel(grid;
            Ek=1e-2, Ra=1e4, include_magnetic=false, include_composition=false)
        @test model.clock isa GeoDynamo.Clock
        @test model.clock.time == 0.0
        @test model.clock.iteration == 0
    end

end
