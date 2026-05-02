using Test

@testset "User API" begin
    @testset "Grid types" begin
        g = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU(); lmax=8, nr=16)
        @test g.lmax == 8
        @test g.nr == 16
        @test g.arch isa GeoDynamo.CPU
        @test g.mmax == 8

        gb = GeoDynamo.SphericalBallGrid(GeoDynamo.CPU(); lmax=4, nr=8)
        @test gb.lmax == 4
        @test gb.nr == 8
    end

    @testset "Grid constructors default to CPU architecture" begin
        shell = GeoDynamo.SphericalShellGrid(lmax=8, nr=16)
        @test shell.arch isa GeoDynamo.CPU
        @test shell.lmax == 8
        @test shell.nr == 16

        ball = GeoDynamo.SphericalBallGrid(lmax=4, nr=8)
        @test ball.arch isa GeoDynamo.CPU
        @test ball.lmax == 4
        @test ball.nr == 8
    end

    @testset "BoundaryConditions codes" begin
        ns = GeoDynamo.NoSlip()
        sf = GeoDynamo.StressFree()
        ft = GeoDynamo.FixedTemperature(0.0)
        ff = GeoDynamo.FixedFlux(1.0)

        @test GeoDynamo._velocity_bc_code(GeoDynamo.BoundaryConditions(inner=ns, outer=ns)) == 1
        @test GeoDynamo._velocity_bc_code(GeoDynamo.BoundaryConditions(inner=ns, outer=sf)) == 2
        @test GeoDynamo._velocity_bc_code(GeoDynamo.BoundaryConditions(inner=sf, outer=ns)) == 3
        @test GeoDynamo._velocity_bc_code(GeoDynamo.BoundaryConditions(inner=sf, outer=sf)) == 4
        @test GeoDynamo._thermal_bc_code(GeoDynamo.BoundaryConditions(inner=ff, outer=ft)) == 3
    end

    @testset "Schedule types" begin
        ctx = GeoDynamo._ScheduleContext(1.0, 100, 5.0)
        @test GeoDynamo.should_fire(GeoDynamo.IterationInterval(100), ctx) == true
        @test GeoDynamo.should_fire(GeoDynamo.IterationInterval(50),  ctx) == true
        @test GeoDynamo.should_fire(GeoDynamo.IterationInterval(99),  ctx) == false
    end

    @testset "IC types construct" begin
        @test GeoDynamo.RandomPerturbation(amplitude=1e-3, lmax=8) !== nothing
        @test GeoDynamo.ZeroIC() !== nothing
        @test GeoDynamo.FileIC("path.jld2") !== nothing
        @test GeoDynamo.AnalyticIC(:dipole) !== nothing
        @test GeoDynamo.AnalyticIC(:dipole; amplitude=2.0) !== nothing
    end

    @testset "Public exports present" begin
        for sym in (:SphericalShellGrid, :SphericalBallGrid, :CPU, :GPU,
                    :NoSlip, :StressFree, :FixedTemperature, :FixedFlux,
                    :InsulatingMagnetic, :ConductingMagnetic, :BoundaryConditions,
                    :GeodynamoModel, :Simulation,
                    :TimeInterval, :IterationInterval, :WallTimeInterval,
                    :FieldWriter, :CheckpointWriter,
                    :RandomPerturbation, :AnalyticIC, :FileIC, :ZeroIC,
                    :set_initial_condition!, :Callback,
                    :EnergyDiagnostics, :SolenoidalMonitor, :SimulationProgress, :HealthCheck)
            @test Base.isexported(GeoDynamo, sym)
        end
    end
end
