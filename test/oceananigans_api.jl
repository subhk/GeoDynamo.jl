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
        if !MPI.Initialized()
            ;
            MPI.Init();
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid;
            Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false)
        @test model.clock isa GeoDynamo.Clock
        @test model.clock.time == 0.0
        @test model.clock.iteration == 0
    end

    @testset "time_step! advances one step" begin
        using MPI
        if !MPI.Initialized()
            ;
            MPI.Init();
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid;
            Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false)
        GeoDynamo.time_step!(model, 1e-4)
        @test model.clock.iteration == 1
        @test model.clock.last_dt == 1e-4
        @test isfinite(model.clock.time)
    end

    @testset "time_step! with a new dt rebuilds implicit matrices" begin
        using MPI
        if !MPI.Initialized()
            ;
            MPI.Init();
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid;
            Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false)
        old_id = objectid(model.state.implicit_matrices)
        GeoDynamo.time_step!(model, 1e-4)   # differs from the model's default timestep
        @test model.state.parameters.timestep == 1e-4
        @test model.state.runtime.timestep_state.dt == 1e-4
        @test objectid(model.state.implicit_matrices) != old_id   # store was rebuilt
        @test all(isfinite, parent(model.state.fields.temperature.spectral.data_real))
    end

    @testset "Simulation rebuilds implicit matrices for its dt" begin
        using MPI
        if !MPI.Initialized()
            ;
            MPI.Init();
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid;
            Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false)
        old_id = objectid(model.state.implicit_matrices)
        default_dt = model.state.parameters.timestep
        sim = GeoDynamo.Simulation(model; dt = 2e-4, stop_iteration = 1)
        @test model.state.parameters.timestep == 2e-4
        @test model.state.runtime.timestep_state.dt == 2e-4
        @test 2e-4 != default_dt          # sanity: we actually changed it
        @test objectid(model.state.implicit_matrices) != old_id
    end

    @testset "Simulation accepts Δt alias and Real stop_time" begin
        using MPI
        if !MPI.Initialized()
            MPI.Init()
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid;
            Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false)
        # Δt is a convenience alias for the canonical dt (docs / Oceananigans use Δt).
        s1 = GeoDynamo.Simulation(model; Δt = 1e-4, stop_iteration = 1)
        @test s1 isa GeoDynamo.Simulation
        @test s1.dt == 1e-4
        # An integer stop_time must convert (was stop_time::Float64 -> TypeError).
        s2 = GeoDynamo.Simulation(model; dt = 1e-4, stop_time = 1, stop_iteration = 1)
        @test s2.stop_time === 1.0
        # Passing both dt and Δt is ambiguous -> error.
        @test_throws ArgumentError GeoDynamo.Simulation(model; dt = 1e-4, Δt = 2e-4)
        # Neither dt nor Δt -> error (a timestep is required).
        @test_throws Exception GeoDynamo.Simulation(model)
    end

    @testset "OrderedDict writers/callbacks + add_callback!" begin
        using MPI
        if !MPI.Initialized()
            ;
            MPI.Init();
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid;
            Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false)
        sim = GeoDynamo.Simulation(model; dt = 1e-4, stop_iteration = 2)
        @test sim.callbacks isa GeoDynamo.OrderedDict{Symbol, <:Any}
        @test sim.output_writers isa GeoDynamo.OrderedDict{Symbol, <:Any}

        fired = Ref(0)
        GeoDynamo.add_callback!(sim, _ -> (fired[] += 1);
            schedule = GeoDynamo.IterationInterval(1), name = :counter)
        @test haskey(sim.callbacks, :counter)

        GeoDynamo.run!(sim)
        @test fired[] == 2
    end

    @testset "set! applies ICs by field kwarg" begin
        using MPI
        if !MPI.Initialized()
            ;
            MPI.Init();
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 6, mmax = 6, nlat = 16, nlon = 24, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid;
            Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false)
        returned = GeoDynamo.set!(model;
            temperature = GeoDynamo.RandomPerturbation(amplitude = 1e-3, lmax = 4))
        @test returned === model
        temp = parent(model.state.fields.temperature.spectral.data_real)
        @test any(temp .!= 0.0)
    end

    @testset "fields / prognostic_fields accessors" begin
        using MPI
        if !MPI.Initialized()
            ;
            MPI.Init();
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid;
            Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false)
        f = GeoDynamo.fields(model)
        @test f isa NamedTuple
        @test keys(f) == (:velocity, :temperature, :magnetic, :composition)
        @test f.magnetic === nothing
        pf = GeoDynamo.prognostic_fields(model)
        @test :magnetic ∉ keys(pf)
        @test :velocity ∈ keys(pf)
        @test :temperature ∈ keys(pf)
    end

    @testset "summary one-liners" begin
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU(); lmax = 4, nr = 16)
        s = summary(grid)
        @test occursin("SphericalShellGrid", s)
        @test occursin("lmax=4", s)
        c = GeoDynamo.Clock(; time = 1.5, iteration = 3)
        @test occursin("iteration = 3", summary(c))
    end

    @testset "prettytime / prettysummary" begin
        @test GeoDynamo.prettytime(0) == "0 seconds"
        @test GeoDynamo.prettytime(1) == "1 second"
        @test GeoDynamo.prettytime(1e-9) == "1 ns"
        @test GeoDynamo.prettytime(2.5e-6) == "2.500 μs"
        @test GeoDynamo.prettytime(0.012345) == "12.345 ms"
        @test GeoDynamo.prettytime(2.341) == "2.341 seconds"
        @test GeoDynamo.prettytime(90) == "1.500 minutes"
        @test GeoDynamo.prettytime(3600) == "1 hour"
        @test GeoDynamo.prettytime(129600) == "1.500 days"
        @test GeoDynamo.prettytime(Inf) == "Inf days"
        @test GeoDynamo.prettytime(NaN) == "NaN"
        @test GeoDynamo.prettytime(-90) == "-1.500 minutes"

        @test GeoDynamo.prettysummary(0) == "0"
        @test GeoDynamo.prettysummary(1) == "1"
        @test GeoDynamo.prettysummary(0.0) == "0"
        @test GeoDynamo.prettysummary(1.0) == "1"
        @test GeoDynamo.prettysummary(1e-4) == "0.0001"
        @test GeoDynamo.prettysummary(1.5e-7) == "1.5e-7"
        @test GeoDynamo.prettysummary(0.35) == "0.35"
        @test GeoDynamo.prettysummary(Inf) == "Inf"
        @test GeoDynamo.prettysummary(typemax(Int)) == "Inf"
        @test GeoDynamo.prettysummary(NaN) == "NaN"
        @test GeoDynamo.prettysummary(-Inf) == "-Inf"
        @test GeoDynamo.prettysummary(123456.0) == "123456"
    end

    @testset "SpecifiedTimes schedule" begin
        s = GeoDynamo.SpecifiedTimes(0.5, 0.1, 0.1, 1.0)   # unsorted + dup on purpose
        @test s.times == [0.1, 0.5, 1.0]
        ctx(t) = GeoDynamo._ScheduleContext(t, 0, 0.0)
        @test GeoDynamo.should_fire(s, ctx(0.05)) == false
        @test GeoDynamo.should_fire(s, ctx(0.1)) == true     # reaches 0.1
        @test GeoDynamo.should_fire(s, ctx(0.2)) == false    # 0.1 already fired
        @test GeoDynamo.should_fire(s, ctx(0.7)) == true     # passed 0.5
        @test GeoDynamo.should_fire(s, ctx(2.0)) == true     # passed 1.0
        @test GeoDynamo.should_fire(s, ctx(3.0)) == false    # exhausted
    end

    @testset "model field properties" begin
        using MPI
        if !MPI.Initialized()
            MPI.Init()
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4)   # no magnetic/composition
        @test model.velocity === model.state.fields.velocity
        @test model.temperature === model.state.fields.temperature
        @test model.magnetic === nothing
        @test model.composition === nothing
        @test :velocity in propertynames(model)
        @test :magnetic in propertynames(model)
    end

    @testset "Δt canonical property" begin
        using MPI
        if !MPI.Initialized()
            MPI.Init()
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4)
        sim = GeoDynamo.Simulation(model, Δt = 1e-4, stop_iteration = 1)
        @test sim.Δt == 1e-4
        @test sim.dt == 1e-4                       # ASCII field still readable
        sim.Δt = 2e-4
        @test sim.dt == 2e-4
        @test :Δt in propertynames(sim)
        @test model.clock.last_Δt == model.clock.last_dt
        @test :last_Δt in propertynames(model.clock)
    end

    @testset "Oceananigans BC names" begin
        @test GeoDynamo.ValueBoundaryCondition(0.5) == GeoDynamo.FixedTemperature(0.5)
        @test GeoDynamo.FluxBoundaryCondition(1.0) == GeoDynamo.FixedFlux(1.0)
        bcs = GeoDynamo.FieldBoundaryConditions(inner = GeoDynamo.ValueBoundaryCondition(1.0),
                                                outer = GeoDynamo.FluxBoundaryCondition(0.0))
        @test bcs isa GeoDynamo.BoundaryConditions          # alias identity
        @test bcs.inner == GeoDynamo.FixedTemperature(1.0)
        @test sprint(show, GeoDynamo.ValueBoundaryCondition(0.5)) == "ValueBoundaryCondition(0.5)"
        @test sprint(show, bcs) ==
            "FieldBoundaryConditions(inner = ValueBoundaryCondition(1.0), outer = FluxBoundaryCondition(0.0))"
    end

    @testset "boundary_conditions NamedTuple kwarg" begin
        using MPI
        if !MPI.Initialized()
            MPI.Init()
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        t_bcs = GeoDynamo.FieldBoundaryConditions(inner = GeoDynamo.ValueBoundaryCondition(1.0),
                                                  outer = GeoDynamo.ValueBoundaryCondition(0.0))
        model = GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4,
            boundary_conditions = (temperature = t_bcs,))
        @test model.state.parameters.temperature_bcs == t_bcs
        # same field both ways → error
        @test_throws ArgumentError GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4,
            boundary_conditions = (temperature = t_bcs,),
            temperature_bcs = t_bcs)
        # unknown field name → error
        @test_throws ArgumentError GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4,
            boundary_conditions = (pressure = t_bcs,))
    end

    @testset "set! with number / function / array" begin
        using MPI
        if !MPI.Initialized()
            MPI.Init()
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 8, mmax = 8, nlat = 16, nlon = 32, nr = 8, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4)
        phys = GeoDynamo.get_main_physical_field(model.temperature)

        # number → uniform: synthesize back and check the mean
        GeoDynamo.set!(model; temperature = 0.5)
        GeoDynamo.shtnskit_spectral_to_physical!(model.temperature.spectral, phys)
        pd = parent(phys.data)
        @test isapprox(sum(pd) / length(pd), 0.5; atol = 1e-6)

        # function of (r, θ, φ): radial-only profile
        GeoDynamo.set!(model; temperature = (r, θ, φ) -> r)
        GeoDynamo.shtnskit_spectral_to_physical!(model.temperature.spectral, phys)
        domain = model.state.backend.outer_core_domain
        rvals = [domain.r[k, 4] for k in 1:domain.N]
        pd = parent(phys.data)
        @test all(isapprox(pd[1, 1, k], rvals[k]; atol = 1e-3) for k in axes(pd, 3))

        # array of local physical size
        arr = fill(0.25, size(parent(phys.data)))
        GeoDynamo.set!(model; temperature = arr)
        GeoDynamo.shtnskit_spectral_to_physical!(model.temperature.spectral, phys)
        @test isapprox(parent(phys.data)[1, 1, 1], 0.25; atol = 1e-6)

        # wrong-size array → error
        @test_throws ArgumentError GeoDynamo.set!(model; temperature = zeros(2, 2, 2))

        # vector fields reject direct values with a clear error
        @test_throws ArgumentError GeoDynamo.set!(model; velocity = (r, θ, φ) -> 1.0)
    end

    @testset "default callbacks + sim.running" begin
        using MPI
        if !MPI.Initialized()
            MPI.Init()
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4)
        sim = GeoDynamo.Simulation(model, Δt = 1e-4, stop_iteration = 3)
        @test collect(keys(sim.callbacks))[1:4] ==
            [:stop_time_exceeded, :stop_iteration_exceeded,
             :wall_time_limit_exceeded, :nan_checker]
        @test sim.running == false
        GeoDynamo.run!(sim)
        @test sim.running == false
        @test model.clock.iteration == 3          # stopped BY the callback

        model2 = GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4)
        sim2 = GeoDynamo.Simulation(model2, Δt = 1e-4, stop_time = 2.5e-4)
        GeoDynamo.run!(sim2)
        @test model2.clock.time >= 2.5e-4
        @test sim2.running == false

        model3 = GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4)
        fired = Ref(0)
        sim3 = GeoDynamo.Simulation(model3, Δt = 1e-4, stop_iteration = 2,
            callbacks = (progress = GeoDynamo.Callback(s -> fired[] += 1,
                schedule = GeoDynamo.IterationInterval(1)),))
        GeoDynamo.run!(sim3)
        @test fired[] >= 2
        @test collect(keys(sim3.callbacks))[end] == :progress

        # a completed simulation must not advance on a second run!
        iter_after = model3.clock.iteration
        GeoDynamo.run!(sim3)
        @test model3.clock.iteration == iter_after
    end

    @testset "Oceananigans-style printing" begin
        using MPI
        if !MPI.Initialized()
            MPI.Init()
        end
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4)
        sim = GeoDynamo.Simulation(model, Δt = 1e-4, stop_time = 1.0)

        msum = summary(model)
        @test occursin("GeodynamoModel{CPU, Float64}", msum)
        @test occursin("(time = 0, iteration = 0)", msum)

        mshow = sprint(show, MIME"text/plain"(), model)
        @test occursin("├── grid: SphericalShellGrid(CPU, lmax=4, mmax=4, nr=16)", mshow)
        @test occursin("├── timestepper:", mshow)
        @test occursin("└── active: magnetic=false, composition=false", mshow)

        sshow = sprint(show, MIME"text/plain"(), sim)
        @test occursin("Simulation of GeodynamoModel{CPU, Float64}", sshow)
        @test occursin("├── Next time step: 0.0001", sshow)
        @test occursin("├── Elapsed wall time: 0 seconds", sshow)
        @test occursin("├── Stop time: 1", sshow)
        @test occursin("├── Stop iteration: Inf", sshow)
        @test occursin("├── Wall time limit: Inf", sshow)
        @test occursin("├── Callbacks: OrderedDict with 4 entries:", sshow)
        @test occursin("│   ├── stop_time_exceeded => Callback of stop_time_exceeded on IterationInterval(1)", sshow)
        @test occursin("│   └── nan_checker => Callback of nan_checker on IterationInterval(100)", sshow)
        @test occursin("└── Output writers: OrderedDict with no entries", sshow)

        csum = summary(model.clock)
        @test occursin("Clock(time = 0, iteration = 0, last_Δt = 0)", csum)

        @test summary(GeoDynamo.IterationInterval(10)) == "IterationInterval(10)"
        @test summary(GeoDynamo.TimeInterval(0.1)) == "TimeInterval(0.1)"
        @test summary(GeoDynamo.SpecifiedTimes(0.1, 0.5)) == "SpecifiedTimes(0.1, 0.5)"
        cb = GeoDynamo.Callback(s -> nothing, schedule = GeoDynamo.IterationInterval(2))
        @test startswith(summary(cb), "Callback of ")
        @test endswith(summary(cb), "on IterationInterval(2)")
    end
end
