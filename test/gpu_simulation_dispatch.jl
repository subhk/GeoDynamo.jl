using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

# =============================================================================
# Simulation `gpu` kwarg — dense device-state stepping through the public API.
# On this (CPU/Array) backend `gpu = true` exercises the exact code path a GPU
# run takes minus the device transfer, and must reproduce the CPU stepping to
# machine precision (the 5n2/phase6 parity gates guarantee the per-step match;
# here we check the Simulation-level wiring: cached bundle, CPU bootstrap step,
# host sync, clock, callbacks).
# =============================================================================

function _dispatch_model()
    grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
        lmax = 6, mmax = 6, nlat = 14, nlon = 28, nr = 8, nr_inner = 4)
    model = GeoDynamo.GeodynamoModel(grid; Ek = 1e-3, Ra = 1e5,
        include_magnetic = false, include_composition = false)
    # deterministic IC: conductive background + m=1 symmetry breaker
    GeoDynamo.set!(model; temperature = (r, θ, φ) -> (1 - r) + 1e-2 * sin(θ) * cos(φ))
    return model
end

@testset "Simulation GPU stepping dispatch" begin
    NSTEPS = 4

    @testset "gpu=true matches the CPU path" begin
        cpu_model = _dispatch_model()
        cpu_sim = GeoDynamo.Simulation(cpu_model; Δt = 1e-4,
            stop_iteration = NSTEPS, gpu = false)
        GeoDynamo.run!(cpu_sim)
        @test cpu_model.clock.iteration == NSTEPS

        gpu_model = _dispatch_model()
        gpu_sim = GeoDynamo.Simulation(gpu_model; Δt = 1e-4,
            stop_iteration = NSTEPS, gpu = true)
        @test gpu_sim.gpu == true
        GeoDynamo.run!(gpu_sim)
        @test gpu_model.clock.iteration == NSTEPS
        @test gpu_model.clock.time ≈ cpu_model.clock.time
        @test gpu_sim._gpu_state !== nothing          # bundle built + cached

        cfg = cpu_model.state.backend.shtns_config
        nr = cpu_model.state.runtime.outer_core_domain.N
        for (name, fa, fb) in [
                ("temperature", cpu_model.state.fields.temperature.spectral,
                                gpu_model.state.fields.temperature.spectral),
                ("velocity_tor", cpu_model.state.fields.velocity.toroidal,
                                 gpu_model.state.fields.velocity.toroidal),
                ("velocity_pol", cpu_model.state.fields.velocity.poloidal,
                                 gpu_model.state.fields.velocity.poloidal)]
            ar, ai = GeoDynamo.cpu_spectral_to_dense(fa, cfg, nr, Float64)
            br, bi = GeoDynamo.cpu_spectral_to_dense(fb, cfg, nr, Float64)
            @test isapprox(ar, br; atol = 1e-10, rtol = 1e-8)
            @test isapprox(ai, bi; atol = 1e-10, rtol = 1e-8)
        end
    end

    @testset "callbacks fire and clock advances on the gpu path" begin
        model = _dispatch_model()
        fired = Ref(0)
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, stop_iteration = 3, gpu = true,
            callbacks = (counter = GeoDynamo.Callback(_ -> fired[] += 1,
                schedule = GeoDynamo.IterationInterval(1)),))
        GeoDynamo.run!(sim)
        @test fired[] == 3
        @test model.clock.iteration == 3
        @test model.clock.last_dt == 1e-4
        @test all(isfinite, parent(model.state.fields.temperature.spectral.data_real))
    end

    @testset "ERK2 gpu=true matches the CPU ERK2 path" begin
        cpu_model = _dispatch_model()
        cpu_sim = GeoDynamo.Simulation(cpu_model; Δt = 1e-4,
            stop_iteration = NSTEPS, gpu = false, timestepper = GeoDynamo.ERK2())
        GeoDynamo.run!(cpu_sim)

        gpu_model = _dispatch_model()
        gpu_sim = GeoDynamo.Simulation(gpu_model; Δt = 1e-4,
            stop_iteration = NSTEPS, gpu = true, timestepper = GeoDynamo.ERK2())
        @test gpu_sim.gpu == true
        GeoDynamo.run!(gpu_sim)
        @test gpu_model.clock.iteration == NSTEPS
        @test gpu_sim._gpu_erk2 !== nothing           # ERK2 pack built + cached

        cfg = cpu_model.state.backend.shtns_config
        nr = cpu_model.state.runtime.outer_core_domain.N
        for (fa, fb) in [
                (cpu_model.state.fields.temperature.spectral,
                 gpu_model.state.fields.temperature.spectral),
                (cpu_model.state.fields.velocity.toroidal,
                 gpu_model.state.fields.velocity.toroidal),
                (cpu_model.state.fields.velocity.poloidal,
                 gpu_model.state.fields.velocity.poloidal)]
            ar, ai = GeoDynamo.cpu_spectral_to_dense(fa, cfg, nr, Float64)
            br, bi = GeoDynamo.cpu_spectral_to_dense(fb, cfg, nr, Float64)
            @test isapprox(ar, br; atol = 1e-10, rtol = 1e-8)
            @test isapprox(ai, bi; atol = 1e-10, rtol = 1e-8)
        end
    end

    @testset "unsupported timestepper warns and falls back" begin
        model = _dispatch_model()
        local sim
        @test_logs (:warn, r"CNAB2 and ERK2 only") match_mode = :any begin
            sim = GeoDynamo.Simulation(model; Δt = 1e-4, stop_iteration = 1,
                gpu = true, timestepper = GeoDynamo.ThetaMethod())
        end
        @test sim.gpu == false
    end

    @testset ":auto is off on the CPU architecture" begin
        model = _dispatch_model()
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, stop_iteration = 1)
        @test sim.gpu == false
        @test_throws ArgumentError GeoDynamo.Simulation(model; Δt = 1e-4, gpu = :always)
    end
end
