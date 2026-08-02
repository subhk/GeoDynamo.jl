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

    @testset "ExponentialRungeKutta2 gpu=true matches the CPU path" begin
        cpu_model = _dispatch_model()
        cpu_sim = GeoDynamo.Simulation(cpu_model; Δt = 1e-4,
            stop_iteration = NSTEPS, gpu = false, timestepper = GeoDynamo.ExponentialRungeKutta2())
        GeoDynamo.run!(cpu_sim)

        gpu_model = _dispatch_model()
        gpu_sim = GeoDynamo.Simulation(gpu_model; Δt = 1e-4,
            stop_iteration = NSTEPS, gpu = true, timestepper = GeoDynamo.ExponentialRungeKutta2())
        @test gpu_sim.gpu == true
        GeoDynamo.run!(gpu_sim)
        @test gpu_model.clock.iteration == NSTEPS
        @test gpu_sim._gpu_erk2 !== nothing           # ExponentialRungeKutta2 pack built + cached

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

    @testset "unsupported timestepper fails before dispatch" begin
        model = _dispatch_model()
        @test_throws ArgumentError begin
            GeoDynamo.Simulation(model; Δt = 1e-4, stop_iteration = 1,
                gpu = true, timestepper = GeoDynamo.ThetaMethod())
        end
    end

    @testset "gpu_sync = :output matches :every" begin
        a_model = _dispatch_model()
        a_sim = GeoDynamo.Simulation(a_model; Δt = 1e-4, stop_iteration = NSTEPS,
            gpu = true, gpu_sync = :every)
        GeoDynamo.run!(a_sim)

        b_model = _dispatch_model()
        b_sim = GeoDynamo.Simulation(b_model; Δt = 1e-4, stop_iteration = NSTEPS,
            gpu = true, gpu_sync = :output)
        GeoDynamo.run!(b_sim)
        @test b_sim._gpu_dirty == false           # final state synced by run!

        cfg = a_model.state.backend.shtns_config
        nr = a_model.state.runtime.outer_core_domain.N
        for (fa, fb) in [
                (a_model.state.fields.temperature.spectral, b_model.state.fields.temperature.spectral),
                (a_model.state.fields.velocity.poloidal, b_model.state.fields.velocity.poloidal)]
            ar, _ = GeoDynamo.cpu_spectral_to_dense(fa, cfg, nr, Float64)
            br, _ = GeoDynamo.cpu_spectral_to_dense(fb, cfg, nr, Float64)
            @test ar == br                         # identical math, lazy mirror
        end

        @test_throws ArgumentError GeoDynamo.Simulation(a_model; Δt = 1e-4, gpu_sync = :sometimes)
    end

    @testset "gpu_sync = :output honours writer/callback schedules" begin
        # `:output` must skip the device→host copy on steps where nothing reads
        # the host mirror. Merely ATTACHING a writer or callback does not make it
        # a reader — only one whose schedule fires this iteration is.
        scratch = mktempdir()

        # (a) an IterationInterval writer that does not fire must not force a sync.
        model = _dispatch_model()
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, gpu = true, gpu_sync = :output,
            output_writers = (GeoDynamo.FieldWriter(scratch;
                schedule = GeoDynamo.IterationInterval(1000), fields = [:temperature]),))
        for _ in 1:3; GeoDynamo.time_step!(sim); end     # 1 = CPU bootstrap, 2-3 device
        @test model.clock.iteration == 3
        @test sim._gpu_dirty                             # writer never fired ⇒ no sync

        # (b) a stateful schedule cannot be pre-queried without consuming the
        # firing, so it stays conservative and syncs every step.
        tmodel = _dispatch_model()
        tsim = GeoDynamo.Simulation(tmodel; Δt = 1e-4, gpu = true, gpu_sync = :output,
            output_writers = (GeoDynamo.FieldWriter(scratch;
                schedule = GeoDynamo.TimeInterval(1e9), fields = [:temperature]),))
        for _ in 1:3; GeoDynamo.time_step!(tsim); end
        @test tsim._gpu_dirty == false

        # (c) nan_checker reads the host fields, so its OWN registered schedule
        # decides — not a hard-coded interval, and not its function identity.
        nmodel = _dispatch_model()
        nsim = GeoDynamo.Simulation(nmodel; Δt = 1e-4, gpu = true, gpu_sync = :output)
        GeoDynamo.add_callback!(nsim, GeoDynamo.nan_checker;
            schedule = GeoDynamo.IterationInterval(2), name = :nan_checker_fast)
        GeoDynamo.time_step!(nsim)                       # 1: CPU bootstrap
        GeoDynamo.time_step!(nsim)                       # 2: device, 2 % 2 == 0 ⇒ sync
        @test nsim._gpu_dirty == false
        GeoDynamo.time_step!(nsim)                       # 3: device, 3 % 2 != 0 ⇒ no sync
        @test nsim._gpu_dirty

        # (d) the non-`Callback` diagnostic types carry schedules too
        # (EnergyDiagnostics / SolenoidalMonitor / SimulationProgress / HealthCheck)
        # and must be pre-queried the same way, not lumped into "assume it reads".
        dmodel = _dispatch_model()
        dsim = GeoDynamo.Simulation(dmodel; Δt = 1e-4, gpu = true, gpu_sync = :output,
            callbacks = (GeoDynamo.SimulationProgress(
                schedule = GeoDynamo.IterationInterval(1000)),))
        for _ in 1:3; GeoDynamo.time_step!(dsim); end
        @test dsim._gpu_dirty
    end

    @testset "Δt change re-syncs the host mirror before the bundle rebuild" begin
        # gpu_sync = :output deliberately leaves the host SolverState behind the
        # device between syncs. The Δt-change branch of _gpu_time_step! bootstraps
        # the new bundle with a CPU step, so it MUST pull the device state back
        # first — otherwise every unsynced device step is discarded and the fields
        # silently rewind while the clock keeps counting.
        dt1, dt2 = 1e-4, 2e-4

        ref_model = _dispatch_model()
        ref_sim = GeoDynamo.Simulation(ref_model; Δt = dt1, gpu = false)
        for _ in 1:3; GeoDynamo.time_step!(ref_sim); end
        ref_sim.dt = dt2
        for _ in 1:3; GeoDynamo.time_step!(ref_sim); end

        model = _dispatch_model()
        sim = GeoDynamo.Simulation(model; Δt = dt1, gpu = true, gpu_sync = :output)
        for _ in 1:3; GeoDynamo.time_step!(sim); end
        @test sim._gpu_dirty                      # host mirror is behind the device
        sim.dt = dt2
        for _ in 1:3; GeoDynamo.time_step!(sim); end
        GeoDynamo.sync_gpu_host!(sim)

        @test model.clock.iteration == ref_model.clock.iteration
        cfg = ref_model.state.backend.shtns_config
        nr = ref_model.state.runtime.outer_core_domain.N
        for (fa, fb) in [
                (ref_model.state.fields.temperature.spectral,
                 model.state.fields.temperature.spectral),
                (ref_model.state.fields.velocity.toroidal,
                 model.state.fields.velocity.toroidal),
                (ref_model.state.fields.velocity.poloidal,
                 model.state.fields.velocity.poloidal)]
            ar, ai = GeoDynamo.cpu_spectral_to_dense(fa, cfg, nr, Float64)
            br, bi = GeoDynamo.cpu_spectral_to_dense(fb, cfg, nr, Float64)
            @test isapprox(ar, br; atol = 1e-10, rtol = 1e-8)
            @test isapprox(ai, bi; atol = 1e-10, rtol = 1e-8)
        end
    end

    @testset "device steps advance the runtime timestep_state clock" begin
        # The device branch bumps state.step/.time directly; runtime.timestep_state
        # is what get_current_simulation_time and the ERK2 diagnostics read, so the
        # two must not drift apart mid-run.
        model = _dispatch_model()
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, gpu = true)
        for _ in 1:3; GeoDynamo.time_step!(sim); end
        st = model.state
        @test st.runtime.timestep_state.step == st.step
        @test st.runtime.timestep_state.time ≈ st.time
    end

    @testset "sync_gpu_host! is public" begin
        # Driving time_step! directly under gpu_sync = :output leaves the host
        # SolverState behind the device, and run! is the only thing that used to
        # flush it — there was no supported way to read the state mid-run.
        @test isdefined(GeoDynamo, :sync_gpu_host!)
        @test Base.isexported(GeoDynamo, :sync_gpu_host!)

        model = _dispatch_model()
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, gpu = true, gpu_sync = :output)
        for _ in 1:3; GeoDynamo.time_step!(sim); end
        @test sim._gpu_dirty                       # host mirror is stale
        @test GeoDynamo.sync_gpu_host!(sim) === sim
        @test sim._gpu_dirty == false              # ...and now it is not

        # idempotent, and a no-op on the CPU path rather than an error
        @test GeoDynamo.sync_gpu_host!(sim) === sim
        cpu_sim = GeoDynamo.Simulation(_dispatch_model(); Δt = 1e-4, gpu = false)
        GeoDynamo.time_step!(cpu_sim)
        @test GeoDynamo.sync_gpu_host!(cpu_sim) === cpu_sim
    end

    @testset "bundle scope limits are re-checked every step, not just at build" begin
        # The device bundle bakes boundary endpoint values and the internal-source
        # profile at BUILD time, and it is only rebuilt on a Δt change. Mutating
        # either afterwards used to be silently ignored, despite the builder
        # docstring promising a loud rejection.
        function _td_data(::Type{T}) where {T}
            GeoDynamo.bcs.BoundaryData{T}(
                nothing, nothing, T[0.0, 1.0], zeros(T, 4, 8, 2),
                "K", "synthetic time-dependent boundary", "", "temperature",
                true, 4, 8, 2, 1)
        end

        model = _dispatch_model()
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, gpu = true)
        for _ in 1:2; GeoDynamo.time_step!(sim); end     # bundle built + stepping
        model.state.fields.temperature.boundary_condition_set =
            GeoDynamo.bcs.BoundaryConditionSet{Float64}(
                _td_data(Float64), _td_data(Float64), "temperature",
                GeoDynamo.bcs.TEMPERATURE, 0.0)
        @test_throws ErrorException GeoDynamo.time_step!(sim)

        # same for an internal-source profile swapped in after the bundle was packed
        model2 = _dispatch_model()
        sim2 = GeoDynamo.Simulation(model2; Δt = 1e-4, gpu = true)
        for _ in 1:2; GeoDynamo.time_step!(sim2); end
        model2.state.fields.temperature.internal_sources .+= 1.0
        @test_throws ErrorException GeoDynamo.time_step!(sim2)
    end

    @testset ":auto is off on the CPU architecture" begin
        model = _dispatch_model()
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, stop_iteration = 1)
        @test sim.gpu == false
        @test_throws ArgumentError GeoDynamo.Simulation(model; Δt = 1e-4, gpu = :always)
    end
end
