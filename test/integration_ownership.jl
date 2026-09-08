module IntegrationOwnershipTests

using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

function model(; T=Float64)
    grid = SphericalShellGrid(GeoDynamo.CPU(); lmax=2, mmax=2, nlat=8, nlon=8,
        nr=8, nr_inner=4)
    walls = BoundaryConditions(inner=FixedTemperature(0.0), outer=FixedTemperature(0.0))
    GeodynamoModel(grid; T, Ek=1e-2, Ra=1e-8, temperature_bcs=walls,
        include_magnetic=false, include_composition=false)
end

function controls(p; overrides...)
    values = (; (name => getfield(p, name) for name in fieldnames(typeof(p)))...)
    GeoDynamo.SolverParameters(; merge(values, (; overrides...))...)
end

@testset "Integration ownership" begin
    @testset "All clock views share writes and completed advances" begin
        m = model()
        st = m.state
        GeoDynamo.reset_solver_clock!(st; time=0.25, step=7)
        @test m.clock.time == 0.25
        @test m.clock.iteration == 7
        m.clock.time = 0.5
        m.clock.iteration = 9
        @test st.time == st.runtime.timestep_state.time == 0.5
        @test st.step == st.runtime.timestep_state.step == 9
        st.time = 0.75
        st.step = 11
        @test m.clock.time == st.runtime.timestep_state.time == 0.75
        @test m.clock.iteration == st.runtime.timestep_state.step == 11
        GeoDynamo.initialize_solver_fields!(st)
        GeoDynamo.solver_step!(st)
        @test m.clock.time == st.time == st.parameters.timestep
        @test m.clock.iteration == st.step == 1
        @test m.clock.last_dt == st.parameters.timestep
        @test m.clock.last_Δt == m.clock.last_dt
        if MPI.Comm_size(MPI.COMM_WORLD) == 1
            GeoDynamo.gpu_run!(st, 2)
            @test m.clock.iteration == st.step == 3
            @test m.clock.time == st.time
            @test m.clock.last_dt == st.parameters.timestep
        end
    end

    @testset "Float32 and standalone clock compatibility" begin
        c = Clock(time=Float32(0.25), last_dt=Float32(0.1))
        c.time = 0.5
        c.last_Δt = 0.125
        @test c.time === Float32(0.5)
        @test c.last_dt === Float32(0.125)
        @test all(n -> hasproperty(c, n), (:time, :iteration, :stage, :last_dt, :last_Δt))
        m = model(T=Float32)
        GeoDynamo.solver_step!(m.state)
        @test m.clock.time === Float32(m.state.time)
        @test m.clock.last_dt === Float32(m.state.parameters.timestep)
        @test m.clock.iteration == 1
    end

    @testset "Fixed-dt physical changes invalidate all derived operators" begin
        for scheme in (CNAB2(), ExponentialRungeKutta2(), RungeKutta3())
            m = model()
            Simulation(m; dt=1e-5, timestepper=scheme)
            time_step!(m, 1e-5)
            st = m.state
            old_matrices = st.implicit_matrices
            old_caches = st.timestep_caches
            # End time has no influence on the operators or their scratch buffers.
            p = controls(st.parameters; end_time=2.0)
            GeoDynamo._commit_run_controls!(m, p, p.timestep, p.timestep, () -> nothing)
            @test st.implicit_matrices === old_matrices
            @test st.timestep_caches === old_caches
            p = controls(p; Ek=2p.Ek, Pr=2p.Pr)
            GeoDynamo._commit_run_controls!(m, p, p.timestep, p.timestep, () -> nothing)
            @test st.implicit_matrices !== old_matrices
            @test st.timestep_caches !== old_caches
            @test st.timestep_caches.poloidal_split === nothing
            @test st.timestep_caches.erk2_temperature === nothing
            @test all(isnothing, st.timestep_caches.cb3_stage_matrices)
            @test st.timestep_caches.radial_work === old_caches.radial_work
            @test st.timestep_caches.erk2_field_buffers === old_caches.erk2_field_buffers
            expected, _ = GeoDynamo._build_implicit_matrices_dict(Float64,
                st.backend.shtns_config, st.backend.outer_core_domain,
                st.backend.inner_core_domain, p, p.timestep)
            @test st.implicit_matrices[:temperature].system_matrices[1].data ==
                expected[:temperature].system_matrices[1].data
            time_step!(m, p.timestep)
            @test all(isfinite, parent(m.temperature.spectral.data_real))
            before = (st.parameters, st.implicit_matrices, st.timestep_caches)
            p = controls(p; timestep=2p.timestep)
            @test_throws ErrorException GeoDynamo._commit_run_controls!(m, p,
                p.timestep, st.parameters.timestep, () -> error("injected failure"))
            @test st.parameters === before[1]
            @test st.implicit_matrices === before[2]
            @test st.timestep_caches === before[3]
            @test st.runtime.timestep_state.dt == st.parameters.timestep

            # Boundary type changes at the same dt must also replace factors.
            p = controls(st.parameters; temperature_bcs=BoundaryConditions(
                inner=FluxBoundaryCondition(0.0), outer=FixedTemperature(0.0)))
            before_matrices = st.implicit_matrices
            GeoDynamo._commit_run_controls!(m, p, p.timestep, p.timestep, () -> nothing)
            @test st.implicit_matrices !== before_matrices
            @test st.timestep_caches.erk2_temperature === nothing
            expected, _ = GeoDynamo._build_implicit_matrices_dict(Float64,
                st.backend.shtns_config, st.backend.outer_core_domain,
                st.backend.inner_core_domain, p, p.timestep)
            @test st.implicit_matrices[:temperature].system_matrices[1].data ==
                expected[:temperature].system_matrices[1].data
            # Low-level callers get the same validity check at the step boundary.
            before_matrices = st.implicit_matrices
            st.parameters = controls(p; Pr=2p.Pr)
            GeoDynamo.solver_step!(st)
            @test st.implicit_matrices !== before_matrices
        end
    end

    @testset "Operator changes synchronize and release an active device copy" begin
        if MPI.Comm_size(MPI.COMM_WORLD) == 1
            sim = Simulation(model(); dt=1e-5, gpu=true, gpu_sync=:output)
            set!(sim.model; temperature=(r, theta, phi)->0.01*sin(theta)*cos(phi))
            time_step!(sim)
            time_step!(sim)
            @test sim._gpu_dirty
            before_step = sim.model.clock.iteration
            p = controls(sim.model.state.parameters; Ek=0.02)
            GeoDynamo._commit_run_controls!(sim.model, p, p.timestep, p.timestep, () -> nothing)
            @test sim._gpu_state === nothing
            @test !sim._gpu_dirty
            @test sim.model.clock.iteration == before_step
            time_step!(sim)
            @test sim.model.clock.iteration == before_step + 1
            @test sim._gpu_state !== nothing
        end
    end

    @testset "ERK2 diagnostics retain their completed-step cadence" begin
        m = model()
        Simulation(m; dt=1e-5, timestepper=ExponentialRungeKutta2())
        time_step!(m, 1e-5)
        @test isempty(m.state.solenoidal_monitor.timestamps)
        m.clock.iteration = 99
        time_step!(m, 1e-5)
        @test m.state.solenoidal_monitor.timestamps == [100]
        @test m.clock.iteration == 100
    end
end

end # module
