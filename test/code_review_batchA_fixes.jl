# ================================================================================
# Regression tests for batch A of the max-effort src/ review — the CONFIRMED
# correctness findings the crashed synthesis step never surfaced ([15]-[21]).
# ================================================================================
#
#   A19 solver/mainloop.jl:135        rebuild_solver_implicit_matrices! rebuilds from
#                                    the FROZEN backend.parameters, so theta / Ek /
#                                    Pm / BC codes revert to construction time
#   A15 physics/temperature/field.jl:631  set_boundary_conditions! stores the PHYSICAL
#                                    boundary value in a slot every consumer reads as
#                                    the (0,0) spectral coefficient (value*sqrt(4pi))
#   A16 physics/composition/field.jl:564  set_composition_boundary_conditions! broadcasts
#                                    one value across ALL (l,m) modes, unscaled
#   A18 diagnostics/solver.jl:137    energy-drift baseline is total_energy[1], which
#                                    trim_energy_tracker! deletes -> moving window
#   A20 gpu/run.jl:62                gpu_run! always runs the CNAB2 device step,
#                                    whatever parameters.timestepper says
#   A21 api/simulation.jl:512        _gpu_assert_bundle_current never re-checks the
#                                    baked boundary endpoint VALUES
# ================================================================================

using Test
using MPI
using Random
using GeoDynamo

@testset "Max-effort review batch A" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping batch A fix tests"
        return
    end
    MPI.Initialized() || MPI.Init()

    cra_grid() = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
        lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
    cra_model(; kwargs...) = GeoDynamo.GeodynamoModel(cra_grid();
        Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false,
        kwargs...)

    _sqrt4pi = sqrt(4 * pi)

    # ── A19: the implicit rebuild must read the LIVE parameters ───────────────
    @testset "A19 rebuild_solver_implicit_matrices! uses live parameters" begin
        model = cra_model()
        st = model.state
        p0 = st.parameters
        @test st.implicit_matrices[:temperature].theta == 0.5   # baseline

        # theta lives on the timestepper struct; swapping it must reach the operators.
        st.parameters = GeoDynamo.SolverParameters(;
            (f => getfield(p0, f) for f in fieldnames(GeoDynamo.SolverParameters))...,
            timestepper = GeoDynamo.CNAB2(theta = 1.0))
        GeoDynamo.rebuild_solver_implicit_matrices!(st, st.parameters.timestep)
        @test st.implicit_matrices[:temperature].theta == 1.0
        @test st.implicit_matrices[:velocity_tor].theta == 1.0

        # A non-timestepper physical parameter must reach them too.
        before = copy(st.implicit_matrices[:velocity_tor].linear_matrices[1].data)
        st.parameters = GeoDynamo.SolverParameters(;
            (f => getfield(st.parameters, f) for f in fieldnames(GeoDynamo.SolverParameters))...,
            Ek = 10 * p0.Ek)
        GeoDynamo.rebuild_solver_implicit_matrices!(st, st.parameters.timestep)
        @test copy(st.implicit_matrices[:velocity_tor].linear_matrices[1].data) != before

        # End to end: this is what makes the Simulation implicit_theta kwarg real.
        m2 = cra_model()
        GeoDynamo.Simulation(m2; Δt = 2e-4, implicit_theta = 1.0)
        @test m2.state.parameters.timestepper.implicit_theta == 1.0
        @test m2.state.implicit_matrices[:temperature].theta == 1.0
    end

    # ── A15: temperature boundary values are (0,0) COEFFICIENTS ──────────────
    @testset "A15 set_boundary_conditions! stores the (0,0) coefficient" begin
        model = cra_model()
        temp = model.state.fields.temperature
        GeoDynamo.set_boundary_conditions!(temp; inner_value = 1.0, outer_value = 0.25)
        idx = GeoDynamo.get_mode_index(temp.config, 0, 0)
        @test idx > 0
        @test temp.boundary_values[1, idx] ≈ _sqrt4pi * 1.0
        @test temp.boundary_values[2, idx] ≈ _sqrt4pi * 0.25
        # every other mode stays zero
        others = [i for i in 1:temp.config.nlm if i != idx]
        @test all(temp.boundary_values[1, others] .== 0.0)
        @test all(temp.boundary_values[2, others] .== 0.0)
        # same convention the canonical installer uses
        ref = GeoDynamo.create_shtns_temperature_field(Float64,
            model.state.backend.shtns_config, model.state.runtime.outer_core_domain)
        GeoDynamo.apply_scalar_boundary_parameters!(ref, GeoDynamo.BoundaryConditions(
            inner = GeoDynamo.FixedTemperature(1.0),
            outer = GeoDynamo.FixedTemperature(0.25)))
        @test temp.boundary_values[1, idx] ≈ ref.boundary_values[1, idx]
        @test temp.boundary_values[2, idx] ≈ ref.boundary_values[2, idx]
    end

    # ── A16: composition boundary values only on (0,0), scaled ───────────────
    @testset "A16 set_composition_boundary_conditions! sets only the (0,0) mode" begin
        cfg = GeoDynamo.create_shtnskit_config(
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 8)
        shell = GeoDynamo.create_radial_domain(8)
        comp = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        idx = GeoDynamo.get_mode_index(cfg, 0, 0)
        @test idx > 0

        GeoDynamo.set_composition_boundary_conditions!(comp, :fixed, :no_flux, 2.5, -1.0)
        @test all(comp.bc_type_inner .== Int(GeoDynamo.DIRICHLET))
        @test all(comp.bc_type_outer .== Int(GeoDynamo.NEUMANN))
        @test comp.boundary_values[1, idx] ≈ _sqrt4pi * 2.5
        others = [i for i in 1:cfg.nlm if i != idx]
        @test all(comp.boundary_values[1, others] .== 0.0)
        @test all(comp.boundary_values[2, :] .== 0.0)   # no_flux -> zero flux

        # opposite direction, and the stale inner row must be cleared
        GeoDynamo.set_composition_boundary_conditions!(comp, :no_flux, :fixed, 0.0, 4.0)
        @test all(comp.bc_type_inner .== Int(GeoDynamo.NEUMANN))
        @test all(comp.bc_type_outer .== Int(GeoDynamo.DIRICHLET))
        @test comp.boundary_values[2, idx] ≈ _sqrt4pi * 4.0
        @test all(comp.boundary_values[2, others] .== 0.0)
        @test all(comp.boundary_values[1, :] .== 0.0)
    end

    # ── A18: the drift baseline must survive history trimming ────────────────
    @testset "A18 energy drift baseline survives trim_energy_tracker!" begin
        tr = GeoDynamo.create_solver_energy_tracker()
        n = GeoDynamo.SOLVER_MAX_TRACKER_HISTORY + 10
        for i in 1:n
            push!(tr.kinetic_energy, 0.0)
            push!(tr.magnetic_energy, 0.0)
            push!(tr.thermal_energy, 0.0)
            push!(tr.compositional_energy, 0.0)
            push!(tr.total_energy, Float64(i))
            push!(tr.timestamps, i)
            GeoDynamo.trim_energy_tracker!(tr)
        end
        # the front really was trimmed ...
        @test length(tr.total_energy) < n
        @test tr.total_energy[1] != 1.0
        # ... but the true first sample is still the drift baseline
        @test tr.initial_total_energy == 1.0
        @test GeoDynamo._solver_energy_baseline(tr) == 1.0

        # before any trim the baseline is simply the first sample
        fresh = GeoDynamo.create_solver_energy_tracker()
        push!(fresh.total_energy, 7.0)
        @test GeoDynamo._solver_energy_baseline(fresh) == 7.0
    end

    # ── A20: gpu_run! must honour the configured timestepper ─────────────────
    @testset "A20 gpu_run! dispatches on the timestepper" begin
        function erk2_state()
            params = GeoDynamo.SolverParameters(
                geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8,
                nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
                Ek = 1e-3, Ra = 1e4, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
                include_magnetic = false, include_composition = false,
                timestepper = GeoDynamo.ExponentialRungeKutta2())
            st = GeoDynamo.initialize_solver_state(Float64; params = params)
            rng = MersenneTwister(7)
            for f in (st.fields.temperature.spectral,
                st.fields.velocity.toroidal, st.fields.velocity.poloidal)
                dr = parent(f.data_real)
                di = parent(f.data_imag)
                dr .+= 1e-3 .* (rand(rng, size(dr)...) .- 0.5)
                di .+= 1e-3 .* (rand(rng, size(di)...) .- 0.5)
            end
            GeoDynamo.solver_step!(st)      # warm the caches / lagged buffers
            return st
        end

        # (a) the bundle-level loop must actually call the stepper it is given.
        st0 = erk2_state()
        gst0 = GeoDynamo.build_gpu_solver_state(st0)
        calls = Ref(0)
        GeoDynamo.gpu_run!(gst0, 3; step! = _ -> (calls[] += 1; nothing))
        @test calls[] == 3

        # (b) an ERK2 run must reproduce the ERK2 device step, not the CNAB2 one.
        a = erk2_state()
        b = erk2_state()
        GeoDynamo.gpu_run!(a, 2)
        gb = GeoDynamo.build_gpu_solver_state(b)
        erk = GeoDynamo.build_gpu_erk2_state(b)
        for _ in 1:2
            GeoDynamo.gpu_erk2_solver_step!(gb, erk)
        end
        GeoDynamo.sync_gpu_state_to_cpu!(b, gb)
        @test parent(a.fields.temperature.spectral.data_real) ==
              parent(b.fields.temperature.spectral.data_real)
        @test parent(a.fields.velocity.toroidal.data_real) ==
              parent(b.fields.velocity.toroidal.data_real)

        # (c) the step choice is dispatch, and a timestepper with no device step must
        # error rather than fall through to CNAB2. (The solver's own parameter
        # validation rejects such a timestepper today, so this is the guard that keeps
        # a future addition from silently inheriting the CNAB2 step.)
        @test GeoDynamo._gpu_device_step(GeoDynamo.CNAB2(), nothing) ===
              GeoDynamo.gpu_solver_step!
        @test GeoDynamo._gpu_device_step(GeoDynamo.RungeKutta3(), nothing) ===
              GeoDynamo.gpu_cb3_solver_step!
        @test GeoDynamo._gpu_device_step(
            GeoDynamo.ExponentialRungeKutta2(), nothing) !== GeoDynamo.gpu_solver_step!
        @test_throws ErrorException GeoDynamo._gpu_device_step(GeoDynamo.ETD(), nothing)
    end

    # ── A21: the device bundle must notice changed boundary endpoint values ───
    @testset "A21 bundle staleness check covers boundary endpoint values" begin
        model = cra_model()
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, gpu = true)
        GeoDynamo.time_step!(sim)      # CPU bootstrap: builds the bundle
        GeoDynamo.time_step!(sim)      # device step with a current bundle

        temp = model.state.fields.temperature
        idx = GeoDynamo.get_mode_index(temp.config, 0, 0)
        temp.boundary_values[2, idx] += 1.0    # move the baked outer endpoint
        @test_throws ErrorException GeoDynamo.time_step!(sim)

        # restoring it makes the bundle current again
        temp.boundary_values[2, idx] -= 1.0
        GeoDynamo.time_step!(sim)
        @test model.clock.iteration == 3
    end
end
