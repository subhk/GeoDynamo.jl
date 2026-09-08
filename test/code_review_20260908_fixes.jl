module Review20260908Tests

using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

function _review0908_model(; T=Float64, all_fields=false)
    grid = SphericalShellGrid(GeoDynamo.CPU(); lmax=2, mmax=2, nlat=8, nlon=8,
        nr=8, nr_inner=4)
    walls = BoundaryConditions(inner=FixedTemperature(0.0), outer=FixedTemperature(0.0))
    return GeodynamoModel(grid; T, Ek=1e-2, Ra=1e-8,
        temperature_bcs=walls, composition_bcs=walls,
        include_magnetic=all_fields, include_composition=all_fields)
end

function _review0908_seed!(model)
    set!(model; velocity=RandomPerturbation(amplitude=1e-3, lmax=2, seed=17),
        temperature=(r, theta, phi)->0.01*sin(theta)*cos(phi))
    return model
end

@testset "Review 2026-09-08 fixes" begin
    @testset "Changing theta at fixed dt updates the actual operators" begin
        model = _review0908_model(all_fields=true)
        dt = model.state.parameters.timestep
        Simulation(model; dt, implicit_theta=1.0)
        for key in (:temperature, :composition, :velocity_tor, :magnetic_tor)
            @test model.state.implicit_matrices[key].theta == 1.0
        end
        Simulation(model; dt, timestepper=CNAB2(theta=0.5))
        @test model.state.implicit_matrices[:temperature].theta == 0.5
    end

    @testset "CNAB2 extrapolates over the actual step interval" begin
        model = _review0908_model()
        old_dt = 1e-5
        time_step!(model, old_dt)
        Simulation(model; dt=2old_dt)
        field = model.temperature
        for f in (field.spectral, field.nonlinear, field.prev_nonlinear)
            fill!(parent(f.data_real), 0.0)
            fill!(parent(f.data_imag), 0.0)
        end
        # N(t_prev)=0 and N(t_n)=1. Over twice the preceding interval,
        # linear extrapolation has mean 2, rather than the fixed-step 1.5.
        fill!(parent(field.nonlinear.data_real), 1.0)
        GeoDynamo.apply_temperature_implicit_update!(model.state)
        slot = GeoDynamo.local_spectral_storage_slot(field.config, 1)
        if slot !== nothing
            @test GeoDynamo.local_spectral_value(
                parent(field.work_spectral.data_real), slot, 4) ≈ 2.0
        end

        # Poloidal momentum assembles a separate W-equation RHS. With zero P
        # and N_prev, doubling the interval scales its forced solution by 2/1.5.
        velocity = model.velocity
        for f in (velocity.poloidal, velocity.nl_poloidal, velocity.prev_nl_poloidal)
            fill!(parent(f.data_real), 0.0)
            fill!(parent(f.data_imag), 0.0)
        end
        fill!(parent(velocity.nl_poloidal.data_real), 1.0)
        GeoDynamo.apply_velocity_poloidal_implicit_update!(model.state)
        variable_step = copy(parent(velocity.poloidal.data_real))
        fill!(parent(velocity.poloidal.data_real), 0.0)
        model.state.runtime.timestep_state.previous_dt = 2old_dt
        GeoDynamo.apply_velocity_poloidal_implicit_update!(model.state)
        fixed_step = parent(velocity.poloidal.data_real)
        @test any(!iszero, fixed_step)
        @test variable_step ≈ (4 / 3) .* fixed_step rtol=1e-12 atol=1e-14
    end

    @testset "A restart retains the interval of its nonlinear history" begin
        source = _review0908_seed!(_review0908_model(all_fields=true))
        time_step!(source, 1e-5)
        time_step!(source, 2e-5)
        data = GeoDynamo.extract_all_fields(source.state)
        @test data["previous_dt"] == 2e-5
        target = _review0908_model(all_fields=true)
        GeoDynamo.restore_fields_from_restart!(target.state, data)
        GeoDynamo.reset_solver_clock!(target.state; time=source.state.time, step=source.state.step)
        @test target.state.runtime.timestep_state.previous_dt == 2e-5
        for dt in (3e-5, 1e-5)
            time_step!(source, dt)
            time_step!(target, dt)
        end
        a, b = GeoDynamo.extract_all_fields(source.state), GeoDynamo.extract_all_fields(target.state)
        for key in ("temperature_spectral", "composition_spectral", "velocity_toroidal",
            "velocity_poloidal", "magnetic_toroidal", "magnetic_poloidal")
            @test a[key]["real"] ≈ b[key]["real"] rtol=1e-12 atol=1e-14
            @test a[key]["imag"] ≈ b[key]["imag"] rtol=1e-12 atol=1e-14
        end
        # Older files contain no history timestep. They still load, but must
        # bootstrap rather than assume that the new run chose the original dt.
        delete!(data, "previous_dt")
        GeoDynamo.restore_fields_from_restart!(target.state, data)
        @test target.state.runtime.timestep_state.needs_ab2_bootstrap
    end

    @testset "Scalar setters discard pre-reset nonlinear history" begin
        for ic in (0.0, (r, theta, phi)->0.0, ZeroIC())
            model = _review0908_seed!(_review0908_model())
            time_step!(model, 1e-5)
            set!(model; temperature=ic)
            time_step!(model, 1e-5)
            @test all(iszero, parent(model.temperature.spectral.data_real))
            @test all(iszero, parent(model.temperature.spectral.data_imag))
        end
    end

    @testset "GPU field edits preserve current unedited fields" begin
        if MPI.Comm_size(MPI.COMM_WORLD) == 1
            for sync in (:every, :output)
                cpu = Simulation(_review0908_seed!(_review0908_model()); dt=1e-5, gpu=false)
                gpu = Simulation(_review0908_seed!(_review0908_model()); dt=1e-5,
                    gpu=true, gpu_sync=sync)
                for _ in 1:3
                    time_step!(cpu)
                    time_step!(gpu)
                end
                # With :output the GPU host mirror is stale here. A scalar edit
                # must retain the latest velocity while replacing temperature.
                set!(cpu.model; temperature=ZeroIC())
                set!(gpu.model; temperature=ZeroIC())
                for _ in 1:2
                    time_step!(cpu)
                    time_step!(gpu)
                end
                sync_gpu_host!(gpu)
                @test all(iszero, parent(gpu.model.temperature.spectral.data_real))
                for (a,b) in ((cpu.model.velocity.toroidal, gpu.model.velocity.toroidal),
                    (cpu.model.velocity.poloidal, gpu.model.velocity.poloidal))
                    @test parent(a.data_real) ≈ parent(b.data_real) rtol=1e-7 atol=1e-12
                    @test parent(a.data_imag) ≈ parent(b.data_imag) rtol=1e-7 atol=1e-12
                end
            end
        end
    end

    @testset "Float32 models construct and advance" begin
        model = _review0908_model(T=Float32, all_fields=true)
        time_step!(model, 1e-5)
        time_step!(model, 2e-5)
        @test model.clock.iteration == 2
        @test model.clock.last_dt === Float32(2e-5)
        @test eltype(parent(model.temperature.spectral.data_real)) === Float32
        @test all(isfinite, parent(model.temperature.spectral.data_real))
    end
end

end # module
