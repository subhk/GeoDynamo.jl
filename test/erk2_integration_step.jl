using Test
using MPI
using Random

# Full-physics ERK2 integration-step coverage.
#
# Drives one real `solver_step!` with the ERK2 timestepper and BOTH the magnetic
# and composition fields enabled. This exercises the magnetic toroidal/poloidal
# and composition branches of `integrate_solver_erk2_step!` (src/timestep/erk2.jl)
# — the largest uncovered block, only reached when those fields are active.
#
# This uses the canonical solver-state path (SolverParameters -> initialize_simulation
# -> solver_step!), mirroring test/integration_simulation.jl. It does NOT fall back
# to unit-testing the branch helpers because the full state builds and steps cleanly.

@testset "ERK2 full-physics integration step (magnetic + composition)" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping ERK2 integration-step test"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    # Tiny shell so one step is cheap but all the ERK2 sub-systems run.
    params = GeoDynamo.SolverParameters(
        architecture = :cpu,
        geometry = :shell,
        nr = 6,
        nr_inner = 2,
        lmax = 2,
        mmax = 2,
        nlat = 6,
        nlon = 8,
        Ra = 1e4,
        Ek = 1e-2,
        Pr = 1.0,
        Pm = 1.0,
        Sc = 1.0,
        timestep = 1e-6,
        start_time = 0.0,
        end_time = 1e-3,
        stop_iteration = 5,
        include_magnetic = true,
        include_composition = true,
        timestepper = GeoDynamo.ERK2(),
        topography_enabled = false,
        stefan_enabled = false
    )
    @test params.timestepper isa GeoDynamo.ERK2
    @test params.include_magnetic == true
    @test params.include_composition == true

    state = GeoDynamo.initialize_simulation(Float64, params)
    # Seed the global RNG so the random initial condition is deterministic
    # regardless of suite order (prior tests otherwise leave the RNG in a
    # varying state, which can perturb the IC amplitude into a non-finite step).
    Random.seed!(20260604)
    GeoDynamo.initialize_fields!(state)

    # Both optional field sets must actually exist for the magnetic/composition
    # branches of integrate_solver_erk2_step! to execute.
    @test state.fields.magnetic !== nothing
    @test state.fields.composition !== nothing

    # Snapshot one coefficient array per evolving subsystem before the step.
    snap_vel_tor = copy(parent(state.fields.velocity.toroidal.data_real))
    snap_temp = copy(parent(state.fields.temperature.spectral.data_real))
    snap_mag_tor = copy(parent(state.fields.magnetic.toroidal.data_real))
    snap_mag_pol = copy(parent(state.fields.magnetic.poloidal.data_real))
    snap_comp = copy(parent(state.fields.composition.spectral.data_real))

    # One full ERK2 step (dispatches through apply_solver_implicit_step! ->
    # integrate_solver_erk2_step!).
    GeoDynamo.solver_step!(state)
    @test state.step == 1

    # (a) Magnetic toroidal + poloidal coefficients remain finite.
    @test all(isfinite, parent(state.fields.magnetic.toroidal.data_real))
    @test all(isfinite, parent(state.fields.magnetic.toroidal.data_imag))
    @test all(isfinite, parent(state.fields.magnetic.poloidal.data_real))
    @test all(isfinite, parent(state.fields.magnetic.poloidal.data_imag))

    # (b) Composition coefficients remain finite.
    @test all(isfinite, parent(state.fields.composition.spectral.data_real))
    @test all(isfinite, parent(state.fields.composition.spectral.data_imag))

    # Velocity + temperature (always-on subsystems) also stay finite.
    @test all(isfinite, parent(state.fields.velocity.toroidal.data_real))
    @test all(isfinite, parent(state.fields.velocity.poloidal.data_real))
    @test all(isfinite, parent(state.fields.temperature.spectral.data_real))

    # (c) The step actually advanced state: at least one coefficient changed.
    vel_tor_changed = any(snap_vel_tor .!= parent(state.fields.velocity.toroidal.data_real))
    temp_changed = any(snap_temp .!= parent(state.fields.temperature.spectral.data_real))
    mag_tor_changed = any(snap_mag_tor .!= parent(state.fields.magnetic.toroidal.data_real))
    mag_pol_changed = any(snap_mag_pol .!= parent(state.fields.magnetic.poloidal.data_real))
    comp_changed = any(snap_comp .!= parent(state.fields.composition.spectral.data_real))

    @test vel_tor_changed || temp_changed || mag_tor_changed ||
          mag_pol_changed || comp_changed
    # Stronger: the magnetic + composition branches specifically advanced their
    # own fields (proves those uncovered branches ran, not just velocity/temp).
    @test mag_tor_changed
    @test mag_pol_changed
    @test comp_changed

    # NOTE: do not finalize MPI here — other MPI-aware tests in the suite run
    # after this file and rely on the communicator staying alive.
end
