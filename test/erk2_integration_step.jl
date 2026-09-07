using Test
using MPI

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
        nr = 8,
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
        timestepper = GeoDynamo.ExponentialRungeKutta2(),
        topography_enabled = false,
        stefan_enabled = false
    )
    @test params.timestepper isa GeoDynamo.ExponentialRungeKutta2
    @test params.include_magnetic == true
    @test params.include_composition == true

    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_fields!(state)
    @test state.fields.magnetic !== nothing
    @test state.fields.composition !== nothing

    # Stage-4B ERK2 W-split port: the poloidal stage machinery advances
    # V = Ek·D_pol·P with φ1-column influence recovery — full-physics
    # assertions restored (docs/superpowers/plans/2026-06-11-erk2-wsplit-port.md).
    snap_temp = copy(parent(state.fields.temperature.spectral.data_real))
    snap_mag_tor = copy(parent(state.fields.magnetic.toroidal.data_real))
    snap_mag_pol = copy(parent(state.fields.magnetic.poloidal.data_real))
    snap_comp = copy(parent(state.fields.composition.spectral.data_real))

    GeoDynamo.solver_step!(state)
    @test state.step == 1

    # Ek·∂ₜu_tor = Ek·Δu_tor + N_tor becomes
    # ∂ₜu_tor = Δu_tor + N_tor/Ek.  The ERK2 toroidal propagator must
    # therefore use unit diffusivity and scale the nonlinear increment by
    # inv(Ek); using Ek as the propagator diffusivity scales the entire dynamics
    # by Ek and is not equivalent to the CNAB2 mass-matrix formulation.
    vel_tor_cache = state.timestep_caches.erk2_velocity_toroidal
    @test vel_tor_cache !== nothing
    @test vel_tor_cache.diffusivity == 1.0
    vel_tor_buffers = state.timestep_caches.erk2_field_buffers[:velocity_toroidal]
    cfg = state.runtime.shtns_config
    nr = state.runtime.outer_core_domain.N
    forcing_scale_error = 0.0
    profile = zeros(Float64, nr)
    for lm in GeoDynamo.local_spectral_mode_indices(cfg)
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        cache_idx = vel_tor_buffers.cache_lookup[cfg.l_values[lm]]
        for (nl_data, k1_data) in (
                (vel_tor_buffers.n_current_real, vel_tor_buffers.k1_real),
                (vel_tor_buffers.n_current_imag, vel_tor_buffers.k1_imag))
            for r in 1:nr
                profile[r] = GeoDynamo.local_spectral_value(nl_data, slot, r) / params.Ek
            end
            expected_k1 = vel_tor_cache.phi1_full[cache_idx] * profile
            for r in 1:nr
                forcing_scale_error = max(
                    forcing_scale_error,
                    abs(GeoDynamo.local_spectral_value(k1_data, slot, r) - expected_k1[r]))
            end
        end
    end
    @test forcing_scale_error < 1e-10

    @test all(isfinite, parent(state.fields.magnetic.toroidal.data_real))
    @test all(isfinite, parent(state.fields.magnetic.poloidal.data_real))
    @test all(isfinite, parent(state.fields.composition.spectral.data_real))
    @test all(isfinite, parent(state.fields.velocity.toroidal.data_real))
    @test all(isfinite, parent(state.fields.velocity.poloidal.data_real))
    @test all(isfinite, parent(state.fields.temperature.spectral.data_real))

    @test any(snap_temp .!= parent(state.fields.temperature.spectral.data_real))
    @test any(snap_mag_tor .!= parent(state.fields.magnetic.toroidal.data_real))
    @test any(snap_mag_pol .!= parent(state.fields.magnetic.poloidal.data_real))
    @test any(snap_comp .!= parent(state.fields.composition.spectral.data_real))

    # NOTE: do not finalize MPI here — other MPI-aware tests in the suite run
    # after this file and rely on the communicator staying alive.
end
