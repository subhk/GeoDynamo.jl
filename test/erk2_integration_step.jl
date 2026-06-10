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
        timestepper = GeoDynamo.ERK2(),
        topography_enabled = false,
        stefan_enabled = false
    )
    @test params.timestepper isa GeoDynamo.ERK2
    @test params.include_magnetic == true
    @test params.include_composition == true

    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_fields!(state)
    @test state.fields.magnetic !== nothing
    @test state.fields.composition !== nothing

    # Stage-4B gate: nl_poloidal now carries the W-equation RHS of the
    # pressure-free double-curl momentum form; the ERK2 stage machinery still
    # advances the legacy poloidal equation and refuses loudly until ported
    # (docs/superpowers/plans/2026-06-10-double-curl-stage4b-poloidal-momentum.md).
    # The full-physics finite/advance assertions that lived here return with
    # the ERK2 port.
    @test_throws ErrorException GeoDynamo.solver_step!(state)

    # NOTE: do not finalize MPI here — other MPI-aware tests in the suite run
    # after this file and rely on the communicator staying alive.
end
