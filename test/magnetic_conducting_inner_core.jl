using Test
using MPI
using LinearAlgebra

const FINALIZE_MPI_CONDIC = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# Task 1: Opt-in flag — this testset must PASS.
@testset "conducting flag sets CONTINUITY_MAG bc_type_inner" begin
    params = GeoDynamo.SolverParameters(
        architecture = :cpu, geometry = :shell, nr = 16, nr_inner = 8,
        lmax = 4, mmax = 4, nlat = 12, nlon = 16,
        include_magnetic = true, include_composition = false,
        timestepper = GeoDynamo.CNAB2(),
        magnetic_inner_bc = :conducting_inner_core
    )
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_fields!(state)
    mag = state.fields.magnetic
    @test all(==(Int(GeoDynamo.CONTINUITY_MAG)), mag.toroidal.bc_type_inner)
    @test all(==(Int(GeoDynamo.CONTINUITY_MAG)), mag.poloidal.bc_type_inner)
end

# Acceptance test for a CONDUCTING INNER CORE (magnetic).
#
# Physics contract: when the inner core is electrically conducting, the
# magnetic field diffuses across the ICB into the solid inner core. The
# inner-core toroidal/poloidal scalars (toroidal_ic / poloidal_ic) must therefore develop a
# nonzero internal field over time, and at the ICB the field must be continuous
# with the outer-core solution (toroidal_ic[ICB] ≈ toroidal[ICB], same for poloidal).
#
# Enable signal: bc_type_inner == CONTINUITY_MAG on the magnetic tor/pol fields
# (enum bcs.jl:161, "Conducting inner core: ∂B/∂r continuous at ICB").
#
# STATUS: This test is expected to FAIL until the conducting inner core is
# implemented. Today the inner-core fields are allocated, zeroed, and never
# evolved — so it is a runtime proof that the feature is absent. Once the
# two-domain ICB-coupled solve lands, it becomes the GREEN acceptance test.
@testset "Conducting inner core evolves the inner-core magnetic field" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping conducting inner core test"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    params = GeoDynamo.SolverParameters(
        architecture = :cpu,
        geometry = :shell,
        nr = 16,
        nr_inner = 8,
        lmax = 4,
        mmax = 4,
        nlat = 12,
        nlon = 16,
        Ra = 1e4,
        Ek = 1e-2,
        Pr = 1.0,
        Pm = 1.0,
        timestep = 1e-4,
        start_time = 0.0,
        end_time = 1.0,
        stop_iteration = 1000,
        include_magnetic = true,
        include_composition = false,
        timestepper = GeoDynamo.CNAB2(),
        topography_enabled = false,
        stefan_enabled = false,
        # Enable the conducting inner core via the opt-in parameter. This builds
        # the ICB admittances and the conducting Robin inner row (Part 1) and
        # sets CONTINUITY_MAG on the magnetic tor/pol modes during field init.
        magnetic_inner_bc = :conducting_inner_core
    )

    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_fields!(state)

    mag = state.fields.magnetic
    @test mag !== nothing

    # Sanity: the outer-core seed field is nonzero, so a failure below is due to
    # the inner core not evolving — not an all-zero magnetic state.
    @test maximum(abs, parent(mag.poloidal.data_real)) > 0.0

    # The conducting flag must have set CONTINUITY_MAG on the magnetic modes.
    @test all(==(Int(GeoDynamo.CONTINUITY_MAG)), mag.toroidal.bc_type_inner)
    @test all(==(Int(GeoDynamo.CONTINUITY_MAG)), mag.poloidal.bc_type_inner)

    # Advance enough steps for the field to diffuse across the ICB.
    for _ in 1:30
        GeoDynamo.solver_step!(state)
    end

    ic_tor = parent(mag.toroidal_ic.data_real)
    ic_pol = parent(mag.poloidal_ic.data_real)

    # A conducting inner core must develop a nonzero internal field.
    @test maximum(abs, ic_tor) > 1e-12
    @test maximum(abs, ic_pol) > 1e-12

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_CONDIC && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
