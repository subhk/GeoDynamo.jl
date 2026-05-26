using Test
using MPI
using LinearAlgebra

const FINALIZE_MPI_CONDIC = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# Acceptance test for a CONDUCTING INNER CORE (magnetic).
#
# Physics contract: when the inner core is electrically conducting, the
# magnetic field diffuses across the ICB into the solid inner core. The
# inner-core toroidal/poloidal scalars (𝒯ⁱᶜ / 𝒫ⁱᶜ) must therefore develop a
# nonzero internal field over time, and at the ICB the field must be continuous
# with the outer-core solution (𝒯ⁱᶜ[ICB] ≈ 𝒯[ICB], same for poloidal).
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
        max_steps = 1000,
        include_magnetic_field = true,
        include_composition = false,
        timestepper = GeoDynamo.CNAB2(),
        topography_enabled = false,
        stefan_enabled = false,
    )

    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_fields!(state)

    mag = state.fields.magnetic
    @test mag !== nothing

    # Sanity: the outer-core seed field is nonzero, so a failure below is due to
    # the inner core not evolving — not an all-zero magnetic state.
    @test maximum(abs, parent(mag.𝒫.data_real)) > 0.0

    # Enable conducting inner core (continuity at ICB) on all magnetic modes.
    fill!(mag.𝒯.bc_type_inner, Int(GeoDynamo.CONTINUITY_MAG))
    fill!(mag.𝒫.bc_type_inner, Int(GeoDynamo.CONTINUITY_MAG))

    # Advance enough steps for the field to diffuse across the ICB.
    for _ in 1:30
        GeoDynamo.advance_solver_step!(state)
    end

    ic_tor = parent(mag.𝒯ⁱᶜ.data_real)
    ic_pol = parent(mag.𝒫ⁱᶜ.data_real)

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
