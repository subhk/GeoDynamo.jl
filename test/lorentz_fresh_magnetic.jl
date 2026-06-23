using Test, MPI, LinearAlgebra
using GeoDynamo

MPI.Initialized() || MPI.Init()

# ===========================================================================
# Bug: the Lorentz force in the velocity nonlinear pass reads the magnetic
# PHYSICAL buffers (B, j = ∇×B). Those buffers were refreshed only by the
# magnetic nonlinear pass, which runs AFTER the velocity pass in
# compute_solver_nonlinear_terms!. So the velocity Lorentz force used a
# one-step-lagged magnetic field (zero on the first step) — the same lag class
# as the buoyancy bug that was already fixed.
#
# Correct behaviour: the production nonlinear assembly's velocity terms must
# equal the velocity terms computed with B,j physical freshly synthesized from
# the CURRENT spectral magnetic field.
# ===========================================================================

function _lf_state(; dt = 1e-4)
    params = GeoDynamo.SolverParameters(
        architecture = :cpu, geometry = :shell,
        nr = 16, nr_inner = 4, lmax = 8, mmax = 8, nlat = 20, nlon = 40,
        Ra = 1e4, Ek = 1e-2, Pr = 1.0, Pm = 1.0, Sc = 1.0,
        timestep = dt, start_time = 0.0, end_time = 1.0, stop_iteration = 100000,
        include_magnetic = true, include_composition = false,
        timestepper = GeoDynamo.CNAB2(), topography_enabled = false, stefan_enabled = false)
    return GeoDynamo.initialize_simulation(Float64, params)
end

function _seed_magnetic!(state)
    mag = state.fields.magnetic
    dom = state.backend.outer_core_domain
    ri = dom.r[1, 4]; ro = dom.r[dom.N, 4]
    for (fld, amp) in ((mag.poloidal, 1.0e-2), (mag.toroidal, 7.0e-3))
        cfg = fld.config
        for lm in 1:cfg.nlm
            l = cfg.l_values[lm]
            (l in (1, 2)) || continue
            m = cfg.m_values[lm]
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            for r_idx in 1:dom.N
                x = (dom.r[r_idx, 4] - ri) / (ro - ri)
                GeoDynamo.set_local_spectral_value!(parent(fld.data_real), slot, r_idx, amp * sinpi(x))
                if m != 0
                    GeoDynamo.set_local_spectral_value!(parent(fld.data_imag), slot, r_idx, 0.5amp * sinpi(x))
                end
            end
        end
    end
    return state
end

function _zero_magnetic_physical!(mag)
    for comp in (mag.magnetic.r_component, mag.magnetic.θ_component, mag.magnetic.φ_component,
                 mag.current.r_component, mag.current.θ_component, mag.current.φ_component)
        fill!(parent(comp.data), 0.0)
    end
end

_velnl(vel) = (copy(parent(vel.nl_toroidal.data_real)), copy(parent(vel.nl_poloidal.data_real)))

@testset "Lorentz force uses the current magnetic field, not a lagged one" begin
    state = _lf_state()
    GeoDynamo.initialize_solver_fields!(state)   # consume the one-shot IC regeneration
    _seed_magnetic!(state)

    vel = state.fields.velocity
    temp = state.fields.temperature
    comp = state.fields.composition
    mag = state.fields.magnetic
    dom = state.backend.outer_core_domain

    # Keep the temperature physical field consistent across all three evaluations
    # (buoyancy reads it); the production path refreshes it the same way.
    GeoDynamo.scalar_spectral_to_physical!(temp.spectral, GeoDynamo.solver_main_physical_field(temp))

    # REFERENCE: velocity nonlinear with B,j physical freshly synthesized from
    # the current spectral magnetic field.
    GeoDynamo.refresh_magnetic_physical_fields!(mag, dom)
    GeoDynamo.refresh_current_physical_fields!(mag, dom)
    GeoDynamo.solver_compute_velocity_nonlinear!(vel, temp, comp, mag, dom; params = state.parameters)
    ref_tor, ref_pol = _velnl(vel)

    # NON-VACUITY: with B,j physical zeroed the Lorentz term drops, so the
    # velocity nonlinear must differ from the reference — proving the Lorentz
    # contribution is non-trivial and the equality test below is meaningful.
    _zero_magnetic_physical!(mag)
    GeoDynamo.solver_compute_velocity_nonlinear!(vel, temp, comp, mag, dom; params = state.parameters)
    zb_tor, zb_pol = _velnl(vel)
    @test norm(ref_pol - zb_pol) + norm(ref_tor - zb_tor) > 1e-10

    # PRODUCTION: zero the physical buffers again, then run the real assembly.
    # Pre-fix the velocity Lorentz reads the stale (zero) field; post-fix it
    # refreshes the magnetic field first and matches the reference.
    _zero_magnetic_physical!(mag)
    GeoDynamo.compute_solver_nonlinear_terms!(state)
    prod_tor, prod_pol = _velnl(vel)

    @test isapprox(prod_tor, ref_tor; rtol = 1e-10, atol = 1e-12)
    @test isapprox(prod_pol, ref_pol; rtol = 1e-10, atol = 1e-12)
end
