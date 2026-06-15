# Regression guard for the live scalar nonlinear assembly's RADIAL gradient.
#
# `compute_all_gradients_spectral!` (src/physics/nonlinear.jl, SolverGradientWorkspace
# method) now computes ONLY the radial spectral gradient. The θ/φ spectral
# recurrences (and the `apply_geometric_factors_spectral!` 1/r rescale that only
# touches them) were dropped because the live consumer
# `transform_field_and_gradients_to_physical!` reads ONLY `ws.∇r_spec` — it
# recomputes the EXACT tangential physical gradient via raw sphtor synthesis of
# the scalar's own coefficients (S = 𝔽.spectral, T = 0), never reading
# `ws.∇θ_spec` / `ws.∇φ_spec`. Dropping the discarded θ/φ recurrences also
# removes 2 wasted MPI Allreduces per scalar field per step on multi-rank runs.
#
# This test pins the PHYSICS RESULT (unchanged by the refactor): a purely-radial
# l=0 temperature profile advected by a purely-radial flow must yield the
# expected nonzero radial-advection term `-(u_r · ∂r T)`. If the radial gradient
# stopped being computed, ws.∇r_spec → 0 ⇒ the advection collapses to 0 and this
# fails. The fixture is axisymmetric (l=0 only), so the result is identical under
# the old (θ/φ computed) and new (θ/φ skipped) code paths.
#
# NOTE: not registered in test/runtests.jl yet (shared-file ownership) — wire in
# alongside the other nonlinear suites.

using Test
using MPI
using GeoDynamo

@testset "live scalar nonlinear uses the radial gradient" begin
    MPI.Finalized() && return
    MPI.Initialized() || MPI.Init()

    params = GeoDynamo.SolverParameters(
        architecture = :cpu,
        geometry = :shell,
        nr = 16,
        nr_inner = 4,
        lmax = 4,
        mmax = 4,
        nlat = 12,
        nlon = 16,
        timestep = 1e-4,
        stop_iteration = 4,
        include_magnetic = false,
        include_composition = false,
        timestepper = GeoDynamo.CNAB2(),
    )
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_fields!(state)
    # Warm up: compile the hot paths and populate the per-config transform buffers.
    GeoDynamo.solver_step!(state)

    temp = state.fields.temperature
    vel  = state.fields.velocity
    ws   = state.runtime.gradient_workspace
    cfg  = state.backend.shtns_config
    dom  = state.runtime.outer_core_domain

    # --- Fixture: a purely-radial l=0 temperature ramp T(r) = r -----------------
    # Only the (l=0, m=0) coefficient is populated, so the tangential (θ/φ)
    # gradients vanish identically and the entire advection comes from the
    # radial gradient ∂r T.
    real_data = parent(temp.spectral.data_real)
    imag_data = parent(temp.spectral.data_imag)
    fill!(real_data, 0.0)
    fill!(imag_data, 0.0)
    m00 = GeoDynamo.get_mode_index(cfg, 0, 0)
    slot00 = GeoDynamo.local_spectral_storage_slot(cfg, m00)
    @test slot00 !== nothing
    r_range = GeoDynamo.local_range(cfg.pencils.spec, 3)
    for (lr, gr) in enumerate(r_range)
        lr <= size(real_data, 3) || continue
        GeoDynamo.set_local_spectral_value!(real_data, slot00, lr, Float64(dom.r[gr, 4]))
    end

    # --- Fixture: purely-radial advecting flow u_r = 1, u_θ = u_φ = 0 -----------
    fill!(parent(vel.velocity.r_component.data), 1.0)
    fill!(parent(vel.velocity.θ_component.data), 0.0)
    fill!(parent(vel.velocity.φ_component.data), 0.0)

    # Isolate the advection term: no internal heating so advection == -u·∇T.
    fill!(temp.internal_sources, 0.0)

    # --- Run the live scalar nonlinear assembly ---------------------------------
    GeoDynamo.solver_compute_temperature_nonlinear!(temp, vel, dom, ws)

    ∇r_spec = parent(ws.∇r_spec.data_real)
    grad_r  = parent(temp.gradient.r_component.data)
    adv     = parent(temp.advection_physical.data)
    nl_real = parent(temp.nonlinear.data_real)

    # Radial spectral gradient is actually computed (nonzero, finite).
    @test all(isfinite, ∇r_spec)
    @test maximum(abs, ∇r_spec) > 0
    # For T(r)=r the discrete ∂r is ≈ 1 in the (0,0) coefficient at interior nodes.
    interior_slot_vals = [GeoDynamo.local_spectral_value(∇r_spec, slot00, lr)
                          for lr in 2:(length(r_range) - 1)]
    @test all(v -> isapprox(v, 1.0; atol = 1e-6), interior_slot_vals)

    # Radial gradient is synthesized to a nonzero physical field.
    @test all(isfinite, grad_r)
    @test maximum(abs, grad_r) > 0

    # The radial-advection contribution is the expected nonzero -(u_r·∂rT).
    @test all(isfinite, adv)
    @test maximum(abs, adv) > 0
    @test adv ≈ -grad_r atol = 1e-10

    # The product is analyzed back to a nonzero spectral nonlinear term.
    @test maximum(abs, nl_real) > 0

    # The live path leaves the workspace TANGENTIAL spectral gradients zeroed: it
    # no longer fills ws.∇θ_spec / ws.∇φ_spec (θ/φ physical gradients come from a
    # separate raw sphtor synthesis downstream).
    @test all(iszero, parent(ws.∇θ_spec.data_real))
    @test all(iszero, parent(ws.∇θ_spec.data_imag))
    @test all(iszero, parent(ws.∇φ_spec.data_real))
    @test all(iszero, parent(ws.∇φ_spec.data_imag))
end
