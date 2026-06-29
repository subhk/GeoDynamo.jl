using Test
using MPI
using LinearAlgebra

const FINALIZE_MPI_TOPO_COUPLING = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# Shorthand for the internal topography submodule
const topocpl = GeoDynamo.bcs.topography

# Coverage for the previously-untested topography coupling kernels:
#   * bcs/topography/thermal_coupling.jl  — apply_thermal/composition_topography_correction!
#                                            + assemble_thermal_boundary_operator
#   * bcs/topography/velocity_coupling.jl — apply_velocity_topography_correction!
#                                            + assemble_velocity_boundary_operator
#   * bcs/topography/magnetic_coupling.jl — apply_magnetic_topography_correction!
#                                            + assemble_magnetic_boundary_operator
#
# The apply_* functions are driven with real solver-state fields (which carry the
# .spectral / .bc_type_* / .∂r / .∂²r / .domain metadata they probe) plus an
# explicitly constructed ICB+CMB topography and Gaunt cache.
@testset "Topography coupling" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping topography coupling tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    L = 4
    params = GeoDynamo.SolverParameters(
        architecture = :cpu,
        geometry = :shell,
        nr = 16,
        nr_inner = 4,
        lmax = L,
        mmax = L,
        nlat = 12,
        nlon = 16,
        Ra = 1e4,
        Ek = 1e-2,
        Pr = 1.0,
        Pm = 1.0,
        timestep = 1e-4,
        start_time = 0.0,
        end_time = 1e-3,
        stop_iteration = 10,
        include_magnetic = true,
        include_composition = true,
        timestepper = GeoDynamo.CNAB2(),
        topography_enabled = false,
        stefan_enabled = false
    )
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_fields!(state)

    # Give the spectral fields nonzero content so the boundary-derivative caches
    # carry real ∂_r / ∂²_r traces through the coupling sums.
    for f in (state.fields.temperature.spectral,
        state.fields.composition.spectral,
        state.fields.velocity.toroidal, state.fields.velocity.poloidal,
        state.fields.magnetic.toroidal, state.fields.magnetic.poloidal)
        fill!(parent(f.data_real), 0.01)
        fill!(parent(f.data_imag), 0.0)
    end

    # Build ICB + CMB topography with a couple of nonzero modes and a Gaunt cache.
    nlm_topo = topocpl.lm_to_index(L, L, L)          # 15 modes for lmax = 4
    cmb_coeffs = zeros(ComplexF64, nlm_topo)
    icb_coeffs = zeros(ComplexF64, nlm_topo)
    cmb_coeffs[topocpl.lm_to_index(0, 0, L)] = 0.10  # axisymmetric offset
    cmb_coeffs[topocpl.lm_to_index(2, 0, L)] = 0.05  # an l=2 bump
    icb_coeffs[topocpl.lm_to_index(0, 0, L)] = 0.08
    icb_coeffs[topocpl.lm_to_index(2, 0, L)] = 0.04

    topodata = topocpl.create_topography_data(
        icb_coeffs = icb_coeffs, cmb_coeffs = cmb_coeffs,
        icb_radius = 0.35, cmb_radius = 1.0, lmax = L, epsilon = 0.02)
    topocpl.initialize_gaunt_cache!(topodata, L)
    @test topodata.gaunt_cache !== nothing
    @test topodata.icb !== nothing
    @test topodata.cmb !== nothing

    config = topocpl.TopographyCouplingConfig(
        enabled = true,
        velocity_coupling = true,
        magnetic_coupling = true,
        thermal_coupling = true,
        epsilon = 0.02,
        include_shift_terms = true,
        include_slope_terms = true)

    @testset "apply_thermal_topography_correction! (Dirichlet + Neumann)" begin
        bv0 = copy(state.fields.temperature.boundary_values)
        # default BCs are Dirichlet; a conductive profile makes the correction nonzero
        @test topocpl.apply_thermal_topography_correction!(
            state.fields.temperature, topodata, config; T_cond = (r) -> r) === nothing
        @test norm(state.fields.temperature.boundary_values .- bv0) > 0

        # flip outer BC to Neumann (anything != Int(DIRICHLET)) to drive that branch
        state.fields.temperature.bc_type_outer .= 2
        @test topocpl.apply_thermal_topography_correction!(
            state.fields.temperature, topodata, config; T_cond = (r) -> r) === nothing

        # disabled config is a no-op
        off = topocpl.TopographyCouplingConfig(enabled = false)
        bv1 = copy(state.fields.temperature.boundary_values)
        @test topocpl.apply_thermal_topography_correction!(
            state.fields.temperature, topodata, off) === nothing
        @test state.fields.temperature.boundary_values == bv1
    end

    @testset "thermal topography correction does not drift across steps" begin
        # The correction is a LAGGED function of the current field state, applied
        # every step. Applying it twice WITHOUT changing the field state must be
        # idempotent: the base boundary row is re-established each step, so the
        # result is `base - ε·corr(state)`, not `base - 2·ε·corr(state)`. The bug
        # was an in-place `-=` with no per-step base reset ⇒ unbounded drift.
        temp = state.fields.temperature
        temp.bc_type_outer .= Int(GeoDynamo.DIRICHLET)  # deterministic branch
        Tc = (r) -> r

        topocpl.apply_thermal_topography_correction!(temp, topodata, config; T_cond = Tc)
        bv_after_1 = copy(temp.boundary_values)

        topocpl.apply_thermal_topography_correction!(temp, topodata, config; T_cond = Tc)
        bv_after_2 = copy(temp.boundary_values)

        @test bv_after_2 ≈ bv_after_1
    end

    @testset "apply_composition_topography_correction! reuses the thermal path" begin
        @test topocpl.apply_composition_topography_correction!(
            state.fields.composition, topodata, config) === nothing
    end

    @testset "apply_velocity_topography_correction!" begin
        @test topocpl.apply_velocity_topography_correction!(
            state.fields.velocity, topodata, config) === nothing
    end

    @testset "apply_magnetic_topography_correction!" begin
        @test topocpl.apply_magnetic_topography_correction!(
            state.fields.magnetic, topodata, config) === nothing
    end

    @testset "assemble_*_boundary_operator diagonal structure" begin
        gaunt = topodata.gaunt_cache
        cmb = topodata.cmb

        # thermal: Dirichlet -> identity on Θ, Neumann -> identity on ∂Θ
        op_d = topocpl.assemble_thermal_boundary_operator(2, cmb, gaunt, config, :dirichlet)
        @test op_d[(2, 0, :Θ)] == one(ComplexF64)
        @test all(haskey(op_d, (2, m, :Θ)) for m in -2:2)

        op_n = topocpl.assemble_thermal_boundary_operator(2, cmb, gaunt, config, :neumann)
        @test op_n[(2, 0, :dΘ)] == one(ComplexF64)

        # velocity impermeability: diagonal l(l+1)/r_b^2 on poloidal
        op_v = topocpl.assemble_velocity_boundary_operator(2, cmb, gaunt, config, :impermeability)
        @test op_v[(2, 0, :P)] ≈ ComplexF64(2 * 3 / 1.0^2)

        # magnetic CMB (outer): flat diagonal is ∂P + l/r_o P = 0, T = 0
        # (insulating row under B_r = λP/r², exterior P ∝ r^{-l} — same
        # convention as src/bcs/magnetic_bc.jl).
        # (The full config also folds topography coupling into these same keys, so
        # the clean diagonal is checked with shift/slope terms switched off.)
        flat = topocpl.TopographyCouplingConfig(
            enabled = true, include_shift_terms = false,
            include_slope_terms = false, epsilon = 0.02)
        op_m_flat = topocpl.assemble_magnetic_boundary_operator(
            2, cmb, gaunt, flat, GeoDynamo.OUTER_BOUNDARY)
        @test op_m_flat[(2, 0, :dP)] == one(ComplexF64)
        @test op_m_flat[(2, 0, :P)] ≈ ComplexF64(2 / 1.0)
        @test op_m_flat[(2, 0, :T)] == one(ComplexF64)

        # with coupling enabled the operator gains off-diagonal entries
        op_m = topocpl.assemble_magnetic_boundary_operator(
            2, cmb, gaunt, config, GeoDynamo.OUTER_BOUNDARY)
        @test length(op_m) >= length(op_m_flat)
        @test haskey(op_m, (2, 0, :dP))
    end

    # The fill-with-a-constant tests above cannot see two mode-indexing bugs because
    # every spectral slot holds the same value. The two testsets below excite a SINGLE
    # non-axisymmetric mode so a scrambled read or a doubled write becomes visible.
    @testset "Bug 1: boundary-derivative cache reads modes in canonical m-major order" begin
        # The cache is FILLED in canonical m-major order (mode index 1:nlm via
        # local_spectral_storage_slot) but used to be READ with the l-major lm_to_index,
        # so get_cache_value(l,m) returned a different mode's data.
        pol = state.fields.velocity.poloidal
        cfg = pol.config

        # Excite ONLY canonical mode (l=2, m=1) with a known constant radial profile.
        fill!(parent(pol.data_real), 0.0)
        fill!(parent(pol.data_imag), 0.0)
        src = topocpl.get_mode_index(cfg, 2, 1)          # canonical (m-major) index
        @test src > 0
        sslot = topocpl.local_spectral_storage_slot(cfg, src)
        @test sslot !== nothing
        V = 0.7
        parent(pol.data_real)[sslot[1], sslot[2], :] .= V

        cache = topocpl.compute_boundary_derivative_cache(
            pol, state.fields.velocity.∂r, state.fields.velocity.∂²r,
            state.fields.velocity.domain)

        # The two index conventions genuinely disagree for (2,1) — that disagreement is
        # exactly the bug. (m-major canonical index 7 vs l-major lm_to_index 5 at lmax 4.)
        @test topocpl.lm_to_index(2, 1, L) != src

        # Correct mode (2,1) must return V. Under the l-major bug it read the wrong slot
        # (an unexcited mode) and returned 0.
        @test real(topocpl.get_cache_value(cache, 2, 1, GeoDynamo.OUTER_BOUNDARY)) ≈ V

        # (3,0) is the mode the buggy l-major index aliased onto slot `src`
        # (lm_to_index(3,0,4) == 7 == canonical index of (2,1)); the old code therefore
        # returned V here. The fixed canonical read must return 0 (this mode is unexcited).
        @test real(topocpl.get_cache_value(cache, 3, 0, GeoDynamo.OUTER_BOUNDARY)) ≈ 0 atol=1e-12
        # Another unexcited mode also reads zero.
        @test real(topocpl.get_cache_value(cache, 1, 0, GeoDynamo.OUTER_BOUNDARY)) ≈ 0 atol=1e-12
    end

    @testset "Bug 2: apply loop writes each m>=0 slot once (no +/-m double-application)" begin
        # The apply loops used to iterate m in -l:l and write lm_to_spectral_index(l, m),
        # which maps +m and -m to the SAME m>=0 storage slot — so each m != 0 slot was
        # corrected twice. After the fix the OUTER (CMB) row delta for every non-axisymmetric
        # target must equal exactly ONE application of the per-mode correction.
        vel = state.fields.velocity
        pol = vel.poloidal
        tor = vel.toroidal
        cfg = pol.config

        # Single source mode (l=2, m=1) in the poloidal field, toroidal zeroed.
        fill!(parent(pol.data_real), 0.0)
        fill!(parent(pol.data_imag), 0.0)
        fill!(parent(tor.data_real), 0.0)
        fill!(parent(tor.data_imag), 0.0)
        src = topocpl.get_mode_index(cfg, 2, 1)
        sslot = topocpl.local_spectral_storage_slot(cfg, src)
        @test sslot !== nothing
        parent(pol.data_real)[sslot[1], sslot[2], :] .= 0.7

        # Caches exactly as apply_velocity_topography_correction! builds them internally
        # (it does not modify field data, only boundary_values, so these match its caches).
        p_cache = topocpl.compute_boundary_derivative_cache(pol, vel.∂r, vel.∂²r, vel.domain)
        t_cache = topocpl.compute_boundary_derivative_cache(tor, vel.∂r, vel.∂²r, vel.domain)

        ro = topodata.cmb.radius
        @test topocpl.apply_velocity_topography_correction!(vel, topodata, config) === nothing
        bv_once = copy(pol.boundary_values)

        # The correction now re-establishes the base each call (no per-step drift),
        # so the CMB-row (row 2) absolute value equals a SINGLE impermeability
        # correction on top of the zero velocity base. The poloidal boundary row only
        # ever receives the impermeability term (no-slip/stress-free go to the toroidal
        # row), so this is an exact comparison; a +/-m double-application would double it.
        nonzero_seen = false
        for l in 1:L
            for m in 1:min(l, L)
                idx = topocpl.get_mode_index(cfg, l, m)
                (idx <= 0 || idx > size(pol.boundary_values, 2)) && continue
                imp = topocpl.compute_impermeability_correction(
                    l, m, p_cache, t_cache, topodata.cmb, topodata.gaunt_cache,
                    ro, GeoDynamo.OUTER_BOUNDARY, config)
                expected = -config.epsilon * real(imp) * ro^2 / (l * (l + 1))
                actual = pol.boundary_values[2, idx]
                @test actual ≈ expected atol=1e-12 rtol=1e-9
                abs(expected) > 1e-10 && (nonzero_seen = true)
            end
        end

        # Idempotent re-application: applying again from the same field state must
        # reproduce the same boundary rows (drift fix), not accumulate.
        @test topocpl.apply_velocity_topography_correction!(vel, topodata, config) === nothing
        @test pol.boundary_values ≈ bv_once
        # Non-vacuous: at least one m>0 correction is genuinely nonzero, so the equality
        # checks above would fail under the old double-application (which adds the spurious
        # -m pass on top of the +m correction).
        @test nonzero_seen
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_TOPO_COUPLING && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
