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

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_TOPO_COUPLING && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
