using Test
using LinearAlgebra

# Unit tests for the ERK2 cache builders and boundary-condition factory
# functions in src/timestep/erk2.jl. Each test constructs a small radial domain
# + SHTnsKit config and asserts concrete properties (matrix sizes, finiteness,
# boundary-descriptor field values) of the returned structs.

@testset "ERK2 cache builders + BC factories" begin
    # Shared fixture: a small shell with lmax=2 so unique(l_values) = [0,1,2]
    # (exercises l-dependent operator/BC code for more than just l=0) and an
    # (l=1, m=0) mode exists for the rotating-inner-core branch.
    nr = 5
    dom = GeoDynamo.create_radial_domain(nr)
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = 2, mmax = 2, nlat = 6, nlon = 6, nr = dom.N, optimize_decomp = false)
    nl = length(unique(cfg.l_values))   # == 3
    @test nl == 3

    diffusivity = 0.7
    dt = 0.05

    # ---- helper: assert a freshly built ERK2StageCache is well-formed ----
    function check_stage_cache(cache; expect_krylov::Bool)
        @test cache.nr == nr
        @test cache.dt == dt
        @test cache.diffusivity == diffusivity
        @test cache.use_krylov == expect_krylov
        @test cache.mpi_consistent == true
        @test cache.l_values == unique(cfg.l_values)
        # one matrix per unique l-mode, each square (nr, nr) and finite
        for fld in (cache.E_half, cache.E_full,
                    cache.phi1_half, cache.phi1_full, cache.phi2_full)
            @test length(fld) == nl
            for mat in fld
                @test size(mat) == (nr, nr)
                @test all(isfinite, mat)
            end
        end
    end

    # Every dense ERK2 cache eliminates its endpoint constraints from the
    # generator, so the defining property of a propagator is that its OUTPUT
    # satisfies the constraint rows — for any input, without help from the
    # trailing endpoint projection. That is the invariant to pin; the boundary
    # rows of E themselves are overwritten by the projection in
    # prepare_/finalize_solver_erk2_field! and carry no meaning.
    function check_constraint_preserved(cache, spec, l)
        idx = findfirst(==(l), cache.l_values)
        row_in = GeoDynamo.Solver.solver_erk2_constraint_row(
            Float64, spec.inner, 1, l, nr)
        row_out = GeoDynamo.Solver.solver_erk2_constraint_row(
            Float64, spec.outer, nr, l, nr)
        v = [sin(3.1 * i / nr) + 0.4 * cos(7.7 * i / nr) for i in 1:nr]
        for M in (cache.E_half[idx], cache.E_full[idx],
            cache.phi1_half[idx], cache.phi1_full[idx], cache.phi2_full[idx])
            y = M * v
            scale = max(maximum(abs, y), 1e-300)
            @test abs(dot(row_in, y)) < 1e-8 * scale * maximum(abs, row_in)
            @test abs(dot(row_out, y)) < 1e-8 * scale * maximum(abs, row_out)
        end
    end

    @testset "create_solver_erk2_scalar_cache (dense, all 4 BC codes)" begin
        for bc_code in 1:4
            spec = GeoDynamo.Solver.build_solver_erk2_scalar_bc(Float64, dom, bc_code)
            cache = GeoDynamo.create_solver_erk2_scalar_cache(
                Float64, cfg, dom, diffusivity, dt, bc_code; use_krylov = false)
            check_stage_cache(cache; expect_krylov = false)
            # Neumann ends are derivative rows: the propagator only satisfies them
            # because the boundary DOFs were eliminated, not because the trailing
            # projection patched them up one step late.
            for l in cache.l_values
                check_constraint_preserved(cache, spec, l)
            end
            # Pure Dirichlet-Dirichlet (code 1) pins both endpoints to their given
            # values, so the eliminated propagator carries exactly-zero boundary
            # rows (endpoints do not feed back into the interior evolution).
            if bc_code == 1
                E = cache.E_full[1]
                @test all(E[1, :] .== 0.0)
                @test all(E[nr, :] .== 0.0)
            end
        end
    end

    @testset "create_solver_erk2_scalar_cache (krylov path)" begin
        cache = GeoDynamo.create_solver_erk2_scalar_cache(
            Float64, cfg, dom, diffusivity, dt, 1; use_krylov = true, m = 12, tol = 1e-9)
        check_stage_cache(cache; expect_krylov = true)
        @test cache.krylov_m == 12
        @test cache.krylov_tol == 1e-9
        # In the krylov path the five matrix lists all alias the same raw
        # operator (operator_dense), not exponentials — so they are equal.
        @test cache.E_half[1] == cache.E_full[1]
        @test cache.E_full[1] == cache.phi1_full[1]
        @test cache.phi1_full[1] == cache.phi2_full[1]
        # boundary rows of the raw operator were zeroed
        @test all(cache.E_full[1][1, :] .== 0.0)
        @test all(cache.E_full[1][nr, :] .== 0.0)
    end

    @testset "create_solver_erk2_magnetic_toroidal_cache" begin
        spec = GeoDynamo.Solver.build_solver_erk2_magnetic_tor_bc(Float64, dom)
        cache = GeoDynamo.create_solver_erk2_magnetic_toroidal_cache(
            Float64, cfg, dom, diffusivity, dt; use_krylov = false)
        check_stage_cache(cache; expect_krylov = false)
        # Homogeneous Dirichlet: the constraint pins the endpoints to exactly
        # zero, so the eliminated propagators carry zero boundary rows.
        E = cache.E_full[1]
        @test all(E[1, :] .== 0.0)
        @test all(E[nr, :] .== 0.0)
        for l in cache.l_values
            check_constraint_preserved(cache, spec, l)
        end
    end

    @testset "create_solver_erk2_magnetic_poloidal_cache" begin
        spec = GeoDynamo.Solver.build_solver_erk2_magnetic_pol_bc(Float64, dom)
        cache = GeoDynamo.create_solver_erk2_magnetic_poloidal_cache(
            Float64, cfg, dom, diffusivity, dt; use_krylov = false)
        check_stage_cache(cache; expect_krylov = false)
        E = cache.E_full[1]
        @test !(E ≈ Matrix{Float64}(I, nr, nr))
        # Robin rows on both walls: unlike Dirichlet, these are only satisfied
        # because the boundary DOFs were eliminated. Exponentiating the stamped
        # rows instead leaves residuals of the same order as the field.
        for l in cache.l_values
            check_constraint_preserved(cache, spec, l)
        end
    end

    @testset "solver_create_dirichlet_bc" begin
        bc = GeoDynamo.solver_create_dirichlet_bc(Float64, nr)
        @test bc.type == :dirichlet
        @test bc.value == 0.0
        @test length(bc.stencil) == nr
        @test all(bc.stencil .== 0.0)
        @test bc.use_l_correction == false
        # with a nonzero endpoint value
        bc2 = GeoDynamo.solver_create_dirichlet_bc(Float64, nr, 1.25)
        @test bc2.value == 1.25
    end

    @testset "no l=0 Dirichlet pin on ERK2 boundary sides" begin
        # The NN gauge pin belongs to the SINGULAR STEADY conductive solve
        # (scalar_field_solver_common.jl), not to the time-stepping operator.
        # ERK2 used to carry an `l0_dirichlet` flag that pinned l=0 under NN: it
        # enforced the prescribed FLUX as a field VALUE and diverged from the
        # CNAB2/RK3 matrix path. The flag is gone — this guards its return.
        @test !(:l0_dirichlet in fieldnames(GeoDynamo.SolverERK2BoundarySide))

        # Behavioural half: under NN the inner constraint row at l=0 must be the
        # SAME Neumann stencil as any other degree, not an identity pin.
        spec = GeoDynamo.build_solver_erk2_scalar_bc(Float64, dom, 4)
        row0 = GeoDynamo.solver_erk2_constraint_row(Float64, spec.inner, 1, 0, nr)
        row1 = GeoDynamo.solver_erk2_constraint_row(Float64, spec.inner, 1, 1, nr)
        @test row0 == row1                    # no l=0 special case
        @test count(!iszero, row0) > 1        # a stencil row, not an identity pin
    end

    @testset "solver_create_stress_free_tor_bc" begin
        d1 = GeoDynamo.build_radial_derivative_matrix(Float64, 1, dom)
        d1_inner = GeoDynamo.extract_dense_row(d1.data, d1.bandwidth, nr, 1)
        r_inv = Float64(dom.r[1, 3])
        bc = GeoDynamo.solver_create_stress_free_tor_bc(Float64, d1_inner, r_inv)
        @test bc.type == :stress_free_tor
        @test bc.value == 0.0
        @test bc.stencil == d1_inner           # copy of the d1 row
        @test bc.stencil !== d1_inner          # but a distinct array (copied)
        @test bc.r_inv == r_inv
        @test bc.fixed_correction == -r_inv
        @test bc.use_l_correction == false
        @test all(isfinite, bc.stencil)
    end

    @testset "solver_create_noslip_pol_bc" begin
        d1 = GeoDynamo.build_radial_derivative_matrix(Float64, 1, dom)
        d1_inner = GeoDynamo.extract_dense_row(d1.data, d1.bandwidth, nr, 1)
        bc = GeoDynamo.solver_create_noslip_pol_bc(Float64, d1_inner)
        @test bc.type == :noslip_pol
        @test bc.stencil == d1_inner
        @test bc.r_inv == 0.0
        @test bc.fixed_correction == 0.0
        @test bc.use_l_correction == false
    end

    @testset "solver_create_stress_free_pol_bc" begin
        d1 = GeoDynamo.build_radial_derivative_matrix(Float64, 1, dom)
        d2 = GeoDynamo.build_radial_derivative_matrix(Float64, 2, dom)
        d1_inner = GeoDynamo.extract_dense_row(d1.data, d1.bandwidth, nr, 1)
        d2_inner = GeoDynamo.extract_dense_row(d2.data, d2.bandwidth, nr, 1)
        r_inv = Float64(dom.r[1, 3])
        bc = GeoDynamo.solver_create_stress_free_pol_bc(
            Float64, d1_inner, d2_inner, r_inv)
        @test bc.type == :stress_free_pol
        @test bc.stencil ≈ d2_inner .- 2 * r_inv .* d1_inner
        @test bc.use_l_correction == false
        @test all(isfinite, bc.stencil)
    end

    @testset "solver_create_insulating_inner_bc" begin
        d1 = GeoDynamo.build_radial_derivative_matrix(Float64, 1, dom)
        d1_inner = GeoDynamo.extract_dense_row(d1.data, d1.bandwidth, nr, 1)
        r_inv = Float64(dom.r[1, 3])
        bc = GeoDynamo.solver_create_insulating_inner_bc(Float64, d1_inner, r_inv)
        @test bc.type == :insulating_inner
        @test bc.stencil == d1_inner
        @test bc.r_inv == r_inv
        @test bc.l_sign == -1.0
        @test bc.use_l_correction == true
        # Corrected insulating inner row (2026-06-11 audit): (∂r − (l+1)/r)P = 0
        # under B_r = λP/r² ⇒ fixed_correction = −r_inv (was 0 for ∂r − l/r).
        @test bc.fixed_correction == -r_inv
    end

    @testset "solver_create_insulating_outer_bc" begin
        d1 = GeoDynamo.build_radial_derivative_matrix(Float64, 1, dom)
        d1_outer = GeoDynamo.extract_dense_row(d1.data, d1.bandwidth, nr, nr)
        r_inv = Float64(dom.r[nr, 3])
        bc = GeoDynamo.solver_create_insulating_outer_bc(Float64, d1_outer, r_inv)
        @test bc.type == :insulating_outer
        @test bc.stencil == d1_outer
        @test bc.r_inv == r_inv
        @test bc.l_sign == 1.0
        @test bc.use_l_correction == true
        # Corrected insulating outer row (2026-06-11 audit): (∂r + l/r)P = 0
        # under B_r = λP/r² ⇒ fixed_correction = 0 (was r_inv for ∂r + (l+1)/r);
        # verified by the dipole free-decay rate σ = π² (ball_bessel_decay.jl).
        @test bc.fixed_correction == 0.0
    end

    @testset "build_solver_erk2_velocity_tor_bc (no-slip, rot_omega=0)" begin
        spec = GeoDynamo.build_solver_erk2_velocity_tor_bc(
            Float64, dom, 1; config = cfg, rot_omega = 0.0)
        # bc_code 1 -> Dirichlet on both endpoints
        @test spec.inner.type == :dirichlet
        @test spec.outer.type == :dirichlet
        # rot_omega == 0 -> no mode-dependent endpoint values
        @test spec.inner_mode_values === nothing
        @test spec.outer_mode_values === nothing
    end

    @testset "build_solver_erk2_velocity_tor_bc (stress-free, code=4)" begin
        spec = GeoDynamo.build_solver_erk2_velocity_tor_bc(
            Float64, dom, 4; config = cfg, rot_omega = 0.0)
        # bc_code 4 -> stress-free on both endpoints
        @test spec.inner.type == :stress_free_tor
        @test spec.outer.type == :stress_free_tor
        @test spec.inner.r_inv == Float64(dom.r[1, 3])
        @test spec.outer.r_inv == Float64(dom.r[nr, 3])
        @test spec.inner_mode_values === nothing
    end

    @testset "build_solver_erk2_velocity_tor_bc (rotating inner core)" begin
        rot_omega = 2.5
        spec = GeoDynamo.build_solver_erk2_velocity_tor_bc(
            Float64, dom, 1; config = cfg, rot_omega = rot_omega)
        @test spec.inner.type == :dirichlet
        @test spec.inner_mode_values !== nothing
        @test length(spec.inner_mode_values) == length(cfg.l_values)
        # only the (l=1, m=0) mode carries a nonzero endpoint value
        r_inner = Float64(dom.r[1, 4])
        expected = rot_omega * r_inner
        for i in eachindex(cfg.l_values)
            if cfg.l_values[i] == 1 && cfg.m_values[i] == 0
                @test spec.inner_mode_values[i] ≈ expected atol = 1e-12
            else
                @test spec.inner_mode_values[i] == 0.0
            end
        end
        # exactly one nonzero entry
        @test count(!iszero, spec.inner_mode_values) == 1
    end

    @testset "build_solver_erk2_velocity_pol_bc (no-slip & stress-free)" begin
        # code 1 -> no-slip both endpoints
        spec1 = GeoDynamo.build_solver_erk2_velocity_pol_bc(Float64, dom, 1)
        @test spec1.inner.type == :noslip_pol
        @test spec1.outer.type == :noslip_pol
        @test spec1.inner_mode_values === nothing
        # code 4 -> stress-free both endpoints
        spec4 = GeoDynamo.build_solver_erk2_velocity_pol_bc(Float64, dom, 4)
        @test spec4.inner.type == :stress_free_pol
        @test spec4.outer.type == :stress_free_pol
        # stencil lengths match nr
        @test length(spec1.inner.stencil) == nr
        @test length(spec4.outer.stencil) == nr

        d1 = GeoDynamo.build_radial_derivative_matrix(Float64, 1, dom)
        d2 = GeoDynamo.build_radial_derivative_matrix(Float64, 2, dom)
        d1_inner = GeoDynamo.extract_dense_row(d1.data, d1.bandwidth, nr, 1)
        d1_outer = GeoDynamo.extract_dense_row(d1.data, d1.bandwidth, nr, nr)
        d2_inner = GeoDynamo.extract_dense_row(d2.data, d2.bandwidth, nr, 1)
        d2_outer = GeoDynamo.extract_dense_row(d2.data, d2.bandwidth, nr, nr)
        expected_inner = d2_inner .- 2 * Float64(dom.r[1, 3]) .* d1_inner
        expected_outer = d2_outer .- 2 * Float64(dom.r[nr, 3]) .* d1_outer
        @test spec4.inner.stencil ≈ expected_inner
        @test spec4.outer.stencil ≈ expected_outer
    end
end
