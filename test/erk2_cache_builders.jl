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

    @testset "create_solver_erk2_scalar_cache (dense, all 4 BC codes)" begin
        for bc_code in 1:4
            cache = GeoDynamo.create_solver_erk2_scalar_cache(
                Float64, cfg, dom, diffusivity, dt, bc_code; use_krylov = false)
            check_stage_cache(cache; expect_krylov = false)
            # Dirichlet rows (index 1 and nr) are zeroed in the operator, so the
            # dense exponential E_full has identity-like boundary rows: row 1 of
            # exp(0-row operator) == e_1, row nr == e_nr. Verify for first mode.
            E = cache.E_full[1]
            @test E[1, 1] ≈ 1.0 atol = 1e-12
            @test E[nr, nr] ≈ 1.0 atol = 1e-12
            @test all(abs.(E[1, 2:nr]) .< 1e-12)
            @test all(abs.(E[nr, 1:(nr - 1)]) .< 1e-12)
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
        cache = GeoDynamo.create_solver_erk2_magnetic_toroidal_cache(
            Float64, cfg, dom, diffusivity, dt; use_krylov = false)
        check_stage_cache(cache; expect_krylov = false)
        # embedded homogeneous Dirichlet rows -> identity boundary rows in E_full
        E = cache.E_full[1]
        @test E[1, 1] ≈ 1.0 atol = 1e-12
        @test E[nr, nr] ≈ 1.0 atol = 1e-12
    end

    @testset "create_solver_erk2_magnetic_poloidal_cache" begin
        cache = GeoDynamo.create_solver_erk2_magnetic_poloidal_cache(
            Float64, cfg, dom, diffusivity, dt; use_krylov = false)
        check_stage_cache(cache; expect_krylov = false)
        # The insulating poloidal builder overwrites boundary rows with
        # first-derivative stencils (NOT zero rows), so the boundary rows of the
        # operator are generally non-trivial -> E_full boundary rows differ from
        # the pure-Dirichlet identity. Just assert finiteness (covered above) and
        # that the matrices are genuinely non-identity in the interior.
        E = cache.E_full[1]
        @test !(E ≈ Matrix{Float64}(I, nr, nr))
    end

    @testset "solver_create_dirichlet_bc" begin
        bc = GeoDynamo.solver_create_dirichlet_bc(Float64, nr)
        @test bc.type == :dirichlet
        @test bc.value == 0.0
        @test length(bc.stencil) == nr
        @test all(bc.stencil .== 0.0)
        @test bc.use_l_correction == false
        @test bc.l0_dirichlet == false
        # with a nonzero endpoint value
        bc2 = GeoDynamo.solver_create_dirichlet_bc(Float64, nr, 1.25)
        @test bc2.value == 1.25
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
        d2 = GeoDynamo.build_radial_derivative_matrix(Float64, 2, dom)
        d2_inner = GeoDynamo.extract_dense_row(d2.data, d2.bandwidth, nr, 1)
        bc = GeoDynamo.solver_create_stress_free_pol_bc(Float64, d2_inner)
        @test bc.type == :stress_free_pol
        @test bc.stencil == d2_inner
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
        @test bc.fixed_correction == 0.0
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
        @test bc.fixed_correction == r_inv
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
    end
end
