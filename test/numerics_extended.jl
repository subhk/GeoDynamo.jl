using Test
using LinearAlgebra

# Extended unit coverage for src/solver/numerics.jl — targets the residual
# uncovered numeric helpers NOT already exercised by:
#   * test/numerics_phi_functions.jl       (phi1_series / phi2_series / rcond_estimate)
#   * test/numerics_banded_and_vector.jl   (solver_build_banded_A / solver_factorize_banded /
#                                           solve_banded! / apply_banded_full! /
#                                           solver_phi1_action_krylov main path / cpu_* vector helpers)
#   * test/imex_implicit_steps.jl          (timestep/imex.jl implicit-step solvers)
#
# Functions covered here (all in src/solver/numerics.jl):
#   * build_radial_derivative_matrix  — finite-difference banded derivative operator
#   * krylov_exp_action!              — workspace Arnoldi exp(dt·A)·v  (was 0% covered)
#   * phi1_action_krylov!            — workspace + delegate-to-allocating φ₁ action
#   * GeoDynamo.exp_action_krylov / exp_action_krylov!  — root-level reusable wrappers
#   * solver_phi1_action_krylov      — the small-dt linearised early-return branch
#   * solver_compute_phi1_function   — tiny-norm series branch + well-conditioned LU path
#   * solver_compute_phi2_function   — tiny-norm series branch + well-conditioned LU path
#   * reset_solver_phi2_monitor!     — conditioning-monitor reset
#   * report_solver_phi2_conditioning — periodic report (rank-0, resets the monitor)
#   * solver_enforce_ball_vector_regularity! — zero l≥1 modes at the ball centre (r=1)
#   * scalar_field_data_and_config   — scalar-field container dispatch (all 3 branches + error)
#
# QUALITY: every @test pins a KNOWN-CORRECT value — an analytic derivative, a dense
# matrix-exponential reference, a closed-form scalar φ factor, an exact zeroing/identity,
# or an exact field equality. Deterministic; single-rank (these paths are comm-free).
#
# SKIPPED (not reachable serially without pathological/over/underflow input — documented
# in the agent report): the φ₁/φ₂ "non-finite LU result → series" fallbacks
# (numerics.jl 1956-1957 / 2053-2054), the φ₁/φ₂ "series ALSO threw → identity/zero"
# inner-catch (1965-1966 / 2066-2067), domain_bandwidth's radial_laplacian fallback +
# error (188-191), build_radial_derivative_matrix's insufficient-stencil / Vandermonde-
# failure errors (221-225 / 231-235), get_bc_vectors' loaded-cache + all-nothing branches
# (135 / 153), and GPU / multi-rank-Allreduce paths.

const _NE = GeoDynamo

@testset "numerics.jl extended coverage" begin

    # ---------------------------------------------------------------------
    # build_radial_derivative_matrix: differentiate a known cubic polynomial.
    # The finite-difference stencil is exact (to round-off) for polynomials of
    # degree ≤ stencil order at interior nodes, where the stencil is centred.
    # ---------------------------------------------------------------------
    @testset "build_radial_derivative_matrix matches analytic derivative" begin
        nr = 12
        dom = _NE.create_radial_domain(nr)
        r = dom.r[1:nr, 4]                      # physical radial coordinate
        # p(r) = 2 + 3r - r^2 + 0.5 r^3
        p = @. 2 + 3r - r^2 + 0.5 * r^3
        dp_exact = @. 3 - 2r + 1.5 * r^2        # p'
        d2p_exact = @. -2 + 3r                  # p''

        D1 = _NE.build_radial_derivative_matrix(Float64, 1, dom)
        D2 = _NE.build_radial_derivative_matrix(Float64, 2, dom)

        # Operator is the production banded type of order nr.
        @test D1 isa _NE.BandedOperator{Float64}
        @test D1.size == nr
        @test D2.size == nr

        out1 = zeros(Float64, nr)
        out2 = zeros(Float64, nr)
        _NE.apply_radial_derivative!(out1, D1, p)
        _NE.apply_radial_derivative!(out2, D2, p)

        # Interior nodes (one bandwidth in from each end) use a centred stencil;
        # there the derivative of a cubic is reproduced to machine precision.
        bw = D1.bandwidth
        interior = (bw + 1):(nr - bw)
        @test maximum(abs.(out1[interior] .- dp_exact[interior])) < 1.0e-10
        @test maximum(abs.(out2[interior] .- d2p_exact[interior])) < 1.0e-9

        # The single-arg convenience method (default eltype) gives the same operator.
        D1b = _NE.build_radial_derivative_matrix(1, dom)
        @test D1b.data == D1.data
    end

    # ---------------------------------------------------------------------
    # krylov_exp_action! (workspace Arnoldi) — exp(dt·A)·v for a banded operator.
    # Reference: the dense matrix exponential of the same operator.
    # ---------------------------------------------------------------------
    @testset "krylov_exp_action! reproduces the dense matrix exponential" begin
        n = 6
        # Symmetric tridiagonal -2 on diagonal, +1 off-diagonal (bandwidth 1).
        data = zeros(Float64, 3, n)
        for j in 1:n
            data[2, j] = -2.0
        end
        for j in 1:(n - 1)
            data[1, j + 1] = 1.0   # super-diagonal at band row 1
            data[3, j] = 1.0       # sub-diagonal at band row 3
        end
        Aop = _NE.BandedOperator{Float64}(data, 1, n)
        apply_A!(out, v) = _NE.apply_banded_full!(out, Aop, v)
        Adense = _NE.solver_banded_to_dense(Aop)

        v = Float64[1.0, -1.0, 2.0, 0.5, -0.3, 1.1]
        dt = 0.1
        ref = exp(dt .* Adense) * v

        work = _NE.SolverKrylovWork{Float64}()
        dest = similar(v)
        ret = _NE.krylov_exp_action!(dest, apply_A!, v, dt, work; m = 20, tol = 1.0e-12)
        @test ret === dest                                   # writes in place, returns dest
        @test maximum(abs.(dest .- ref)) < 1.0e-10

        # Workspace and allocating variants agree to round-off.
        nonws = _NE.krylov_exp_action(apply_A!, v, dt; m = 20, tol = 1.0e-12)
        @test maximum(abs.(dest .- nonws)) < 1.0e-12

        # Guard branches (all return-early paths):
        z = _NE.krylov_exp_action!(zeros(Float64, n), apply_A!, zeros(Float64, n), dt, work)
        @test z == zeros(Float64, n)                         # zero input ⇒ zero output

        empty_dest = Float64[]
        @test _NE.krylov_exp_action!(empty_dest, apply_A!, Float64[], dt, work) == Float64[]

        # |dt| < eps ⇒ exp(dt·A)v ≈ v (the copy-through branch).
        tiny = _NE.krylov_exp_action!(similar(v), apply_A!, v, 0.0, work)
        @test tiny == v
    end

    # ---------------------------------------------------------------------
    # phi1_action_krylov!  —  φ₁(dt·A)·v.  For A = λ·I this equals
    # ((e^{λΔt}-1)/(λΔt))·v.  Covers BOTH the workspace path and the
    # work===nothing delegate-to-allocating path.
    # ---------------------------------------------------------------------
    @testset "phi1_action_krylov! matches the closed-form φ₁ factor" begin
        n = 4
        λ = -0.3
        dt = 0.05
        data = zeros(Float64, 3, n)            # diagonal λ, bandwidth 1
        for j in 1:n
            data[2, j] = λ
        end
        Aop = _NE.BandedOperator{Float64}(data, 1, n)
        A_lu = _NE.solver_factorize_banded(Aop)
        apply_A!(out, v) = _NE.apply_banded_full!(out, Aop, v)
        v = Float64[1.0, 2.0, -1.0, 0.5]
        factor = (exp(λ * dt) - 1) / (λ * dt)

        work = _NE.SolverKrylovWork{Float64}()
        dest = similar(v)
        ret = _NE.phi1_action_krylov!(dest, apply_A!, A_lu, v, dt; work = work)
        @test ret === dest
        @test maximum(abs.(dest .- factor .* v)) < 1.0e-12

        # work === nothing ⇒ delegates to solver_phi1_action_krylov, same answer.
        dest2 = similar(v)
        _NE.phi1_action_krylov!(dest2, apply_A!, A_lu, v, dt; work = nothing)
        @test maximum(abs.(dest2 .- factor .* v)) < 1.0e-12

        # Zero-input norm guard (workspace path): exact zero out.
        zdest = _NE.phi1_action_krylov!(
            zeros(Float64, n), apply_A!, A_lu, zeros(Float64, n), dt; work = work)
        @test zdest == zeros(Float64, n)
    end

    # ---------------------------------------------------------------------
    # solver_phi1_action_krylov small-dt linearisation early return.
    # When |dt| < 1e-8 AND the local exponential scale is below sqrt(eps),
    # the routine returns the first-order term  v + (dt/2)·A·v  without
    # building a Krylov space.  Use A = I and dt = 1e-12.
    # ---------------------------------------------------------------------
    @testset "solver_phi1_action_krylov small-dt early return" begin
        n = 3
        data = zeros(Float64, 3, n)            # A = identity (bandwidth 1)
        for j in 1:n
            data[2, j] = 1.0
        end
        Aop = _NE.BandedOperator{Float64}(data, 1, n)
        A_lu = _NE.solver_factorize_banded(Aop)
        apply_A!(out, v) = _NE.apply_banded_full!(out, Aop, v)

        v = Float64[1.0, 2.0, 3.0]
        dt = 1.0e-12
        result = _NE.solver_phi1_action_krylov(apply_A!, A_lu, v, dt)
        # Closed form of the branch:  v + (dt/2)·A·v = v + (dt/2)·v.
        expected = v .+ (dt / 2) .* v
        @test result == expected
    end

    # ---------------------------------------------------------------------
    # GeoDynamo.exp_action_krylov / exp_action_krylov!  (root-level wrappers).
    # Both must equal exp(dt·A)·v; exercise the work===nothing and work-supplied
    # branches of the in-place form.
    # ---------------------------------------------------------------------
    @testset "exp_action_krylov root-level wrappers" begin
        n = 4
        data = zeros(Float64, 3, n)            # A = -0.5·I
        for j in 1:n
            data[2, j] = -0.5
        end
        Aop = _NE.BandedOperator{Float64}(data, 1, n)
        apply_A!(out, v) = _NE.apply_banded_full!(out, Aop, v)
        v = Float64[1.0, -1.0, 2.0, 0.5]
        dt = 0.1
        ref = exp(-0.5 * dt) .* v             # scalar exponential on each component

        e = _NE.exp_action_krylov(apply_A!, v, dt)
        @test maximum(abs.(e .- ref)) < 1.0e-12

        d1 = similar(v)
        _NE.exp_action_krylov!(d1, apply_A!, v, dt)               # work === nothing
        @test maximum(abs.(d1 .- ref)) < 1.0e-12

        work = _NE.SolverKrylovWork{Float64}()
        d2 = similar(v)
        _NE.exp_action_krylov!(d2, apply_A!, v, dt; work = work)  # work supplied
        @test maximum(abs.(d2 .- ref)) < 1.0e-12
    end

    # ---------------------------------------------------------------------
    # solver_compute_phi1_function:
    #   (a) tiny-norm branch  ‖A‖ < sqrt(eps)  → series expansion.
    #   (b) well-conditioned LU path  → φ₁(A) = (e^A - I)·A⁻¹.
    # ---------------------------------------------------------------------
    @testset "solver_compute_phi1_function (series + LU paths)" begin
        # (a) tiny norm: equals the series expansion exactly (same code path).
        tiny = 1.0e-10 * Matrix{Float64}(I, 3, 3)
        @test _NE.solver_compute_phi1_function(tiny, exp(tiny)) ==
              _NE.phi1_series(tiny)

        # (b) well-conditioned matrix: LU path equals (expA - I) A⁻¹ ...
        A = [0.5 0.1 0.0
             0.2 0.4 0.1
             0.0 0.1 0.3]
        expA = exp(A)
        phi1 = _NE.solver_compute_phi1_function(A, expA)
        @test maximum(abs.(phi1 .- (expA - I) / A)) < 1.0e-10
        # ... and agrees with the Taylor series for this small, well-conditioned A.
        @test maximum(abs.(phi1 .- _NE.phi1_series(A))) < 1.0e-10

        # Non-finite input throws (validation branch).
        @test_throws ArgumentError _NE.solver_compute_phi1_function(
            fill(NaN, 2, 2), Matrix{Float64}(I, 2, 2))
    end

    # ---------------------------------------------------------------------
    # solver_compute_phi2_function:
    #   (a) tiny-norm branch → series.
    #   (b) well-conditioned LU path → φ₂(A) = (e^A - I - A)·A⁻².
    # ---------------------------------------------------------------------
    @testset "solver_compute_phi2_function (series + LU paths)" begin
        _NE.reset_solver_phi2_monitor!()

        tiny = 1.0e-10 * Matrix{Float64}(I, 3, 3)
        @test _NE.solver_compute_phi2_function(tiny, exp(tiny)) ==
              _NE.phi2_series(tiny)

        A = [0.5 0.1 0.0
             0.2 0.4 0.1
             0.0 0.1 0.3]
        expA = exp(A)
        phi2 = _NE.solver_compute_phi2_function(A, expA; l = 2)
        # φ₂(A) = A⁻¹ (A⁻¹ (e^A - I - A)).
        ref = A \ (A \ (expA - I - A))
        @test maximum(abs.(phi2 .- ref)) < 1.0e-10
        @test maximum(abs.(phi2 .- _NE.phi2_series(A))) < 1.0e-10

        @test_throws ArgumentError _NE.solver_compute_phi2_function(
            fill(Inf, 2, 2), Matrix{Float64}(I, 2, 2); l = 1)
    end

    # ---------------------------------------------------------------------
    # φ₂ conditioning monitor: reset_solver_phi2_monitor! clears all counters;
    # report_solver_phi2_conditioning at a step past the interval resets the
    # monitor and records the report step.
    # ---------------------------------------------------------------------
    @testset "phi2 conditioning monitor reset + report" begin
        m = _NE.SOLVER_PHI2_MONITOR
        # Dirty the monitor, then reset.
        m.worst_rcond = 1.0e-4
        m.worst_l = 9
        m.series_expansion_count = 5
        m.lu_failure_count = 3
        m.nonfinite_count = 2
        m.last_report_step = 42
        _NE.reset_solver_phi2_monitor!()
        @test m.worst_rcond == 1.0
        @test m.worst_l == 0
        @test m.series_expansion_count == 0
        @test m.lu_failure_count == 0
        @test m.nonfinite_count == 0
        @test m.last_report_step == 0

        # Below the interval: no report, monitor untouched.
        m.series_expansion_count = 7
        m.last_report_step = 0
        _NE.report_solver_phi2_conditioning(50; interval = 100)
        @test m.series_expansion_count == 7      # not reset
        @test m.last_report_step == 0            # not advanced

        # Past the interval: emits the report, resets counters, advances the step.
        _NE.report_solver_phi2_conditioning(1000; interval = 100)
        @test m.series_expansion_count == 0      # reset happened
        @test m.last_report_step == 1000
    end

    # ---------------------------------------------------------------------
    # solver_enforce_ball_vector_regularity!: at the ball centre (radial index 1)
    # every spherical-harmonic mode with degree l ≥ 1 must be zeroed (regularity),
    # while the l = 0 mode and all radial levels r ≥ 2 are left untouched.
    # ---------------------------------------------------------------------
    @testset "solver_enforce_ball_vector_regularity! zeros l≥1 at the centre" begin
        lmax = 3
        nr = 4
        cfg = _NE.create_shtnskit_config(
            lmax = lmax, mmax = lmax, nlat = 8, nlon = 8, nr = nr, optimize_decomp = false)
        dom = _NE.create_radial_domain(nr)
        pencil = cfg.pencils.spec
        tor = _NE.create_shtns_spectral_field(Float64, cfg, dom, pencil)
        pol = _NE.create_shtns_spectral_field(Float64, cfg, dom, pencil)

        # Known distinct nonzero value at every (mode, radius).
        val_r(lm, r) = 1.0 + lm + r
        val_i(lm, r) = 2.0 + lm + r
        for sf in (tor, pol)
            sr = parent(sf.data_real)
            si = parent(sf.data_imag)
            for lm in _NE.local_spectral_mode_indices(cfg)
                slot = _NE.local_spectral_storage_slot(cfg, lm)
                slot === nothing && continue
                for r in 1:nr
                    _NE.set_local_spectral_value!(sr, slot, r, val_r(lm, r))
                    _NE.set_local_spectral_value!(si, slot, r, val_i(lm, r))
                end
            end
        end

        t_out, p_out = _NE.solver_enforce_ball_vector_regularity!(tor, pol)
        @test t_out === tor
        @test p_out === pol

        centre_l_ge1_max = 0.0
        l0_centre_ok = true
        deep_ok = true
        for sf in (tor, pol)
            sr = parent(sf.data_real)
            si = parent(sf.data_imag)
            for lm in _NE.local_spectral_mode_indices(cfg)
                slot = _NE.local_spectral_storage_slot(cfg, lm)
                slot === nothing && continue
                l = cfg.l_values[lm]
                c_r = _NE.local_spectral_value(sr, slot, 1)
                c_i = _NE.local_spectral_value(si, slot, 1)
                if l >= 1
                    centre_l_ge1_max = max(centre_l_ge1_max, abs(c_r), abs(c_i))
                else
                    l0_centre_ok &= (c_r == val_r(lm, 1)) && (c_i == val_i(lm, 1))
                end
                for r in 2:nr
                    deep_ok &= (_NE.local_spectral_value(sr, slot, r) == val_r(lm, r)) &&
                               (_NE.local_spectral_value(si, slot, r) == val_i(lm, r))
                end
            end
        end
        @test centre_l_ge1_max == 0.0     # all l ≥ 1 centre modes zeroed
        @test l0_centre_ok                 # l = 0 centre mode preserved
        @test deep_ok                      # interior radial levels preserved
    end

    # ---------------------------------------------------------------------
    # scalar_field_data_and_config: container dispatch.  Three accepted shapes
    # plus the unsupported-container error.  NamedTuples reproduce the exact
    # hasproperty/getproperty behaviour the function relies on.
    # ---------------------------------------------------------------------
    @testset "scalar_field_data_and_config dispatch" begin
        d = zeros(Float64, 2, 2, 3)
        cfg_marker = (sentinel = :cfg,)

        # Branch 1: field has .data and .config directly.
        dat1, c1 = _NE.scalar_field_data_and_config((data = d, config = cfg_marker))
        @test dat1 === d
        @test c1 === cfg_marker

        # Branch 2: field wraps a .temperature with .data/.config.
        dat2, c2 = _NE.scalar_field_data_and_config(
            (temperature = (data = d, config = cfg_marker),))
        @test dat2 === d
        @test c2 === cfg_marker

        # Branch 3: field wraps a .composition with .data/.config.
        dat3, c3 = _NE.scalar_field_data_and_config(
            (composition = (data = d, config = cfg_marker),))
        @test dat3 === d
        @test c3 === cfg_marker

        # Unsupported container ⇒ error.
        @test_throws ErrorException _NE.scalar_field_data_and_config((unrelated = 1,))
    end
end
