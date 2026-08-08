# ================================================================================
# Regression tests for batch B of the max-effort src/ review — the PLAUSIBLE
# correctness findings [23]-[27] (structurally verified, trigger env/config gated).
# ================================================================================
#
#   B23 solver/numerics.jl:391        the Krylov loop answers every failure with
#                                    `kmax = j; break` and then returns the truncated,
#                                    finite, badly under-converged action
#   B24 fields/scalar_operators.jl:930  _TAU_CACHE / _INFLUENCE_CACHE are unlocked
#                                    global IdDicts (the sibling _MODE_INDEX_CACHE
#                                    is lock-guarded)
#   B25 core/parameters.jl:314       `while current_dir != "/"` is a POSIX-only root
#                                    sentinel; a Windows/UNC root is a dirname fixed
#                                    point, so the walk never terminates
#   B26 gpu/erk2_state.jl:192        vel_tor_spec never gets with_boundary_mode_values,
#                                    unlike the CPU sibling and unlike temperature
#   B27 timestep/erk2/cache.jl:813   the ERK2 stage-cache validity test omits every
#                                    BOUNDARY input, though the eliminated constraint
#                                    rows are baked into the propagators
# ================================================================================

using Test
using MPI
using Random
using LinearAlgebra
using GeoDynamo

_crb_wsn(s) = replace(s, r"\s+" => "")
_crb_occ(pat::AbstractString, src) = occursin(_crb_wsn(pat), _crb_wsn(src))
const CRB_PARAMS_SRC = read(
    joinpath(normpath(joinpath(@__DIR__, "..")), "src", "core", "parameters.jl"), String)

@testset "Max-effort review batch B" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping batch B fix tests"
        return
    end
    MPI.Initialized() || MPI.Init()

    # ── B23: an under-converged Krylov action must not pass as an answer ──────
    @testset "B23 krylov_exp_action reports non-convergence" begin
        nr = 24
        # A well-conditioned negative-definite operator converges easily.
        A = zeros(Float64, nr, nr)
        for i in 1:nr
            A[i, i] = -1.0 - 0.01 * i
        end
        Aop!(y, x) = (mul!(y, A, x); y)
        v = ones(Float64, nr)

        ok = GeoDynamo.krylov_exp_action(Aop!, v, 1e-3; m = 20, tol = 1e-10)
        @test all(isfinite, ok)
        # reference: the true action of a diagonal generator
        ref = [exp(1e-3 * A[i, i]) * v[i] for i in 1:nr]
        @test isapprox(ok, ref; rtol = 1e-6)

        # A basis far too small for the requested tolerance must FAIL rather than
        # return a finite, quietly wrong 2-term approximation. (The operator is
        # deliberately MILD: a stiff one underflows exp(dt*H11) to zero, which makes
        # both sides of the residual test zero and legitimately reports convergence
        # at j = 1 — so a stiff probe would prove nothing here.)
        @test_throws ErrorException GeoDynamo.krylov_exp_action(
            Aop!, v, 1.0; m = 2, tol = 1e-14)
    end

    # ── B24: the tau/influence caches must be lock-guarded like their sibling ─
    @testset "B24 scalar operator caches are lock-guarded" begin
        @test GeoDynamo._TAU_CACHE_LOCK isa ReentrantLock
        @test GeoDynamo._INFLUENCE_CACHE_LOCK isa ReentrantLock

        # Concurrent first-touch from several threads must not corrupt the cache.
        GeoDynamo.clear_scalar_field_caches!()
        domains = [GeoDynamo.create_radial_domain(8 + i) for i in 1:4]
        results = Vector{Any}(undef, length(domains))
        Threads.@threads for i in eachindex(domains)
            results[i] = GeoDynamo._get_tau_cache(domains[i])
        end
        @test all(r -> r isa GeoDynamo._TauCache, results)
        for (i, d) in enumerate(domains)
            @test results[i].nr == d.N
            # a second call returns the same cached object
            @test GeoDynamo._get_tau_cache(d) === results[i]
        end
    end

    # ── B25: the upward package-root walk must terminate on any filesystem ────
    @testset "B25 package-root walk stops at a dirname fixed point" begin
        # The Windows/UNC hang cannot be reproduced on POSIX (dirname("C:\\") is a
        # fixed point only on Windows), so the portable termination PREDICATE is
        # tested directly and the POSIX-only sentinel is pinned out of the source.
        @test GeoDynamo._parent_dir("/tmp") == "/"
        @test GeoDynamo._parent_dir("/") === nothing        # fixed point -> stop
        @test GeoDynamo._parent_dir(GeoDynamo._parent_dir("/tmp")) === nothing

        # walking from a directory with no GeoDynamo Project.toml above it must
        # terminate and fall back, not spin
        @test GeoDynamo.find_package_root() isa String
        @test !_crb_occ("while current_dir != \"/\"", CRB_PARAMS_SRC)
    end

    # ── B26: the GPU ERK2 pack must carry the toroidal per-mode endpoints ─────
    @testset "B26 build_gpu_erk2_state attaches vel_tor mode values" begin
        params = GeoDynamo.SolverParameters(
            geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8,
            nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
            Ek = 1e-3, Ra = 1e4, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
            include_magnetic = false, include_composition = false,
            timestepper = GeoDynamo.ExponentialRungeKutta2())
        st = GeoDynamo.initialize_solver_state(Float64; params = params)
        GeoDynamo.initialize_fields!(st)

        # a static, non-zero per-mode toroidal endpoint (the case _gpu_assert_static_bcs
        # explicitly accepts as "baked")
        tor = st.fields.velocity.toroidal
        tor.boundary_values[1, 2] = 0.75

        erk = GeoDynamo.build_gpu_erk2_state(st)
        # index 2 of the canonical m-major order is (l=1, m=0) -> val_r[l+1, m+1]
        cfg = st.backend.shtns_config
        @test cfg.l_values[2] == 1 && cfg.m_values[2] == 0
        @test erk.velocity_tor.bc.inner.val_r[2, 1] == 0.75
        # and the modes that carry no prescribed endpoint stay at the spec target
        @test erk.velocity_tor.bc.inner.val_r[3, 1] == 0.0
        # temperature already worked; assert the two paths now agree in shape
        @test size(erk.velocity_tor.bc.inner.val_r) == size(erk.temperature.bc.inner.val_r)
    end

    # ── B27: the ERK2 stage cache must notice a changed boundary structure ────
    @testset "B27 ERK2 stage cache tracks its boundary structure" begin
        cfg = GeoDynamo.create_shtnskit_config(
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 8)
        dom = GeoDynamo.create_radial_domain(8)
        dt = 1e-4
        nu = 1.0

        # code 3 = ND, code 4 = NN -> a DIFFERENT eliminated inner/outer row pair
        c3 = GeoDynamo._get_or_build_erk2_scalar_cache(
            nothing, "probe", nu, Float64, cfg, dom, dt, 3)
        @test c3.bc_signature == GeoDynamo._get_or_build_erk2_scalar_cache(
            c3, "probe", nu, Float64, cfg, dom, dt, 3).bc_signature
        # same dt / nr / diffusivity, different BC code -> must NOT be reused
        c4 = GeoDynamo._get_or_build_erk2_scalar_cache(
            c3, "probe", nu, Float64, cfg, dom, dt, 4)
        @test c4 !== c3
        @test c4.bc_signature != c3.bc_signature
        @test c4.E_full[1] != c3.E_full[1]      # the propagators really did change

        # inner_regularity swaps the inner row for the center-regularity row
        cr = GeoDynamo._get_or_build_erk2_scalar_cache(
            c3, "probe", nu, Float64, cfg, dom, dt, 3; inner_regularity = true)
        @test cr !== c3
        @test cr.bc_signature != c3.bc_signature

        # the generic (velocity-like) getter tracks dpol_operator the same way
        g1 = GeoDynamo._get_or_build_erk2_cache(
            nothing, "probe", nu, Float64, cfg, dom, dt)
        g2 = GeoDynamo._get_or_build_erk2_cache(
            g1, "probe", nu, Float64, cfg, dom, dt; dpol_operator = true)
        @test g2 !== g1
        @test g2.bc_signature != g1.bc_signature

        # an unchanged call is still reused (the fix must not defeat memoization)
        @test GeoDynamo._get_or_build_erk2_scalar_cache(
            c4, "probe", nu, Float64, cfg, dom, dt, 4) === c4
    end
end
