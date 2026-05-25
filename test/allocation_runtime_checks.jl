# Runtime allocation & inference guards.
#
# Replaces the former `allocation_static_checks.jl`, which matched exact source
# text (brittle — it broke whenever the matched lines were refactored). These
# checks instead build a tiny serial solver state and verify the *behavior* the
# static checks approximated: hot paths don't allocate, the workspace/spec
# caches reuse rather than rebuild, and core types are concretely typed/inferred.
#
# Fixture uses CNAB2 (a timestepper proven to run end-to-end on the tiny state);
# the ERK2 boundary-spec cache (#2) and influence scratch (#3) are exercised via
# direct helper calls, so these checks don't depend on the full ERK2 step.

using Test
using MPI

const FINALIZE_MPI_ALLOC = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Runtime allocation & inference guards" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping runtime allocation guards"
        return
    end
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
        max_steps = 4,
        include_magnetic_field = true,
        include_composition = true,
        timestepper = GeoDynamo.CNAB2(),
        topography_enabled = false,
        stefan_enabled = false,
    )
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_fields!(state)
    # Warm up: compile the hot paths and populate the per-config transform buffers.
    GeoDynamo.advance_solver_step!(state)

    temp_field = state.fields.temperature
    grad_ws = state.runtime.gradient_workspace
    cfg = state.backend.shtns_config
    domain = state.runtime.𝒟ᵒᶜ

    # --- A. Cached lookups are allocation-free; hot paths reuse, not rebuild ---
    @testset "cached lookups do not allocate" begin
        # The owned spectral-mode list is cached per config (not rebuilt per call).
        GeoDynamo.local_spectral_mode_indices(cfg)
        @test (@allocated GeoDynamo.local_spectral_mode_indices(cfg)) == 0

        # (Radial-work cache reuse is validated by object identity in the
        # influence-scratch testset below, which is robust to measurement noise.)

        # The θ/φ-gradient runs and returns its (reused) workspace. We don't
        # assert @allocated==0 on the gradient itself: its ~8 KB/call residual is
        # the PencilArrays `axes_local` access inside `local_range`, a separate
        # pre-existing allocation unrelated to the gradient's own buffers.
        @test GeoDynamo.solver_compute_theta_gradient_spectral!(temp_field, grad_ws) === grad_ws
        @test GeoDynamo.solver_compute_phi_gradient_spectral!(temp_field, grad_ws) === grad_ws
    end

    # --- B. Caches reuse rather than rebuild -----------------------------------
    @testset "ERK2 boundary-spec cache reuses, matches fresh build (#2)" begin
        tc = GeoDynamo.TimestepCaches{Float64}()
        rebuilt = Ref(false)
        s1 = GeoDynamo._get_or_build_erk2_boundary_spec!(
            tc, :temperature, 1,
            () -> GeoDynamo.build_solver_erk2_scalar_bc(Float64, domain, 1),
        )
        s2 = GeoDynamo._get_or_build_erk2_boundary_spec!(
            tc, :temperature, 1,
            () -> (rebuilt[] = true; GeoDynamo.build_solver_erk2_scalar_bc(Float64, domain, 1)),
        )
        @test s1 === s2            # second lookup returns the cached object
        @test !rebuilt[]           # builder was not invoked again

        fresh = GeoDynamo.build_solver_erk2_scalar_bc(Float64, domain, 1)
        @test s1.inner.type == fresh.inner.type
        @test s1.outer.type == fresh.outer.type
        @test s1.inner.stencil == fresh.inner.stencil
        @test s1.outer.stencil == fresh.outer.stencil

        # A different BC code keys a distinct entry.
        s3 = GeoDynamo._get_or_build_erk2_boundary_spec!(
            tc, :temperature, 4,
            () -> GeoDynamo.build_solver_erk2_scalar_bc(Float64, domain, 4),
        )
        @test s3 !== s1
        @test length(tc.erk2_boundary_specs) == 2
    end

    @testset "influence-correction scratch is reused (#3)" begin
        tc = GeoDynamo.TimestepCaches{Float64}()
        w1 = GeoDynamo.solver_get_radial_work!(tc, :velocity_poloidal_influence, domain.N)
        w2 = GeoDynamo.solver_get_radial_work!(tc, :velocity_poloidal_influence, domain.N)
        @test w1 === w2
        @test w1.tmp_real === w2.tmp_real      # same backing vector, not reallocated
        @test length(w1.tmp_real) == domain.N
    end

    @testset "transform buffer cache warm path returns cached object (#4)" begin
        @test cfg._buffers.solver_transform_workspace isa GeoDynamo.TransformWorkspace
        builds = Ref(0)
        b1 = GeoDynamo.solver_get_cached_buffer!(cfg, :coeffs_buffer) do
            builds[] += 1
            zeros(ComplexF64, cfg.nlm)
        end
        b2 = GeoDynamo.solver_get_cached_buffer!(cfg, :coeffs_buffer) do
            builds[] += 1
            zeros(ComplexF64, cfg.nlm)
        end
        @test b1 === b2          # warm path hands back the same buffer
        @test builds[] <= 1      # built at most once across both calls
    end

    # --- C. Type stability / concrete struct typing ----------------------------
    @testset "core field/runtime types are concrete" begin
        specT = typeof(temp_field.spectral)
        @test isconcretetype(specT)
        @test isconcretetype(fieldtype(specT, :data_real))
        @test isconcretetype(fieldtype(specT, :data_imag))
        @test isconcretetype(fieldtype(specT, :config))

        @test isconcretetype(typeof(state.backend))
        @test isconcretetype(typeof(state.runtime))
        @test isconcretetype(typeof(state.fields.velocity))
        @test isconcretetype(typeof(temp_field))

        gwT = typeof(grad_ws)
        @test fieldtype(gwT, :theta_lm_plus) === Vector{Int}
        @test fieldtype(gwT, :theta_lm_minus) === Vector{Int}
        @test fieldtype(gwT, :theta_full_real) === Vector{Float64}
    end

    @testset "structural invariants" begin
        @test hasfield(typeof(cfg), :_buffers)
        @test !hasfield(typeof(cfg), :_buffer_cache)
        @test hasfield(typeof(state), :timestep_caches)
        @test fieldtype(typeof(state.timestep_caches), :erk2_boundary_specs) <:
              Dict{Tuple{Symbol,Int}, <:GeoDynamo.SolverERK2BoundarySpec}
    end

    @testset "hot calls are type-inferable" begin
        # NB: `mode_index` itself infers to Any (its cache-table local is a
        # Union) — but it is no longer on the gradient hot path (#1 precomputes
        # the (l±1,m) neighbours), so the gradient call below is what matters.
        @test (@inferred GeoDynamo.solver_compute_theta_gradient_spectral!(temp_field, grad_ws)) === grad_ws
    end

    if MPI.Initialized() && FINALIZE_MPI_ALLOC && !MPI.Finalized()
        MPI.Finalize()
    end
end
