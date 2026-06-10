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

# Measure allocations through a function barrier so the arguments are concretely
# typed inside the measurement. Calling `@allocated f(x)` directly on non-const
# test locals would count call-site dynamic dispatch/boxing and mask the real
# per-call allocation of `f`.
_alloc_theta_grad(f, ws) = @allocated GeoDynamo.compute_theta_gradient_spectral!(f, ws)
_alloc_phi_grad(f, ws) = @allocated GeoDynamo.compute_phi_gradient_spectral!(f, ws)
_alloc_mode_indices(cfg) = @allocated GeoDynamo.local_spectral_mode_indices(cfg)
# Mirrors the per-mode BC-enforcement pattern in prepare/finalize_solver_erk2_field!:
# read the mode-value slots off the spec, enforce both endpoints with the overrides.
function _alloc_enforce_bc(stage, spec, l, nr, lm_idx)
    return @allocated begin
        iv = GeoDynamo.boundary_mode_value(spec.inner_mode_values, lm_idx)
        ov = GeoDynamo.boundary_mode_value(spec.outer_mode_values, lm_idx)
        GeoDynamo.solver_enforce_erk2_bc!(stage, spec.inner, 1, l, nr; value_override = iv)
        GeoDynamo.solver_enforce_erk2_bc!(stage, spec.outer, nr, l, nr; value_override = ov)
    end
end

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
        stop_iteration = 4,
        include_magnetic = true,
        include_composition = true,
        timestepper = GeoDynamo.CNAB2(),
        topography_enabled = false,
        stefan_enabled = false
    )
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_fields!(state)
    # Warm up: compile the hot paths and populate the per-config transform buffers.
    GeoDynamo.solver_step!(state)

    temp_field = state.fields.temperature
    grad_ws = state.runtime.gradient_workspace
    cfg = state.backend.shtns_config
    domain = state.runtime.outer_core_domain

    # --- A. Cached lookups are allocation-free; hot paths reuse, not rebuild ---
    @testset "hot paths do not allocate" begin
        # The owned spectral-mode list is cached per config (not rebuilt per call).
        GeoDynamo.local_spectral_mode_indices(cfg)
        @test _alloc_mode_indices(cfg) == 0

        # The θ/φ-gradient kernels write in place into the (concretely-typed)
        # workspace spectral fields, so they are allocation-free once warmed.
        # (A non-concrete workspace field type silently reintroduces ~8 KB/call
        # of per-element boxing — this guards against that regression.)
        GeoDynamo.compute_theta_gradient_spectral!(temp_field, grad_ws)
        GeoDynamo.compute_phi_gradient_spectral!(temp_field, grad_ws)
        @test _alloc_theta_grad(temp_field, grad_ws) == 0
        @test _alloc_phi_grad(temp_field, grad_ws) == 0
    end

    # --- A2. θ-gradient batches the cross-rank spectral gather ------------------
    @testset "θ-gradient issues one gather reduction, not one per radial level (#5)" begin
        # The (l,m)<->(l±1,m) recurrence needs the full spectrum gathered across
        # ranks. Gathering per radial level issues O(nr) collectives; batching all
        # radial levels into a single buffer issues O(1). `_THETA_GATHER_REDUCE_COUNT`
        # tracks gather-reduce passes (== MPI collective count under multi-rank).
        # nr=16 here, so the former per-level path registered 2*16 = 32.
        GeoDynamo._THETA_GATHER_REDUCE_COUNT[] = 0
        GeoDynamo.compute_theta_gradient_spectral!(temp_field, grad_ws)
        @test GeoDynamo._THETA_GATHER_REDUCE_COUNT[] == 2
    end

    # --- A3. Scalar transforms minimise the cross-rank collective count --------
    @testset "scalar transforms issue one batched θ_comm spectral collective (#6)" begin
        # Phase-3 DistTransposePlan path: each scalar transform performs exactly ONE
        # θ_comm m-axis redistribution (synthesis: Allgatherv in spec_storage_to_solve!;
        # analysis: Allreduce in solve_to_spec_storage!), batched over ALL radial
        # levels (independent of nr).  The heavy Legendre/FFT work is θ-distributed
        # inside dist_synthesis!/dist_analysis! and is NOT counted here.  No full-grid
        # gather and no per-radial-level collective occur.
        GeoDynamo._SCALAR_GATHER_REDUCE_COUNT[] = 0
        GeoDynamo.scalar_spectral_to_physical!(temp_field.spectral, temp_field.temperature)
        @test GeoDynamo._SCALAR_GATHER_REDUCE_COUNT[] == 1

        GeoDynamo._SCALAR_GATHER_REDUCE_COUNT[] = 0
        GeoDynamo.scalar_physical_to_spectral!(temp_field.temperature, temp_field.spectral)
        @test GeoDynamo._SCALAR_GATHER_REDUCE_COUNT[] == 1
    end

    # --- B. Caches reuse rather than rebuild -----------------------------------
    @testset "ERK2 boundary-spec cache reuses, matches fresh build (#2)" begin
        tc = GeoDynamo.TimestepCaches{Float64}()
        rebuilt = Ref(false)
        s1 = GeoDynamo._get_or_build_erk2_boundary_spec!(
            tc, :temperature, 1,
            () -> GeoDynamo.build_solver_erk2_scalar_bc(Float64, domain, 1)
        )
        s2 = GeoDynamo._get_or_build_erk2_boundary_spec!(
            tc, :temperature, 1,
            () -> (rebuilt[] = true;
                GeoDynamo.build_solver_erk2_scalar_bc(Float64, domain, 1))
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
            () -> GeoDynamo.build_solver_erk2_scalar_bc(Float64, domain, 4)
        )
        @test s3 !== s1
        @test length(tc.erk2_boundary_specs) == 2
    end

    @testset "influence-correction scratch is reused (#3)" begin
        tc = GeoDynamo.TimestepCaches{Float64}()
        w1 = GeoDynamo.get_radial_work!(tc, :velocity_poloidal_influence, domain.N)
        w2 = GeoDynamo.get_radial_work!(tc, :velocity_poloidal_influence, domain.N)
        @test w1 === w2
        @test w1.tmp_real === w2.tmp_real      # same backing vector, not reallocated
        @test length(w1.tmp_real) == domain.N
    end

    @testset "transform buffer cache warm path returns cached object (#4)" begin
        @test cfg._buffers.solver_transform_workspace isa GeoDynamo.TransformWorkspace
        builds = Ref(0)
        b1 = GeoDynamo.solver_get_cached_buffer!(cfg, :coeffs_buffer) do
            builds[] += 1
            zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        end
        b2 = GeoDynamo.solver_get_cached_buffer!(cfg, :coeffs_buffer) do
            builds[] += 1
            zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
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
        # Workspace spectral fields must be concretely typed — a bare
        # `SpectralFieldType{T}` (params unbound) is non-concrete and boxes in
        # the gradient hot loop.
        @test isconcretetype(fieldtype(gwT, :∇θ_spec))
        @test isconcretetype(fieldtype(gwT, :∇φ_spec))
        @test isconcretetype(fieldtype(gwT, :∇r_spec))
        @test fieldtype(gwT, :theta_lm_plus) === Vector{Int}
        @test fieldtype(gwT, :theta_lm_minus) === Vector{Int}
        @test fieldtype(gwT, :theta_full_real) === Matrix{Float64}
    end

    @testset "structural invariants" begin
        @test hasfield(typeof(cfg), :_buffers)
        @test !hasfield(typeof(cfg), :_buffer_cache)
        @test hasfield(typeof(state), :timestep_caches)
        @test fieldtype(typeof(state.timestep_caches), :erk2_boundary_specs) <:
              Dict{Tuple{Symbol, Int}, <:GeoDynamo.SolverERK2BoundarySpec}
    end

    @testset "hot calls are type-inferable" begin
        # NB: `mode_index` itself infers to Any (its cache-table local is a
        # Union) — but it is no longer on the gradient hot path (#1 precomputes
        # the (l±1,m) neighbours), so the gradient call below is what matters.
        @test (@inferred GeoDynamo.compute_theta_gradient_spectral!(temp_field, grad_ws)) ===
              grad_ws
    end

    @testset "ERK2 boundary-spec fields are concrete; BC enforcement is alloc-free" begin
        # SolverERK2BoundarySpec used to type its mode-value slots as
        # Union{Nothing, AbstractVector{T}} — an abstract field. Every per-mode
        # read in prepare/finalize_solver_erk2_field! then boxed the
        # value_override (~51 B × thousands of calls/step, ~590 KB/step on ERK2).
        # The struct is parameterized on the concrete slot types instead; these
        # guards pin that.
        T = Float64
        nr_bc = 16
        inner_side = GeoDynamo.solver_create_dirichlet_bc(T, nr_bc)
        outer_side = GeoDynamo.solver_create_dirichlet_bc(T, nr_bc)
        base_spec = GeoDynamo.SolverERK2BoundarySpec{T}(inner_side, outer_side)
        bvals = rand(T, 2, 8)
        spec = GeoDynamo.with_boundary_mode_values(
            base_spec, view(bvals, 1, :), view(bvals, 2, :))

        # Field types concrete per instance (views attached ⇒ SubArray slots;
        # imag left off ⇒ Nothing slots).
        @test isconcretetype(fieldtype(typeof(spec), :inner_mode_values))
        @test isconcretetype(fieldtype(typeof(spec), :outer_mode_values))
        @test fieldtype(typeof(spec), :inner_mode_values_imag) === Nothing
        @test fieldtype(typeof(spec), :outer_mode_values_imag) === Nothing
        # Base (all-nothing) spec is fully concrete too.
        @test isconcretetype(fieldtype(typeof(base_spec), :inner_mode_values))

        # The per-mode enforcement pattern from prepare/finalize must not
        # allocate once the spec is concrete (warm call first).
        stage = zeros(T, nr_bc)
        _alloc_enforce_bc(stage, spec, 1, nr_bc, 1)
        @test _alloc_enforce_bc(stage, spec, 1, nr_bc, 1) == 0
    end

    if MPI.Initialized() && FINALIZE_MPI_ALLOC && !MPI.Finalized()
        MPI.Finalize()
    end
end
