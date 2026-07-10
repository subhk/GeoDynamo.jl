using Test

_alloc_topography_cache_index(cache) = @allocated GeoDynamo.bcs.topography._cache_index(cache, 1, 0)
_alloc_topography_cache_value(cache) = @allocated GeoDynamo.bcs.topography.get_cache_value(
    cache, 1, 0, GeoDynamo.OUTER_BOUNDARY)

function _weakly_cache_boundary_base()
    topo = GeoDynamo.bcs.topography
    boundary = reshape([1.0, 2.0], 1, 2)
    topo.reset_boundary_to_base!(boundary)
    return WeakRef(boundary)
end

@testset "Stability Regressions" begin
    @testset "gpu scratch allocation hook is inferred" begin
        ci = only(code_typed(GeoDynamo.gpu_scratch_zeros, (Type{Float64}, Int, Int); optimize = true))
        @test ci.second !== Any
    end

    @testset "topography spectral compatibility is not Any" begin
        @test GeoDynamo.bcs.topography.SHTnsSpecField !== Any
    end

    @testset "topography derivative cache lookups are concrete and allocation-free" begin
        topo = GeoDynamo.bcs.topography
        cfg = GeoDynamo.create_shtnskit_config(
            lmax = 2, mmax = 2, nlat = 8, nlon = 16, nr = 4)
        z = zeros(ComplexF64, cfg.nlm)
        cache = topo.BoundaryDerivativeCache{Float64}(
            cfg.lmax, cfg.mmax, cfg.nlm, cfg,
            z, z, z, z, nothing, nothing)

        topo.get_cache_value(cache, 1, 0, GeoDynamo.OUTER_BOUNDARY)
        @test isconcretetype(fieldtype(typeof(cache), :config))
        @test _alloc_topography_cache_index(cache) == 0
        @test _alloc_topography_cache_value(cache) == 0
    end

    @testset "topography boundary bases do not retain simulations" begin
        topo = GeoDynamo.bcs.topography
        @test isdefined(topo, :clear_boundary_value_base_cache!)
        if isdefined(topo, :clear_boundary_value_base_cache!)
            topo.clear_boundary_value_base_cache!()
            boundary = reshape([1.0, 2.0], 1, 2)
            topo.reset_boundary_to_base!(boundary)
            fill!(boundary, 0.0)
            topo.reset_boundary_to_base!(boundary)
            @test boundary == reshape([1.0, 2.0], 1, 2)

            topo.clear_boundary_value_base_cache!()
            weak_boundary = _weakly_cache_boundary_base()
            GC.gc()
            GC.gc()
            @test weak_boundary.value === nothing
            @test isempty(topo._BOUNDARY_VALUE_BASE)

            topo.clear_boundary_value_base_cache!()
            @test isempty(topo._BOUNDARY_VALUE_BASE)
        end
    end

    @testset "single-thread solver avoids task-spawning implicit path" begin
        expected = Threads.nthreads() > 1
        @test GeoDynamo._solver_can_thread_implicit_updates(GeoDynamo.CNAB2()) == expected
    end

    @testset "scalar cache cleanup helpers empty global caches" begin
        cfg = GeoDynamo.create_shtnskit_config(
            lmax = 2, mmax = 2, nlat = 8, nlon = 16, nr = 4)
        GeoDynamo.get_mode_index(cfg, 0, 0)
        @test !isempty(GeoDynamo._MODE_INDEX_CACHE)

        domain = GeoDynamo.create_shell_radial_domain(4)
        GeoDynamo._TAU_CACHE[domain] = GeoDynamo._TauCache(
            4, zeros(4), zeros(4), 0.0, 0.0, 0.0, 0.0
        )
        GeoDynamo._INFLUENCE_CACHE[domain] = GeoDynamo._InfluenceCache(
            4, zeros(4), zeros(4), zeros(2, 2)
        )

        GeoDynamo.clear_scalar_field_caches!()

        @test isempty(GeoDynamo._MODE_INDEX_CACHE)
        @test isempty(GeoDynamo._TAU_CACHE)
        @test isempty(GeoDynamo._INFLUENCE_CACHE)
    end

    @testset "solver mode-index cache deduplicates equivalent configs" begin
        empty!(GeoDynamo.SOLVER_MODE_INDEX_CACHE)

        function populate_solver_mode_cache()
            for _ in 1:3
                cfg = GeoDynamo.create_shtnskit_config(
                    lmax = 2, mmax = 2, nlat = 8, nlon = 16, nr = 4)
                GeoDynamo.mode_index(cfg, 0, 0)
            end
            return nothing
        end

        populate_solver_mode_cache()
        GC.gc()
        GC.gc()

        @test length(GeoDynamo.SOLVER_MODE_INDEX_CACHE) <= 1
    end

    @testset "SHTnsBuffers replaces _buffer_cache" begin
        @test !hasfield(GeoDynamo.SHTnsKitConfig, :_buffer_cache)
        @test hasfield(GeoDynamo.SHTnsKitConfig, :_buffers)
        @test GeoDynamo.SHTnsBuffers isa DataType
        b = GeoDynamo.SHTnsBuffers()
        @test b.sht_plan === nothing
        @test b.synth_out === nothing
        @test b.solver_transform_workspace === nothing
    end

    @testset "ERK2 influence cache invalidates when non-dt parameters change" begin
        params = GeoDynamo.SolverParameters(
            architecture = :cpu,
            geometry = :shell,
            nr = 8,
            nr_inner = 4,
            lmax = 4,
            mmax = 4,
            nlat = 12,
            nlon = 16,
            timestep = 1e-4,
            stop_iteration = 1,
            timestepper = GeoDynamo.ExponentialRungeKutta2(),
            include_magnetic = false,
            include_composition = false,
            topography_enabled = false,
            stefan_enabled = false
        )
        state = GeoDynamo.initialize_simulation(Float64, params)
        tc = state.timestep_caches
        cfg = state.runtime.shtns_config
        domain = state.runtime.outer_core_domain

        base = GeoDynamo.get_solver_erk2_influence_matrices!(
            tc,
            :velocity_poloidal,
            Float64,
            cfg,
            domain,
            params.Ek,
            params.timestep,
            1;
            theta = 0.5
        )
        changed_bc = GeoDynamo.get_solver_erk2_influence_matrices!(
            tc,
            :velocity_poloidal,
            Float64,
            cfg,
            domain,
            params.Ek,
            params.timestep,
            4;
            theta = 0.5
        )
        changed_theta = GeoDynamo.get_solver_erk2_influence_matrices!(
            tc,
            :velocity_poloidal,
            Float64,
            cfg,
            domain,
            params.Ek,
            params.timestep,
            4;
            theta = 0.5 + 0.1
        )

        @test changed_bc !== base
        @test changed_theta !== changed_bc
    end

    @testset "unsupported EAB2 state is rejected before cache preparation" begin
        params = GeoDynamo.SolverParameters(
            architecture = :cpu,
            geometry = :shell,
            nr = 8,
            nr_inner = 4,
            lmax = 4,
            mmax = 4,
            nlat = 12,
            nlon = 16,
            timestep = 1e-4,
            stop_iteration = 1,
            timestepper = GeoDynamo.ExponentialAdamsBashforth2(),
            include_magnetic = true,
            include_composition = true,
            topography_enabled = false,
            stefan_enabled = false
        )
        @test_throws ArgumentError GeoDynamo.initialize_simulation(Float64, params)
    end

    @testset "sync_spectral_history! copies real and imaginary coefficients" begin
        cfg = GeoDynamo.create_shtnskit_config(
            lmax = 2, mmax = 2, nlat = 8, nlon = 16, nr = 4)
        domain = GeoDynamo.create_shell_radial_domain(4)
        current = GeoDynamo.create_shtns_spectral_field(Float64, cfg, domain, cfg.pencils.spec)
        previous = GeoDynamo.create_shtns_spectral_field(Float64, cfg, domain, cfg.pencils.spec)

        fill!(parent(current.data_real), 3.0)
        fill!(parent(current.data_imag), -2.0)
        fill!(parent(previous.data_real), 0.0)
        fill!(parent(previous.data_imag), 0.0)

        GeoDynamo.sync_spectral_history!(previous, current)

        @test parent(previous.data_real) == parent(current.data_real)
        @test parent(previous.data_imag) == parent(current.data_imag)
    end

    @testset "Reynolds stress uses spherical quadrature weights" begin
        cfg = GeoDynamo.create_shtnskit_config(
            lmax = 3, mmax = 3, nlat = 8, nlon = 12, nr = 6)
        domain = GeoDynamo.create_shell_radial_domain(6)
        velocity = GeoDynamo.create_shtns_velocity_fields(Float64, cfg, domain;
            params = GeoDynamo.SolverParameters(geometry = :shell))

        v_r = parent(velocity.velocity.r_component.data)
        v_θ = parent(velocity.velocity.θ_component.data)
        v_φ = parent(velocity.velocity.φ_component.data)

        r_range = GeoDynamo.range_local(cfg.pencils.r, 3)
        θ_range = GeoDynamo.range_local(cfg.pencils.r, 1)
        local_size = size(v_r)

        fill!(v_r, 0.0)
        fill!(v_θ, 0.0)
        fill!(v_φ, 0.0)

        local_weight_sum = 0.0
        local_rr_sum = 0.0
        local_rθ_sum = 0.0
        for k in 1:local_size[3]
            r_idx = first(r_range) + k - 1
            if r_idx > domain.N
                continue
            end
            radial_weight = domain.r[r_idx, 4]^2 * domain.integration_weights[r_idx]
            for j in 1:local_size[2]
                for i in 1:local_size[1]
                    theta_idx = θ_range[i]
                    weight = radial_weight * cfg.gauss_weights[theta_idx] * (2π / cfg.nlon)
                    linear_idx = i + (j - 1) * local_size[1] +
                                 (k - 1) * local_size[1] * local_size[2]
                    u_r = domain.r[r_idx, 4]
                    u_θ = cos(cfg.theta_grid[theta_idx])
                    u_φ = 0.5
                    v_r[linear_idx] = u_r
                    v_θ[linear_idx] = u_θ
                    v_φ[linear_idx] = u_φ
                    local_weight_sum += weight
                    local_rr_sum += weight * u_r^2
                    local_rθ_sum += weight * u_r * u_θ
                end
            end
        end

        total_weight = GeoDynamo.MPI.Allreduce(local_weight_sum, GeoDynamo.MPI.SUM, GeoDynamo.get_comm())
        expected_rr = GeoDynamo.MPI.Allreduce(local_rr_sum, GeoDynamo.MPI.SUM, GeoDynamo.get_comm()) /
                      total_weight
        expected_rθ = GeoDynamo.MPI.Allreduce(local_rθ_sum, GeoDynamo.MPI.SUM, GeoDynamo.get_comm()) /
                      total_weight

        R_rr, _, _, R_rθ, _, _ = GeoDynamo.compute_reynolds_stress(velocity)
        @test R_rr ≈ expected_rr
        @test R_rθ ≈ expected_rθ
    end

    @testset "invalid scalar batch helper is removed from velocity module" begin
        @test !isdefined(GeoDynamo, :batch_velocity_transforms!)
    end

    @testset "TimestepCaches replaces dict caches" begin
        @test !hasfield(GeoDynamo.SolverState, :etd_caches)
        @test !hasfield(GeoDynamo.SolverState, :erk2_caches)
        @test !hasfield(GeoDynamo.SolverState, :erk2_influence_matrices)
        @test hasfield(GeoDynamo.SolverState, :timestep_caches)
        tc = GeoDynamo.TimestepCaches{Float64}()
        @test tc.etd_velocity_toroidal === nothing
        @test tc.erk2_velocity_toroidal === nothing
        @test tc.erk2_influence_velocity_poloidal === nothing
    end

    @testset "SolverTransformBuffers replaces TransformWorkspace.cache" begin
        @test !hasfield(GeoDynamo.TransformWorkspace, :cache)
        @test hasfield(GeoDynamo.TransformWorkspace, :buffers)
        @test GeoDynamo.SolverTransformBuffers isa UnionAll
        tb = GeoDynamo.SolverTransformBuffers{Float64}()
        @test tb.vector_coeffs_1 === nothing
        @test tb.generic_slice === nothing
    end
end
