function initialize_composition_field!(state::SolverState{T,<:AbstractArchitecture}) where T
    composition = state.fields.composition
    composition === nothing && return state

    spec_real = parent(composition.spectral.data_real)
    spec_imag = parent(composition.spectral.data_imag)
    fill!(spec_real, zero(T))
    fill!(spec_imag, zero(T))

    lm_range = local_spectral_mode_indices(composition.config)
    r_range = local_range(composition.config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        lm_idx <= composition.config.nlm || continue
        l = composition.config.l_values[lm_idx]
        m = composition.config.m_values[lm_idx]
        slot = local_spectral_storage_slot(composition.config, lm_idx)

        for r_idx in r_range
            if l == 0 && m == 0
                # Orthonormal SH (Y_0^0 = 1/√(4π)): the uniform physical mean
                # composition 0.5 is stored as the (0,0) coefficient 0.5·√(4π),
                # matching apply_scalar_boundary_parameters! and the temperature
                # conductive IC so physical-space buoyancy sees the intended value.
                set_local_spectral_value!(spec_real, slot, r_idx, sqrt(4 * T(π)) * T(0.5))
            elseif 1 <= l <= 3
                amplitude = T(1e-4)
                set_local_spectral_value!(
                    spec_real,
                    slot,
                    r_idx,
                    amplitude * (rand(T) - T(0.5)),
                )
                if m > 0
                    set_local_spectral_value!(
                        spec_imag,
                        slot,
                        r_idx,
                        amplitude * (rand(T) - T(0.5)),
                    )
                end
            end
        end
    end

    return state
end

function solver_compute_composition_nonlinear!(
    𝔽::CompositionFieldType{T},
    vel_fields,
    𝒟ᵒᶜ::RadialDomainType,
    ws::SolverGradientWorkspace{T};
    geometry::Symbol=solver_default_geometry(), ) where T

    t_start = timing_enabled() ? mpi_wtime() : 0.0

    solver_zero_scalar_work_arrays!(𝔽)
    zero_gradient_workspace!(ws)

    # Composition shares the scalar transport path with temperature, but has no
    # internal source term here; only advection is transformed back to spectral.
    if timing_enabled()
        t_spectral = mpi_wtime()
    end
    compute_all_gradients_spectral!(𝔽, 𝒟ᵒᶜ, ws)
    if timing_enabled()
        𝔽.spectral_time[] += mpi_wtime() - t_spectral
    end

    if timing_enabled()
        t_transform = mpi_wtime()
    end
    transform_field_and_gradients_to_physical!(𝔽, ws)
    if timing_enabled()
        𝔽.transform_time[] += mpi_wtime() - t_transform
    end

    if vel_fields !== nothing
        solver_compute_scalar_advection_local!(𝔽, vel_fields)
    end

    if timing_enabled()
        t_transform = mpi_wtime()
    end

    scalar_nonlinear_to_spectral!(
        𝔽.advection_physical,
        𝔽.nonlinear,
        geometry,
    )
    if timing_enabled()
        𝔽.transform_time[] += mpi_wtime() - t_transform
        𝔽.computation_time[] += mpi_wtime() - t_start
    end
    return 𝔽
end

function apply_composition_implicit_update!(state::SolverState{T,<:AbstractArchitecture}) where T
    composition = state.fields.composition
    composition === nothing && return state

    runtime = state.runtime
    diffusivity = state.parameters.Pm / state.parameters.Sc
    bc = get_bc_vectors(composition)
    timestepper = state.parameters.timestepper
    dt = state.parameters.timestep

    # Keep composition on the same timestep contract as temperature: CNAB2 uses
    # boundary RHS rows, while EAB2 carries a boundary spec for endpoint repair.
    if timestepper isa CNAB2
        matrices = state.implicit_matrices[:composition]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :composition,
            matrices.system_matrices[1].size,
        )
        solver_build_rhs_cnab2!(
            composition.work_spectral,
            composition.spectral,
            composition.nonlinear,
            composition.prev_nonlinear,
            dt,
            matrices;
            work=radial_work,
        )
        solver_solve_composition_implicit_step!(
            composition.spectral,
            composition.work_spectral,
            matrices;
            bc_inner=bc.inner_real,
            bc_outer=bc.outer_real,
            bc_inner_imag=bc.inner_imag,
            bc_outer_imag=bc.outer_imag,
            work=radial_work,
        )
    elseif timestepper isa EAB2
        alu_map = (state.timestep_caches.etd_composition::EAB2CacheEntry{T}).map
        radial_work = get_radial_work!(
            state.timestep_caches,
            :composition,
            runtime.𝒟ᵒᶜ.N,
        )
        composition_bc_code = _composition_bc_code(state.parameters.composition_bcs)
        scalar_bc = build_solver_erk2_scalar_bc(T, runtime.𝒟ᵒᶜ, composition_bc_code)
        bc_spec = with_boundary_mode_values(
            scalar_bc,
            bc.inner_real,
            bc.outer_real,
            bc.inner_imag,
            bc.outer_imag,
        )
        solver_eab2_update_krylov_cached!(
            composition.spectral,
            composition.nonlinear,
            composition.prev_nonlinear,
            alu_map,
            runtime.𝒟ᵒᶜ,
            diffusivity,
            runtime.shtns_config,
            dt;
            m=_timestepper_krylov_dimension(timestepper, state.parameters),
            tol=_timestepper_krylov_tolerance(timestepper, state.parameters),
            bc_spec=bc_spec,
            krylov_work=radial_work,
        )
    else
        matrices = state.implicit_matrices[:composition]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :composition,
            matrices.system_matrices[1].size,
        )
        solver_solve_composition_implicit_step!(
            composition.spectral,
            composition.nonlinear,
            matrices;
            bc_inner=bc.inner_real,
            bc_outer=bc.outer_real,
            bc_inner_imag=bc.inner_imag,
            bc_outer_imag=bc.outer_imag,
            work=radial_work,
        )
    end

    return state
end

function queue_composition_implicit_update!(
    operations::Vector{Function},
    state::SolverState{T,<:AbstractArchitecture},
) where T
    state.fields.composition === nothing && return operations
    push!(operations, () -> apply_composition_implicit_update!(state))
    return operations
end
