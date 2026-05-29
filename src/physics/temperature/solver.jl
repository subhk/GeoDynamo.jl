function _shell_conductive_temperature(params::SolverParameters, r)
    η = params.radius_ratio
    ri = η / (1.0 - η)
    ro = 1.0 / (1.0 - η)
    return ri * ro / (ro - ri) * (1.0 / r - 1.0 / ro)
end

function _ball_conductive_temperature(::SolverParameters, r)
    return 1.0 - r^2
end

function initialize_temperature_field!(state::SolverState{T,<:AbstractArchitecture}) where T
    temperature = state.fields.temperature
    domain = state.backend.outer_core_domain

    spec_real = parent(temperature.spectral.data_real)
    spec_imag = parent(temperature.spectral.data_imag)
    fill!(spec_real, zero(T))
    fill!(spec_imag, zero(T))

    lm_range = local_spectral_mode_indices(temperature.config)
    r_range = local_range(temperature.config.pencils.spec, 3)

    conductive_profile =
        state.parameters.geometry === :ball ? _ball_conductive_temperature : _shell_conductive_temperature

    @inbounds for lm_idx in lm_range
        lm_idx <= temperature.config.nlm || continue
        l = temperature.config.l_values[lm_idx]
        m = temperature.config.m_values[lm_idx]
        slot = local_spectral_storage_slot(temperature.config, lm_idx)

        for r_idx in r_range
            if l == 0 && m == 0
                r = domain.r[r_idx, 4]
                # Orthonormal SH (Y_0^0 = 1/√(4π)): the physical conductive
                # profile is stored as the (0,0) coefficient value·√(4π), the
                # same convention used for boundary values in
                # apply_scalar_boundary_parameters!, so the field starts on the
                # FixedTemperature boundary condition rather than √(4π) away.
                set_local_spectral_value!(
                    spec_real,
                    slot,
                    r_idx,
                    sqrt(4 * T(π)) * T(conductive_profile(state.parameters, r)),
                )
            elseif 1 <= l <= 4
                amplitude = T(1e-3)
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

function solver_compute_temperature_nonlinear!(
    temp_𝔽::TemperatureFieldType{T},
    vel_fields,
    𝒟ᵒᶜ::RadialDomainType,
    ws::SolverGradientWorkspace{T};
    geometry::Symbol=solver_default_geometry(),
) where T
    t_start = timing_enabled() ? mpi_wtime() : 0.0

    solver_zero_scalar_work_arrays!(temp_𝔽)
    zero_gradient_workspace!(ws)

    # Scalar nonlinear terms are formed as physical-space advection/source
    # fields and transformed back to spectral coefficients for timestepping.
    if timing_enabled()
        t_spectral = mpi_wtime()
    end
    compute_all_gradients_spectral!(temp_𝔽, 𝒟ᵒᶜ, ws)
    if timing_enabled()
        temp_𝔽.spectral_time[] += mpi_wtime() - t_spectral
    end

    if timing_enabled()
        t_transform = mpi_wtime()
    end
    transform_field_and_gradients_to_physical!(temp_𝔽, ws)
    if timing_enabled()
        temp_𝔽.transform_time[] += mpi_wtime() - t_transform
    end

    if vel_fields !== nothing
        solver_compute_scalar_advection_local!(temp_𝔽, vel_fields)
    end
    solver_add_internal_sources_local!(temp_𝔽, 𝒟ᵒᶜ)

    if timing_enabled()
        t_transform = mpi_wtime()
    end
    scalar_nonlinear_to_spectral!(
        temp_𝔽.advection_physical,
        temp_𝔽.nonlinear,
        geometry,
    )
    if timing_enabled()
        temp_𝔽.transform_time[] += mpi_wtime() - t_transform
        temp_𝔽.computation_time[] += mpi_wtime() - t_start
    end
    return temp_𝔽
end

function apply_temperature_implicit_update!(state::SolverState{T,<:AbstractArchitecture}) where T
    temperature = state.fields.temperature
    runtime = state.runtime
    diffusivity = state.parameters.Pm / state.parameters.Pr
    bc = get_bc_vectors(temperature)
    timestepper = state.parameters.timestepper
    dt = state.parameters.timestep

    # CNAB2 solves an implicit radial system with boundary RHS values; EAB2
    # advances by exponential action and then enforces the same endpoint spec.
    if timestepper isa CNAB2
        matrices = state.implicit_matrices[:temperature]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :temperature,
            matrices.system_matrices[1].size,
        )
        solver_build_rhs_cnab2!(
            temperature.work_spectral,
            temperature.spectral,
            temperature.nonlinear,
            temperature.prev_nonlinear,
            dt,
            matrices;
            work=radial_work,
        )
        solver_solve_temperature_implicit_step!(
            temperature.spectral,
            temperature.work_spectral,
            matrices;
            bc_inner=bc.inner_real,
            bc_outer=bc.outer_real,
            bc_inner_imag=bc.inner_imag,
            bc_outer_imag=bc.outer_imag,
            work=radial_work,
        )
    elseif timestepper isa EAB2
        alu_map = (state.timestep_caches.etd_temperature::EAB2CacheEntry{T}).map
        radial_work = get_radial_work!(
            state.timestep_caches,
            :temperature,
            runtime.𝒟ᵒᶜ.N,
        )
        temperature_bc_code = _thermal_bc_code(state.parameters.temperature_bcs)
        scalar_bc = build_solver_erk2_scalar_bc(T, runtime.𝒟ᵒᶜ, temperature_bc_code)
        bc_spec = with_boundary_mode_values(
            scalar_bc,
            bc.inner_real,
            bc.outer_real,
            bc.inner_imag,
            bc.outer_imag,
        )
        solver_eab2_update_krylov_cached!(
            temperature.spectral,
            temperature.nonlinear,
            temperature.prev_nonlinear,
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
        matrices = state.implicit_matrices[:temperature]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :temperature,
            matrices.system_matrices[1].size,
        )
        solver_solve_temperature_implicit_step!(
            temperature.spectral,
            temperature.nonlinear,
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

