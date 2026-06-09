function _shell_conductive_temperature(params::SolverParameters, r)
    η = params.radius_ratio
    ri = η / (1.0 - η)
    ro = 1.0 / (1.0 - η)
    return ri * ro / (ro - ri) * (1.0 / r - 1.0 / ro)
end

function _ball_conductive_temperature(::SolverParameters, r)
    return 1.0 - r^2
end

function initialize_temperature_field!(state::SolverState{
        T, <:AbstractArchitecture}) where {T}
    temperature = state.fields.temperature
    domain = state.backend.outer_core_domain

    spec_real = parent(temperature.spectral.data_real)
    spec_imag = parent(temperature.spectral.data_imag)
    fill!(spec_real, zero(T))
    fill!(spec_imag, zero(T))

    lm_range = local_spectral_mode_indices(temperature.config)
    r_range = local_range(temperature.config.pencils.spec, 3)

    conductive_profile = state.parameters.geometry === :ball ?
                         _ball_conductive_temperature : _shell_conductive_temperature

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
                    sqrt(4 * T(π)) * T(conductive_profile(state.parameters, r))
                )
            elseif 1 <= l <= 4
                amplitude = T(1e-3)
                set_local_spectral_value!(
                    spec_real,
                    slot,
                    r_idx,
                    amplitude * (rand(T) - T(0.5))
                )
                if m > 0
                    set_local_spectral_value!(
                        spec_imag,
                        slot,
                        r_idx,
                        amplitude * (rand(T) - T(0.5))
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
        outer_core_domain::RadialDomainType,
        ws::SolverGradientWorkspace{T};
        geometry::Symbol = solver_default_geometry(),
) where {T}
    return _solver_compute_scalar_nonlinear!(
        temp_𝔽, vel_fields, outer_core_domain, ws;
        add_internal_sources = true,
        geometry = geometry,
    )
end

function apply_temperature_implicit_update!(state::SolverState{
        T, <:AbstractArchitecture}) where {T}
    return _apply_scalar_implicit_update!(
        state,
        state.fields.temperature,
        :temperature,
        state.parameters.Pm / state.parameters.Pr,
        _thermal_bc_code(state.parameters.temperature_bcs),
        solver_solve_temperature_implicit_step!,
        state.timestep_caches.etd_temperature,
    )
end

function queue_temperature_implicit_update!(
        operations::Vector{Function},
        state::SolverState{T, <:AbstractArchitecture}
) where {T}
    push!(operations, () -> apply_temperature_implicit_update!(state))
    return operations
end
