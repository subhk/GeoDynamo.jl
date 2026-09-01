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

    # BC + source-aware conductive (0,0) profile. The default closed-form
    # branches (a+b/r shell, 1−r² ball) ignore internal heating and the actual
    # boundary VALUES; the shared helper solves the discrete l=0 BVP with the
    # solver's own Laplacian + boundary rows so the IC is a true discrete
    # equilibrium of the implicit step (and sustains the source so it stays one).
    geom = state.parameters.geometry
    if geom === :ball && state.parameters.internal_heating === nothing
        # Backward-compat: ball with no explicit heating keeps the closed-form
        # 1 − r² conductive IC and zero internal source (a decaying transient,
        # exactly as before). Leave internal_sources at zero — do NOT call the
        # helper; this branch uses _ball_conductive_temperature below.
        cond_c = nothing   # fall through to the existing closed-form branch
    else
        # default_H = 0: the old `6κT` ball value was dead code (the ball default
        # takes the closed-form branch above and never reached here), so 0 is
        # equivalent and cleaner. κT = Pm/Pr is the temperature implicit diffusivity.
        cond_c = apply_scalar_conductive_l0!(temperature, domain, geom,
            state.parameters.internal_heating,
            state.parameters.Pm / state.parameters.Pr, 0.0)
    end

    @inbounds for lm_idx in lm_range
        lm_idx <= temperature.config.nlm || continue
        l = temperature.config.l_values[lm_idx]
        m = temperature.config.m_values[lm_idx]
        slot = local_spectral_storage_slot(temperature.config, lm_idx)

        for r_idx in r_range
            if l == 0 && m == 0
                # cond_c !== nothing ⇒ the (0,0) coefficient was already written
                # by apply_scalar_conductive_l0! above; only the closed-form
                # backward-compat fallback (ball, no heating) needs writing here.
                if cond_c === nothing
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
                end
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
        physical_fresh::Bool = false,
) where {T}
    return _solver_compute_scalar_nonlinear!(
        temp_𝔽, vel_fields, outer_core_domain, ws;
        add_internal_sources = true,
        physical_fresh = physical_fresh,
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
