function initialize_composition_field!(state::SolverState{
        T, <:AbstractArchitecture}) where {T}
    composition = state.fields.composition
    composition === nothing && return state

    spec_real = parent(composition.spectral.data_real)
    spec_imag = parent(composition.spectral.data_imag)
    fill!(spec_real, zero(T))
    fill!(spec_imag, zero(T))

    lm_range = local_spectral_mode_indices(composition.config)
    r_range = local_range(composition.config.pencils.spec, 3)
    domain = state.backend.outer_core_domain

    # BC + source-aware conductive (0,0) profile. The previous default branch
    # linearly interpolated the inner/outer boundary coefficients, ignoring the
    # actual compositional source. The shared helper solves the discrete l=0 BVP
    # with the solver's own Laplacian + boundary rows so the IC is a true discrete
    # equilibrium of the implicit step (and sustaining the source via
    # internal_sources keeps it one). Composition has no nonzero geometry default
    # source: default_H = 0 for both shell and ball, so for the default 0/0 BC
    # this yields C ≡ 0 — identical to the old linear-interp-of-(0,0) default,
    # preserving backward compatibility. κC = Pm/Sc is the implicit diffusivity.
    geom = state.parameters.geometry
    apply_scalar_conductive_l0!(composition, domain, geom,
        state.parameters.compositional_source,
        state.parameters.Pm / state.parameters.Sc, 0.0)

    @inbounds for lm_idx in lm_range
        lm_idx <= composition.config.nlm || continue
        l = composition.config.l_values[lm_idx]
        m = composition.config.m_values[lm_idx]
        slot = local_spectral_storage_slot(composition.config, lm_idx)

        for r_idx in r_range
            if l == 0 && m == 0
                # The BC + source-aware (0,0) coefficient was already written by
                # apply_scalar_conductive_l0! above; nothing to do here.
            elseif 1 <= l <= 3
                amplitude = T(1e-4)
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

function solver_compute_composition_nonlinear!(
        𝔽::CompositionFieldType{T},
        vel_fields,
        outer_core_domain::RadialDomainType,
        ws::SolverGradientWorkspace{T};
        physical_fresh::Bool = false) where {T}
    return _solver_compute_scalar_nonlinear!(
        𝔽, vel_fields, outer_core_domain, ws;
        add_internal_sources = true,
        physical_fresh = physical_fresh,
    )
end

function apply_composition_implicit_update!(state::SolverState{
        T, <:AbstractArchitecture}) where {T}
    composition = state.fields.composition
    composition === nothing && return state
    return _apply_scalar_implicit_update!(
        state,
        composition,
        :composition,
        state.parameters.Pm / state.parameters.Sc,
        _composition_bc_code(state.parameters.composition_bcs),
        solver_solve_composition_implicit_step!,
        state.timestep_caches.etd_composition,
    )
end
