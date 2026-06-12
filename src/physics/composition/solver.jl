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

    # Background (0,0) profile consistent with the prescribed composition BCs:
    # interpolate between the inner/outer boundary coefficients (already √(4π)
    # scaled by apply_scalar_boundary_parameters!) so the IC matches the
    # boundaries instead of jumping. For the default 0/0 BC this is simply 0.
    domain = state.backend.outer_core_domain
    ri = domain.r[1, 4]
    ro = domain.r[domain.N, 4]
    m00 = get_mode_index(composition.config, 0, 0)
    inner_bv = m00 > 0 ? composition.boundary_values[1, m00] : zero(T)
    outer_bv = m00 > 0 ? composition.boundary_values[2, m00] : zero(T)

    @inbounds for lm_idx in lm_range
        lm_idx <= composition.config.nlm || continue
        l = composition.config.l_values[lm_idx]
        m = composition.config.m_values[lm_idx]
        slot = local_spectral_storage_slot(composition.config, lm_idx)

        for r_idx in r_range
            if l == 0 && m == 0
                # BC-consistent background: linear interpolation (in radius)
                # between the √(4π)-scaled inner/outer boundary coefficients, so
                # the conductive seed satisfies the composition boundary values.
                r = domain.r[r_idx, 4]
                frac = ro > ri ? (r - ri) / (ro - ri) : zero(T)
                set_local_spectral_value!(spec_real, slot, r_idx,
                    inner_bv + (outer_bv - inner_bv) * T(frac))
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
        ws::SolverGradientWorkspace{T}) where {T}
    return _solver_compute_scalar_nonlinear!(
        𝔽, vel_fields, outer_core_domain, ws;
        add_internal_sources = false,
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

function queue_composition_implicit_update!(
        operations::Vector{Function},
        state::SolverState{T, <:AbstractArchitecture}
) where {T}
    state.fields.composition === nothing && return operations
    push!(operations, () -> apply_composition_implicit_update!(state))
    return operations
end
