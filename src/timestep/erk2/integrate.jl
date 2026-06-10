# ERK2 integration: field buffers, stage execution, residual logging, and the step entry point.

"""
    SolverERK2FieldBuffers(u, nl, cache)

Allocate ERK2 work buffers matching one spectral field, its nonlinear term, and
the selected stage cache.
"""
function SolverERK2FieldBuffers(
        u::SpectralFieldType{T},
        nl::SpectralFieldType{T},
        cache::ERK2StageCache{T}
) where {T}
    isempty(cache.E_full) && error("ERK2 cache has no precomputed matrices")
    real_data = parent(u.data_real)
    imag_data = parent(u.data_imag)
    nl_real = parent(nl.data_real)
    nl_imag = parent(nl.data_imag)
    cache_lookup = Dict{Int, Int}(l => idx for (idx, l) in enumerate(cache.l_values))
    nr = size(cache.E_full[1], 1)
    workspace = [zeros(T, nr) for _ in 1:8]

    # Zero-initialize the work buffers. `similar` returns UNINITIALIZED memory;
    # any buffer read before it is written on the first step then sees whatever
    # the allocator last left there — zeros on a fresh process (so this looked
    # fine in isolation) but a previous computation's values under the full test
    # suite, which made the magnetic-poloidal step nondeterministic / non-finite.
    z(x) = fill!(similar(x), zero(T))
    return SolverERK2FieldBuffers{T}(
        z(real_data),
        z(imag_data),
        z(real_data),
        z(imag_data),
        z(real_data),
        z(imag_data),
        z(nl_real),
        z(nl_imag),
        z(nl_real),
        z(nl_imag),
        cache_lookup,
        nr,
        workspace
    )
end

function erk2_field_buffers_match(
        buffers::SolverERK2FieldBuffers{T},
        u::SpectralFieldType{T},
        nl::SpectralFieldType{T},
        cache::ERK2StageCache{T}
) where {T}
    size(buffers.linear_real) == size(parent(u.data_real)) || return false
    size(buffers.linear_imag) == size(parent(u.data_imag)) || return false
    size(buffers.n_current_real) == size(parent(nl.data_real)) || return false
    size(buffers.n_current_imag) == size(parent(nl.data_imag)) || return false
    isempty(cache.E_full) && return false
    buffers.nr == size(cache.E_full[1], 1) || return false
    length(buffers._ws) >= 8 || return false
    @inbounds for i in eachindex(cache.l_values)
        get(buffers.cache_lookup, cache.l_values[i], 0) == i || return false
    end
    return true
end

function get_solver_erk2_field_buffers!(
        caches::TimestepCaches{T},
        key::Symbol,
        u::SpectralFieldType{T},
        nl::SpectralFieldType{T},
        cache::ERK2StageCache{T}
) where {T}
    buffers = get(caches.erk2_field_buffers, key, nothing)
    if buffers === nothing || !erk2_field_buffers_match(buffers, u, nl, cache)
        buffers = SolverERK2FieldBuffers(u, nl, cache)
        caches.erk2_field_buffers[key] = buffers
    end
    return buffers
end

"""
    prepare_solver_erk2_field!(buffers, u, nl, cache, config, dt; bc_spec=nothing)

Prepare the first ERK2 stage for one field.

This computes the full-step linear term, the first nonlinear increment, and
the half-step provisional state used for recomputing nonlinear terms.
"""
function prepare_solver_erk2_field!(
        buffers::SolverERK2FieldBuffers{T},
        u::SpectralFieldType{T},
        nl::SpectralFieldType{T},
        cache::ERK2StageCache{T},
        config::SHTnsConfigType,
        dt::Float64;
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing
) where {T}
    cache.use_krylov &&
        error("Krylov-based ERK2 caches are not supported in staged integration")

    u_real = parent(u.data_real)
    u_imag = parent(u.data_imag)
    nl_real = parent(nl.data_real)
    nl_imag = parent(nl.data_imag)

    copyto!(buffers.n_current_real, nl_real)
    copyto!(buffers.n_current_imag, nl_imag)

    r_range = local_range(u.pencil, 3)

    nr = buffers.nr
    ur, ui, nr_vec, ni_vec = buffers._ws[1], buffers._ws[2], buffers._ws[3], buffers._ws[4]
    linear_tmp, k1_tmp,
    stage_tmp,
    stage_phi_tmp = buffers._ws[5], buffers._ws[6], buffers._ws[7], buffers._ws[8]
    half_dt = T(dt) / T(2)

    nlm_total = u.nlm

    linear_real = buffers.linear_real
    linear_imag = buffers.linear_imag
    k1_real = buffers.k1_real
    k1_imag = buffers.k1_imag
    stage_real = buffers.stage_real
    stage_imag = buffers.stage_imag

    # Each spherical-harmonic mode owns a COMPLETE radial profile on a single
    # rank (the spectral pencil keeps the radial dimension local), so every ERK2
    # stage operator (E, φ₁) is applied on the owning rank with no inter-rank
    # communication. Non-owners skip the mode entirely. The previous per-mode
    # Allreduce was redundant: it summed the owner's profile against all-zero
    # contributions from the other ranks and never changed the owner's result.
    for lm_idx in 1:nlm_total
        slot = local_spectral_storage_slot(config, lm_idx)
        slot === nothing && continue

        l = config.l_values[lm_idx]
        cache_idx = get(buffers.cache_lookup, l, 0)
        cache_idx == 0 && error("Missing ERK2 cache entry for l=$l")

        E_full = cache.E_full[cache_idx]
        E_half = cache.E_half[cache_idx]
        phi1_full = cache.phi1_full[cache_idx]
        phi1_half = cache.phi1_half[cache_idx]

        fill!(ur, zero(T))
        fill!(ui, zero(T))
        fill!(nr_vec, zero(T))
        fill!(ni_vec, zero(T))
        gather_local_radial_profile!(ur, ui, u_real, u_imag, slot, r_range)
        gather_local_radial_profile!(
            nr_vec,
            ni_vec,
            buffers.n_current_real,
            buffers.n_current_imag,
            slot,
            r_range
        )

        # Real component
        LA.mul!(linear_tmp, E_full, ur)
        LA.mul!(k1_tmp, phi1_full, nr_vec)
        @inbounds for r in r_range
            set_local_spectral_value!(linear_real, slot, r, linear_tmp[r])
            set_local_spectral_value!(k1_real, slot, r, k1_tmp[r])
        end

        LA.mul!(stage_tmp, E_half, ur)
        LA.mul!(stage_phi_tmp, phi1_half, nr_vec)
        @. stage_tmp = stage_tmp + half_dt * stage_phi_tmp
        if bc_spec !== nothing
            inner_val = boundary_mode_value(bc_spec.inner_mode_values, lm_idx)
            outer_val = boundary_mode_value(bc_spec.outer_mode_values, lm_idx)
            solver_enforce_erk2_bc!(
                stage_tmp, bc_spec.inner, 1, l, nr; value_override = inner_val)
            solver_enforce_erk2_bc!(
                stage_tmp, bc_spec.outer, nr, l, nr; value_override = outer_val)
        else
            stage_tmp[1] = zero(T)
            stage_tmp[nr] = zero(T)
        end
        @inbounds for r in r_range
            set_local_spectral_value!(stage_real, slot, r, stage_tmp[r])
        end

        # Imag component
        LA.mul!(linear_tmp, E_full, ui)
        LA.mul!(k1_tmp, phi1_full, ni_vec)
        @inbounds for r in r_range
            set_local_spectral_value!(linear_imag, slot, r, linear_tmp[r])
            set_local_spectral_value!(k1_imag, slot, r, k1_tmp[r])
        end

        LA.mul!(stage_tmp, E_half, ui)
        LA.mul!(stage_phi_tmp, phi1_half, ni_vec)
        @. stage_tmp = stage_tmp + half_dt * stage_phi_tmp
        if bc_spec !== nothing
            inner_val_i = boundary_mode_value(bc_spec.inner_mode_values_imag, lm_idx)
            outer_val_i = boundary_mode_value(bc_spec.outer_mode_values_imag, lm_idx)
            solver_enforce_erk2_bc!(
                stage_tmp, bc_spec.inner, 1, l, nr; value_override = inner_val_i)
            solver_enforce_erk2_bc!(
                stage_tmp, bc_spec.outer, nr, l, nr; value_override = outer_val_i)
        else
            stage_tmp[1] = zero(T)
            stage_tmp[nr] = zero(T)
        end
        @inbounds for r in r_range
            set_local_spectral_value!(stage_imag, slot, r, stage_tmp[r])
        end
    end

    return buffers
end

"""
    GeoDynamo.erk2_prepare_field!(buffers, u, nl, cache, config, dt; bc_spec=nothing)

Public wrapper for preparing the provisional ERK2 stage for one field.
"""
function GeoDynamo.erk2_prepare_field!(
        buffers::SolverERK2FieldBuffers{T},
        u::SpectralFieldType{T},
        nl::SpectralFieldType{T},
        cache::GeoDynamo.ERK2Cache{T},
        config::SHTnsConfigType,
        dt::Float64;
        bc_spec::Union{Nothing, GeoDynamo.ERK2BoundarySpec{T}} = nothing
) where {T}
    return prepare_solver_erk2_field!(
        buffers,
        u,
        nl,
        cache,
        config,
        dt;
        bc_spec
    )
end

"""
    apply_solver_erk2_stage!(buffers, u)

Overwrite `u` with the provisional half-step ERK2 stage stored in `buffers`.
"""
function apply_solver_erk2_stage!(
        buffers::SolverERK2FieldBuffers{T},
        u::SpectralFieldType{T}
) where {T}
    parent(u.data_real) .= buffers.stage_real
    parent(u.data_imag) .= buffers.stage_imag
    return u
end

"""
    GeoDynamo.erk2_apply_stage!(buffers, u)

Public wrapper that writes the provisional ERK2 stage into a field.
"""
function GeoDynamo.erk2_apply_stage!(
        buffers::SolverERK2FieldBuffers{T},
        u::SpectralFieldType{T}
) where {T}
    apply_solver_erk2_stage!(buffers, u)
end

"""
    store_solver_erk2_stage_nonlinear!(buffers, nl)

Store nonlinear terms evaluated at the provisional ERK2 stage.
"""
function store_solver_erk2_stage_nonlinear!(
        buffers::SolverERK2FieldBuffers{T},
        nl::SpectralFieldType{T}
) where {T}
    copyto!(buffers.stage_nl_real, parent(nl.data_real))
    copyto!(buffers.stage_nl_imag, parent(nl.data_imag))
    return buffers
end

"""
    GeoDynamo.erk2_store_stage_nonlinear!(buffers, nl)

Public wrapper for storing nonlinear terms evaluated at the ERK2 stage.
"""
function GeoDynamo.erk2_store_stage_nonlinear!(
        buffers::SolverERK2FieldBuffers{T},
        nl::SpectralFieldType{T}
) where {T}
    store_solver_erk2_stage_nonlinear!(buffers, nl)
end

"""
    finalize_solver_erk2_field!(buffers, u, cache, config, dt; bc_spec=nothing)

Write the accepted ERK2 update back into `u`.

The final state combines the full-step linear propagation, the first nonlinear
increment, and the `phi2` correction from the staged nonlinear residual.
"""
function finalize_solver_erk2_field!(
        buffers::SolverERK2FieldBuffers{T},
        u::SpectralFieldType{T},
        cache::ERK2StageCache{T},
        config::SHTnsConfigType,
        dt::Float64;
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing
) where {T}
    cache.use_krylov &&
        error("Krylov-based ERK2 caches are not supported in staged integration")

    u_real = parent(u.data_real)
    u_imag = parent(u.data_imag)
    r_range = local_range(u.pencil, 3)

    nr = buffers.nr
    tmp_linear, tmp_k1,
    tmp_Nn, tmp_stage = buffers._ws[1], buffers._ws[2], buffers._ws[3], buffers._ws[4]
    delta, correction,
    result,
    result_real_profile = buffers._ws[5], buffers._ws[6], buffers._ws[7], buffers._ws[8]

    nlm_total = u.nlm

    # Per-mode finalize on the owning rank only. The inputs (linear, k1, N_n, stage
    # N) are owner-local per-mode radial profiles (the spectral pencil keeps the
    # radial dimension local), so no Allreduce is needed — the previous per-mode
    # reduction summed owner data against all-zero contributions from other ranks.
    # Non-owners skip the mode entirely.
    for lm_idx in 1:nlm_total
        slot = local_spectral_storage_slot(config, lm_idx)
        slot === nothing && continue

        l = config.l_values[lm_idx]
        cache_idx = get(buffers.cache_lookup, l, 0)
        cache_idx == 0 && error("Missing ERK2 cache entry for l=$l")
        phi2 = cache.phi2_full[cache_idx]

        # Real component
        fill!(tmp_linear, zero(T));
        fill!(tmp_k1, zero(T))
        fill!(tmp_Nn, zero(T));
        fill!(tmp_stage, zero(T))
        gather_local_radial_profile!(
            tmp_linear, tmp_k1, buffers.linear_real, buffers.k1_real, slot, r_range)
        gather_local_radial_profile!(
            tmp_Nn, tmp_stage, buffers.n_current_real, buffers.stage_nl_real, slot, r_range)

        delta .= tmp_stage
        @. delta = delta - tmp_Nn
        LA.mul!(correction, phi2, delta)
        @. result = tmp_linear + dt * tmp_k1 + T(2) * dt * correction
        if bc_spec !== nothing
            inner_val = boundary_mode_value(bc_spec.inner_mode_values, lm_idx)
            outer_val = boundary_mode_value(bc_spec.outer_mode_values, lm_idx)
            solver_enforce_erk2_bc!(
                result, bc_spec.inner, 1, l, nr; value_override = inner_val)
            solver_enforce_erk2_bc!(
                result, bc_spec.outer, nr, l, nr; value_override = outer_val)
        else
            result[1] = zero(T)
            result[nr] = zero(T)
        end
        copy!(result_real_profile, result)

        # Imag component
        fill!(tmp_linear, zero(T));
        fill!(tmp_k1, zero(T))
        fill!(tmp_Nn, zero(T));
        fill!(tmp_stage, zero(T))
        gather_local_radial_profile!(
            tmp_linear, tmp_k1, buffers.linear_imag, buffers.k1_imag, slot, r_range)
        gather_local_radial_profile!(
            tmp_Nn, tmp_stage, buffers.n_current_imag, buffers.stage_nl_imag, slot, r_range)

        delta .= tmp_stage
        @. delta = delta - tmp_Nn
        LA.mul!(correction, phi2, delta)
        @. result = tmp_linear + dt * tmp_k1 + T(2) * dt * correction
        if bc_spec !== nothing
            inner_val_i = boundary_mode_value(bc_spec.inner_mode_values_imag, lm_idx)
            outer_val_i = boundary_mode_value(bc_spec.outer_mode_values_imag, lm_idx)
            solver_enforce_erk2_bc!(
                result, bc_spec.inner, 1, l, nr; value_override = inner_val_i)
            solver_enforce_erk2_bc!(
                result, bc_spec.outer, nr, l, nr; value_override = outer_val_i)
        else
            result[1] = zero(T)
            result[nr] = zero(T)
        end

        scatter_local_radial_profile!(
            u_real, u_imag, result_real_profile, result, slot, r_range)
    end

    solver_synchronize_pencil_transforms!(u)
    return u
end

"""
    GeoDynamo.erk2_finalize_field!(buffers, u, cache, config, dt; bc_spec=nothing)

Public wrapper for writing the accepted ERK2 update back into a field.
"""
function GeoDynamo.erk2_finalize_field!(
        buffers::SolverERK2FieldBuffers{T},
        u::SpectralFieldType{T},
        cache::GeoDynamo.ERK2Cache{T},
        config::SHTnsConfigType,
        dt::Float64;
        bc_spec::Union{Nothing, GeoDynamo.ERK2BoundarySpec{T}} = nothing
) where {T}
    return finalize_solver_erk2_field!(
        buffers,
        u,
        cache,
        config,
        dt;
        bc_spec
    )
end

"""
    solver_erk2_stage_residual_stats(buffers)

Return global max and L2 norms of the staged nonlinear residual.
"""
function solver_erk2_stage_residual_stats(buffers::SolverERK2FieldBuffers{T}) where {T}
    stage_real = buffers.stage_nl_real
    stage_imag = buffers.stage_nl_imag
    base_real = buffers.n_current_real
    base_imag = buffers.n_current_imag

    local_max = zero(T)
    local_sum = zero(T)

    @inbounds for idx in eachindex(stage_real)
        diff = stage_real[idx] - base_real[idx]
        mag = abs(diff)
        mag > local_max && (local_max = mag)
        local_sum += abs2(diff)
    end

    @inbounds for idx in eachindex(stage_imag)
        diff = stage_imag[idx] - base_imag[idx]
        mag = abs(diff)
        mag > local_max && (local_max = mag)
        local_sum += abs2(diff)
    end

    if mpi_initialized() && mpi_comm_size() > 1
        comm = mpi_comm()
        global_max = allreduce_max(local_max, comm)
        global_sum = allreduce_sum(local_sum, comm)
    else
        global_max = local_max
        global_sum = local_sum
    end

    return (max = global_max, l2 = sqrt(global_sum))
end

"""
    GeoDynamo.erk2_stage_residual_stats(buffers)

Public wrapper returning global max and L2 ERK2 stage-residual norms.
"""
function GeoDynamo.erk2_stage_residual_stats(buffers::SolverERK2FieldBuffers{T}) where {T}
    solver_erk2_stage_residual_stats(buffers)
end

"""
    maybe_log_solver_erk2_stage_residual!(label, buffers, step)

Log ERK2 stage residual diagnostics when diagnostics are enabled and `step`
matches the configured interval.
"""
function maybe_log_solver_erk2_stage_residual!(
        label::Symbol,
        buffers::SolverERK2FieldBuffers,
        step::Int
)
    SOLVER_ERK2_DIAGNOSTICS_ENABLED[] || return nothing
    interval = SOLVER_ERK2_DIAGNOSTICS_INTERVAL[]
    interval <= 0 && return nothing
    (step % interval == 0) || return nothing

    stats = solver_erk2_stage_residual_stats(buffers)
    @info "ERK2 stage residual" field=label step=step max_residual=stats.max l2_residual=stats.l2
    return stats
end

"""
    GeoDynamo.maybe_log_erk2_stage_residual!(label, buffers, step)

Public wrapper for conditional ERK2 stage-residual logging.
"""
function GeoDynamo.maybe_log_erk2_stage_residual!(
        label::Symbol,
        buffers::SolverERK2FieldBuffers,
        step::Int
)
    maybe_log_solver_erk2_stage_residual!(label, buffers, step)
end

"""
    restore_solver_erk2_nonlinear_terms!(state, temp_buffers, vel_tor_buffers, vel_pol_buffers, mag_tor_buffers, mag_pol_buffers, comp_buffers)

Restore nonlinear terms from pre-stage buffers after an ERK2 step.

The staged nonlinear evaluation temporarily overwrites solver nonlinear fields;
this restores the accepted-step histories expected by the rest of the solver.
"""
function restore_solver_erk2_nonlinear_terms!(
        state::SolverState,
        temp_buffers,
        vel_tor_buffers,
        vel_pol_buffers,
        mag_tor_buffers,
        mag_pol_buffers,
        comp_buffers
)
    copyto!(parent(state.fields.temperature.nonlinear.data_real), temp_buffers.n_current_real)
    copyto!(parent(state.fields.temperature.nonlinear.data_imag), temp_buffers.n_current_imag)
    copyto!(parent(state.fields.velocity.nl_toroidal.data_real), vel_tor_buffers.n_current_real)
    copyto!(parent(state.fields.velocity.nl_toroidal.data_imag), vel_tor_buffers.n_current_imag)
    copyto!(parent(state.fields.velocity.nl_poloidal.data_real), vel_pol_buffers.n_current_real)
    copyto!(parent(state.fields.velocity.nl_poloidal.data_imag), vel_pol_buffers.n_current_imag)

    if mag_tor_buffers !== nothing
        copyto!(parent(state.fields.magnetic.nl_toroidal.data_real), mag_tor_buffers.n_current_real)
        copyto!(parent(state.fields.magnetic.nl_toroidal.data_imag), mag_tor_buffers.n_current_imag)
        copyto!(parent(state.fields.magnetic.nl_poloidal.data_real), mag_pol_buffers.n_current_real)
        copyto!(parent(state.fields.magnetic.nl_poloidal.data_imag), mag_pol_buffers.n_current_imag)
    end

    if comp_buffers !== nothing
        copyto!(parent(state.fields.composition.nonlinear.data_real), comp_buffers.n_current_real)
        copyto!(parent(state.fields.composition.nonlinear.data_imag), comp_buffers.n_current_imag)
    end

    return state
end

# Fetch a cached ERK2 boundary spec for `(role, bc_code)`, building it once via
# `builder` on first request. The derivative stencils a spec carries depend only
# on the radial domain and BC code — both fixed for a run — so this avoids
# rebuilding them (each build runs N dense Vandermonde solves) every timestep.
# Per-step endpoint values are attached separately via
# `with_boundary_mode_values`, so the cached base spec is never mutated.
function _get_or_build_erk2_boundary_spec!(
        caches::TimestepCaches{T},
        role::Symbol,
        bc_code::Int,
        builder::F
) where {T, F}
    specs = caches.erk2_boundary_specs
    key = (role, bc_code)
    cached = get(specs, key, nothing)
    cached === nothing || return cached
    spec = builder()::SolverERK2BoundarySpec{T}
    specs[key] = spec
    return spec
end

"""
    integrate_solver_erk2_step!(state)

Run one full ERK2 timestep for all active solver fields.

The routine builds/reuses field caches, prepares provisional stages, recomputes
nonlinear terms at those stages, finalizes each field, applies the
velocity-poloidal influence correction, and restores nonlinear histories.
"""
function integrate_solver_erk2_step!(state::SolverState{
        T, <:AbstractArchitecture}) where {T}
    params = state.parameters
    runtime = state.runtime
    domain = state.backend.outer_core_domain
    nr = domain.N
    velocity_bc_code = _velocity_bc_code(params.velocity_bcs)
    temperature_bc_code = _thermal_bc_code(params.temperature_bcs)
    composition_bc_code = _composition_bc_code(params.composition_bcs)
    theta = _timestepper_implicit_theta(params.timestepper, params)

    # Build the boundary embedding for each active field up front so the stage
    # march can stay uniform across temperature, velocity, magnetic, and composition.
    temp_bc = _get_or_build_erk2_boundary_spec!(
        state.timestep_caches, :temperature, temperature_bc_code,
        () -> build_solver_erk2_scalar_bc(T, domain, temperature_bc_code)
    )
    temp_bc_values = get_bc_vectors(state.fields.temperature)
    temp_bc = with_boundary_mode_values(
        temp_bc,
        temp_bc_values.inner_real,
        temp_bc_values.outer_real,
        temp_bc_values.inner_imag,
        temp_bc_values.outer_imag
    )
    vel_tor_bc = _get_or_build_erk2_boundary_spec!(
        state.timestep_caches, :velocity_tor, velocity_bc_code,
        () -> build_solver_erk2_velocity_tor_bc(
            T,
            domain,
            velocity_bc_code;
            config = runtime.shtns_config,
            rot_omega = 0.0
        )
    )
    vel_pol_bc = _get_or_build_erk2_boundary_spec!(
        state.timestep_caches, :velocity_pol, velocity_bc_code,
        () -> build_solver_erk2_velocity_pol_bc(T, domain, velocity_bc_code)
    )

    # Velocity poloidal evolution needs an influence operator so the accepted
    # step satisfies the no-penetration constraint after ERK2 finalization.
    vel_pol_influence = get_solver_erk2_influence_matrices!(
        state.timestep_caches,
        :velocity_poloidal,
        T,
        runtime.shtns_config,
        runtime.outer_core_domain,
        params.Ek,
        params.timestep,
        velocity_bc_code;
        theta = theta
    )

    temp_cache = get_solver_erk2_temperature_cache!(
        state.timestep_caches,
        params.Pm / params.Pr,
        T,
        runtime.shtns_config,
        runtime.outer_core_domain,
        params.timestep,
        temperature_bc_code;
        use_krylov = false
    )
    temp_buffers = get_solver_erk2_field_buffers!(
        state.timestep_caches,
        :temperature,
        state.fields.temperature.spectral,
        state.fields.temperature.nonlinear,
        temp_cache
    )
    prepare_solver_erk2_field!(
        temp_buffers,
        state.fields.temperature.spectral,
        state.fields.temperature.nonlinear,
        temp_cache,
        runtime.shtns_config,
        params.timestep;
        bc_spec = temp_bc
    )

    vel_tor_cache = get_solver_erk2_cache!(
        state.timestep_caches,
        :velocity_toroidal,
        params.Ek,
        T,
        runtime.shtns_config,
        runtime.outer_core_domain,
        params.timestep;
        use_krylov = false,
        bc_spec = vel_tor_bc
    )
    vel_tor_buffers = get_solver_erk2_field_buffers!(
        state.timestep_caches,
        :velocity_toroidal,
        state.fields.velocity.toroidal,
        state.fields.velocity.nl_toroidal,
        vel_tor_cache
    )
    prepare_solver_erk2_field!(
        vel_tor_buffers,
        state.fields.velocity.toroidal,
        state.fields.velocity.nl_toroidal,
        vel_tor_cache,
        runtime.shtns_config,
        params.timestep;
        bc_spec = vel_tor_bc
    )

    vel_pol_cache = get_solver_erk2_cache!(
        state.timestep_caches,
        :velocity_poloidal,
        params.Ek,
        T,
        runtime.shtns_config,
        runtime.outer_core_domain,
        params.timestep;
        use_krylov = false,
        bc_spec = vel_pol_bc
    )
    vel_pol_buffers = get_solver_erk2_field_buffers!(
        state.timestep_caches,
        :velocity_poloidal,
        state.fields.velocity.poloidal,
        state.fields.velocity.nl_poloidal,
        vel_pol_cache
    )
    prepare_solver_erk2_field!(
        vel_pol_buffers,
        state.fields.velocity.poloidal,
        state.fields.velocity.nl_poloidal,
        vel_pol_cache,
        runtime.shtns_config,
        params.timestep;
        bc_spec = vel_pol_bc
    )

    mag_tor_buffers = nothing
    mag_pol_buffers = nothing
    mag_tor_cache = nothing
    mag_pol_cache = nothing
    mag_tor_bc = nothing
    mag_pol_bc = nothing
    if params.include_magnetic && state.fields.magnetic !== nothing
        mag_tor_bc = _get_or_build_erk2_boundary_spec!(
            state.timestep_caches, :magnetic_tor, 0,
            () -> build_solver_erk2_magnetic_tor_bc(T, nr)
        )
        mag_pol_bc = _get_or_build_erk2_boundary_spec!(
            state.timestep_caches, :magnetic_pol, 0,
            () -> build_solver_erk2_magnetic_pol_bc(T, domain)
        )

        mag_tor_cache = get_solver_erk2_magnetic_toroidal_cache!(
            state.timestep_caches,
            1.0,
            T,
            runtime.shtns_config,
            runtime.outer_core_domain,
            params.timestep;
            use_krylov = false
        )
        mag_tor_buffers = get_solver_erk2_field_buffers!(
            state.timestep_caches,
            :magnetic_toroidal,
            state.fields.magnetic.toroidal,
            state.fields.magnetic.nl_toroidal,
            mag_tor_cache
        )
        prepare_solver_erk2_field!(
            mag_tor_buffers,
            state.fields.magnetic.toroidal,
            state.fields.magnetic.nl_toroidal,
            mag_tor_cache,
            runtime.shtns_config,
            params.timestep;
            bc_spec = mag_tor_bc
        )

        mag_pol_cache = get_solver_erk2_magnetic_poloidal_cache!(
            state.timestep_caches,
            1.0,
            T,
            runtime.shtns_config,
            runtime.outer_core_domain,
            params.timestep;
            use_krylov = false
        )
        mag_pol_buffers = get_solver_erk2_field_buffers!(
            state.timestep_caches,
            :magnetic_poloidal,
            state.fields.magnetic.poloidal,
            state.fields.magnetic.nl_poloidal,
            mag_pol_cache
        )
        prepare_solver_erk2_field!(
            mag_pol_buffers,
            state.fields.magnetic.poloidal,
            state.fields.magnetic.nl_poloidal,
            mag_pol_cache,
            runtime.shtns_config,
            params.timestep;
            bc_spec = mag_pol_bc
        )
    end

    comp_buffers = nothing
    comp_cache = nothing
    comp_bc = nothing
    if state.fields.composition !== nothing
        comp_bc = _get_or_build_erk2_boundary_spec!(
            state.timestep_caches, :composition, composition_bc_code,
            () -> build_solver_erk2_scalar_bc(T, domain, composition_bc_code)
        )
        comp_bc_values = get_bc_vectors(state.fields.composition)
        comp_bc = with_boundary_mode_values(
            comp_bc,
            comp_bc_values.inner_real,
            comp_bc_values.outer_real,
            comp_bc_values.inner_imag,
            comp_bc_values.outer_imag
        )
        comp_cache = get_solver_erk2_composition_cache!(
            state.timestep_caches,
            params.Pm / params.Sc,
            T,
            runtime.shtns_config,
            runtime.outer_core_domain,
            params.timestep,
            composition_bc_code;
            use_krylov = false
        )
        comp_buffers = get_solver_erk2_field_buffers!(
            state.timestep_caches,
            :composition,
            state.fields.composition.spectral,
            state.fields.composition.nonlinear,
            comp_cache
        )
        prepare_solver_erk2_field!(
            comp_buffers,
            state.fields.composition.spectral,
            state.fields.composition.nonlinear,
            comp_cache,
            runtime.shtns_config,
            params.timestep;
            bc_spec = comp_bc
        )
    end

    # Stage application advances each field to the provisional ERK2 state using
    # the cached linear operator and the nonlinear data from the previous step.
    apply_solver_erk2_stage!(temp_buffers, state.fields.temperature.spectral)
    apply_solver_erk2_stage!(vel_tor_buffers, state.fields.velocity.toroidal)
    apply_solver_erk2_stage!(vel_pol_buffers, state.fields.velocity.poloidal)
    if mag_tor_buffers !== nothing
        apply_solver_erk2_stage!(mag_tor_buffers, state.fields.magnetic.toroidal)
        apply_solver_erk2_stage!(mag_pol_buffers, state.fields.magnetic.poloidal)
    end
    if comp_buffers !== nothing
        apply_solver_erk2_stage!(comp_buffers, state.fields.composition.spectral)
    end

    # Recompute nonlinear terms on the staged fields before storing the accepted
    # ERK2 increment for each subsystem.
    compute_solver_nonlinear_terms!(state)

    store_solver_erk2_stage_nonlinear!(temp_buffers, state.fields.temperature.nonlinear)
    maybe_log_solver_erk2_stage_residual!(:temperature, temp_buffers, runtime.timestep_state.step)
    store_solver_erk2_stage_nonlinear!(vel_tor_buffers, state.fields.velocity.nl_toroidal)
    maybe_log_solver_erk2_stage_residual!(:velocity_toroidal, vel_tor_buffers, runtime.timestep_state.step)
    store_solver_erk2_stage_nonlinear!(vel_pol_buffers, state.fields.velocity.nl_poloidal)
    maybe_log_solver_erk2_stage_residual!(:velocity_poloidal, vel_pol_buffers, runtime.timestep_state.step)

    if mag_tor_buffers !== nothing
        store_solver_erk2_stage_nonlinear!(mag_tor_buffers, state.fields.magnetic.nl_toroidal)
        maybe_log_solver_erk2_stage_residual!(:magnetic_toroidal, mag_tor_buffers, runtime.timestep_state.step)
        store_solver_erk2_stage_nonlinear!(mag_pol_buffers, state.fields.magnetic.nl_poloidal)
        maybe_log_solver_erk2_stage_residual!(:magnetic_poloidal, mag_pol_buffers, runtime.timestep_state.step)
    end

    if comp_buffers !== nothing
        store_solver_erk2_stage_nonlinear!(comp_buffers, state.fields.composition.nonlinear)
        maybe_log_solver_erk2_stage_residual!(:composition, comp_buffers, runtime.timestep_state.step)
    end

    # Finalization writes the accepted ERK2 state back to the solver-owned
    # spectral fields and then reapplies the poloidal influence correction.
    finalize_solver_erk2_field!(
        temp_buffers,
        state.fields.temperature.spectral,
        temp_cache,
        runtime.shtns_config,
        params.timestep;
        bc_spec = temp_bc
    )
    finalize_solver_erk2_field!(
        vel_tor_buffers,
        state.fields.velocity.toroidal,
        vel_tor_cache,
        runtime.shtns_config,
        params.timestep;
        bc_spec = vel_tor_bc
    )
    finalize_solver_erk2_field!(
        vel_pol_buffers,
        state.fields.velocity.poloidal,
        vel_pol_cache,
        runtime.shtns_config,
        params.timestep;
        bc_spec = vel_pol_bc
    )
    pol_nr = size(parent(state.fields.velocity.poloidal.data_real), 3)
    apply_solver_velocity_poloidal_influence_correction!(
        state.fields.velocity.poloidal, vel_pol_influence, runtime.shtns_config;
        work = get_radial_work!(state.timestep_caches, :velocity_poloidal_influence, pol_nr).tmp_real
    )

    if mag_tor_buffers !== nothing
        finalize_solver_erk2_field!(
            mag_tor_buffers,
            state.fields.magnetic.toroidal,
            mag_tor_cache,
            runtime.shtns_config,
            params.timestep;
            bc_spec = mag_tor_bc
        )
        finalize_solver_erk2_field!(
            mag_pol_buffers,
            state.fields.magnetic.poloidal,
            mag_pol_cache,
            runtime.shtns_config,
            params.timestep;
            bc_spec = mag_pol_bc
        )
    end

    if comp_buffers !== nothing
        finalize_solver_erk2_field!(
            comp_buffers,
            state.fields.composition.spectral,
            comp_cache,
            runtime.shtns_config,
            params.timestep;
            bc_spec = comp_bc
        )
    end

    report_solver_phi2_conditioning(runtime.timestep_state.step; interval = 100)
    run_diagnostics!(state; interval = 100)

    restore_solver_erk2_nonlinear_terms!(
        state,
        temp_buffers,
        vel_tor_buffers,
        vel_pol_buffers,
        mag_tor_buffers,
        mag_pol_buffers,
        comp_buffers
    )

    return state
end
