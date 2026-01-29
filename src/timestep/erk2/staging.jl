# ================================================================================
# ERK2 helper utilities for staged evaluation
# ================================================================================

struct ERK2FieldBuffers{T}
    linear_real::Array{T,3}
    linear_imag::Array{T,3}
    k1_real::Array{T,3}
    k1_imag::Array{T,3}
    stage_real::Array{T,3}
    stage_imag::Array{T,3}
    n_current_real::Array{T,3}
    n_current_imag::Array{T,3}
    stage_nl_real::Array{T,3}
    stage_nl_imag::Array{T,3}
    cache_lookup::Dict{Int,Int}
    nr::Int
end

function ERK2FieldBuffers(u::SHTnsSpecField{T}, nl::SHTnsSpecField{T}, cache::ERK2Cache{T}) where T
    isempty(cache.E_full) && error("ERK2 cache has no precomputed matrices")
    real_data = parent(u.data_real)
    imag_data = parent(u.data_imag)
    nl_real = parent(nl.data_real)
    nl_imag = parent(nl.data_imag)
    cache_lookup = Dict{Int,Int}(l => idx for (idx, l) in enumerate(cache.l_values))
    nr = size(cache.E_full[1], 1)
    return ERK2FieldBuffers{T}(
        similar(real_data), similar(imag_data),
        similar(real_data), similar(imag_data),
        similar(real_data), similar(imag_data),
        similar(nl_real), similar(nl_imag),
        similar(nl_real), similar(nl_imag),
        cache_lookup, nr
    )
end

"""
    erk2_prepare_field!(buffers, u, nl, cache, config, dt)

Prepare ERK2 first stage by computing linear evolution and k1 terms.

# MPI Safety
Uses global loop bounds (1:nlm) to ensure all processes call Allreduce
the same number of times, preventing deadlock with uneven lm distribution.
"""
function erk2_prepare_field!(buffers::ERK2FieldBuffers{T}, u::SHTnsSpecField{T},
                             nl::SHTnsSpecField{T}, cache::ERK2Cache{T},
                             config::SHTnsKitConfig, dt::Float64;
                             bc_spec::Union{ERK2BoundarySpec{T}, Nothing}=nothing) where T
    cache.use_krylov && error("Krylov-based ERK2 caches are not supported in staged integration")

    u_real = parent(u.data_real)
    u_imag = parent(u.data_imag)
    nl_real = parent(nl.data_real)
    nl_imag = parent(nl.data_imag)

    copyto!(buffers.n_current_real, nl_real)
    copyto!(buffers.n_current_imag, nl_imag)

    lm_range = get_local_range(u.pencil, 1)
    r_range = get_local_range(u.pencil, 3)

    nr = buffers.nr
    ur = zeros(T, nr)
    ui = similar(ur)
    nr_vec = similar(ur)
    ni_vec = similar(ur)
    linear_tmp = similar(ur)
    k1_tmp = similar(ur)
    stage_tmp = similar(ur)
    stage_phi_tmp = similar(ur)
    half_dt = T(dt) / T(2)

    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    nlm_total = u.nlm

    linear_real = buffers.linear_real
    linear_imag = buffers.linear_imag
    k1_real = buffers.k1_real
    k1_imag = buffers.k1_imag
    stage_real = buffers.stage_real
    stage_imag = buffers.stage_imag

    # Use GLOBAL loop bounds to ensure all processes call Allreduce same number of times
    for lm_idx in 1:nlm_total
        # Check if this process owns this lm mode
        owns_mode = lm_idx in lm_range

        l = config.l_values[lm_idx]
        cache_idx = get(buffers.cache_lookup, l, nothing)
        cache_idx === nothing && error("Missing ERK2 cache entry for l=$l")

        E_full = cache.E_full[cache_idx]
        E_half = cache.E_half[cache_idx]
        phi1_full = cache.phi1_full[cache_idx]
        phi1_half = cache.phi1_half[cache_idx]

        fill!(ur, zero(T)); fill!(ui, zero(T))
        fill!(nr_vec, zero(T)); fill!(ni_vec, zero(T))

        # Only fill if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[r] = u_real[ll, 1, lr]
                    ui[r] = u_imag[ll, 1, lr]
                    nr_vec[r] = buffers.n_current_real[ll, 1, lr]
                    ni_vec[r] = buffers.n_current_imag[ll, 1, lr]
                end
            end
        end

        # ALL processes call Allreduce together (collective operation)
        if multi
            Allreduce!(ur, MPI.SUM, comm)
            Allreduce!(ui, MPI.SUM, comm)
            Allreduce!(nr_vec, MPI.SUM, comm)
            Allreduce!(ni_vec, MPI.SUM, comm)
        end

        mul!(linear_tmp, E_full, ur)
        mul!(k1_tmp, phi1_full, nr_vec)

        # Scatter back only if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    linear_real[ll, 1, lr] = linear_tmp[r]
                    k1_real[ll, 1, lr] = k1_tmp[r]
                end
            end
        end

        mul!(stage_tmp, E_half, ur)
        mul!(stage_phi_tmp, phi1_half, nr_vec)
        @. stage_tmp = stage_tmp + half_dt * stage_phi_tmp
        # Enforce boundary conditions (real part)
        if bc_spec !== nothing
            inner_val = bc_spec.inner_mode_values !== nothing && lm_idx <= length(bc_spec.inner_mode_values) ?
                        bc_spec.inner_mode_values[lm_idx] : nothing
            outer_val = bc_spec.outer_mode_values !== nothing && lm_idx <= length(bc_spec.outer_mode_values) ?
                        bc_spec.outer_mode_values[lm_idx] : nothing
            enforce_erk2_bc!(stage_tmp, bc_spec.inner, 1, l, nr; value_override=inner_val)
            enforce_erk2_bc!(stage_tmp, bc_spec.outer, nr, l, nr; value_override=outer_val)
        else
            stage_tmp[1] = zero(T)
            stage_tmp[nr] = zero(T)
        end

        # Scatter stage results only if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    stage_real[ll, 1, lr] = stage_tmp[r]
                end
            end
        end

        mul!(linear_tmp, E_full, ui)
        mul!(k1_tmp, phi1_full, ni_vec)

        # Scatter imaginary linear/k1 results only if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_imag, 3)
                    linear_imag[ll, 1, lr] = linear_tmp[r]
                    k1_imag[ll, 1, lr] = k1_tmp[r]
                end
            end
        end

        mul!(stage_tmp, E_half, ui)
        mul!(stage_phi_tmp, phi1_half, ni_vec)
        @. stage_tmp = stage_tmp + half_dt * stage_phi_tmp
        # Enforce boundary conditions
        if bc_spec !== nothing
            enforce_erk2_bc!(stage_tmp, bc_spec.inner, 1, l, nr)
            enforce_erk2_bc!(stage_tmp, bc_spec.outer, nr, l, nr)
        else
            stage_tmp[1] = zero(T)
            stage_tmp[nr] = zero(T)
        end

        # Scatter imaginary stage results only if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_imag, 3)
                    stage_imag[ll, 1, lr] = stage_tmp[r]
                end
            end
        end
    end

    return buffers
end

erk2_apply_stage!(buffers::ERK2FieldBuffers{T}, u::SHTnsSpecField{T}) where T = begin
    parent(u.data_real) .= buffers.stage_real
    parent(u.data_imag) .= buffers.stage_imag
    return u
end

erk2_store_stage_nonlinear!(buffers::ERK2FieldBuffers{T}, nl::SHTnsSpecField{T}) where T = begin
    parent_nl_real = parent(nl.data_real)
    parent_nl_imag = parent(nl.data_imag)
    copyto!(buffers.stage_nl_real, parent_nl_real)
    copyto!(buffers.stage_nl_imag, parent_nl_imag)
    return buffers
end

"""
    erk2_finalize_field!(buffers, u, cache, config, dt)

Finalize ERK2 second stage by applying phi2 correction.

# MPI Safety
Uses global loop bounds (1:nlm) to ensure all processes call Allreduce
the same number of times, preventing deadlock with uneven lm distribution.
"""
function erk2_finalize_field!(buffers::ERK2FieldBuffers{T}, u::SHTnsSpecField{T},
                              cache::ERK2Cache{T}, config::SHTnsKitConfig, dt::Float64;
                              bc_spec::Union{ERK2BoundarySpec{T}, Nothing}=nothing) where T
    cache.use_krylov && error("Krylov-based ERK2 caches are not supported in staged integration")

    u_real = parent(u.data_real)
    u_imag = parent(u.data_imag)

    lm_range = get_local_range(u.pencil, 1)
    r_range = get_local_range(u.pencil, 3)

    nr = buffers.nr
    tmp_linear = zeros(T, nr)
    tmp_k1 = similar(tmp_linear)
    tmp_Nn = similar(tmp_linear)
    tmp_stage = similar(tmp_linear)
    delta = similar(tmp_linear)
    correction = similar(tmp_linear)
    result = similar(tmp_linear)

    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    nlm_total = u.nlm

    # Use GLOBAL loop bounds to ensure all processes call Allreduce same number of times
    for lm_idx in 1:nlm_total
        # Check if this process owns this lm mode
        owns_mode = lm_idx in lm_range

        l = config.l_values[lm_idx]
        cache_idx = get(buffers.cache_lookup, l, nothing)
        cache_idx === nothing && error("Missing ERK2 cache entry for l=$l")
        phi2 = cache.phi2_full[cache_idx]

        # Initialize buffers - all processes
        fill!(tmp_linear, zero(T))
        fill!(tmp_k1, zero(T))
        fill!(tmp_Nn, zero(T))
        fill!(tmp_stage, zero(T))

        # Only fill if this process owns the mode (REAL part)
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    tmp_linear[r] = buffers.linear_real[ll, 1, lr]
                    tmp_k1[r] = buffers.k1_real[ll, 1, lr]
                    tmp_Nn[r] = buffers.n_current_real[ll, 1, lr]
                    tmp_stage[r] = buffers.stage_nl_real[ll, 1, lr]
                end
            end
        end

        # ALL processes call Allreduce together (collective operation)
        if multi
            Allreduce!(tmp_linear, MPI.SUM, comm)
            Allreduce!(tmp_k1, MPI.SUM, comm)
            Allreduce!(tmp_Nn, MPI.SUM, comm)
            Allreduce!(tmp_stage, MPI.SUM, comm)
        end

        # ERK2 final formula (Hochbruck & Ostermann 2010, c₂ = 1/2):
        # u^{n+1} = exp(hA)·u + h·φ₁(hA)·N + (h/c₂)·φ₂(hA)·(N_stage - N)
        # With c₂ = 1/2, the correction coefficient is 1/c₂ = 2.
        delta .= tmp_stage
        @. delta = delta - tmp_Nn
        mul!(correction, phi2, delta)
        @. result = tmp_linear + dt * tmp_k1 + T(2) * dt * correction
        # Enforce boundary conditions (real part)
        if bc_spec !== nothing
            inner_val = bc_spec.inner_mode_values !== nothing && lm_idx <= length(bc_spec.inner_mode_values) ?
                        bc_spec.inner_mode_values[lm_idx] : nothing
            outer_val = bc_spec.outer_mode_values !== nothing && lm_idx <= length(bc_spec.outer_mode_values) ?
                        bc_spec.outer_mode_values[lm_idx] : nothing
            enforce_erk2_bc!(result, bc_spec.inner, 1, l, nr; value_override=inner_val)
            enforce_erk2_bc!(result, bc_spec.outer, nr, l, nr; value_override=outer_val)
        else
            result[1] = zero(T)
            result[nr] = zero(T)
        end

        # Scatter back only if this process owns the mode (REAL part)
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    u_real[ll, 1, lr] = result[r]
                end
            end
        end

        # Reset buffers for imaginary part - all processes
        fill!(tmp_linear, zero(T))
        fill!(tmp_k1, zero(T))
        fill!(tmp_Nn, zero(T))
        fill!(tmp_stage, zero(T))

        # Only fill if this process owns the mode (IMAG part)
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_imag, 3)
                    tmp_linear[r] = buffers.linear_imag[ll, 1, lr]
                    tmp_k1[r] = buffers.k1_imag[ll, 1, lr]
                    tmp_Nn[r] = buffers.n_current_imag[ll, 1, lr]
                    tmp_stage[r] = buffers.stage_nl_imag[ll, 1, lr]
                end
            end
        end

        # ALL processes call Allreduce together (collective operation)
        if multi
            Allreduce!(tmp_linear, MPI.SUM, comm)
            Allreduce!(tmp_k1, MPI.SUM, comm)
            Allreduce!(tmp_Nn, MPI.SUM, comm)
            Allreduce!(tmp_stage, MPI.SUM, comm)
        end

        delta .= tmp_stage
        @. delta = delta - tmp_Nn
        mul!(correction, phi2, delta)
        @. result = tmp_linear + dt * tmp_k1 + T(2) * dt * correction
        # Enforce boundary conditions
        if bc_spec !== nothing
            enforce_erk2_bc!(result, bc_spec.inner, 1, l, nr)
            enforce_erk2_bc!(result, bc_spec.outer, nr, l, nr)
        else
            result[1] = zero(T)
            result[nr] = zero(T)
        end

        # Scatter back only if this process owns the mode (IMAG part)
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_imag, 3)
                    u_imag[ll, 1, lr] = result[r]
                end
            end
        end
    end

    synchronize_pencil_transforms!(u)
    return u
end

"""
    erk2_stage_residual_stats(buffers) -> NamedTuple

Compute diagnostic statistics for the difference between stage nonlinear terms
and the base-step nonlinear terms.
"""
function erk2_stage_residual_stats(buffers::ERK2FieldBuffers{T}) where T
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

    if MPI.Initialized() && MPI.Comm_size(get_comm()) > 1
        comm = get_comm()
        global_max = MPI.allreduce(local_max, MPI.MAX, comm)
        global_sum = MPI.allreduce(local_sum, MPI.SUM, comm)
    else
        global_max = local_max
        global_sum = local_sum
    end

    return (max=global_max, l2=sqrt(global_sum))
end

"""
    maybe_log_erk2_stage_residual!(label, buffers, step)

Emit a diagnostic log entry when ERK2 diagnostics are enabled.
"""
function maybe_log_erk2_stage_residual!(label::Symbol, buffers::ERK2FieldBuffers, step::Int)
    ERK2_DIAGNOSTICS_ENABLED[] || return nothing
    interval = ERK2_DIAGNOSTICS_INTERVAL[]
    interval <= 0 && return nothing
    (step % interval == 0) || return nothing

    stats = erk2_stage_residual_stats(buffers)
    @info "ERK2 stage residual" field=label step=step max_residual=stats.max l2_residual=stats.l2
    return stats
end

# Exports are handled by main module
