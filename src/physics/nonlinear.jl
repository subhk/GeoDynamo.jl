function solver_zero_gradient_workspace!(ws::SolverGradientWorkspace{T}) where T
    fill!(parent(ws.∇θ_spec.data_real), zero(T))
    fill!(parent(ws.∇θ_spec.data_imag), zero(T))
    fill!(parent(ws.∇φ_spec.data_real), zero(T))
    fill!(parent(ws.∇φ_spec.data_imag), zero(T))
    fill!(parent(ws.∇r_spec.data_real), zero(T))
    fill!(parent(ws.∇r_spec.data_imag), zero(T))
    return ws
end

GeoDynamo.zero_gradient_workspace!(ws::SolverGradientWorkspace{T}) where {T} =
    solver_zero_gradient_workspace!(ws)

function solver_compute_theta_gradient_spectral!(
    𝔽::ScalarFieldType{T},
    ws::SolverGradientWorkspace{T},
) where T
    spec_real = parent(𝔽.spectral.data_real)
    spec_imag = parent(𝔽.spectral.data_imag)
    ∇θ_real = parent(ws.∇θ_spec.data_real)
    ∇θ_imag = parent(ws.∇θ_spec.data_imag)

    lm_range = local_spectral_mode_indices(𝔽.config)
    r_range = local_range(𝔽.config.pencils.spec, 3)
    nlm = 𝔽.config.nlm

    comm = mpi_comm()
    multi = mpi_initialized() && mpi_comm_size(comm) > 1

    full_real = ws.theta_full_real
    full_imag = ws.theta_full_imag
    length(full_real) == nlm || error("theta-gradient workspace real buffer has length $(length(full_real)); expected $nlm")
    length(full_imag) == nlm || error("theta-gradient workspace imaginary buffer has length $(length(full_imag)); expected $nlm")

    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r > size(∇θ_real, 3)
            continue
        end

        fill!(full_real, zero(T))
        fill!(full_imag, zero(T))
        for lm_idx in lm_range
            if lm_idx <= nlm
                slot = local_spectral_storage_slot(𝔽.config, lm_idx)
                if slot !== nothing
                    full_real[lm_idx] = local_spectral_value(spec_real, slot, local_r)
                    full_imag[lm_idx] = local_spectral_value(spec_imag, slot, local_r)
                end
            end
        end
        if multi
            allreduce_sum_in_place!(full_real, comm)
            allreduce_sum_in_place!(full_imag, comm)
        end

        for lm_idx in lm_range
            if lm_idx <= nlm
                slot = local_spectral_storage_slot(𝔽.config, lm_idx)
                if slot === nothing
                    continue
                end

                l = 𝔽.config.l_values[lm_idx]
                m = 𝔽.config.m_values[lm_idx]
                abs_m = abs(m)

                dtheta_real = zero(T)
                dtheta_imag = zero(T)

                if l < 𝔽.config.lmax
                    # Neighbor (l+1, m) storage index is precomputed once in the
                    # workspace; avoids hashing the full mode arrays every call.
                    lm_plus = ws.theta_lm_plus[lm_idx]
                    if lm_plus > 0 && lm_plus <= nlm
                        A_plus = T(l) * sqrt(T((l + abs_m + 1) * (l - abs_m + 1)) /
                                             T((2 * l + 1) * (2 * l + 3)))
                        dtheta_real += A_plus * full_real[lm_plus]
                        dtheta_imag += A_plus * full_imag[lm_plus]
                    end
                end

                if l > abs_m
                    lm_minus = ws.theta_lm_minus[lm_idx]
                    if lm_minus > 0 && lm_minus <= nlm
                        A_minus = -T(l + 1) * sqrt(T((l + abs_m) * (l - abs_m)) /
                                                   T((2 * l - 1) * (2 * l + 1)))
                        dtheta_real += A_minus * full_real[lm_minus]
                        dtheta_imag += A_minus * full_imag[lm_minus]
                    end
                end

                set_local_spectral_value!(∇θ_real, slot, local_r, dtheta_real)
                set_local_spectral_value!(∇θ_imag, slot, local_r, dtheta_imag)
            end
        end
    end

    return ws
end

GeoDynamo.compute_theta_gradient_spectral!(
    𝔽::ScalarFieldType{T},
    ws::SolverGradientWorkspace{T},
) where {T} = solver_compute_theta_gradient_spectral!(𝔽, ws)

function solver_compute_phi_gradient_spectral!(
    𝔽::ScalarFieldType{T},
    ws::SolverGradientWorkspace{T},
) where T
    spec_real = parent(𝔽.spectral.data_real)
    spec_imag = parent(𝔽.spectral.data_imag)
    ∇φ_real = parent(ws.∇φ_spec.data_real)
    ∇φ_imag = parent(ws.∇φ_spec.data_imag)

    lm_range = local_spectral_mode_indices(𝔽.config)
    r_range = local_range(𝔽.config.pencils.spec, 3)

    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(∇φ_real, 3)
            for lm_idx in lm_range
                if lm_idx <= 𝔽.config.nlm
                    slot = local_spectral_storage_slot(𝔽.config, lm_idx)
                    m = 𝔽.config.m_values[lm_idx]
                    if slot !== nothing
                        set_local_spectral_value!(
                            ∇φ_real,
                            slot,
                            local_r,
                            -T(m) * local_spectral_value(spec_imag, slot, local_r),
                        )
                        set_local_spectral_value!(
                            ∇φ_imag,
                            slot,
                            local_r,
                            T(m) * local_spectral_value(spec_real, slot, local_r),
                        )
                    end
                end
            end
        end
    end

    return ws
end

GeoDynamo.compute_phi_gradient_spectral!(
    𝔽::ScalarFieldType{T},
    ws::SolverGradientWorkspace{T},
) where {T} = solver_compute_phi_gradient_spectral!(𝔽, ws)

function solver_compute_radial_gradient_spectral!(
    𝔽::ScalarFieldType{T},
    domain::RadialDomainType,
    ws::SolverGradientWorkspace{T},
) where T
    spec_real = parent(𝔽.spectral.data_real)
    spec_imag = parent(𝔽.spectral.data_imag)
    ∇r_real = parent(ws.∇r_spec.data_real)
    ∇r_imag = parent(ws.∇r_spec.data_imag)

    lm_range = local_spectral_mode_indices(𝔽.config)
    r_range = local_range(𝔽.config.pencils.spec, 3)
    nr = domain.N
    bandwidth = 𝔽.∂r.bandwidth

    if first(r_range) != 1 || last(r_range) != nr
        @assert first(r_range) == 1 && last(r_range) == nr (
            "compute_radial_gradient_spectral! requires full radial domain on each rank. " *
            "Got r_range=$(r_range) but nr=$nr. Radial MPI decomposition is not supported " *
            "for banded radial derivatives without halo exchange.")
    end

    @inbounds for lm_idx in lm_range
        if lm_idx <= 𝔽.config.nlm
            slot = local_spectral_storage_slot(𝔽.config, lm_idx)
            slot === nothing && continue
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(∇r_real, 3)
                    dr_real = zero(T)
                    dr_imag = zero(T)

                    for j in max(1, r_idx - bandwidth):min(nr, r_idx + bandwidth)
                        local_j = j - first(r_range) + 1
                        band_row = bandwidth + 1 + r_idx - j
                        if 1 <= band_row <= 2 * bandwidth + 1
                            coeff = 𝔽.∂r.data[band_row, j]
                            dr_real += coeff * local_spectral_value(spec_real, slot, local_j)
                            dr_imag += coeff * local_spectral_value(spec_imag, slot, local_j)
                        end
                    end

                    set_local_spectral_value!(∇r_real, slot, local_r, dr_real)
                    set_local_spectral_value!(∇r_imag, slot, local_r, dr_imag)
                end
            end
        end
    end

    return ws
end

GeoDynamo.compute_radial_gradient_spectral!(
    𝔽::ScalarFieldType{T},
    domain::RadialDomainType,
    ws::SolverGradientWorkspace{T},
) where {T} = solver_compute_radial_gradient_spectral!(𝔽, domain, ws)

function solver_apply_geometric_factors_spectral!(
    ws::SolverGradientWorkspace{T},
    𝔽::ScalarFieldType{T},
    domain::RadialDomainType,
) where T
    ∇θ_real = parent(ws.∇θ_spec.data_real)
    ∇θ_imag = parent(ws.∇θ_spec.data_imag)
    ∇φ_real = parent(ws.∇φ_spec.data_real)
    ∇φ_imag = parent(ws.∇φ_spec.data_imag)

    r_range = local_range(𝔽.config.pencils.spec, 3)
    lm_range = local_spectral_mode_indices(𝔽.config)

    @inbounds for r_idx in r_range
        if r_idx <= domain.N
            local_r = r_idx - first(r_range) + 1
            r_val = domain.r[r_idx, 4]
            if r_val == 0.0
                for lm_idx in lm_range
                    slot = local_spectral_storage_slot(𝔽.config, lm_idx)
                    if slot !== nothing && local_r <= size(∇θ_real, 3)
                        set_local_spectral_value!(∇θ_real, slot, local_r, zero(T))
                        set_local_spectral_value!(∇θ_imag, slot, local_r, zero(T))
                        set_local_spectral_value!(∇φ_real, slot, local_r, zero(T))
                        set_local_spectral_value!(∇φ_imag, slot, local_r, zero(T))
                    end
                end
            else
                r⁻¹ = domain.r[r_idx, 3]
                for lm_idx in lm_range
                    slot = local_spectral_storage_slot(𝔽.config, lm_idx)
                    if slot !== nothing && local_r <= size(∇θ_real, 3)
                        set_local_spectral_value!(
                            ∇θ_real,
                            slot,
                            local_r,
                            local_spectral_value(∇θ_real, slot, local_r) * r⁻¹,
                        )
                        set_local_spectral_value!(
                            ∇θ_imag,
                            slot,
                            local_r,
                            local_spectral_value(∇θ_imag, slot, local_r) * r⁻¹,
                        )
                        set_local_spectral_value!(
                            ∇φ_real,
                            slot,
                            local_r,
                            local_spectral_value(∇φ_real, slot, local_r) * r⁻¹,
                        )
                        set_local_spectral_value!(
                            ∇φ_imag,
                            slot,
                            local_r,
                            local_spectral_value(∇φ_imag, slot, local_r) * r⁻¹,
                        )
                    end
                end
            end
        end
    end

    return ws
end

GeoDynamo.apply_geometric_factors_spectral!(
    ws::SolverGradientWorkspace{T},
    𝔽::ScalarFieldType{T},
    domain::RadialDomainType,
) where {T} = solver_apply_geometric_factors_spectral!(ws, 𝔽, domain)

function solver_compute_all_gradients_spectral!(
    𝔽::ScalarFieldType{T},
    domain::RadialDomainType,
    ws::SolverGradientWorkspace{T},) where T

    solver_compute_theta_gradient_spectral!(𝔽, ws)
    solver_compute_phi_gradient_spectral!(𝔽, ws)
    solver_compute_radial_gradient_spectral!(𝔽, domain, ws)
    solver_apply_geometric_factors_spectral!(ws, 𝔽, domain)

    return ws
end

GeoDynamo.compute_all_gradients_spectral!(
    𝔽::ScalarFieldType{T},
    domain::RadialDomainType,
    ws::SolverGradientWorkspace{T},
) where {T} = solver_compute_all_gradients_spectral!(𝔽, domain, ws)

solver_main_physical_field(𝔽::ScalarFieldType) =
    error("Solver scalar transform does not support $(typeof(𝔽))")

solver_main_physical_field(𝔽::TemperatureFieldType{T}) where T = 𝔽.temperature

solver_main_physical_field(𝔽::CompositionFieldType{T}) where T = 𝔽.composition

function solver_scalar_spectral_to_physical!(
    spec::SpectralFieldType{T},
    phys::PhysicalFieldType{T},) where T

    config = spec.config
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    phys_data = parent(phys.data)

    @assert size(phys_data, 3) == size(spec_real_data, 3) (
        "Radial dimension mismatch: physical=$(size(phys_data, 3)) vs " *
        "spectral=$(size(spec_real_data, 3)). SH transforms require radial to be local."
    )

    plan = get_sht_plan(config._buffers)
    synth_out = get_synth_out(config._buffers)
    axes_local = phys.pencil.axes_local

    for r_local in axes(phys_data, 3)
        coeffs_matrix = solver_collect_scalar_coefficients(
            spec_real_data,
            spec_imag_data,
            r_local,
            config,
        )

        if plan !== nothing && synth_out !== nothing
            solver_synthesize_scalar!(plan, synth_out, coeffs_matrix)
            local_synth = @view synth_out[axes_local[1], axes_local[2]]
            solver_store_physical_slice!(phys_data, local_synth, r_local, config)
        else
            phys_slice = solver_synthesize_scalar(config, coeffs_matrix)
            local_slice = @view phys_slice[axes_local[1], axes_local[2]]
            solver_store_physical_slice!(phys_data, local_slice, r_local, config)
        end
    end

    return phys
end

@inline function solver_synthesize_scalar!(plan, synth_out, coeffs_matrix)
    return sht_synthesis!(plan, synth_out, coeffs_matrix)
end

@inline function solver_synthesize_scalar(config, coeffs_matrix)
    return sht_synthesis(config, coeffs_matrix)
end

@inline function solver_analyze_scalar!(plan, anal_out, phys_slice)
    return sht_analysis!(plan, anal_out, phys_slice)
end

@inline function solver_analyze_scalar(config, phys_slice)
    return sht_analysis(config, phys_slice)
end

function solver_extract_physical_slice!(
    slice_buffer::Matrix{T},
    phys_data,
    r_local,
    config;
    axes_local::Union{Nothing,Tuple}=nothing,
) where T
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_extract_physical_slice(
            slice_buffer,
            phys_data,
            r_local,
            config;
            axes_local=axes_local,
        )
    end
    return solver_cpu_extract_physical_slice!(
        slice_buffer,
        phys_data,
        r_local,
        config;
        axes_local=axes_local,
    )
end

function solver_cpu_extract_physical_slice!(
    slice_buffer::Matrix{T},
    phys_data,
    r_local,
    config;
    axes_local::Union{Nothing,Tuple}=nothing,
) where T
    nlat, nlon = config.nlat, config.nlon
    fill!(slice_buffer, zero(T))

    has_local_data = r_local <= size(phys_data, 3)

    if axes_local !== nothing
        θ_range = axes_local[1]
        φ_range = axes_local[2]
        if has_local_data
            Threads.@threads for i_local in 1:size(phys_data, 1)
                i_global = θ_range[i_local]
                for j_local in 1:size(phys_data, 2)
                    j_global = φ_range[j_local]
                    slice_buffer[i_global, j_global] = phys_data[i_local, j_local, r_local]
                end
            end
        end
    else
        common_i_range = 1:min(size(phys_data, 1), nlat, size(slice_buffer, 1))
        common_j_range = 1:min(size(phys_data, 2), nlon, size(slice_buffer, 2))
        if has_local_data
            Threads.@threads for i in common_i_range
                for j in common_j_range
                    slice_buffer[i, j] = phys_data[i, j, r_local]
                end
            end
        end
    end

    return slice_buffer
end

function solver_extract_physical_slice(
    phys_data,
    r_local,
    config;
    axes_local::Union{Nothing,Tuple}=nothing,
)
    nlat, nlon = config.nlat, config.nlon
    slice_buffer = solver_get_cached_buffer!(config, :solver_generic_slice_buffer) do
        solver_workspace_zeros(config, eltype(phys_data), nlat, nlon)
    end::Matrix{Float64}
    gathered_buffer = solver_get_cached_buffer!(config, :solver_generic_slice_buffer_gathered) do
        solver_workspace_zeros(config, eltype(phys_data), nlat, nlon)
    end::Matrix{Float64}
    solver_extract_physical_slice!(
        slice_buffer,
        phys_data,
        r_local,
        config;
        axes_local=axes_local,
    )
    allreduce_sum!(slice_buffer, gathered_buffer)
    return gathered_buffer
end

function solver_store_scalar_coefficients!(
    spec_real,
    spec_imag,
    coeffs_matrix,
    r_local,
    config,
)
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_store_scalar_coefficients(
            spec_real,
            spec_imag,
            coeffs_matrix,
            r_local,
            config,
        )
    end
    return solver_cpu_store_scalar_coefficients!(
        spec_real,
        spec_imag,
        coeffs_matrix,
        r_local,
        config,
    )
end

function solver_cpu_store_scalar_coefficients!(
    spec_real,
    spec_imag,
    coeffs_matrix,
    r_local,
    config,
)
    matrix_lmax = size(coeffs_matrix, 1) - 1
    matrix_mmax = size(coeffs_matrix, 2) - 1

    local_modes = local_spectral_mode_indices(config)

    Threads.@threads for lm_idx in local_modes
        slot = local_spectral_storage_slot(config, lm_idx)
        slot === nothing && continue
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        if r_local <= size(spec_real, 3) && r_local <= size(spec_imag, 3)
            if l <= matrix_lmax && m <= matrix_mmax
                coeff = coeffs_matrix[l + 1, m + 1]
                set_local_spectral_value!(spec_real, slot, r_local, real(coeff))
                set_local_spectral_value!(spec_imag, slot, r_local, m == 0 ? 0.0 : imag(coeff))
            else
                set_local_spectral_value!(spec_real, slot, r_local, 0.0)
                set_local_spectral_value!(spec_imag, slot, r_local, 0.0)
            end
        end
    end

    return nothing
end

function solver_scalar_physical_to_spectral!(
    phys::PhysicalFieldType{T},
    spec::SpectralFieldType{T},
) where T
    config = spec.config

    phys_data = parent(phys.data)
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)

    @assert size(phys_data, 3) == size(spec_real_data, 3) (
        "Radial dimension mismatch in solver scalar analysis. " *
        "SH transforms require radial to be local."
    )

    plan = get_sht_plan(config._buffers)
    anal_out = get_anal_out(config._buffers)
    phys_axes_local = phys.pencil.axes_local

    for r_local in axes(phys_data, 3)
        phys_slice = solver_extract_physical_slice(
            phys_data,
            r_local,
            config;
            axes_local=phys_axes_local,
        )

        if plan !== nothing && anal_out !== nothing
            solver_analyze_scalar!(plan, anal_out, phys_slice)
            solver_store_scalar_coefficients!(
                spec_real_data,
                spec_imag_data,
                anal_out,
                r_local,
                config,
            )
        else
            coeffs_matrix = solver_analyze_scalar(config, phys_slice)
            solver_store_scalar_coefficients!(
                spec_real_data,
                spec_imag_data,
                coeffs_matrix,
                r_local,
                config,
            )
        end
    end

    return spec
end

function solver_collect_scalar_coefficients(spec_real, spec_imag, r_local, config)
    lmax, mmax = config.lmax, config.mmax

    coeffs_buffer = solver_get_cached_buffer!(config, :coeffs_buffer) do
        solver_workspace_zeros(config, ComplexF64, lmax + 1, mmax + 1)
    end::Matrix{ComplexF64}

    solver_fill_scalar_coeff_buffer!(
        coeffs_buffer,
        spec_real,
        spec_imag,
        r_local,
        config,
    )

    coeffs_gathered = solver_get_cached_buffer!(config, :coeffs_buffer_gathered) do
        solver_workspace_zeros(config, ComplexF64, lmax + 1, mmax + 1)
    end::Matrix{ComplexF64}

    allreduce_sum!(coeffs_buffer, coeffs_gathered)

    return coeffs_gathered
end

@inline function solver_workspace_zeros(config, ::Type{T}, dims...) where {T}
    workspace = config._buffers.solver_transform_workspace
    if workspace isa TransformWorkspace && !(workspace.arch isa CPU)
        return solver_gpu_scratch_zeros(T, dims...)
    end
    return zeros(T, dims...)
end

const _SOLVER_BUFFERS_KEY_MAP = Dict{Symbol, Symbol}(
    :solver_vector_coeffs_buffer_1          => :vector_coeffs_1,
    :solver_vector_coeffs_buffer_2          => :vector_coeffs_2,
    :solver_vector_coeffs_gathered_1        => :vector_coeffs_gathered_1,
    :solver_vector_coeffs_gathered_2        => :vector_coeffs_gathered_2,
    :solver_pol_rad_coeffs_buffer           => :pol_rad_coeffs,
    :solver_vector_component_buffer_vt      => :vector_component_vt,
    :solver_vector_component_buffer_vp      => :vector_component_vp,
    :solver_generic_slice_buffer            => :generic_slice,
    :solver_generic_slice_buffer_gathered   => :generic_slice_gathered,
    :coeffs_buffer                          => :coeffs_buffer,
    :coeffs_buffer_gathered                 => :coeffs_gathered,
)

@inline function _solver_buffer_field(::Val{key}) where {key}
    error("solver_get_cached_buffer!: unknown key $(repr(key)). Add it to SolverTransformBuffers and _SOLVER_BUFFERS_KEY_MAP.")
end

for (key, field) in _SOLVER_BUFFERS_KEY_MAP
    @eval @inline _solver_buffer_field(::Val{$(QuoteNode(key))}) = Val{$(QuoteNode(field))}()
end

@inline function solver_get_cached_buffer!(create_func::F, config, key::Symbol) where {F}
    return solver_get_cached_buffer!(create_func, config, Val(key))
end

@inline function solver_get_cached_buffer!(create_func::F, config, ::Val{key}) where {F,key}
    return _solver_get_cached_buffer_field!(create_func, config, _solver_buffer_field(Val(key)))
end

@inline function _solver_get_cached_buffer_field!(create_func::F, config, ::Val{field}) where {F,field}
    workspace = config._buffers.solver_transform_workspace
    # Fallback: no solver workspace installed (should not happen in production)
    workspace isa TransformWorkspace || return create_func()
    buffers = workspace.buffers
    # Warm path: a buffer slot, once built, is never resized or cleared, so a
    # populated slot can be returned without taking the lock. This routine runs
    # per radial level, per transform, every timestep, so the lock + closure on
    # the hot path was pure overhead.
    cached = getfield(buffers, field)
    cached === nothing || return cached
    # Cold path: build under the lock, re-checking in case another task filled
    # the slot first (double-checked locking).
    return lock(solver_buffer_cache_lock()) do
        existing = getfield(buffers, field)
        existing === nothing || return existing
        created = create_func()
        setfield!(buffers, field, created)
        return created
    end
end

function allreduce_sum!(sendbuf, recvbuf)
    return allreduce_sum_buffers!(sendbuf, recvbuf)
end

@inline function solver_lookup_lm(idx::Int, config)
    if 1 <= idx <= length(config.l_values)
        return config.l_values[idx], config.m_values[idx]
    end
    return -1, -1
end

function solver_fill_scalar_coeff_buffer!(
    coeffs_buffer::Matrix{ComplexF64},
    spec_real,
    spec_imag,
    r_local,
    config,
)
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_fill_scalar_coeff_buffer(
            coeffs_buffer,
            spec_real,
            spec_imag,
            r_local,
            config,
        )
    end
    return solver_cpu_fill_scalar_coeff_buffer!(
        coeffs_buffer,
        spec_real,
        spec_imag,
        r_local,
        config,
    )
end

function solver_cpu_fill_scalar_coeff_buffer!(
    coeffs_buffer::Matrix{ComplexF64},
    spec_real,
    spec_imag,
    r_local,
    config,
)
    buffer_lmax = size(coeffs_buffer, 1) - 1
    buffer_mmax = size(coeffs_buffer, 2) - 1

    fill!(coeffs_buffer, zero(ComplexF64))

    local_modes = local_spectral_mode_indices(config)

    Threads.@threads for lm_idx in local_modes
        slot = local_spectral_storage_slot(config, lm_idx)
        slot === nothing && continue
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        if r_local <= size(spec_real, 3) && r_local <= size(spec_imag, 3) &&
           l <= buffer_lmax && m <= buffer_mmax
            real_part = local_spectral_value(spec_real, slot, r_local)
            imag_part = local_spectral_value(spec_imag, slot, r_local)
            coeffs_buffer[l + 1, m + 1] = complex(real_part, imag_part)
        end
    end

    return coeffs_buffer
end

function solver_store_physical_slice!(phys_data, phys_slice, r_local, config)
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_store_physical_slice(phys_data, phys_slice, r_local)
    end
    return solver_cpu_store_physical_slice!(phys_data, phys_slice, r_local)
end

function solver_cpu_store_physical_slice!(phys_data, phys_slice, r_local)
    common_i_range = 1:min(size(phys_data, 1), size(phys_slice, 1))
    common_j_range = 1:min(size(phys_data, 2), size(phys_slice, 2))

    Threads.@threads for i in common_i_range
        for j in common_j_range
            if r_local <= size(phys_data, 3)
                phys_data[i, j, r_local] = phys_slice[i, j]
            end
        end
    end

    return phys_data
end

function solver_apply_scalar_transform_batch!(
    spectral_fields::Vector{SpectralFieldType{T}},
    physical_fields::Vector{PhysicalFieldType{T}},
) where T
    @assert length(spectral_fields) == length(physical_fields)

    # Keep transforms sequential because each synthesis path uses MPI collectives.
    for field_idx in eachindex(spectral_fields)
        solver_scalar_spectral_to_physical!(
            spectral_fields[field_idx],
            physical_fields[field_idx],
        )
    end
    return nothing
end

function solver_transform_field_and_gradients_to_physical!(
    𝔽::ScalarFieldType{T},
    ws::SolverGradientWorkspace{T},
) where T
    main_physical_field = solver_main_physical_field(𝔽)
    solver_scalar_spectral_to_physical!(𝔽.spectral, main_physical_field)
    solver_scalar_spectral_to_physical!(ws.∇θ_spec, 𝔽.gradient.θ_component)
    solver_scalar_spectral_to_physical!(ws.∇φ_spec, 𝔽.gradient.φ_component)
    solver_scalar_spectral_to_physical!(ws.∇r_spec, 𝔽.gradient.r_component)
    return 𝔽
end

GeoDynamo.transform_field_and_gradients_to_physical!(
    𝔽::ScalarFieldType{T},
    ws::SolverGradientWorkspace{T},
) where {T} = solver_transform_field_and_gradients_to_physical!(𝔽, ws)

function solver_zero_scalar_work_arrays!(𝔽::ScalarFieldType{T}) where T
    fill!(parent(𝔽.work_spectral.data_real), zero(T))
    fill!(parent(𝔽.work_spectral.data_imag), zero(T))
    fill!(parent(𝔽.work_physical.data), zero(T))
    fill!(parent(𝔽.advection_physical.data), zero(T))
    fill!(parent(𝔽.nonlinear.data_real), zero(T))
    fill!(parent(𝔽.nonlinear.data_imag), zero(T))
    return 𝔽
end

function solver_compute_scalar_advection_local!(
    𝔽::ScalarFieldType{T},
    vel_fields,
) where T
    u_r = parent(vel_fields.velocity.r_component.data)
    u_θ = parent(vel_fields.velocity.θ_component.data)
    u_φ = parent(vel_fields.velocity.φ_component.data)

    ∇r = parent(𝔽.gradient.r_component.data)
    ∇θ = parent(𝔽.gradient.θ_component.data)
    ∇φ = parent(𝔽.gradient.φ_component.data)

    advection = parent(𝔽.advection_physical.data)

    n = length(advection)
    if length(u_r) != n || length(∇r) != n
        error(
            "Advection array size mismatch: advection=$n, u_r=$(length(u_r)), ∇r=$(length(∇r)). " *
            "Check that velocity and gradient fields share the same physical grid.",
        )
    end

    @inbounds @simd for idx in 1:n
        advection[idx] = -(u_r[idx] * ∇r[idx] +
                           u_θ[idx] * ∇θ[idx] +
                           u_φ[idx] * ∇φ[idx])
    end

    return 𝔽
end

function solver_add_internal_sources_local!(
    𝔽::ScalarFieldType{T},
    domain::RadialDomainType,
) where T
    advection = parent(𝔽.advection_physical.data)

    if !all(iszero, 𝔽.internal_sources)
        nlat_local, nlon_local, nr_local = size(𝔽.advection_physical.data)
        r_range = local_range(𝔽.config.pencils.r, 3)

        @inbounds for k in 1:nr_local
            r_idx = k + first(r_range) - 1
            if r_idx <= length(𝔽.internal_sources) && r_idx <= domain.N
                source_value = 𝔽.internal_sources[r_idx]

                @simd for j in 1:nlon_local
                    for i in 1:nlat_local
                        idx = i + (j - 1) * nlat_local + (k - 1) * nlat_local * nlon_local
                        if idx <= length(advection)
                            advection[idx] += source_value
                        end
                    end
                end
            end
        end
    end

    return 𝔽
end

function solver_enforce_ball_scalar_regularity!(spec::SpectralFieldType)
    cfg = spec.config
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)

    lm_range = local_spectral_mode_indices(cfg)
    r_range = local_range(cfg.pencils.spec, 3)

    if !(1 in r_range)
        return spec
    end

    r_local_idx = 1 - first(r_range) + 1
    T = eltype(spec_real)

    @inbounds for lm_idx in lm_range
        if lm_idx <= cfg.nlm
            slot = local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            l = cfg.l_values[lm_idx]
            if l > 0
                set_local_spectral_value!(spec_real, slot, r_local_idx, zero(T))
                set_local_spectral_value!(spec_imag, slot, r_local_idx, zero(T))
            end
        end
    end

    return spec
end

function solver_ball_scalar_physical_to_spectral!(
    phys::PhysicalFieldType{T},
    spec::SpectralFieldType{T},
) where T
    solver_scalar_physical_to_spectral!(phys, spec)
    solver_enforce_ball_scalar_regularity!(spec)
    return spec
end

function solver_scalar_nonlinear_to_spectral!(
    phys::PhysicalFieldType{T},
    spec::SpectralFieldType{T},
    geometry::Symbol,
) where T
    if geometry === :ball
        return solver_ball_scalar_physical_to_spectral!(phys, spec)
    end
    return solver_scalar_physical_to_spectral!(phys, spec)
end

function solver_compute_velocity_nonlinear!(
    velocity_fields::VelocityFieldsType{T},
    temperature_field,
    composition_field,
    magnetic_field,
    domain::RadialDomainType;
    geometry::Symbol=solver_default_geometry(),
    params::Union{Nothing,SolverParameters}=nothing,
) where T
    solver_params = isnothing(params) ? create_solver_parameters() : params
    solver_prepare_velocity_fields!(velocity_fields, domain)
    solver_accumulate_velocity_nonlinear_terms!(
        velocity_fields,
        temperature_field,
        composition_field,
        magnetic_field,
        domain,
        solver_params,
    )
    solver_finish_velocity_nonlinear!(velocity_fields; geometry)
    return velocity_fields
end

function solver_compute_magnetic_nonlinear!(
    magnetic_fields::MagneticFieldsType{T},
    velocity_fields,
    outer_domain::RadialDomainType,
    inner_domain::RadialDomainType,
    rotation_rate::Float64=0.0;
    geometry::Symbol=solver_default_geometry(),
) where T
    solver_prepare_magnetic_fields!(magnetic_fields, outer_domain)
    solver_apply_magnetic_nonlinear_terms!(
        magnetic_fields,
        velocity_fields;
        geometry,
        rotation_rate,
    )
    return magnetic_fields
end

function GeoDynamo.compute_temperature_nonlinear!(
    temp_𝔽::TemperatureFieldType{T},
    vel_fields,
    𝒟ᵒᶜ::RadialDomainType,
    ws::SolverGradientWorkspace{T};
    geometry::Symbol=solver_default_geometry(),
) where {T}
    return solver_compute_temperature_nonlinear!(
        temp_𝔽,
        vel_fields,
        𝒟ᵒᶜ,
        ws;
        geometry,
    )
end

function GeoDynamo.compute_composition_nonlinear!(
    𝔽::CompositionFieldType{T},
    vel_fields,
    𝒟ᵒᶜ::RadialDomainType,
    ws::SolverGradientWorkspace{T};
    geometry::Symbol=solver_default_geometry(),
) where {T}
    return solver_compute_composition_nonlinear!(
        𝔽,
        vel_fields,
        𝒟ᵒᶜ,
        ws;
        geometry,
    )
end

function compute_solver_nonlinear_terms!(state::SolverState)
    # Velocity nonlinear terms define the advecting flow shared by every other
    # subsystem, so that path runs first.
    solver_compute_velocity_nonlinear!(
        state.fields.velocity,
        state.fields.temperature,
        state.fields.composition,
        state.fields.magnetic,
        state.backend.outer_core_domain,
        params=state.parameters,
        geometry=state.parameters.geometry,
    )

    if state.parameters.include_magnetic_field && state.fields.magnetic !== nothing
        # Ball runs reuse the outer-core domain in both slots; shell runs carry
        # a distinct inner-core domain that the magnetic coupling needs.
        inner_domain = isnothing(state.backend.inner_core_domain) ? state.backend.outer_core_domain : state.backend.inner_core_domain
        solver_compute_magnetic_nonlinear!(
            state.fields.magnetic,
            state.fields.velocity,
            state.backend.outer_core_domain,
            inner_domain,
        )
    end

    # Scalar nonlinear terms share one gradient workspace, so they run after the
    # vector paths have finished with the current state.
    solver_compute_temperature_nonlinear!(
        state.fields.temperature,
        state.fields.velocity,
        state.backend.outer_core_domain,
        state.runtime.gradient_workspace,
    )

    if state.fields.composition !== nothing
        solver_compute_composition_nonlinear!(
            state.fields.composition,
            state.fields.velocity,
            state.backend.outer_core_domain,
            state.runtime.gradient_workspace,
        )
    end

    return state
end
