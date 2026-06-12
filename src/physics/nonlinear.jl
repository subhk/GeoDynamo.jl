function zero_gradient_workspace!(ws::SolverGradientWorkspace{T}) where {T}
    fill!(parent(ws.∇θ_spec.data_real), zero(T))
    fill!(parent(ws.∇θ_spec.data_imag), zero(T))
    fill!(parent(ws.∇φ_spec.data_real), zero(T))
    fill!(parent(ws.∇φ_spec.data_imag), zero(T))
    fill!(parent(ws.∇r_spec.data_real), zero(T))
    fill!(parent(ws.∇r_spec.data_imag), zero(T))
    return ws
end

# Number of cross-rank spectral gather-reduce passes performed by the most recent
# `compute_theta_gradient_spectral!` call. Under multi-rank MPI this equals the
# number of MPI collectives issued; the batched gather keeps it at 2 (real +
# imaginary) regardless of the radial-level count, versus 2*nr for a per-level
# gather. Tests reset it to 0 and assert it stays at 2.
const _THETA_GATHER_REDUCE_COUNT = Ref(0)

function compute_theta_gradient_spectral!(
        𝔽::ScalarFieldType{T},
        ws::SolverGradientWorkspace{T}
) where {T}
    spec_real = parent(𝔽.spectral.data_real)
    spec_imag = parent(𝔽.spectral.data_imag)
    ∇θ_real = parent(ws.∇θ_spec.data_real)
    ∇θ_imag = parent(ws.∇θ_spec.data_imag)

    lm_range = local_spectral_mode_indices(𝔽.config)
    r_range = local_range(𝔽.config.pencils.spec, 3)
    nlm = 𝔽.config.nlm

    comm = mpi_comm()
    multi = mpi_initialized() && mpi_comm_size(comm) > 1

    # Gather scratch shaped (nlm, nr_local): column `local_r` holds the full
    # spectrum at that radial level. Stacking every level lets the cross-rank
    # sum use one collective per component instead of one per level.
    full_real = ws.theta_full_real
    full_imag = ws.theta_full_imag
    size(full_real, 1) == nlm ||
        error("theta-gradient workspace real buffer has $(size(full_real, 1)) modes; expected $nlm")
    size(full_imag, 1) == nlm ||
        error("theta-gradient workspace imaginary buffer has $(size(full_imag, 1)) modes; expected $nlm")

    # Phase 1: scatter this rank's owned modes into every radial column. Cleared
    # first so modes owned by other ranks contribute zero before the sum.
    fill!(full_real, zero(T))
    fill!(full_imag, zero(T))
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r > size(∇θ_real, 3)
            continue
        end
        for lm_idx in lm_range
            if lm_idx <= nlm
                slot = local_spectral_storage_slot(𝔽.config, lm_idx)
                if slot !== nothing
                    full_real[lm_idx, local_r] = local_spectral_value(spec_real, slot, local_r)
                    full_imag[lm_idx, local_r] = local_spectral_value(spec_imag, slot, local_r)
                end
            end
        end
    end

    # Phase 2: one collective per component over all radial levels at once
    # (the former per-level reduce issued 2*nr collectives).
    _THETA_GATHER_REDUCE_COUNT[] += 1
    multi && allreduce_sum_in_place!(full_real, comm)
    _THETA_GATHER_REDUCE_COUNT[] += 1
    multi && allreduce_sum_in_place!(full_imag, comm)

    # Phase 3: apply the ∂/∂θ recurrence using the gathered full spectrum.
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r > size(∇θ_real, 3)
            continue
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
                        dtheta_real += A_plus * full_real[lm_plus, local_r]
                        dtheta_imag += A_plus * full_imag[lm_plus, local_r]
                    end
                end

                if l > abs_m
                    lm_minus = ws.theta_lm_minus[lm_idx]
                    if lm_minus > 0 && lm_minus <= nlm
                        A_minus = -T(l + 1) * sqrt(T((l + abs_m) * (l - abs_m)) /
                                       T((2 * l - 1) * (2 * l + 1)))
                        dtheta_real += A_minus * full_real[lm_minus, local_r]
                        dtheta_imag += A_minus * full_imag[lm_minus, local_r]
                    end
                end

                set_local_spectral_value!(∇θ_real, slot, local_r, dtheta_real)
                set_local_spectral_value!(∇θ_imag, slot, local_r, dtheta_imag)
            end
        end
    end

    return ws
end

function compute_phi_gradient_spectral!(
        𝔽::ScalarFieldType{T},
        ws::SolverGradientWorkspace{T}
) where {T}
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
                            -T(m) * local_spectral_value(spec_imag, slot, local_r)
                        )
                        set_local_spectral_value!(
                            ∇φ_imag,
                            slot,
                            local_r,
                            T(m) * local_spectral_value(spec_real, slot, local_r)
                        )
                    end
                end
            end
        end
    end

    return ws
end

function compute_radial_gradient_spectral!(
        𝔽::ScalarFieldType{T},
        domain::RadialDomainType,
        ws::SolverGradientWorkspace{T}
) where {T}
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
                            dr_real += coeff *
                                       local_spectral_value(spec_real, slot, local_j)
                            dr_imag += coeff *
                                       local_spectral_value(spec_imag, slot, local_j)
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

function apply_geometric_factors_spectral!(
        ws::SolverGradientWorkspace{T},
        𝔽::ScalarFieldType{T},
        domain::RadialDomainType
) where {T}
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
                            local_spectral_value(∇θ_real, slot, local_r) * r⁻¹
                        )
                        set_local_spectral_value!(
                            ∇θ_imag,
                            slot,
                            local_r,
                            local_spectral_value(∇θ_imag, slot, local_r) * r⁻¹
                        )
                        set_local_spectral_value!(
                            ∇φ_real,
                            slot,
                            local_r,
                            local_spectral_value(∇φ_real, slot, local_r) * r⁻¹
                        )
                        set_local_spectral_value!(
                            ∇φ_imag,
                            slot,
                            local_r,
                            local_spectral_value(∇φ_imag, slot, local_r) * r⁻¹
                        )
                    end
                end
            end
        end
    end

    return ws
end

function compute_all_gradients_spectral!(
        𝔽::ScalarFieldType{T},
        domain::RadialDomainType,
        ws::SolverGradientWorkspace{T}) where {T}
    compute_theta_gradient_spectral!(𝔽, ws)
    compute_phi_gradient_spectral!(𝔽, ws)
    compute_radial_gradient_spectral!(𝔽, domain, ws)
    apply_geometric_factors_spectral!(ws, 𝔽, domain)

    return ws
end

function solver_main_physical_field(𝔽::ScalarFieldType)
    error("Solver scalar transform does not support $(typeof(𝔽))")
end

solver_main_physical_field(𝔽::TemperatureFieldType{T}) where {T} = 𝔽.temperature

solver_main_physical_field(𝔽::CompositionFieldType{T}) where {T} = 𝔽.composition

# Number of batched cross-rank SPECTRAL collectives issued by the most recent
# scalar transform.  The Phase-3 path performs exactly ONE θ_comm m-axis
# redistribution per call (synthesis: Allgatherv in spec_storage_to_solve!;
# analysis: Allreduce in solve_to_spec_storage!), batched over ALL radial levels
# (independent of nr).  The heavy Legendre/FFT work is θ-distributed inside
# dist_synthesis!/dist_analysis! and is NOT counted here.
const _SCALAR_GATHER_REDUCE_COUNT = Ref(0)

# Number of times the Phase-3 DistTransposePlan scalar path has run (synthesis +
# analysis each increment once).  Used by tests to assert the new path is wired in.
const _SCALAR_DISTTRANSPOSE_COUNT = Ref(0)

# Number of times the Phase-3 DistTransposePlan VECTOR (sphtor) path has run.  Both
# the solver path (vector_spectral_to_physical! / vector_physical_to_spectral!) and
# the non-solver path (shtnskit_vector_synthesis! / shtnskit_vector_analysis!)
# increment this once per call.  Used by tests to assert the new path is wired in.
const _VECTOR_DISTTRANSPOSE_COUNT = Ref(0)

# ---------------------------------------------------------------------------
# Phase 3 (Task 2): scalar transforms via DistTransposePlan.
#
# Storage contract (config.pencils.spec, decomp (2,1)):
#   parent(spec.data_real/imag) = (l_local, m_local, nr)  — l over r_comm,
#   m over θ_comm, r LOCAL (full nr on every rank).
#
# Physical field (phys on config.pencils.r, decomp (1,3)):
#   parent(phys.data) = (nlat_local, nlon, nr_local)  — θ over θ_comm,
#   φ LOCAL, r over r_comm.
#
# Plan spatial buffer (allocate_spatial): parent = (nlon, nlat_local, nlev)
#   — φ LOCAL, θ over θ_comm, lev = nr_local.  The first two axes are the
#   transpose of phys.data's; r_local ↔ lev align (plan nlev = phys r_local).
# ---------------------------------------------------------------------------

function _build_scalar_scratch(config, plan)
    Alm      = SHTnsKit.allocate_spectral(plan)
    fspatial = SHTnsKit.allocate_spatial(plan)
    scratch  = _get_disttranspose_scratch(config, plan)
    # Share scratch.solve (the adapter's persistent buffer) so that
    # from_spec_solve! can use the cached Transposition plan t_bwd.
    # All scalar transform calls are sequential — no aliasing hazard.
    solve = scratch.solve
    return (; Alm, fspatial, solve)
end

function _scalar_scratch(config, plan)
    b = config._buffers
    sc = b.p3_scalar_scratch
    sc !== nothing && return sc
    lock(_DISTTRANSPOSE_LOCK) do
        if b.p3_scalar_scratch === nothing
            b.p3_scalar_scratch = _build_scalar_scratch(config, plan)
        end
        return b.p3_scalar_scratch
    end
end

function _build_vector_scratch(config, plan)
    Slm    = SHTnsKit.allocate_spectral(plan)   # spheroidal/poloidal
    Tlm    = SHTnsKit.allocate_spectral(plan)   # toroidal
    Vr_alm = SHTnsKit.allocate_spectral(plan)   # radial-scaled poloidal (l(l+1)/r²·P)
    Vt     = SHTnsKit.allocate_spatial(plan)    # θ-tangential (nlon, nlat_local, nlev)
    Vp     = SHTnsKit.allocate_spatial(plan)    # φ-tangential
    Vr     = SHTnsKit.allocate_spatial(plan)    # radial (scalar synthesis of vr coeffs)
    scratch = _get_disttranspose_scratch(config, plan)
    # Share scratch.solve (the adapter's persistent buffer) so that
    # from_spec_solve! can use the cached Transposition plan t_bwd.
    # The vector path calls from_spec_solve! up to three times (S, T, Vr) but
    # always sequentially, so sharing scratch.solve is safe.
    solve = scratch.solve
    # Storage-layout slabs (l_local, m_local, nr) for the solenoidal S and v_r
    # coefficient fields, computed where r is fully local (r-dist support).
    spec_dims = length.(PencilArrays.range_local(config.pencils.spec))
    Ssto_re  = zeros(Float64, spec_dims)
    Ssto_im  = zeros(Float64, spec_dims)
    Vrsto_re = zeros(Float64, spec_dims)
    Vrsto_im = zeros(Float64, spec_dims)
    return (; Slm, Tlm, Vr_alm, Vt, Vp, Vr, solve,
              Ssto_re, Ssto_im, Vrsto_re, Vrsto_im)
end

function _vector_scratch(config, plan)
    b = config._buffers
    sc = b.p3_vector_scratch
    sc !== nothing && return sc
    lock(_DISTTRANSPOSE_LOCK) do
        if b.p3_vector_scratch === nothing
            b.p3_vector_scratch = _build_vector_scratch(config, plan)
        end
        return b.p3_vector_scratch
    end
end

"""
    _clear_p3_transform_caches!(config)

Clear all five Phase-3 scratch slots for `config`.  All slots live on
`config._buffers` (set to `nothing` directly): the DistTransposePlan,
DistTranspose scratch, m-bridge, scalar-scratch, and vector-scratch.
Called by `clear_buffer_cache!` so transient configs (e.g. across a test
suite) do not accumulate stale entries.
"""
function _clear_p3_transform_caches!(config)
    # Nil the five config-owned slots under the build lock so a concurrent
    # lazy build can't observe a half-cleared `_buffers` holder.
    lock(_DISTTRANSPOSE_LOCK) do
        config._buffers.disttranspose_plan    = nothing
        config._buffers.disttranspose_scratch = nothing
        config._buffers.disttranspose_mbridge = nothing
        config._buffers.p3_scalar_scratch     = nothing
        config._buffers.p3_vector_scratch     = nothing
    end
    return nothing
end

function scalar_spectral_to_physical!(
        spec::SpectralFieldType{T},
        phys::PhysicalFieldType{T}) where {T}
    config = spec.config
    plan   = get_disttranspose_plan(config)

    sc       = _scalar_scratch(config, plan)
    Alm      = sc.Alm
    fspatial = sc.fspatial
    solve    = sc.solve

    # 1. spec storage → solve (l/r already aligned; ONE θ_comm m-axis collective).
    _SCALAR_GATHER_REDUCE_COUNT[] += 1
    spec_storage_to_solve!(config, solve, parent(spec.data_real), parent(spec.data_imag), plan)

    # 2. solve → Alm (batched r_comm l-transpose; r-full → r-local).
    from_spec_solve!(config, Alm, solve, plan)

    # 3. Distributed synthesis: Alm → fspatial (φ,θ,lev), batched over all nr_local.
    SHTnsKit.dist_synthesis!(plan, fspatial, Alm)

    # 4. Copy fspatial (nlon, nlat_local, lev) → phys.data (nlat_local, nlon, nr_local)
    #    with the first-two-axis transpose; lev ↔ r_local are local per rank.
    #    Function barrier: sc is ::Any from IdDict cache; parent(fspatial) would be
    #    ::Any and box every element.  Pass the parents as concrete-typed args.
    _copy_spatial_to_physical!(parent(phys.data), parent(fspatial))

    _SCALAR_DISTTRANSPOSE_COUNT[] += 1
    return phys
end

function extract_physical_slice!(
        slice_buffer::AbstractMatrix{T},
        phys_data,
        r_local,
        config;
        axes_local::Union{Nothing, Tuple} = nothing
) where {T}
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_extract_physical_slice(
            slice_buffer,
            phys_data,
            r_local,
            config;
            axes_local = axes_local
        )
    end
    return cpu_extract_physical_slice!(
        slice_buffer,
        phys_data,
        r_local,
        config;
        axes_local = axes_local
    )
end

function cpu_extract_physical_slice!(
        slice_buffer::AbstractMatrix{T},
        phys_data,
        r_local,
        config;
        axes_local::Union{Nothing, Tuple} = nothing
) where {T}
    nlat, nlon = config.nlat, config.nlon
    fill!(slice_buffer, zero(T))

    has_local_data = r_local <= size(phys_data, 3)

    if axes_local !== nothing
        θ_range = axes_local[1]
        φ_range = axes_local[2]
        if has_local_data
            for i_local in 1:size(phys_data, 1)
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
            for i in common_i_range
                for j in common_j_range
                    slice_buffer[i, j] = phys_data[i, j, r_local]
                end
            end
        end
    end

    return slice_buffer
end

function extract_physical_slice(
        phys_data,
        r_local,
        config;
        axes_local::Union{Nothing, Tuple} = nothing
)
    nlat, nlon = config.nlat, config.nlon
    slice_buffer = solver_get_cached_buffer!(config, :solver_generic_slice_buffer) do
        workspace_zeros(config, eltype(phys_data), nlat, nlon)
    end::Matrix{Float64}
    gathered_buffer = solver_get_cached_buffer!(
        config, :solver_generic_slice_buffer_gathered) do
        workspace_zeros(config, eltype(phys_data), nlat, nlon)
    end::Matrix{Float64}
    extract_physical_slice!(
        slice_buffer,
        phys_data,
        r_local,
        config;
        axes_local = axes_local
    )
    allreduce_sum!(slice_buffer, gathered_buffer)
    return gathered_buffer
end

function store_scalar_coefficients!(
        spec_real,
        spec_imag,
        coeffs_matrix,
        r_local,
        config
)
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_store_scalar_coefficients(
            spec_real,
            spec_imag,
            coeffs_matrix,
            r_local,
            config
        )
    end
    return cpu_store_scalar_coefficients!(
        spec_real,
        spec_imag,
        coeffs_matrix,
        r_local,
        config
    )
end

function cpu_store_scalar_coefficients!(
        spec_real,
        spec_imag,
        coeffs_matrix,
        r_local,
        config
)
    matrix_lmax = size(coeffs_matrix, 1) - 1
    matrix_mmax = size(coeffs_matrix, 2) - 1

    local_modes = local_spectral_mode_indices(config)

    for lm_idx in local_modes
        slot = local_spectral_storage_slot(config, lm_idx)
        slot === nothing && continue
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        if r_local <= size(spec_real, 3) && r_local <= size(spec_imag, 3)
            if l <= matrix_lmax && m <= matrix_mmax
                coeff = coeffs_matrix[l + 1, m + 1]
                set_local_spectral_value!(spec_real, slot, r_local, real(coeff))
                set_local_spectral_value!(spec_imag, slot, r_local, m == 0 ? 0.0 :
                                                                    imag(coeff))
            else
                set_local_spectral_value!(spec_real, slot, r_local, 0.0)
                set_local_spectral_value!(spec_imag, slot, r_local, 0.0)
            end
        end
    end

    return nothing
end

function scalar_physical_to_spectral!(
        phys::PhysicalFieldType{T},
        spec::SpectralFieldType{T}
) where {T}
    config = spec.config
    plan   = get_disttranspose_plan(config)

    sc       = _scalar_scratch(config, plan)
    Alm      = sc.Alm
    fspatial = sc.fspatial

    # 1. Copy phys.data (nlat_local, nlon, nr_local) → fspatial (nlon, nlat_local, lev)
    #    with the first-two-axis transpose; lev ↔ r_local are local per rank.
    #    Function barrier: sc is ::Any from IdDict cache; parent(fspatial) would be
    #    ::Any and box every element.  Pass the parents as concrete-typed args.
    _copy_physical_to_spatial!(parent(fspatial), parent(phys.data))

    # 2. Distributed analysis: fspatial → Alm (batched over all nr_local).
    SHTnsKit.dist_analysis!(plan, Alm, fspatial)

    # 3. Alm → solve (batched r_comm l-transpose; r-local → r-full).  Returns the
    #    cached scratch.solve, consumed immediately below.
    solve = GeoDynamo.to_spec_solve(config, Alm, plan)

    # 4. solve → spec storage (l/r already aligned; ONE θ_comm m-axis collective).
    _SCALAR_GATHER_REDUCE_COUNT[] += 1
    solve_to_spec_storage!(config, parent(spec.data_real), parent(spec.data_imag), solve, plan)

    _SCALAR_DISTTRANSPOSE_COUNT[] += 1
    return spec
end

function collect_scalar_coefficients(spec_real, spec_imag, r_local, config)
    lmax, mmax = config.lmax, config.mmax

    coeffs_buffer = solver_get_cached_buffer!(config, :coeffs_buffer) do
        workspace_zeros(config, ComplexF64, lmax + 1, mmax + 1)
    end::Matrix{ComplexF64}

    fill_scalar_coeff_buffer!(
        coeffs_buffer,
        spec_real,
        spec_imag,
        r_local,
        config
    )

    coeffs_gathered = solver_get_cached_buffer!(config, :coeffs_buffer_gathered) do
        workspace_zeros(config, ComplexF64, lmax + 1, mmax + 1)
    end::Matrix{ComplexF64}

    allreduce_sum!(coeffs_buffer, coeffs_gathered)

    return coeffs_gathered
end

@inline function workspace_zeros(config, ::Type{T}, dims...) where {T}
    workspace = config._buffers.solver_transform_workspace
    if workspace isa TransformWorkspace && !(workspace.arch isa CPU)
        return solver_gpu_scratch_zeros(T, dims...)
    end
    return zeros(T, dims...)
end

const _SOLVER_BUFFERS_KEY_MAP = Dict{Symbol, Symbol}(
    :solver_vector_coeffs_buffer_1 => :vector_coeffs_1,
    :solver_vector_coeffs_buffer_2 => :vector_coeffs_2,
    :solver_vector_coeffs_gathered_1 => :vector_coeffs_gathered_1,
    :solver_vector_coeffs_gathered_2 => :vector_coeffs_gathered_2,
    :solver_pol_rad_coeffs_buffer => :pol_rad_coeffs,
    :solver_vector_component_buffer_vt => :vector_component_vt,
    :solver_vector_component_buffer_vp => :vector_component_vp,
    :solver_generic_slice_buffer => :generic_slice,
    :solver_generic_slice_buffer_gathered => :generic_slice_gathered,
    :coeffs_buffer => :coeffs_buffer,
    :coeffs_buffer_gathered => :coeffs_gathered
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

@inline function solver_get_cached_buffer!(create_func::F, config, ::Val{key}) where {
        F, key}
    return _solver_get_cached_buffer_field!(create_func, config, _solver_buffer_field(Val(key)))
end

@inline function _solver_get_cached_buffer_field!(create_func::F, config, ::Val{field}) where {
        F, field}
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

@inline function lookup_lm(idx::Int, config)
    if 1 <= idx <= length(config.l_values)
        return config.l_values[idx], config.m_values[idx]
    end
    return -1, -1
end

function fill_scalar_coeff_buffer!(
        coeffs_buffer::AbstractMatrix{ComplexF64},
        spec_real,
        spec_imag,
        r_local,
        config
)
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_fill_scalar_coeff_buffer(
            coeffs_buffer,
            spec_real,
            spec_imag,
            r_local,
            config
        )
    end
    return cpu_fill_scalar_coeff_buffer!(
        coeffs_buffer,
        spec_real,
        spec_imag,
        r_local,
        config
    )
end

function cpu_fill_scalar_coeff_buffer!(
        coeffs_buffer::AbstractMatrix{ComplexF64},
        spec_real,
        spec_imag,
        r_local,
        config
)
    buffer_lmax = size(coeffs_buffer, 1) - 1
    buffer_mmax = size(coeffs_buffer, 2) - 1

    fill!(coeffs_buffer, zero(ComplexF64))

    local_modes = local_spectral_mode_indices(config)

    for lm_idx in local_modes
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

function cpu_store_physical_slice!(phys_data, phys_slice, r_local)
    common_i_range = 1:min(size(phys_data, 1), size(phys_slice, 1))
    common_j_range = 1:min(size(phys_data, 2), size(phys_slice, 2))

    for i in common_i_range
        for j in common_j_range
            if r_local <= size(phys_data, 3)
                phys_data[i, j, r_local] = phys_slice[i, j]
            end
        end
    end

    return phys_data
end

function apply_scalar_transform_batch!(
        spectral_fields::Vector{SpectralFieldType{T}},
        physical_fields::Vector{PhysicalFieldType{T}}
) where {T}
    @assert length(spectral_fields) == length(physical_fields)

    # Keep transforms sequential because each synthesis path uses MPI collectives.
    for field_idx in eachindex(spectral_fields)
        scalar_spectral_to_physical!(
            spectral_fields[field_idx],
            physical_fields[field_idx]
        )
    end
    return nothing
end

function transform_field_and_gradients_to_physical!(
        𝔽::ScalarFieldType{T},
        ws::SolverGradientWorkspace{T},
        domain::RadialDomainType
) where {T}
    main_physical_field = solver_main_physical_field(𝔽)
    scalar_spectral_to_physical!(𝔽.spectral, main_physical_field)
    # EXACT tangential gradient via the raw sphtor synthesis of the scalar's
    # own coefficients: S = 𝔽.spectral, T = 0 gives (∂θf, (1/sinθ)∂φf) on the
    # unit sphere; dividing by r yields the TRUE physical gradient components.
    # The legacy route synthesized the θ/φ spectral-recurrence outputs, which
    # are sinθ-weighted projections (the θ-recurrence encodes sinθ·∂θY) — that
    # silently sinθ-weighted the tangential advection u·∇f (suppressed toward
    # the poles). 𝔽.work_spectral is zero here (solver_zero_scalar_work_arrays!
    # ran at the start of the pass) and serves as the zero toroidal input.
    vector_spectral_to_physical!(
        𝔽.work_spectral,           # zero toroidal
        𝔽.spectral,                # spheroidal = the scalar itself
        𝔽.gradient;
        raw_spheroidal = true
    )
    _scale_tangential_gradient_by_inv_r!(𝔽, domain)
    scalar_spectral_to_physical!(ws.∇r_spec, 𝔽.gradient.r_component)
    return 𝔽
end

# gradient.θ/φ hold the unit-sphere tangential derivatives after the sphtor
# synthesis; divide by r per radial level for the physical gradient.
function _scale_tangential_gradient_by_inv_r!(𝔽, domain)
    gθ = parent(𝔽.gradient.θ_component.data)
    gφ = parent(𝔽.gradient.φ_component.data)
    cfg = 𝔽.config
    r_range = local_range(cfg.pencils.r, 3)
    @inbounds for k in axes(gθ, 3)
        r_idx = k + first(r_range) - 1
        rinv = (1 <= r_idx <= domain.N) ? domain.r[r_idx, 3] : 0.0
        for j in axes(gθ, 2), i in axes(gθ, 1)
            gθ[i, j, k] *= rinv
            gφ[i, j, k] *= rinv
        end
    end
    return 𝔽
end

function solver_zero_scalar_work_arrays!(𝔽::ScalarFieldType{T}) where {T}
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
        vel_fields
) where {T}
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
        domain::RadialDomainType
) where {T}
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

function scalar_nonlinear_to_spectral!(
        phys::PhysicalFieldType{T},
        spec::SpectralFieldType{T}
) where {T}
    # geometry-blind: the ball grid has no r=0 node (off-center grid);
    # regularity at r=0 lives in the implicit-matrix Robin rows, not here.
    return scalar_physical_to_spectral!(phys, spec)
end

function solver_compute_velocity_nonlinear!(
        velocity_fields::VelocityFieldsType{T},
        temperature_field,
        composition_field,
        magnetic_field,
        domain::RadialDomainType;
        params::Union{Nothing, SolverParameters} = nothing
) where {T}
    solver_params = isnothing(params) ? create_solver_parameters() : params
    prepare_velocity_fields!(velocity_fields, domain)
    accumulate_velocity_nonlinear_terms!(
        velocity_fields,
        temperature_field,
        composition_field,
        magnetic_field,
        domain,
        solver_params
    )
    finish_velocity_nonlinear!(velocity_fields)
    return velocity_fields
end

function solver_compute_magnetic_nonlinear!(
        magnetic_fields::MagneticFieldsType{T},
        velocity_fields,
        outer_domain::RadialDomainType,
        inner_domain::RadialDomainType,
        rotation_rate::Float64 = 0.0
) where {T}
    prepare_magnetic_fields!(magnetic_fields, outer_domain)
    apply_magnetic_nonlinear_terms!(
        magnetic_fields,
        velocity_fields;
        rotation_rate
    )
    return magnetic_fields
end

# NOTE: `geometry` is accepted for API back-compat only — the unified scalar
# nonlinear path is geometry-blind (off-center ball grid; no r=0 node).
function GeoDynamo.compute_temperature_nonlinear!(
        temp_𝔽::TemperatureFieldType{T},
        vel_fields,
        outer_core_domain::RadialDomainType,
        ws::SolverGradientWorkspace{T};
        geometry::Symbol = solver_default_geometry()
) where {T}
    return solver_compute_temperature_nonlinear!(
        temp_𝔽,
        vel_fields,
        outer_core_domain,
        ws
    )
end

# NOTE: `geometry` is accepted for API back-compat only — see above.
function GeoDynamo.compute_composition_nonlinear!(
        𝔽::CompositionFieldType{T},
        vel_fields,
        outer_core_domain::RadialDomainType,
        ws::SolverGradientWorkspace{T};
        geometry::Symbol = solver_default_geometry()
) where {T}
    return solver_compute_composition_nonlinear!(
        𝔽,
        vel_fields,
        outer_core_domain,
        ws
    )
end

function compute_solver_nonlinear_terms!(state::SolverState)
    # Buoyancy reads the scalar PHYSICAL fields during the velocity force
    # assembly below. Refresh them from the current spectral state FIRST —
    # historically they were only populated by each scalar's own nonlinear
    # pass (which runs AFTER velocity), so buoyancy saw zeros on the first
    # step and a one-step-lagged temperature afterwards (an O(dt)
    # inconsistency). One extra scalar synthesis per field per step;
    # the scalar passes later re-do it (acceptable; optimization noted).
    scalar_spectral_to_physical!(
        state.fields.temperature.spectral,
        solver_main_physical_field(state.fields.temperature))
    if state.fields.composition !== nothing
        scalar_spectral_to_physical!(
            state.fields.composition.spectral,
            solver_main_physical_field(state.fields.composition))
    end

    # Velocity nonlinear terms define the advecting flow shared by every other
    # subsystem, so that path runs first.
    solver_compute_velocity_nonlinear!(
        state.fields.velocity,
        state.fields.temperature,
        state.fields.composition,
        state.fields.magnetic,
        state.backend.outer_core_domain,
        params = state.parameters
    )

    if state.parameters.include_magnetic && state.fields.magnetic !== nothing
        # Ball runs reuse the outer-core domain in both slots; shell runs carry
        # a distinct inner-core domain that the magnetic coupling needs.
        inner_domain = isnothing(state.backend.inner_core_domain) ?
                       state.backend.outer_core_domain : state.backend.inner_core_domain
        solver_compute_magnetic_nonlinear!(
            state.fields.magnetic,
            state.fields.velocity,
            state.backend.outer_core_domain,
            inner_domain
        )
    end

    # Scalar nonlinear terms share one gradient workspace, so they run after the
    # vector paths have finished with the current state.
    solver_compute_temperature_nonlinear!(
        state.fields.temperature,
        state.fields.velocity,
        state.backend.outer_core_domain,
        state.runtime.gradient_workspace
    )

    if state.fields.composition !== nothing
        solver_compute_composition_nonlinear!(
            state.fields.composition,
            state.fields.velocity,
            state.backend.outer_core_domain,
            state.runtime.gradient_workspace
        )
    end

    return state
end
