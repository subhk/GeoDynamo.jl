# ERK2 influence matrices: poloidal-velocity boundary-condition correction.

"""
    _get_or_build_erk2_influence_entry(existing, T, config, domain, diffusivity, dt, velocity_bc_code; theta)

Build or reuse the velocity-poloidal influence correction cache.

The cache key includes a hash of the radial grid so boundary-correction
operators are refreshed when the domain geometry changes.
"""
function _get_or_build_erk2_influence_entry(
        existing::Union{ERK2InfluenceCacheEntry{T}, Nothing},
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64,
        velocity_bc_code::Int;
        theta::Float64
)::ERK2InfluenceCacheEntry{T} where {T}
    domain_hash = hash(domain.r)
    needs_refresh = existing === nothing ||
                    existing.diffusivity != diffusivity ||
                    existing.dt != dt ||
                    existing.theta != theta ||
                    existing.velocity_bc_code != velocity_bc_code ||
                    existing.lmax != config.lmax ||
                    existing.mmax != config.mmax ||
                    existing.nlat != config.nlat ||
                    existing.nlon != config.nlon ||
                    existing.nr != domain.N ||
                    existing.domain_hash != domain_hash

    if needs_refresh
        matrices = create_solver_velocity_poloidal_influence_matrices(
            T,
            config,
            domain,
            diffusivity,
            dt,
            velocity_bc_code;
            theta = theta
        )
        return ERK2InfluenceCacheEntry{T}(
            matrices,
            diffusivity,
            dt,
            theta,
            velocity_bc_code,
            config.lmax,
            config.mmax,
            config.nlat,
            config.nlon,
            domain.N,
            domain_hash
        )
    end

    return existing::ERK2InfluenceCacheEntry{T}
end

"""
    create_solver_velocity_poloidal_influence_matrices(T, config, domain, diffusivity, dt, velocity_bc_code; theta)

Build Green-function influence operators used to correct velocity-poloidal
endpoint constraints after ERK2 finalization.
"""
function create_solver_velocity_poloidal_influence_matrices(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64,
        velocity_bc_code::Int;
        theta::Float64
) where {T}
    l_values = unique(config.l_values)
    nr = domain.N

    first_derivative = build_radial_derivative_matrix(T, 1, domain)
    second_derivative = build_radial_derivative_matrix(T, 2, domain)
    bandwidth = second_derivative.bandwidth
    r_inv_sq = @views domain.r[1:nr, 2]

    # Poloidal diffusion operator D_pol = ∂² − l(l+1)/r² (bare ∂², NO 2/r term):
    # the influence Green functions MUST use the SAME operator as the paired main
    # solve (create_velocity_poloidal_matrices, also D_pol) for the correction to
    # cancel the boundary residuals. The −l(l+1)/r² shift is added per-l below.
    base_data = T.(diffusivity .* second_derivative.data)
    influence_matrices = Dict{Int, ERK2InfluenceOp{T}}()

    for l in l_values
        l == 0 && continue

        operator_data = copy(base_data)
        l_factor = Float64(l * (l + 1))
        @inbounds for n in 1:nr
            operator_data[bandwidth + 1, n] -= T(diffusivity * l_factor * r_inv_sq[n])
        end

        inv_dt = T(1 / dt)
        θ = T(theta)

        green_system_data = copy(operator_data)
        green_system_data .*= -θ
        green_system_data[bandwidth + 1, :] .+= inv_dt

        @inbounds for j in 1:(1 + bandwidth)
            green_system_data[bandwidth + 1 + 1 - j, j] = zero(T)
        end
        @inbounds for j in (nr - bandwidth):nr
            green_system_data[bandwidth + 1 + nr - j, j] = zero(T)
        end

        green_system_data[bandwidth + 1, 1] = one(T)
        green_system_data[bandwidth + 1, nr] = one(T)

        green_system = BandedOperator{T}(green_system_data, bandwidth, nr)
        green_lu = solver_factorize_banded(green_system)

        physical_system_data = copy(operator_data)
        physical_system_data .*= -θ
        physical_system_data[bandwidth + 1, :] .+= inv_dt

        @inbounds for j in 1:(1 + bandwidth)
            physical_system_data[bandwidth + 1 + 1 - j, j] = zero(T)
        end
        @inbounds for j in (nr - bandwidth):nr
            physical_system_data[bandwidth + 1 + nr - j, j] = zero(T)
        end

        if velocity_bc_code == 1 || velocity_bc_code == 2
            @inbounds for j in 1:(1 + bandwidth)
                physical_system_data[bandwidth + 1 + 1 - j, j] = first_derivative.data[bandwidth + 1 + 1 - j, j]
            end
        else
            # Stress-free inner: P″ − (2/r)P′ = 0 (no-tangential-stress), matching
            # the main solve; the bare P″ row missed the −2P′/r metric term.
            r2_in = T(2 / domain.r[1, 4])
            @inbounds for j in 1:(1 + bandwidth)
                physical_system_data[bandwidth + 1 + 1 - j, j] =
                    second_derivative.data[bandwidth + 1 + 1 - j, j] -
                    r2_in * first_derivative.data[bandwidth + 1 + 1 - j, j]
            end
        end

        if velocity_bc_code == 1 || velocity_bc_code == 3
            @inbounds for j in (nr - bandwidth):nr
                physical_system_data[bandwidth + 1 + nr - j, j] = first_derivative.data[bandwidth + 1 + nr - j, j]
            end
        else
            # Stress-free outer: P″ − (2/r)P′ = 0.
            r2_out = T(2 / domain.r[nr, 4])
            @inbounds for j in (nr - bandwidth):nr
                physical_system_data[bandwidth + 1 + nr - j, j] =
                    second_derivative.data[bandwidth + 1 + nr - j, j] -
                    r2_out * first_derivative.data[bandwidth + 1 + nr - j, j]
            end
        end

        physical_system = BandedOperator{T}(physical_system_data, bandwidth, nr)
        physical_lu = solver_factorize_banded(physical_system)

        Gre = zeros(T, nr, 2)
        rhs = zeros(T, nr)

        rhs[1] = one(T)
        solve_banded!(rhs, green_lu, rhs)
        rhs[1] = zero(T)
        rhs[nr] = zero(T)
        solve_banded!(rhs, physical_lu, rhs)
        Gre[:, 1] = rhs

        fill!(rhs, zero(T))
        rhs[nr] = one(T)
        solve_banded!(rhs, green_lu, rhs)
        rhs[1] = zero(T)
        rhs[nr] = zero(T)
        solve_banded!(rhs, physical_lu, rhs)
        Gre[:, 2] = rhs

        invG = zeros(T, 2, 2)
        invG[1, 1] = Gre[1, 1]
        invG[1, 2] = Gre[1, 2]
        invG[2, 1] = Gre[nr, 1]
        invG[2, 2] = Gre[nr, 2]

        det = invG[1, 1] * invG[2, 2] - invG[1, 2] * invG[2, 1]
        max_elem = max(abs(invG[1, 1]), abs(invG[2, 2]), abs(invG[1, 2]), abs(invG[2, 1]))
        relative_det = max_elem > zero(T) ? abs(det) / (max_elem^2) : abs(det)
        if relative_det > pivot_tol(T) && abs(det) > zero(T)
            inv_det = one(T) / det
            if !isfinite(inv_det)
                @error "Solver ERK2 influence matrix inversion overflow for l=$l (det=$det). Zeroing correction matrix."
                invG .= zero(T)
                influence_matrices[l] = ERK2InfluenceOp{T}(Gre, invG, l)
                continue
            end
            tmp = invG[1, 1]
            invG[1, 1] = invG[2, 2] * inv_det
            invG[2, 2] = tmp * inv_det
            invG[1, 2] = -invG[1, 2] * inv_det
            invG[2, 1] = -invG[2, 1] * inv_det
        else
            @error "Solver ERK2 influence matrix is near-singular for l=$l (det=$det, relative_det=$relative_det, max_elem=$max_elem)."
            invG .= zero(T)
        end

        influence_matrices[l] = ERK2InfluenceOp{T}(Gre, invG, l)
    end

    return influence_matrices
end

"""
    get_solver_erk2_influence_matrices!(cache, key, T, config, domain, diffusivity, dt, velocity_bc_code; theta)

Return velocity-poloidal influence matrices from `TimestepCaches`.

Only `:velocity_poloidal` is valid here because the correction enforces the
poloidal boundary constraints after the ERK2 field update.
"""
function get_solver_erk2_influence_matrices!(
        cache::TimestepCaches{T},
        key::Symbol,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64,
        velocity_bc_code::Int;
        theta::Float64
) where {T}
    key === :velocity_poloidal || throw(ArgumentError(
        "get_solver_erk2_influence_matrices!: only :velocity_poloidal key supported for TimestepCaches; got $key"
    ))
    cache.erk2_influence_velocity_poloidal = _get_or_build_erk2_influence_entry(
        cache.erk2_influence_velocity_poloidal,
        T,
        config,
        domain,
        diffusivity,
        dt,
        velocity_bc_code;
        theta = theta
    )
    return (cache.erk2_influence_velocity_poloidal::ERK2InfluenceCacheEntry{T}).matrices
end

"""
    apply_solver_influence_matrix_correction!(result, influence, bc_inner_val=0, bc_outer_val=0)

Project one dense radial profile back onto the requested endpoint constraints
using a precomputed two-column influence operator.
"""
function apply_solver_influence_matrix_correction!(
        result::AbstractVector{T},
        influence::ERK2InfluenceOp{T},
        bc_inner_val::T = zero(T),
        bc_outer_val::T = zero(T)
) where {T}
    nr = length(result)
    delta_inner = result[1] - bc_inner_val
    delta_outer = result[nr] - bc_outer_val

    c1 = influence.invG[1, 1] * delta_inner + influence.invG[1, 2] * delta_outer
    c2 = influence.invG[2, 1] * delta_inner + influence.invG[2, 2] * delta_outer

    @inbounds for i in 1:nr
        result[i] -= c1 * influence.Gre[i, 1] + c2 * influence.Gre[i, 2]
    end

    return result
end

"""
    apply_solver_velocity_poloidal_influence_correction!(field, influence_matrices, config)

Apply the velocity-poloidal endpoint influence correction to all local spectral
modes in a distributed field.
"""
function apply_solver_velocity_poloidal_influence_correction!(
        field::SpectralFieldType{T},
        influence_matrices::Dict{Int, ERK2InfluenceOp{T}},
        config::SHTnsConfigType;
        work::Union{Vector{T}, Nothing} = nothing
) where {T}
    u_real = parent(field.data_real)
    u_imag = parent(field.data_imag)
    lm_range = local_spectral_mode_indices(config)
    nr = size(u_real, 3)
    # Reuse a caller-provided scratch vector when correctly sized; only allocate
    # on the fallback path (e.g. external callers that pass no workspace).
    tmp = (work !== nothing && length(work) == nr) ? work : Vector{T}(undef, nr)

    @inbounds for lm_idx in lm_range
        l = config.l_values[lm_idx]
        l == 0 && continue
        !haskey(influence_matrices, l) && continue

        influence = influence_matrices[l]
        slot = local_spectral_storage_slot(config, lm_idx)

        for ir in 1:nr
            tmp[ir] = local_spectral_value(u_real, slot, ir)
        end
        apply_solver_influence_matrix_correction!(tmp, influence, zero(T), zero(T))
        for ir in 1:nr
            set_local_spectral_value!(u_real, slot, ir, tmp[ir])
        end

        for ir in 1:nr
            tmp[ir] = local_spectral_value(u_imag, slot, ir)
        end
        apply_solver_influence_matrix_correction!(tmp, influence, zero(T), zero(T))
        for ir in 1:nr
            set_local_spectral_value!(u_imag, slot, ir, tmp[ir])
        end
    end

    return field
end

"""
    GeoDynamo.create_velocity_poloidal_influence_matrices(T, config, domain, diffusivity, dt, velocity_bcs; theta=0.5)

Public wrapper for building velocity-poloidal influence correction operators.
"""
function GeoDynamo.create_velocity_poloidal_influence_matrices(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64,
        velocity_bcs::BoundaryConditions;
        theta::Float64 = 0.5
) where {T}
    create_solver_velocity_poloidal_influence_matrices(
        T,
        config,
        domain,
        diffusivity,
        dt,
        _velocity_bc_code(velocity_bcs);
        theta
    )
end

"""
    GeoDynamo.apply_influence_matrix_correction!(result, influence, bc_inner_val=0, bc_outer_val=0)

Public wrapper for applying a single radial-profile influence correction.
"""
function GeoDynamo.apply_influence_matrix_correction!(
        result::AbstractVector{T},
        influence::ERK2InfluenceOp{T},
        bc_inner_val::T = zero(T),
        bc_outer_val::T = zero(T)
) where {T}
    apply_solver_influence_matrix_correction!(
        result,
        influence,
        bc_inner_val,
        bc_outer_val
    )
end

"""
    GeoDynamo.apply_velocity_poloidal_influence_correction!(field, influence_matrices, config)

Public wrapper for applying influence corrections to velocity-poloidal fields.
"""
function GeoDynamo.apply_velocity_poloidal_influence_correction!(
        field::SpectralFieldType{T},
        influence_matrices::Dict{Int, ERK2InfluenceOp{T}},
        config::SHTnsConfigType
) where {T}
    apply_solver_velocity_poloidal_influence_correction!(
        field,
        influence_matrices,
        config
    )
end
