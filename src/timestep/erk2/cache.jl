# ERK2 cache lifecycle: cache builders, memoized accessors, and bundle persistence.

"""
    create_solver_erk2_scalar_cache(T, config, domain, diffusivity, dt, boundary_condition; ...)

Precompute ERK2 propagators for scalar fields with embedded boundary rows.

The cache stores one set of matrices per unique spherical-harmonic degree in
`config`. Dense matrices are precomputed unless `use_krylov=true`, in which
case operator matrices are stored for Krylov actions.
"""
function create_solver_erk2_scalar_cache(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64,
        boundary_condition::Int;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    laplacian = build_radial_laplacian(domain)
    nr = domain.N
    bandwidth = laplacian.bandwidth
    r_inv_sq = @views domain.r[1:nr, 2]
    l_values = unique(config.l_values)

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    bc_desc = ["DD", "DN", "ND", "NN"][clamp(boundary_condition, 1, 4)]
    if mpi_rank() == 0
        @info "Creating solver ERK2 scalar cache (type=$bc_desc, ν=$diffusivity)"
    end

    for l in l_values
        operator_data = diffusivity .* laplacian.data
        operator_dense = solver_banded_to_dense(BandedOperator(operator_data, bandwidth, nr))
        l_factor = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            operator_dense[n, n] -= diffusivity * l_factor * r_inv_sq[n]
        end

        operator_dense[1, :] .= zero(T)
        operator_dense[nr, :] .= zero(T)

        if use_krylov
            push!(E_half, operator_dense)
            push!(E_full, operator_dense)
            push!(phi1_half, operator_dense)
            push!(phi1_full, operator_dense)
            push!(phi2_full, operator_dense)
        else
            operator_half = (dt / 2) .* operator_dense
            operator_full = dt .* operator_dense

            E_half_l = exp(operator_half)
            E_full_l = exp(operator_full)
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))

            phi1_half_l = solver_compute_phi1_function(operator_half, E_half_l)
            phi1_full_l = solver_compute_phi1_function(operator_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            phi2_full_l = solver_compute_phi2_function(operator_full, E_full_l; l = l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
        end
    end

    mpi_barrier!()

    return ERK2StageCache{T}(
        dt,
        diffusivity,
        nr,
        l_values,
        E_half,
        E_full,
        phi1_half,
        phi1_full,
        phi2_full,
        use_krylov,
        m,
        tol,
        true
    )
end

"""
    create_solver_erk2_cache(T, config, domain, diffusivity, dt; bc_spec=nothing, ...)

Precompute generic ERK2 propagators for velocity-like spectral fields.
"""
function create_solver_erk2_cache(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8,
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        dpol_operator::Bool = false
) where {T}
    # dpol_operator: build on D_pol = d²/dr² − l(l+1)/r² (poloidal potentials
    # under the Stage-2 solenoidal convention) instead of the full scalar
    # Laplacian (Stage-4B ERK2 W-split port).
    laplacian = dpol_operator ? create_derivative_matrix(Float64, 2, domain) :
                build_radial_laplacian(domain)
    nr = domain.N
    r_inv_sq = @views domain.r[1:nr, 2]
    l_values = unique(config.l_values)

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    if mpi_rank() == 0
        method_name = use_krylov ? "Krylov" : "dense"
        @info "Creating solver ERK2 cache for $(length(l_values)) l-modes with $method_name methods"
    end

    for l in l_values
        operator_data = diffusivity .* laplacian.data
        operator_dense = solver_banded_to_dense(
            BandedOperator(operator_data, laplacian.bandwidth, nr),
        )
        l_factor = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            operator_dense[n, n] -= diffusivity * l_factor * r_inv_sq[n]
        end

        if l == 0
            operator_dense[1, :] .= zero(T)
            operator_dense[nr, :] .= zero(T)
        end

        if use_krylov
            push!(E_half, operator_dense)
            push!(E_full, operator_dense)
            push!(phi1_half, operator_dense)
            push!(phi1_full, operator_dense)
            push!(phi2_full, operator_dense)
        else
            operator_half = (dt / 2) .* operator_dense
            operator_full = dt .* operator_dense

            E_half_l = exp(operator_half)
            E_full_l = exp(operator_full)
            if !all(isfinite, E_half_l) || !all(isfinite, E_full_l)
                @error "Non-finite solver ERK2 matrix exponential for l=$l (dt=$dt, ||A||=$(opnorm(operator_dense)))"
            end
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))

            phi1_half_l = solver_compute_phi1_function(operator_half, E_half_l)
            phi1_full_l = solver_compute_phi1_function(operator_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            phi2_full_l = solver_compute_phi2_function(operator_full, E_full_l; l = l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
        end
    end

    mpi_barrier!()

    return ERK2StageCache{T}(
        dt,
        diffusivity,
        nr,
        l_values,
        E_half,
        E_full,
        phi1_half,
        phi1_full,
        phi2_full,
        use_krylov,
        m,
        tol,
        true
    )
end

"""
    create_solver_erk2_magnetic_toroidal_cache(T, config, domain, diffusivity, dt; ...)

Precompute ERK2 propagators for magnetic toroidal fields with embedded
homogeneous Dirichlet boundary rows.
"""
function create_solver_erk2_magnetic_toroidal_cache(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    laplacian = build_radial_laplacian(domain)
    nr = domain.N
    bandwidth = laplacian.bandwidth
    r_inv_sq = @views domain.r[1:nr, 2]
    l_values = unique(config.l_values)

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    if mpi_rank() == 0
        @info "Creating solver ERK2 cache for magnetic toroidal with embedded Dirichlet BCs"
    end

    for l in l_values
        operator_data = diffusivity .* laplacian.data
        operator_dense = solver_banded_to_dense(BandedOperator(operator_data, bandwidth, nr))
        l_factor = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            operator_dense[n, n] -= diffusivity * l_factor * r_inv_sq[n]
        end

        operator_dense[1, :] .= zero(T)
        operator_dense[nr, :] .= zero(T)

        if use_krylov
            push!(E_half, operator_dense)
            push!(E_full, operator_dense)
            push!(phi1_half, operator_dense)
            push!(phi1_full, operator_dense)
            push!(phi2_full, operator_dense)
        else
            operator_half = (dt / 2) .* operator_dense
            operator_full = dt .* operator_dense

            E_half_l = exp(operator_half)
            E_full_l = exp(operator_full)
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))

            phi1_half_l = solver_compute_phi1_function(operator_half, E_half_l)
            phi1_full_l = solver_compute_phi1_function(operator_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            phi2_full_l = solver_compute_phi2_function(operator_full, E_full_l; l = l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
        end
    end

    mpi_barrier!()

    return ERK2StageCache{T}(
        dt,
        diffusivity,
        nr,
        l_values,
        E_half,
        E_full,
        phi1_half,
        phi1_full,
        phi2_full,
        use_krylov,
        m,
        tol,
        true
    )
end

"""
    create_solver_erk2_magnetic_poloidal_cache(T, config, domain, diffusivity, dt; ...)

Precompute ERK2 propagators for magnetic poloidal fields with embedded
insulating boundary rows.
"""
function create_solver_erk2_magnetic_poloidal_cache(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    laplacian = build_radial_laplacian(domain)
    first_derivative = build_radial_derivative_matrix(T, 1, domain)
    nr = domain.N
    bandwidth = laplacian.bandwidth
    r_inv_sq = @views domain.r[1:nr, 2]
    r_inv = @views domain.r[1:nr, 3]
    l_values = unique(config.l_values)

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    if mpi_rank() == 0
        @info "Creating solver ERK2 cache for magnetic poloidal with embedded insulating BCs"
    end

    for l in l_values
        operator_data = diffusivity .* laplacian.data
        operator_dense = solver_banded_to_dense(BandedOperator(operator_data, bandwidth, nr))
        l_factor = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            operator_dense[n, n] -= diffusivity * l_factor * r_inv_sq[n]
        end

        operator_dense[1, :] .= zero(T)
        operator_dense[nr, :] .= zero(T)

        for j in max(1, 1 - bandwidth):min(nr, 1 + bandwidth)
            band_idx = bandwidth + 1 + 1 - j
            if 1 <= band_idx <= 2 * bandwidth + 1
                operator_dense[1, j] = T(first_derivative.data[band_idx, j])
            end
        end
        operator_dense[1, 1] -= T(l) * r_inv[1]

        for j in max(1, nr - bandwidth):min(nr, nr + bandwidth)
            band_idx = bandwidth + 1 + nr - j
            if 1 <= band_idx <= 2 * bandwidth + 1
                operator_dense[nr, j] = T(first_derivative.data[band_idx, j])
            end
        end
        operator_dense[nr, nr] += T(l + 1) * r_inv[nr]

        if use_krylov
            push!(E_half, operator_dense)
            push!(E_full, operator_dense)
            push!(phi1_half, operator_dense)
            push!(phi1_full, operator_dense)
            push!(phi2_full, operator_dense)
        else
            operator_half = (dt / 2) .* operator_dense
            operator_full = dt .* operator_dense

            E_half_l = exp(operator_half)
            E_full_l = exp(operator_full)
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))

            phi1_half_l = solver_compute_phi1_function(operator_half, E_half_l)
            phi1_full_l = solver_compute_phi1_function(operator_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            phi2_full_l = solver_compute_phi2_function(operator_full, E_full_l; l = l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
        end
    end

    mpi_barrier!()

    return ERK2StageCache{T}(
        dt,
        diffusivity,
        nr,
        l_values,
        E_half,
        E_full,
        phi1_half,
        phi1_full,
        phi2_full,
        use_krylov,
        m,
        tol,
        true
    )
end

"""
    GeoDynamo.create_erk2_cache(T, config, domain, diffusivity, dt; ...)

Public wrapper for constructing a generic ERK2 stage cache.
"""
function GeoDynamo.create_erk2_cache(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8,
        bc_spec::Union{Nothing, GeoDynamo.ERK2BoundarySpec{T}} = nothing
) where {T}
    return create_solver_erk2_cache(
        T,
        config,
        domain,
        diffusivity,
        dt;
        use_krylov,
        m,
        tol,
        bc_spec
    )
end

"""
    GeoDynamo.create_erk2_cache_scalar(T, config, domain, diffusivity, dt, boundary_condition; ...)

Public wrapper for constructing scalar-field ERK2 caches with embedded
boundary conditions.
"""
function GeoDynamo.create_erk2_cache_scalar(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64,
        boundary_condition::Int;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    return create_solver_erk2_scalar_cache(
        T,
        config,
        domain,
        diffusivity,
        dt,
        boundary_condition;
        use_krylov,
        m,
        tol
    )
end

"""
    GeoDynamo.create_erk2_cache_temperature(T, config, domain, diffusivity, dt, temperature_bcs; ...)

Create the ERK2 cache used by temperature fields.
"""
function GeoDynamo.create_erk2_cache_temperature(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64,
        temperature_bcs::BoundaryConditions;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    GeoDynamo.create_erk2_cache_scalar(
        T,
        config,
        domain,
        diffusivity,
        dt,
        _thermal_bc_code(temperature_bcs);
        use_krylov,
        m,
        tol
    )
end

"""
    GeoDynamo.create_erk2_cache_composition(T, config, domain, diffusivity, dt, composition_bcs; ...)

Create the ERK2 cache used by composition fields.
"""
function GeoDynamo.create_erk2_cache_composition(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64,
        composition_bcs::BoundaryConditions;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    GeoDynamo.create_erk2_cache_scalar(
        T,
        config,
        domain,
        diffusivity,
        dt,
        _composition_bc_code(composition_bcs);
        use_krylov,
        m,
        tol
    )
end

"""
    GeoDynamo.create_erk2_cache_magnetic_toroidal(T, config, domain, diffusivity, dt; ...)

Create the ERK2 cache used by magnetic toroidal fields.
"""
function GeoDynamo.create_erk2_cache_magnetic_toroidal(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    return create_solver_erk2_magnetic_toroidal_cache(
        T,
        config,
        domain,
        diffusivity,
        dt;
        use_krylov,
        m,
        tol
    )
end

"""
    GeoDynamo.create_erk2_cache_magnetic_poloidal(T, config, domain, diffusivity, dt; ...)

Create the ERK2 cache used by magnetic poloidal fields.
"""
function GeoDynamo.create_erk2_cache_magnetic_poloidal(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    return create_solver_erk2_magnetic_poloidal_cache(
        T,
        config,
        domain,
        diffusivity,
        dt;
        use_krylov,
        m,
        tol
    )
end

"""
    _get_or_build_erk2_cache(existing, label, diffusivity, T, config, domain, dt; ...)

Build or reuse an ERK2 stage cache for velocity-like fields.

Callers own the storage location; this helper only decides whether the existing
cache still matches the current grid, timestep, diffusivity, and method flags.
"""
function _get_or_build_erk2_cache(
        existing::Union{ERK2StageCache{T}, Nothing},
        label::AbstractString,
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8,
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        dpol_operator::Bool = false
)::ERK2StageCache{T} where {T}
    nr = domain.N
    needs_rebuild = existing === nothing ||
                    existing.diffusivity != diffusivity ||
                    existing.nr != nr ||
                    existing.dt != dt ||
                    existing.use_krylov != use_krylov ||
                    !existing.mpi_consistent ||
                    existing.l_values != unique(config.l_values)

    if needs_rebuild
        if mpi_rank() == 0
            @info "Creating solver $label ERK2 cache (ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        return create_solver_erk2_cache(
            T,
            config,
            domain,
            diffusivity,
            dt;
            use_krylov,
            m,
            tol,
            bc_spec,
            dpol_operator
        )
    end

    return existing::ERK2StageCache{T}
end

"""
    _get_or_build_erk2_scalar_cache(existing, label, diffusivity, T, config, domain, dt, boundary_condition; ...)

Build or reuse an ERK2 stage cache for scalar fields.
"""
function _get_or_build_erk2_scalar_cache(
        existing::Union{ERK2StageCache{T}, Nothing},
        label::AbstractString,
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64,
        boundary_condition::Int;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
)::ERK2StageCache{T} where {T}
    nr = domain.N
    needs_rebuild = existing === nothing ||
                    existing.diffusivity != diffusivity ||
                    existing.nr != nr ||
                    existing.dt != dt ||
                    existing.use_krylov != use_krylov ||
                    !existing.mpi_consistent ||
                    existing.l_values != unique(config.l_values)

    if needs_rebuild
        bc_desc = ["DD", "DN", "ND", "NN"][clamp(boundary_condition, 1, 4)]
        if mpi_rank() == 0
            @info "Creating solver $label ERK2 cache (type=$bc_desc, ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        return create_solver_erk2_scalar_cache(
            T,
            config,
            domain,
            diffusivity,
            dt,
            boundary_condition;
            use_krylov,
            m,
            tol
        )
    end

    return existing::ERK2StageCache{T}
end

"""
    get_solver_erk2_temperature_cache!(caches, diffusivity, T, config, domain, dt, temperature_bc_code; ...)

Return the solver-owned temperature ERK2 cache, rebuilding it when timestep,
grid, diffusivity, Krylov settings, or boundary conditions no longer match.
"""
function get_solver_erk2_temperature_cache!(
        caches::TimestepCaches{T},
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64,
        temperature_bc_code::Int;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    caches.erk2_temperature = _get_or_build_erk2_scalar_cache(
        caches.erk2_temperature,
        "temperature",
        diffusivity,
        T,
        config,
        domain,
        dt,
        temperature_bc_code;
        use_krylov = use_krylov,
        m = m,
        tol = tol
    )
    return caches.erk2_temperature::ERK2StageCache{T}
end

"""
    get_solver_erk2_composition_cache!(caches, diffusivity, T, config, domain, dt, composition_bc_code; ...)

Return the solver-owned composition ERK2 cache with the same compatibility
checks used for the temperature scalar cache.
"""
function get_solver_erk2_composition_cache!(
        caches::TimestepCaches{T},
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64,
        composition_bc_code::Int;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    caches.erk2_composition = _get_or_build_erk2_scalar_cache(
        caches.erk2_composition,
        "composition",
        diffusivity,
        T,
        config,
        domain,
        dt,
        composition_bc_code;
        use_krylov = use_krylov,
        m = m,
        tol = tol
    )
    return caches.erk2_composition::ERK2StageCache{T}
end

"""
    get_solver_erk2_cache!(caches, Val(:velocity_toroidal), diffusivity, T, config, domain, dt; ...)

Return the velocity-toroidal ERK2 cache from `TimestepCaches`.

This concrete overload avoids runtime `Symbol` dispatch in the main solver
step while still sharing the generic rebuild checks.
"""
function get_solver_erk2_cache!(
        caches::TimestepCaches{T},
        ::Val{:velocity_toroidal},
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8,
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing
) where {T}
    caches.erk2_velocity_toroidal = _get_or_build_erk2_cache(
        caches.erk2_velocity_toroidal,
        "velocity_toroidal",
        diffusivity,
        T,
        config,
        domain,
        dt;
        use_krylov = use_krylov,
        m = m,
        tol = tol,
        bc_spec = bc_spec
    )
    return caches.erk2_velocity_toroidal::ERK2StageCache{T}
end

"""
    get_solver_erk2_cache!(caches, Val(:velocity_poloidal), diffusivity, T, config, domain, dt; ...)

Return the velocity-poloidal ERK2 cache from `TimestepCaches`.
"""
function get_solver_erk2_cache!(
        caches::TimestepCaches{T},
        ::Val{:velocity_poloidal},
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8,
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        dpol_operator::Bool = false
) where {T}
    caches.erk2_velocity_poloidal = _get_or_build_erk2_cache(
        caches.erk2_velocity_poloidal,
        "velocity_poloidal",
        diffusivity,
        T,
        config,
        domain,
        dt;
        use_krylov = use_krylov,
        m = m,
        tol = tol,
        bc_spec = bc_spec,
        dpol_operator = dpol_operator
    )
    return caches.erk2_velocity_poloidal::ERK2StageCache{T}
end

"""
    get_solver_erk2_cache!(caches, key, diffusivity, T, config, domain, dt; ...)

Compatibility shim that dispatches legacy `Symbol` keys to the concrete
velocity cache overloads.
"""
function get_solver_erk2_cache!(
        caches::TimestepCaches{T},
        key::Symbol,
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8,
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        dpol_operator::Bool = false
) where {T}
    if key === :velocity_toroidal
        return get_solver_erk2_cache!(
            caches, Val(:velocity_toroidal), diffusivity, T, config, domain, dt;
            use_krylov = use_krylov, m = m, tol = tol, bc_spec = bc_spec
        )
    elseif key === :velocity_poloidal
        return get_solver_erk2_cache!(
            caches, Val(:velocity_poloidal), diffusivity, T, config, domain, dt;
            use_krylov = use_krylov, m = m, tol = tol, bc_spec = bc_spec,
            dpol_operator = dpol_operator
        )
    else
        error("get_solver_erk2_cache!: unsupported key $key for TimestepCaches")
    end
end

"""
    get_solver_erk2_magnetic_toroidal_cache!(caches, diffusivity, T, config, domain, dt; ...)

Return or rebuild the magnetic-toroidal ERK2 cache stored in `TimestepCaches`.
"""
function get_solver_erk2_magnetic_toroidal_cache!(
        caches::TimestepCaches{T},
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    nr = domain.N
    existing = caches.erk2_magnetic_toroidal
    needs_rebuild = existing === nothing ||
                    existing.diffusivity != diffusivity ||
                    existing.nr != nr ||
                    existing.dt != dt ||
                    existing.use_krylov != use_krylov ||
                    !existing.mpi_consistent ||
                    existing.l_values != unique(config.l_values)

    if needs_rebuild
        if mpi_rank() == 0
            @info "Creating solver magnetic toroidal ERK2 cache (ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        caches.erk2_magnetic_toroidal = create_solver_erk2_magnetic_toroidal_cache(
            T,
            config,
            domain,
            diffusivity,
            dt;
            use_krylov,
            m,
            tol
        )
    end

    return caches.erk2_magnetic_toroidal::ERK2StageCache{T}
end

"""
    get_solver_erk2_magnetic_poloidal_cache!(caches, diffusivity, T, config, domain, dt; ...)

Return or rebuild the magnetic-poloidal ERK2 cache stored in `TimestepCaches`.
"""
function get_solver_erk2_magnetic_poloidal_cache!(
        caches::TimestepCaches{T},
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    nr = domain.N
    existing = caches.erk2_magnetic_poloidal
    needs_rebuild = existing === nothing ||
                    existing.diffusivity != diffusivity ||
                    existing.nr != nr ||
                    existing.dt != dt ||
                    existing.use_krylov != use_krylov ||
                    !existing.mpi_consistent ||
                    existing.l_values != unique(config.l_values)

    if needs_rebuild
        if mpi_rank() == 0
            @info "Creating solver magnetic poloidal ERK2 cache (ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        caches.erk2_magnetic_poloidal = create_solver_erk2_magnetic_poloidal_cache(
            T,
            config,
            domain,
            diffusivity,
            dt;
            use_krylov,
            m,
            tol
        )
    end

    return caches.erk2_magnetic_poloidal::ERK2StageCache{T}
end

"""
    GeoDynamo.save_erk2_cache_bundle(path, caches; metadata=Dict())

Persist compatible ERK2 stage caches and metadata to a JLD2 file.
"""
function GeoDynamo.save_erk2_cache_bundle(
        path::AbstractString,
        caches::AbstractDict{Symbol, <:Any};
        metadata::Dict{String, Any} = Dict{String, Any}()
)
    bundle = Dict{Symbol, Any}()
    for (key, value) in caches
        cache = compat_normalize_old_erk2_cache_entry(value)
        cache === nothing && continue
        bundle[key] = cache
    end

    meta = Dict{String, Any}(metadata)
    meta["created_at"] = get(meta, "created_at", string(GeoDynamo.now()))
    GeoDynamo.jldopen(path, "w") do file
        file["caches"] = bundle
        file["metadata"] = meta
    end
    return path
end

"""
    GeoDynamo.load_erk2_cache_bundle(path)

Load ERK2 cache bundle data and metadata from a JLD2 file.
"""
function GeoDynamo.load_erk2_cache_bundle(path::AbstractString)
    caches = Dict{Symbol, Any}()
    metadata = Dict{String, Any}()
    GeoDynamo.jldopen(path, "r") do file
        caches = Dict{Symbol, Any}(file["caches"])
        metadata = haskey(file, "metadata") ? Dict{String, Any}(file["metadata"]) :
                   Dict{String, Any}()
    end
    return caches, metadata
end

"""
    GeoDynamo.install_erk2_cache_bundle!(target, bundle)

Install cache entries from a loaded bundle into a target cache dictionary.
"""
function GeoDynamo.install_erk2_cache_bundle!(
        target::Dict{Symbol, Any},
        bundle::AbstractDict{Symbol, <:Any}
)
    for (key, value) in bundle
        cache = compat_normalize_old_erk2_cache_entry(value)
        cache === nothing && continue
        target[key] = cache
    end
    return target
end

"""
    GeoDynamo.install_erk2_cache_bundle!(target::Dict{Symbol, ERK2StageCache{T}}, bundle)

Typed cache-bundle installer used by solver-local cache dictionaries.
"""
function GeoDynamo.install_erk2_cache_bundle!(
        target::Dict{Symbol, ERK2StageCache{T}},
        bundle::AbstractDict{Symbol, <:Any}
) where {T}
    for (key, value) in bundle
        cache = compat_normalize_old_erk2_cache_entry(value)
        cache === nothing && continue
        target[key] = compat_solver_erk2_cache(cache)
    end
    return target
end

"""
    GeoDynamo.load_erk2_cache_bundle!(target, path)

Load a cache bundle from disk, install it into `target`, and return metadata.
"""
function GeoDynamo.load_erk2_cache_bundle!(
        target::Union{
            Dict{Symbol, Any},
            Dict{Symbol, ERK2StageCache{T}}
        },
        path::AbstractString
) where {T}
    bundle, metadata = GeoDynamo.load_erk2_cache_bundle(path)
    GeoDynamo.install_erk2_cache_bundle!(target, bundle)
    return metadata
end
