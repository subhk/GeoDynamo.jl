#
# ERK2 staged exponential timestep support
#
# Developer map:
# - Boundary descriptors encode endpoint constraints and how to enforce them on
#   dense radial profiles.
# - Cache constructors precompute per-degree linear propagators (`E`, `phi1`,
#   `phi2`) for each active field family.
# - Cache getters rebuild only when physical parameters, timestep, grid, or
#   Krylov settings change.
# - Field buffers hold the provisional stage, current nonlinear terms, and
#   finalization work arrays for one field.
# - `integrate_solver_erk2_step!` orchestrates the full solver step across all
#   active fields.
#
@eval GeoDynamo begin
    function enforce_erk2_bc! end
    function create_dirichlet_bc end
    function create_neumann_bc end
    function create_stress_free_tor_bc end
    function create_noslip_pol_bc end
    function create_stress_free_pol_bc end
    function create_insulating_inner_bc end
    function create_insulating_outer_bc end
    function set_erk2_diagnostics_interval! end
    function enable_erk2_diagnostics! end
    function disable_erk2_diagnostics! end
    function erk2_diagnostics_enabled end
    function erk2_diagnostics_interval end
    function create_erk2_cache end
    function create_erk2_cache_scalar end
    function create_erk2_cache_temperature end
    function create_erk2_cache_composition end
    function create_erk2_cache_magnetic_toroidal end
    function create_erk2_cache_magnetic_poloidal end
    function compute_phi1_function end
    function compute_phi2_function end
    function reset_phi2_monitor! end
    function report_phi2_conditioning end
    function erk2_prepare_field! end
    function erk2_apply_stage! end
    function erk2_store_stage_nonlinear! end
    function erk2_finalize_field! end
    function erk2_stage_residual_stats end
    function maybe_log_erk2_stage_residual! end
    function create_velocity_poloidal_influence_matrices end
    function apply_influence_matrix_correction! end
    function apply_velocity_poloidal_influence_correction! end
    function save_erk2_cache_bundle end
    function load_erk2_cache_bundle end
    function install_erk2_cache_bundle! end
    function load_erk2_cache_bundle! end
end

# `SolverERK2BoundarySide` and `SolverERK2BoundarySpec` are defined in
# `solver/state.jl` (next to the other timestep-cache types) so they can be
# stored in `TimestepCaches.erk2_boundary_specs`. The helpers that build and
# operate on them live here.

function with_boundary_mode_values(
        spec::SolverERK2BoundarySpec{T},
        inner_real::Union{Nothing, AbstractVector{T}},
        outer_real::Union{Nothing, AbstractVector{T}},
        inner_imag::Union{Nothing, AbstractVector{T}} = nothing,
        outer_imag::Union{Nothing, AbstractVector{T}} = nothing
) where {T}
    # Keep the same derivative stencils and BC types, but attach the actual
    # mode-indexed endpoint values used during stage/final enforcement.
    return SolverERK2BoundarySpec{T}(
        spec.inner,
        spec.outer,
        inner_real,
        outer_real,
        inner_imag,
        outer_imag
    )
end

const ERK2Cache = ERK2StageCache
const Phi2ConditioningMonitor = SolverPhi2ConditioningMonitor
const PHI2_MONITOR = SOLVER_PHI2_MONITOR
const ERK2BoundarySide = SolverERK2BoundarySide
const ERK2BoundarySpec = SolverERK2BoundarySpec
const ERK2InfluenceMatrix = ERK2InfluenceOp

"""
    GeoDynamo.ERK2Cache{T}(...)

Compatibility constructor for the public ERK2 cache type.

The solver now stores stage caches as `ERK2StageCache`; this constructor keeps
older call sites that instantiate `GeoDynamo.ERK2Cache` directly working.
"""
function GeoDynamo.ERK2Cache{T}(
        dt::Float64,
        l_values::Vector{Int},
        E_half::Vector{Matrix{T}},
        E_full::Vector{Matrix{T}},
        phi1_half::Vector{Matrix{T}},
        phi1_full::Vector{Matrix{T}},
        phi2_full::Vector{Matrix{T}},
        use_krylov::Bool,
        krylov_m::Int,
        krylov_tol::Float64,
        mpi_consistent::Bool
) where {T}
    nr = isempty(E_half) ? 0 : size(E_half[1], 1)
    return ERK2StageCache{T}(
        dt,
        NaN,
        nr,
        l_values,
        E_half,
        E_full,
        phi1_half,
        phi1_full,
        phi2_full,
        use_krylov,
        krylov_m,
        krylov_tol,
        mpi_consistent
    )
end

"""
    GeoDynamo.ERK2Cache(args...)

Legacy constructor that builds a `Float64` ERK2 cache when no element type is
specified explicitly.
"""
GeoDynamo.ERK2Cache(args...) = GeoDynamo.ERK2Cache{Float64}(args...)

@inline compat_solver_erk2_cache(cache::ERK2StageCache{T}) where {T} = cache
@inline compat_old_erk2_cache(cache::ERK2StageCache{T}) where {T} = cache

"""
    compat_normalize_old_erk2_cache_entry(entry)

Extract an `ERK2StageCache` from legacy cache bundle entries.

Older code sometimes stores cache entries directly and sometimes under a
`:cache` key; unsupported entries return `nothing`.
"""
function compat_normalize_old_erk2_cache_entry(entry)
    if entry isa ERK2StageCache
        return compat_old_erk2_cache(entry)
    elseif entry isa Dict
        return compat_normalize_old_erk2_cache_entry(get(entry, :cache, nothing))
    elseif entry === nothing
        return nothing
    else
        return nothing
    end
end

"""
    solver_enforce_erk2_bc!(result, bc_side, boundary_idx, l, nr; value_override=nothing)

Enforce one ERK2 endpoint boundary condition on a dense radial profile.

Dirichlet conditions assign the endpoint directly. Derivative-based conditions
solve the endpoint value from the stored stencil and optional `l`-dependent
correction while leaving interior values unchanged.
"""
function solver_enforce_erk2_bc!(
        result::AbstractVector{T},
        bc_side::SolverERK2BoundarySide{T},
        boundary_idx::Int,
        l::Int,
        nr::Int;
        value_override::Union{T, Nothing} = nothing
) where {T}
    b = boundary_idx
    effective_value = value_override !== nothing ? value_override : bc_side.value

    if bc_side.l0_dirichlet && l == 0
        result[b] = effective_value
        return result
    end

    if bc_side.type === :dirichlet
        result[b] = effective_value
        return result
    end

    self_coeff = bc_side.stencil[b] + bc_side.fixed_correction
    if bc_side.use_l_correction
        self_coeff += bc_side.l_sign * T(l) * bc_side.r_inv
    end

    off_diag_sum = zero(T)
    @inbounds for j in 1:nr
        if j != b
            off_diag_sum += bc_side.stencil[j] * result[j]
        end
    end

    if abs(self_coeff) > pivot_tol(T)
        result[b] = (effective_value - off_diag_sum) / self_coeff
    else
        result[b] = effective_value
    end

    return result
end

"""
    GeoDynamo.enforce_erk2_bc!(result, bc_side, boundary_idx, l, nr; value_override=nothing)

Public wrapper for enforcing one ERK2 boundary condition on a radial profile.
"""
function GeoDynamo.enforce_erk2_bc!(
        result::AbstractVector{T},
        bc_side::SolverERK2BoundarySide{T},
        boundary_idx::Int,
        l::Int,
        nr::Int;
        value_override::Union{T, Nothing} = nothing
) where {T}
    return solver_enforce_erk2_bc!(
        result,
        bc_side,
        boundary_idx,
        l,
        nr;
        value_override
    )
end

"""
    solver_create_dirichlet_bc(T, nr, value=zero(T))

Create a fixed-value endpoint descriptor for ERK2 radial profiles.
"""
function solver_create_dirichlet_bc(::Type{T}, nr::Int, value::T = zero(T)) where {T}
    stencil = zeros(T, nr)
    return SolverERK2BoundarySide{T}(
        :dirichlet,
        value,
        stencil,
        zero(T),
        zero(T),
        false,
        zero(T),
        false
    )
end

"""
    GeoDynamo.create_dirichlet_bc(T, nr, value=zero(T))

Create a public fixed-value ERK2 endpoint descriptor.
"""
function GeoDynamo.create_dirichlet_bc(::Type{T}, nr::Int, value::T = zero(T)) where {T}
    solver_create_dirichlet_bc(T, nr, value)
end

"""
    solver_create_neumann_bc(T, d1_row, value=zero(T); l0_dirichlet=false)

Create a first-derivative endpoint descriptor.
"""
function solver_create_neumann_bc(
        ::Type{T},
        d1_row::Vector{T},
        value::T = zero(T);
        l0_dirichlet::Bool = false
) where {T}
    return SolverERK2BoundarySide{T}(
        :neumann,
        value,
        copy(d1_row),
        zero(T),
        zero(T),
        false,
        zero(T),
        l0_dirichlet
    )
end

"""
    GeoDynamo.create_neumann_bc(T, d1_row, value=zero(T); l0_dirichlet=false)

Create a public first-derivative ERK2 endpoint descriptor.
"""
function GeoDynamo.create_neumann_bc(
        ::Type{T},
        d1_row::Vector{T},
        value::T = zero(T);
        l0_dirichlet::Bool = false
) where {T}
    return solver_create_neumann_bc(T, d1_row, value; l0_dirichlet)
end

"""
    solver_create_stress_free_tor_bc(T, d1_row, r_inv)

Create the toroidal velocity stress-free endpoint descriptor.
"""
function solver_create_stress_free_tor_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where {T}
    return SolverERK2BoundarySide{T}(
        :stress_free_tor,
        zero(T),
        copy(d1_row),
        r_inv,
        zero(T),
        false,
        -r_inv,
        false
    )
end

"""
    GeoDynamo.create_stress_free_tor_bc(T, d1_row, r_inv)

Create a public toroidal-velocity stress-free endpoint descriptor.
"""
function GeoDynamo.create_stress_free_tor_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where {T}
    solver_create_stress_free_tor_bc(T, d1_row, r_inv)
end

"""
    solver_create_noslip_pol_bc(T, d1_row)

Create the poloidal velocity no-slip endpoint descriptor.
"""
function solver_create_noslip_pol_bc(::Type{T}, d1_row::Vector{T}) where {T}
    return SolverERK2BoundarySide{T}(
        :noslip_pol,
        zero(T),
        copy(d1_row),
        zero(T),
        zero(T),
        false,
        zero(T),
        false
    )
end

"""
    GeoDynamo.create_noslip_pol_bc(T, d1_row)

Create a public poloidal-velocity no-slip endpoint descriptor.
"""
function GeoDynamo.create_noslip_pol_bc(::Type{T}, d1_row::Vector{T}) where {T}
    solver_create_noslip_pol_bc(T, d1_row)
end

"""
    solver_create_stress_free_pol_bc(T, d2_row)

Create the poloidal velocity stress-free endpoint descriptor.
"""
function solver_create_stress_free_pol_bc(::Type{T}, d2_row::Vector{T}) where {T}
    return SolverERK2BoundarySide{T}(
        :stress_free_pol,
        zero(T),
        copy(d2_row),
        zero(T),
        zero(T),
        false,
        zero(T),
        false
    )
end

"""
    GeoDynamo.create_stress_free_pol_bc(T, d2_row)

Create a public poloidal-velocity stress-free endpoint descriptor.
"""
function GeoDynamo.create_stress_free_pol_bc(::Type{T}, d2_row::Vector{T}) where {T}
    solver_create_stress_free_pol_bc(T, d2_row)
end

"""
    solver_create_insulating_inner_bc(T, d1_row, r_inv)

Create the inner insulating magnetic poloidal endpoint descriptor.
"""
function solver_create_insulating_inner_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where {T}
    return SolverERK2BoundarySide{T}(
        :insulating_inner,
        zero(T),
        copy(d1_row),
        r_inv,
        -one(T),
        true,
        zero(T),
        false
    )
end

"""
    GeoDynamo.create_insulating_inner_bc(T, d1_row, r_inv)

Create a public inner-boundary insulating magnetic-poloidal descriptor.
"""
function GeoDynamo.create_insulating_inner_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where {T}
    solver_create_insulating_inner_bc(T, d1_row, r_inv)
end

"""
    solver_create_insulating_outer_bc(T, d1_row, r_inv)

Create the outer insulating magnetic poloidal endpoint descriptor.
"""
function solver_create_insulating_outer_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where {T}
    return SolverERK2BoundarySide{T}(
        :insulating_outer,
        zero(T),
        copy(d1_row),
        r_inv,
        one(T),
        true,
        r_inv,
        false
    )
end

"""
    GeoDynamo.create_insulating_outer_bc(T, d1_row, r_inv)

Create a public outer-boundary insulating magnetic-poloidal descriptor.
"""
function GeoDynamo.create_insulating_outer_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where {T}
    solver_create_insulating_outer_bc(T, d1_row, r_inv)
end

"""
    GeoDynamo.set_erk2_diagnostics_interval!(interval)

Set how often ERK2 stage residual diagnostics are reported.
"""
function GeoDynamo.set_erk2_diagnostics_interval!(interval::Int)
    interval <= 0 && error("ERK2 diagnostics interval must be positive, got $interval")
    SOLVER_SHARED_ERK2_DIAGNOSTICS_INTERVAL[] = interval
    set_erk2_diagnostics!(SOLVER_SHARED_ERK2_DIAGNOSTICS_ENABLED[], interval)
    return interval
end

"""
    GeoDynamo.enable_erk2_diagnostics!(; interval=...)

Enable ERK2 residual diagnostics and optionally update the reporting interval.
"""
function GeoDynamo.enable_erk2_diagnostics!(;
        interval::Int = SOLVER_SHARED_ERK2_DIAGNOSTICS_INTERVAL[])
    GeoDynamo.set_erk2_diagnostics_interval!(interval)
    SOLVER_SHARED_ERK2_DIAGNOSTICS_ENABLED[] = true
    set_erk2_diagnostics!(true, interval)
    return nothing
end

"""
    GeoDynamo.disable_erk2_diagnostics!()

Disable ERK2 residual diagnostics without changing the configured interval.
"""
function GeoDynamo.disable_erk2_diagnostics!()
    SOLVER_SHARED_ERK2_DIAGNOSTICS_ENABLED[] = false
    set_erk2_diagnostics!(false, SOLVER_SHARED_ERK2_DIAGNOSTICS_INTERVAL[])
    return nothing
end

"""
    GeoDynamo.erk2_diagnostics_enabled()

Return whether ERK2 residual diagnostics are currently enabled.
"""
GeoDynamo.erk2_diagnostics_enabled() = SOLVER_SHARED_ERK2_DIAGNOSTICS_ENABLED[]

"""
    GeoDynamo.erk2_diagnostics_interval()

Return the configured ERK2 residual diagnostics interval.
"""
GeoDynamo.erk2_diagnostics_interval() = SOLVER_SHARED_ERK2_DIAGNOSTICS_INTERVAL[]

"""
    build_solver_erk2_scalar_bc(T, domain, boundary_condition)

Translate scalar boundary-condition codes into an ERK2 boundary specification.

Boundary codes follow the existing scalar convention: DD, DN, ND, and NN for
codes 1 through 4.
"""
function build_solver_erk2_scalar_bc(::Type{T}, domain::RadialDomainType, boundary_condition::Int) where {T}
    nr = domain.N
    d1 = build_radial_derivative_matrix(T, 1, domain)
    d1_inner = extract_dense_row(d1.data, d1.bandwidth, nr, 1)
    d1_outer = extract_dense_row(d1.data, d1.bandwidth, nr, nr)

    inner = boundary_condition == 1 || boundary_condition == 2 ?
            solver_create_dirichlet_bc(T, nr) :
            solver_create_neumann_bc(T, d1_inner; l0_dirichlet = (boundary_condition == 4))

    outer = boundary_condition == 1 || boundary_condition == 3 ?
            solver_create_dirichlet_bc(T, nr) :
            solver_create_neumann_bc(T, d1_outer)

    return SolverERK2BoundarySpec{T}(inner, outer)
end

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
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing
) where {T}
    laplacian = build_radial_laplacian(domain)
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
    GeoDynamo.compute_phi1_function(A, expA)

Public compatibility wrapper for the solver-local phi1 matrix-function helper.
"""
function GeoDynamo.compute_phi1_function(A::Matrix{T}, expA::Matrix{T}) where {T}
    solver_compute_phi1_function(A, expA)
end

"""
    GeoDynamo.compute_phi2_function(A, expA; l=0)

Public compatibility wrapper for the solver-local phi2 matrix-function helper.
"""
function GeoDynamo.compute_phi2_function(A::Matrix{T}, expA::Matrix{T}; l::Int = 0) where {T}
    solver_compute_phi2_function(A, expA; l = l)
end

"""
    GeoDynamo.reset_phi2_monitor!()

Clear accumulated phi2 conditioning diagnostics.
"""
GeoDynamo.reset_phi2_monitor!() = reset_solver_phi2_monitor!()

"""
    GeoDynamo.report_phi2_conditioning(step; interval=100)

Report phi2 conditioning diagnostics on the requested interval.
"""
function GeoDynamo.report_phi2_conditioning(step::Int; interval::Int = 100)
    report_solver_phi2_conditioning(step; interval = interval)
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
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing
)::ERK2StageCache{T} where {T}
    nr = domain.N
    needs_rebuild = existing === nothing ||
                    existing.diffusivity != diffusivity ||
                    existing.nr != nr ||
                    existing.dt != dt ||
                    existing.use_krylov != use_krylov ||
                    !existing.mpi_consistent ||
                    existing.l_values != config.l_values

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
            bc_spec
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
                    existing.l_values != config.l_values

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
    build_solver_erk2_velocity_tor_bc(T, domain, velocity_bc_code; config=nothing, rot_omega=0.0)

Create ERK2 boundary descriptors for the velocity toroidal component.

When a rotating inner core is requested, the `(l=1, m=0)` mode gets a
mode-dependent inner boundary value.
"""
function build_solver_erk2_velocity_tor_bc(
        ::Type{T},
        domain::RadialDomainType,
        velocity_bc_code::Int;
        config::Union{SHTnsConfigType, Nothing} = nothing,
        rot_omega::Float64 = 0.0
) where {T}
    nr = domain.N
    d1 = build_radial_derivative_matrix(T, 1, domain)
    d1_inner = extract_dense_row(d1.data, d1.bandwidth, nr, 1)
    d1_outer = extract_dense_row(d1.data, d1.bandwidth, nr, nr)
    r_inv_inner = T(domain.r[1, 3])
    r_inv_outer = T(domain.r[nr, 3])

    inner = velocity_bc_code == 1 || velocity_bc_code == 2 ?
            solver_create_dirichlet_bc(T, nr) :
            solver_create_stress_free_tor_bc(T, d1_inner, r_inv_inner)

    outer = velocity_bc_code == 1 || velocity_bc_code == 3 ?
            solver_create_dirichlet_bc(T, nr) :
            solver_create_stress_free_tor_bc(T, d1_outer, r_inv_outer)

    inner_mode_values = nothing
    if rot_omega != 0.0 && (velocity_bc_code == 1 || velocity_bc_code == 2) &&
       config !== nothing
        r_inner = T(domain.r[1, 4])
        inner_mode_values = zeros(T, length(config.l_values))
        for lm_idx in eachindex(config.l_values)
            if config.l_values[lm_idx] == 1 && config.m_values[lm_idx] == 0
                inner_mode_values[lm_idx] = T(rot_omega) * r_inner
            end
        end
    end

    return SolverERK2BoundarySpec{T}(
        inner, outer, inner_mode_values, nothing, nothing, nothing)
end

"""
    build_solver_erk2_velocity_pol_bc(T, domain, velocity_bc_code)

Create ERK2 boundary descriptors for the velocity poloidal component.
"""
function build_solver_erk2_velocity_pol_bc(::Type{T}, domain::RadialDomainType, velocity_bc_code::Int) where {T}
    nr = domain.N
    d1 = build_radial_derivative_matrix(T, 1, domain)
    d2 = build_radial_derivative_matrix(T, 2, domain)
    d1_inner = extract_dense_row(d1.data, d1.bandwidth, nr, 1)
    d1_outer = extract_dense_row(d1.data, d1.bandwidth, nr, nr)
    d2_inner = extract_dense_row(d2.data, d2.bandwidth, nr, 1)
    d2_outer = extract_dense_row(d2.data, d2.bandwidth, nr, nr)

    inner = velocity_bc_code == 1 || velocity_bc_code == 2 ?
            solver_create_noslip_pol_bc(T, d1_inner) :
            solver_create_stress_free_pol_bc(T, d2_inner)

    outer = velocity_bc_code == 1 || velocity_bc_code == 3 ?
            solver_create_noslip_pol_bc(T, d1_outer) :
            solver_create_stress_free_pol_bc(T, d2_outer)

    return SolverERK2BoundarySpec{T}(inner, outer)
end

"""
    build_solver_erk2_magnetic_tor_bc(T, nr)

Create homogeneous Dirichlet boundary descriptors for magnetic toroidal fields.
"""
function build_solver_erk2_magnetic_tor_bc(::Type{T}, nr::Int) where {T}
    return SolverERK2BoundarySpec{T}(
        solver_create_dirichlet_bc(T, nr),
        solver_create_dirichlet_bc(T, nr)
    )
end

"""
    build_solver_erk2_magnetic_pol_bc(T, domain)

Create insulating boundary descriptors for magnetic poloidal fields.
"""
function build_solver_erk2_magnetic_pol_bc(::Type{T}, domain::RadialDomainType) where {T}
    nr = domain.N
    d1 = build_radial_derivative_matrix(T, 1, domain)
    d1_inner = extract_dense_row(d1.data, d1.bandwidth, nr, 1)
    d1_outer = extract_dense_row(d1.data, d1.bandwidth, nr, nr)
    r_inv_inner = T(domain.r[1, 3])
    r_inv_outer = T(domain.r[nr, 3])

    inner = solver_create_insulating_inner_bc(T, d1_inner, r_inv_inner)
    outer = solver_create_insulating_outer_bc(T, d1_outer, r_inv_outer)
    return SolverERK2BoundarySpec{T}(inner, outer)
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
    laplacian = build_radial_laplacian(domain)
    bandwidth = laplacian.bandwidth
    r_inv_sq = @views domain.r[1:nr, 2]

    base_data = T.(diffusivity .* laplacian.data)
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
            @inbounds for j in 1:(1 + bandwidth)
                physical_system_data[bandwidth + 1 + 1 - j, j] = second_derivative.data[bandwidth + 1 + 1 - j, j]
            end
        end

        if velocity_bc_code == 1 || velocity_bc_code == 3
            @inbounds for j in (nr - bandwidth):nr
                physical_system_data[bandwidth + 1 + nr - j, j] = first_derivative.data[bandwidth + 1 + nr - j, j]
            end
        else
            @inbounds for j in (nr - bandwidth):nr
                physical_system_data[bandwidth + 1 + nr - j, j] = second_derivative.data[bandwidth + 1 + nr - j, j]
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
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing
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
        bc_spec = bc_spec
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
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing
) where {T}
    if key === :velocity_toroidal
        return get_solver_erk2_cache!(
            caches, Val(:velocity_toroidal), diffusivity, T, config, domain, dt;
            use_krylov = use_krylov, m = m, tol = tol, bc_spec = bc_spec
        )
    elseif key === :velocity_poloidal
        return get_solver_erk2_cache!(
            caches, Val(:velocity_poloidal), diffusivity, T, config, domain, dt;
            use_krylov = use_krylov, m = m, tol = tol, bc_spec = bc_spec
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
                    existing.l_values != config.l_values

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
                    existing.l_values != config.l_values

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

    return SolverERK2FieldBuffers{T}(
        similar(real_data),
        similar(imag_data),
        similar(real_data),
        similar(imag_data),
        similar(real_data),
        similar(imag_data),
        similar(nl_real),
        similar(nl_imag),
        similar(nl_real),
        similar(nl_imag),
        cache_lookup,
        nr,
        workspace
    )
end

const ERK2FieldBuffers = SolverERK2FieldBuffers

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
    if params.include_magnetic_field && state.fields.magnetic !== nothing
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
