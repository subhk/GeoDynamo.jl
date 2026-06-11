# ERK2 boundary conditions: endpoint BC constructors, enforcement, and per-field BC-spec builders.

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

Create the inner insulating magnetic poloidal endpoint descriptor,
(∂r − (l+1)/r)P = 0: under B_r = λP/r² the interior vacuum solution is
P ∝ r^{l+1} (B = −∇Φ, Φ ∝ r^l regular at the origin ⇒ B_r ∝ r^{l−1} = λP/r²).
Encoded as l_sign = −1, fixed_correction = −r_inv (self_coeff =
d1 − l·r_inv − r_inv). Matches the banded row in
`create_magnetic_poloidal_matrices`.
"""
function solver_create_insulating_inner_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where {T}
    return SolverERK2BoundarySide{T}(
        :insulating_inner,
        zero(T),
        copy(d1_row),
        r_inv,
        -one(T),
        true,
        -r_inv,
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

Create the outer insulating magnetic poloidal endpoint descriptor,
(∂r + l/r)P = 0: under B_r = λP/r² the exterior vacuum solution is P ∝ r^{−l}
(B = −∇Φ, Φ ∝ r^{−(l+1)} ⇒ B_r ∝ r^{−(l+2)} = λP/r²). Encoded as l_sign = +1,
fixed_correction = 0 (self_coeff = d1 + l·r_inv). Verified by the classic
full-sphere dipole free-decay rate σ = π² (test/ball_bessel_decay.jl); matches
the banded row in `create_magnetic_poloidal_matrices`.
"""
function solver_create_insulating_outer_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where {T}
    return SolverERK2BoundarySide{T}(
        :insulating_outer,
        zero(T),
        copy(d1_row),
        r_inv,
        one(T),
        true,
        zero(T),
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
