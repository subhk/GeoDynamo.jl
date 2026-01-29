# ================================================================================
# Exponential 2nd Order Runge-Kutta (ERK2) Implementation
# ================================================================================

# ---- ERK2 Boundary Condition Specification ----

"""
    ERK2BoundarySide{T}

Encodes how to enforce a boundary condition at one boundary (inner or outer)
for the ERK2 exponential integrator.

All BC types are expressed as a linear constraint:
    sum_j stencil[j] * u[j] + l_correction * u[b] = value

which is solved for u[b]:
    u[b] = (value - sum_{j≠b} stencil[j] * u[j]) / (stencil[b] + l_correction)

BC types and their encoding:
  - :dirichlet         → stencil = delta_{b}, value = BC_val  (u[b] = BC_val)
  - :neumann           → stencil = d1[b,:], value = q         (du/dr = q)
  - :stress_free_tor   → stencil = d1[b,:], l_correction = -1/r_b  (dT/dr - T/r = 0)
  - :noslip_pol        → stencil = d1[b,:], value = 0         (dP/dr = 0)
  - :stress_free_pol   → stencil = d2[b,:], value = 0         (d²P/dr² = 0)
  - :insulating_inner  → stencil = d1[b,:], l_correction = -l/r_b  (d/dr - l/r)P = 0
  - :insulating_outer  → stencil = d1[b,:], l_correction = +(l+1)/r_b  (d/dr + (l+1)/r)P = 0
"""
struct ERK2BoundarySide{T}
    type::Symbol              # BC type identifier
    value::T                  # Prescribed RHS value (BC_val for Dirichlet, flux for Neumann)
    stencil::Vector{T}        # Dense derivative stencil row (length nr)
    r_inv::T                  # 1/r at this boundary
    l_sign::T                 # Sign/multiplier for l-dependent correction (+1, -1, or 0)
    use_l_correction::Bool    # Whether correction is l-dependent
    fixed_correction::T       # Fixed additive correction to diagonal (e.g., -1/r for stress-free tor)
    l0_dirichlet::Bool        # Override to Dirichlet for l=0 (avoids underdetermined NN systems)
end

"""
    ERK2BoundarySpec{T}

Complete boundary condition specification for both boundaries.

`inner_mode_values` and `outer_mode_values` are optional per-mode value overrides
(indexed by lm_idx). When provided, they override `bc_side.value` for specific modes.
Used e.g. for rotating inner core: T(l=1,m=0) = rot_omega * r_inner.
"""
struct ERK2BoundarySpec{T}
    inner::ERK2BoundarySide{T}
    outer::ERK2BoundarySide{T}
    inner_mode_values::Union{Nothing, Vector{T}}
    outer_mode_values::Union{Nothing, Vector{T}}
end

# Convenience constructor without per-mode values
function ERK2BoundarySpec{T}(inner::ERK2BoundarySide{T}, outer::ERK2BoundarySide{T}) where T
    return ERK2BoundarySpec{T}(inner, outer, nothing, nothing)
end

"""
    enforce_erk2_bc!(result, bc_side, boundary_idx, l, nr)

Enforce a boundary condition on the result vector at the given boundary index.
Modifies `result[boundary_idx]` to satisfy the linear constraint encoded in `bc_side`.
"""
function enforce_erk2_bc!(result::AbstractVector{T}, bc_side::ERK2BoundarySide{T},
                          boundary_idx::Int, l::Int, nr::Int;
                          value_override::Union{T, Nothing}=nothing) where T
    b = boundary_idx
    effective_value = value_override !== nothing ? value_override : bc_side.value

    # l=0 override: use Dirichlet to avoid underdetermined NN systems
    # (matches CNAB2 treatment: pin value at inner boundary for l=0 when both are Neumann)
    if bc_side.l0_dirichlet && l == 0
        result[b] = effective_value
        return
    end

    # Dirichlet: just set the value directly
    if bc_side.type === :dirichlet
        result[b] = effective_value
        return
    end

    # Compute the effective diagonal coefficient at boundary:
    # self_coeff = stencil[b] + fixed_correction + l_sign * l * r_inv (if l-dependent)
    self_coeff = bc_side.stencil[b] + bc_side.fixed_correction
    if bc_side.use_l_correction
        self_coeff += bc_side.l_sign * T(l) * bc_side.r_inv
    end

    # Compute the off-diagonal contribution: sum_{j≠b} stencil[j] * result[j]
    off_diag_sum = zero(T)
    @inbounds for j in 1:nr
        if j != b
            off_diag_sum += bc_side.stencil[j] * result[j]
        end
    end

    # Solve: self_coeff * u[b] + off_diag_sum = value
    # → u[b] = (value - off_diag_sum) / self_coeff
    if abs(self_coeff) > eps(T) * T(100)
        result[b] = (effective_value - off_diag_sum) / self_coeff
    else
        # Degenerate case (e.g., l=0 with insulating BC): fall back to value
        result[b] = effective_value
    end
end

"""
    create_dirichlet_bc(T, nr, value) -> ERK2BoundarySide{T}

Create a Dirichlet BC side (u = value at boundary).
"""
function create_dirichlet_bc(::Type{T}, nr::Int, value::T=zero(T)) where T
    stencil = zeros(T, nr)
    return ERK2BoundarySide{T}(:dirichlet, value, stencil, zero(T), zero(T), false, zero(T), false)
end

"""
    create_neumann_bc(T, d1_row, value) -> ERK2BoundarySide{T}

Create a Neumann BC side (du/dr = value at boundary).
"""
function create_neumann_bc(::Type{T}, d1_row::Vector{T}, value::T=zero(T); l0_dirichlet::Bool=false) where T
    nr = length(d1_row)
    return ERK2BoundarySide{T}(:neumann, value, copy(d1_row), zero(T), zero(T), false, zero(T), l0_dirichlet)
end

"""
    create_stress_free_tor_bc(T, d1_row, r_inv) -> ERK2BoundarySide{T}

Create a stress-free toroidal BC side: dT/dr - T/r = 0.
"""
function create_stress_free_tor_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where T
    return ERK2BoundarySide{T}(:stress_free_tor, zero(T), copy(d1_row), r_inv, zero(T), false, -r_inv, false)
end

"""
    create_noslip_pol_bc(T, d1_row) -> ERK2BoundarySide{T}

Create a no-slip poloidal BC side: dP/dr = 0.
"""
function create_noslip_pol_bc(::Type{T}, d1_row::Vector{T}) where T
    return ERK2BoundarySide{T}(:noslip_pol, zero(T), copy(d1_row), zero(T), zero(T), false, zero(T), false)
end

"""
    create_stress_free_pol_bc(T, d2_row) -> ERK2BoundarySide{T}

Create a stress-free poloidal BC side: d²P/dr² = 0.
"""
function create_stress_free_pol_bc(::Type{T}, d2_row::Vector{T}) where T
    return ERK2BoundarySide{T}(:stress_free_pol, zero(T), copy(d2_row), zero(T), zero(T), false, zero(T), false)
end

"""
    create_insulating_inner_bc(T, d1_row, r_inv) -> ERK2BoundarySide{T}

Create an insulating magnetic poloidal inner BC: (d/dr - l/r)P = 0.
The l-factor is applied dynamically during enforcement.
"""
function create_insulating_inner_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where T
    return ERK2BoundarySide{T}(:insulating_inner, zero(T), copy(d1_row), r_inv, -one(T), true, zero(T), false)
end

"""
    create_insulating_outer_bc(T, d1_row, r_inv) -> ERK2BoundarySide{T}

Create an insulating magnetic poloidal outer BC: (d/dr + (l+1)/r)P = 0.
The (l+1) factor is split: fixed_correction = +1/r (the "+1" part),
and l_sign = +1 with use_l_correction = true (the "l" part).
Total correction = l_sign*l*r_inv + fixed_correction = (l+1)/r.
"""
function create_insulating_outer_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where T
    return ERK2BoundarySide{T}(:insulating_outer, zero(T), copy(d1_row), r_inv, one(T), true, r_inv, false)
end
