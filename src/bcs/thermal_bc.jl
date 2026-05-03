# ================================================================================
# Temperature Boundary Conditions - Matrix-Embedded Approach
# ================================================================================
#
# Temperature uses the shared scalar boundary implementation in
# `src/bcs/scalar_bc.jl`. This file keeps the temperature-specific public API
# and names used by existing callers.

"""
    create_temperature_matrices(config, domain, diffusivity, dt;
                                 temperature_bc_code, theta, T)

Create implicit time-stepping matrices for the temperature equation with
Dirichlet/Neumann boundary rows embedded in the radial system matrix.
"""
function create_temperature_matrices(
    config::SHTnsKitConfig,
    domain::RadialDomain,
    diffusivity::Float64,
    dt::Float64;
    temperature_bc_code::Int,
    theta::Float64=0.5,
    T::Type{<:Number}=Float64,
)
    return create_scalar_matrices(
        config,
        domain,
        diffusivity,
        dt;
        scalar_bc_code=temperature_bc_code,
        theta,
        T,
    )
end

"""
    set_temperature_rhs_bc!(rhs_real, rhs_imag, slot, nr; ...)

Set temperature boundary RHS values for one local spectral mode.
"""
function set_temperature_rhs_bc!(
    rhs_real::AbstractArray{T},
    rhs_imag::AbstractArray{T},
    slot::CartesianIndex{2},
    nr::Int;
    inner_value::T=zero(T),
    outer_value::T=zero(T),
    inner_value_imag::T=zero(T),
    outer_value_imag::T=zero(T),
) where T
    return set_scalar_rhs_bc!(
        rhs_real,
        rhs_imag,
        slot,
        nr;
        inner_value,
        outer_value,
        inner_value_imag,
        outer_value_imag,
    )
end

"""
    solve_temperature_implicit_step!(solution, rhs, matrices; ...)

Solve the implicit temperature system with boundary conditions embedded in the
matrix and boundary values supplied as mode-indexed RHS vectors.
"""
function solve_temperature_implicit_step!(
    solution::SHTnsSpecField{T},
    rhs::SHTnsSpecField{T},
    matrices::SHTnsImplicitMatrices{T};
    bc_inner::Union{AbstractVector{T},Nothing}=nothing,
    bc_outer::Union{AbstractVector{T},Nothing}=nothing,
    bc_inner_imag::Union{AbstractVector{T},Nothing}=nothing,
    bc_outer_imag::Union{AbstractVector{T},Nothing}=nothing,
) where T
    return solve_scalar_implicit_step!(
        solution,
        rhs,
        matrices;
        bc_inner,
        bc_outer,
        bc_inner_imag,
        bc_outer_imag,
    )
end
