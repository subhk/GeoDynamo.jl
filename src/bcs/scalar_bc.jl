# ================================================================================
# Shared Scalar Boundary Conditions - Matrix-Embedded Approach
# ================================================================================
#
# Temperature and composition use the same scalar radial operator and the same
# Dirichlet/Neumann boundary-row embedding. The public temperature/composition
# files keep field-specific names and documentation; this file holds the common
# implementation so the two paths cannot drift.

@inline _scalar_inner_is_dirichlet(scalar_bc_code::Int) = scalar_bc_code == 1 ||
                                                          scalar_bc_code == 2

@inline _scalar_outer_is_dirichlet(scalar_bc_code::Int) = scalar_bc_code == 1 ||
                                                          scalar_bc_code == 3

@inline function _scalar_mode_bc_value(values::Union{AbstractVector{T}, Nothing}, lm_idx::Int) where {T}
    return values !== nothing && lm_idx <= length(values) ? values[lm_idx] : zero(T)
end

function _zero_scalar_boundary_rows!(system_data::AbstractMatrix{T}, bw::Int, N::Int) where {T}
    @inbounds for j in 1:(1 + bw)
        system_data[bw + 1 + 1 - j, j] = zero(T)
    end
    @inbounds for j in (N - bw):N
        system_data[bw + 1 + N - j, j] = zero(T)
    end
    return system_data
end

function _apply_scalar_boundary_rows!(
        system_data::AbstractMatrix{T},
        d1_data::AbstractMatrix{T},
        scalar_bc_code::Int,
        l::Int,
        bw::Int,
        N::Int,
        inner_regularity::Bool = false,
        r_inv_inner::Float64 = 0.0
) where {T}
    if inner_regularity
        # Ball center regularity: Θ ~ r^l ⇒ Θ′(r₁) = l·Θ(r₁)/r₁  (β = l;
        # l=0 reduces to Θ′(r₁)=0). Exact to leading order; consistency
        # error O(r₁²) shrinks as N⁻² (see ball design spec §5).
        @inbounds for j in 1:(1 + bw)
            system_data[bw + 1 + 1 - j, j] = d1_data[bw + 1 + 1 - j, j]
        end
        system_data[bw + 1, 1] -= T(l * r_inv_inner)
    elseif _scalar_inner_is_dirichlet(scalar_bc_code)
        system_data[bw + 1, 1] = one(T)
    else
        @inbounds for j in 1:(1 + bw)
            system_data[bw + 1 + 1 - j, j] = d1_data[bw + 1 + 1 - j, j]
        end
    end

    if _scalar_outer_is_dirichlet(scalar_bc_code)
        system_data[bw + 1, N] = one(T)
    else
        @inbounds for j in (N - bw):N
            system_data[bw + 1 + N - j, j] = d1_data[bw + 1 + N - j, j]
        end
    end

    # NOTE: double-Neumann (scalar_bc_code == 4) needs NO special l == 0 handling.
    # The time-stepping operator (mass/dt) I − θ κ L is a positive-shifted
    # Helmholtz operator and stays non-singular under pure-Neumann rows on every
    # degree, including the l = 0 mean mode — the (mass/dt) I shift lifts the
    # bare-Laplacian constant null space. A former Dirichlet "pin" here overwrote
    # the inner Neumann derivative row with an identity row, which then imposed
    # the inner FLUX datum as a prescribed VALUE on the mean mode (wrong
    # units/condition for nonzero inner flux). The genuinely singular steady
    # solve lives in conductive_profile_solve, which gauge-fixes separately.

    return system_data
end

"""
    create_scalar_matrices(config, domain, diffusivity, dt; scalar_bc_code, theta, T,
                           inner_regularity)

Create implicit scalar time-stepping matrices with Dirichlet/Neumann boundary
conditions embedded in the first and last radial rows.

When `inner_regularity = true` the inner boundary row implements the ball
centre regularity condition Θ′(r₁) = l·Θ(r₁)/r₁ (Robin with β = l; reduces
to Neumann Θ′=0 for l=0). This is appropriate for full-sphere geometry where
there is no physical inner boundary; the outer row is still determined by
`scalar_bc_code`. Defaults to `false` (shell geometry, standard BCs on both
ends).
"""
function create_scalar_matrices(
        config::SHTnsKitConfig,
        domain::RadialDomain,
        diffusivity::Float64,
        dt::Float64;
        scalar_bc_code::Int,
        theta::Float64 = 0.5,
        T::Type{<:Number} = Float64,
        inner_regularity::Bool = false
)
    unique_l = unique(config.l_values)
    laplacian = create_radial_laplacian(domain)
    r_inv_sq = @views domain.r[1:domain.N, 2]

    base_data = T.(diffusivity .* laplacian.data)
    system_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    linear_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    factorizations = Vector{BandedLU{T}}(undef, length(unique_l))
    l_values = Vector{Int}(undef, length(unique_l))
    lookup = Dict{Int, Int}()

    d1_matrix = create_derivative_matrix(T, 1, domain)
    bw = radial_bandwidth(domain)
    N = domain.N

    inv_dt = T(1 / dt)
    minus_theta = -T(theta)

    for (idx, l) in enumerate(unique_l)
        l_values[idx] = l
        lookup[l] = idx

        linear_data = copy(base_data)
        l_factor = Float64(l * (l + 1))
        @inbounds for n in 1:N
            linear_data[bw + 1, n] -= T(diffusivity * l_factor * r_inv_sq[n])
        end

        linear_matrix = BandedMatrix{T}(copy(linear_data), bw, N)

        system_data = copy(linear_data)
        system_data .*= minus_theta
        system_data[bw + 1, :] .+= inv_dt

        _zero_scalar_boundary_rows!(system_data, bw, N)
        _apply_scalar_boundary_rows!(system_data, d1_matrix.data,
            scalar_bc_code, l, bw, N, inner_regularity, domain.r[1, 3])

        system_matrix = BandedMatrix{T}(system_data, bw, N)
        system_matrices[idx] = system_matrix
        linear_matrices[idx] = linear_matrix
        factorizations[idx] = factorize_banded(system_matrix)
    end

    return SHTnsImplicitMatrices{T}(
        system_matrices,
        factorizations,
        linear_matrices,
        l_values,
        lookup,
        theta
    )
end

"""
    set_scalar_rhs_bc!(rhs_real, rhs_imag, slot, nr; inner_value, outer_value, ...)

Set inner and outer radial RHS entries for one local spectral mode.
"""
function set_scalar_rhs_bc!(
        rhs_real::AbstractArray{T},
        rhs_imag::AbstractArray{T},
        slot::CartesianIndex{2},
        nr::Int;
        inner_value::T = zero(T),
        outer_value::T = zero(T),
        inner_value_imag::T = zero(T),
        outer_value_imag::T = zero(T)
) where {T}
    set_local_spectral_value!(rhs_real, slot, 1, inner_value)
    set_local_spectral_value!(rhs_imag, slot, 1, inner_value_imag)
    set_local_spectral_value!(rhs_real, slot, nr, outer_value)
    set_local_spectral_value!(rhs_imag, slot, nr, outer_value_imag)
    return nothing
end

"""
    solve_scalar_implicit_step!(solution, rhs, matrices; bc_inner, bc_outer, ...)

Solve a matrix-embedded scalar implicit step for each locally owned spectral
mode. Boundary vectors are global mode-indexed arrays; missing values default
to homogeneous real and imaginary endpoint values.
"""
function solve_scalar_implicit_step!(
        solution::SHTnsSpecField{T},
        rhs::SHTnsSpecField{T},
        matrices::SHTnsImplicitMatrices{T};
        bc_inner::Union{AbstractVector{T}, Nothing} = nothing,
        bc_outer::Union{AbstractVector{T}, Nothing} = nothing,
        bc_inner_imag::Union{AbstractVector{T}, Nothing} = nothing,
        bc_outer_imag::Union{AbstractVector{T}, Nothing} = nothing
) where {T}
    sol_real = parent(solution.data_real)
    sol_imag = parent(solution.data_imag)
    rhs_real = parent(rhs.data_real)
    rhs_imag = parent(rhs.data_imag)

    lm_range = local_spectral_mode_indices(solution.config)
    nr = matrices.system_matrices[1].size
    tmp_r = Vector{T}(undef, nr)
    tmp_i = Vector{T}(undef, nr)

    @inbounds for lm_idx in lm_range
        slot = local_spectral_storage_slot(solution.config, lm_idx)
        slot === nothing && continue

        l = solution.config.l_values[lm_idx]
        idx = get(matrices.lookup, l, nothing)
        idx === nothing && continue

        for ir in 1:nr
            tmp_r[ir] = local_spectral_value(rhs_real, slot, ir)
            tmp_i[ir] = local_spectral_value(rhs_imag, slot, ir)
        end

        tmp_r[1] = _scalar_mode_bc_value(bc_inner, lm_idx)
        tmp_i[1] = _scalar_mode_bc_value(bc_inner_imag, lm_idx)
        tmp_r[nr] = _scalar_mode_bc_value(bc_outer, lm_idx)
        tmp_i[nr] = _scalar_mode_bc_value(bc_outer_imag, lm_idx)

        solve_banded!(tmp_r, matrices.factorizations[idx], tmp_r)
        solve_banded!(tmp_i, matrices.factorizations[idx], tmp_i)

        for ir in 1:nr
            set_local_spectral_value!(sol_real, slot, ir, tmp_r[ir])
            set_local_spectral_value!(sol_imag, slot, ir, tmp_i[ir])
        end
    end

    return solution
end
