
# Banded matrix structure for radial derivatives
struct BandedMatrix{T}
    data::Matrix{T}           # Banded storage
    bandwidth::Int            # Half-bandwidth
    size::Int                 # Matrix size
end

# =============================
# Banded LU factorization/solve
# =============================

struct BandedLU{T}
    lu::Matrix{T}     # In-place LU factors in banded storage
    bandwidth::Int
    size::Int
end

@inline function _band_row(i::Int, j::Int, bw::Int)
    return bw + 1 + i - j
end

function factorize_banded(A::BandedMatrix{T}) where {T}
    N = A.size
    bw = A.bandwidth
    lu = copy(A.data)

    @inbounds for k in 1:(N - 1)
        # Pivot (no pivoting for banded SPD-like operators)
        piv_row = _band_row(k, k, bw)
        if !(1 <= piv_row <= 2*bw+1)
            continue
        end
        piv = lu[piv_row, k]

        # Check for singular matrix (zero or near-zero pivot)
        tol = pivot_tol(T)
        if abs(piv) < tol
            error("Singular matrix detected during LU factorization at pivot $k. " *
                  "Pivot value = $piv, which is below tolerance $tol. " *
                  "This may indicate:\n" *
                  "  1. Ill-conditioned derivative matrix\n" *
                  "  2. Incorrect boundary conditions\n" *
                  "  3. Numerical instability in grid setup")
        end

        # Eliminate entries below pivot within bandwidth
        i_max = min(N, k + bw)
        for i in (k + 1):i_max
            row = _band_row(i, k, bw)
            if 1 <= row <= 2*bw+1
                L = lu[row, k] / piv
                lu[row, k] = L  # store L below diagonal
                # Update row i for columns within band
                j_max = min(N, k + bw)
                for j in (k + 1):j_max
                    col = _band_row(i, j, bw)
                    if 1 <= col <= 2*bw+1
                        urow = _band_row(k, j, bw)
                        if 1 <= urow <= 2*bw+1
                            lu[col, j] -= L * lu[urow, j]
                        end
                    end
                end
            end
        end
    end
    return BandedLU{T}(lu, bw, N)
end

"""
    solve_banded!(x, lu, b)

Solve the banded linear system `lu * x = b` in-place.

**In-place aliasing `x === b` is explicitly supported and safe.** The forward
substitution sweep processes rows in ascending order: at step `i`, it reads
`x[j]` for `j < i` (already holding the intermediate result `y[j]`) and `b[i]`
(not yet overwritten when aliased). The back substitution sweep processes rows
in descending order with the same safe access pattern.

WARNING: Do not change the loop ordering without verifying aliasing safety.
"""
function solve_banded!(x::Vector{T}, lu::BandedLU{T}, b::Vector{T}) where {T}
    N = lu.size
    bw = lu.bandwidth
    # Forward substitution: L y = b (L has unit diagonal)
    # NOTE: x === b aliasing is safe — see docstring
    @inbounds for i in 1:N
        s = zero(T)
        j_min = max(1, i - bw)
        for j in j_min:(i - 1)
            row = _band_row(i, j, bw)
            if 1 <= row <= 2*bw+1
                s += lu.lu[row, j] * x[j]
            end
        end
        x[i] = b[i] - s
    end

    # Back substitution: U x = y
    @inbounds for i in N:-1:1
        s = zero(T)
        j_max = min(N, i + bw)
        for j in (i + 1):j_max
            row = _band_row(i, j, bw)
            if 1 <= row <= 2*bw+1
                s += lu.lu[row, j] * x[j]
            end
        end
        diag_row = _band_row(i, i, bw)
        diag_val = lu.lu[diag_row, i]

        # Check for zero diagonal during back substitution
        tol = pivot_tol(T)
        if abs(diag_val) < tol
            error("Zero diagonal detected during back substitution at row $i. " *
                  "Diagonal value = $diag_val, which is below tolerance $tol. " *
                  "The LU factorization is singular.")
        end

        x[i] = (x[i] - s) / diag_val
    end
    return x
end

function create_derivative_matrix(::Type{T}, order::Int, domain::RadialDomain) where {T}
    # Create finite difference matrix for given derivative order
    N = domain.N
    bandwidth = isempty(domain.dr_matrices) ?
                error("create_derivative_matrix: cannot infer bandwidth — " *
                      "domain.dr_matrices is empty; ensure the domain was " *
                      "constructed with pre-computed derivative matrices") :
                (size(domain.dr_matrices[1], 1) - 1) ÷ 2
    calc_T = promote_type(T, eltype(domain.r))

    # Initialize banded matrix storage
    data = zeros(T, 2*bandwidth + 1, N)

    # Compute finite difference coefficients using Chebyshev points
    for n in 1:N
        left = max(1, n - bandwidth)
        right = min(N, n + bandwidth)
        stencil_size = right - left + 1

        # Vandermonde matrix for interpolation
        V = ones(calc_T, stencil_size, stencil_size)
        points = calc_T.(domain.r[left:right, 4])  # r values
        center = calc_T(domain.r[n, 4])

        for j in 2:stencil_size
            for i in 1:stencil_size
                V[i, j] = V[i, j - 1] * (points[i] - center)
            end
        end

        # Solve for derivative coefficients with error handling
        rhs = zeros(calc_T, stencil_size)
        if order + 1 > stencil_size
            error("Insufficient stencil size ($stencil_size) for derivative order $order at grid point $n. " *
                  "Need at least $(order + 1) points. Consider increasing radial_bandwidth or " *
                  "using a coarser grid near boundaries.")
        end
        rhs[order + 1] = calc_T(factorial(order))

        # Solve for derivative coefficients
        # We need the (order+1)-th ROW of V^(-1), not the column.
        # Row k of V^(-1) = (V^(-T) * e_k)^T, so we solve V^T * w = rhs
        # which gives w = V^(-T) * rhs = transpose of row (order+1) of V^(-1)
        coeffs = try
            transpose(V) \ rhs
        catch e
            error("Failed to solve Vandermonde system at grid point $n. " *
                  "This indicates severe ill-conditioning. " *
                  "Consider using a different grid spacing or derivative method. " *
                  "Original error: $e")
        end

        # Store in banded format
        for (i, idx) in enumerate(left:right)
            band_row = bandwidth + 1 + n - idx
            if 1 <= band_row <= 2*bandwidth + 1
                data[band_row, idx] = T(coeffs[i])
            end
        end
    end

    return BandedMatrix{T}(data, bandwidth, N)
end

function create_derivative_matrix(order::Int, domain::RadialDomain)
    return create_derivative_matrix(eltype(domain.r), order, domain)
end

function create_radial_laplacian(::Type{T}, domain::RadialDomain) where {T}
    # d²/dr² + (2/r) d/dr
    d2_matrix = create_derivative_matrix(T, 2, domain)
    d1_matrix = create_derivative_matrix(T, 1, domain)

    bandwidth = d2_matrix.bandwidth
    laplacian_data = copy(d2_matrix.data)

    # Add (2/r) * d/dr term
    for n in 1:domain.N
        r_inv = T(domain.r[n, 3])  # 1/r
        for j in max(1, n - bandwidth):min(domain.N, n + bandwidth)
            band_row = bandwidth + 1 + n - j
            laplacian_data[band_row, j] += T(2) * r_inv * d1_matrix.data[band_row, j]
        end
    end

    return BandedMatrix{T}(laplacian_data, bandwidth, domain.N)
end

function create_radial_laplacian(domain::RadialDomain)
    return create_radial_laplacian(eltype(domain.r), domain)
end

"""
    _populate_radial_operators!(domain) -> domain

Fill `domain.dr_matrices` and `domain.radial_laplacian` with the finite-difference
operators for this grid, in place.

The `RadialDomain` constructors allocate these as zero matrices. Without this
step they stay zero: every current consumer recomputes via
`create_derivative_matrix`/`create_radial_laplacian`, but any code that reads the
stored fields directly would silently get zeros (wrong derivatives, no error).
Populating them keeps the cached operators consistent with a fresh recompute.

Degenerate grids with `N < length(dr_matrices) + 1` lack enough stencil points for
the highest-order derivative; their operators are left zero (as before). Physical
runs use `N` far larger than the stencil order, so this only affects toy domains
that never build radial operators anyway.
"""
function _populate_radial_operators!(domain::RadialDomain)
    norders = length(domain.dr_matrices)
    domain.N >= norders + 1 || return domain
    for order in 1:norders
        domain.dr_matrices[order] .= create_derivative_matrix(order, domain).data
    end
    domain.radial_laplacian .= create_radial_laplacian(domain).data
    return domain
end

function apply_∂r!(output::Vector{T},
        matrix::BandedMatrix{T},
        input::Vector{T}) where {T}
    N = matrix.size
    bandwidth = matrix.bandwidth

    fill!(output, zero(T))

    @inbounds for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            if 1 <= band_row <= 2*bandwidth + 1
                output[i] += matrix.data[band_row, j] * input[j]
            end
        end
    end
end

# Matrix-vector multiplication for BandedMatrix
function Base.:*(matrix::BandedMatrix{T}, v::AbstractVector{S}) where {T, S}
    N = matrix.size
    bandwidth = matrix.bandwidth
    R = promote_type(T, S)
    output = zeros(R, N)

    @inbounds for j in 1:min(N, length(v))
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            if 1 <= band_row <= 2*bandwidth + 1
                output[i] += matrix.data[band_row, j] * v[j]
            end
        end
    end

    return output
end

#export BandedMatrix, BandedLU, create_derivative_matrix, create_radial_laplacian,
#       apply_banded_matrix!, factorize_banded, solve_banded!

#end
