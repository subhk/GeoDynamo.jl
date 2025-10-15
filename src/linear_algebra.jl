    
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

function factorize_banded(A::BandedMatrix{T}) where T
    N  = A.size
    bw = A.bandwidth
    lu = copy(A.data)

    @inbounds for k in 1:N-1
        # Pivot (no pivoting for banded SPD-like operators)
        piv_row = _band_row(k, k, bw)
        if !(1 <= piv_row <= 2*bw+1)
            continue
        end
        piv = lu[piv_row, k]
        # Eliminate entries below pivot within bandwidth
        i_max = min(N, k + bw)
        for i in k+1:i_max
            row = _band_row(i, k, bw)
            if 1 <= row <= 2*bw+1
                L = lu[row, k] / piv
                lu[row, k] = L  # store L below diagonal
                # Update row i for columns within band
                j_max = min(N, k + bw)
                for j in k+1:j_max
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

function solve_banded!(x::Vector{T}, lu::BandedLU{T}, b::Vector{T}) where T
    N  = lu.size
    bw = lu.bandwidth
    # Forward substitution: L y = b (L has unit diagonal)
    @inbounds for i in 1:N
        s = zero(T)
        j_min = max(1, i - bw)
        for j in j_min:i-1
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
        for j in i+1:j_max
            row = _band_row(i, j, bw)
            if 1 <= row <= 2*bw+1
                s += lu.lu[row, j] * x[j]
            end
        end
        diag_row = _band_row(i, i, bw)
        x[i] = (x[i] - s) / lu.lu[diag_row, i]
    end
    return x
end

function create_derivative_matrix(order::Int, domain::RadialDomain)
    # Create finite difference matrix for given derivative order
    N = domain.N
    bandwidth = i_KL
    
    # Initialize banded matrix storage
    data = zeros(2*bandwidth + 1, N)
    
    # Compute finite difference coefficients using Chebyshev points
    for n in 1:N
        left = max(1, n - bandwidth)
        right = min(N, n + bandwidth)
        stencil_size = right - left + 1
        
        # Vandermonde matrix for interpolation
        V = ones(stencil_size, stencil_size)
        points = domain.r[left:right, 4]  # r values
        
        for j in 2:stencil_size
            for i in 1:stencil_size
                V[i, j] = V[i, j-1] * (points[i] - domain.r[n, 4])
            end
        end
        
        # Solve for derivative coefficients
        rhs = zeros(stencil_size)
        if order <= stencil_size
            rhs[order + 1] = factorial(order)
        end
        
        coeffs = V \ rhs
        
        # Store in banded format
        for (i, idx) in enumerate(left:right)
            band_row = bandwidth + 1 + n - idx
            if 1 <= band_row <= 2*bandwidth + 1
                data[band_row, idx] = coeffs[i]
            end
        end
    end
    
    return BandedMatrix(data, bandwidth, N)
end

function create_radial_laplacian(domain::RadialDomain)
    # d²/dr² + (2/r) d/dr
    d2_matrix = create_derivative_matrix(2, domain)
    d1_matrix = create_derivative_matrix(1, domain)
    
    laplacian_data = copy(d2_matrix.data)
    
    # Add (2/r) * d/dr term
    for n in 1:domain.N
        r_inv = domain.r[n, 3]  # 1/r
        for j in max(1, n - i_KL):min(domain.N, n + i_KL)
            band_row = i_KL + 1 + n - j
            laplacian_data[band_row, j] += 2.0 * r_inv * d1_matrix.data[band_row, j]
        end
    end
    
    return BandedMatrix(laplacian_data, i_KL, domain.N)
end

# Apply banded matrix to PencilArray data
function apply_banded_matrix!(output::SHTnsSpectralField{T}, 
                             matrix::BandedMatrix{T}, 
                             input::SHTnsSpectralField{T}) where T
    # Get local data portions
    out_real = parent(output.data_real)
    out_imag = parent(output.data_imag)
    in_real  = parent(input.data_real)
    in_imag  = parent(input.data_imag)
    
    # Get local indices
    local_indices = get_local_indices(input.pencil)
    
    # Apply matrix only to local data
    for idx in CartesianIndices(local_indices)
        if idx[3] <= matrix.size  # Check radial dimension
            # Apply to real part
            apply_banded_vector_local!(view(out_real, idx[1], idx[2], :), 
                                      matrix, 
                                      view(in_real, idx[1], idx[2], :))
            # Apply to imaginary part
            apply_banded_vector_local!(view(out_imag, idx[1], idx[2], :), 
                                      matrix, 
                                      view(in_imag, idx[1], idx[2], :))
        end
    end

    return nothing
end

# Helper function to apply derivative matrix
function apply_banded_vector_local!(output::AbstractVector{T}, 
                                   matrix::BandedMatrix{T}, 
                                   input::AbstractVector{T}) where T
    fill!(output, zero(T))
    N = min(matrix.size, length(input))
    bandwidth = matrix.bandwidth
    
    for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            if i <= length(output)
                band_row = bandwidth + 1 + i - j
                output[i] += matrix.data[band_row, j] * input[j]
            end
        end
    end

    return nothing
end

function apply_derivative_matrix!(output::Vector{T}, 
                                matrix::BandedMatrix{T}, 
                                input::Vector{T}) where T
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


#export BandedMatrix, BandedLU, create_derivative_matrix, create_radial_laplacian,
#       apply_banded_matrix!, factorize_banded, solve_banded!


#end
