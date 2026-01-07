# ================================================================================
# Boundary Derivative Cache Utilities
# ================================================================================

import ...apply_derivative_matrix!

"""
    BoundaryDerivativeCache{T}

Cache of boundary values and radial derivatives for a spectral field.
All values are stored for m >= 0 (SHTnsKit triangular indexing) and
are expanded to negative m using conjugate symmetry.
"""
struct BoundaryDerivativeCache{T<:AbstractFloat}
    lmax::Int
    mmax::Int
    nlm::Int
    value_inner::Vector{Complex{T}}
    value_outer::Vector{Complex{T}}
    d1_inner::Vector{Complex{T}}
    d1_outer::Vector{Complex{T}}
    d2_inner::Union{Vector{Complex{T}}, Nothing}
    d2_outer::Union{Vector{Complex{T}}, Nothing}
end

"""
    compute_boundary_derivative_cache(field, dr_matrix, d2r_matrix, domain) -> BoundaryDerivativeCache

Compute boundary values and radial derivatives for all modes using the
full radial profile (MPI-safe). This is expensive but avoids per-mode
Allreduce inside tight coupling loops.
"""
function compute_boundary_derivative_cache(field::SHTnsSpectralField{T},
                                           dr_matrix,
                                           d2r_matrix,
                                           domain) where T
    nlm = field.config.nlm
    lmax = field.config.lmax
    mmax = field.config.mmax
    nr = domain.N

    values_inner = zeros(Complex{T}, nlm)
    values_outer = zeros(Complex{T}, nlm)
    d1_inner = zeros(Complex{T}, nlm)
    d1_outer = zeros(Complex{T}, nlm)
    d2_inner = d2r_matrix === nothing ? nothing : zeros(Complex{T}, nlm)
    d2_outer = d2r_matrix === nothing ? nothing : zeros(Complex{T}, nlm)

    data_real = parent(field.data_real)
    data_imag = parent(field.data_imag)
    lm_range = field.pencil.axes_local[1]
    r_range = field.pencil.axes_local[3]

    comm = get_comm()

    profile_real = zeros(T, nr)
    profile_imag = zeros(T, nr)
    gathered_real = zeros(T, nr)
    gathered_imag = zeros(T, nr)
    dprofile_real = zeros(T, nr)
    dprofile_imag = zeros(T, nr)
    d2profile_real = d2r_matrix === nothing ? zeros(T, 0) : zeros(T, nr)
    d2profile_imag = d2r_matrix === nothing ? zeros(T, 0) : zeros(T, nr)

    for lm_idx in 1:nlm
        fill!(profile_real, zero(T))
        fill!(profile_imag, zero(T))

        if lm_idx in lm_range
            local_lm = lm_idx - first(lm_range) + 1
            @inbounds for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(data_real, 3) && r_idx <= nr
                    profile_real[r_idx] = data_real[local_lm, 1, local_r]
                    profile_imag[r_idx] = data_imag[local_lm, 1, local_r]
                end
            end
        end

        MPI.Allreduce!(profile_real, gathered_real, MPI.SUM, comm)
        MPI.Allreduce!(profile_imag, gathered_imag, MPI.SUM, comm)

        values_inner[lm_idx] = complex(gathered_real[1], gathered_imag[1])
        values_outer[lm_idx] = complex(gathered_real[nr], gathered_imag[nr])

        apply_derivative_matrix!(dprofile_real, dr_matrix, gathered_real)
        apply_derivative_matrix!(dprofile_imag, dr_matrix, gathered_imag)
        d1_inner[lm_idx] = complex(dprofile_real[1], dprofile_imag[1])
        d1_outer[lm_idx] = complex(dprofile_real[nr], dprofile_imag[nr])

        if d2r_matrix !== nothing
            apply_derivative_matrix!(d2profile_real, d2r_matrix, gathered_real)
            apply_derivative_matrix!(d2profile_imag, d2r_matrix, gathered_imag)
            d2_inner[lm_idx] = complex(d2profile_real[1], d2profile_imag[1])
            d2_outer[lm_idx] = complex(d2profile_real[nr], d2profile_imag[nr])
        end
    end

    return BoundaryDerivativeCache{T}(lmax, mmax, nlm,
                                      values_inner, values_outer,
                                      d1_inner, d1_outer,
                                      d2_inner, d2_outer)
end

function _cache_index(cache::BoundaryDerivativeCache, l::Int, m::Int)
    if l > cache.lmax || abs(m) > l || abs(m) > cache.mmax
        return 0
    end
    return lm_to_index(l, abs(m), cache.lmax)
end

function _apply_m_conjugate(val::Complex{T}, m::Int) where T
    if m >= 0
        return val
    end
    phase = iseven(-m) ? one(T) : -one(T)
    return phase * conj(val)
end

function get_cache_value(cache::BoundaryDerivativeCache{T},
                         l::Int, m::Int,
                         location::BoundaryLocation) where T
    idx = _cache_index(cache, l, m)
    if idx <= 0 || idx > cache.nlm
        return zero(Complex{T})
    end
    val = location == INNER_BOUNDARY ? cache.value_inner[idx] : cache.value_outer[idx]
    return _apply_m_conjugate(val, m)
end

function get_cache_d1(cache::BoundaryDerivativeCache{T},
                      l::Int, m::Int,
                      location::BoundaryLocation) where T
    idx = _cache_index(cache, l, m)
    if idx <= 0 || idx > cache.nlm
        return zero(Complex{T})
    end
    val = location == INNER_BOUNDARY ? cache.d1_inner[idx] : cache.d1_outer[idx]
    return _apply_m_conjugate(val, m)
end

function get_cache_d2(cache::BoundaryDerivativeCache{T},
                      l::Int, m::Int,
                      location::BoundaryLocation) where T
    cache.d2_inner === nothing && return zero(Complex{T})
    idx = _cache_index(cache, l, m)
    if idx <= 0 || idx > cache.nlm
        return zero(Complex{T})
    end
    val = location == INNER_BOUNDARY ? cache.d2_inner[idx] : cache.d2_outer[idx]
    return _apply_m_conjugate(val, m)
end
