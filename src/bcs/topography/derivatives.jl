# ================================================================================
# Boundary Derivative Cache Utilities
# ================================================================================

# Note: `numerics/banded_operators.jl` is loaded after `bcs` in the module.
# hierarchy, so we keep a local banded-matrix apply here to avoid ordering issues.

"""
    __apply_∂r!(output, matrix, input)

Local banded-matrix apply for derivative operators.
"""
function __apply_∂r!(output::Vector{T}, matrix, input::Vector{T}) where {T}
    N = matrix.size
    bandwidth = matrix.bandwidth

    fill!(output, zero(T))

    @inbounds for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            if 1 <= band_row <= 2 * bandwidth + 1
                output[i] += matrix.data[band_row, j] * input[j]
            end
        end
    end

    return output
end

"""
    BoundaryDerivativeCache{T}

Cache of boundary values and radial derivatives for a spectral field.
All values are stored for m >= 0 (SHTnsKit triangular indexing) and
are expanded to negative m using conjugate symmetry.

`config` is retained so reads index modes in the SAME canonical m-major order
the cache was filled in (via `local_spectral_storage_slot` / `get_mode_index`).
"""
struct BoundaryDerivativeCache{T <: AbstractFloat}
    lmax::Int
    mmax::Int
    nlm::Int
    config::Any
    value_inner::Vector{Complex{T}}
    value_outer::Vector{Complex{T}}
    d1_inner::Vector{Complex{T}}
    d1_outer::Vector{Complex{T}}
    d2_inner::Union{Vector{Complex{T}}, Nothing}
    d2_outer::Union{Vector{Complex{T}}, Nothing}
end

"""
    compute_boundary_derivative_cache(field, ∂r, ∂²r, domain) -> BoundaryDerivativeCache

Compute boundary values and radial derivatives for all modes using the
full radial profile (MPI-safe). This is expensive but avoids per-mode
Allreduce inside tight coupling loops.
"""
function compute_boundary_derivative_cache(field,
        ∂r,
        ∂²r,
        domain)
    nlm = field.config.nlm
    lmax = field.config.lmax
    mmax = field.config.mmax
    nr = domain.N

    # Infer element type from field data
    T = eltype(parent(field.data_real))

    values_inner = zeros(Complex{T}, nlm)
    values_outer = zeros(Complex{T}, nlm)
    d1_inner = zeros(Complex{T}, nlm)
    d1_outer = zeros(Complex{T}, nlm)
    d2_inner = ∂²r === nothing ? nothing : zeros(Complex{T}, nlm)
    d2_outer = ∂²r === nothing ? nothing : zeros(Complex{T}, nlm)

    data_real = parent(field.data_real)
    data_imag = parent(field.data_imag)
    r_range = field.pencil.axes_local[3]

    comm = get_comm()

    # Gather every mode's radial profile in ONE collective (per real/imag) instead
    # of 2·nlm per-mode Allreduces. Each rank fills only its owned modes' columns
    # (zeros elsewhere) via local_spectral_storage_slot ownership (nothing ⇒ not
    # held; the old `lm_idx in axes_local[1]` l-slot guard wrongly dropped all m>0
    # modes), then SUM reconstructs the full set on every rank.
    all_real = zeros(T, nr, nlm)
    all_imag = zeros(T, nr, nlm)
    @inbounds for lm_idx in 1:nlm
        slot = local_spectral_storage_slot(field.config, lm_idx)
        slot === nothing && continue
        gather_local_radial_profile!(view(all_real, :, lm_idx), view(all_imag, :, lm_idx),
            data_real, data_imag, slot, r_range)
    end
    if MPI.Comm_size(comm) > 1
        MPI.Allreduce!(all_real, MPI.SUM, comm)
        MPI.Allreduce!(all_imag, MPI.SUM, comm)
    end

    gathered_real = zeros(T, nr)
    gathered_imag = zeros(T, nr)
    dprofile_real = zeros(T, nr)
    dprofile_imag = zeros(T, nr)
    d2profile_real = ∂²r === nothing ? zeros(T, 0) : zeros(T, nr)
    d2profile_imag = ∂²r === nothing ? zeros(T, 0) : zeros(T, nr)

    @inbounds for lm_idx in 1:nlm
        # Copy the gathered column into the reused work vectors (__apply_∂r! needs
        # a contiguous Vector, not a view).
        for r_idx in 1:nr
            gathered_real[r_idx] = all_real[r_idx, lm_idx]
            gathered_imag[r_idx] = all_imag[r_idx, lm_idx]
        end

        values_inner[lm_idx] = complex(gathered_real[1], gathered_imag[1])
        values_outer[lm_idx] = complex(gathered_real[nr], gathered_imag[nr])

        __apply_∂r!(dprofile_real, ∂r, gathered_real)
        __apply_∂r!(dprofile_imag, ∂r, gathered_imag)
        d1_inner[lm_idx] = complex(dprofile_real[1], dprofile_imag[1])
        d1_outer[lm_idx] = complex(dprofile_real[nr], dprofile_imag[nr])

        if ∂²r !== nothing
            __apply_∂r!(d2profile_real, ∂²r, gathered_real)
            __apply_∂r!(d2profile_imag, ∂²r, gathered_imag)
            d2_inner[lm_idx] = complex(d2profile_real[1], d2profile_imag[1])
            d2_outer[lm_idx] = complex(d2profile_real[nr], d2profile_imag[nr])
        end
    end

    return BoundaryDerivativeCache{T}(lmax, mmax, nlm, field.config,
        values_inner, values_outer,
        d1_inner, d1_outer,
        d2_inner, d2_outer)
end

# Canonical m-major mode index of (l, m) for a full triangular layout
# (m in 0:mmax, l in m:lmax) — matches the SHTnsKit config's mode ordering. Used as
# a config-free fallback for directly-built caches (unit tests) that carry no config.
@inline function _canonical_mode_index(l::Int, m::Int, lmax::Int)
    return m * (lmax + 1) - (m * (m - 1)) ÷ 2 + (l - m + 1)
end

function _cache_index(cache::BoundaryDerivativeCache, l::Int, m::Int)
    if l > cache.lmax || abs(m) > l || abs(m) > cache.mmax
        return 0
    end
    # The cache is FILLED in canonical m-major order (compute_boundary_derivative_cache
    # writes values_inner[lm_idx] for lm_idx in 1:nlm via local_spectral_storage_slot).
    # Read with the matching canonical index — lm_to_index is l-major and would scramble
    # modes (e.g. lmax=4: (l=2,m=1) is canonical index 7 but lm_to_index gives 5).
    cfg = cache.config
    cfg === nothing && return _canonical_mode_index(l, abs(m), cache.lmax)
    return get_mode_index(cfg, l, abs(m))
end

function _apply_m_conjugate(val::Complex{T}, m::Int) where {T}
    if m >= 0
        return val
    end
    phase = iseven(-m) ? one(T) : -one(T)
    return phase * conj(val)
end

function get_cache_value(cache::BoundaryDerivativeCache{T},
        l::Int, m::Int,
        location::BoundaryLocation) where {T}
    idx = _cache_index(cache, l, m)
    if idx <= 0 || idx > cache.nlm
        return zero(Complex{T})
    end
    val = location == INNER_BOUNDARY ? cache.value_inner[idx] : cache.value_outer[idx]
    return _apply_m_conjugate(val, m)
end

function get_cache_d1(cache::BoundaryDerivativeCache{T},
        l::Int, m::Int,
        location::BoundaryLocation) where {T}
    idx = _cache_index(cache, l, m)
    if idx <= 0 || idx > cache.nlm
        return zero(Complex{T})
    end
    val = location == INNER_BOUNDARY ? cache.d1_inner[idx] : cache.d1_outer[idx]
    return _apply_m_conjugate(val, m)
end

function get_cache_d2(cache::BoundaryDerivativeCache{T},
        l::Int, m::Int,
        location::BoundaryLocation) where {T}
    cache.d2_inner === nothing && return zero(Complex{T})
    idx = _cache_index(cache, l, m)
    if idx <= 0 || idx > cache.nlm
        return zero(Complex{T})
    end
    val = location == INNER_BOUNDARY ? cache.d2_inner[idx] : cache.d2_outer[idx]
    return _apply_m_conjugate(val, m)
end
