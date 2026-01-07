# ================================================================================
# Gaunt Tensor Computation for Topography Coupling (SHTnsKit-based)
# ================================================================================
#
# This file implements the computation of Gaunt-type tensor integrals needed
# for spherical harmonic coupling due to boundary topography, using SHTnsKit
# for efficient spherical harmonic transforms.
#
# Basic Gaunt integral:
#   G_{ℓm,ℓ'm',LM} = ∫ Y_ℓ^{m*} Y_{ℓ'}^{m'} Y_L^M dΩ
#
# Gradient Gaunt integral (for slope coupling):
#   G^{(∇)}_{ℓm,ℓ'm',LM} = ∫ Y_ℓ^{m*} ∇_H Y_{ℓ'}^{m'} · ∇_H Y_L^M dΩ
#
# Cross Gaunt integral (for toroidal coupling):
#   G^{(×)}_{ℓm,ℓ'm',LM} = ∫ Y_ℓ^{m*} r̂ · (∇_H Y_{ℓ'}^{m'} × ∇_H Y_L^M) dΩ
#
# The gradient Gaunt can be computed efficiently using the identity:
#   G^{(∇)} = (1/2)[ℓ'(ℓ'+1) + L(L+1) - ℓ(ℓ+1)] G
#
# ================================================================================

using LinearAlgebra

# ================================================================================
# Gaunt Tensor Cache Structure
# ================================================================================

"""
    GauntTensorCache{T}

Cache structure for precomputed Gaunt tensors using SHTnsKit for efficiency.

Stores sparse representations of Gaunt tensor elements for efficient lookup
during boundary condition application.

# Fields
- `lmax::Int`: Maximum spherical harmonic degree
- `lmax_topo::Int`: Maximum degree for topography
- `G::Dict`: Basic Gaunt integrals G_{ℓm,ℓ'm',LM}
- `G_grad::Dict`: Gradient Gaunt integrals G^{(∇)}
- `G_cross::Dict`: Cross Gaunt integrals G^{(×)}
- `sht_config`: SHTnsKit configuration for transforms
- `is_precomputed::Bool`: Whether tensors have been precomputed
"""
mutable struct GauntTensorCache{T<:AbstractFloat}
    lmax::Int
    lmax_topo::Int
    G::Dict{NTuple{6,Int}, T}           # (ℓ,m,ℓ',m',L,M) -> value
    G_grad::Dict{NTuple{6,Int}, T}      # Gradient Gaunt
    G_cross::Dict{NTuple{6,Int}, T}     # Cross Gaunt
    sht_config::Any                      # SHTnsKit configuration
    nlat::Int                            # Number of latitude points
    nlon::Int                            # Number of longitude points
    theta::Vector{T}                     # Gauss-Legendre theta points
    weights::Vector{T}                   # Gauss-Legendre weights
    is_precomputed::Bool
end

"""
    GauntTensorCache{T}(lmax::Int, lmax_topo::Int; nth::Int=0, nph::Int=0) where T

Create a Gaunt tensor cache with SHTnsKit configuration.

# Arguments
- `lmax::Int`: Maximum spherical harmonic degree for fields
- `lmax_topo::Int`: Maximum degree for topography
- `nth::Int`: Number of theta points (default: 2*max(lmax,lmax_topo) + 2)
- `nph::Int`: Number of phi points (default: 2*nth)
"""
function GauntTensorCache{T}(lmax::Int, lmax_topo::Int; nth::Int=0, nph::Int=0) where T
    # Set default quadrature resolution (need enough points for triple products)
    # For triple product of Y_l1^m1 * Y_l2^m2 * Y_L^M, need at least 3*lmax points
    lmax_total = lmax + lmax_topo
    nth = nth > 0 ? nth : 3 * lmax_total ÷ 2 + 2
    nph = nph > 0 ? nph : 2 * nth

    # Create SHTnsKit configuration for the quadrature grid
    # Using lmax_total to ensure we can represent all products
    sht_config = SHTnsKit.create_gauss_config(lmax_total, nth; mmax=lmax_total, nlon=nph)

    # Get Gauss-Legendre points and weights from SHTnsKit
    theta, weights = _get_gauss_legendre_from_shtns(sht_config, nth)

    return GauntTensorCache{T}(
        lmax, lmax_topo,
        Dict{NTuple{6,Int}, T}(),
        Dict{NTuple{6,Int}, T}(),
        Dict{NTuple{6,Int}, T}(),
        sht_config,
        nth, nph,
        theta, weights,
        false
    )
end

# Convenience constructor with default Float64
GauntTensorCache(lmax::Int, lmax_topo::Int; kwargs...) =
    GauntTensorCache{Float64}(lmax, lmax_topo; kwargs...)

"""
    _get_gauss_legendre_from_shtns(sht_config, nlat)

Extract Gauss-Legendre quadrature points and weights.
SHTnsKit uses Gauss-Legendre points internally.
"""
function _get_gauss_legendre_from_shtns(sht_config, nlat::Int)
    # Compute Gauss-Legendre nodes and weights
    # These are the same points SHTnsKit uses internally
    theta = zeros(Float64, nlat)
    weights = zeros(Float64, nlat)

    for i in 1:nlat
        # SHTnsKit uses cos(θ) as the Legendre argument
        # Gauss-Legendre nodes for P_n on [-1, 1]
        mu, w = _gauss_legendre_point(nlat, i)
        theta[i] = acos(mu)  # Convert to colatitude
        weights[i] = w
    end

    return theta, weights
end

"""
    _gauss_legendre_point(n, i)

Compute the i-th Gauss-Legendre quadrature point and weight for n points.
"""
function _gauss_legendre_point(n::Int, i::Int)
    # Initial guess using Chebyshev approximation
    z = cos(π * (i - 0.25) / (n + 0.5))

    # Newton iteration for Legendre polynomial root
    for _ in 1:50
        p1 = 1.0
        p2 = 0.0

        for j in 1:n
            p3 = p2
            p2 = p1
            p1 = ((2j - 1) * z * p2 - (j - 1) * p3) / j
        end

        # Derivative: P'_n(z) = n(zP_n - P_{n-1})/(z² - 1)
        pp = n * (z * p1 - p2) / (z^2 - 1)

        z_old = z
        z = z - p1 / pp

        if abs(z - z_old) < 1e-15
            break
        end
    end

    # Weight
    pp = n * (z * 1.0 - 0.0)  # Recompute derivative at root
    p1 = 1.0
    p2 = 0.0
    for j in 1:n
        p3 = p2
        p2 = p1
        p1 = ((2j - 1) * z * p2 - (j - 1) * p3) / j
    end
    pp = n * (z * p1 - p2) / (z^2 - 1)
    w = 2.0 / ((1 - z^2) * pp^2)

    return (z, w)
end

# ================================================================================
# SHTnsKit-based Spherical Harmonic Evaluation
# ================================================================================

"""
    evaluate_spherical_harmonics_grid(l::Int, m::Int, cache::GauntTensorCache{T}) where T

Evaluate Y_l^m on the Gauss-Legendre × uniform φ grid using SHTnsKit.

Returns a (nlat, nlon) matrix of complex values.
"""
function evaluate_spherical_harmonics_grid(l::Int, m::Int, cache::GauntTensorCache{T}) where T
    if l > cache.lmax + cache.lmax_topo || abs(m) > l
        return zeros(Complex{T}, cache.nlat, cache.nlon)
    end

    # Create spectral coefficients with only (l,m) mode set to 1
    lmax_total = cache.lmax + cache.lmax_topo
    coeffs = zeros(Complex{Float64}, lmax_total + 1, lmax_total + 1)

    mabs = abs(m)
    coeffs[l + 1, mabs + 1] = 1.0

    # Use SHTnsKit synthesis to get Y_l^m on the grid
    ylm_grid = SHTnsKit.synthesis(cache.sht_config, coeffs; real_output=false)
    ylm = Complex{T}.(ylm_grid)
    if m < 0
        phase = iseven(mabs) ? one(T) : -one(T)
        return phase .* conj.(ylm)
    end
    return ylm
end

"""
    evaluate_spherical_harmonic_gradient_grid(l::Int, m::Int, cache::GauntTensorCache{T}) where T

Evaluate ∇_H Y_l^m on the Gauss-Legendre × uniform φ grid using SHTnsKit.

Returns (grad_theta, grad_phi) matrices of complex values.
"""
function evaluate_spherical_harmonic_gradient_grid(l::Int, m::Int, cache::GauntTensorCache{T}) where T
    if l > cache.lmax + cache.lmax_topo || abs(m) > l || l == 0
        zero_grid = zeros(Complex{T}, cache.nlat, cache.nlon)
        return (zero_grid, zero_grid)
    end

    # Create spectral coefficients with only (l,m) mode set to 1
    lmax_total = cache.lmax + cache.lmax_topo
    coeffs = zeros(Complex{Float64}, lmax_total + 1, lmax_total + 1)

    mabs = abs(m)
    coeffs[l + 1, mabs + 1] = 1.0

    # Use SHTnsKit's gradient function if available
    try
        grad_theta, grad_phi = SHTnsKit.SH_to_grad_spat(cache.sht_config, coeffs; real_output=false)
        grad_theta_t = Complex{T}.(grad_theta)
        grad_phi_t = Complex{T}.(grad_phi)
        if m < 0
            phase = iseven(mabs) ? one(T) : -one(T)
            return (phase .* conj.(grad_theta_t), phase .* conj.(grad_phi_t))
        end
        return (grad_theta_t, grad_phi_t)
    catch e
        # Fallback: compute gradient numerically
        return _compute_gradient_fallback(l, m, cache)
    end
end

"""
    _compute_gradient_fallback(l, m, cache)

Fallback gradient computation using finite differences on the Y_l^m grid.
"""
function _compute_gradient_fallback(l::Int, m::Int, cache::GauntTensorCache{T}) where T
    ylm = evaluate_spherical_harmonics_grid(l, m, cache)

    nlat, nlon = cache.nlat, cache.nlon
    theta = cache.theta
    dphi = 2π / nlon

    grad_theta = zeros(Complex{T}, nlat, nlon)
    grad_phi = zeros(Complex{T}, nlat, nlon)

    # ∂Y/∂θ using central differences
    for j in 1:nlon
        for i in 2:nlat-1
            grad_theta[i, j] = (ylm[i+1, j] - ylm[i-1, j]) / (theta[i+1] - theta[i-1])
        end
        # One-sided at boundaries
        grad_theta[1, j] = (ylm[2, j] - ylm[1, j]) / (theta[2] - theta[1])
        grad_theta[nlat, j] = (ylm[nlat, j] - ylm[nlat-1, j]) / (theta[nlat] - theta[nlat-1])
    end

    # (1/sin θ) ∂Y/∂φ using central differences with periodic boundary
    for i in 1:nlat
        sin_theta = sin(theta[i])
        if abs(sin_theta) < 1e-10
            continue
        end
        for j in 1:nlon
            jp1 = j == nlon ? 1 : j + 1
            jm1 = j == 1 ? nlon : j - 1
            grad_phi[i, j] = (ylm[i, jp1] - ylm[i, jm1]) / (2 * dphi * sin_theta)
        end
    end

    return (grad_theta, grad_phi)
end

# ================================================================================
# Gaunt Tensor Computation using SHTnsKit Grids
# ================================================================================

"""
    compute_gaunt_tensor(l1::Int, m1::Int, l2::Int, m2::Int, L::Int, M::Int,
                         cache::GauntTensorCache{T}) where T

Compute the basic Gaunt integral using SHTnsKit for spherical harmonic evaluation:
G_{l1,m1,l2,m2,L,M} = ∫ Y_{l1}^{m1*} Y_{l2}^{m2} Y_L^M dΩ

Uses numerical quadrature on Gauss-Legendre × uniform φ grid.
"""
function compute_gaunt_tensor(l1::Int, m1::Int, l2::Int, m2::Int, L::Int, M::Int,
                              cache::GauntTensorCache{T}) where T
    # Selection rules (quick rejection)
    # Azimuthal: m1 = m2 + M
    if m1 != m2 + M
        return zero(T)
    end

    # Triangle inequality
    if abs(l1 - l2) > L || l1 + l2 < L
        return zero(T)
    end

    # Parity rule
    if isodd(l1 + l2 + L)
        return zero(T)
    end

    # Evaluate spherical harmonics on grid
    Y1 = evaluate_spherical_harmonics_grid(l1, m1, cache)
    Y2 = evaluate_spherical_harmonics_grid(l2, m2, cache)
    YL = evaluate_spherical_harmonics_grid(L, M, cache)

    # Numerical integration using Gauss-Legendre quadrature
    nlat, nlon = cache.nlat, cache.nlon
    dphi = 2π / nlon
    theta = cache.theta
    weights = cache.weights

    result = zero(Complex{T})

    for i in 1:nlat
        w = weights[i]

        for j in 1:nlon
            # Y_l1^{m1*} * Y_l2^{m2} * Y_L^M
            integrand = conj(Y1[i, j]) * Y2[i, j] * YL[i, j]
            result += w * dphi * integrand
        end
    end

    # Gaunt integrals are real for real spherical harmonics
    return T(real(result))
end

"""
    compute_gradient_gaunt_tensor(l1::Int, m1::Int, l2::Int, m2::Int, L::Int, M::Int,
                                  cache::GauntTensorCache{T}) where T

Compute the gradient Gaunt integral using the analytic identity:
G^{(∇)}_{l1,m1,l2,m2,L,M} = (1/2)[ℓ2(ℓ2+1) + L(L+1) - ℓ1(ℓ1+1)] G_{l1,m1,l2,m2,L,M}

This is more efficient and accurate than numerical gradient computation.
"""
function compute_gradient_gaunt_tensor(l1::Int, m1::Int, l2::Int, m2::Int, L::Int, M::Int,
                                       cache::GauntTensorCache{T}) where T
    # Use analytic identity relating gradient Gaunt to basic Gaunt
    # ∫ Y_1^* (∇_H Y_2 · ∇_H Y_L) dΩ = (1/2)[l2(l2+1) + L(L+1) - l1(l1+1)] G

    # First compute basic Gaunt
    G = compute_gaunt_tensor(l1, m1, l2, m2, L, M, cache)

    if abs(G) < 1e-15
        return zero(T)
    end

    # Gradient of l=0 is zero
    if l2 == 0 || L == 0
        return zero(T)
    end

    # Apply the identity
    factor = T(0.5) * (l2 * (l2 + 1) + L * (L + 1) - l1 * (l1 + 1))

    return factor * G
end

"""
    compute_cross_gaunt_tensor(l1::Int, m1::Int, l2::Int, m2::Int, L::Int, M::Int,
                               cache::GauntTensorCache{T}) where T

Compute the cross Gaunt integral:
G^{(×)}_{l1,m1,l2,m2,L,M} = ∫ Y_{l1}^{m1*} r̂ · (∇_H Y_{l2}^{m2} × ∇_H Y_L^M) dΩ

This integral couples toroidal and poloidal modes through topography.
Uses SHTnsKit for gradient evaluation when available.
"""
function compute_cross_gaunt_tensor(l1::Int, m1::Int, l2::Int, m2::Int, L::Int, M::Int,
                                    cache::GauntTensorCache{T}) where T
    # Azimuthal selection rule
    if m1 != m2 + M
        return zero(T)
    end

    # Zero if either gradient is zero
    if l2 == 0 || L == 0
        return zero(T)
    end

    # Evaluate Y_1^* on grid
    Y1 = evaluate_spherical_harmonics_grid(l1, m1, cache)

    # Evaluate gradients
    grad2_theta, grad2_phi = evaluate_spherical_harmonic_gradient_grid(l2, m2, cache)
    gradL_theta, gradL_phi = evaluate_spherical_harmonic_gradient_grid(L, M, cache)

    # Numerical integration
    nlat, nlon = cache.nlat, cache.nlon
    dphi = 2π / nlon
    theta = cache.theta
    weights = cache.weights

    result = zero(Complex{T})

    for i in 1:nlat
        sin_theta = sin(theta[i])
        if abs(sin_theta) < 1e-10
            continue
        end
        w = weights[i]

        for j in 1:nlon
            # Cross product in spherical coords:
            # r̂ · (A × B) = A_θ B_φ - A_φ B_θ  (for surface vectors)
            cross_r = grad2_theta[i, j] * gradL_phi[i, j] - grad2_phi[i, j] * gradL_theta[i, j]

            integrand = conj(Y1[i, j]) * cross_r
            result += w * dphi * integrand
        end
    end

    # Cross product gives imaginary values for real spherical harmonics
    return T(imag(result))
end

# ================================================================================
# Alternative: Gaunt from Wigner 3j Symbols (Analytic)
# ================================================================================

"""
    compute_gaunt_from_wigner3j(l1::Int, m1::Int, l2::Int, m2::Int, L::Int, M::Int)

Compute Gaunt coefficient using Wigner 3j symbols (analytic formula).

G = sqrt((2l1+1)(2l2+1)(2L+1)/(4π)) * (l1 l2 L; 0 0 0) * (l1 l2 L; -m1 m2 M)

This is faster than numerical integration for production use.
"""
function compute_gaunt_from_wigner3j(l1::Int, m1::Int, l2::Int, m2::Int, L::Int, M::Int)
    # Selection rules
    if -m1 + m2 + M != 0
        return 0.0
    end
    if abs(l1 - l2) > L || l1 + l2 < L
        return 0.0
    end
    if isodd(l1 + l2 + L)
        return 0.0
    end

    # Compute prefactor
    prefactor = sqrt((2l1 + 1) * (2l2 + 1) * (2L + 1) / (4π))

    # Compute Wigner 3j symbols
    w3j_000 = _wigner_3j(l1, l2, L, 0, 0, 0)
    w3j_mmM = _wigner_3j(l1, l2, L, -m1, m2, M)

    return prefactor * w3j_000 * w3j_mmM
end

"""
    _wigner_3j(l1, l2, l3, m1, m2, m3)

Compute Wigner 3j symbol using Racah formula.
"""
function _wigner_3j(l1::Int, l2::Int, l3::Int, m1::Int, m2::Int, m3::Int)
    # Selection rules
    if m1 + m2 + m3 != 0
        return 0.0
    end
    if abs(m1) > l1 || abs(m2) > l2 || abs(m3) > l3
        return 0.0
    end
    if l1 + l2 < l3 || abs(l1 - l2) > l3
        return 0.0
    end

    # Use Racah formula with BigInt for factorials
    J = l1 + l2 + l3

    # Triangle coefficient
    tri = sqrt(factorial(big(l1 + l2 - l3)) * factorial(big(l1 - l2 + l3)) *
               factorial(big(-l1 + l2 + l3)) / factorial(big(J + 1)))

    # Row factors
    r1 = sqrt(factorial(big(l1 - m1)) * factorial(big(l1 + m1)))
    r2 = sqrt(factorial(big(l2 - m2)) * factorial(big(l2 + m2)))
    r3 = sqrt(factorial(big(l3 - m3)) * factorial(big(l3 + m3)))

    # Sum over k
    kmin = max(0, l2 - l3 - m1, l1 - l3 + m2)
    kmax = min(l1 + l2 - l3, l1 - m1, l2 + m2)

    sum_val = big(0.0)
    for k in kmin:kmax
        num = big((-1)^k)
        den = factorial(big(k)) *
              factorial(big(l1 + l2 - l3 - k)) *
              factorial(big(l1 - m1 - k)) *
              factorial(big(l2 + m2 - k)) *
              factorial(big(l3 - l2 + m1 + k)) *
              factorial(big(l3 - l1 - m2 + k))
        sum_val += num / den
    end

    phase = iseven(l1 - l2 - m3) ? 1 : -1

    return Float64(phase * tri * r1 * r2 * r3 * sum_val)
end

# ================================================================================
# Precomputation and Caching
# ================================================================================

"""
    precompute_gaunt_tensors!(cache::GauntTensorCache{T};
                              verbose::Bool=true, use_wigner::Bool=true) where T

Precompute all non-zero Gaunt tensors up to lmax.

# Arguments
- `verbose::Bool=true`: Print progress information
- `use_wigner::Bool=true`: Use analytic Wigner 3j formula (faster and more accurate)
"""
function precompute_gaunt_tensors!(cache::GauntTensorCache{T};
                                   verbose::Bool=true,
                                   use_wigner::Bool=true) where T
    lmax = cache.lmax
    lmax_t = cache.lmax_topo

    if verbose && get_rank() == 0
        @info "Precomputing Gaunt tensors" lmax=lmax lmax_topo=lmax_t method=(use_wigner ? "Wigner 3j" : "SHTnsKit numerical")
    end

    count_G = 0
    count_grad = 0
    count_cross = 0

    # Iterate over all mode combinations
    for l1 in 0:lmax
        for m1 in -l1:l1
            for l2 in 0:lmax
                for m2 in -l2:l2
                    for L in 0:lmax_t
                        # M is determined by selection rule m1 = m2 + M
                        M = m1 - m2
                        if abs(M) > L
                            continue
                        end

                        # Compute basic Gaunt
                        if use_wigner
                            G_val = T(compute_gaunt_from_wigner3j(l1, m1, l2, m2, L, M))
                        else
                            G_val = compute_gaunt_tensor(l1, m1, l2, m2, L, M, cache)
                        end

                        if abs(G_val) > 1e-14
                            cache.G[(l1, m1, l2, m2, L, M)] = G_val
                            count_G += 1

                            # Gradient Gaunt from identity (always use analytic)
                            if l2 > 0 && L > 0
                                factor = T(0.5) * (l2 * (l2 + 1) + L * (L + 1) - l1 * (l1 + 1))
                                G_grad_val = factor * G_val
                                if abs(G_grad_val) > 1e-14
                                    cache.G_grad[(l1, m1, l2, m2, L, M)] = G_grad_val
                                    count_grad += 1
                                end
                            end
                        end

                        # Cross Gaunt (requires numerical integration)
                        if !use_wigner && l2 > 0 && L > 0
                            G_cross_val = compute_cross_gaunt_tensor(l1, m1, l2, m2, L, M, cache)
                            if abs(G_cross_val) > 1e-14
                                cache.G_cross[(l1, m1, l2, m2, L, M)] = G_cross_val
                                count_cross += 1
                            end
                        end
                    end
                end
            end
        end
    end

    cache.is_precomputed = true

    if verbose && get_rank() == 0
        @info "Gaunt tensors precomputed" G_nonzero=count_G G_grad_nonzero=count_grad G_cross_nonzero=count_cross
    end

    return cache
end

# ================================================================================
# Cache Accessors
# ================================================================================

"""
    get_gaunt_tensor(cache::GauntTensorCache{T}, l1, m1, l2, m2, L, M) where T

Get the basic Gaunt tensor G_{l1,m1,l2,m2,L,M} from cache.
Returns 0 if not found (implies zero due to selection rules).
"""
function get_gaunt_tensor(cache::GauntTensorCache{T}, l1::Int, m1::Int,
                          l2::Int, m2::Int, L::Int, M::Int) where T
    return get(cache.G, (l1, m1, l2, m2, L, M), zero(T))
end

"""
    get_gradient_gaunt(cache::GauntTensorCache{T}, l1, m1, l2, m2, L, M) where T

Get the gradient Gaunt tensor G^{(∇)}_{l1,m1,l2,m2,L,M} from cache.
"""
function get_gradient_gaunt(cache::GauntTensorCache{T}, l1::Int, m1::Int,
                            l2::Int, m2::Int, L::Int, M::Int) where T
    return get(cache.G_grad, (l1, m1, l2, m2, L, M), zero(T))
end

"""
    get_cross_gaunt(cache::GauntTensorCache{T}, l1, m1, l2, m2, L, M) where T

Get the cross Gaunt tensor G^{(×)}_{l1,m1,l2,m2,L,M} from cache.
"""
function get_cross_gaunt(cache::GauntTensorCache{T}, l1::Int, m1::Int,
                         l2::Int, m2::Int, L::Int, M::Int) where T
    key = (l1, m1, l2, m2, L, M)
    if haskey(cache.G_cross, key)
        return cache.G_cross[key]
    end

    # Lazy compute when missing (needed if precompute ran with use_wigner=true).
    if l2 == 0 || L == 0 || m1 != m2 + M
        return zero(T)
    end

    val = compute_cross_gaunt_tensor(l1, m1, l2, m2, L, M, cache)
    if abs(val) > 1e-14
        cache.G_cross[key] = val
    end
    return val
end

# ================================================================================
# Utility: On-the-fly Gaunt Computation (for modes not in cache)
# ================================================================================

"""
    gaunt_on_the_fly(l1, m1, l2, m2, L, M; use_wigner=true)

Compute a single Gaunt coefficient on-the-fly.
Useful when only a few coefficients are needed.
"""
function gaunt_on_the_fly(l1::Int, m1::Int, l2::Int, m2::Int, L::Int, M::Int;
                          use_wigner::Bool=true)
    if use_wigner
        return compute_gaunt_from_wigner3j(l1, m1, l2, m2, L, M)
    else
        # Create temporary cache for numerical computation
        lmax_total = max(l1, l2, L)
        cache = GauntTensorCache{Float64}(lmax_total, lmax_total)
        return compute_gaunt_tensor(l1, m1, l2, m2, L, M, cache)
    end
end

"""
    gradient_gaunt_from_basic(l1::Int, l2::Int, L::Int, G_basic::T) where T

Compute gradient Gaunt using the identity:
G^{(∇)} = (1/2)[ℓ2(ℓ2+1) + L(L+1) - ℓ1(ℓ1+1)] G

This avoids computing gradients explicitly.
"""
function gradient_gaunt_from_basic(l1::Int, l2::Int, L::Int, G_basic::T) where T
    if abs(G_basic) < 1e-15
        return zero(T)
    end
    factor = T(0.5) * (l2 * (l2 + 1) + L * (L + 1) - l1 * (l1 + 1))
    return factor * G_basic
end
