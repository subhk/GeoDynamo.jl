"""
SPECTRAL VERSION: Proper divergence computation for pressure RHS using SHTnsKit

This file shows the correct way to compute ∇·F for the pressure Poisson equation
using spectral methods for angular derivatives (more accurate than finite differences).

The key issue: Forces are NOT solenoidal, so toroidal-poloidal decomposition
doesn't properly capture their divergence. We compute divergence directly.

Methods provided:
1. compute_divergence_spectral_angular! - Uses SHTnsKit for angular derivatives (RECOMMENDED)
2. compute_divergence_physical! - Uses finite differences (fallback)
"""

using LinearAlgebra
using Statistics

# Try to load SHTnsKit for spectral methods
const HAS_SHTNSKIT = try
    @eval using SHTnsKit
    true
catch
    false
end

# =============================================================================
# SPECTRAL METHOD: Angular derivatives via SHTnsKit (RECOMMENDED)
# =============================================================================

"""
    compute_divergence_spectral_angular!(div_field, vector_field, domain, config)

Compute divergence using SPECTRAL methods for angular derivatives.

For a vector field V = (V_r, V_θ, V_φ) in spherical coordinates:
    ∇·V = (1/r²)∂(r²V_r)/∂r + (1/(r·sinθ))∂(sinθ·V_θ)/∂θ + (1/(r·sinθ))∂V_φ/∂φ

Implementation:
- Term 1 (radial): Finite differences (standard practice in spectral codes)
- Term 2 (θ): Spectral derivative using SHTnsKit
- Term 3 (φ): Spectral derivative (multiply by im*m in spectral space)

This is MORE ACCURATE than finite differences for angular terms because:
- Spectral derivatives are exact to machine precision
- No truncation error from finite difference stencils
- Properly handles pole singularities via spherical harmonic basis
"""
function compute_divergence_spectral_angular!(div_data::AbstractArray{T,3},
                                               V_r::AbstractArray{T,3},
                                               V_θ::AbstractArray{T,3},
                                               V_φ::AbstractArray{T,3},
                                               r_grid::AbstractVector{<:Real},
                                               theta_grid::AbstractVector{<:Real},
                                               sht_config) where T

    HAS_SHTNSKIT || error("SHTnsKit required for spectral divergence computation")

    nlat, nlon, nr = size(V_r)
    lmax = sht_config.lmax
    mmax = sht_config.mmax

    # Precompute trigonometric factors
    sinθ = sin.(theta_grid)
    cosθ = cos.(theta_grid)
    sinθ_inv = 1.0 ./ max.(sinθ, 1e-12)

    # Process each radial shell
    for k in 1:nr
        r = r_grid[k]
        r_inv = 1.0 / r
        r_inv2 = r_inv^2

        # Extract 2D slices at this radius
        Vr_slice = @view V_r[:, :, k]
        Vθ_slice = @view V_θ[:, :, k]
        Vφ_slice = @view V_φ[:, :, k]

        # =====================================================================
        # TERM 1: (1/r²)∂(r²V_r)/∂r = (2/r)V_r + ∂V_r/∂r
        # Use finite differences for radial derivative (standard practice)
        # =====================================================================
        dVr_dr = zeros(T, nlat, nlon)
        if nr > 1
            if k == 1
                # Forward difference at inner boundary
                dr = r_grid[k+1] - r
                @. dVr_dr = (V_r[:, :, k+1] - Vr_slice) / dr
            elseif k == nr
                # Backward difference at outer boundary
                dr = r - r_grid[k-1]
                @. dVr_dr = (Vr_slice - V_r[:, :, k-1]) / dr
            else
                # Central difference in interior
                dr = r_grid[k+1] - r_grid[k-1]
                @. dVr_dr = (V_r[:, :, k+1] - V_r[:, :, k-1]) / dr
            end
        end
        term1 = @. r_inv2 * (2.0 * r * Vr_slice + r^2 * dVr_dr)

        # =====================================================================
        # TERM 2: (1/(r·sinθ))∂(sinθ·V_θ)/∂θ
        # Use spectral method: compute sinθ*V_θ, then ∂/∂θ spectrally
        # =====================================================================
        # Compute sinθ * V_θ in physical space
        sinθ_Vθ = zeros(T, nlat, nlon)
        for j in 1:nlat
            @. sinθ_Vθ[j, :] = sinθ[j] * Vθ_slice[j, :]
        end

        # Transform to spectral space
        sinθ_Vθ_lm = SHTnsKit.analysis(sht_config, sinθ_Vθ)

        # Apply ∂/∂θ operator spectrally
        # For scalar field f expanded as f = Σ f_lm Y_lm:
        # ∂f/∂θ involves coupling between (l,m) and (l±1,m) modes
        # We use SHTnsKit's gradient function which returns (∂f/∂θ, (1/sinθ)∂f/∂φ)
        d_sinθVθ_dθ, _ = SHTnsKit.SH_to_grad_spat(sht_config, sinθ_Vθ_lm)

        # Build term 2: (1/(r sinθ)) * d(sinθ V_θ)/dθ
        term2 = zeros(T, nlat, nlon)
        for j in 1:nlat
            @. term2[j, :] = r_inv * sinθ_inv[j] * d_sinθVθ_dθ[j, :]
        end

        # =====================================================================
        # TERM 3: (1/(r·sinθ))∂V_φ/∂φ
        # In spectral space: ∂/∂φ → multiply by im*m
        # =====================================================================
        # Transform V_φ to spectral space
        Vφ_lm = SHTnsKit.analysis(sht_config, Vφ_slice)

        # Apply ∂/∂φ operator: multiply by im*m
        dVφ_dφ_lm = zeros(ComplexF64, lmax+1, mmax+1)
        for l in 0:lmax
            for m in 0:min(l, mmax)
                dVφ_dφ_lm[l+1, m+1] = im * m * Vφ_lm[l+1, m+1]
            end
        end

        # Transform back to physical space
        dVφ_dφ = SHTnsKit.synthesis(sht_config, dVφ_dφ_lm; real_output=true)

        # Build term 3: (1/(r sinθ)) * dV_φ/dφ
        term3 = zeros(T, nlat, nlon)
        for j in 1:nlat
            @. term3[j, :] = r_inv * sinθ_inv[j] * dVφ_dφ[j, :]
        end

        # =====================================================================
        # Total divergence at this radius
        # =====================================================================
        @. div_data[:, :, k] = term1 + term2 + term3
    end

    return div_data
end


"""
    spectral_phi_derivative!(dfdφ, f, sht_config)

Compute ∂f/∂φ spectrally by multiplying spectral coefficients by im*m.
This is EXACT (no truncation error) for band-limited functions.
"""
function spectral_phi_derivative!(dfdφ::AbstractMatrix{T}, f::AbstractMatrix{T},
                                   sht_config) where T
    HAS_SHTNSKIT || error("SHTnsKit required")

    lmax = sht_config.lmax
    mmax = sht_config.mmax

    # Transform to spectral space
    f_lm = SHTnsKit.analysis(sht_config, f)

    # Apply ∂/∂φ: multiply by im*m
    df_lm = zeros(ComplexF64, lmax+1, mmax+1)
    for l in 0:lmax
        for m in 0:min(l, mmax)
            df_lm[l+1, m+1] = im * m * f_lm[l+1, m+1]
        end
    end

    # Transform back
    dfdφ .= SHTnsKit.synthesis(sht_config, df_lm; real_output=true)
    return dfdφ
end


"""
    spectral_theta_derivative!(dfdθ, f, sht_config)

Compute ∂f/∂θ spectrally using SHTnsKit's gradient operator.
This uses the proper spherical harmonic recurrence relations.
"""
function spectral_theta_derivative!(dfdθ::AbstractMatrix{T}, f::AbstractMatrix{T},
                                     sht_config) where T
    HAS_SHTNSKIT || error("SHTnsKit required")

    # Transform to spectral space
    f_lm = SHTnsKit.analysis(sht_config, f)

    # Use SHTnsKit's gradient: returns (∂f/∂θ, (1/sinθ)∂f/∂φ)
    grad_θ, _ = SHTnsKit.SH_to_grad_spat(sht_config, f_lm)

    dfdθ .= grad_θ
    return dfdθ
end


# =============================================================================
# FINITE DIFFERENCE METHOD (Fallback when SHTnsKit unavailable)
# =============================================================================

"""
    compute_divergence_physical!(div_data, V_r, V_θ, V_φ, r_grid, theta_grid, phi_grid)

Compute divergence using finite differences in all directions.
This is the FALLBACK method when SHTnsKit is not available.

For a vector field V = (V_r, V_θ, V_φ) in spherical coordinates:
    ∇·V = (1/r²)∂(r²V_r)/∂r + (1/(r·sinθ))∂(sinθ·V_θ)/∂θ + (1/(r·sinθ))∂V_φ/∂φ
"""
function compute_divergence_physical!(div_data::AbstractArray{T,3},
                                       V_r::AbstractArray{T,3},
                                       V_θ::AbstractArray{T,3},
                                       V_φ::AbstractArray{T,3},
                                       r_grid::AbstractVector{<:Real},
                                       theta_grid::AbstractVector{<:Real},
                                       phi_grid::AbstractVector{<:Real}) where T

    nlat, nlon, nr = size(V_r)

    # Compute divergence at each point
    for k in 1:nr
        r = r_grid[k]
        r_inv = 1.0 / r
        r_inv2 = r_inv^2

        for j in 1:nlat
            sin_theta = sin(theta_grid[j])
            sin_theta_inv = 1.0 / max(sin_theta, 1e-10)

            for i in 1:nlon
                # Term 1: (1/r²)∂(r²V_r)/∂r
                if k == 1 && nr > 1
                    dVr_dr = (V_r[j, i, k+1] - V_r[j, i, k]) / (r_grid[k+1] - r)
                elseif k == nr && k > 1
                    dVr_dr = (V_r[j, i, k] - V_r[j, i, k-1]) / (r - r_grid[k-1])
                elseif nr > 2
                    dVr_dr = (V_r[j, i, k+1] - V_r[j, i, k-1]) / (r_grid[k+1] - r_grid[k-1])
                else
                    dVr_dr = 0.0
                end
                term1 = r_inv2 * (2.0 * r * V_r[j, i, k] + r^2 * dVr_dr)

                # Term 2: (1/(r·sinθ))∂(sinθ·V_θ)/∂θ
                if j == 1
                    d_sinθ_Vθ = sin(theta_grid[j+1]) * V_θ[j+1, i, k] - sin_theta * V_θ[j, i, k]
                    d_sinθ_Vθ /= (theta_grid[j+1] - theta_grid[j])
                elseif j == nlat
                    d_sinθ_Vθ = sin_theta * V_θ[j, i, k] - sin(theta_grid[j-1]) * V_θ[j-1, i, k]
                    d_sinθ_Vθ /= (theta_grid[j] - theta_grid[j-1])
                else
                    d_sinθ_Vθ = sin(theta_grid[j+1]) * V_θ[j+1, i, k] - sin(theta_grid[j-1]) * V_θ[j-1, i, k]
                    d_sinθ_Vθ /= (theta_grid[j+1] - theta_grid[j-1])
                end
                term2 = r_inv * sin_theta_inv * d_sinθ_Vθ

                # Term 3: (1/(r·sinθ))∂V_φ/∂φ (periodic in φ)
                i_next = i == nlon ? 1 : i + 1
                i_prev = i == 1 ? nlon : i - 1
                dphi = phi_grid[2] - phi_grid[1]  # Assume uniform
                dVφ_dφ = (V_φ[j, i_next, k] - V_φ[j, i_prev, k]) / (2 * dphi)
                term3 = r_inv * sin_theta_inv * dVφ_dφ

                # Total divergence
                div_data[j, i, k] = term1 + term2 + term3
            end
        end
    end

    return div_data
end


# =============================================================================
# MAIN INTERFACE: Automatically selects best method
# =============================================================================

"""
    compute_divergence!(div_data, V_r, V_θ, V_φ, r_grid, theta_grid, phi_grid;
                        sht_config=nothing, method=:auto)

Compute divergence of a vector field V = (V_r, V_θ, V_φ) in spherical coordinates.

Methods:
- :spectral - Use SHTnsKit for angular derivatives (most accurate)
- :physical - Use finite differences (fallback)
- :auto - Use spectral if SHTnsKit available, else physical

The divergence formula is:
    ∇·V = (1/r²)∂(r²V_r)/∂r + (1/(r·sinθ))∂(sinθ·V_θ)/∂θ + (1/(r·sinθ))∂V_φ/∂φ

Note: Radial derivatives always use finite differences (standard in spectral codes).
      Angular derivatives use spectral methods when available (more accurate).
"""
function compute_divergence!(div_data::AbstractArray{T,3},
                              V_r::AbstractArray{T,3},
                              V_θ::AbstractArray{T,3},
                              V_φ::AbstractArray{T,3},
                              r_grid::AbstractVector{<:Real},
                              theta_grid::AbstractVector{<:Real},
                              phi_grid::AbstractVector{<:Real};
                              sht_config=nothing,
                              method::Symbol=:auto) where T

    # Select method
    use_spectral = if method == :auto
        HAS_SHTNSKIT && sht_config !== nothing
    elseif method == :spectral
        HAS_SHTNSKIT || error("SHTnsKit required for spectral method")
        sht_config !== nothing || error("sht_config required for spectral method")
        true
    else
        false
    end

    if use_spectral
        println("  Using SPECTRAL angular derivatives (SHTnsKit)")
        compute_divergence_spectral_angular!(div_data, V_r, V_θ, V_φ, r_grid, theta_grid, sht_config)
    else
        println("  Using FINITE DIFFERENCE angular derivatives")
        compute_divergence_physical!(div_data, V_r, V_θ, V_φ, r_grid, theta_grid, phi_grid)
    end

    return div_data
end


"""
    create_sht_config_for_grid(nlat, nlon; lmax=nothing, mmax=nothing)

Create an SHTnsKit configuration for a given grid size.
"""
function create_sht_config_for_grid(nlat::Int, nlon::Int; lmax=nothing, mmax=nothing)
    HAS_SHTNSKIT || error("SHTnsKit required")

    lmax = lmax === nothing ? nlat - 2 : lmax
    mmax = mmax === nothing ? lmax : mmax

    cfg = SHTnsKit.create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon, norm=:orthonormal)
    return cfg
end


# =============================================================================
# Module initialization message
# =============================================================================

println("""
================================================================================
SPECTRAL Pressure Solver Components Loaded
================================================================================

Key functions:
  - compute_divergence!                    Main interface (auto-selects method)
  - compute_divergence_spectral_angular!   Spectral angular derivatives (RECOMMENDED)
  - compute_divergence_physical!           Finite difference fallback
  - spectral_phi_derivative!               ∂f/∂φ via im*m multiplication
  - spectral_theta_derivative!             ∂f/∂θ via SHTnsKit gradient

SHTnsKit available: $(HAS_SHTNSKIT)

Why spectral angular derivatives?
  ✓ EXACT to machine precision for band-limited functions
  ✓ No truncation error from finite difference stencils
  ✓ Properly handles pole singularities via spherical harmonic basis
  ✓ Consistent with GeoDynamo.jl's spectral approach

Note: Radial derivatives use finite differences (standard practice in spectral codes)
================================================================================
""")
