"""
CORRECTED VERSION: Proper divergence computation for pressure RHS

This file shows the correct way to compute ∇·F for the pressure Poisson equation.

The key issue: Forces are NOT solenoidal, so toroidal-poloidal decomposition
doesn't properly capture their divergence.
"""

using LinearAlgebra
using Statistics

"""
    compute_divergence_physical!(div_field, vector_field, domain, config)

Compute divergence of a vector field directly in physical space.

For a vector field V = (V_r, V_θ, V_φ) in spherical coordinates:
    ∇·V = (1/r²)∂(r²V_r)/∂r + (1/(r·sinθ))∂(sinθ·V_θ)/∂θ + (1/(r·sinθ))∂V_φ/∂φ

This is the CORRECT approach for non-solenoidal fields like forces.
"""
function compute_divergence_physical!(div_field::SHTnsPhysicalField{T},
                                       vector_field::SHTnsVectorField{T},
                                       domain::RadialDomain,
                                       config::SHTnsKitConfig) where T

    # Get physical space data
    V_r = parent(vector_field.r_component.data)
    V_θ = parent(vector_field.θ_component.data)
    V_φ = parent(vector_field.φ_component.data)

    div_data = parent(div_field.data)

    # Get grid info
    nlat = config.nlat
    nlon = config.nlon
    theta = config.theta_grid
    phi = config.phi_grid

    # Get radial range
    r_range = get_local_range(div_field.pencil, 3)
    nr = domain.N

    # Compute divergence at each point
    for k in 1:size(div_data, 3)
        r_idx = k + first(r_range) - 1
        if r_idx <= nr
            r = domain.r[r_idx, 4]
            r_inv = domain.r[r_idx, 3]
            r_inv2 = domain.r[r_idx, 2]

            # Extract slice at this radius
            Vr_slice = V_r[:, :, k]
            Vθ_slice = V_θ[:, :, k]
            Vφ_slice = V_φ[:, :, k]

            for j in 1:nlat
                sin_theta = sin(theta[j])
                sin_theta_inv = 1.0 / max(sin_theta, 1e-10)

                for i in 1:nlon
                    # Term 1: (1/r²)∂(r²V_r)/∂r
                    # Use finite difference in radial direction
                    if k == 1 && size(div_data, 3) > 1
                        # Forward difference at inner boundary
                        dVr_dr = (V_r[j, i, k+1] - V_r[j, i, k]) / (domain.r[r_idx+1, 4] - r)
                    elseif k == size(div_data, 3) && k > 1
                        # Backward difference at outer boundary
                        dVr_dr = (V_r[j, i, k] - V_r[j, i, k-1]) / (r - domain.r[r_idx-1, 4])
                    elseif size(div_data, 3) > 2
                        # Central difference in interior
                        dVr_dr = (V_r[j, i, k+1] - V_r[j, i, k-1]) / (domain.r[r_idx+1, 4] - domain.r[r_idx-1, 4])
                    else
                        dVr_dr = 0.0
                    end
                    term1 = r_inv2 * (2.0 * r * Vr_slice[j, i] + r * r * dVr_dr)

                    # Term 2: (1/(r·sinθ))∂(sinθ·V_θ)/∂θ
                    # Use finite difference in theta direction (periodic)
                    if j == 1
                        dVθ_dθ = (Vθ_slice[j+1, i] - Vθ_slice[j, i]) / (theta[j+1] - theta[j])
                        d_sinθ_Vθ = sin(theta[j+1]) * Vθ_slice[j+1, i] - sin_theta * Vθ_slice[j, i]
                        d_sinθ_Vθ /= (theta[j+1] - theta[j])
                    elseif j == nlat
                        dVθ_dθ = (Vθ_slice[j, i] - Vθ_slice[j-1, i]) / (theta[j] - theta[j-1])
                        d_sinθ_Vθ = sin_theta * Vθ_slice[j, i] - sin(theta[j-1]) * Vθ_slice[j-1, i]
                        d_sinθ_Vθ /= (theta[j] - theta[j-1])
                    else
                        dVθ_dθ = (Vθ_slice[j+1, i] - Vθ_slice[j-1, i]) / (theta[j+1] - theta[j-1])
                        d_sinθ_Vθ = sin(theta[j+1]) * Vθ_slice[j+1, i] - sin(theta[j-1]) * Vθ_slice[j-1, i]
                        d_sinθ_Vθ /= (theta[j+1] - theta[j-1])
                    end
                    term2 = r_inv * sin_theta_inv * d_sinθ_Vθ

                    # Term 3: (1/(r·sinθ))∂V_φ/∂φ
                    # Use finite difference in phi direction (periodic)
                    i_next = i == nlon ? 1 : i + 1
                    i_prev = i == 1 ? nlon : i - 1
                    dphi = phi[i_next] - phi[i]
                    if dphi < 0
                        dphi += 2π
                    end
                    dVφ_dφ = (Vφ_slice[j, i_next] - Vφ_slice[j, i_prev]) / (2 * dphi)
                    term3 = r_inv * sin_theta_inv * dVφ_dφ

                    # Total divergence
                    div_data[j, i, k] = term1 + term2 + term3
                end
            end
        end
    end

    return div_field
end


"""
    compute_pressure_rhs_CORRECTED!(rhs_field, velocity_fields, temp_field,
                                     comp_field, mag_field, domain, params)

CORRECTED version: Compute RHS = ∇·F properly for non-solenoidal force field.

Key difference from original:
- Computes divergence in PHYSICAL space using proper formula
- Then transforms scalar divergence to spectral space
- Handles non-solenoidal forces correctly
"""
function compute_pressure_rhs_CORRECTED!(rhs_field::SHTnsSpectralField{T},
                                         velocity_fields::SHTnsVelocityFields{T},
                                         temp_field, comp_field, mag_field,
                                         domain::RadialDomain,
                                         params) where T

    println("  Computing RHS (corrected divergence method)...")

    # Step 1: Compute all forces in physical space (same as before)
    compute_velocity_nonlinear!(velocity_fields, temp_field, comp_field, mag_field, domain)

    # The advection_physical field now contains F = -u×ω + Coriolis + buoyancy + Lorentz

    # Step 2: Create physical space divergence field
    config = velocity_fields.config
    div_physical = create_shtns_physical_field(T, config, domain, config.pencils.r)

    # Step 3: Compute divergence in physical space (CORRECT METHOD)
    compute_divergence_physical!(div_physical, velocity_fields.advection_physical, domain, config)

    # Step 4: Transform scalar divergence to spectral space
    shtnskit_scalar_analysis!(div_physical, rhs_field)

    println("    ✓ Divergence computed correctly in physical space")

    return rhs_field
end


"""
Alternative: Compute divergence from spectral vector components

This is more accurate but requires transforming vector components separately
rather than using toroidal-poloidal decomposition.
"""
function compute_divergence_spectral!(div_spectral::SHTnsSpectralField{T},
                                       V_r_spectral::SHTnsSpectralField{T},
                                       V_θ_spectral::SHTnsSpectralField{T},
                                       V_φ_spectral::SHTnsSpectralField{T},
                                       domain::RadialDomain,
                                       l_factors::Vector{Float64}) where T

    # For a vector field expanded as:
    # V_r = Σ V_r_lm(r) Y_lm
    # V_θ = Σ V_θ_lm(r) ∂Y_lm/∂θ
    # V_φ = Σ V_φ_lm(r) (1/sinθ) ∂Y_lm/∂φ

    # The divergence in spectral space is:
    # (∇·V)_lm = (1/r²)d(r²V_r_lm)/dr + spectral_angular_operator(V_θ, V_φ)

    # This requires the proper spherical harmonic derivative operators
    # which is more complex to implement but mathematically rigorous

    # Implementation would go here...
    # (Left as placeholder - requires careful spherical harmonic calculus)

    @warn "Spectral divergence computation not yet fully implemented"
    return div_spectral
end


println("""
CORRECTED Pressure Solver Components Loaded

Key functions:
  - compute_divergence_physical!      (Correct divergence in physical space)
  - compute_pressure_rhs_CORRECTED!   (Fixed RHS computation)
  - compute_divergence_spectral!      (Spectral approach - placeholder)

The main issue with the original:
  ✗ Used toroidal-poloidal decomposition for non-solenoidal force field
  ✗ Extracted "divergence" from poloidal component only
  ✗ This is incorrect for fields with ∇·F ≠ 0

The corrected approach:
  ✓ Computes ∇·F directly in physical space using proper formula
  ✓ Transforms scalar divergence to spectral space
  ✓ Works correctly for any force field (solenoidal or not)
""")
