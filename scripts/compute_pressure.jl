#!/usr/bin/env julia
"""
    compute_pressure.jl

Compute pressure field by solving the pressure Poisson equation derived from
the divergence of the momentum equation in a rotating, magnetohydrodynamic system.

# Theory

For incompressible flow (∇·u = 0), taking the divergence of the momentum equation:
    ∂u/∂t + u×ω = -∇p/ρ + 2Ω(ẑ×u) + ν∇²u + buoyancy + Lorentz

gives the pressure Poisson equation:
    ∇²p = -ρ ∇·(u×ω) + ρ ∇·(buoyancy forces) + ρ ∇·(Lorentz force)

In spectral space (spherical harmonics), this becomes:
    For each (l,m) mode: [d²/dr² + 2/r d/dr - l(l+1)/r²] p_lm = RHS_lm

# Boundary Conditions

The solver uses **Neumann boundary conditions** derived from the radial momentum equation:

At boundaries where velocity is constrained (no-slip or stress-free):
    ∂p/∂r|_boundary = (buoyancy + Lorentz)_r

This is physically correct and consistent with the momentum equation. The boundary
forces are computed from temperature, composition, and magnetic field values at
the inner and outer boundaries.

For the l=0 (spherically symmetric) mode, an additional constraint p(r_mid) = 0
is applied to fix the arbitrary pressure constant.

# Method: Divergence Computation (v2.0 Physical)

Force fields that appear in the pressure Poisson equation are not solenoidal,
so their divergence cannot be inferred from a toroidal–poloidal representation.
The production solver therefore constructs the RHS entirely in physical space:

1. **Assemble the total force** `F = -u×ω + 2Ω×u + buoyancy + Lorentz` on the
   Gauss–Legendre grid used by SHTnsKit.
2. **Evaluate the spherical-coordinate divergence** using second-order finite
   differences in r, θ, and φ.
3. **Transform the scalar divergence** back to spectral space before solving
   the radial Poisson problem mode-by-mode.

This pipeline preserves the full divergence information and matches the fix
described in `scripts/DIVERGENCE_CORRECTION.md`. An experimental spectral-only
implementation lives in `scripts/compute_pressure_SPECTRAL.jl`.

# Usage

```julia
using GeoDynamo
include("scripts/compute_pressure.jl")

# Compute pressure from output file
pressure = compute_pressure_from_output("outputs/fields_00100.nc")

# Or compute pressure from field structures directly
pressure = compute_pressure_poisson(velocity_fields, temp_fields, mag_fields, domain)
```
"""

using NCDatasets
using LinearAlgebra
using Printf
using Statistics
using Glob

# Import GeoDynamo modules
if !isdefined(Main, :GeoDynamo)
    using GeoDynamo
end

# Import MPI for parallel processing
if !isdefined(Main, :MPI)
    using MPI
end

"""
    compute_divergence_physical!(div_field::SHTnsPhysicalField{T},
                                  vector_field::SHTnsVectorField{T},
                                  domain::RadialDomain,
                                  config::SHTnsKitConfig) where T

Compute the divergence of a general vector field directly in physical space.

Force fields in the momentum equation are not solenoidal, so their divergence
must be evaluated component-wise rather than inferred from a toroidal-poloidal
representation. This routine applies the exact spherical-coordinate formula
with finite differences in r, θ, and φ.
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
                    if k == 1 && size(div_data, 3) > 1
                        dVr_dr = (V_r[j, i, k+1] - V_r[j, i, k]) / (domain.r[r_idx+1, 4] - r)
                    elseif k == size(div_data, 3) && k > 1
                        dVr_dr = (V_r[j, i, k] - V_r[j, i, k-1]) / (r - domain.r[r_idx-1, 4])
                    elseif size(div_data, 3) > 2
                        dVr_dr = (V_r[j, i, k+1] - V_r[j, i, k-1]) /
                                 (domain.r[r_idx+1, 4] - domain.r[r_idx-1, 4])
                    else
                        dVr_dr = 0.0
                    end
                    term1 = r_inv2 * (2.0 * r * Vr_slice[j, i] + r * r * dVr_dr)

                    # Term 2: (1/(r·sinθ))∂(sinθ·V_θ)/∂θ
                    if j == 1
                        d_sinθ_Vθ = sin(theta[j+1]) * Vθ_slice[j+1, i] - sin_theta * Vθ_slice[j, i]
                        d_sinθ_Vθ /= (theta[j+1] - theta[j])
                    elseif j == nlat
                        d_sinθ_Vθ = sin_theta * Vθ_slice[j, i] - sin(theta[j-1]) * Vθ_slice[j-1, i]
                        d_sinθ_Vθ /= (theta[j] - theta[j-1])
                    else
                        d_sinθ_Vθ = sin(theta[j+1]) * Vθ_slice[j+1, i] - sin(theta[j-1]) * Vθ_slice[j-1, i]
                        d_sinθ_Vθ /= (theta[j+1] - theta[j-1])
                    end
                    term2 = r_inv * sin_theta_inv * d_sinθ_Vθ

                    # Term 3: (1/(r·sinθ))∂V_φ/∂φ (periodic in φ)
                    i_next = i == nlon ? 1 : i + 1
                    i_prev = i == 1 ? nlon : i - 1
                    dphi = phi[i_next] - phi[i]
                    if dphi < 0
                        dphi += 2π
                    end
                    dVφ_dφ = (Vφ_slice[j, i_next] - Vφ_slice[j, i_prev]) / (2 * dphi)
                    term3 = r_inv * sin_theta_inv * dVφ_dφ

                    div_data[j, i, k] = term1 + term2 + term3
                end
            end
        end
    end

    return div_field
end


"""
    compute_pressure_rhs!(rhs_field::SHTnsSpectralField{T},
                          velocity_fields::SHTnsVelocityFields{T},
                          temp_field, comp_field, mag_field,
                          domain::RadialDomain,
                          params;
                          method::Symbol=:physical) where T

Compute the right-hand side of the pressure Poisson equation:
    ∇²p = RHS = ∇·F

where F = -u×ω + Coriolis + buoyancy + Lorentz (total force field).

# Physics
From momentum equation:
    ∂u/∂t = -u×ω - ∇p/ρ + 2Ω(ẑ×u) + ν∇²u + buoyancy + Lorentz

Taking divergence (using ∇·u = 0):
    ∇²p = -ρ[∇·(u×ω) - ∇·(2Ω ẑ×u) + ∇·(buoyancy) + ∇·(Lorentz)]

# Implementation (v2.0 Physical)
1. Compute forces F in physical space
2. Compute ∇·F in physical space using finite differences (see `compute_divergence_physical!`)
3. Transform the scalar divergence to spectral space
"""
function compute_pressure_rhs!(rhs_field::SHTnsSpectralField{T},
                                velocity_fields::SHTnsVelocityFields{T},
                                temp_field, comp_field, mag_field,
                                domain::RadialDomain,
                                params;
                                method::Symbol=:physical) where T

    # Step 1: Compute all forces in physical space
    compute_velocity_nonlinear!(velocity_fields, temp_field, comp_field, mag_field, domain)

    # The advection_physical field now contains F = -u×ω + Coriolis + buoyancy + Lorentz

    config = velocity_fields.config

    if method != :physical
        error("Only method = :physical is supported in scripts/compute_pressure.jl. " *
              "See scripts/compute_pressure_SPECTRAL.jl for experimental options.")
    end

    # Compute divergence in physical space, then transform
    div_physical = create_shtns_physical_field(T, config, domain, config.pencils.r)
    compute_divergence_physical!(div_physical, velocity_fields.advection_physical, domain, config)
    shtnskit_scalar_analysis!(div_physical, rhs_field)

    return rhs_field
end


"""
    compute_boundary_forces(velocity_fields, temp_field, comp_field,
                            mag_field, domain, params, lm_idx)

Compute the radial forces at boundaries for a specific spectral mode,
needed for Neumann pressure boundary conditions.

From the radial momentum equation at boundaries where velocity is constrained:
    ∂p/∂r|_boundary = (buoyancy + Lorentz + viscous_stress)_r

For stress-free boundaries, viscous stress = 0.
For no-slip boundaries, viscous stress is included.

Returns (force_inner, force_outer) for the given (l,m) mode.
"""
function compute_boundary_forces(velocity_fields, temp_field, comp_field,
                                  mag_field, domain, params, lm_idx)
    T = eltype(velocity_fields.toroidal.data_real)

    # Initialize forces
    force_inner = zero(T)
    force_outer = zero(T)

    # Get parameters for scaling
    rossby_factor = params.d_Pm / params.d_E

    prepare_velocity_field_for_forces!(velocity_fields, domain)

    # Buoyancy contribution at boundaries
    if temp_field !== nothing
        buoyancy_factor = rossby_factor * (params.d_Pm / params.d_Pr) * params.d_Ra

        # For spectral field, extract boundary values for this mode
        if isa(temp_field, SHTnsPhysicalField)
            # Physical space - take mean at boundaries
            T_data = parent(temp_field.data)
            if size(T_data, 3) >= 2
                T_inner = mean(T_data[:, :, 1])
                T_outer = mean(T_data[:, :, end])
                force_inner += buoyancy_factor * T_inner
                force_outer += buoyancy_factor * T_outer
            end
        else
            # Spectral field - use mode coefficients at boundaries
            temp_real = parent(temp_field.data_real)
            if lm_idx <= size(temp_real, 1) && size(temp_real, 3) >= 2
                # Real part at boundaries (mode-dependent)
                T_inner = temp_real[lm_idx, 1, 1]
                T_outer = temp_real[lm_idx, 1, end]
                force_inner += buoyancy_factor * T_inner
                force_outer += buoyancy_factor * T_outer
            end
        end
    end

    # Compositional buoyancy
    if comp_field !== nothing
        comp_factor = rossby_factor * (params.d_Pm / params.d_Sc) * params.d_Ra_C

        if isa(comp_field, SHTnsPhysicalField)
            C_data = parent(comp_field.data)
            if size(C_data, 3) >= 2
                C_inner = mean(C_data[:, :, 1])
                C_outer = mean(C_data[:, :, end])
                force_inner += comp_factor * C_inner
                force_outer += comp_factor * C_outer
            end
        else
            comp_real = parent(comp_field.data_real)
            if lm_idx <= size(comp_real, 1) && size(comp_real, 3) >= 2
                C_inner = comp_real[lm_idx, 1, 1]
                C_outer = comp_real[lm_idx, 1, end]
                force_inner += comp_factor * C_inner
                force_outer += comp_factor * C_outer
            end
        end
    end

    # Lorentz force contribution (j×B)_r at boundaries
    if mag_field !== nothing
        lorentz_factor = rossby_factor
        prepare_magnetic_force_fields!(mag_field, domain)

        lorentz_inner = get_boundary_lorentz_average(mag_field, domain, :inner)
        lorentz_outer = get_boundary_lorentz_average(mag_field, domain, :outer)

        force_inner += lorentz_factor * lorentz_inner
        force_outer += lorentz_factor * lorentz_outer
    end

    # Viscous stress contribution for no-slip boundaries
    if params.i_vel_bc == 1  # no-slip
        visc_factor = rossby_factor
        visc_inner = get_boundary_viscous_average(velocity_fields, domain, :inner)
        visc_outer = get_boundary_viscous_average(velocity_fields, domain, :outer)
        force_inner += visc_factor * visc_inner
        force_outer += visc_factor * visc_outer
    end

    return (force_inner, force_outer)
end

function prepare_velocity_field_for_forces!(velocity_fields::SHTnsVelocityFields, domain::RadialDomain)
    cache = velocity_fields.boundary_interpolation_cache
    if get(cache, "velocity_ready", false)
        return velocity_fields
    end
    shtnskit_vector_synthesis!(velocity_fields.toroidal, velocity_fields.poloidal,
                               velocity_fields.velocity; domain=domain)
    cache["velocity_ready"] = true
    return velocity_fields
end

function get_boundary_viscous_average(velocity_fields::SHTnsVelocityFields{T},
                                      domain::RadialDomain,
                                      boundary::Symbol) where T
    boundary_idx = boundary === :inner ? 1 : domain.N
    neighbor_idx = boundary === :inner ? min(domain.N, boundary_idx + 1) : max(1, boundary_idx - 1)

    vel_r = parent(velocity_fields.velocity.r_component.data)
    r_range = get_local_range(velocity_fields.velocity.r_component.pencil, 3)

    local_sum = zero(T)
    local_count = zero(Int)

    if boundary_idx in r_range && neighbor_idx in r_range
        local_boundary = boundary_idx - first(r_range) + 1
        local_neighbor = neighbor_idx - first(r_range) + 1
        if 1 <= local_boundary <= size(vel_r, 3) && 1 <= local_neighbor <= size(vel_r, 3)
            dr = abs(domain.r[neighbor_idx, 4] - domain.r[boundary_idx, 4])
            if dr > 0
                if boundary === :inner
                    derivative = (vel_r[:, :, local_neighbor] - vel_r[:, :, local_boundary]) / dr
                else
                    derivative = (vel_r[:, :, local_boundary] - vel_r[:, :, local_neighbor]) / dr
                end
                local_sum = sum(derivative)
                local_count = length(derivative)
            end
        end
    end

    if MPI.Initialized()
        total_sum = MPI.Allreduce(local_sum, +, MPI.COMM_WORLD)
        total_count = MPI.Allreduce(local_count, +, MPI.COMM_WORLD)
    else
        total_sum = local_sum
        total_count = local_count
    end

    return total_count == 0 ? zero(T) : total_sum / T(total_count)
end

function prepare_magnetic_force_fields!(mag_field::SHTnsMagneticFields, domain::RadialDomain)
    cache = mag_field.boundary_cache
    if get(cache, "lorentz_ready", false)
        return mag_field
    end

    # Populate physical magnetic field and current density needed for j×B evaluation
    shtnskit_vector_synthesis!(mag_field.toroidal, mag_field.poloidal, mag_field.magnetic; domain=domain)
    compute_current_density_spectral!(mag_field, domain)
    shtnskit_vector_synthesis!(mag_field.work_tor, mag_field.work_pol, mag_field.current; domain=domain)

    cache["lorentz_ready"] = true
    return mag_field
end

function get_boundary_lorentz_average(mag_field::SHTnsMagneticFields{T},
                                      domain::RadialDomain,
                                      boundary::Symbol) where T
    boundary_idx = boundary === :inner ? 1 : domain.N

    j_θ = parent(mag_field.current.θ_component.data)
    j_φ = parent(mag_field.current.φ_component.data)
    B_θ = parent(mag_field.magnetic.θ_component.data)
    B_φ = parent(mag_field.magnetic.φ_component.data)

    r_range = get_local_range(mag_field.current.r_component.pencil, 3)

    local_sum = zero(T)
    local_count = zero(Int)

    if boundary_idx in r_range
        local_r = boundary_idx - first(r_range) + 1
        if 1 <= local_r <= size(j_θ, 3)
            lorentz_slice = j_θ[:, :, local_r] .* B_φ[:, :, local_r] .-
                            j_φ[:, :, local_r] .* B_θ[:, :, local_r]
            local_sum = sum(lorentz_slice)
            local_count = length(lorentz_slice)
        end
    end

    if MPI.Initialized()
        total_sum = MPI.Allreduce(local_sum, +, MPI.COMM_WORLD)
        total_count = MPI.Allreduce(local_count, +, MPI.COMM_WORLD)
    else
        total_sum = local_sum
        total_count = local_count
    end

    return total_count == 0 ? zero(T) : total_sum / T(total_count)
end


"""
    solve_pressure_poisson!(pressure_field::SHTnsSpectralField{T},
                            rhs_field::SHTnsSpectralField{T},
                            domain::RadialDomain,
                            velocity_fields::SHTnsVelocityFields{T},
                            temp_field, comp_field, mag_field,
                            params) where T

Solve the pressure Poisson equation in spectral space with Neumann boundary conditions:
    ∇²p = RHS

For each (l,m) mode:
    [d²/dr² + 2/r d/dr - l(l+1)/r²] p_lm(r) = RHS_lm(r)

Boundary conditions (Neumann - physically correct):
    ∂p/∂r|_inner = (buoyancy + Lorentz)_r at inner boundary
    ∂p/∂r|_outer = (buoyancy + Lorentz)_r at outer boundary

These come from the radial momentum equation at boundaries where u is constrained.
"""
function solve_pressure_poisson!(pressure_field::SHTnsSpectralField{T},
                                  rhs_field::SHTnsSpectralField{T},
                                  domain::RadialDomain,
                                  velocity_fields::SHTnsVelocityFields{T},
                                  temp_field, comp_field, mag_field,
                                  params) where T

    p_real = parent(pressure_field.data_real)
    p_imag = parent(pressure_field.data_imag)
    rhs_real = parent(rhs_field.data_real)
    rhs_imag = parent(rhs_field.data_imag)

    config = velocity_fields.config
    lm_range = get_local_range(pressure_field.pencil, 1)
    r_range = get_local_range(pressure_field.pencil, 3)
    nr = domain.N

    # Get radial operators
    dr_matrix = velocity_fields.dr_matrix
    d2r_matrix = velocity_fields.d2r_matrix

    # Buffers for each mode
    rhs_profile_real = zeros(T, nr)
    rhs_profile_imag = zeros(T, nr)
    sol_profile_real = zeros(T, nr)
    sol_profile_imag = zeros(T, nr)

    # Solve for each spectral mode
    Threads.@threads for lm_idx in lm_range
        if lm_idx <= pressure_field.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = velocity_fields.l_factors[lm_idx]

            # Extract RHS profile
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(rhs_real, 3)
                    rhs_profile_real[r_idx] = rhs_real[local_lm, 1, local_r]
                    rhs_profile_imag[r_idx] = rhs_imag[local_lm, 1, local_r]
                end
            end

            # Build Poisson operator: ∇² = d²/dr² + 2/r d/dr - l(l+1)/r²
            # This is a banded matrix problem

            # For simplicity, use a direct matrix inversion approach
            # Build full operator matrix (in practice, use banded solver)
            A = zeros(T, nr, nr)

            # Fill operator matrix
            for i in 1:nr
                r_val = domain.r[i, 4]
                r_inv = domain.r[i, 3]
                r_inv2 = domain.r[i, 2]

                # Add second derivative operator
                for j in 1:nr
                    if abs(i - j) <= d2r_matrix.bandwidth
                        band_row = d2r_matrix.bandwidth + 1 + i - j
                        if 1 <= band_row <= 2*d2r_matrix.bandwidth + 1
                            A[i, j] += d2r_matrix.data[band_row, j]
                        end
                    end
                end

                # Add first derivative operator with 2/r factor
                for j in 1:nr
                    if abs(i - j) <= dr_matrix.bandwidth
                        band_row = dr_matrix.bandwidth + 1 + i - j
                        if 1 <= band_row <= 2*dr_matrix.bandwidth + 1
                            A[i, j] += 2.0 * r_inv * dr_matrix.data[band_row, j]
                        end
                    end
                end

                # Subtract l(l+1)/r² (diagonal)
                if r_val > 0
                    A[i, i] -= l_factor * r_inv2
                end
            end

            # Apply boundary conditions (Neumann - physically derived from momentum equation)
            # Compute forces at boundaries for this spectral mode
            force_inner, force_outer = compute_boundary_forces(
                velocity_fields, temp_field, comp_field, mag_field, domain, params, lm_idx
            )

            # Inner boundary: ∂p/∂r = force_inner (Neumann BC)
            # Replace first row with derivative operator
            A[1, :] .= 0
            for j in 1:nr
                if abs(1 - j) <= dr_matrix.bandwidth
                    band_row = dr_matrix.bandwidth + 1 + 1 - j
                    if 1 <= band_row <= 2*dr_matrix.bandwidth + 1
                        A[1, j] = dr_matrix.data[band_row, j]
                    end
                end
            end
            rhs_profile_real[1] = force_inner
            rhs_profile_imag[1] = 0  # Imaginary part typically zero at boundary

            # Outer boundary: ∂p/∂r = force_outer (Neumann BC)
            # Replace last row with derivative operator
            A[nr, :] .= 0
            for j in 1:nr
                if abs(nr - j) <= dr_matrix.bandwidth
                    band_row = dr_matrix.bandwidth + 1 + nr - j
                    if 1 <= band_row <= 2*dr_matrix.bandwidth + 1
                        A[nr, j] = dr_matrix.data[band_row, j]
                    end
                end
            end
            rhs_profile_real[nr] = force_outer
            rhs_profile_imag[nr] = 0

            # Fix pressure constant for l=0 mode to avoid singular system
            # For l=0, Neumann BCs at both boundaries make system singular
            # Add constraint: p(r_middle) = 0 to fix the arbitrary constant
            if lm_idx == 1 && l_factor < 1e-10  # l=0 mode
                mid_point = nr ÷ 2
                A[mid_point, :] .= 0
                A[mid_point, mid_point] = 1
                rhs_profile_real[mid_point] = 0
                rhs_profile_imag[mid_point] = 0
            end

            # Solve linear system
            try
                sol_profile_real = A \ rhs_profile_real
                sol_profile_imag = A \ rhs_profile_imag
            catch e
                @warn "Failed to solve Poisson equation for mode $lm_idx: $e"
                fill!(sol_profile_real, 0)
                fill!(sol_profile_imag, 0)
            end

            # Store solution
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(p_real, 3)
                    p_real[local_lm, 1, local_r] = sol_profile_real[r_idx]
                    p_imag[local_lm, 1, local_r] = sol_profile_imag[r_idx]
                end
            end
        end
    end

    return pressure_field
end


"""
    compute_pressure_poisson(velocity_fields::SHTnsVelocityFields{T},
                             temp_field, comp_field, mag_field,
                             domain::RadialDomain,
                             params) where T

Main function to compute pressure by solving the pressure Poisson equation.

Returns a tuple (pressure_spectral, pressure_physical) containing the pressure
field in both spectral and physical space representations.
"""
function compute_pressure_poisson(velocity_fields::SHTnsVelocityFields{T},
                                   temp_field, comp_field, mag_field,
                                   domain::RadialDomain,
                                   params) where T

    println("Computing pressure via Poisson equation...")

    # Create fields for pressure and RHS
    config = velocity_fields.config
    pressure_spectral = create_shtns_spectral_field(T, config, domain, config.pencils.spec)
    rhs_spectral = create_shtns_spectral_field(T, config, domain, config.pencils.spec)

    # Step 1: Compute RHS = ∇·(force terms) using the corrected physical-space method
    println("  Computing RHS (physical-space divergence - v2.0)...")
    compute_pressure_rhs!(rhs_spectral, velocity_fields, temp_field, comp_field, mag_field, domain, params;
                          method=:physical)

    # Step 2: Solve Poisson equation ∇²p = RHS with Neumann BCs
    println("  Solving Poisson equation (Neumann BCs from momentum equation)...")
    solve_pressure_poisson!(pressure_spectral, rhs_spectral, domain, velocity_fields,
                            temp_field, comp_field, mag_field, params)

    # Step 3: Transform to physical space
    println("  Transforming to physical space...")
    pressure_physical = create_shtns_physical_field(T, config, domain, config.pencils.r)
    shtnskit_scalar_synthesis!(pressure_spectral, pressure_physical)

    println("  Done!")

    return (pressure_spectral, pressure_physical)
end


"""
    save_pressure_field(filename::String, pressure_physical::SHTnsPhysicalField{T},
                        domain::RadialDomain, config::SHTnsKitConfig) where T

Save pressure field to NetCDF file.
"""
function save_pressure_field(filename::String,
                              pressure_physical::SHTnsPhysicalField{T},
                              domain::RadialDomain,
                              config::SHTnsKitConfig) where T

    println("Saving pressure to $filename...")

    # Get local data
    p_data = parent(pressure_physical.data)

    # Create NetCDF file
    NCDataset(filename, "c") do ds
        # Define dimensions
        defDim(ds, "radius", domain.N)
        defDim(ds, "theta", config.nlat)
        defDim(ds, "phi", config.nlon)

        # Define coordinate variables
        defVar(ds, "r", Float64, ("radius",))[:] = domain.r[:, 4]
        defVar(ds, "theta", Float64, ("theta",))[:] = config.theta_grid
        defVar(ds, "phi", Float64, ("phi",))[:] = config.phi_grid

        # Define pressure variable
        pressure_var = defVar(ds, "pressure", Float32, ("theta", "phi", "radius"))
        pressure_var.attrib["long_name"] = "Pressure field"
        pressure_var.attrib["units"] = "dimensionless"

        # Write local portion
        r_range = get_local_range(pressure_physical.pencil, 3)
        local_r_start = first(r_range)
        local_r_count = length(r_range)

        if local_r_count > 0 && size(p_data, 3) > 0
            pressure_var[:, :, local_r_start:local_r_start+local_r_count-1] = p_data
        end

        # Add metadata
        ds.attrib["description"] = "Pressure field from Poisson equation"
        ds.attrib["creation_date"] = string(Dates.now())
    end

    println("  Saved!")
end


"""
    load_spectral_field_from_nc(nc_file, field_name::String,
                                 config::SHTnsKitConfig, domain::RadialDomain,
                                 ::Type{T}) where T

Load a spectral field (toroidal or poloidal) from NetCDF file.
"""
function load_spectral_field_from_nc(nc_file, field_name::String,
                                      config::SHTnsKitConfig, domain::RadialDomain,
                                      ::Type{T}) where T

    # Create the spectral field structure
    spec_field = create_shtns_spectral_field(T, config, domain, config.pencils.spec)

    # Read real and imaginary parts
    real_data = ncread(nc_file, field_name * "_real")
    imag_data = ncread(nc_file, field_name * "_imag")

    # Get local ranges
    lm_range = get_local_range(spec_field.pencil, 1)
    r_range = get_local_range(spec_field.pencil, 3)

    # Copy data to local portion
    spec_real = parent(spec_field.data_real)
    spec_imag = parent(spec_field.data_imag)

    for lm_idx in lm_range
        if lm_idx <= spec_field.nlm
            local_lm = lm_idx - first(lm_range) + 1
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(spec_real, 3) && r_idx <= size(real_data, 2)
                    spec_real[local_lm, 1, local_r] = T(real_data[lm_idx, r_idx])
                    spec_imag[local_lm, 1, local_r] = T(imag_data[lm_idx, r_idx])
                end
            end
        end
    end

    return spec_field
end


"""
    load_physical_field_from_nc(nc_file, field_name::String,
                                 config::SHTnsKitConfig, domain::RadialDomain,
                                 ::Type{T}) where T

Load a physical field from NetCDF file.
"""
function load_physical_field_from_nc(nc_file, field_name::String,
                                      config::SHTnsKitConfig, domain::RadialDomain,
                                      ::Type{T}) where T

    # Create the physical field structure
    phys_field = create_shtns_physical_field(T, config, domain, config.pencils.r)

    # Check if field exists in file
    if !(field_name in keys(nc_file.vars))
        @warn "Field $field_name not found in NetCDF file"
        return nothing
    end

    # Read data
    data = ncread(nc_file, field_name)  # [theta, phi, r]

    # Get local range
    r_range = get_local_range(phys_field.pencil, 3)

    # Copy to local portion
    phys_data = parent(phys_field.data)

    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(phys_data, 3) && r_idx <= size(data, 3)
            phys_data[:, :, local_r] = T.(data[:, :, r_idx])
        end
    end

    return phys_field
end


"""
    compute_pressure_from_output(input_file::String; output_file::String="")

Load velocity and other fields from output file and compute pressure.

# Arguments
- `input_file`: Path to NetCDF output file containing velocity fields
- `output_file`: Optional path to save pressure field (default: auto-generated)

# Returns
- Tuple of (pressure_spectral, pressure_physical)

# Example
```julia
# From command line:
# julia --project scripts/compute_pressure.jl output/geodynamo_rank_0000_time_100.nc

# From REPL:
using GeoDynamo
include("scripts/compute_pressure.jl")
pressure = compute_pressure_from_output("output/geodynamo_rank_0000_time_100.nc")
```
"""
function compute_pressure_from_output(input_file::String;
                                       output_file::String="",
                                       params=nothing)

    println("=" ^ 70)
    println("PRESSURE POISSON SOLVER")
    println("=" ^ 70)
    println("Input file: $input_file")

    # Check file exists
    if !isfile(input_file)
        error("Input file not found: $input_file")
    end

    # Load or use default parameters
    if params === nothing
        params = get_parameters()
    end

    println("\nLoading NetCDF data...")

    # Open NetCDF file
    nc = NetCDF.open(input_file)

    try
        # Read grid information
        r_vals = ncread(nc, "r")
        theta_vals = ncread(nc, "theta")
        phi_vals = ncread(nc, "phi")

        nr = length(r_vals)
        nlat = length(theta_vals)
        nlon = length(phi_vals)

        # Read spectral mode information
        l_values = ncread(nc, "l_values")
        m_values = ncread(nc, "m_values")
        nlm = length(l_values)

        lmax = maximum(l_values)
        mmax = maximum(m_values)

        println("  Grid: nr=$nr, nlat=$nlat, nlon=$nlon")
        println("  Spectral: lmax=$lmax, mmax=$mmax, nlm=$nlm")

        # Create configuration
        config = create_shtnskit_config(lmax, mmax, nlat, nlon)
        pencils = create_pencil_topology(config)

        # Update config with pencils
        config = SHTnsKitConfig(
            config.lmax, config.mmax, config.nlat, config.nlon,
            config.nlm, config.l_values, config.m_values,
            config.theta_grid, config.phi_grid, config.quad_weights,
            pencils
        )

        # Create radial domain
        domain = create_radial_domain(nr)
        domain.r[:, 4] .= r_vals  # Use actual radial values from file

        # Recompute powers of r
        for n in 1:nr
            for p in 1:7
                if p != 4
                    domain.r[n, p] = domain.r[n, 4]^(p - 4)
                end
            end
        end

        println("\nLoading velocity fields...")

        # Create velocity fields structure
        velocity_fields = create_shtns_velocity_fields(Float64, config, domain, pencils)

        # Load toroidal and poloidal velocity components
        velocity_fields.toroidal = load_spectral_field_from_nc(
            nc, "velocity_toroidal", config, domain, Float64
        )
        velocity_fields.poloidal = load_spectral_field_from_nc(
            nc, "velocity_poloidal", config, domain, Float64
        )

        println("  ✓ Velocity fields loaded")

        # Load temperature field (if present)
        temp_field = nothing
        if "temperature" in keys(nc.vars)
            println("\nLoading temperature field...")
            temp_field = load_physical_field_from_nc(nc, "temperature", config, domain, Float64)
            println("  ✓ Temperature field loaded")
        end

        # Load composition field (if present)
        comp_field = nothing
        if "composition" in keys(nc.vars)
            println("\nLoading composition field...")
            comp_field = load_physical_field_from_nc(nc, "composition", config, domain, Float64)
            println("  ✓ Composition field loaded")
        end

        # Load magnetic fields (if present)
        mag_field = nothing
        if "magnetic_toroidal_real" in keys(nc.vars)
            println("\nLoading magnetic fields...")
            mag_field = create_shtns_magnetic_fields(Float64, config, domain, domain, pencils, pencils.spec)
            mag_field.toroidal = load_spectral_field_from_nc(
                nc, "magnetic_toroidal", config, domain, Float64
            )
            mag_field.poloidal = load_spectral_field_from_nc(
                nc, "magnetic_poloidal", config, domain, Float64
            )
            prepare_magnetic_force_fields!(mag_field, domain)
            println("  ✓ Magnetic fields loaded")
        end

        NetCDF.close(nc)

        println("\n" * "=" ^ 70)
        println("Computing pressure...")
        println("=" ^ 70)

        # Compute pressure
        pressure_spectral, pressure_physical = compute_pressure_poisson(
            velocity_fields, temp_field, comp_field, mag_field, domain, params
        )

        # Save output
        if isempty(output_file)
            # Auto-generate output filename
            dir = dirname(input_file)
            base = basename(input_file)
            base = replace(base, r"\.nc$" => "")
            output_file = joinpath(dir, base * "_pressure.nc")
        end

        println("\n" * "=" ^ 70)
        save_pressure_field(output_file, pressure_physical, domain, config)
        println("=" ^ 70)

        println("\n✓ Pressure computation complete!")
        println("  Output: $output_file")

        return (pressure_spectral, pressure_physical)

    catch e
        NetCDF.close(nc)
        rethrow(e)
    end
end


"""
    process_multiple_files(file_pattern::String; output_dir::String="")

Process multiple output files matching a glob pattern.

# Example
```julia
# Process all output files
process_multiple_files("output/geodynamo_rank_0000_time_*.nc")

# Save to specific directory
process_multiple_files("output/*.nc"; output_dir="pressure_results/")
```
"""
function process_multiple_files(file_pattern::String; output_dir::String="")
    # Find matching files
    files = filter(isfile, Glob.glob(file_pattern))

    if isempty(files)
        @warn "No files found matching pattern: $file_pattern"
        return
    end

    println("Found $(length(files)) files to process")

    # Create output directory if specified
    if !isempty(output_dir) && !isdir(output_dir)
        mkpath(output_dir)
        println("Created output directory: $output_dir")
    end

    # Process each file
    for (i, input_file) in enumerate(files)
        println("\n" * "=" ^ 70)
        println("Processing file $i / $(length(files))")
        println("=" ^ 70)

        try
            # Generate output filename
            output_file = if !isempty(output_dir)
                base = basename(input_file)
                base = replace(base, r"\.nc$" => "")
                joinpath(output_dir, base * "_pressure.nc")
            else
                ""  # Auto-generate in same directory
            end

            compute_pressure_from_output(input_file; output_file=output_file)

        catch e
            @error "Failed to process $input_file" exception=(e, catch_backtrace())
            continue
        end
    end

    println("\n" * "=" ^ 70)
    println("Batch processing complete!")
    println("Processed $(length(files)) files")
    println("=" ^ 70)
end


# ================================================================================
# Command-line interface
# ================================================================================

"""
Main function for command-line execution.
"""
function main(args=ARGS)
    if length(args) == 0
        println("""
Usage: julia --project scripts/compute_pressure.jl INPUT_FILE [OUTPUT_FILE]

Arguments:
  INPUT_FILE   - Path to NetCDF output file from GeoDynamo simulation
  OUTPUT_FILE  - (Optional) Path for output pressure file

Examples:
  # Single file
  julia --project scripts/compute_pressure.jl output/geodynamo_rank_0000_time_100.nc

  # Specify output location
  julia --project scripts/compute_pressure.jl output/fields.nc results/pressure.nc

  # With MPI (for parallel processing)
  mpiexecjl -n 4 julia --project scripts/compute_pressure.jl output/fields.nc

From Julia REPL:
  using GeoDynamo
  include("scripts/compute_pressure.jl")
  pressure = compute_pressure_from_output("output/geodynamo_rank_0000_time_100.nc")
""")
        return 1
    end

    input_file = args[1]
    output_file = length(args) >= 2 ? args[2] : ""

    # Initialize MPI if not already initialized
    if !MPI.Initialized()
        MPI.Init()
    end

    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    try
        # Only rank 0 prints header
        if rank == 0
            compute_pressure_from_output(input_file; output_file=output_file)
        else
            # Other ranks participate in computation but don't print
            compute_pressure_from_output(input_file; output_file=output_file)
        end

        if rank == 0
            println("\nAll done!")
        end

        return 0

    catch e
        if rank == 0
            println("\nERROR: $e")
            if isa(e, InterruptException)
                println("Interrupted by user")
            else
                showerror(stdout, e, catch_backtrace())
            end
        end
        return 1
    end
end


# Run main if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    # Initialize GeoDynamo
    if !isdefined(Main, :GeoDynamo)
        using GeoDynamo
    end

    # Initialize parameters
    initialize_parameters()
    initialize_global_parameters!()

    exit_code = main(ARGS)
    exit(exit_code)
else
    # Script was included, just show info
    println("""
Pressure Poisson Solver loaded.

Main functions:
  - compute_pressure_from_output(input_file; output_file="")
    Load fields from NetCDF and compute pressure

  - compute_pressure_poisson(velocity_fields, temp, comp, mag, domain, params)
    Compute pressure from field structures directly

  - save_pressure_field(filename, pressure_physical, domain, config)
    Save pressure field to NetCDF

Usage from command line:
  julia --project scripts/compute_pressure.jl INPUT_FILE [OUTPUT_FILE]

Usage from REPL:
  include("scripts/compute_pressure.jl")
  pressure = compute_pressure_from_output("output/fields.nc")
""")
