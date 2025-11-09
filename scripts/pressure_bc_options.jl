"""
Pressure Boundary Condition Options for Poisson Solver

This file documents different boundary condition implementations for the
pressure Poisson equation in geodynamo simulations.
"""

"""
    apply_pressure_bc_dirichlet!(A, rhs_real, rhs_imag, nr)

Apply simple Dirichlet boundary conditions: p = 0 at both boundaries.

# Pros
- Simple to implement
- Fixes the arbitrary constant in pressure
- Computationally stable

# Cons
- Not physically derived from momentum equation
- May introduce small errors near boundaries
"""
function apply_pressure_bc_dirichlet!(A, rhs_real, rhs_imag, nr)
    # Inner boundary: p = 0
    A[1, :] .= 0
    A[1, 1] = 1
    rhs_real[1] = 0
    rhs_imag[1] = 0

    # Outer boundary: p = 0
    A[nr, :] .= 0
    A[nr, nr] = 1
    rhs_real[nr] = 0
    rhs_imag[nr] = 0
end


"""
    apply_pressure_bc_neumann!(A, rhs_real, rhs_imag, nr, dr_matrix,
                                force_r_inner, force_r_outer)

Apply Neumann boundary conditions derived from radial momentum equation:
    ∂p/∂r = (force terms)_r at boundaries

# Physics
At boundaries where velocity is constrained, the radial momentum equation gives:
    ∂p/∂r|_boundary = (buoyancy + Lorentz + viscous stress)_r

For no-slip (u = 0):
    ∂p/∂r = (viscous stress + buoyancy + Lorentz)_r

For stress-free (u_r = 0, ∂u_tangential/∂r = 0):
    ∂p/∂r = (buoyancy + Lorentz)_r  [viscous stress = 0]

# Arguments
- `A`: Poisson operator matrix
- `rhs_real, rhs_imag`: Right-hand side profiles
- `nr`: Number of radial points
- `dr_matrix`: First derivative operator
- `force_r_inner`: Radial force at inner boundary (from RHS computation)
- `force_r_outer`: Radial force at outer boundary (from RHS computation)

# Implementation
The Neumann BC ∂p/∂r = f is implemented by replacing the boundary row
of the matrix equation with the discrete derivative operator.
"""
function apply_pressure_bc_neumann!(A, rhs_real, rhs_imag, nr, dr_matrix,
                                     force_r_inner, force_r_outer)
    # Inner boundary: ∂p/∂r = force_r_inner
    A[1, :] .= 0
    # Apply derivative operator at first point
    for j in 1:nr
        if abs(1 - j) <= dr_matrix.bandwidth
            band_row = dr_matrix.bandwidth + 1 + 1 - j
            if 1 <= band_row <= 2*dr_matrix.bandwidth + 1
                A[1, j] = dr_matrix.data[band_row, j]
            end
        end
    end
    rhs_real[1] = force_r_inner
    rhs_imag[1] = 0  # Typically imaginary part is zero at boundary

    # Outer boundary: ∂p/∂r = force_r_outer
    A[nr, :] .= 0
    # Apply derivative operator at last point
    for j in 1:nr
        if abs(nr - j) <= dr_matrix.bandwidth
            band_row = dr_matrix.bandwidth + 1 + nr - j
            if 1 <= band_row <= 2*dr_matrix.bandwidth + 1
                A[nr, j] = dr_matrix.data[band_row, j]
            end
        end
    end
    rhs_real[nr] = force_r_outer
    rhs_imag[nr] = 0
end


"""
    apply_pressure_bc_mixed!(A, rhs_real, rhs_imag, nr, dr_matrix)

Apply mixed boundary conditions:
- Inner: Dirichlet p = 0 (fixes arbitrary constant)
- Outer: Neumann ∂p/∂r = 0 (natural outflow condition)

This is a common compromise that:
1. Fixes the pressure constant at inner boundary
2. Allows natural pressure adjustment at outer boundary
"""
function apply_pressure_bc_mixed!(A, rhs_real, rhs_imag, nr, dr_matrix)
    # Inner boundary: Dirichlet p = 0
    A[1, :] .= 0
    A[1, 1] = 1
    rhs_real[1] = 0
    rhs_imag[1] = 0

    # Outer boundary: Neumann ∂p/∂r = 0
    A[nr, :] .= 0
    for j in 1:nr
        if abs(nr - j) <= dr_matrix.bandwidth
            band_row = dr_matrix.bandwidth + 1 + nr - j
            if 1 <= band_row <= 2*dr_matrix.bandwidth + 1
                A[nr, j] = dr_matrix.data[band_row, j]
            end
        end
    end
    rhs_real[nr] = 0
    rhs_imag[nr] = 0
end


"""
    compute_boundary_forces(velocity_fields, temp_field, comp_field,
                            mag_field, domain, params)

Compute the radial forces at boundaries needed for Neumann pressure BC.

Returns (force_inner, force_outer) where each is the radial component
of (buoyancy + Lorentz + viscous_stress) evaluated at the boundary.

# For No-Slip Boundaries
force = (∂²u_r/∂r² + viscous terms) + (buoyancy)_r + (Lorentz)_r

# For Stress-Free Boundaries
force = (buoyancy)_r + (Lorentz)_r  [viscous stress = 0]
"""
function compute_boundary_forces(velocity_fields, temp_field, comp_field,
                                  mag_field, domain, params)
    T = eltype(velocity_fields.toroidal.data_real)

    # Initialize forces
    force_inner = zero(T)
    force_outer = zero(T)

    # Buoyancy contribution at boundaries
    if temp_field !== nothing
        # Thermal buoyancy at boundaries
        rossby_factor = params.d_Pm / params.d_E
        buoyancy_factor = rossby_factor * (params.d_Pm / params.d_Pr) * params.d_Ra

        # Average temperature at boundaries (simplified - could be mode-dependent)
        if isa(temp_field, SHTnsPhysicalField)
            T_data = parent(temp_field.data)
            T_inner = mean(T_data[:, :, 1])
            T_outer = mean(T_data[:, :, end])
        else
            # If spectral, would need to transform
            T_inner = 0
            T_outer = 0
        end

        force_inner += buoyancy_factor * T_inner
        force_outer += buoyancy_factor * T_outer
    end

    # Compositional buoyancy
    if comp_field !== nothing
        rossby_factor = params.d_Pm / params.d_E
        comp_factor = rossby_factor * (params.d_Pm / params.d_Sc) * params.d_Ra_C

        if isa(comp_field, SHTnsPhysicalField)
            C_data = parent(comp_field.data)
            C_inner = mean(C_data[:, :, 1])
            C_outer = mean(C_data[:, :, end])
        else
            C_inner = 0
            C_outer = 0
        end

        force_inner += comp_factor * C_inner
        force_outer += comp_factor * C_outer
    end

    # Lorentz force contribution
    if mag_field !== nothing
        # Would need to compute (j×B)_r at boundaries
        # Simplified for now - in practice, evaluate from spectral coefficients
        force_inner += 0  # Compute from mag_field
        force_outer += 0
    end

    # Viscous stress for no-slip boundaries
    # For stress-free, this is zero by definition
    # For no-slip, would compute from ∂²u_r/∂r² at boundary
    if params.i_vel_bc == 1  # no-slip
        # Viscous contribution (simplified)
        # force_inner += viscous_term_inner
        # force_outer += viscous_term_outer
    end

    return (force_inner, force_outer)
end


"""
Summary of Boundary Condition Options for Pressure Poisson Equation

# Option 1: Dirichlet (Current Implementation)
BC: p = 0 at both boundaries

Pros:
  ✓ Simple and stable
  ✓ Fixes arbitrary constant in pressure
  ✓ Common in literature

Cons:
  ✗ Not physically derived
  ✗ May have errors near boundaries

Use when: Quick results needed, boundary effects not critical


# Option 2: Neumann (Physically Correct)
BC: ∂p/∂r = (forces)_r at boundaries

From radial momentum equation at boundaries:
  - No-slip: ∂p/∂r = viscous_stress + buoyancy + Lorentz
  - Stress-free: ∂p/∂r = buoyancy + Lorentz

Pros:
  ✓ Physically consistent with momentum equation
  ✓ Correct pressure gradient at boundaries
  ✓ Better accuracy overall

Cons:
  ✗ More complex to implement
  ✗ Needs force evaluation at boundaries
  ✗ Pressure constant must be fixed separately

Use when: High accuracy needed, especially near boundaries


# Option 3: Mixed (Recommended Compromise)
BC: p = 0 at inner, ∂p/∂r = 0 at outer

Pros:
  ✓ Fixes pressure constant at inner boundary
  ✓ Natural outflow at outer boundary
  ✓ Good balance of simplicity and accuracy

Cons:
  ✗ Inner BC still not fully physical

Use when: Good general-purpose choice for most simulations


# Implementation Notes

1. For incompressible flow, pressure is unique only up to constant
   → Need at least one Dirichlet BC to fix this

2. For full Neumann BCs on both boundaries:
   → Must add constraint ∫p dV = 0 to fix constant
   → Or use pressure with mean removed

3. Boundary forces depend on:
   - Velocity BC type (no-slip vs stress-free)
   - Presence of buoyancy
   - Presence of magnetic field
   - Viscous terms

4. For l=0 mode:
   - Special treatment may be needed
   - Represents spherically symmetric component
   - Sets overall pressure level
"""

# Example: How to modify solve_pressure_poisson! to use different BCs
function solve_pressure_poisson_with_neumann!(pressure_field, rhs_field,
                                               domain, velocity_fields,
                                               temp_field, comp_field, mag_field,
                                               params)
    # ... [setup code same as before] ...

    # Compute boundary forces
    force_inner, force_outer = compute_boundary_forces(
        velocity_fields, temp_field, comp_field, mag_field, domain, params
    )

    # In the mode loop, replace BC application with:
    # apply_pressure_bc_neumann!(A, rhs_profile_real, rhs_profile_imag,
    #                             nr, dr_matrix, force_inner, force_outer)

    # ... [rest of solver] ...
end

println("""
Pressure Boundary Condition Options Loaded

Available BC implementations:
  - apply_pressure_bc_dirichlet!  (p = 0 at boundaries)
  - apply_pressure_bc_neumann!    (∂p/∂r = forces at boundaries)
  - apply_pressure_bc_mixed!      (p = 0 inner, ∂p/∂r = 0 outer)

Helper functions:
  - compute_boundary_forces       (evaluate force terms at boundaries)

See function docstrings for details on each option.
""")
