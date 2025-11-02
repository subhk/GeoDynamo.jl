# ================================================================================
# Magnetic Field Boundary Conditions
# ================================================================================

# Note: This file is included within the BoundaryConditions module
# All necessary packages are imported at the module level

"""
    load_magnetic_boundary_conditions!(magnetic_field, boundary_specs::Dict)

Load magnetic field boundary conditions from various sources.

# Arguments
- `magnetic_field`: SHTnsMagneticField structure
- `boundary_specs`: Dictionary specifying boundary sources

# Examples
```julia
# Insulating inner, potential field outer
boundary_specs = Dict(
    :inner => (:insulating, 0.0),
    :outer => (:potential_field, "geomagnetic_coefficients.nc")
)

# Perfect conductor boundaries
boundary_specs = Dict(
    :inner => (:perfect_conductor, 0.0),
    :outer => (:perfect_conductor, 0.0)
)

# NetCDF files for both boundaries
boundary_specs = Dict(
    :inner => "cmb_magnetic.nc",
    :outer => "surface_magnetic.nc"
)
```
"""
function load_magnetic_boundary_conditions!(magnetic_field, boundary_specs::Dict)
    
    if get_rank() == 0
        @info "Loading magnetic field boundary conditions..."
    end
    
    # Determine boundary types
    inner_spec = get(boundary_specs, :inner, nothing)
    outer_spec = get(boundary_specs, :outer, nothing)
    
    if inner_spec === nothing || outer_spec === nothing
        throw(ArgumentError("Both :inner and :outer boundary specifications required"))
    end
    
    # Load or generate boundary data
    if isa(inner_spec, String) && isa(outer_spec, String)
        # Both from NetCDF files
        boundary_set = load_magnetic_boundaries_from_files(inner_spec, outer_spec, magnetic_field.config)
    elseif isa(inner_spec, String) && isa(outer_spec, Tuple)
        # Inner from file, outer programmatic
        boundary_set = create_hybrid_magnetic_boundaries(inner_spec, outer_spec, magnetic_field.config)
    elseif isa(inner_spec, Tuple) && isa(outer_spec, String)
        # Inner programmatic, outer from file
        boundary_set = create_hybrid_magnetic_boundaries(outer_spec, inner_spec, magnetic_field.config, swap_boundaries=true)
    elseif isa(inner_spec, Tuple) && isa(outer_spec, Tuple)
        # Both programmatic
        boundary_set = create_programmatic_magnetic_boundaries(inner_spec, outer_spec, magnetic_field.config)
    else
        throw(ArgumentError("Invalid boundary specification format"))
    end
    
    # Store boundary conditions in field
    magnetic_field.boundary_condition_set = boundary_set
    magnetic_field.boundary_time_index[] = 1
    
    # Create interpolation cache
    magnetic_field.boundary_interpolation_cache = create_magnetic_interpolation_cache(boundary_set, magnetic_field.config)
    
    # Apply initial boundary conditions
    apply_magnetic_boundary_conditions!(magnetic_field)
    
    if get_rank() == 0
        print_boundary_info(boundary_set)
        @info "Magnetic field boundary conditions loaded successfully"
    end
    
    return magnetic_field
end

"""
    load_magnetic_boundaries_from_files(inner_file::String, outer_file::String, config)

Load magnetic field boundary conditions from NetCDF files.
"""
function load_magnetic_boundaries_from_files(inner_file::String, outer_file::String, config)
    
    # Validate files exist
    for file in [inner_file, outer_file]
        if !isfile(file)
            throw(ArgumentError("Magnetic boundary file not found: $file"))
        end
    end
    
    # Read boundary data
    inner_data = read_netcdf_boundary_data(inner_file, precision=config.T)
    outer_data = read_netcdf_boundary_data(outer_file, precision=config.T)
    
    # Update field type for magnetic field
    inner_data.field_type = "magnetic"
    outer_data.field_type = "magnetic"
    
    # Validate vector field dimensions (should have 3 components: B_r, B_θ, B_φ)
    if inner_data.ncomponents != 3 || outer_data.ncomponents != 3
        throw(ArgumentError("Magnetic boundary conditions require 3 components (B_r, B_θ, B_φ)"))
    end
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "magnetic")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "magnetic", MAGNETIC, time()
    )
    
    return boundary_set
end

"""
    create_hybrid_magnetic_boundaries(file_spec::String, prog_spec::Tuple, config; swap_boundaries=false)

Create hybrid magnetic boundaries (one from file, one programmatic).
"""
function create_hybrid_magnetic_boundaries(file_spec::String, prog_spec::Tuple, config; swap_boundaries=false)
    
    # Load file-based boundary
    file_data = read_netcdf_boundary_data(file_spec, precision=config.T)
    file_data.field_type = "magnetic"
    
    # Create programmatic boundary
    pattern, amplitude = prog_spec[1], prog_spec[2]
    parameters = length(prog_spec) >= 3 ? prog_spec[3] : Dict()
    
    prog_data = create_programmatic_magnetic_boundary(
        pattern, config, amplitude; parameters=parameters
    )
    
    # Ensure same grid resolution
    if file_data.nlat != config.nlat || file_data.nlon != config.nlon
        # Interpolate file data to config grid
        theta_target = collect(range(0, π, length=config.nlat))
        phi_target = collect(range(0, 2π, length=config.nlon+1)[1:end-1])
        
        interpolated_values = interpolate_boundary_to_grid(file_data, theta_target, phi_target, 1)
        
        file_data = create_boundary_data(
            interpolated_values, "magnetic";
            theta=theta_target, phi=phi_target, time=nothing,
            units=file_data.units, description=file_data.description,
            file_path=file_data.file_path
        )
    end
    
    # Assign boundaries based on swap_boundaries flag
    if swap_boundaries
        inner_data, outer_data = prog_data, file_data
    else
        inner_data, outer_data = file_data, prog_data
    end
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "magnetic")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "magnetic", MAGNETIC, time()
    )
    
    return boundary_set
end

"""
    create_programmatic_magnetic_boundaries(inner_spec::Tuple, outer_spec::Tuple, config)

Create fully programmatic magnetic boundaries.
"""
function create_programmatic_magnetic_boundaries(inner_spec::Tuple, outer_spec::Tuple, config)
    
    # Create inner boundary
    inner_pattern, inner_amplitude = inner_spec[1], inner_spec[2]
    inner_parameters = length(inner_spec) >= 3 ? inner_spec[3] : Dict()
    
    inner_data = create_programmatic_magnetic_boundary(
        inner_pattern, config, inner_amplitude; parameters=inner_parameters
    )
    
    # Create outer boundary
    outer_pattern, outer_amplitude = outer_spec[1], outer_spec[2]
    outer_parameters = length(outer_spec) >= 3 ? outer_spec[3] : Dict()
    
    outer_data = create_programmatic_magnetic_boundary(
        outer_pattern, config, outer_amplitude; parameters=outer_parameters
    )
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "magnetic")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "magnetic", MAGNETIC, time()
    )
    
    return boundary_set
end

"""
    create_programmatic_magnetic_boundary(pattern::Symbol, config, amplitude::Real=1.0; 
                                        parameters::Dict=Dict())

Create programmatically generated magnetic boundary conditions.

# Available patterns:
- `:insulating` - Insulating boundary (B_r = 0, ∂B_tan/∂r = 0)
- `:perfect_conductor` - Perfect conductor (B_tan = 0)
- `:dipole` - Dipolar magnetic field pattern
- `:quadrupole` - Quadrupolar magnetic field pattern
- `:potential_field` - Potential field from spherical harmonic coefficients
- `:uniform_field` - Uniform magnetic field
- `:custom` - User-defined magnetic field function
"""
function create_programmatic_magnetic_boundary(pattern::Symbol, config, amplitude::Real=1.0;
                                             parameters::Dict=Dict())
    
    # Create coordinate grids
    nlat, nlon = config.nlat, config.nlon
    theta = collect(range(0, π, length=nlat))
    phi = collect(range(0, 2π, length=nlon+1)[1:end-1])
    
    # Initialize magnetic field components array [nlat, nlon, 3] for (B_r, B_θ, B_φ)
    values = zeros(config.T, nlat, nlon, 3)
    
    # Generate magnetic field pattern
    if pattern == :insulating
        # Insulating boundary: No current can flow across boundary (σ = 0)
        # Physical constraint: J_n = 0 at boundary
        # From Ampère's law: ∇×B = μ₀J, so (∇×B)_r = μ₀J_r = 0
        # This means: (1/(r sin θ))[∂(B_φ sin θ)/∂θ - ∂B_θ/∂φ] = 0
        #
        # For potential field matching (typical insulating BC):
        # - B_r matches internal/external potential field
        # - Tangential components match potential field (continuous)
        # - The key is that field lines cross the boundary normally
        #
        # Use a dipole potential field as default:
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                # Dipolar potential field pattern (appropriate for insulating boundary)
                values[i, j, 1] = amplitude * 2 * cos(θ)     # B_r (radial component)
                values[i, j, 2] = amplitude * sin(θ)         # B_θ (from potential)
                values[i, j, 3] = 0.0                        # B_φ = 0 (axisymmetric dipole)
            end
        end
        description = "Insulating boundary condition (potential field, J_n=0)"
        
    elseif pattern == :perfect_conductor
        # Perfect conductor: B_tangential = 0 at boundary
        # Physical constraint: Tangential E and H vanish inside perfect conductor
        # This gives B_θ = B_φ = 0 at the boundary
        # B_r is determined by ∇·B = 0 (solenoidal constraint) and matching
        #
        # Use dipole radial pattern as reasonable default for B_r:
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = amplitude * 2 * cos(θ)  # B_r (dipolar pattern)
                values[i, j, 2] = 0.0                     # B_θ = 0 (perfect conductor)
                values[i, j, 3] = 0.0                     # B_φ = 0 (perfect conductor)
            end
        end
        description = "Perfect conductor boundary condition (B_tan=0)"
        
    elseif pattern == :dipole
        # Dipolar magnetic field: B ∝ (2cos(θ)ê_r + sin(θ)ê_θ)
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = amplitude * 2 * cos(θ)     # B_r
                values[i, j, 2] = amplitude * sin(θ)         # B_θ
                values[i, j, 3] = 0.0                        # B_φ = 0 (axisymmetric)
            end
        end
        description = "Dipolar magnetic field (amplitude = $amplitude T)"
        
    elseif pattern == :quadrupole
        # Quadrupolar field: B_r ∝ (3cos²θ - 1), B_θ ∝ sin(2θ)
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = amplitude * (3 * cos(θ)^2 - 1)  # B_r
                values[i, j, 2] = amplitude * sin(2 * θ)           # B_θ  
                values[i, j, 3] = 0.0                              # B_φ = 0
            end
        end
        description = "Quadrupolar magnetic field (amplitude = $amplitude T)"
        
    elseif pattern == :potential_field
        # Potential field from spherical harmonic coefficients
        # Requires coefficients in parameters
        if !haskey(parameters, "coefficients")
            # Default to dipole if no coefficients provided
            return create_programmatic_magnetic_boundary(:dipole, config, amplitude; parameters=parameters)
        end
        
        coeffs = parameters["coefficients"]
        lmax = get(parameters, "lmax", 10)
        
        # Calculate field from potential using SHTnsKit
        potential_field = calculate_potential_field_boundary(coeffs, theta, phi, lmax)
        values[:, :, :] = amplitude * potential_field
        
        description = "Potential magnetic field (lmax = $lmax, amplitude = $amplitude T)"
        
    elseif pattern == :uniform_field
        # Uniform magnetic field in specified direction
        direction = get(parameters, "direction", [0.0, 0.0, 1.0])  # Default: z-direction
        direction = direction ./ norm(direction)  # Normalize
        
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                # Convert Cartesian direction to spherical components
                # B_r = B⃗ · ê_r = Bₓsin(θ)cos(φ) + Bᵧsin(θ)sin(φ) + Bᵤcos(θ)
                # B_θ = B⃗ · ê_θ = Bₓcos(θ)cos(φ) + Bᵧcos(θ)sin(φ) - Bᵤsin(θ)
                # B_φ = B⃗ · ê_φ = -Bₓsin(φ) + Bᵧcos(φ)
                
                Bx, By, Bz = direction
                sin_theta, cos_theta = sin(θ), cos(θ)
                sin_phi, cos_phi = sin(φ), cos(φ)
                
                values[i, j, 1] = amplitude * (Bx*sin_theta*cos_phi + By*sin_theta*sin_phi + Bz*cos_theta)  # B_r
                values[i, j, 2] = amplitude * (Bx*cos_theta*cos_phi + By*cos_theta*sin_phi - Bz*sin_theta)  # B_θ
                values[i, j, 3] = amplitude * (-Bx*sin_phi + By*cos_phi)                                     # B_φ
            end
        end
        description = "Uniform magnetic field (direction = $direction, amplitude = $amplitude T)"
        
    elseif pattern == :custom
        # User-defined magnetic field function
        if !haskey(parameters, "function")
            throw(ArgumentError("Custom magnetic pattern requires 'function' in parameters"))
        end
        
        user_func = parameters["function"]
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                B_r, B_theta, B_phi = user_func(θ, φ)
                values[i, j, 1] = amplitude * B_r
                values[i, j, 2] = amplitude * B_theta
                values[i, j, 3] = amplitude * B_phi
            end
        end
        description = "Custom magnetic field pattern (amplitude = $amplitude T)"
        
    else
        throw(ArgumentError("Unknown magnetic field pattern: $pattern"))
    end
    
    # Create BoundaryData structure
    boundary_data = create_boundary_data(
        values, "magnetic";
        theta=theta, phi=phi, time=nothing,
        units="T",
        description=description,
        file_path="programmatic"
    )

    # Add pattern information for constraint enforcement
    boundary_data.pattern = pattern
    boundary_data.amplitude = amplitude

    return boundary_data
end

"""
    calculate_potential_field_boundary(coeffs::Dict, theta::Vector, phi::Vector, lmax::Int)

Calculate magnetic field from spherical harmonic coefficients of the potential.
"""
function calculate_potential_field_boundary(coeffs::Dict, theta::Vector, phi::Vector, lmax::Int)
    
    nlat, nlon = length(theta), length(phi)
    B_field = zeros(nlat, nlon, 3)
    
    # Calculate field components from potential derivatives
    for (i, θ) in enumerate(theta)
        for (j, φ) in enumerate(phi)
            
            B_r = 0.0
            B_theta = 0.0
            B_phi = 0.0
            
            # Sum over spherical harmonic modes
            for l in 1:lmax
                for m in -l:l
                    # Get coefficient for this (l,m) mode
                    coeff_key = "$(l)_$(m)"
                    if haskey(coeffs, coeff_key)
                        coeff = coeffs[coeff_key]
                        
                        # Calculate spherical harmonic and derivatives
                        Ylm = spherical_harmonic(l, m, θ, φ)
                        dYlm_dtheta = spherical_harmonic_theta_derivative(l, m, θ, φ)
                        dYlm_dphi = spherical_harmonic_phi_derivative(l, m, θ, φ)

                        # Magnetic field from potential: B = -∇V
                        # For V = Σ V_lm(r) Y_lm(θ,φ)
                        #
                        # Assuming INTERNAL potential: V_lm(r) = A_lm r^l
                        # Then ∂V_lm/∂r = l * A_lm r^(l-1) = l/r * V_lm
                        # At boundary r=r₀: B_r = -l * V_lm/r₀ * Y_lm
                        #
                        # For EXTERNAL potential: V_lm(r) = B_lm r^(-l-1)
                        # Then ∂V_lm/∂r = -(l+1) * B_lm r^(-l-2) = -(l+1)/r * V_lm
                        # At boundary r=r₀: B_r = (l+1) * V_lm/r₀ * Y_lm
                        #
                        # Using external potential (typical for outer boundary):
                        # B_r = -∂V/∂r = (l+1)/r * V_lm * Y_lm
                        # B_θ = -1/r * ∂V/∂θ = -1/r * V_lm * dY_lm/dθ
                        # B_φ = -1/(r sin θ) * ∂V/∂φ = -1/(r sin θ) * V_lm * dY_lm/dφ
                        #
                        # At r=1 (normalized boundary):
                        B_r += (l + 1) * coeff * Ylm
                        B_theta += -coeff * dYlm_dtheta
                        B_phi += -coeff * dYlm_dphi / (sin(θ) + 1e-15)  # Avoid division by zero
                    end
                end
            end
            
            B_field[i, j, 1] = B_r
            B_field[i, j, 2] = B_theta
            B_field[i, j, 3] = B_phi
        end
    end
    
    return B_field
end

"""
    spherical_harmonic(l::Int, m::Int, theta::Real, phi::Real)

Compute spherical harmonic Y_l^m(θ, φ).
"""
function spherical_harmonic(l::Int, m::Int, theta::Real, phi::Real)
    # Simplified implementation - in practice would use SHTnsKit
    # For now, return basic patterns for common modes
    
    if l == 1 && m == 0
        return cos(theta)  # Y₁₀
    elseif l == 1 && m == 1
        return sin(theta) * cos(phi)  # Y₁₁ (real part)
    elseif l == 1 && m == -1
        return sin(theta) * sin(phi)  # Y₁₁ (imaginary part)
    elseif l == 2 && m == 0
        return 0.5 * (3 * cos(theta)^2 - 1)  # Y₂₀
    else
        return 0.0  # Placeholder for other modes
    end
end

"""
    spherical_harmonic_theta_derivative(l::Int, m::Int, theta::Real, phi::Real)

Compute ∂Y_l^m/∂θ.
"""
function spherical_harmonic_theta_derivative(l::Int, m::Int, theta::Real, phi::Real)
    # Simplified implementation
    if l == 1 && m == 0
        return -sin(theta)
    elseif l == 1 && m == 1
        return cos(theta) * cos(phi)
    elseif l == 1 && m == -1
        return cos(theta) * sin(phi)
    elseif l == 2 && m == 0
        return -3 * cos(theta) * sin(theta)
    else
        return 0.0
    end
end

"""
    spherical_harmonic_phi_derivative(l::Int, m::Int, theta::Real, phi::Real)

Compute ∂Y_l^m/∂φ.
"""
function spherical_harmonic_phi_derivative(l::Int, m::Int, theta::Real, phi::Real)
    # Simplified implementation
    if l == 1 && m == 0
        return 0.0
    elseif l == 1 && m == 1
        return -sin(theta) * sin(phi)
    elseif l == 1 && m == -1
        return sin(theta) * cos(phi)
    elseif l == 2 && m == 0
        return 0.0
    else
        return 0.0
    end
end

"""
    create_magnetic_interpolation_cache(boundary_set::BoundaryConditionSet, config)

Create interpolation cache for magnetic boundaries.
"""
function create_magnetic_interpolation_cache(boundary_set::BoundaryConditionSet, config)
    
    cache = Dict{String, Any}()
    
    # Create target grid (simulation grid)
    theta_target = collect(range(0, π, length=config.nlat))
    phi_target = collect(range(0, 2π, length=config.nlon+1)[1:end-1])
    
    # Create interpolation caches
    cache["inner"] = create_interpolation_cache(boundary_set.inner_boundary, theta_target, phi_target)
    cache["outer"] = create_interpolation_cache(boundary_set.outer_boundary, theta_target, phi_target)
    
    return cache
end

"""
    apply_magnetic_boundary_conditions!(magnetic_field, time_index::Int=1)

Apply magnetic field boundary conditions to the field.
"""
function apply_magnetic_boundary_conditions!(magnetic_field, time_index::Int=1)
    
    if magnetic_field.boundary_condition_set === nothing
        @warn "No boundary conditions loaded for magnetic field"
        return magnetic_field
    end
    
    boundary_set = magnetic_field.boundary_condition_set
    cache = magnetic_field.boundary_interpolation_cache
    
    # Interpolate boundary data to simulation grid
    inner_physical = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
    outer_physical = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
    
    # Transform to spectral space using proper magnetic QST decomposition
    # For magnetic fields: B = Q ê_r + S_tangential + T_tangential
    # where Q is radial, S is spheroidal (potential-like), T is toroidal (solenoidal)

    # Extract components
    B_r_inner = inner_physical[:, :, 1]     # Radial component
    B_theta_inner = inner_physical[:, :, 2]  # Theta component
    B_phi_inner = inner_physical[:, :, 3]    # Phi component

    B_r_outer = outer_physical[:, :, 1]
    B_theta_outer = outer_physical[:, :, 2]
    B_phi_outer = outer_physical[:, :, 3]

    # Convert to QST coefficients using SHTnsKit (if available)
    try
        # Inner boundary
        Q_inner, S_inner, T_inner = magnetic_to_qst_coefficients(
            B_r_inner, B_theta_inner, B_phi_inner, magnetic_field.config
        )

        # Outer boundary
        Q_outer, S_outer, T_outer = magnetic_to_qst_coefficients(
            B_r_outer, B_theta_outer, B_phi_outer, magnetic_field.config
        )

        # Map to toroidal-poloidal structure:
        # For magnetic fields: toroidal ~ T (purely tangential), poloidal ~ Q + S (radial + potential)
        magnetic_field.toroidal.boundary_values[1, :] = T_inner   # Inner toroidal
        magnetic_field.toroidal.boundary_values[2, :] = T_outer   # Outer toroidal

        # Combine Q and S for poloidal
        # For solenoidal fields (∇·B = 0), the decomposition is:
        # - Q: radial component (B_r)
        # - S: should be zero for purely solenoidal fields
        # - T: tangential toroidal component
        magnetic_field.poloidal.boundary_values[1, :] = Q_inner  # Inner poloidal (radial)
        magnetic_field.poloidal.boundary_values[2, :] = Q_outer  # Outer poloidal (radial)

        # Check S component magnitude to verify solenoidal assumption
        S_norm_inner = sqrt(sum(abs2, S_inner))
        S_norm_outer = sqrt(sum(abs2, S_outer))
        Q_norm_inner = sqrt(sum(abs2, Q_inner))
        Q_norm_outer = sqrt(sum(abs2, Q_outer))
        T_norm_inner = sqrt(sum(abs2, T_inner))
        T_norm_outer = sqrt(sum(abs2, T_outer))

        if (S_norm_inner > 0.01 * max(Q_norm_inner, T_norm_inner) ||
            S_norm_outer > 0.01 * max(Q_norm_outer, T_norm_outer)) && get_rank() == 0
            @warn """
            Non-negligible spheroidal (S) component detected in magnetic boundary conditions!
            S_inner: $(S_norm_inner) vs Q_inner: $(Q_norm_inner), T_inner: $(T_norm_inner)
            S_outer: $(S_norm_outer) vs Q_outer: $(Q_norm_outer), T_outer: $(T_norm_outer)
            Ratios: S_inner/Q_inner = $(S_norm_inner/max(Q_norm_inner, 1e-10))
                    S_outer/Q_outer = $(S_norm_outer/max(Q_norm_outer, 1e-10))

            This suggests the magnetic boundary conditions may not be solenoidal (∇·B ≠ 0).
            For magnetic fields, we expect ∇·B = 0, which implies S should be negligible.

            The current implementation assumes solenoidal fields and IGNORES the S component.
            If your boundary conditions are non-solenoidal, the code needs extension to handle
            the spheroidal component properly.
            """
        end

    catch e
        error_msg = """
        Failed to perform proper QST decomposition of magnetic field boundary conditions.
        Error: $e

        The QST decomposition is mathematically required for correct magnetic boundary conditions.
        The previous fallback (treating B_r as toroidal and magnitude of tangential as poloidal)
        was fundamentally incorrect because:
        1. It confused radial and tangential components
        2. It took magnitudes of vector components (destroying directional information)
        3. It treated components as independent scalars

        Proper decomposition requires:
        - Q: radial component (B_r)
        - S: curl-free tangential part
        - T: solenoidal tangential part
        where B_θ and B_φ are coupled through S and T.

        Possible solutions:
        1. Check that SHTnsKit.spat_to_SHqst is properly installed and configured
        2. Verify grid dimensions (nlat=$(size(B_r_inner,1)), nlon=$(size(B_r_inner,2)))
           are compatible with lmax=$(magnetic_field.config.lmax)
        3. Ensure SHTnsKit configuration is properly initialized
        """
        throw(ErrorException(error_msg))
    end
    
    # Update time index
    magnetic_field.boundary_time_index[] = time_index

    # Enforce magnetic boundary condition constraints based on boundary pattern
    boundary_set = magnetic_field.boundary_condition_set

    # Determine boundary constraint type
    primary_constraint = :potential_field  # Default

    # Check for pattern information in boundary data
    if hasfield(typeof(boundary_set.inner_boundary), :pattern)
        primary_constraint = boundary_set.inner_boundary.pattern
    elseif hasfield(typeof(boundary_set.outer_boundary), :pattern)
        primary_constraint = boundary_set.outer_boundary.pattern
    elseif hasfield(typeof(boundary_set), :constraint_type)
        # Use explicit constraint if specified
        primary_constraint = boundary_set.constraint_type
    end

    # Apply the determined constraint
    enforce_magnetic_boundary_constraints!(magnetic_field, primary_constraint)

    return magnetic_field
end

"""
    update_time_dependent_magnetic_boundaries!(magnetic_field, current_time::Float64)

Update time-dependent magnetic boundary conditions.
"""
function update_time_dependent_magnetic_boundaries!(magnetic_field, current_time::Float64)
    
    if magnetic_field.boundary_condition_set === nothing
        return magnetic_field
    end
    
    boundary_set = magnetic_field.boundary_condition_set
    
    # Check if boundaries are time-dependent
    if !boundary_set.inner_boundary.is_time_dependent && !boundary_set.outer_boundary.is_time_dependent
        return magnetic_field  # Nothing to update
    end
    
    # Find time index for current time
    time_index = find_boundary_time_index(boundary_set, current_time)
    
    # Only update if time index has changed
    if time_index != magnetic_field.boundary_time_index[]
        apply_magnetic_boundary_conditions!(magnetic_field, time_index)
        
        if get_rank() == 0
            @info "Updated magnetic boundaries to time index $time_index (t=$current_time)"
        end
    end
    
    return magnetic_field
end

"""
    get_current_magnetic_boundaries(magnetic_field)

Get current magnetic field boundary conditions.
"""
function get_current_magnetic_boundaries(magnetic_field)
    
    if magnetic_field.boundary_condition_set === nothing
        return Dict(:error => "No boundary conditions loaded")
    end
    
    boundary_set = magnetic_field.boundary_condition_set
    time_index = magnetic_field.boundary_time_index[]
    cache = magnetic_field.boundary_interpolation_cache
    
    # Get current boundary data
    inner_physical = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
    outer_physical = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
    
    # Get spectral coefficients
    inner_toroidal_spectral = magnetic_field.toroidal.boundary_values[1, :]
    outer_toroidal_spectral = magnetic_field.toroidal.boundary_values[2, :]
    inner_poloidal_spectral = magnetic_field.poloidal.boundary_values[1, :]
    outer_poloidal_spectral = magnetic_field.poloidal.boundary_values[2, :]
    
    return Dict(
        :inner_physical => inner_physical,
        :outer_physical => outer_physical,
        :inner_toroidal_spectral => inner_toroidal_spectral,
        :outer_toroidal_spectral => outer_toroidal_spectral,
        :inner_poloidal_spectral => inner_poloidal_spectral,
        :outer_poloidal_spectral => outer_poloidal_spectral,
        :time_index => time_index,
        :metadata => Dict(
            "field_name" => boundary_set.field_name,
            "source" => "file_based",
            "inner_file" => boundary_set.inner_boundary.file_path,
            "outer_file" => boundary_set.outer_boundary.file_path,
            "creation_time" => boundary_set.creation_time,
            "components" => ["B_r", "B_theta", "B_phi"]
        )
    )
end

"""
    set_programmatic_magnetic_boundaries!(magnetic_field, inner_spec::Tuple, outer_spec::Tuple)

Set programmatic magnetic boundary conditions.
"""
function set_programmatic_magnetic_boundaries!(magnetic_field, inner_spec::Tuple, outer_spec::Tuple)
    
    boundary_specs = Dict(:inner => inner_spec, :outer => outer_spec)
    return load_magnetic_boundary_conditions!(magnetic_field, boundary_specs)
end

"""
    validate_magnetic_boundary_files(boundary_specs::Dict, config)

Validate magnetic field boundary condition files.
"""
function validate_magnetic_boundary_files(boundary_specs::Dict, config)
    
    inner_spec = get(boundary_specs, :inner, nothing)
    outer_spec = get(boundary_specs, :outer, nothing)
    
    errors = String[]
    
    # Validate file specifications
    if isa(inner_spec, String)
        try
            validate_netcdf_boundary_file(inner_spec, ["magnetic", "b", "B"])
            # Check vector components
            inner_data = read_netcdf_boundary_data(inner_spec, precision=config.T)
            if inner_data.ncomponents != 3
                push!(errors, "Inner magnetic file must have 3 components (B_r, B_theta, B_phi)")
            end
        catch e
            push!(errors, "Inner boundary file error: $e")
        end
    end
    
    if isa(outer_spec, String)
        try
            validate_netcdf_boundary_file(outer_spec, ["magnetic", "b", "B"])
            # Check vector components
            outer_data = read_netcdf_boundary_data(outer_spec, precision=config.T)
            if outer_data.ncomponents != 3
                push!(errors, "Outer magnetic file must have 3 components (B_r, B_theta, B_phi)")
            end
        catch e
            push!(errors, "Outer boundary file error: $e")
        end
    end
    
    # If both are files, check compatibility
    if isa(inner_spec, String) && isa(outer_spec, String)
        try
            inner_data = read_netcdf_boundary_data(inner_spec, precision=config.T)
            outer_data = read_netcdf_boundary_data(outer_spec, precision=config.T)
            validate_boundary_compatibility(inner_data, outer_data, "magnetic")
        catch e
            push!(errors, "Boundary compatibility error: $e")
        end
    end
    
    if !isempty(errors)
        error_msg = "Magnetic boundary validation failed:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
    
    return true
end

"""
    magnetic_to_qst_coefficients(B_r, B_theta, B_phi, config)

Convert physical magnetic field components to QST spectral coefficients.

For magnetic fields:
- Q: Radial component coefficients (B_r)
- S: Spheroidal tangential component coefficients
- T: Toroidal tangential component coefficients
"""
function magnetic_to_qst_coefficients(B_r, B_theta, B_phi, config)

    try
        # Use SHTnsKit for proper QST decomposition (it's imported at module level)
        # Create SHTnsKit configuration
        lmax = config.lmax
        nlat, nlon = size(B_r)

        shtconfig = SHTnsKit.create_gauss_config(lmax, nlat; mmax=config.mmax, nlon=nlon)

        # Transform to QST coefficients using SHTnsKit - returns (lmax+1)×(mmax+1) matrices
        Q_matrix, S_matrix, T_matrix = SHTnsKit.spat_to_SHqst(shtconfig, B_r, B_theta, B_phi)

        # Clean up configuration
        SHTnsKit.destroy_config(shtconfig)

        # Convert matrix format to 1D spectral coefficient arrays
        # The boundary code expects 1D arrays of length nlm
        mmax = config.mmax
        nlm = config.nlm

        Q_coeffs = zeros(eltype(B_r), nlm)
        S_coeffs = zeros(eltype(B_r), nlm)
        T_coeffs = zeros(eltype(B_r), nlm)

        # Map from (l,m) matrix to linear index following the simulation's lm-indexing
        # Typically: idx increases as we loop over l, then m (with appropriate m range)
        idx = 0
        for l in 0:lmax
            # Determine m range for this l
            # For complex harmonics: m from -min(l,mmax) to +min(l,mmax)
            # For real harmonics stored efficiently: m from 0 to min(l,mmax)
            m_max = min(l, mmax)

            for m in 0:m_max
                idx += 1
                if idx <= nlm && (l+1) <= size(Q_matrix, 1) && (m+1) <= size(Q_matrix, 2)
                    # SHTnsKit matrices are typically (lmax+1) × (mmax+1) in size
                    # Extract real parts (boundary conditions are typically real)
                    Q_coeffs[idx] = real(Q_matrix[l+1, m+1])
                    S_coeffs[idx] = real(S_matrix[l+1, m+1])
                    T_coeffs[idx] = real(T_matrix[l+1, m+1])
                elseif idx <= nlm
                    # If matrix is smaller than expected, zero-pad
                    Q_coeffs[idx] = zero(eltype(B_r))
                    S_coeffs[idx] = zero(eltype(B_r))
                    T_coeffs[idx] = zero(eltype(B_r))
                end
            end
        end

        # Verify we processed the expected number of modes
        if idx != nlm && get_rank() == 0
            @warn "Magnetic QST extraction: processed $idx modes but nlm=$(nlm). Check lm-indexing consistency."
        end

        return Q_coeffs, S_coeffs, T_coeffs
    catch e
        error_msg = """
        Failed to perform proper QST decomposition of magnetic field components.
        Error: $e

        The QST decomposition is mathematically required for correct magnetic boundary conditions.
        Treating B_r, B_θ, B_φ as independent scalar fields (the old fallback) is incorrect
        because the tangential components B_θ and B_φ are coupled through the spheroidal-toroidal
        decomposition in spherical geometry.

        For magnetic fields:
        - Q: radial component (transforms like scalar: B_r)
        - S: spheroidal tangential (curl-free, from ∇ψ)
        - T: toroidal tangential (solenoidal, from ∇×(T r̂))

        The relationship between (B_θ, B_φ) and (S, T) is non-trivial and requires proper
        vector spherical harmonic decomposition.

        Possible solutions:
        1. Check that SHTnsKit.spat_to_SHqst is properly installed and configured
        2. Verify grid dimensions (nlat=$(size(B_r,1)), nlon=$(size(B_r,2)))
           are compatible with lmax=$(config.lmax)
        3. Ensure SHTnsKit configuration matches the simulation configuration
        """
        throw(ErrorException(error_msg))
    end
end

# Function moved to main BoundaryConditions module to avoid duplication

"""
    enforce_magnetic_boundary_constraints!(magnetic_field, bc_type::Symbol)

Enforce magnetic boundary condition constraints based on magnetohydrodynamic physics.

# Boundary condition types:
- `:insulating` - Insulating boundary (σ = 0): No normal current (J_n = 0)
  * Physical interpretation: (∇×B)_r = μ₀J_r = 0 at boundary
  * Implementation: Potential field matching (B continuous, tangential current-free)
  * Spectral: Typically Dirichlet BC for both components matching external potential

- `:perfect_conductor` - Perfect conductor (σ → ∞): Zero tangential magnetic field
  * Physical interpretation: B_tangential = 0, B_r free (from ∇·B = 0)
  * Implementation: Tangential components zero, radial component determined by matching
  * Spectral: Toroidal (T) = 0 Dirichlet, Poloidal (Q) matches

- `:potential_field` - General potential field matching
  * Implementation: Match external potential field at boundary
  * Spectral: Dirichlet BC for both toroidal and poloidal components

# Notes:
For most geodynamo applications:
- Inner boundary (ICB): Often insulating or potential field
- Outer boundary (CMB): Often insulating (matching Earth's mantle)
"""
function enforce_magnetic_boundary_constraints!(magnetic_field, bc_type::Symbol)

    if bc_type == :insulating
        # Insulating boundary: J_n = 0, which gives (∇×B)_r = 0
        # For potential field matching (typical insulating implementation):
        # - All components match external/internal potential field
        # - Use Dirichlet BC with values from potential field calculation
        #
        # Note: The boundary values should already be set from potential field
        # calculation in apply_magnetic_boundary_conditions!

        # Both components use Dirichlet BC (matching potential field)
        fill!(magnetic_field.toroidal.bc_type_inner, Int(DIRICHLET))
        fill!(magnetic_field.toroidal.bc_type_outer, Int(DIRICHLET))
        fill!(magnetic_field.poloidal.bc_type_inner, Int(DIRICHLET))
        fill!(magnetic_field.poloidal.bc_type_outer, Int(DIRICHLET))

    elseif bc_type == :perfect_conductor
        # Perfect conductor: B_tangential = 0 at boundary
        # Physical: Tangential E and H vanish, giving B_θ = B_φ = 0
        # Spectral: Toroidal component T = 0 (controls tangential field)
        # Radial component (poloidal/Q) is non-zero and determined by ∇·B = 0

        # Set toroidal components to zero (tangential field = 0)
        fill!(magnetic_field.toroidal.boundary_values, 0.0)
        fill!(magnetic_field.toroidal.bc_type_inner, Int(DIRICHLET))  # T = 0 enforced
        fill!(magnetic_field.toroidal.bc_type_outer, Int(DIRICHLET))  # T = 0 enforced

        # Poloidal/radial component uses Dirichlet BC from computed values
        # (determined by ∇·B = 0 and matching conditions)
        fill!(magnetic_field.poloidal.bc_type_inner, Int(DIRICHLET))
        fill!(magnetic_field.poloidal.bc_type_outer, Int(DIRICHLET))

    elseif bc_type == :potential_field
        # Potential field boundary: match external field
        # Both components use computed boundary values as Dirichlet BC

        fill!(magnetic_field.toroidal.bc_type_inner, 1)  # Dirichlet
        fill!(magnetic_field.toroidal.bc_type_outer, 1)  # Dirichlet
        fill!(magnetic_field.poloidal.bc_type_inner, 1)  # Dirichlet
        fill!(magnetic_field.poloidal.bc_type_outer, 1)  # Dirichlet

    elseif bc_type == :custom
        # Custom boundary conditions - leave arrays as set by user
        @info "Custom magnetic boundary conditions - user must set bc_type arrays"

    else
        @warn "Unknown magnetic boundary condition type: $bc_type, using potential_field"
        enforce_magnetic_boundary_constraints!(magnetic_field, :potential_field)
    end

    return magnetic_field
end

export load_magnetic_boundary_conditions!, set_programmatic_magnetic_boundaries!
export update_time_dependent_magnetic_boundaries!, get_current_magnetic_boundaries
export validate_magnetic_boundary_files, create_programmatic_magnetic_boundary
export magnetic_to_qst_coefficients, enforce_magnetic_boundary_constraints!