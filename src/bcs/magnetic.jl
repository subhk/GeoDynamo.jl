# ================================================================================
# Magnetic Field Boundary Conditions
# ================================================================================

# Note: This file is included within the bcs module
# All necessary packages are imported at the module level

"""
    load_magnetic_boundary_conditions!(ℬ, boundary_specs::Dict)

Load magnetic field boundary conditions from various sources.

# Arguments
- `ℬ`: SHTnsMagneticField structure
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
function load_magnetic_boundary_conditions!(ℬ, boundary_specs::Dict)
    
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
        boundary_set = load_magnetic_boundaries_from_files(inner_spec, outer_spec, ℬ.config)
    elseif isa(inner_spec, String) && isa(outer_spec, Tuple)
        # Inner from file, outer programmatic
        boundary_set = create_hybrid_magnetic_boundaries(inner_spec, outer_spec, ℬ.config)
    elseif isa(inner_spec, Tuple) && isa(outer_spec, String)
        # Inner programmatic, outer from file
        boundary_set = create_hybrid_magnetic_boundaries(outer_spec, inner_spec, ℬ.config, swap_boundaries=true)
    elseif isa(inner_spec, Tuple) && isa(outer_spec, Tuple)
        # Both programmatic
        boundary_set = create_programmatic_magnetic_boundaries(inner_spec, outer_spec, ℬ.config)
    else
        throw(ArgumentError("Invalid boundary specification format"))
    end
    
    cache = create_magnetic_interpolation_cache(boundary_set, ℬ.config)

    if hasfield(typeof(ℬ), :boundary_condition_set)
        ℬ.boundary_condition_set = boundary_set
        if hasfield(typeof(ℬ), :boundary_time_index)
            ℬ.boundary_time_index[] = 1
        end
        ℬ.boundary_interpolation_cache = cache
    else
        if !isdefined(@__MODULE__, :_magnetic_boundary_cache)
            global _magnetic_boundary_cache = Dict{UInt64, Any}()
        end
        field_id = objectid(ℬ)
        _magnetic_boundary_cache[field_id] = Dict(
            :boundary_set => boundary_set,
            :interpolation_cache => cache,
            :time_index => 1
        )
    end

    # Apply initial boundary conditions
    apply_magnetic_boundary_conditions!(ℬ)
    
    if get_rank() == 0
        print_boundary_info(boundary_set)
        @info "Magnetic field boundary conditions loaded successfully"
    end
    
    return ℬ
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
    # Note: field_type is set automatically by read_netcdf_boundary_data based on file contents
    inner_data = read_netcdf_boundary_data(inner_file, precision=config.T)
    outer_data = read_netcdf_boundary_data(outer_file, precision=config.T)

    # Validate vector field dimensions (should have 3 components: Bᵣ, Bθ, Bφ)
    if inner_data.ncomponents != 3 || outer_data.ncomponents != 3
        throw(ArgumentError("Magnetic boundary conditions require 3 components (Bᵣ, Bθ, Bφ)"))
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
    # Note: field_type is set automatically by read_netcdf_boundary_data based on file contents
    file_data = read_netcdf_boundary_data(file_spec, precision=config.T)

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
- `:insulating` - Insulating boundary (potential field matching, (∇×B)_r = 0)
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
    
    # Initialize magnetic field components array [nlat, nlon, 3] for (Bᵣ, Bθ, Bφ)
    values = zeros(config.T, nlat, nlon, 3)
    
    # Generate magnetic field pattern
    if pattern == :insulating
        # Insulating boundary: No current can flow across boundary (σ = 0)
        # Physical constraint: J_n = 0 at boundary
        # From Ampère's law: ∇×B = μ₀J, so (∇×B)_r = μ₀J_r = 0
        # This means: (1/(r sin θ))[∂(Bφ sin θ)/∂θ - ∂Bθ/∂φ] = 0
        #
        # For potential field matching (typical insulating BC):
        # - Bᵣ matches internal/external potential field
        # - Tangential components match potential field (continuous)
        # - The key is that field lines cross the boundary normally
        #
        # Use a dipole potential field as default:
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                # Dipolar potential field pattern (appropriate for insulating boundary)
                values[i, j, 1] = amplitude * 2 * cos(θ)     # Bᵣ (radial component)
                values[i, j, 2] = amplitude * sin(θ)         # Bθ (from potential)
                values[i, j, 3] = 0.0                        # Bφ = 0 (axisymmetric dipole)
            end
        end
        description = "Insulating boundary condition (potential field, J_n=0)"
        
    elseif pattern == :perfect_conductor
        # Perfect conductor: B_tangential = 0 at boundary
        # Physical constraint: Tangential E and H vanish inside perfect conductor
        # This gives Bθ = Bφ = 0 at the boundary
        # Bᵣ is determined by ∇·B = 0 (solenoidal constraint) and matching
        #
        # Use dipole radial pattern as reasonable default for Bᵣ:
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = amplitude * 2 * cos(θ)  # Bᵣ (dipolar pattern)
                values[i, j, 2] = 0.0                     # Bθ = 0 (perfect conductor)
                values[i, j, 3] = 0.0                     # Bφ = 0 (perfect conductor)
            end
        end
        description = "Perfect conductor boundary condition (B_tan=0)"
        
    elseif pattern == :dipole
        # Dipolar magnetic field: B ∝ (2cos(θ)ê_r + sin(θ)ê_θ)
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = amplitude * 2 * cos(θ)     # Bᵣ
                values[i, j, 2] = amplitude * sin(θ)         # Bθ
                values[i, j, 3] = 0.0                        # Bφ = 0 (axisymmetric)
            end
        end
        description = "Dipolar magnetic field (amplitude = $amplitude T)"
        
    elseif pattern == :quadrupole
        # Quadrupolar field: Bᵣ ∝ (3cos²θ - 1), Bθ ∝ sin(2θ)
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = amplitude * (3 * cos(θ)^2 - 1)  # Bᵣ
                values[i, j, 2] = amplitude * sin(2 * θ)           # Bθ  
                values[i, j, 3] = 0.0                              # Bφ = 0
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
                # Bᵣ = B⃗ · ê_r = Bₓsin(θ)cos(φ) + Bᵧsin(θ)sin(φ) + Bᵤcos(θ)
                # Bθ = B⃗ · ê_θ = Bₓcos(θ)cos(φ) + Bᵧcos(θ)sin(φ) - Bᵤsin(θ)
                # Bφ = B⃗ · ê_φ = -Bₓsin(φ) + Bᵧcos(φ)
                
                Bx, By, Bz = direction
                sin_theta, cos_theta = sin(θ), cos(θ)
                sin_phi, cos_phi = sin(φ), cos(φ)
                
                values[i, j, 1] = amplitude * (Bx*sin_theta*cos_phi + By*sin_theta*sin_phi + Bz*cos_theta)  # Bᵣ
                values[i, j, 2] = amplitude * (Bx*cos_theta*cos_phi + By*cos_theta*sin_phi - Bz*sin_theta)  # Bθ
                values[i, j, 3] = amplitude * (-Bx*sin_phi + By*cos_phi)                                     # Bφ
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
                Bᵣ, Bθ, Bφ = user_func(θ, φ)
                values[i, j, 1] = amplitude * Bᵣ
                values[i, j, 2] = amplitude * Bθ
                values[i, j, 3] = amplitude * Bφ
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

    # Note: Pattern information is encoded in the description string
    # and will be parsed later if needed for constraint enforcement

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
            
            Bᵣ = 0.0
            Bθ = 0.0
            Bφ = 0.0
            
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
                        # At boundary r=r₀: Bᵣ = -l * V_lm/r₀ * Y_lm
                        #
                        # For EXTERNAL potential: V_lm(r) = B_lm r^(-l-1)
                        # Then ∂V_lm/∂r = -(l+1) * B_lm r^(-l-2) = -(l+1)/r * V_lm
                        # At boundary r=r₀: Bᵣ = (l+1) * V_lm/r₀ * Y_lm
                        #
                        # Using external potential (typical for outer boundary):
                        # Bᵣ = -∂V/∂r = (l+1)/r * V_lm * Y_lm
                        # Bθ = -1/r * ∂V/∂θ = -1/r * V_lm * dY_lm/dθ
                        # Bφ = -1/(r sin θ) * ∂V/∂φ = -1/(r sin θ) * V_lm * dY_lm/dφ
                        #
                        # At r=1 (normalized boundary):
                        Bᵣ += (l + 1) * coeff * Ylm
                        Bθ += -coeff * dYlm_dtheta
                        Bφ += -coeff * dYlm_dphi / (sin(θ) + 1e-15)  # Avoid division by zero
                    end
                end
            end
            
            B_field[i, j, 1] = Bᵣ
            B_field[i, j, 2] = Bθ
            B_field[i, j, 3] = Bφ
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
    apply_magnetic_boundary_conditions!(ℬ, time_index::Int=1)

Apply magnetic field boundary conditions to the field.
"""
function apply_magnetic_boundary_conditions!(ℬ, time_index::Int=1)
    
    boundary_set, cache = get_magnetic_boundary_data(ℬ)
    if boundary_set === nothing || cache === nothing
        @warn "No boundary conditions loaded for magnetic field"
        return ℬ
    end
    
    # Interpolate boundary data to simulation grid
    inner_physical = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
    outer_physical = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
    
    # Transform to spectral space using proper magnetic QST decomposition
    # For magnetic fields: B = Q ê_r + S_tangential + T_tangential
    # where Q is radial, S is spheroidal (potential-like), T is toroidal (solenoidal)

    # Extract components
    Bᵣ_inner = inner_physical[:, :, 1]     # Radial component
    Bθ_inner = inner_physical[:, :, 2]  # Theta component
    Bφ_inner = inner_physical[:, :, 3]    # Phi component

    Bᵣ_outer = outer_physical[:, :, 1]
    Bθ_outer = outer_physical[:, :, 2]
    Bφ_outer = outer_physical[:, :, 3]

    # Convert to QST coefficients using SHTnsKit (if available)
    try
        # Inner boundary
        Q_inner, S_inner, T_inner = magnetic_to_qst_coefficients(
            Bᵣ_inner, Bθ_inner, Bφ_inner, ℬ.config
        )

        # Outer boundary
        Q_outer, S_outer, T_outer = magnetic_to_qst_coefficients(
            Bᵣ_outer, Bθ_outer, Bφ_outer, ℬ.config
        )

        # Map to toroidal-poloidal structure:
        # For magnetic fields: toroidal ~ T (purely tangential), poloidal ~ Q + S (radial + potential)
        ℬ.𝒯.boundary_values[1, :] = T_inner   # Inner toroidal
        ℬ.𝒯.boundary_values[2, :] = T_outer   # Outer toroidal

        # Combine Q and S for poloidal
        # For solenoidal fields (∇·B = 0), the decomposition is:
        # - Q: radial component (Bᵣ)
        # - S: should be zero for purely solenoidal fields
        # - T: tangential toroidal component
        ℬ.𝒫.boundary_values[1, :] = Q_inner  # Inner poloidal (radial)
        ℬ.𝒫.boundary_values[2, :] = Q_outer  # Outer poloidal (radial)

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
        The previous fallback (treating Bᵣ as toroidal and magnitude of tangential as poloidal)
        was fundamentally incorrect because:
        1. It confused radial and tangential components
        2. It took magnitudes of vector components (destroying directional information)
        3. It treated components as independent scalars

        Proper decomposition requires:
        - Q: radial component (Bᵣ)
        - S: curl-free tangential part
        - T: solenoidal tangential part
        where Bθ and Bφ are coupled through S and T.

        Possible solutions:
        1. Check that SHTnsKit.spat_to_SHqst is properly installed and configured
        2. Verify grid dimensions (nlat=$(size(Bᵣ_inner,1)), nlon=$(size(Bᵣ_inner,2)))
           are compatible with lmax=$(ℬ.config.lmax)
        3. Ensure SHTnsKit configuration is properly initialized
        """
        throw(ErrorException(error_msg))
    end
    
    # Update time index
    update_magnetic_time_index!(ℬ, time_index)

    # Enforce magnetic boundary condition constraints based on boundary pattern
    # Infer constraint type from boundary description strings

    # Determine boundary constraint type by parsing description strings
    primary_constraint = :potential_field  # Default

    # Check inner boundary description
    inner_desc = lowercase(boundary_set.inner_boundary.description)
    if occursin("insulating", inner_desc)
        primary_constraint = :insulating
    elseif occursin("perfect conductor", inner_desc) || occursin("perfect_conductor", inner_desc)
        primary_constraint = :perfect_conductor
    elseif occursin("potential", inner_desc)
        primary_constraint = :potential_field
    end

    # Override with outer boundary if it's more specific
    outer_desc = lowercase(boundary_set.outer_boundary.description)
    if occursin("insulating", outer_desc) && primary_constraint == :potential_field
        primary_constraint = :insulating
    elseif occursin("perfect conductor", outer_desc) || occursin("perfect_conductor", outer_desc)
        primary_constraint = :perfect_conductor
    end

    # Validate insulating boundaries if applicable
    if primary_constraint == :insulating && get_rank() == 0
        # Check inner boundary
        if occursin("insulating", inner_desc)
            is_valid_inner, violation_inner = validate_insulating_boundary(
                inner_physical[:, :, 1], inner_physical[:, :, 2], inner_physical[:, :, 3],
                boundary_set.inner_boundary.theta, boundary_set.inner_boundary.phi;
                tolerance=0.05  # 5% relative error allowed
            )
            if !is_valid_inner
                @warn """
                Inner boundary violates insulating condition!
                Maximum |(∇×B)_r| / |B| = $(violation_inner)

                For insulating boundaries, (∇×B)_r = 0 is required (no normal current).
                Your boundary data may not be from a potential field.

                Recommendations:
                1. If using file-based boundaries, ensure they're computed from a potential field
                2. Use programmatic :insulating pattern for guaranteed correctness
                3. Consider computing matching tangential components from Bᵣ

                Proceeding with Dirichlet BC, but results may be unphysical.
                """
            end
        end

        # Check outer boundary
        if occursin("insulating", outer_desc)
            is_valid_outer, violation_outer = validate_insulating_boundary(
                outer_physical[:, :, 1], outer_physical[:, :, 2], outer_physical[:, :, 3],
                boundary_set.outer_boundary.theta, boundary_set.outer_boundary.phi;
                tolerance=0.05
            )
            if !is_valid_outer
                @warn """
                Outer boundary violates insulating condition!
                Maximum |(∇×B)_r| / |B| = $(violation_outer)

                See recommendations above for inner boundary.
                """
            end
        end
    end

    # Apply the determined constraint
    enforce_magnetic_boundary_constraints!(ℬ, primary_constraint)

    return ℬ
end

"""
    update_time_dependent_magnetic_boundaries!(ℬ, current_time::Float64)

Update time-dependent magnetic boundary conditions.
"""
function update_time_dependent_magnetic_boundaries!(ℬ, current_time::Float64)
    
    boundary_set, _ = get_magnetic_boundary_data(ℬ)
    if boundary_set === nothing
        return ℬ
    end
    
    # Check if boundaries are time-dependent
    if !boundary_set.inner_boundary.is_time_dependent && !boundary_set.outer_boundary.is_time_dependent
        return ℬ  # Nothing to update
    end
    
    # Find time index for current time
    time_index = find_boundary_time_index(boundary_set, current_time)
    current_time_index = get_magnetic_time_index(ℬ)
    
    # Only update if time index has changed
    if time_index != current_time_index
        apply_magnetic_boundary_conditions!(ℬ, time_index)
        
        if get_rank() == 0
            @info "Updated magnetic boundaries to time index $time_index (t=$current_time)"
        end
    end
    
    return ℬ
end

"""
    get_current_magnetic_boundaries(ℬ)

Get current magnetic field boundary conditions.
"""
function get_current_magnetic_boundaries(ℬ)
    
    boundary_set, cache = get_magnetic_boundary_data(ℬ)
    if boundary_set === nothing || cache === nothing
        return Dict(:error => "No boundary conditions loaded")
    end
    
    time_index = get_magnetic_time_index(ℬ)
    
    # Get current boundary data
    inner_physical = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
    outer_physical = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
    
    # Get spectral coefficients
    innerᵀ_spectral = ℬ.𝒯.boundary_values[1, :]
    outerᵀ_spectral = ℬ.𝒯.boundary_values[2, :]
    innerᴾ_spectral = ℬ.𝒫.boundary_values[1, :]
    outerᴾ_spectral = ℬ.𝒫.boundary_values[2, :]
    
    return Dict(
        :inner_physical => inner_physical,
        :outer_physical => outer_physical,
        :innerᵀ_spectral => innerᵀ_spectral,
        :outerᵀ_spectral => outerᵀ_spectral,
        :innerᴾ_spectral => innerᴾ_spectral,
        :outerᴾ_spectral => outerᴾ_spectral,
        :time_index => time_index,
        :metadata => Dict(
            "field_name" => boundary_set.field_name,
            "source" => "file_based",
            "inner_file" => boundary_set.inner_boundary.file_path,
            "outer_file" => boundary_set.outer_boundary.file_path,
            "creation_time" => boundary_set.creation_time,
            "components" => ["Bᵣ", "Bθ", "Bφ"]
        )
    )
end

"""
    set_programmatic_magnetic_boundaries!(ℬ, inner_spec::Tuple, outer_spec::Tuple)

Set programmatic magnetic boundary conditions.
"""
function set_programmatic_magnetic_boundaries!(ℬ, inner_spec::Tuple, outer_spec::Tuple)
    
    boundary_specs = Dict(:inner => inner_spec, :outer => outer_spec)
    return load_magnetic_boundary_conditions!(ℬ, boundary_specs)
end

"""
    get_magnetic_boundary_data(ℬ)

Return `(boundary_set, cache)` for the magnetic field, falling back to a
module-level cache when the field struct lacks boundary storage.
"""
function get_magnetic_boundary_data(ℬ)
    if isdefined(@__MODULE__, :_magnetic_boundary_cache)
        field_id = objectid(ℬ)
        if haskey(_magnetic_boundary_cache, field_id)
            data = _magnetic_boundary_cache[field_id]
            return data[:boundary_set], data[:interpolation_cache]
        end
    end

    if hasfield(typeof(ℬ), :boundary_condition_set)
        return ℬ.boundary_condition_set,
               ℬ.boundary_interpolation_cache
    end

    return nothing, nothing
end

"""
    get_magnetic_time_index(ℬ)

Fetch the currently active boundary time index, honoring legacy cache storage.
"""
function get_magnetic_time_index(ℬ)
    if isdefined(@__MODULE__, :_magnetic_boundary_cache)
        field_id = objectid(ℬ)
        if haskey(_magnetic_boundary_cache, field_id)
            return _magnetic_boundary_cache[field_id][:time_index]
        end
    end

    if hasfield(typeof(ℬ), :boundary_time_index)
        return ℬ.boundary_time_index[]
    end

    return 1
end

"""
    update_magnetic_time_index!(ℬ, time_index)

Persist the provided time index to both the magnetic field structure and the
module-level cache when available.
"""
function update_magnetic_time_index!(ℬ, time_index::Int)
    if isdefined(@__MODULE__, :_magnetic_boundary_cache)
        field_id = objectid(ℬ)
        if haskey(_magnetic_boundary_cache, field_id)
            _magnetic_boundary_cache[field_id][:time_index] = time_index
        end
    end

    if hasfield(typeof(ℬ), :boundary_time_index)
        ℬ.boundary_time_index[] = time_index
    end
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
                push!(errors, "Inner magnetic file must have 3 components (Bᵣ, Bθ, Bφ)")
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
                push!(errors, "Outer magnetic file must have 3 components (Bᵣ, Bθ, Bφ)")
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
    compute_radial_curl_component(Bᵣ, Bθ, Bφ, theta, phi)

Compute the radial component of ∇×B to verify insulating boundary conditions.

For insulating boundaries, (∇×B)_r should be zero.
In spherical coordinates:
    (∇×B)_r = (1/(r sin θ))[∂(Bφ sin θ)/∂θ - ∂Bθ/∂φ]

# Arguments
- `Bᵣ, Bθ, Bφ`: Magnetic field components on boundary [nlat, nlon]
- `theta, phi`: Coordinate arrays

# Returns
- `curl_r`: Radial component of curl [nlat, nlon]
- `max_curl`: Maximum absolute value of (∇×B)_r
"""
function compute_radial_curl_component(Bᵣ, Bθ, Bφ, theta, phi)
    nlat, nlon = size(Bᵣ)
    curl_r = zeros(eltype(Bᵣ), nlat, nlon)

    # Compute derivatives using finite differences
    for i in 2:nlat-1
        for j in 1:nlon
            θ = theta[i]
            sin_theta = sin(θ)

            # Handle periodicity in phi
            j_plus = (j == nlon) ? 1 : j + 1
            j_minus = (j == 1) ? nlon : j - 1

            # Compute ∂(Bφ sin θ)/∂θ
            Bφ_sin_theta_plus = Bφ[i+1, j] * sin(theta[i+1])
            Bφ_sin_theta_minus = Bφ[i-1, j] * sin(theta[i-1])
            dtheta = theta[i+1] - theta[i-1]
            d_Bphi_sintheta_dtheta = (Bφ_sin_theta_plus - Bφ_sin_theta_minus) / dtheta

            # Compute ∂Bθ/∂φ
            dphi = phi[j_plus] - phi[j_minus]
            if dphi < 0  # Handle wrap-around
                dphi += 2π
            end
            dBtheta_dphi = (Bθ[i, j_plus] - Bθ[i, j_minus]) / dphi

            # (∇×B)_r = (1/(r sin θ))[∂(Bφ sin θ)/∂θ - ∂Bθ/∂φ]
            # At boundary r = constant, so we ignore the r factor
            curl_r[i, j] = (d_Bphi_sintheta_dtheta - dBtheta_dphi) / (sin_theta + 1e-15)
        end
    end

    max_curl = maximum(abs, curl_r)
    return curl_r, max_curl
end

"""
    validate_insulating_boundary(Bᵣ, Bθ, Bφ, theta, phi; tolerance=1e-2)

Validate that a magnetic field satisfies the insulating boundary condition (∇×B)_r = 0.

# Arguments
- `Bᵣ, Bθ, Bφ`: Magnetic field components [nlat, nlon]
- `theta, phi`: Coordinate arrays
- `tolerance`: Maximum allowed |(∇×B)_r| / |B|

# Returns
- `is_valid`: Boolean indicating if condition is satisfied
- `max_violation`: Maximum |(∇×B)_r| / |B|
"""
function validate_insulating_boundary(Bᵣ, Bθ, Bφ, theta, phi; tolerance=1e-2)
    curl_r, max_curl = compute_radial_curl_component(Bᵣ, Bθ, Bφ, theta, phi)

    # Compute typical field magnitude for normalization
    B_magnitude = sqrt.(Bᵣ.^2 .+ Bθ.^2 .+ Bφ.^2)
    typical_B = _Statistics.mean(B_magnitude)

    # Relative violation
    max_violation = max_curl / (typical_B + 1e-15)
    is_valid = max_violation < tolerance

    return is_valid, max_violation
end

"""
    magnetic_to_qst_coefficients(Bᵣ, Bθ, Bφ, config)

Convert physical magnetic field components to QST spectral coefficients.

For magnetic fields:
- Q: Radial component coefficients (Bᵣ)
- S: Spheroidal tangential component coefficients
- T: Toroidal tangential component coefficients
"""
function magnetic_to_qst_coefficients(Bᵣ, Bθ, Bφ, config)

    try
        # Use SHTnsKit for proper QST decomposition (it's imported at module level)
        # Create SHTnsKit configuration
        lmax = config.lmax
        nlat, nlon = size(Bᵣ)

        shtconfig = SHTnsKit.create_gauss_config(lmax, nlat; mmax=config.mmax, nlon=nlon)

        # Transform to QST coefficients using SHTnsKit - returns (lmax+1)×(mmax+1) matrices
        Q_matrix, S_matrix, T_matrix = SHTnsKit.spat_to_SHqst(shtconfig, Bᵣ, Bθ, Bφ)

        # Clean up configuration
        SHTnsKit.destroy_config(shtconfig)

        # Convert matrix format to 1D spectral coefficient arrays
        # The boundary code expects 1D arrays of length nlm
        mmax = config.mmax
        nlm = config.nlm

        Q_coeffs = zeros(eltype(Bᵣ), nlm)
        S_coeffs = zeros(eltype(Bᵣ), nlm)
        T_coeffs = zeros(eltype(Bᵣ), nlm)

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
                    Q_coeffs[idx] = zero(eltype(Bᵣ))
                    S_coeffs[idx] = zero(eltype(Bᵣ))
                    T_coeffs[idx] = zero(eltype(Bᵣ))
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
        Treating Bᵣ, Bθ, Bφ as independent scalar fields (the old fallback) is incorrect
        because the tangential components Bθ and Bφ are coupled through the spheroidal-toroidal
        decomposition in spherical geometry.

        For magnetic fields:
        - Q: radial component (transforms like scalar: Bᵣ)
        - S: spheroidal tangential (curl-free, from ∇ψ)
        - T: toroidal tangential (solenoidal, from ∇×(T r̂))

        The relationship between (Bθ, Bφ) and (S, T) is non-trivial and requires proper
        vector spherical harmonic decomposition.

        Possible solutions:
        1. Check that SHTnsKit.spat_to_SHqst is properly installed and configured
        2. Verify grid dimensions (nlat=$(size(Bᵣ,1)), nlon=$(size(Bᵣ,2)))
           are compatible with lmax=$(config.lmax)
        3. Ensure SHTnsKit configuration matches the simulation configuration
        """
        throw(ErrorException(error_msg))
    end
end

# Function moved to main bcs module to avoid duplication

"""
    enforce_magnetic_boundary_constraints!(ℬ, bc_type::Symbol)

Enforce magnetic boundary condition constraints based on magnetohydrodynamic physics.

# Boundary condition types:
- `:insulating` - Insulating boundary (σ = 0): No normal current (J_n = 0)
  * Physical interpretation: (∇×B)_r = μ₀J_r = 0 at boundary
  * Implementation: Potential field matching (B continuous, tangential current-free)
  * Spectral: Typically Dirichlet BC for both components matching external potential

- `:conducting_inner_core` - Conducting inner core (finite σ) with insulating exterior
  * Physical interpretation: Field diffuses through inner core, continuous at ICB
  * Implementation:
    - At center (r=0): (∂/∂r - l/r) P = 0 (regularity)
    - At ICB: ∂B/∂r continuous across interface
    - At outer boundary: insulating (∂/∂r + (l+1)/r) P = 0
  * Spectral: Continuity condition at ICB, insulating at exterior

- `:perfect_conductor` - Perfect conductor (σ → ∞): Zero tangential magnetic field
  * Physical interpretation: B_tangential = 0, Bᵣ free (from ∇·B = 0)
  * Implementation: Tangential components zero, radial component determined by matching
  * Spectral: Toroidal (T) = 0 Dirichlet, Poloidal (Q) matches

- `:potential_field` - General potential field matching
  * Implementation: Match external potential field at boundary
  * Spectral: Dirichlet BC for both toroidal and poloidal components

# Notes:
For most geodynamo applications:
- Inner boundary (ICB): Often insulating or conducting inner core
- Outer boundary (CMB): Often insulating (matching Earth's mantle)
"""
function enforce_magnetic_boundary_constraints!(ℬ, bc_type::Symbol)

    if bc_type == :insulating
        # Insulating boundary conditions:
        # - Toroidal: B_tor = 0 at both boundaries (Dirichlet)
        # - Poloidal inner: (∂/∂r - l/r) B_pol = 0 (NEUMANN_MAG_INNER)
        # - Poloidal outer: (∂/∂r + (l+1)/r) B_pol = 0 (NEUMANN_MAG_OUTER)
        #
        # These conditions ensure the magnetic field matches a potential field
        # solution outside the conducting region.

        # Toroidal: B_tor = 0 at both boundaries
        fill!(ℬ.𝒯.boundary_values, 0.0)
        fill!(ℬ.𝒯.bc_type_inner, Int(DIRICHLET))
        fill!(ℬ.𝒯.bc_type_outer, Int(DIRICHLET))

        # Poloidal: l-dependent derivative conditions
        # Inner: (∂/∂r - l/r) P = 0  →  field decays as r^l inside
        # Outer: (∂/∂r + (l+1)/r) P = 0  →  field decays as r^{-(l+1)} outside
        fill!(ℬ.𝒫.boundary_values, 0.0)  # RHS = 0 for homogeneous BC
        fill!(ℬ.𝒫.bc_type_inner, Int(NEUMANN_MAG_INNER))
        fill!(ℬ.𝒫.bc_type_outer, Int(NEUMANN_MAG_OUTER))

    elseif bc_type == :conducting_inner_core
        # Conducting inner core boundary conditions:
        # The inner core has finite electrical conductivity, so magnetic field
        # can diffuse through it. At the ICB, B and ∂B/∂r must be continuous.
        #
        # Toroidal at ICB: ∂BTor/∂r continuous (jump condition)
        # Poloidal at ICB: ∂BPol/∂r continuous
        # Outer boundary: still insulating

        # Toroidal:
        # - Inner (ICB): continuity of ∂BTor/∂r
        # - Outer: B_tor = 0 (insulating)
        fill!(ℬ.𝒯.boundary_values, 0.0)
        fill!(ℬ.𝒯.bc_type_inner, Int(CONTINUITY_MAG))  # Continuity at ICB
        fill!(ℬ.𝒯.bc_type_outer, Int(DIRICHLET))       # BTor = 0 at exterior

        # Poloidal:
        # - Inner (ICB): continuity of ∂BPol/∂r
        # - Outer: insulating (∂/∂r + (l+1)/r) P = 0
        fill!(ℬ.𝒫.boundary_values, 0.0)
        fill!(ℬ.𝒫.bc_type_inner, Int(CONTINUITY_MAG))      # Continuity at ICB
        fill!(ℬ.𝒫.bc_type_outer, Int(NEUMANN_MAG_OUTER))   # Insulating exterior

    elseif bc_type == :perfect_conductor
        # Perfect conductor: B_tangential = 0 at boundary
        # Physical: Tangential E and H vanish, giving Bθ = Bφ = 0
        # Spectral: Toroidal component T = 0 (controls tangential field)
        # Radial component (poloidal/Q) is non-zero and determined by ∇·B = 0

        # Set toroidal components to zero (tangential field = 0)
        fill!(ℬ.𝒯.boundary_values, 0.0)
        fill!(ℬ.𝒯.bc_type_inner, Int(DIRICHLET))  # T = 0 enforced
        fill!(ℬ.𝒯.bc_type_outer, Int(DIRICHLET))  # T = 0 enforced

        # Poloidal/radial component uses Dirichlet BC from computed values
        # (determined by ∇·B = 0 and matching conditions)
        fill!(ℬ.𝒫.bc_type_inner, Int(DIRICHLET))
        fill!(ℬ.𝒫.bc_type_outer, Int(DIRICHLET))

    elseif bc_type == :potential_field
        # Potential field boundary: match external field
        # Both components use computed boundary values as Dirichlet BC

        fill!(ℬ.𝒯.bc_type_inner, Int(DIRICHLET))
        fill!(ℬ.𝒯.bc_type_outer, Int(DIRICHLET))
        fill!(ℬ.𝒫.bc_type_inner, Int(DIRICHLET))
        fill!(ℬ.𝒫.bc_type_outer, Int(DIRICHLET))

    elseif bc_type == :custom
        # Custom boundary conditions - leave arrays as set by user
        @info "Custom magnetic boundary conditions - user must set bc_type arrays"

    else
        @warn "Unknown magnetic boundary condition type: $bc_type, using potential_field"
        enforce_magnetic_boundary_constraints!(ℬ, :potential_field)
    end

    return ℬ
end

export load_magnetic_boundary_conditions!, set_programmatic_magnetic_boundaries!
export update_time_dependent_magnetic_boundaries!, get_current_magnetic_boundaries
export validate_magnetic_boundary_files, create_programmatic_magnetic_boundary
export magnetic_to_qst_coefficients, enforce_magnetic_boundary_constraints!
export compute_radial_curl_component, validate_insulating_boundary
