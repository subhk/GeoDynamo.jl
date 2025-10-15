# ================================================================================
# Programmatic Boundary Condition Generation
# ================================================================================

using LinearAlgebra
using Statistics

"""
    create_programmatic_boundary(pattern::Symbol, config, amplitude::Real=1.0;
                                parameters::Dict=Dict(), description::String="")

Create programmatically generated boundary conditions.

# Available patterns:
- `:uniform` - Uniform value
- `:y11` - Y₁₁ spherical harmonic  
- `:plume` - Gaussian plume pattern
- `:hemisphere` - Hemispherical pattern
- `:dipole` - Dipolar pattern (Y₁₀)
- `:quadrupole` - Quadrupolar pattern
- `:custom` - User-defined function
"""
function create_programmatic_boundary(pattern::Symbol, config, amplitude::Real=1.0;
                                    parameters::Dict=Dict(), description::String="", 
                                    field_type::String="temperature")
    
    # Create coordinate grids
    nlat, nlon = config.nlat, config.nlon
    theta = collect(range(0, π, length=nlat))
    phi = collect(range(0, 2π, length=nlon+1)[1:end-1])
    
    # Initialize data array
    values = zeros(eltype(amplitude), nlat, nlon)
    
    # Generate pattern
    if pattern == :uniform
        values .= amplitude
        
    elseif pattern == :y11
        # Y₁₁ spherical harmonic: sin(θ)cos(φ)
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j] = amplitude * sin(θ) * cos(φ)
            end
        end
        
    elseif pattern == :plume
        # Gaussian plume pattern
        width = get(parameters, "width", π/6)
        center_theta = get(parameters, "center_theta", π/2)
        center_phi = get(parameters, "center_phi", 0.0)
        
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                # Angular distance from plume center
                angular_dist = acos(cos(θ) * cos(center_theta) + 
                                  sin(θ) * sin(center_theta) * cos(φ - center_phi))
                values[i, j] = amplitude * exp(-(angular_dist / width)^2)
            end
        end
        
    elseif pattern == :hemisphere
        # Hemispherical pattern
        axis = get(parameters, "axis", "z")  # Options: "x", "y", "z"
        
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                if axis == "z"
                    values[i, j] = amplitude * (θ <= π/2 ? 1.0 : 0.0)
                elseif axis == "x" 
                    values[i, j] = amplitude * (sin(θ) * cos(φ) >= 0 ? 1.0 : 0.0)
                elseif axis == "y"
                    values[i, j] = amplitude * (sin(θ) * sin(φ) >= 0 ? 1.0 : 0.0)
                else
                    throw(ArgumentError("Invalid axis for hemisphere pattern: $axis"))
                end
            end
        end
        
    elseif pattern == :dipole
        # Dipolar pattern (Y₁₀): cos(θ)
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j] = amplitude * cos(θ)
            end
        end
        
    elseif pattern == :quadrupole
        # Quadrupolar pattern: P₂(cos θ) = (3cos²θ - 1)/2
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j] = amplitude * 0.5 * (3 * cos(θ)^2 - 1)
            end
        end
        
    elseif pattern == :custom
        # User-defined function
        if !haskey(parameters, "function")
            throw(ArgumentError("Custom pattern requires 'function' in parameters"))
        end
        
        user_func = parameters["function"]
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j] = amplitude * user_func(θ, φ)
            end
        end
        
    else
        throw(ArgumentError("Unknown programmatic pattern: $pattern"))
    end
    
    # Create BoundaryData structure
    if isempty(description)
        description = "Programmatic $pattern pattern (amplitude=$amplitude)"
    end
    
    return create_boundary_data(
        values, field_type;
        theta=theta, phi=phi, time=nothing,
        units=get_default_units(determine_field_type_from_name(field_type)),
        description=description,
        file_path="programmatic"
    )
end

"""
    create_time_dependent_programmatic_boundary(pattern::Symbol, config, time_span::Tuple{Real, Real}, 
                                               ntime::Int; amplitude::Real=1.0, 
                                               parameters::Dict=Dict(), description::String="",
                                               field_type::String="temperature")

Create time-dependent programmatically generated boundary conditions.
"""
function create_time_dependent_programmatic_boundary(pattern::Symbol, config, 
                                                   time_span::Tuple{Real, Real}, ntime::Int;
                                                   amplitude::Real=1.0, parameters::Dict=Dict(),
                                                   description::String="", field_type::String="temperature")
    
    # Create coordinate grids
    nlat, nlon = config.nlat, config.nlon
    theta = collect(range(0, π, length=nlat))
    phi = collect(range(0, 2π, length=nlon+1)[1:end-1])
    time_coords = collect(range(time_span[1], time_span[2], length=ntime))
    
    # Initialize data array
    values = zeros(eltype(amplitude), nlat, nlon, ntime)
    
    # Time evolution parameters
    time_factor = get(parameters, "time_factor", 1.0)
    phase_offset = get(parameters, "phase_offset", 0.0)
    
    # Generate time-dependent pattern
    for (t, time_val) in enumerate(time_coords)
        time_phase = time_factor * time_val + phase_offset
        
        if pattern == :uniform
            values[:, :, t] .= amplitude
            
        elseif pattern == :y11
            # Rotating Y₁₁: sin(θ)cos(φ + ωt)
            for (i, θ) in enumerate(theta)
                for (j, φ) in enumerate(phi)
                    values[i, j, t] = amplitude * sin(θ) * cos(φ + time_phase)
                end
            end
            
        elseif pattern == :plume
            # Moving plume pattern
            width = get(parameters, "width", π/6)
            center_theta = get(parameters, "center_theta", π/2)
            center_phi_base = get(parameters, "center_phi", 0.0)
            
            # Move plume center over time
            center_phi = center_phi_base + time_phase
            
            for (i, θ) in enumerate(theta)
                for (j, φ) in enumerate(phi)
                    angular_dist = acos(cos(θ) * cos(center_theta) + 
                                      sin(θ) * sin(center_theta) * cos(φ - center_phi))
                    values[i, j, t] = amplitude * exp(-(angular_dist / width)^2)
                end
            end
            
        elseif pattern == :hemisphere
            # Rotating hemisphere
            axis_rotation = time_phase
            
            for (i, θ) in enumerate(theta)
                for (j, φ) in enumerate(phi)
                    # Rotate coordinate system
                    x = sin(θ) * cos(φ + axis_rotation)
                    values[i, j, t] = amplitude * (x >= 0 ? 1.0 : 0.0)
                end
            end
            
        elseif pattern == :dipole
            # Precessing dipole
            for (i, θ) in enumerate(theta)
                for (j, φ) in enumerate(phi)
                    # Simple precession: cos(θ)cos(ωt) + sin(θ)sin(φ)sin(ωt)
                    values[i, j, t] = amplitude * (cos(θ) * cos(time_phase) + 
                                                 sin(θ) * sin(φ) * sin(time_phase))
                end
            end
            
        elseif pattern == :custom
            # User-defined time-dependent function
            if !haskey(parameters, "function")
                throw(ArgumentError("Custom pattern requires 'function' in parameters"))
            end
            
            user_func = parameters["function"]
            for (i, θ) in enumerate(theta)
                for (j, φ) in enumerate(phi)
                    values[i, j, t] = amplitude * user_func(θ, φ, time_val)
                end
            end
            
        else
            throw(ArgumentError("Unknown programmatic pattern: $pattern"))
        end
    end
    
    # Create BoundaryData structure
    if isempty(description)
        description = "Time-dependent programmatic $pattern pattern (amplitude=$amplitude)"
    end
    
    return create_boundary_data(
        values, field_type;
        theta=theta, phi=phi, time=time_coords,
        units=get_default_units(determine_field_type_from_name(field_type)),
        description=description,
        file_path="programmatic"
    )
end

"""
    combine_programmatic_patterns(patterns::Vector{Tuple{Symbol, Real}}, config; 
                                 parameters::Vector{Dict}=Dict[], description::String="")

Combine multiple programmatic patterns with different amplitudes.
"""
function combine_programmatic_patterns(patterns::Vector{Tuple{Symbol, Real}}, config; 
                                     parameters::Vector{Dict}=Dict[], description::String="",
                                     field_type::String="temperature")
    
    if isempty(patterns)
        throw(ArgumentError("At least one pattern must be specified"))
    end
    
    # Ensure parameters vector has correct length
    if length(parameters) < length(patterns)
        for i in (length(parameters)+1):length(patterns)
            push!(parameters, Dict())
        end
    end
    
    # Create first pattern as base
    pattern1, amplitude1 = patterns[1]
    combined_boundary = create_programmatic_boundary(
        pattern1, config, amplitude1; 
        parameters=parameters[1], field_type=field_type
    )
    
    # Add additional patterns
    for i in 2:length(patterns)
        pattern_i, amplitude_i = patterns[i]
        boundary_i = create_programmatic_boundary(
            pattern_i, config, amplitude_i; 
            parameters=parameters[i], field_type=field_type
        )
        
        # Add to combined result
        combined_boundary.values .+= boundary_i.values
    end
    
    # Update description
    if isempty(description)
        pattern_names = [string(p[1]) for p in patterns]
        description = "Combined programmatic patterns: " * join(pattern_names, " + ")
    end
    combined_boundary.description = description
    
    return combined_boundary
end

"""
    add_noise_to_boundary(boundary_data::BoundaryData, noise_amplitude::Real, 
                         noise_type::Symbol=:gaussian)

Add noise to existing boundary data.
"""
function add_noise_to_boundary(boundary_data::BoundaryData, noise_amplitude::Real, 
                              noise_type::Symbol=:gaussian)
    
    noisy_values = copy(boundary_data.values)
    
    if noise_type == :gaussian
        noise = noise_amplitude * randn(size(noisy_values))
    elseif noise_type == :uniform
        noise = noise_amplitude * (2 * rand(size(noisy_values)) .- 1)
    else
        throw(ArgumentError("Unknown noise type: $noise_type"))
    end
    
    noisy_values .+= noise
    
    # Create new boundary data with noise added
    return BoundaryData(
        boundary_data.theta, boundary_data.phi, boundary_data.time,
        noisy_values, boundary_data.units, 
        boundary_data.description * " + $(noise_type) noise",
        boundary_data.file_path, boundary_data.field_type,
        boundary_data.is_time_dependent, boundary_data.nlat, 
        boundary_data.nlon, boundary_data.ntime, boundary_data.ncomponents
    )
end

"""
    smooth_boundary_data(boundary_data::BoundaryData, smoothing_radius::Real)

Apply spatial smoothing to boundary data.
"""
function smooth_boundary_data(boundary_data::BoundaryData, smoothing_radius::Real)
    
    smoothed_values = copy(boundary_data.values)
    
    if boundary_data.theta === nothing || boundary_data.phi === nothing
        @warn "Cannot smooth boundary data without coordinate information"
        return boundary_data
    end
    
    theta = boundary_data.theta
    phi = boundary_data.phi
    
    # Apply Gaussian smoothing kernel
    for time_idx in 1:boundary_data.ntime
        if boundary_data.is_time_dependent
            if boundary_data.ncomponents == 1
                data_slice = smoothed_values[:, :, time_idx]
            else
                # Smooth each component separately
                for comp in 1:boundary_data.ncomponents
                    data_slice = @view smoothed_values[:, :, time_idx, comp]
                    apply_gaussian_smoothing!(data_slice, theta, phi, smoothing_radius)
                end
                continue
            end
        else
            if boundary_data.ncomponents == 1
                data_slice = smoothed_values
            else
                for comp in 1:boundary_data.ncomponents
                    data_slice = @view smoothed_values[:, :, comp]
                    apply_gaussian_smoothing!(data_slice, theta, phi, smoothing_radius)
                end
                continue
            end
        end
        
        apply_gaussian_smoothing!(data_slice, theta, phi, smoothing_radius)
    end
    
    # Create new boundary data with smoothed values
    return BoundaryData(
        boundary_data.theta, boundary_data.phi, boundary_data.time,
        smoothed_values, boundary_data.units,
        boundary_data.description * " (smoothed)",
        boundary_data.file_path, boundary_data.field_type,
        boundary_data.is_time_dependent, boundary_data.nlat,
        boundary_data.nlon, boundary_data.ntime, boundary_data.ncomponents
    )
end

"""
    apply_gaussian_smoothing!(data::AbstractMatrix, theta::Vector, phi::Vector, radius::Real)

Apply Gaussian smoothing kernel to 2D data array.
"""
function apply_gaussian_smoothing!(data::AbstractMatrix, theta::Vector, phi::Vector, radius::Real)
    
    nlat, nlon = size(data)
    original_data = copy(data)
    
    for i in 1:nlat
        for j in 1:nlon
            weighted_sum = 0.0
            weight_total = 0.0
            
            # Apply smoothing kernel in neighborhood
            for ii in 1:nlat
                for jj in 1:nlon
                    # Calculate angular distance
                    angular_dist = acos(cos(theta[i]) * cos(theta[ii]) + 
                                      sin(theta[i]) * sin(theta[ii]) * 
                                      cos(phi[j] - phi[jj]))
                    
                    # Gaussian weight
                    weight = exp(-(angular_dist / radius)^2)
                    
                    weighted_sum += weight * original_data[ii, jj]
                    weight_total += weight
                end
            end
            
            data[i, j] = weighted_sum / weight_total
        end
    end
end

export create_programmatic_boundary, create_time_dependent_programmatic_boundary
export combine_programmatic_patterns, add_noise_to_boundary, smooth_boundary_data