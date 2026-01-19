# ================================================================================
# Velocity Boundary Conditions
# ================================================================================

# Note: This file is included within the bcs module
# All necessary packages are imported at the module level

"""
    load_velocity_boundary_conditions!(velocity_field, boundary_specs::Dict)

Load velocity boundary conditions from various sources.

# Arguments
- `velocity_field`: SHTnsVelocityField structure
- `boundary_specs`: Dictionary specifying boundary sources

# Examples
```julia
# No-slip boundaries at both surfaces
boundary_specs = Dict(
    :inner => (:no_slip, 0.0),
    :outer => (:no_slip, 0.0)
)

# Stress-free boundaries
boundary_specs = Dict(
    :inner => (:stress_free, 0.0),
    :outer => (:stress_free, 0.0)
)

# NetCDF file for inner, no-slip for outer
boundary_specs = Dict(
    :inner => "cmb_velocity.nc",
    :outer => (:no_slip, 0.0)
)
```
"""
function load_velocity_boundary_conditions!(velocity_field, boundary_specs::Dict)
    
    if get_rank() == 0
        @info "Loading velocity boundary conditions..."
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
        boundary_set = load_velocity_boundaries_from_files(inner_spec, outer_spec, velocity_field.config)
    elseif isa(inner_spec, String) && isa(outer_spec, Tuple)
        # Inner from file, outer programmatic
        boundary_set = create_hybrid_velocity_boundaries(inner_spec, outer_spec, velocity_field.config)
    elseif isa(inner_spec, Tuple) && isa(outer_spec, String)
        # Inner programmatic, outer from file
        boundary_set = create_hybrid_velocity_boundaries(outer_spec, inner_spec, velocity_field.config, swap_boundaries=true)
    elseif isa(inner_spec, Tuple) && isa(outer_spec, Tuple)
        # Both programmatic
        boundary_set = create_programmatic_velocity_boundaries(inner_spec, outer_spec, velocity_field.config)
    else
        throw(ArgumentError("Invalid boundary specification format"))
    end
    
    cache = create_velocity_interpolation_cache(boundary_set, velocity_field.config)

    if hasfield(typeof(velocity_field), :boundary_condition_set)
        velocity_field.boundary_condition_set = boundary_set
        if hasfield(typeof(velocity_field), :boundary_time_index)
            velocity_field.boundary_time_index[] = 1
        end
        velocity_field.boundary_interpolation_cache = cache
    else
        if !isdefined(@__MODULE__, :_velocity_boundary_cache)
            global _velocity_boundary_cache = Dict{UInt64, Any}()
        end
        field_id = objectid(velocity_field)
        _velocity_boundary_cache[field_id] = Dict(
            :boundary_set => boundary_set,
            :interpolation_cache => cache,
            :time_index => 1
        )
    end

    # Apply initial boundary conditions
    apply_velocity_boundary_conditions!(velocity_field)
    
    if get_rank() == 0
        print_boundary_info(boundary_set)
        @info "Velocity boundary conditions loaded successfully"
    end
    
    return velocity_field
end

"""
    load_velocity_boundaries_from_files(inner_file::String, outer_file::String, config)

Load velocity boundary conditions from NetCDF files.
"""
function load_velocity_boundaries_from_files(inner_file::String, outer_file::String, config)
    
    # Validate files exist
    for file in [inner_file, outer_file]
        if !isfile(file)
            throw(ArgumentError("Velocity boundary file not found: $file"))
        end
    end
    
    # Read boundary data
    inner_data = read_netcdf_boundary_data(inner_file, precision=config.T)
    outer_data = read_netcdf_boundary_data(outer_file, precision=config.T)
    
    # Update field type for velocity
    inner_data.field_type = "velocity"
    outer_data.field_type = "velocity"
    
    # Validate vector field dimensions (should have 3 components: r, θ, φ)
    if inner_data.ncomponents != 3 || outer_data.ncomponents != 3
        throw(ArgumentError("Velocity boundary conditions require 3 components (r, θ, φ)"))
    end
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "velocity")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "velocity", VELOCITY, time()
    )
    
    return boundary_set
end

"""
    create_hybrid_velocity_boundaries(file_spec::String, prog_spec::Tuple, config; swap_boundaries=false)

Create hybrid velocity boundaries (one from file, one programmatic).
"""
function create_hybrid_velocity_boundaries(file_spec::String, prog_spec::Tuple, config; swap_boundaries=false)
    
    # Load file-based boundary
    file_data = read_netcdf_boundary_data(file_spec, precision=config.T)
    file_data.field_type = "velocity"
    
    # Create programmatic boundary
    pattern, amplitude = prog_spec[1], prog_spec[2]
    parameters = length(prog_spec) >= 3 ? prog_spec[3] : Dict()
    
    prog_data = create_programmatic_velocity_boundary(
        pattern, config, amplitude; parameters=parameters
    )
    
    # Ensure same grid resolution
    if file_data.nlat != config.nlat || file_data.nlon != config.nlon
        # Interpolate file data to config grid
        theta_target = collect(range(0, π, length=config.nlat))
        phi_target = collect(range(0, 2π, length=config.nlon+1)[1:end-1])
        
        interpolated_values = interpolate_boundary_to_grid(file_data, theta_target, phi_target, 1)
        
        file_data = create_boundary_data(
            interpolated_values, "velocity";
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
    validate_boundary_compatibility(inner_data, outer_data, "velocity")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "velocity", VELOCITY, time()
    )
    
    return boundary_set
end

"""
    create_programmatic_velocity_boundaries(inner_spec::Tuple, outer_spec::Tuple, config)

Create fully programmatic velocity boundaries.
"""
function create_programmatic_velocity_boundaries(inner_spec::Tuple, outer_spec::Tuple, config)
    
    # Create inner boundary
    inner_pattern, inner_amplitude = inner_spec[1], inner_spec[2]
    inner_parameters = length(inner_spec) >= 3 ? inner_spec[3] : Dict()
    
    inner_data = create_programmatic_velocity_boundary(
        inner_pattern, config, inner_amplitude; parameters=inner_parameters
    )
    
    # Create outer boundary
    outer_pattern, outer_amplitude = outer_spec[1], outer_spec[2]
    outer_parameters = length(outer_spec) >= 3 ? outer_spec[3] : Dict()
    
    outer_data = create_programmatic_velocity_boundary(
        outer_pattern, config, outer_amplitude; parameters=outer_parameters
    )
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "velocity")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "velocity", VELOCITY, time()
    )
    
    return boundary_set
end

"""
    create_programmatic_velocity_boundary(pattern::Symbol, config, amplitude::Real=1.0; 
                                        parameters::Dict=Dict())

Create programmatically generated velocity boundary conditions.

# Available patterns:
- `:no_slip` - Zero velocity at boundary (amplitude ignored)
- `:stress_free` - Zero stress at boundary (amplitude ignored)  
- `:uniform_rotation` - Uniform rotation with angular velocity amplitude
- `:differential_rotation` - Differential rotation pattern
- `:zonal_flow` - Zonal (east-west) flow pattern
- `:meridional_flow` - Meridional (north-south) flow pattern
- `:custom` - User-defined velocity function
"""
function create_programmatic_velocity_boundary(pattern::Symbol, config, amplitude::Real=1.0;
                                             parameters::Dict=Dict())
    
    # Create coordinate grids
    nlat, nlon = config.nlat, config.nlon
    theta = collect(range(0, π, length=nlat))
    phi = collect(range(0, 2π, length=nlon+1)[1:end-1])
    
    # Initialize velocity components array [nlat, nlon, 3] for (v_r, v_θ, v_φ)
    values = zeros(config.T, nlat, nlon, 3)
    
    # Generate velocity pattern
    if pattern == :no_slip
        # All velocity components are zero (already initialized)
        description = "No-slip boundary condition (zero velocity)"
        
    elseif pattern == :stress_free
        # Stress-free boundary conditions:
        # v_r = 0 (no penetration) - Dirichlet BC
        # ∂v_θ/∂r - v_θ/r = 0 (zero tangential stress) - Neumann BC
        # ∂v_φ/∂r - v_φ/r = 0 (zero tangential stress) - Neumann BC
        #
        # For a stress-free boundary:
        # - Radial component is constrained to zero (Dirichlet)
        # - Tangential components satisfy Neumann conditions (zero tangential stress)
        #
        # Note: The tangential velocity values set here are placeholders and will NOT
        # be enforced as Dirichlet BCs. Only the radial component v_r=0 is enforced.
        # The tangential components are determined by the Neumann BC during solving.
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = 0.0  # v_r = 0 (Dirichlet BC - enforced)
                values[i, j, 2] = 0.0  # v_θ (placeholder - Neumann BC will be used)
                values[i, j, 3] = 0.0  # v_φ (placeholder - Neumann BC will be used)
            end
        end
        description = "Stress-free boundary condition (v_r=0 Dirichlet, tangential Neumann)"
        
    elseif pattern == :uniform_rotation
        # Uniform rotation with angular velocity = amplitude
        omega = amplitude
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = 0.0  # v_r = 0
                values[i, j, 2] = 0.0  # v_θ = 0  
                values[i, j, 3] = omega * sin(θ)  # v_φ = ω sin(θ)
            end
        end
        description = "Uniform rotation (ω = $omega rad/s)"
        
    elseif pattern == :differential_rotation
        # Differential rotation: ω(θ) = ω₀ sin²(θ)
        omega0 = amplitude
        for (i, θ) in enumerate(theta)
            omega_theta = omega0 * sin(θ)^2
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = 0.0  # v_r = 0
                values[i, j, 2] = 0.0  # v_θ = 0
                values[i, j, 3] = omega_theta * sin(θ)  # v_φ = ω(θ) sin(θ)
            end
        end
        description = "Differential rotation (ω₀ = $omega0 rad/s)"
        
    elseif pattern == :zonal_flow
        # East-west flow pattern: v_φ = amplitude * sin(n*θ)
        n = get(parameters, "wavenumber", 1)
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = 0.0  # v_r = 0
                values[i, j, 2] = 0.0  # v_θ = 0
                values[i, j, 3] = amplitude * sin(n * θ)  # v_φ
            end
        end
        description = "Zonal flow (n = $n, amplitude = $amplitude)"
        
    elseif pattern == :meridional_flow
        # North-south flow pattern: v_θ = amplitude * cos(m*φ)
        m = get(parameters, "wavenumber", 1)
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = 0.0  # v_r = 0
                values[i, j, 2] = amplitude * cos(m * φ)  # v_θ
                values[i, j, 3] = 0.0  # v_φ = 0
            end
        end
        description = "Meridional flow (m = $m, amplitude = $amplitude)"
        
    elseif pattern == :custom
        # User-defined velocity function
        if !haskey(parameters, "function")
            throw(ArgumentError("Custom velocity pattern requires 'function' in parameters"))
        end
        
        user_func = parameters["function"]
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                v_r, v_theta, v_phi = user_func(θ, φ)
                values[i, j, 1] = amplitude * v_r
                values[i, j, 2] = amplitude * v_theta
                values[i, j, 3] = amplitude * v_phi
            end
        end
        description = "Custom velocity pattern (amplitude = $amplitude)"
        
    else
        throw(ArgumentError("Unknown velocity pattern: $pattern"))
    end
    
    # Create BoundaryData structure
    return create_boundary_data(
        values, "velocity";
        theta=theta, phi=phi, time=nothing,
        units="m/s",
        description=description,
        file_path="programmatic"
    )
end

"""
    create_velocity_interpolation_cache(boundary_set::BoundaryConditionSet, config)

Create interpolation cache for velocity boundaries.
"""
function create_velocity_interpolation_cache(boundary_set::BoundaryConditionSet, config)
    
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
    infer_velocity_bc_type(boundary::BoundaryData)

Infer boundary condition type (Dirichlet or Neumann) from boundary metadata.
"""
function infer_velocity_bc_type(boundary::BoundaryData)
    desc = lowercase(boundary.description)
    if occursin("stress-free", desc)
        return Int(NEUMANN)
    else
        return Int(DIRICHLET)
    end
end

"""
    apply_velocity_boundary_conditions!(velocity_field, time_index::Int=1)

Apply velocity boundary conditions to the field.
"""
function apply_velocity_boundary_conditions!(velocity_field, time_index::Int=1)
    
    boundary_set, cache = get_velocity_boundary_data(velocity_field)
    if boundary_set === nothing || cache === nothing
        @warn "No boundary conditions loaded for velocity field"
        return velocity_field
    end
    
    # Interpolate boundary data to simulation grid
    inner_physical = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
    outer_physical = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
    
    # Transform to spectral space for each velocity component
    # IMPORTANT: For spherical coordinates, the velocity field decomposition is:
    # - Toroidal component: related to ∇ × (T r̂), where T is the toroidal scalar
    # - Poloidal component: related to ∇ × ∇ × (P r̂), where P is the poloidal scalar

    # For no-slip boundaries: all velocity components (v_r, v_θ, v_φ) = 0
    # For stress-free boundaries: v_r = 0, ∂v_θ/∂r - v_θ/r = 0, ∂v_φ/∂r - v_φ/r = 0

    # Compute toroidal scalar from velocity components
    # In spherical coordinates: v_r is related to poloidal scalar, not toroidal
    # Toroidal field contributes only to tangential components

    # For proper spectral boundary conditions, we need to compute the scalars T and P
    # from the physical velocity components

    # Convert velocity components to QST spectral coefficients
    # Using proper SHTnsKit QST decomposition for 3D vector fields
    inner_Q, inner_S, inner_T = velocity_to_qst_coefficients(
        inner_physical[:, :, 1], inner_physical[:, :, 2], inner_physical[:, :, 3], velocity_field.config
    )
    outer_Q, outer_S, outer_T = velocity_to_qst_coefficients(
        outer_physical[:, :, 1], outer_physical[:, :, 2], outer_physical[:, :, 3], velocity_field.config
    )
    
    # Apply QST coefficients to boundary arrays
    # Note: Current field structure uses "toroidal" and "poloidal" names
    # but in QST decomposition: "poloidal" → Q (radial), "toroidal" → T (tangential toroidal)

    # Q component (radial) - stored in "poloidal" field for backward compatibility
    velocity_field.poloidal.boundary_values[1, :] .= inner_Q  # Inner boundary (radial component)
    velocity_field.poloidal.boundary_values[2, :] .= outer_Q  # Outer boundary (radial component)

    # T component (tangential toroidal) - stored in "toroidal" field
    velocity_field.toroidal.boundary_values[1, :] .= inner_T  # Inner boundary (toroidal component)
    velocity_field.toroidal.boundary_values[2, :] .= outer_T  # Outer boundary (toroidal component)

    # Update boundary condition type metadata based on pattern descriptions
    # For stress-free: radial (poloidal/Q) is Dirichlet, tangential (toroidal/T) is Neumann
    # For no-slip: both radial and tangential are Dirichlet
    inner_desc = lowercase(boundary_set.inner_boundary.description)
    outer_desc = lowercase(boundary_set.outer_boundary.description)

    # Set BC types for poloidal (radial) component
    if occursin("stress-free", inner_desc) || occursin("stress free", inner_desc)
        fill!(velocity_field.poloidal.bc_type_inner, Int(DIRICHLET))  # v_r = 0 (Dirichlet)
        fill!(velocity_field.toroidal.bc_type_inner, Int(NEUMANN))     # ∂v_tangential/∂r (Neumann)
    else
        inner_bc_type = infer_velocity_bc_type(boundary_set.inner_boundary)
        fill!(velocity_field.poloidal.bc_type_inner, inner_bc_type)
        fill!(velocity_field.toroidal.bc_type_inner, inner_bc_type)
    end

    if occursin("stress-free", outer_desc) || occursin("stress free", outer_desc)
        fill!(velocity_field.poloidal.bc_type_outer, Int(DIRICHLET))  # v_r = 0 (Dirichlet)
        fill!(velocity_field.toroidal.bc_type_outer, Int(NEUMANN))     # ∂v_tangential/∂r (Neumann)
    else
        outer_bc_type = infer_velocity_bc_type(boundary_set.outer_boundary)
        fill!(velocity_field.poloidal.bc_type_outer, outer_bc_type)
        fill!(velocity_field.toroidal.bc_type_outer, outer_bc_type)
    end

    # S component (tangential spheroidal/curl-free) handling:
    #
    # For INCOMPRESSIBLE (solenoidal) flows where ∇·v = 0:
    # - The S component should be zero (or negligible) since solenoidal fields have no curl-free part
    # - The current field structure (toroidal + poloidal) is sufficient for solenoidal fields
    # - "poloidal" actually stores Q (radial), "toroidal" stores T (tangential divergence-free)
    #
    # For COMPRESSIBLE flows:
    # - Would need a separate spheroidal field component to store S
    # - Currently NOT implemented - code assumes incompressible flow
    #
    # Check S component magnitude to verify solenoidal assumption:
    S_norm_inner = sqrt(sum(abs2, inner_S))
    S_norm_outer = sqrt(sum(abs2, outer_S))
    T_norm_inner = sqrt(sum(abs2, inner_T))
    T_norm_outer = sqrt(sum(abs2, outer_T))

    if (S_norm_inner > 0.01 * T_norm_inner || S_norm_outer > 0.01 * T_norm_outer) && get_rank() == 0
        @warn """
        Non-negligible spheroidal (S) component detected in velocity boundary conditions!
        S_inner/T_inner = $(S_norm_inner/max(T_norm_inner, 1e-10))
        S_outer/T_outer = $(S_norm_outer/max(T_norm_outer, 1e-10))

        This suggests the boundary conditions may not be solenoidal (∇·v ≠ 0).
        The current implementation assumes incompressible flow and IGNORES the S component.

        For compressible flows, the code would need to be extended to include a spheroidal field.
        """
    end
    
    # Update time index
    update_velocity_time_index!(velocity_field, time_index)

    return velocity_field
end

"""
    update_time_dependent_velocity_boundaries!(velocity_field, current_time::Float64)

Update time-dependent velocity boundary conditions.
"""
function update_time_dependent_velocity_boundaries!(velocity_field, current_time::Float64)
    
    boundary_set, _ = get_velocity_boundary_data(velocity_field)
    if boundary_set === nothing
        return velocity_field
    end
    
    # Check if boundaries are time-dependent
    if !boundary_set.inner_boundary.is_time_dependent && !boundary_set.outer_boundary.is_time_dependent
        return velocity_field  # Nothing to update
    end
    
    # Find time index for current time
    time_index = find_boundary_time_index(boundary_set, current_time)
    current_time_index = get_velocity_time_index(velocity_field)
    
    # Only update if time index has changed
    if time_index != current_time_index
        apply_velocity_boundary_conditions!(velocity_field, time_index)
        
        if get_rank() == 0
            @info "Updated velocity boundaries to time index $time_index (t=$current_time)"
        end
    end
    
    return velocity_field
end

"""
    get_current_velocity_boundaries(velocity_field)

Get current velocity boundary conditions.
"""
function get_current_velocity_boundaries(velocity_field)
    
    boundary_set, cache = get_velocity_boundary_data(velocity_field)
    if boundary_set === nothing || cache === nothing
        return Dict(:error => "No boundary conditions loaded")
    end
    
    time_index = get_velocity_time_index(velocity_field)
    
    # Get current boundary data
    inner_physical = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
    outer_physical = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
    
    # Get spectral coefficients
    innerᵀ_spectral = velocity_field.toroidal.boundary_values[1, :]
    outerᵀ_spectral = velocity_field.toroidal.boundary_values[2, :]
    innerᴾ_spectral = velocity_field.poloidal.boundary_values[1, :]
    outerᴾ_spectral = velocity_field.poloidal.boundary_values[2, :]
    
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
            "components" => ["v_r", "v_theta", "v_phi"]
        )
    )
end

"""
    set_programmatic_velocity_boundaries!(velocity_field, inner_spec::Tuple, outer_spec::Tuple)

Set programmatic velocity boundary conditions.
"""
function set_programmatic_velocity_boundaries!(velocity_field, inner_spec::Tuple, outer_spec::Tuple)
    
    boundary_specs = Dict(:inner => inner_spec, :outer => outer_spec)
    return load_velocity_boundary_conditions!(velocity_field, boundary_specs)
end

"""
    get_velocity_boundary_data(velocity_field)

Return `(boundary_set, cache)` for the provided velocity field, falling back to
module-level storage when the struct does not carry boundary metadata.
"""
function get_velocity_boundary_data(velocity_field)
    if isdefined(@__MODULE__, :_velocity_boundary_cache)
        field_id = objectid(velocity_field)
        if haskey(_velocity_boundary_cache, field_id)
            data = _velocity_boundary_cache[field_id]
            return data[:boundary_set], data[:interpolation_cache]
        end
    end

    if hasfield(typeof(velocity_field), :boundary_condition_set)
        return velocity_field.boundary_condition_set, velocity_field.boundary_interpolation_cache
    end

    return nothing, nothing
end

"""
    get_velocity_time_index(velocity_field)

Fetch the current boundary time index for a velocity field, honoring fallback storage.
"""
function get_velocity_time_index(velocity_field)
    if isdefined(@__MODULE__, :_velocity_boundary_cache)
        field_id = objectid(velocity_field)
        if haskey(_velocity_boundary_cache, field_id)
            return _velocity_boundary_cache[field_id][:time_index]
        end
    end

    if hasfield(typeof(velocity_field), :boundary_time_index)
        return velocity_field.boundary_time_index[]
    end

    return 1
end

"""
    update_velocity_time_index!(velocity_field, time_index)

Update cached time indices for velocity boundary conditions in both the field
and the module-level fallback cache (when present).
"""
function update_velocity_time_index!(velocity_field, time_index::Int)
    if isdefined(@__MODULE__, :_velocity_boundary_cache)
        field_id = objectid(velocity_field)
        if haskey(_velocity_boundary_cache, field_id)
            _velocity_boundary_cache[field_id][:time_index] = time_index
        end
    end

    if hasfield(typeof(velocity_field), :boundary_time_index)
        velocity_field.boundary_time_index[] = time_index
    end
end

"""
    validate_velocity_boundary_files(boundary_specs::Dict, config)

Validate velocity boundary condition files.
"""
function validate_velocity_boundary_files(boundary_specs::Dict, config)
    
    inner_spec = get(boundary_specs, :inner, nothing)
    outer_spec = get(boundary_specs, :outer, nothing)
    
    errors = String[]
    
    # Validate file specifications
    if isa(inner_spec, String)
        try
            validate_netcdf_boundary_file(inner_spec, ["velocity", "u", "v", "w"])
            # Check vector components
            inner_data = read_netcdf_boundary_data(inner_spec, precision=config.T)
            if inner_data.ncomponents != 3
                push!(errors, "Inner velocity file must have 3 components (v_r, v_theta, v_phi)")
            end
        catch e
            push!(errors, "Inner boundary file error: $e")
        end
    end
    
    if isa(outer_spec, String)
        try
            validate_netcdf_boundary_file(outer_spec, ["velocity", "u", "v", "w"])
            # Check vector components
            outer_data = read_netcdf_boundary_data(outer_spec, precision=config.T)
            if outer_data.ncomponents != 3
                push!(errors, "Outer velocity file must have 3 components (v_r, v_theta, v_phi)")
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
            validate_boundary_compatibility(inner_data, outer_data, "velocity")
        catch e
            push!(errors, "Boundary compatibility error: $e")
        end
    end
    
    if !isempty(errors)
        error_msg = "Velocity boundary validation failed:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
    
    return true
end

"""
    velocity_to_qst_coefficients(v_r, v_theta, v_phi, config)

Convert physical velocity components (v_r, v_θ, v_φ) to QST spectral coefficients
by using proper SHTnsKit decomposition.

The QST decomposition used by SHTnsKit:
- Q: Radial component coefficients (transforms like scalar field)
- S: Spheroidal horizontal component coefficients (curl-free part)
- T: Toroidal horizontal component coefficients (divergence-free part)

# Arguments
- `v_r`: Radial velocity component [nlat, nlon]
- `v_theta`: Colatitude velocity component [nlat, nlon]
- `v_phi`: Azimuthal velocity component [nlat, nlon]
- `config`: SHTnsKit configuration

# Returns
- `(Q_coeffs, S_coeffs, T_coeffs)`: QST spectral coefficients
"""
function velocity_to_qst_coefficients(v_r, v_theta, v_phi, config)

    # Use SHTnsKit's proper QST decomposition for 3D vector fields
    # Q: radial component (scalar-like)
    # S: spheroidal horizontal component (curl-free part)
    # T: toroidal horizontal component (divergence-free part)

    # Q component: radial velocity (scalar-like transform using SHTnsKit)
    Q_coeffs = shtns_physical_to_spectral(v_r, config)

    # S and T components: proper spheroidal-toroidal decomposition
    # Using SHTnsKit's spat_to_SHsphtor function for horizontal components
    try
        # Create temporary SHTConfig for the decomposition
        nlat, nlon = size(v_theta)
        shtconfig = SHTnsKit.SHTConfig(config.lmax; nlat=nlat, nlon=nlon)

        # Use SHTnsKit's proper spheroidal-toroidal decomposition
        S_matrix, T_matrix = SHTnsKit.spat_to_SHsphtor(shtconfig, v_theta, v_phi)

        # Convert matrices to coefficient vectors using proper lm-indexing
        # The indexing must match the convention used elsewhere in the code
        nlm = config.nlm
        S_coeffs = zeros(config.T, nlm)
        T_coeffs = zeros(config.T, nlm)

        # Extract coefficients following the same lm-indexing as the main simulation
        # Typically: idx increases as we loop over l, then m (with appropriate m range)
        idx = 0
        for l in 0:config.lmax
            # Determine the m range for this l
            # For complex harmonics: m from -min(l,mmax) to +min(l,mmax)
            # For real harmonics stored efficiently: m from 0 to min(l,mmax)
            m_max = min(l, config.mmax)

            for m in 0:m_max
                idx += 1
                if idx <= nlm && (l+1) <= size(S_matrix, 1) && (m+1) <= size(S_matrix, 2)
                    # SHTnsKit matrices are typically (lmax+1) × (mmax+1) in size
                    # Extract real part for consistency (boundary conditions are typically real)
                    S_coeffs[idx] = real(S_matrix[l+1, m+1])
                    T_coeffs[idx] = real(T_matrix[l+1, m+1])
                elseif idx <= nlm
                    # If matrix is smaller than expected, zero-pad
                    S_coeffs[idx] = zero(config.T)
                    T_coeffs[idx] = zero(config.T)
                end
            end
        end

        # Verify we processed the expected number of modes
        if idx != nlm && get_rank() == 0
            @warn "QST extraction: processed $idx modes but nlm=$(nlm). Check lm-indexing consistency."
        end

    catch e
        error_msg = """
        Failed to perform proper spheroidal-toroidal decomposition of velocity field.
        Error: $e

        The QST decomposition is mathematically required for correct velocity boundary conditions.
        Treating velocity components as independent scalar fields (the old fallback) is incorrect
        because it ignores the coupling between v_θ and v_φ in spherical coordinates.

        Possible solutions:
        1. Check that SHTnsKit.spat_to_SHsphtor is properly installed and configured
        2. Verify that the grid dimensions (nlat=$(size(v_theta,1)), nlon=$(size(v_theta,2)))
           are compatible with lmax=$(config.lmax)
        3. Ensure the SHTnsKit configuration is properly initialized
        """
        throw(ErrorException(error_msg))
    end

    return Q_coeffs, S_coeffs, T_coeffs
end

# Function moved to main bcs module to avoid duplication

"""
    enforce_velocity_boundary_constraints!(velocity_field, bc_type::Symbol=:no_slip)

Enforce specific velocity boundary constraints based on boundary condition type.

NOTE: This function is for programmatically setting boundary constraints.
For boundary conditions loaded from files or other sources, use
load_velocity_boundary_conditions!() instead.

# Arguments
- `velocity_field`: Velocity field structure with toroidal and poloidal components
- `bc_type`: Type of boundary condition (:no_slip, :stress_free, :impermeable)

# Boundary Condition Mapping (for solenoidal/incompressible flows):
- No-slip: v_r = v_θ = v_φ = 0 → Q = T = 0 at boundaries (Dirichlet)
- Stress-free: v_r = 0 (Dirichlet), tangential stress = 0 → Q = 0 (Dirichlet), ∂T/∂r = T/r (Neumann)
- Impermeable: v_r = 0 → Q = 0 at boundaries (Dirichlet), T unconstrained

# Field naming convention:
- velocity_field.poloidal actually stores Q (radial component)
- velocity_field.toroidal actually stores T (tangential toroidal component)
- S component (spheroidal) is zero for solenoidal flows
"""
function enforce_velocity_boundary_constraints!(velocity_field, bc_type::Symbol=:no_slip)

    if bc_type == :no_slip
        # No-slip: all velocity components = 0 at boundaries
        # Q = T = 0 with Dirichlet BCs

        if hasfield(typeof(velocity_field), :poloidal) && hasfield(typeof(velocity_field.poloidal), :boundary_values)
            fill!(velocity_field.poloidal.boundary_values, 0.0)  # Q = 0 (radial)
            fill!(velocity_field.poloidal.bc_type_inner, Int(DIRICHLET))
            fill!(velocity_field.poloidal.bc_type_outer, Int(DIRICHLET))
        end

        if hasfield(typeof(velocity_field), :toroidal) && hasfield(typeof(velocity_field.toroidal), :boundary_values)
            fill!(velocity_field.toroidal.boundary_values, 0.0)  # T = 0 (toroidal)
            fill!(velocity_field.toroidal.bc_type_inner, Int(DIRICHLET))
            fill!(velocity_field.toroidal.bc_type_outer, Int(DIRICHLET))
        end

    elseif bc_type == :stress_free
        # Stress-free: v_r = 0 (Dirichlet), zero tangential stress (Neumann)
        # Q = 0 (Dirichlet), ∂T/∂r = T/r (Neumann, enforced by apply_velocity_flux_bc_spectral!)

        if hasfield(typeof(velocity_field), :poloidal) && hasfield(typeof(velocity_field.poloidal), :boundary_values)
            fill!(velocity_field.poloidal.boundary_values, 0.0)  # Q = 0 (radial)
            fill!(velocity_field.poloidal.bc_type_inner, Int(DIRICHLET))
            fill!(velocity_field.poloidal.bc_type_outer, Int(DIRICHLET))
        end

        if hasfield(typeof(velocity_field), :toroidal)
            # Toroidal component uses Neumann BC (tangential stress = 0)
            # Boundary values are not enforced for Neumann BCs
            if hasfield(typeof(velocity_field.toroidal), :bc_type_inner)
                fill!(velocity_field.toroidal.bc_type_inner, Int(NEUMANN))
                fill!(velocity_field.toroidal.bc_type_outer, Int(NEUMANN))
            end
        end

    elseif bc_type == :impermeable
        # Impermeable: v_r = 0 only (Dirichlet), tangential components unconstrained
        # Q = 0 (Dirichlet), T unconstrained

        if hasfield(typeof(velocity_field), :poloidal) && hasfield(typeof(velocity_field.poloidal), :boundary_values)
            fill!(velocity_field.poloidal.boundary_values, 0.0)  # Q = 0 (radial)
            fill!(velocity_field.poloidal.bc_type_inner, Int(DIRICHLET))
            fill!(velocity_field.poloidal.bc_type_outer, Int(DIRICHLET))
        end
        # Toroidal component BC types remain unchanged (solver determines values)

    else
        throw(ArgumentError("Unknown velocity boundary condition type: $bc_type. Use :no_slip, :stress_free, or :impermeable"))
    end

    return velocity_field
end

export load_velocity_boundary_conditions!, set_programmatic_velocity_boundaries!
export update_time_dependent_velocity_boundaries!, get_current_velocity_boundaries
export validate_velocity_boundary_files, create_programmatic_velocity_boundary
export velocity_to_qst_coefficients, enforce_velocity_boundary_constraints!
