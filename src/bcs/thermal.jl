# ================================================================================
# Temperature Boundary Conditions
# ================================================================================

# Temperature boundary conditions are implemented within the bcs module
# All required types and functions are available within the module scope

"""
    load_temperature_boundary_conditions!(temp_field, boundary_specs::Dict)

Load temperature boundary conditions from various sources.

# Arguments
- `temp_field`: SHTnsTemperatureField structure
- `boundary_specs`: Dictionary specifying boundary sources

# Examples
```julia
# NetCDF files for both boundaries
boundary_specs = Dict(
    :inner => "cmb_temperature.nc",
    :outer => "surface_temperature.nc"
)

# Mixed NetCDF and programmatic
boundary_specs = Dict(
    :inner => "cmb_temperature.nc", 
    :outer => (:uniform, 300.0)
)
```
"""
function load_temperature_boundary_conditions!(temp_field, boundary_specs::Dict)
    
    if get_rank() == 0
        @info "Loading temperature boundary conditions..."
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
        boundary_set = load_temperature_boundaries_from_files(inner_spec, outer_spec, temp_field.config)
    elseif isa(inner_spec, String) && isa(outer_spec, Tuple)
        # Inner from file, outer programmatic
        boundary_set = create_hybrid_temperature_boundaries(inner_spec, outer_spec, temp_field.config)
    elseif isa(inner_spec, Tuple) && isa(outer_spec, String)
        # Inner programmatic, outer from file
        boundary_set = create_hybrid_temperature_boundaries(outer_spec, inner_spec, temp_field.config; swap_boundaries=true)
    elseif isa(inner_spec, Tuple) && isa(outer_spec, Tuple)
        # Both programmatic
        boundary_set = create_programmatic_temperature_boundaries(inner_spec, outer_spec, temp_field.config)
    else
        throw(ArgumentError("Invalid boundary specification format"))
    end
    
    if hasfield(typeof(temp_field), :boundary_condition_set)
        temp_field.boundary_condition_set = boundary_set
        temp_field.boundary_time_index[] = 1
        temp_field.boundary_interpolation_cache = create_temperature_interpolation_cache(boundary_set, temp_field.config)
    else
        # Legacy fallback for field structures without direct storage
        if !isdefined(@__MODULE__, :_temperature_boundary_cache)
            global _temperature_boundary_cache = Dict{UInt64, Any}()
        end

        field_id = objectid(temp_field)
        _temperature_boundary_cache[field_id] = Dict(
            :boundary_set => boundary_set,
            :interpolation_cache => create_temperature_interpolation_cache(boundary_set, temp_field.config),
            :time_index => 1
        )
    end

    # Apply initial boundary conditions (field-level storage preferred when available)
    apply_temperature_boundary_conditions!(temp_field)

    if get_rank() == 0
        print_boundary_info(boundary_set)
        @info "Temperature boundary conditions loaded successfully"
    end

    return temp_field
end

"""
    load_temperature_boundaries_from_files(inner_file::String, outer_file::String, config)

Load temperature boundary conditions from NetCDF files.
"""
function load_temperature_boundaries_from_files(inner_file::String, outer_file::String, config)
    
    # Validate files exist
    for file in [inner_file, outer_file]
        if !isfile(file)
            throw(ArgumentError("Temperature boundary file not found: $file"))
        end
    end
    
    # Read boundary data
    inner_data = read_netcdf_boundary_data(inner_file, precision=config.T)
    outer_data = read_netcdf_boundary_data(outer_file, precision=config.T)
    
    # Create new data structures with correct field type for temperature
    inner_data = create_boundary_data(
        inner_data.values, "temperature";
        theta=inner_data.theta, phi=inner_data.phi, time=inner_data.time,
        units=inner_data.units, description=inner_data.description,
        file_path=inner_data.file_path
    )
    
    outer_data = create_boundary_data(
        outer_data.values, "temperature";
        theta=outer_data.theta, phi=outer_data.phi, time=outer_data.time,
        units=outer_data.units, description=outer_data.description,
        file_path=outer_data.file_path
    )
    
    # Validate temperature ranges
    validate_temperature_range(inner_data)
    validate_temperature_range(outer_data)
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "temperature")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "temperature", TEMPERATURE, time()
    )
    
    return boundary_set
end

"""
    create_hybrid_temperature_boundaries(file_spec::String, prog_spec::Tuple, config; swap_boundaries=false)

Create hybrid temperature boundaries (one from file, one programmatic).
"""
function create_hybrid_temperature_boundaries(file_spec::String, prog_spec::Tuple, config; swap_boundaries=false)
    
    # Load file-based boundary
    temp_data = read_netcdf_boundary_data(file_spec, precision=config.T)
    file_data = create_boundary_data(
        temp_data.values, "temperature";
        theta=temp_data.theta, phi=temp_data.phi, time=temp_data.time,
        units=temp_data.units, description=temp_data.description,
        file_path=temp_data.file_path
    )
    validate_temperature_range(file_data)
    
    # Create programmatic boundary
    pattern, amplitude = prog_spec[1], prog_spec[2]
    parameters = length(prog_spec) >= 3 ? prog_spec[3] : Dict()
    
    prog_data = create_programmatic_boundary(
        pattern, config, amplitude; 
        parameters=parameters, field_type="temperature"
    )
    
    # Validate temperature range for programmatic data
    validate_temperature_range(prog_data)
    
    # Ensure same grid resolution
    if file_data.nlat != config.nlat || file_data.nlon != config.nlon
        # Interpolate file data to config grid
        theta_target = collect(range(0, π, length=config.nlat))
        phi_target = collect(range(0, 2π, length=config.nlon+1)[1:end-1])
        
        interpolated_values = interpolate_boundary_to_grid(file_data, theta_target, phi_target, 1)
        
        file_data = create_boundary_data(
            interpolated_values, "temperature";
            theta=theta_target, phi=phi_target, time=nothing,
            units=file_data.units, description=file_data.description,
            file_path=file_data.file_path
        )
        
        validate_temperature_range(file_data)
    end
    
    # Assign boundaries based on swap_boundaries flag
    if swap_boundaries
        inner_data, outer_data = prog_data, file_data
    else
        inner_data, outer_data = file_data, prog_data
    end
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "temperature")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "temperature", TEMPERATURE, time()
    )
    
    return boundary_set
end

"""
    create_programmatic_temperature_boundaries(inner_spec::Tuple, outer_spec::Tuple, config)

Create fully programmatic temperature boundaries.
"""
function create_programmatic_temperature_boundaries(inner_spec::Tuple, outer_spec::Tuple, config)
    
    # Create inner boundary
    inner_pattern, inner_amplitude = inner_spec[1], inner_spec[2]
    inner_parameters = length(inner_spec) >= 3 ? inner_spec[3] : Dict()
    
    inner_data = create_programmatic_boundary(
        inner_pattern, config, inner_amplitude;
        parameters=inner_parameters, field_type="temperature"
    )
    
    # Create outer boundary
    outer_pattern, outer_amplitude = outer_spec[1], outer_spec[2]
    outer_parameters = length(outer_spec) >= 3 ? outer_spec[3] : Dict()
    
    outer_data = create_programmatic_boundary(
        outer_pattern, config, outer_amplitude;
        parameters=outer_parameters, field_type="temperature"
    )
    
    # Validate temperature ranges
    validate_temperature_range(inner_data)
    validate_temperature_range(outer_data)
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "temperature")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "temperature", TEMPERATURE, time()
    )
    
    return boundary_set
end

"""
    create_temperature_interpolation_cache(boundary_set::BoundaryConditionSet, config)

Create interpolation cache for temperature boundaries.
"""
function create_temperature_interpolation_cache(boundary_set::BoundaryConditionSet, config)
    
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
    apply_temperature_boundary_conditions!(temp_field, time_index::Int=1)

Apply temperature boundary conditions to the field.
"""
function apply_temperature_boundary_conditions!(temp_field, time_index::Int=1)
    
    # Get boundary data (from field)
    boundary_set, cache = get_temperature_boundary_data(temp_field)
    if boundary_set === nothing
        @warn "No boundary conditions loaded for temperature field"
        return temp_field
    end
    
    # Interpolate boundary data to simulation grid
    inner_physical = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
    outer_physical = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
    
    # Transform to spectral space using SHTnsKit
    inner_spectral = shtns_physical_to_spectral(inner_physical, temp_field.config)
    outer_spectral = shtns_physical_to_spectral(outer_physical, temp_field.config)

    # Apply to boundary arrays
    temp_field.boundary_values[1, :] = inner_spectral  # Inner boundary
    temp_field.boundary_values[2, :] = outer_spectral  # Outer boundary

    # Set boundary condition types based on boundary specifications
    # Default to Dirichlet (fixed temperature), but check for flux BCs
    inner_bc_type = infer_temperature_bc_type(boundary_set.inner_boundary)
    outer_bc_type = infer_temperature_bc_type(boundary_set.outer_boundary)

    # Validate that BC data matches BC type (only on rank 0 to avoid spam)
    if get_rank() == 0
        validate_temperature_flux_boundary(boundary_set.inner_boundary, inner_bc_type)
        validate_temperature_flux_boundary(boundary_set.outer_boundary, outer_bc_type)
    end

    fill!(temp_field.bc_type_inner, inner_bc_type)
    fill!(temp_field.bc_type_outer, outer_bc_type)

    # Update time index
    update_temperature_time_index!(temp_field, time_index)

    return temp_field
end

"""
    infer_temperature_bc_type(boundary::BoundaryData)

Infer boundary condition type (Dirichlet or Neumann) from boundary metadata.

Checks the description field for keywords:
- "flux", "neumann", "heat flux" → NEUMANN (∂T/∂r specified)
- Otherwise → DIRICHLET (T specified)
"""
function infer_temperature_bc_type(boundary::BoundaryData)
    desc = lowercase(boundary.description)

    # Check for flux/Neumann BC indicators
    if occursin("flux", desc) || occursin("neumann", desc) ||
       occursin("heat flux", desc) || occursin("gradient", desc)
        return Int(NEUMANN)
    else
        # Default to Dirichlet (fixed temperature)
        return Int(DIRICHLET)
    end
end

"""
    update_time_dependent_temperature_boundaries!(temp_field, current_time::Float64)

Update time-dependent temperature boundary conditions.
"""
function update_time_dependent_temperature_boundaries!(temp_field, current_time::Float64)
    
    boundary_set, _ = get_temperature_boundary_data(temp_field)
    if boundary_set === nothing
        return temp_field
    end
    
    # Check if boundaries are time-dependent
    if !boundary_set.inner_boundary.is_time_dependent && !boundary_set.outer_boundary.is_time_dependent
        return temp_field  # Nothing to update
    end
    
    # Find time index for current time
    time_index = find_temperature_boundary_time_index(boundary_set, current_time)
    
    # Only update if time index has changed
    current_time_index = get_temperature_time_index(temp_field)
    if time_index != current_time_index
        apply_temperature_boundary_conditions!(temp_field, time_index)
        
        if get_rank() == 0
            @info "Updated temperature boundaries to time index $time_index (t=$current_time)"
        end
    end
    
    return temp_field
end

"""
    find_temperature_boundary_time_index(boundary_set::BoundaryConditionSet, current_time::Float64)

Find the appropriate time index for the current simulation time.
"""
function find_temperature_boundary_time_index(boundary_set::BoundaryConditionSet, current_time::Float64)
    
    # Use time coordinates from inner boundary (both should be compatible)
    time_coords = boundary_set.inner_boundary.time
    
    if time_coords === nothing
        return 1  # Time-independent
    end
    
    # Find closest time index
    if current_time <= time_coords[1]
        return 1
    elseif current_time >= time_coords[end]
        return length(time_coords)
    else
        # Linear search for closest time
        for i in 1:(length(time_coords)-1)
            if time_coords[i] <= current_time <= time_coords[i+1]
                # Choose closer time point
                if abs(current_time - time_coords[i]) <= abs(current_time - time_coords[i+1])
                    return i
                else
                    return i + 1
                end
            end
        end
    end
    
    return 1  # Fallback
end

# Function moved to main BoundaryConditions module to avoid duplication

"""
    get_temperature_boundary_data(temp_field)

Get boundary data from field or fallback cache.
"""
function get_temperature_boundary_data(temp_field)
    # First check if we have data in the module-level cache
    if isdefined(@__MODULE__, :_temperature_boundary_cache)
        field_id = objectid(temp_field)
        if haskey(_temperature_boundary_cache, field_id)
            data = _temperature_boundary_cache[field_id]
            return data[:boundary_set], data[:interpolation_cache]
        end
    end

    # Fallback to field storage (for legacy compatibility)
    boundary_set = temp_field.boundary_condition_set
    cache = temp_field.boundary_interpolation_cache
    return boundary_set, cache
end

"""
    get_temperature_time_index(temp_field)

Get current time index from field or fallback cache.
"""
function get_temperature_time_index(temp_field)
    # Check cache first
    if isdefined(@__MODULE__, :_temperature_boundary_cache)
        field_id = objectid(temp_field)
        if haskey(_temperature_boundary_cache, field_id)
            return _temperature_boundary_cache[field_id][:time_index]
        end
    end

    # Fallback to field storage
    return temp_field.boundary_time_index[]
end

"""
    update_temperature_time_index!(temp_field, time_index::Int)

Update time index in field or fallback cache.
"""
function update_temperature_time_index!(temp_field, time_index::Int)
    # Update cache if it exists
    if isdefined(@__MODULE__, :_temperature_boundary_cache)
        field_id = objectid(temp_field)
        if haskey(_temperature_boundary_cache, field_id)
            _temperature_boundary_cache[field_id][:time_index] = time_index
            temp_field.boundary_time_index[] = time_index
            return
        end
    end

    # Fallback to field storage
    temp_field.boundary_time_index[] = time_index
end

"""
    validate_temperature_range(boundary_data::BoundaryData)

Validate that temperature values are in a reasonable physical range.
"""
function validate_temperature_range(boundary_data::BoundaryData)
    
    min_val = minimum(boundary_data.values)
    max_val = maximum(boundary_data.values)
    
    # Check for reasonable temperature range (assuming Kelvin or dimensionless)
    if min_val < 0.0
        @warn "Temperature values below zero: min = $min_val (assuming Kelvin)"
    end
    
    if max_val > 10000.0  # Arbitrary but reasonable upper bound
        @warn "Very high temperature values: max = $max_val"
    end
    
    return boundary_data
end

"""
    get_current_temperature_boundaries(temp_field)

Get current temperature boundary conditions.
"""
function get_current_temperature_boundaries(temp_field)
    
    boundary_set, cache = get_temperature_boundary_data(temp_field)
    if boundary_set === nothing
        return Dict(:error => "No boundary conditions loaded")
    end
    
    time_index = get_temperature_time_index(temp_field)
    
    # Get current boundary data
    inner_physical = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
    outer_physical = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
    
    # Get spectral coefficients
    inner_spectral = temp_field.boundary_values[1, :]
    outer_spectral = temp_field.boundary_values[2, :]
    
    return Dict(
        :inner_physical => inner_physical,
        :outer_physical => outer_physical,
        :inner_spectral => inner_spectral,
        :outer_spectral => outer_spectral,
        :time_index => time_index,
        :metadata => Dict(
            "field_name" => boundary_set.field_name,
            "source" => "file_based",
            "inner_file" => boundary_set.inner_boundary.file_path,
            "outer_file" => boundary_set.outer_boundary.file_path,
            "creation_time" => boundary_set.creation_time
        )
    )
end

"""
    set_programmatic_temperature_boundaries!(temp_field, inner_spec::Tuple, outer_spec::Tuple)

Set programmatic temperature boundary conditions.
"""
function set_programmatic_temperature_boundaries!(temp_field, inner_spec::Tuple, outer_spec::Tuple)
    
    boundary_specs = Dict(:inner => inner_spec, :outer => outer_spec)
    return load_temperature_boundary_conditions!(temp_field, boundary_specs)
end

"""
    validate_temperature_boundary_files(boundary_specs::Dict, config)

Validate temperature boundary condition files.
"""
function validate_temperature_boundary_files(boundary_specs::Dict, config)
    
    inner_spec = get(boundary_specs, :inner, nothing)
    outer_spec = get(boundary_specs, :outer, nothing)
    
    errors = String[]
    
    # Validate file specifications
    if isa(inner_spec, String)
        try
            validate_netcdf_boundary_file(inner_spec, ["temperature"])
        catch e
            push!(errors, "Inner boundary file error: $e")
        end
    end
    
    if isa(outer_spec, String)
        try
            validate_netcdf_boundary_file(outer_spec, ["temperature"])
        catch e
            push!(errors, "Outer boundary file error: $e")
        end
    end
    
    # If both are files, check compatibility
    if isa(inner_spec, String) && isa(outer_spec, String)
        try
            inner_data = read_netcdf_boundary_data(inner_spec, precision=config.T)
            outer_data = read_netcdf_boundary_data(outer_spec, precision=config.T)
            validate_boundary_compatibility(inner_data, outer_data, "temperature")
        catch e
            push!(errors, "Boundary compatibility error: $e")
        end
    end
    
    if !isempty(errors)
        error_msg = "Temperature boundary validation failed:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
    
    return true
end

"""
    create_layered_temperature_boundary(config, layer_specs::Vector{Tuple{Real, Real, Real}})

Create layered temperature boundary conditions.

# Arguments
- `layer_specs`: Vector of (colatitude_start, colatitude_end, temperature) tuples

# Example
```julia
# Create three-layer temperature structure
layer_specs = [
    (0.0, π/3, 1000.0),    # High temperature in top layer
    (π/3, 2π/3, 500.0),   # Medium temperature in middle layer  
    (2π/3, π, 100.0)      # Low temperature in bottom layer
]
```
"""
function create_layered_temperature_boundary(config, layer_specs::Vector{Tuple{Real, Real, Real}})
    
    if isempty(layer_specs)
        throw(ArgumentError("At least one layer specification required"))
    end
    
    # Create coordinate grids
    theta = collect(range(0, π, length=config.nlat))
    phi = collect(range(0, 2π, length=config.nlon+1)[1:end-1])
    
    # Initialize temperature array
    values = zeros(config.T, config.nlat, config.nlon)
    
    # Apply layered structure
    for (i, θ) in enumerate(theta)
        for (θ_start, θ_end, temperature) in layer_specs
            if θ_start <= θ <= θ_end
                values[i, :] .= temperature
                break
            end
        end
    end
    
    # Create boundary data
    boundary_data = create_boundary_data(
        values, "temperature";
        theta=theta, phi=phi, time=nothing,
        units="K",
        description="Layered temperature boundary ($(length(layer_specs)) layers)",
        file_path="programmatic"
    )
    
    # Validate temperature range
    validate_temperature_range(boundary_data)
    
    return boundary_data
end

"""
    apply_temperature_boundaries!(temp_field, boundary_set::BoundaryConditionSet; time::Float64=0.0)

Apply temperature boundary conditions from a BoundaryConditionSet to a temperature field.
This is a convenience wrapper that loads the boundary set and applies it.
"""
function apply_temperature_boundaries!(temp_field, boundary_set::BoundaryConditionSet; time::Float64=0.0)

    # Store boundary conditions in a module-level cache since the field structure
    # expects Union{Dict{String, Any}, Nothing} but we have BoundaryConditionSet
    if !isdefined(@__MODULE__, :_temperature_boundary_cache)
        global _temperature_boundary_cache = Dict{UInt64, Any}()
    end

    field_id = objectid(temp_field)
    _temperature_boundary_cache[field_id] = Dict(
        :boundary_set => boundary_set,
        :interpolation_cache => create_temperature_interpolation_cache(boundary_set, temp_field.config),
        :time_index => 1
    )

    # Apply boundary conditions (no need to modify immutable field structure)
    apply_temperature_boundary_conditions!(temp_field)

    return temp_field
end

"""
    enforce_temperature_boundary_constraints!(temp_field, bc_spec::Dict)

Enforce specific temperature boundary constraints based on user specification.

# Arguments
- `temp_field`: Temperature field structure
- `bc_spec`: Dictionary specifying boundary condition types

# BC Specification Format
```julia
bc_spec = Dict(
    :inner => :dirichlet,  # or :neumann, :flux
    :outer => :dirichlet,  # or :neumann, :flux
    :inner_value => 1000.0,  # for Dirichlet
    :outer_value => 300.0,   # for Dirichlet
    :inner_flux => 0.1,      # for Neumann (∂T/∂r)
    :outer_flux => 0.0       # for Neumann (∂T/∂r)
)
```

# Boundary Condition Types
- `:dirichlet` - Fixed temperature: T = T₀ at boundary
- `:neumann` or `:flux` - Fixed heat flux: ∂T/∂r = q at boundary
- `:mixed` - Mixed/Robin: α T + β ∂T/∂r = γ (future)

# Physical Interpretation
- **Dirichlet (Fixed T)**: Typical for prescribed temperature boundaries
  * Inner (CMB): Fixed temperature from core evolution
  * Outer (surface): Fixed temperature from atmospheric coupling

- **Neumann (Fixed flux)**: Typical for heat flux boundaries
  * Inner: Uniform heating from below (∂T/∂r = q)
  * Outer: Zero flux (insulated, ∂T/∂r = 0)

# Examples
```julia
# Fixed temperature at both boundaries
enforce_temperature_boundary_constraints!(temp_field, Dict(
    :inner => :dirichlet, :inner_value => 1000.0,
    :outer => :dirichlet, :outer_value => 300.0
))

# Uniform heating from below, insulated top
enforce_temperature_boundary_constraints!(temp_field, Dict(
    :inner => :flux, :inner_flux => 0.1,
    :outer => :flux, :outer_flux => 0.0
))
```
"""
function enforce_temperature_boundary_constraints!(temp_field, bc_spec::Dict)

    # Process inner boundary
    inner_type = get(bc_spec, :inner, :dirichlet)
    if inner_type == :dirichlet
        # Fixed temperature - keep as Dirichlet (already set)
        fill!(temp_field.bc_type_inner, Int(DIRICHLET))

        # Set boundary value if provided
        if haskey(bc_spec, :inner_value)
            inner_val = bc_spec[:inner_value]
            # Set uniform value in l=0,m=0 mode (spherically symmetric)
            temp_field.boundary_values[1, 1] = inner_val
            # Zero out other modes for uniform BC
            temp_field.boundary_values[1, 2:end] .= 0.0
        end

    elseif inner_type == :neumann || inner_type == :flux
        # Fixed heat flux - use Neumann BC
        fill!(temp_field.bc_type_inner, Int(NEUMANN))

        # Set flux value if provided
        if haskey(bc_spec, :inner_flux)
            flux_val = bc_spec[:inner_flux]
            # Flux is stored in boundary_values for Neumann BCs
            # For uniform flux: set l=0,m=0 mode to flux value
            temp_field.boundary_values[1, 1] = flux_val
            temp_field.boundary_values[1, 2:end] .= 0.0
        end

    else
        throw(ArgumentError("Unknown inner boundary type: $inner_type. Use :dirichlet or :neumann/:flux"))
    end

    # Process outer boundary
    outer_type = get(bc_spec, :outer, :dirichlet)
    if outer_type == :dirichlet
        fill!(temp_field.bc_type_outer, Int(DIRICHLET))

        if haskey(bc_spec, :outer_value)
            outer_val = bc_spec[:outer_value]
            temp_field.boundary_values[2, 1] = outer_val
            temp_field.boundary_values[2, 2:end] .= 0.0
        end

    elseif outer_type == :neumann || outer_type == :flux
        fill!(temp_field.bc_type_outer, Int(NEUMANN))

        if haskey(bc_spec, :outer_flux)
            flux_val = bc_spec[:outer_flux]
            temp_field.boundary_values[2, 1] = flux_val
            temp_field.boundary_values[2, 2:end] .= 0.0
        end

    else
        throw(ArgumentError("Unknown outer boundary type: $outer_type. Use :dirichlet or :neumann/:flux"))
    end

    return temp_field
end

"""
    validate_flux_boundary_values(values::Array, field_type::String;
                                  typical_value_range::Tuple=(0.0, 1e10),
                                  check_units::Bool=true)

Validate that boundary values are physically reasonable for flux boundary conditions.

For heat flux boundaries:
- Values should be in W/m² or dimensionless heat flux
- Typical range: 0.01 - 1000 W/m² for Earth
- Check that values aren't accidentally in Kelvin (which would be 100-1000× larger)

# Arguments
- `values`: Boundary data array
- `field_type`: Type of field ("temperature", "composition")
- `typical_value_range`: Expected range for flux values
- `check_units`: Whether to perform unit sanity checks

# Returns
- `is_valid`: Boolean
- `warnings`: Array of warning messages
"""
function validate_flux_boundary_values(values::Array, field_type::String;
                                      typical_value_range::Tuple=(0.0, 1e10),
                                      check_units::Bool=true)
    warnings = String[]
    is_valid = true

    # Compute statistics
    values_abs = abs.(values)
    min_val = minimum(values_abs[values_abs .> 1e-15])
    max_val = maximum(values_abs)
    mean_val = _Statistics.mean(values_abs[values_abs .> 1e-15])

    # Check if values are in expected range
    min_expected, max_expected = typical_value_range

    if max_val > max_expected
        push!(warnings, """
            Maximum flux value ($(max_val)) exceeds expected range ($(max_expected)).

            Possible issues:
            1. Values may be in wrong units (e.g., temperature K instead of flux W/m²)
            2. Data may need dimensional scaling
            3. Flux values may be unrealistic for the physical system
            """)
        is_valid = false
    end

    if field_type == "temperature" && check_units
        # For temperature: check if values look like temperature instead of flux
        # Typical Earth: CMB heat flux ~ 0.1 W/m², temperatures ~ 1000-4000 K

        if mean_val > 100.0  # If mean "flux" > 100, probably temperature in K
            push!(warnings, """
                Heat flux values unusually large (mean = $(mean_val)).

                CRITICAL: Are these temperature values (K) instead of flux (W/m²)?

                Expected heat flux range: 0.01 - 100 W/m² for Earth-like bodies
                Your values: $(min_val) - $(max_val)

                If these are temperature values, change BC type to DIRICHLET!
                """)
            is_valid = false
        end

        # Check for dimensionless flux (should be O(1) or less)
        if minimum(values_abs) < 1e-6 && maximum(values_abs) < 1e-3
            push!(warnings, """
                Heat flux values very small (< 1e-3).

                This might be correct for:
                - Dimensionless formulation with small Rayleigh number
                - Very weak heating

                Or it might indicate:
                - Missing dimensional scaling
                - Incorrect units
                """)
        end
    end

    return is_valid, warnings
end

"""
    validate_temperature_flux_boundary(boundary::BoundaryData, bc_type::Int)

Validate that a temperature boundary condition has correct data type for its BC type.

For Neumann (flux) BCs: ensures values represent heat flux, not temperature
For Dirichlet BCs: ensures values represent temperature
"""
function validate_temperature_flux_boundary(boundary::BoundaryData, bc_type::Int)

    if bc_type == Int(NEUMANN)
        # For flux BC, values should be flux (W/m² or dimensionless)
        # NOT temperature (K)

        # Check units field
        units_lower = lowercase(boundary.units)
        is_temp_units = occursin("k", units_lower) || occursin("kelvin", units_lower) ||
                       occursin("celsius", units_lower) || occursin("°", units_lower)

        if is_temp_units
            @warn """
            ⚠️ CRITICAL: Neumann (flux) BC specified but units indicate temperature!

            Boundary: $(boundary.description)
            Units: $(boundary.units)
            BC type: NEUMANN (expects flux)

            Neumann BCs require heat FLUX values (W/m² or ∂T/∂r), not temperature!

            Fix options:
            1. Change description to remove "flux" keywords → will use DIRICHLET
            2. Provide actual flux data with correct units (W/m² or dimensionless)
            3. Set units to "W/m²" or "dimensionless" to match data type

            Proceeding, but results will be INCORRECT if data is actually temperature!
            """
        end

        # Validate flux values are reasonable
        is_valid, warnings = validate_flux_boundary_values(
            boundary.values, "temperature";
            typical_value_range=(0.0, 1000.0),  # W/m²
            check_units=true
        )

        if !is_valid && get_rank() == 0
            for warning in warnings
                @warn warning
            end
        end

    else  # DIRICHLET
        # For fixed temperature, values should be temperature (K)

        # Check if values look like flux instead of temperature
        values_abs = abs.(boundary.values)
        max_val = maximum(values_abs)

        if max_val < 10.0  # If max temperature < 10 K, probably flux values
            @warn """
            Dirichlet (fixed T) BC but values very small (max = $(max_val)).

            Expected temperature: 100-10000 K for planetary interiors
            Your values: 0 - $(max_val)

            Possible issues:
            1. Values might be heat flux instead of temperature
            2. Missing dimensional scaling
            3. Using dimensionless temperature with small scale

            Check if BC type should be NEUMANN instead!
            """
        end
    end

    return nothing
end

export load_temperature_boundary_conditions!, set_programmatic_temperature_boundaries!
export update_time_dependent_temperature_boundaries!, get_current_temperature_boundaries
export validate_temperature_boundary_files
export validate_flux_boundary_values, validate_temperature_flux_boundary, create_layered_temperature_boundary
export apply_temperature_boundaries!, enforce_temperature_boundary_constraints!
export infer_temperature_bc_type
