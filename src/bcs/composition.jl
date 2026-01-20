# ================================================================================
# Composition Boundary Conditions
# ================================================================================

# Composition boundary conditions are implemented within the bcs module
# All required types and functions are available within the module scope

"""
    load_composition_boundary_conditions!(𝔽, boundary_specs::Dict)

Load composition boundary conditions from various sources.

# Arguments
- `𝔽`: SHTnsCompositionField structure
- `boundary_specs`: Dictionary specifying boundary sources

# Examples
```julia
# NetCDF files for both boundaries
boundary_specs = Dict(
    :inner => "cmb_composition.nc",
    :outer => "surface_composition.nc"
)

# Mixed NetCDF and programmatic
boundary_specs = Dict(
    :inner => "cmb_composition.nc", 
    :outer => (:uniform, 0.1)  # 10% concentration
)
```
"""
function load_composition_boundary_conditions!(𝔽, boundary_specs::Dict)
    
    if get_rank() == 0
        @info "Loading composition boundary conditions..."
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
        boundary_set = load_composition_boundaries_from_files(inner_spec, outer_spec, 𝔽.config)
    elseif isa(inner_spec, String) && isa(outer_spec, Tuple)
        # Inner from file, outer programmatic
        boundary_set = create_hybrid_composition_boundaries(inner_spec, outer_spec, 𝔽.config)
    elseif isa(inner_spec, Tuple) && isa(outer_spec, String)
        # Inner programmatic, outer from file
        boundary_set = create_hybrid_composition_boundaries(outer_spec, inner_spec, 𝔽.config; swap_boundaries=true)
    elseif isa(inner_spec, Tuple) && isa(outer_spec, Tuple)
        # Both programmatic
        boundary_set = create_programmatic_composition_boundaries(inner_spec, outer_spec, 𝔽.config)
    else
        throw(ArgumentError("Invalid boundary specification format"))
    end
    
    # Store boundary conditions in field (if supported by field structure)
    if hasfield(typeof(𝔽), :boundary_condition_set)
        𝔽.boundary_condition_set = boundary_set
        𝔽.boundary_time_index[] = 1
        # Create interpolation cache
        𝔽.boundary_interpolation_cache = create_composition_interpolation_cache(boundary_set, 𝔽.config)
    else
        # Fallback for legacy field structure - store in global Dict
        # This is temporary until field structures are updated
        if !isdefined(@__MODULE__, :_composition_boundary_cache)
            global _composition_boundary_cache = Dict{UInt64, Any}()
        end
        field_id = objectid(𝔽)
        _composition_boundary_cache[field_id] = Dict(
            :boundary_set => boundary_set,
            :interpolation_cache => create_composition_interpolation_cache(boundary_set, 𝔽.config),
            :time_index => 1
        )
    end
    
    # Apply initial boundary conditions
    apply_composition_boundary_conditions!(𝔽)
    
    if get_rank() == 0
        print_boundary_info(boundary_set)
        @info "Composition boundary conditions loaded successfully"
    end
    
    return 𝔽
end

"""
    load_composition_boundaries_from_files(inner_file::String, outer_file::String, config)

Load composition boundary conditions from NetCDF files.
"""
function load_composition_boundaries_from_files(inner_file::String, outer_file::String, config)
    
    # Validate files exist
    for file in [inner_file, outer_file]
        if !isfile(file)
            throw(ArgumentError("Composition boundary file not found: $file"))
        end
    end
    
    # Read boundary data
    inner_data = read_netcdf_boundary_data(inner_file, precision=config.T)
    outer_data = read_netcdf_boundary_data(outer_file, precision=config.T)
    
    # Create new data structures with correct field type for composition
    inner_data = create_boundary_data(
        inner_data.values, "composition";
        theta=inner_data.theta, phi=inner_data.phi, time=inner_data.time,
        units=inner_data.units, description=inner_data.description,
        file_path=inner_data.file_path
    )
    
    outer_data = create_boundary_data(
        outer_data.values, "composition";
        theta=outer_data.theta, phi=outer_data.phi, time=outer_data.time,
        units=outer_data.units, description=outer_data.description,
        file_path=outer_data.file_path
    )
    
    # Validate composition range [0, 1]
    validate_composition_range(inner_data)
    validate_composition_range(outer_data)
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "composition")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "composition", COMPOSITION, time()
    )
    
    return boundary_set
end

"""
    validate_composition_range(boundary_data::BoundaryData)

Validate that composition values are in the physical range [0, 1].
"""
function validate_composition_range(boundary_data::BoundaryData)
    
    min_val = minimum(boundary_data.values)
    max_val = maximum(boundary_data.values)
    
    if min_val < 0.0 || max_val > 1.0
        @warn "Composition values outside physical range [0, 1]: [$min_val, $max_val]"
    end
    
    # Clamp values to physical range
    boundary_data.values .= clamp.(boundary_data.values, 0.0, 1.0)
    
    return boundary_data
end

"""
    create_hybrid_composition_boundaries(file_spec::String, prog_spec::Tuple, config; swap_boundaries=false)

Create hybrid composition boundaries (one from file, one programmatic).
"""
function create_hybrid_composition_boundaries(file_spec::String, prog_spec::Tuple, config; swap_boundaries=false)
    
    # Load file-based boundary
    temp_data = read_netcdf_boundary_data(file_spec, precision=config.T)
    file_data = create_boundary_data(
        temp_data.values, "composition";
        theta=temp_data.theta, phi=temp_data.phi, time=temp_data.time,
        units=temp_data.units, description=temp_data.description,
        file_path=temp_data.file_path
    )
    validate_composition_range(file_data)
    
    # Create programmatic boundary
    pattern, amplitude = prog_spec[1], prog_spec[2]
    parameters = length(prog_spec) >= 3 ? prog_spec[3] : Dict()
    
    prog_data = create_programmatic_boundary(
        pattern, config, amplitude; 
        parameters=parameters, field_type="composition"
    )
    
    # Validate composition range for programmatic data
    validate_composition_range(prog_data)
    
    # Ensure same grid resolution
    if file_data.nlat != config.nlat || file_data.nlon != config.nlon
        # Interpolate file data to config grid
        theta_target = collect(range(0, π, length=config.nlat))
        phi_target = collect(range(0, 2π, length=config.nlon+1)[1:end-1])
        
        interpolated_values = interpolate_boundary_to_grid(file_data, theta_target, phi_target, 1)
        
        file_data = create_boundary_data(
            interpolated_values, "composition";
            theta=theta_target, phi=phi_target, time=nothing,
            units=file_data.units, description=file_data.description,
            file_path=file_data.file_path
        )
        
        validate_composition_range(file_data)
    end
    
    # Assign boundaries based on swap_boundaries flag
    if swap_boundaries
        inner_data, outer_data = prog_data, file_data
    else
        inner_data, outer_data = file_data, prog_data
    end
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "composition")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "composition", COMPOSITION, time()
    )
    
    return boundary_set
end

"""
    create_programmatic_composition_boundaries(inner_spec::Tuple, outer_spec::Tuple, config)

Create fully programmatic composition boundaries.
"""
function create_programmatic_composition_boundaries(inner_spec::Tuple, outer_spec::Tuple, config)
    
    # Create inner boundary
    inner_pattern, inner_amplitude = inner_spec[1], inner_spec[2]
    inner_parameters = length(inner_spec) >= 3 ? inner_spec[3] : Dict()
    
    inner_data = create_programmatic_boundary(
        inner_pattern, config, inner_amplitude;
        parameters=inner_parameters, field_type="composition"
    )
    
    # Create outer boundary
    outer_pattern, outer_amplitude = outer_spec[1], outer_spec[2]
    outer_parameters = length(outer_spec) >= 3 ? outer_spec[3] : Dict()
    
    outer_data = create_programmatic_boundary(
        outer_pattern, config, outer_amplitude;
        parameters=outer_parameters, field_type="composition"
    )
    
    # Validate composition ranges
    validate_composition_range(inner_data)
    validate_composition_range(outer_data)
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "composition")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "composition", COMPOSITION, time()
    )
    
    return boundary_set
end

"""
    create_composition_interpolation_cache(boundary_set::BoundaryConditionSet, config)

Create interpolation cache for composition boundaries.
"""
function create_composition_interpolation_cache(boundary_set::BoundaryConditionSet, config)
    
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
    apply_composition_boundary_conditions!(𝔽, time_index::Int=1)

Apply composition boundary conditions to the field.
"""
function apply_composition_boundary_conditions!(𝔽, time_index::Int=1)
    
    # Get boundary data (from field or fallback cache)
    boundary_set, cache = get_composition_boundary_data(𝔽)
    if boundary_set === nothing
        @warn "No boundary conditions loaded for composition field"
        return 𝔽
    end
    
    # Interpolate boundary data to simulation grid
    inner_physical = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
    outer_physical = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
    
    # Clamp to valid composition range after interpolation
    inner_physical .= clamp.(inner_physical, 0.0, 1.0)
    outer_physical .= clamp.(outer_physical, 0.0, 1.0)
    
    # Transform to spectral space using SHTnsKit
    inner_spectral = shtns_physical_to_spectral(inner_physical, 𝔽.config)
    outer_spectral = shtns_physical_to_spectral(outer_physical, 𝔽.config)
    
    # Apply to boundary arrays
    𝔽.boundary_values[1, :] = inner_spectral  # Inner boundary
    𝔽.boundary_values[2, :] = outer_spectral  # Outer boundary

    # Set boundary condition types based on boundary specifications
    # Default to Dirichlet (fixed composition), but check for flux BCs
    inner_bc_type = infer_composition_bc_type(boundary_set.inner_boundary)
    outer_bc_type = infer_composition_bc_type(boundary_set.outer_boundary)

    fill!(𝔽.bc_type_inner, inner_bc_type)
    fill!(𝔽.bc_type_outer, outer_bc_type)

    # Update time index (in field or fallback cache)
    update_composition_time_index!(𝔽, time_index)

    return 𝔽
end

"""
    infer_composition_bc_type(boundary::BoundaryData)

Infer boundary condition type (Dirichlet or Neumann) from boundary metadata.

Checks the description field for keywords:
- "flux", "neumann" → NEUMANN (∂C/∂r specified)
- Otherwise → DIRICHLET (C specified)
"""
function infer_composition_bc_type(boundary::BoundaryData)
    desc = lowercase(boundary.description)

    # Check for flux/Neumann BC indicators
    if occursin("flux", desc) || occursin("neumann", desc) ||
       occursin("gradient", desc)
        return Int(NEUMANN)
    else
        # Default to Dirichlet (fixed composition)
        return Int(DIRICHLET)
    end
end

"""
    update_time_dependent_composition_boundaries!(𝔽, current_time::Float64)

Update time-dependent composition boundary conditions.
"""
function update_time_dependent_composition_boundaries!(𝔽, current_time::Float64)
    
    boundary_set, _ = get_composition_boundary_data(𝔽)
    if boundary_set === nothing
        return 𝔽
    end
    
    # Check if boundaries are time-dependent
    if !boundary_set.inner_boundary.is_time_dependent && !boundary_set.outer_boundary.is_time_dependent
        return 𝔽  # Nothing to update
    end
    
    # Find time index for current time  
    time_index = find_composition_boundary_time_index(boundary_set, current_time)
    
    # Only update if time index has changed
    current_time_index = get_composition_time_index(𝔽)
    if time_index != current_time_index
        apply_composition_boundary_conditions!(𝔽, time_index)
        
        if get_rank() == 0
            @info "Updated composition boundaries to time index $time_index (t=$current_time)"
        end
    end
    
    return 𝔽
end

"""
    get_current_composition_boundaries(𝔽)

Get current composition boundary conditions.
"""
function get_current_composition_boundaries(𝔽)
    
    boundary_set, cache = get_composition_boundary_data(𝔽)
    if boundary_set === nothing
        return Dict(:error => "No boundary conditions loaded")
    end
    
    time_index = get_composition_time_index(𝔽)
    
    # Get current boundary data
    inner_physical = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
    outer_physical = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
    
    # Get spectral coefficients
    inner_spectral = 𝔽.boundary_values[1, :]
    outer_spectral = 𝔽.boundary_values[2, :]
    
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
            "creation_time" => boundary_set.creation_time,
            "valid_range" => [0.0, 1.0]
        )
    )
end

"""
    set_programmatic_composition_boundaries!(𝔽, inner_spec::Tuple, outer_spec::Tuple)

Set programmatic composition boundary conditions.
"""
function set_programmatic_composition_boundaries!(𝔽, inner_spec::Tuple, outer_spec::Tuple)
    
    boundary_specs = Dict(:inner => inner_spec, :outer => outer_spec)
    return load_composition_boundary_conditions!(𝔽, boundary_specs)
end

"""
    validate_composition_boundary_files(boundary_specs::Dict, config)

Validate composition boundary condition files.
"""
function validate_composition_boundary_files(boundary_specs::Dict, config)
    
    inner_spec = get(boundary_specs, :inner, nothing)
    outer_spec = get(boundary_specs, :outer, nothing)
    
    errors = String[]
    
    # Validate file specifications
    if isa(inner_spec, String)
        try
            validate_netcdf_boundary_file(inner_spec, ["composition", "xi"])
            # Check composition range
            inner_data = read_netcdf_boundary_data(inner_spec, precision=config.T)
            min_val, max_val = extrema(inner_data.values)
            if min_val < 0.0 || max_val > 1.0
                push!(errors, "Inner composition values outside valid range [0,1]: [$min_val, $max_val]")
            end
        catch e
            push!(errors, "Inner boundary file error: $e")
        end
    end
    
    if isa(outer_spec, String)
        try
            validate_netcdf_boundary_file(outer_spec, ["composition", "xi"])
            # Check composition range
            outer_data = read_netcdf_boundary_data(outer_spec, precision=config.T)
            min_val, max_val = extrema(outer_data.values)
            if min_val < 0.0 || max_val > 1.0
                push!(errors, "Outer composition values outside valid range [0,1]: [$min_val, $max_val]")
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
            validate_boundary_compatibility(inner_data, outer_data, "composition")
        catch e
            push!(errors, "Boundary compatibility error: $e")
        end
    end
    
    if !isempty(errors)
        error_msg = "Composition boundary validation failed:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
    
    return true
end

"""
    create_layered_composition_boundary(config, layer_specs::Vector{Tuple{Real, Real, Real}})

Create layered composition boundary conditions.

# Arguments
- `layer_specs`: Vector of (colatitude_start, colatitude_end, composition) tuples

# Example
```julia
# Create three-layer composition structure
layer_specs = [
    (0.0, π/3, 0.8),     # High composition in top layer
    (π/3, 2π/3, 0.4),    # Medium composition in middle layer  
    (2π/3, π, 0.1)       # Low composition in bottom layer
]
```
"""
function create_layered_composition_boundary(config, layer_specs::Vector{Tuple{Real, Real, Real}})
    
    if isempty(layer_specs)
        throw(ArgumentError("At least one layer specification required"))
    end
    
    # Create coordinate grids
    theta = collect(range(0, π, length=config.nlat))
    phi = collect(range(0, 2π, length=config.nlon+1)[1:end-1])
    
    # Initialize composition array
    values = zeros(config.T, config.nlat, config.nlon)
    
    # Apply layered structure
    for (i, θ) in enumerate(theta)
        for (θ_start, θ_end, composition) in layer_specs
            if θ_start <= θ <= θ_end
                values[i, :] .= composition
                break
            end
        end
    end
    
    # Validate composition range
    values .= clamp.(values, 0.0, 1.0)
    
    # Create boundary data
    return create_boundary_data(
        values, "composition";
        theta=theta, phi=phi, time=nothing,
        units="dimensionless",
        description="Layered composition boundary ($(length(layer_specs)) layers)",
        file_path="programmatic"
    )
end

"""
    find_composition_boundary_time_index(boundary_set::BoundaryConditionSet, current_time::Float64)

Find the appropriate time index for the current simulation time.
"""
function find_composition_boundary_time_index(boundary_set::BoundaryConditionSet, current_time::Float64)
    
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

# Function moved to main bcs module to avoid duplication

"""
    get_composition_boundary_data(𝔽)

Get boundary data from field or fallback cache.
"""
function get_composition_boundary_data(𝔽)
    if hasfield(typeof(𝔽), :boundary_condition_set)
        boundary_set = 𝔽.boundary_condition_set
        cache = 𝔽.boundary_interpolation_cache
        return boundary_set, cache
    else
        # Use fallback cache
        if isdefined(@__MODULE__, :_composition_boundary_cache)
            field_id = objectid(𝔽)
            if haskey(_composition_boundary_cache, field_id)
                data = _composition_boundary_cache[field_id]
                return data[:boundary_set], data[:interpolation_cache]
            end
        end
        return nothing, nothing
    end
end

"""
    get_composition_time_index(𝔽)

Get current time index from field or fallback cache.
"""
function get_composition_time_index(𝔽)
    if hasfield(typeof(𝔽), :boundary_time_index)
        return 𝔽.boundary_time_index[]
    else
        # Use fallback cache
        if isdefined(@__MODULE__, :_composition_boundary_cache)
            field_id = objectid(𝔽)
            if haskey(_composition_boundary_cache, field_id)
                return _composition_boundary_cache[field_id][:time_index]
            end
        end
        return 1
    end
end

"""
    update_composition_time_index!(𝔽, time_index::Int)

Update time index in field or fallback cache.
"""
function update_composition_time_index!(𝔽, time_index::Int)
    if hasfield(typeof(𝔽), :boundary_time_index)
        𝔽.boundary_time_index[] = time_index
    else
        # Use fallback cache
        if isdefined(@__MODULE__, :_composition_boundary_cache)
            field_id = objectid(𝔽)
            if haskey(_composition_boundary_cache, field_id)
                _composition_boundary_cache[field_id][:time_index] = time_index
            end
        end
    end
end

"""
    apply_composition_boundaries!(𝔽, boundary_set::BoundaryConditionSet; time::Float64=0.0)

Apply composition boundary conditions from a BoundaryConditionSet to a composition field.
This is a convenience wrapper that loads the boundary set and applies it.
"""
function apply_composition_boundaries!(𝔽, boundary_set::BoundaryConditionSet; time::Float64=0.0)

    # Store boundary conditions in field (if supported)
    if hasfield(typeof(𝔽), :boundary_condition_set)
        𝔽.boundary_condition_set = boundary_set
        𝔽.boundary_time_index[] = 1
        𝔽.boundary_interpolation_cache = create_composition_interpolation_cache(boundary_set, 𝔽.config)
    else
        # Fallback for legacy field structure
        if !isdefined(@__MODULE__, :_composition_boundary_cache)
            global _composition_boundary_cache = Dict{UInt64, Any}()
        end
        field_id = objectid(𝔽)
        _composition_boundary_cache[field_id] = Dict(
            :boundary_set => boundary_set,
            :interpolation_cache => create_composition_interpolation_cache(boundary_set, 𝔽.config),
            :time_index => 1
        )
    end

    # Apply boundary conditions
    apply_composition_boundary_conditions!(𝔽)

    return 𝔽
end

"""
    enforce_composition_boundary_constraints!(𝔽, bc_spec::Dict)

Enforce specific composition boundary constraints based on user specification.

# Arguments
- `𝔽`: Composition field structure
- `bc_spec`: Dictionary specifying boundary condition types

# BC Specification Format
```julia
bc_spec = Dict(
    :inner => :dirichlet,  # or :neumann, :flux
    :outer => :dirichlet,  # or :neumann, :flux
    :inner_value => 1.0,   # for Dirichlet (0.0-1.0)
    :outer_value => 0.0,   # for Dirichlet (0.0-1.0)
    :inner_flux => 0.01,   # for Neumann (∂C/∂r)
    :outer_flux => 0.0     # for Neumann (∂C/∂r)
)
```

# Boundary Condition Types
- `:dirichlet` - Fixed composition: C = C₀ at boundary
- `:neumann` or `:flux` - Fixed compositional flux: ∂C/∂r = q at boundary

# Physical Interpretation
- **Dirichlet (Fixed C)**: Typical for prescribed composition boundaries
  * Inner (ICB): Fixed light element concentration at inner core
  * Outer (CMB): Fixed composition at core-mantle boundary

- **Neumann (Fixed flux)**: Typical for compositional flux boundaries
  * Inner: Compositional release/freezing (∂C/∂r = q)
  * Outer: Zero flux (no exchange, ∂C/∂r = 0)

# Examples
```julia
# Fixed composition at both boundaries
enforce_composition_boundary_constraints!(𝔽, Dict(
    :inner => :dirichlet, :inner_value => 1.0,
    :outer => :dirichlet, :outer_value => 0.0
))

# Compositional release from below, sealed top
enforce_composition_boundary_constraints!(𝔽, Dict(
    :inner => :flux, :inner_flux => 0.01,
    :outer => :flux, :outer_flux => 0.0
))
```
"""
function enforce_composition_boundary_constraints!(𝔽, bc_spec::Dict)

    # Process inner boundary
    inner_type = get(bc_spec, :inner, :dirichlet)
    if inner_type == :dirichlet
        # Fixed composition - use Dirichlet BC
        fill!(𝔽.bc_type_inner, Int(DIRICHLET))

        # Set boundary value if provided (must be in [0,1])
        if haskey(bc_spec, :inner_value)
            inner_val = clamp(bc_spec[:inner_value], 0.0, 1.0)
            # Set uniform value in l=0,m=0 mode
            𝔽.boundary_values[1, 1] = inner_val
            𝔽.boundary_values[1, 2:end] .= 0.0
        end

    elseif inner_type == :neumann || inner_type == :flux
        # Fixed compositional flux - use Neumann BC
        fill!(𝔽.bc_type_inner, Int(NEUMANN))

        # Set flux value if provided
        if haskey(bc_spec, :inner_flux)
            flux_val = bc_spec[:inner_flux]
            𝔽.boundary_values[1, 1] = flux_val
            𝔽.boundary_values[1, 2:end] .= 0.0
        end

    else
        throw(ArgumentError("Unknown inner boundary type: $inner_type. Use :dirichlet or :neumann/:flux"))
    end

    # Process outer boundary
    outer_type = get(bc_spec, :outer, :dirichlet)
    if outer_type == :dirichlet
        fill!(𝔽.bc_type_outer, Int(DIRICHLET))

        if haskey(bc_spec, :outer_value)
            outer_val = clamp(bc_spec[:outer_value], 0.0, 1.0)
            𝔽.boundary_values[2, 1] = outer_val
            𝔽.boundary_values[2, 2:end] .= 0.0
        end

    elseif outer_type == :neumann || outer_type == :flux
        fill!(𝔽.bc_type_outer, Int(NEUMANN))

        if haskey(bc_spec, :outer_flux)
            flux_val = bc_spec[:outer_flux]
            𝔽.boundary_values[2, 1] = flux_val
            𝔽.boundary_values[2, 2:end] .= 0.0
        end

    else
        throw(ArgumentError("Unknown outer boundary type: $outer_type. Use :dirichlet or :neumann/:flux"))
    end

    return 𝔽
end

export load_composition_boundary_conditions!, set_programmatic_composition_boundaries!
export update_time_dependent_composition_boundaries!, get_current_composition_boundaries
export validate_composition_boundary_files, create_layered_composition_boundary
export apply_composition_boundaries!, enforce_composition_boundary_constraints!
export infer_composition_bc_type