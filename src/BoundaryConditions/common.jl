# ================================================================================
# Common Boundary Condition Utilities and Data Structures
# ================================================================================

"""
    BoundaryData{T}

Common data structure for boundary condition data from any source.
"""
struct BoundaryData{T<:AbstractFloat}
    # Spatial coordinates (if provided)
    theta::Union{Vector{T}, Nothing}      # Colatitude coordinates [rad]
    phi::Union{Vector{T}, Nothing}        # Longitude coordinates [rad]
    
    # Time coordinate (if time-dependent)
    time::Union{Vector{T}, Nothing}       # Time values
    
    # Boundary values (can be scalar, vector, or tensor depending on field)
    values::Array{T}                      # Boundary values
    
    # Metadata
    units::String                         # Physical units
    description::String                   # Field description
    file_path::String                     # Source file path
    field_type::String                    # Type of field (temperature, velocity, etc.)
    
    # Validation info
    is_time_dependent::Bool               # Whether data varies in time
    nlat::Int                            # Number of latitude points
    nlon::Int                            # Number of longitude points
    ntime::Int                           # Number of time points (1 if time-independent)
    ncomponents::Int                     # Number of field components (1 for scalar, 3 for vector, etc.)
end

"""
    BoundaryConditionSet{T}

Complete set of boundary conditions for inner and outer boundaries.
"""
struct BoundaryConditionSet{T<:AbstractFloat}
    inner_boundary::BoundaryData{T}       # Inner boundary data
    outer_boundary::BoundaryData{T}       # Outer boundary data
    field_name::String                    # Field name (temperature, velocity, etc.)
    field_type::FieldType                 # Field type enum
    creation_time::Float64               # When the boundary set was created
end

"""
    BoundaryCache{T}

Cache structure for storing processed boundary data.
"""
struct BoundaryCache{T}
    interpolated_data::Dict{String, Array{T}}     # Cached interpolated data by time index
    spectral_coefficients::Dict{String, Array{T}} # Cached spectral coefficients
    last_update_time::Float64                     # Last cache update time
    cache_keys::Vector{String}                    # Available cache keys
end

# Constructor for empty cache
function BoundaryCache{T}() where T
    return BoundaryCache{T}(
        Dict{String, Array{T}}(),
        Dict{String, Array{T}}(),
        0.0,
        String[]
    )
end

"""
    create_boundary_data(values::Array{T}, field_type::String; 
                        theta=nothing, phi=nothing, time=nothing,
                        units="", description="", file_path="programmatic") where T

Create a BoundaryData structure from raw data.
"""
function create_boundary_data(values::Array{T}, field_type::String; 
                            theta=nothing, phi=nothing, time=nothing,
                            units="", description="", file_path="programmatic") where T
    
    dims = size(values)
    is_time_dependent = length(dims) >= 3
    ncomponents = length(dims) >= 4 ? dims[4] : (length(dims) >= 3 && !is_time_dependent ? dims[3] : 1)
    
    if length(dims) == 2
        nlat, nlon = dims
        ntime = 1
    elseif length(dims) == 3
        if is_time_dependent
            nlat, nlon, ntime = dims
        else
            nlat, nlon = dims[1:2]
            ntime = 1
            ncomponents = dims[3]
        end
    elseif length(dims) == 4
        nlat, nlon, ntime, ncomponents = dims
        is_time_dependent = ntime > 1
    else
        throw(ArgumentError("Unsupported array dimensions: $dims"))
    end
    
    return BoundaryData{T}(
        theta, phi, time, values, units, description, file_path, field_type,
        is_time_dependent, nlat, nlon, ntime, ncomponents
    )
end

"""
    validate_boundary_compatibility(inner::BoundaryData, outer::BoundaryData, field_name::String)

Validate that inner and outer boundary data are compatible.
"""
function validate_boundary_compatibility(inner::BoundaryData, outer::BoundaryData, field_name::String)
    errors = String[]
    
    # Check spatial resolution
    if inner.nlat != outer.nlat
        push!(errors, "Mismatched nlat: inner=$(inner.nlat), outer=$(outer.nlat)")
    end
    
    if inner.nlon != outer.nlon
        push!(errors, "Mismatched nlon: inner=$(inner.nlon), outer=$(outer.nlon)")
    end
    
    # Check field components
    if inner.ncomponents != outer.ncomponents
        push!(errors, "Mismatched components: inner=$(inner.ncomponents), outer=$(outer.ncomponents)")
    end
    
    # Check time dependency
    if inner.is_time_dependent != outer.is_time_dependent
        push!(errors, "Time dependency mismatch: inner=$(inner.is_time_dependent), outer=$(outer.is_time_dependent)")
    end
    
    if inner.is_time_dependent && inner.ntime != outer.ntime
        push!(errors, "Mismatched ntime: inner=$(inner.ntime), outer=$(outer.ntime)")
    end
    
    # Check coordinate consistency (if both have coordinates)
    if inner.theta !== nothing && outer.theta !== nothing
        if !isapprox(inner.theta, outer.theta, rtol=1e-10)
            push!(errors, "Theta coordinates don't match between files")
        end
    end
    
    if inner.phi !== nothing && outer.phi !== nothing
        if !isapprox(inner.phi, outer.phi, rtol=1e-10)
            push!(errors, "Phi coordinates don't match between files")
        end
    end
    
    if !isempty(errors)
        error_msg = "$field_name boundary compatibility validation failed:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
    
    if get_rank() == 0
        @info "$field_name boundary files are compatible ($(inner.nlat)×$(inner.nlon)×$(inner.ncomponents), time_dep=$(inner.is_time_dependent))"
    end
end

"""
    get_boundary_statistics(boundary_data::BoundaryData)

Get statistical information about boundary data.
"""
function get_boundary_statistics(boundary_data::BoundaryData)
    values = boundary_data.values
    
    stats = Dict{String, Any}(
        "min" => minimum(values),
        "max" => maximum(values), 
        "mean" => mean(values),
        "std" => std(values),
        "shape" => size(values),
        "units" => boundary_data.units,
        "description" => boundary_data.description,
        "field_type" => boundary_data.field_type,
        "time_dependent" => boundary_data.is_time_dependent,
        "ntime" => boundary_data.ntime,
        "ncomponents" => boundary_data.ncomponents,
        "file_path" => boundary_data.file_path
    )
    
    return stats
end

"""
    print_boundary_data_info(boundary_data::BoundaryData, prefix::String="")

Print detailed information about boundary data.
"""
function print_boundary_data_info(boundary_data::BoundaryData, prefix::String="")
    println("$(prefix)Field Type: $(boundary_data.field_type)")
    println("$(prefix)Grid: $(boundary_data.nlat) × $(boundary_data.nlon)")
    println("$(prefix)Components: $(boundary_data.ncomponents)")
    println("$(prefix)Time steps: $(boundary_data.ntime)")
    println("$(prefix)Units: $(boundary_data.units)")
    
    values = boundary_data.values
    println("$(prefix)Range: [$(round(minimum(values), digits=3)), $(round(maximum(values), digits=3))]")
    println("$(prefix)Mean: $(round(mean(values), digits=3))")
    
    # Extract filename from path
    filename = basename(boundary_data.file_path)
    println("$(prefix)Source: $(filename)")
end

"""
    print_boundary_info(boundary_set::BoundaryConditionSet)

Print comprehensive information about a boundary condition set.
"""
function print_boundary_info(boundary_set::BoundaryConditionSet)
    field_name = uppercase(boundary_set.field_name)
    
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║                  $field_name BOUNDARY CONDITIONS                  ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    
    println("║ INNER BOUNDARY:                                              ║")
    print_boundary_data_info(boundary_set.inner_boundary, "║ ")
    
    println("║                                                              ║")
    println("║ OUTER BOUNDARY:                                              ║") 
    print_boundary_data_info(boundary_set.outer_boundary, "║ ")
    
    println("╚══════════════════════════════════════════════════════════════╝")
end

"""
    determine_field_type_from_name(field_name::String) -> FieldType

Determine field type from field name string.
"""
function determine_field_type_from_name(field_name::String)
    field_lower = lowercase(field_name)
    
    if occursin("temp", field_lower) || occursin("thermal", field_lower)
        return TEMPERATURE
    elseif occursin("comp", field_lower) || occursin("concentration", field_lower) || occursin("xi", field_lower)
        return COMPOSITION
    elseif occursin("veloc", field_lower) || occursin("flow", field_lower) || field_lower == "u"
        return VELOCITY
    elseif occursin("magn", field_lower) || occursin("magnetic", field_lower) || field_lower == "b"
        return MAGNETIC
    else
        throw(ArgumentError("Cannot determine field type from name: $field_name"))
    end
end

"""
    get_default_units(field_type::FieldType) -> String

Get default units for a field type.
"""
function get_default_units(field_type::FieldType)
    if field_type == TEMPERATURE
        return "K"
    elseif field_type == COMPOSITION
        return "dimensionless"
    elseif field_type == VELOCITY
        return "m/s"
    elseif field_type == MAGNETIC
        return "T"
    else
        return "unknown"
    end
end

"""
    get_default_boundary_type(field_type::FieldType, location::BoundaryLocation) -> BoundaryType

Get default boundary condition type for a field at a specific location.
"""
function get_default_boundary_type(field_type::FieldType, location::BoundaryLocation)
    if field_type == TEMPERATURE
        return DIRICHLET  # Fixed temperature
    elseif field_type == COMPOSITION
        return DIRICHLET  # Fixed composition
    elseif field_type == VELOCITY
        return DIRICHLET  # No-slip (zero velocity)
    elseif field_type == MAGNETIC
        if location == INNER_BOUNDARY
            return NEUMANN    # Insulating inner boundary
        else
            return DIRICHLET  # Potential field at outer boundary
        end
    else
        return DIRICHLET
    end
end

"""
    cache_boundary_data!(cache::BoundaryCache{T}, key::String, data::Array{T}) where T

Cache processed boundary data.
"""
function cache_boundary_data!(cache::BoundaryCache{T}, key::String, data::Array{T}) where T
    cache.interpolated_data[key] = data
    if !(key in cache.cache_keys)
        push!(cache.cache_keys, key)
    end
    return cache
end

"""
    get_cached_data(cache::BoundaryCache{T}, key::String) where T

Retrieve cached boundary data.
"""
function get_cached_data(cache::BoundaryCache{T}, key::String) where T
    return get(cache.interpolated_data, key, nothing)
end

"""
    clear_boundary_cache!(cache::BoundaryCache)

Clear all cached boundary data.
"""
function clear_boundary_cache!(cache::BoundaryCache)
    empty!(cache.interpolated_data)
    empty!(cache.spectral_coefficients)
    empty!(cache.cache_keys)
    cache.last_update_time = 0.0
    return cache
end

# ================================================================================
# MPI and parallel utilities
# ================================================================================

"""
    get_comm()

Get MPI communicator (with fallback).
"""
function get_comm()
    if MPI.Initialized()
        return MPI.COMM_WORLD
    else
        @warn "MPI not initialized, using serial mode"
        return nothing
    end
end

"""
    get_rank()

Get MPI rank (with fallback).
"""
function get_rank()
    if MPI.Initialized()
        return MPI.Comm_rank(MPI.COMM_WORLD)
    else
        return 0
    end
end

"""
    get_nprocs()

Get number of MPI processes (with fallback).
"""
function get_nprocs()
    if MPI.Initialized()
        return MPI.Comm_size(MPI.COMM_WORLD)
    else
        return 1
    end
end

# ================================================================================
# Export common utilities
# ================================================================================

export BoundaryData, BoundaryConditionSet, BoundaryCache
export create_boundary_data, validate_boundary_compatibility
export get_boundary_statistics, print_boundary_data_info, print_boundary_info
export determine_field_type_from_name, get_default_units, get_default_boundary_type
export cache_boundary_data!, get_cached_data, clear_boundary_cache!
export get_comm, get_rank, get_nprocs