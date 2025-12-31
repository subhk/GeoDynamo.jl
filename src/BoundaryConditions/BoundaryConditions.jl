# ================================================================================
# BoundaryConditions Module - Unified Boundary Condition System
# ================================================================================
#
# This module provides a unified interface for handling boundary conditions
# for all field types in geodynamo simulations:
# - Temperature boundary conditions
# - Composition boundary conditions
# - Velocity boundary conditions
# - Magnetic field boundary conditions
#
# Features:
# - NetCDF file-based boundary conditions
# - Programmatic boundary generation
# - Time-dependent boundary conditions
# - MPI parallelization support
# - PencilArrays and PencilFFTs integration
# - Automatic grid interpolation
# - Comprehensive error handling
#
# ================================================================================
# BOUNDARY CONDITION TYPES SUMMARY
# ================================================================================
#
# +-------------+------------------+----------------------------------+
# | Field       | BC Type          | Physical Meaning                 |
# +-------------+------------------+----------------------------------+
# | Temperature | DIRICHLET        | Fixed temperature (T = T₀)       |
# |             | NEUMANN          | Fixed heat flux (∂T/∂r = q)      |
# +-------------+------------------+----------------------------------+
# | Composition | DIRICHLET        | Fixed composition (C = C₀)       |
# |             | NEUMANN          | Fixed mass flux (∂C/∂r = q)      |
# |             |                  | Default: no-flux (q = 0)         |
# +-------------+------------------+----------------------------------+
# | Velocity    | DIRICHLET        | No-slip (u = 0)                  |
# | (Poloidal)  |                  | or impermeable (v_r = 0)         |
# +-------------+------------------+----------------------------------+
# | Velocity    | DIRICHLET        | No-slip (T = 0)                  |
# | (Toroidal)  | NEUMANN          | Stress-free (∂T/∂r = T/r)        |
# |             |                  | NOTE: Not simple Neumann!        |
# +-------------+------------------+----------------------------------+
# | Magnetic    | DIRICHLET        | All types (insulating,           |
# |             |                  | perfect conductor, potential)    |
# +-------------+------------------+----------------------------------+
#
# ================================================================================
# FILE ORGANIZATION
# ================================================================================
#
# BoundaryConditions/
# ├── BoundaryConditions.jl   # Main module (this file)
# ├── common.jl               # Shared types and utilities
# ├── netcdf_io.jl            # NetCDF file reading/writing
# ├── interpolation.jl        # Grid interpolation
# ├── programmatic.jl         # Programmatic BC generation
# ├── thermal.jl              # Temperature BCs
# ├── composition.jl          # Composition BCs
# ├── velocity.jl             # Velocity BCs
# └── magnetic.jl             # Magnetic field BCs
#
# ================================================================================
# USAGE PATTERN
# ================================================================================
#
#   # 1. Load boundary conditions from file or programmatically
#   boundary_specs = Dict(
#       :inner => "cmb_temperature.nc",
#       :outer => (:uniform, 300.0)
#   )
#   load_temperature_boundary_conditions!(temp_field, boundary_specs)
#
#   # 2. BCs are automatically applied during field initialization
#   # 3. Update time-dependent BCs during simulation loop
#   update_time_dependent_temperature_boundaries!(temp_field, current_time)
#
# ================================================================================
# DEBUGGING TIPS
# ================================================================================
#
# 1. Check BC type assignment: `println(field.bc_type_inner)`
# 2. Verify boundary values: `println(field.boundary_values[1, 1:5])`
# 3. For MPI issues, ensure all processes load same boundary files
# 4. For interpolation issues, check grid compatibility with source data
#
# ================================================================================

module BoundaryConditions

# Import packages from parent GeoDynamo module scope
using MPI
using PencilArrays
using PencilFFTs
using SHTnsKit
using NCDatasets
using LinearAlgebra
using Base.Threads
using Dates

# For Julia 1.10 compatibility: Define simple statistics functions
# to avoid module resolution issues with Statistics stdlib in submodules
_mean(x) = sum(x) / length(x)
_std(x) = begin
    m = _mean(x)
    n = length(x)
    n <= 1 && return zero(eltype(x))
    sqrt(sum((xi - m)^2 for xi in x) / (n - 1))
end

# Create a module-like object for compatibility with existing code
module _Statistics
    import ..BoundaryConditions: _mean as mean, _std as std
end

# ================================================================================
# Core Boundary Condition Types and Interfaces
# ================================================================================

"""
    AbstractBoundaryCondition{T}

Abstract base type for all boundary conditions.
"""
abstract type AbstractBoundaryCondition{T} end

"""
    BoundaryLocation

Enumeration for boundary locations.
"""
@enum BoundaryLocation begin
    INNER_BOUNDARY = 1  # Inner core boundary (ICB)
    OUTER_BOUNDARY = 2  # Outer boundary (CMB or surface)
end

"""
    BoundaryType

Enumeration for boundary condition types.
"""
@enum BoundaryType begin
    DIRICHLET = 1      # Fixed value boundary condition
    NEUMANN = 2        # Fixed flux/gradient boundary condition
    MIXED = 3          # Mixed boundary condition
    ROBIN = 4          # Robin boundary condition (linear combination)
end

"""
    FieldType

Enumeration for different physical field types.
"""
@enum FieldType begin
    TEMPERATURE = 1
    COMPOSITION = 2
    VELOCITY = 3
    MAGNETIC = 4
end

# ================================================================================
# Export core types and enums
# ================================================================================

export AbstractBoundaryCondition
export BoundaryLocation, INNER_BOUNDARY, OUTER_BOUNDARY
export BoundaryType, DIRICHLET, NEUMANN, MIXED, ROBIN
export FieldType, TEMPERATURE, COMPOSITION, VELOCITY, MAGNETIC

# ================================================================================
# Include specialized boundary condition modules
# ================================================================================

include("common.jl")           # Common utilities and data structures
include("netcdf_io.jl")        # NetCDF file I/O functionality
include("interpolation.jl")    # Grid interpolation utilities
include("programmatic.jl")     # Programmatic boundary generation

# Field-specific boundary condition modules
include("thermal.jl")          # Temperature boundary conditions
include("composition.jl")      # Composition boundary conditions
include("velocity.jl")         # Velocity boundary conditions
include("magnetic.jl")         # Magnetic field boundary conditions

# Integration modules
include("integration.jl")      # Integration with field structures
include("timestepping.jl")     # Integration with timestepping

# ================================================================================
# Unified Interface Functions
# ================================================================================

"""
    load_boundary_conditions!(field, field_type::FieldType, boundary_specs::Dict)

Unified interface to load boundary conditions for any field type.

# Arguments
- `field`: Field structure to apply boundary conditions to
- `field_type`: Type of field (TEMPERATURE, COMPOSITION, VELOCITY, MAGNETIC)
- `boundary_specs`: Dictionary specifying boundary condition sources

# Examples
```julia
# Temperature boundaries
    load_boundary_conditions!(temp_field, TEMPERATURE, Dict(
        :inner => "cmb_temperature.nc",
        :outer => "surface_temperature.nc"
    ))

# Velocity boundaries (no-slip at both boundaries)
    load_boundary_conditions!(velocity_field, VELOCITY, Dict(
        :inner => (:no_slip, 0.0),
        :outer => (:no_slip, 0.0)
    ))

# Magnetic boundaries (potential field at outer boundary)
    load_boundary_conditions!(magnetic_field, MAGNETIC, Dict(
        :inner => (:insulating, 0.0),
        :outer => (:potential_field, "field_coefficients.nc")
    ))
```
"""

function load_boundary_conditions!(field, field_type::FieldType, boundary_specs::Dict)
    if field_type == TEMPERATURE
        return load_temperature_boundary_conditions!(field, boundary_specs)
    elseif field_type == COMPOSITION
        return load_composition_boundary_conditions!(field, boundary_specs)
    elseif field_type == VELOCITY
        return load_velocity_boundary_conditions!(field, boundary_specs)
    elseif field_type == MAGNETIC
        return load_magnetic_boundary_conditions!(field, boundary_specs)
    else
        throw(ArgumentError("Unknown field type: $field_type"))
    end
end

"""
    update_time_dependent_boundaries!(field, field_type::FieldType, current_time::Float64)

Update time-dependent boundary conditions for any field type.
"""
function update_time_dependent_boundaries!(field, field_type::FieldType, current_time::Float64)
    if field_type == TEMPERATURE
        return update_time_dependent_temperature_boundaries!(field, current_time)
    elseif field_type == COMPOSITION
        return update_time_dependent_composition_boundaries!(field, current_time)
    elseif field_type == VELOCITY
        return update_time_dependent_velocity_boundaries!(field, current_time)
    elseif field_type == MAGNETIC
        return update_time_dependent_magnetic_boundaries!(field, current_time)
    else
        return field  # No updates for unknown field types
    end
end

"""
    validate_boundary_files(field_type::FieldType, boundary_specs::Dict, config)

Validate boundary condition files for any field type.
"""
function validate_boundary_files(field_type::FieldType, boundary_specs::Dict, config)
    if field_type == TEMPERATURE
        return validate_temperature_boundary_files(boundary_specs, config)
    elseif field_type == COMPOSITION
        return validate_composition_boundary_files(boundary_specs, config)
    elseif field_type == VELOCITY
        return validate_velocity_boundary_files(boundary_specs, config)
    elseif field_type == MAGNETIC
        return validate_magnetic_boundary_files(boundary_specs, config)
    else
        throw(ArgumentError("Unknown field type: $field_type"))
    end
end

"""
    get_current_boundaries(field, field_type::FieldType)

Get current boundary values for any field type.
"""
function get_current_boundaries(field, field_type::FieldType)
    if field_type == TEMPERATURE
        return get_current_temperature_boundaries(field)
    elseif field_type == COMPOSITION
        return get_current_composition_boundaries(field)
    elseif field_type == VELOCITY
        return get_current_velocity_boundaries(field)
    elseif field_type == MAGNETIC
        return get_current_magnetic_boundaries(field)
    else
        return Dict(:error => "Unknown field type: $field_type")
    end
end

"""
    print_boundary_summary(field, field_type::FieldType)

Print a summary of loaded boundary conditions for any field type.
"""
function print_boundary_summary(field, field_type::FieldType)
    boundaries = get_current_boundaries(field, field_type)
    field_name = string(field_type)
    
    println("╔═══════════════════════════════════════════════════════════════╗")
    println("║                 $(uppercase(field_name)) BOUNDARY SUMMARY                    ║")
    println("╠═══════════════════════════════════════════════════════════════╣")
    
    if haskey(boundaries, :metadata)
        metadata = boundaries[:metadata]
        println("║ Source: $(get(metadata, "source", "unknown"))                              ║")
        
        if haskey(metadata, "inner_file")
            inner_file = basename(get(metadata, "inner_file", ""))
            outer_file = basename(get(metadata, "outer_file", ""))
            println("║ Inner file: $(inner_file)                           ║")
            println("║ Outer file: $(outer_file)                           ║")
        end
        
        if haskey(boundaries, :time_index)
            println("║ Time index: $(boundaries[:time_index])                                      ║")
        end
    end
    
    println("╚═══════════════════════════════════════════════════════════════╝")
end

# ================================================================================
# Export unified interface functions
# ================================================================================

export load_boundary_conditions!, update_time_dependent_boundaries!
export validate_boundary_files, get_current_boundaries, print_boundary_summary

# Re-export field-specific convenience functions from submodules
export apply_temperature_boundaries!, apply_composition_boundaries!

# ================================================================================
# Module-wide utilities
# ================================================================================

"""
    get_boundary_module_info()

Get information about the boundary conditions module.
"""
# ================================================================================
# Common Utility Functions
# ================================================================================

# ================================================================================
# Cached SHTnsKit Configuration for Boundary Transforms (v1.1.15 optimization)
# ================================================================================
# This avoids recreating configs for each boundary transform call

const _BC_SHTNS_CONFIG_CACHE = Dict{Tuple{Int,Int,Int,Int}, Any}()
const _BC_SHTNS_CONFIG_LOCK = ReentrantLock()

"""
    _get_cached_bc_shtns_config(lmax, mmax, nlat, nlon)

Get or create a cached SHTnsKit configuration for boundary transforms.
Reuses configurations to avoid repeated setup/teardown overhead.
Thread-safe implementation using a lock for the check-then-set pattern.
"""
function _get_cached_bc_shtns_config(lmax::Int, mmax::Int, nlat::Int, nlon::Int)
    key = (lmax, mmax, nlat, nlon)

    # Fast path: check if already cached (no lock needed for read)
    if haskey(_BC_SHTNS_CONFIG_CACHE, key)
        return _BC_SHTNS_CONFIG_CACHE[key]
    end

    # Slow path: need to create config (use lock to prevent race conditions)
    lock(_BC_SHTNS_CONFIG_LOCK) do
        # Double-check after acquiring lock (another thread might have created it)
        if !haskey(_BC_SHTNS_CONFIG_CACHE, key)
            _BC_SHTNS_CONFIG_CACHE[key] = SHTnsKit.create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon)
        end
        return _BC_SHTNS_CONFIG_CACHE[key]
    end
end

"""
    clear_bc_shtns_config_cache!()

Clear the cached SHTnsKit configurations for boundary transforms.
Call this when grid parameters change or to free memory.
Thread-safe implementation.
"""
function clear_bc_shtns_config_cache!()
    lock(_BC_SHTNS_CONFIG_LOCK) do
        for (key, cfg) in _BC_SHTNS_CONFIG_CACHE
            try
                SHTnsKit.destroy_config(cfg)
            catch
                # Ignore errors during cleanup
            end
        end
        empty!(_BC_SHTNS_CONFIG_CACHE)
    end
end

"""
    shtns_physical_to_spectral(physical_data::Matrix{T}, config; return_complex::Bool=false) where T

Transform physical boundary data to spectral coefficients using SHTnsKit.
This is a common utility function used by both thermal and composition modules.

# Arguments
- `physical_data`: 2D array of physical values on (nlat, nlon) grid
- `config`: SHTnsKit configuration with lmax, mmax, nlm fields
- `return_complex`: If true, returns complex coefficients; otherwise real part only

# Returns
Vector of spectral coefficients of length nlm.

# Performance (v1.1.15)
Uses cached SHTnsKit configurations to avoid repeated setup overhead.
"""
function shtns_physical_to_spectral(physical_data::Matrix{T}, config; return_complex::Bool=false) where T

    try
        # Use cached configuration for efficiency (v1.1.15 optimization)
        nlat, nlon = size(physical_data)
        shtconfig = _get_cached_bc_shtns_config(config.lmax, config.mmax, nlat, nlon)

        # Perform forward transform - returns (lmax+1)×(mmax+1) matrix
        coeffs_matrix = SHTnsKit.analysis(shtconfig, physical_data)

        # Convert matrix format to 1D spectral coefficient array
        # The boundary code expects a 1D array of length nlm
        lmax, mmax_val = config.lmax, config.mmax
        nlm = config.nlm

        if return_complex
            coeffs = zeros(Complex{T}, nlm)
        else
            coeffs = zeros(T, nlm)
        end

        # Map from (l,m) matrix to linear index
        # This follows the same indexing as used in the main transform code
        idx = 0
        for l in 0:lmax
            for m in 0:min(l, mmax_val)
                idx += 1
                if idx <= nlm && l < size(coeffs_matrix, 1) && m < size(coeffs_matrix, 2)
                    if return_complex
                        coeffs[idx] = coeffs_matrix[l+1, m+1]
                    else
                        # Extract real part (boundary conditions typically use real values)
                        coeffs[idx] = real(coeffs_matrix[l+1, m+1])
                    end
                end
            end
        end

        return coeffs
    catch e
        @warn "SHTnsKit analysis failed, using fallback: $e"

        # Fallback: simple mean value in l=0 mode
        nlm = config.nlm
        if return_complex
            coeffs = zeros(Complex{T}, nlm)
        else
            coeffs = zeros(T, nlm)
        end

        # Set l=0, m=0 mode to mean value
        if length(coeffs) > 0
            coeffs[1] = _Statistics.mean(physical_data)
        end

        return coeffs
    end
end

"""
    shtns_spectral_to_physical(coeffs::Vector, config, nlat::Int, nlon::Int)

Transform spectral coefficients to physical boundary data using SHTnsKit.
Inverse of shtns_physical_to_spectral.

# Arguments
- `coeffs`: Vector of spectral coefficients of length nlm
- `config`: SHTnsKit configuration with lmax, mmax fields
- `nlat, nlon`: Output grid dimensions

# Returns
Matrix of physical values on (nlat, nlon) grid.
"""
function shtns_spectral_to_physical(coeffs::Vector{T}, config, nlat::Int, nlon::Int) where T
    try
        # Use cached configuration
        shtconfig = _get_cached_bc_shtns_config(config.lmax, config.mmax, nlat, nlon)

        # Convert 1D coefficients to (lmax+1)×(mmax+1) matrix
        lmax, mmax_val = config.lmax, config.mmax
        coeffs_matrix = zeros(ComplexF64, lmax+1, mmax_val+1)

        idx = 0
        for l in 0:lmax
            for m in 0:min(l, mmax_val)
                idx += 1
                if idx <= length(coeffs)
                    coeffs_matrix[l+1, m+1] = complex(coeffs[idx])
                end
            end
        end

        # Perform inverse transform
        physical_data = SHTnsKit.synthesis(shtconfig, coeffs_matrix; real_output=true)
        return physical_data
    catch e
        @warn "SHTnsKit synthesis failed, using fallback: $e"
        # Fallback: uniform field with l=0 value
        physical_data = zeros(Float64, nlat, nlon)
        if length(coeffs) > 0
            fill!(physical_data, real(coeffs[1]))
        end
        return physical_data
    end
end

function get_boundary_module_info()
    return Dict(
        "module_name" => "BoundaryConditions",
        "version" => "1.0.0",
        "supported_fields" => ["temperature", "composition", "velocity", "magnetic"],
        "supported_formats" => ["netcdf", "programmatic", "hybrid"],
        "features" => [
            "MPI parallelization",
            "PencilArrays integration",
            "PencilFFTs support",
            "Time-dependent boundaries",
            "Grid interpolation",
            "Comprehensive validation"
        ]
    )
end

export get_boundary_module_info

end # module BoundaryConditions