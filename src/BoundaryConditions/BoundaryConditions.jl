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

module BoundaryConditions

using MPI
using PencilArrays
using PencilFFTs
using SHTnsKit
using NCDatasets
using LinearAlgebra
using Statistics
using Base.Threads

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

"""
    shtns_physical_to_spectral(physical_data::Matrix{T}, config) where T

Transform physical boundary data to spectral coefficients using SHTnsKit.
This is a common utility function used by both thermal and composition modules.
"""
function shtns_physical_to_spectral(physical_data::Matrix{T}, config) where T

    try
        # Create temporary SHTnsKit configuration
        nlat, nlon = size(physical_data)
        shtconfig = SHTnsKit.create_gauss_config(config.lmax, nlat; mmax=config.mmax, nlon=nlon)

        # Perform forward transform - returns (lmax+1)×(mmax+1) matrix
        coeffs_matrix = SHTnsKit.analysis(shtconfig, physical_data)

        # Clean up configuration
        SHTnsKit.destroy_config(shtconfig)

        # Convert matrix format to 1D spectral coefficient array
        # The boundary code expects a 1D array of length nlm
        lmax, mmax = config.lmax, config.mmax
        nlm = config.nlm
        coeffs = zeros(T, nlm)

        # Map from (l,m) matrix to linear index
        # This follows the same indexing as used in the main transform code
        idx = 0
        for l in 0:lmax
            for m in 0:min(l, mmax)
                idx += 1
                if idx <= nlm && l < size(coeffs_matrix, 1) && m < size(coeffs_matrix, 2)
                    # Extract real part (boundary conditions typically use real values)
                    coeffs[idx] = real(coeffs_matrix[l+1, m+1])
                end
            end
        end

        return coeffs
    catch e
        @warn "SHTnsKit analysis failed, using fallback: $e"

        # Fallback: simple mean value in l=0 mode
        nlm = config.nlm
        coeffs = zeros(T, nlm)

        # Set l=0, m=0 mode to mean value
        if length(coeffs) > 0
            coeffs[1] = mean(physical_data)
        end

        return coeffs
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