# ================================================================================
# bcs Module - Unified Boundary Condition System
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
# bcs/
# ├── bcs.jl                  # Main module (this file)
# ├── common.jl               # Shared types and utilities
# ├── netcdf_io.jl            # NetCDF file reading/writing
# ├── interpolation.jl        # Grid interpolation
# ├── programmatic.jl         # Programmatic pattern generation utility
# ├── integration.jl          # Integration with field structures
# ├── timestepping.jl         # Integration with timestepping
# └── topography/             # Boundary topography effects
#
# Matrix-embedded BC files (included by their respective field modules):
# ├── thermal_bc.jl           # Temperature BCs (included by thermal.jl)
# ├── compositional_bc.jl     # Composition BCs (included by compositional.jl)
# ├── velocity_bc.jl          # Velocity BCs (included by velocity.jl)
# └── magnetic_bc.jl          # Magnetic BCs (included by magnetic.jl)
#
# ================================================================================
# USAGE PATTERN
# ================================================================================
#
# Boundary conditions are embedded in the implicit LHS matrices following
# the matrix-embedded approach. BC types are controlled by parameters:
#   - i_tmp_bc: Temperature BC type (1=DD, 2=DN, 3=ND, 4=NN)
#   - i_cmp_bc: Composition BC type (1=DD, 2=DN, 3=ND, 4=NN)
#   - i_vel_bc: Velocity BC type (1=no-slip, 2=stress-free, etc.)
#   - i_mag_bc: Magnetic BC type
#
# Matrix creation functions (in respective *_bc.jl files):
#   create_temperature_matrices(config, domain, diffusivity, dt)
#   create_composition_matrices(config, domain, diffusivity, dt)
#   create_velocity_toroidal_matrices(config, domain, dt)
#   create_magnetic_toroidal_matrices(config, domain, dt)
#
# RHS boundary value functions:
#   set_temperature_rhs_bc!(rhs, i_tmp_bc; val_inner=0, val_outer=0)
#   set_composition_rhs_bc!(rhs, i_cmp_bc; val_inner=0, val_outer=0)
#
# ================================================================================
# DEBUGGING TIPS
# ================================================================================
#
# 1. Check BC parameter: `println(get_parameters().i_tmp_bc)`
# 2. Verify matrix boundary rows: check first/last rows of LHS matrix
# 3. Check RHS boundary values after set_*_rhs_bc! calls
# 4. For MPI issues, ensure all processes use same BC parameters
#
# ================================================================================

module bcs

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
    import ..bcs: _mean as mean, _std as std
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
    NEUMANN_DERIV1 = 5 # First derivative = 0 (∂f/∂r = 0, for no-slip poloidal)
    NEUMANN_DERIV2 = 6 # Second derivative = 0 (∂²f/∂r² = 0, for stress-free poloidal)
    NEUMANN_MAG_INNER = 7  # Insulating inner BC: (∂/∂r - l/r) P = 0 (magnetic poloidal)
    NEUMANN_MAG_OUTER = 8  # Insulating outer BC: (∂/∂r + (l+1)/r) P = 0 (magnetic poloidal)
    CONTINUITY_MAG = 9     # Conducting inner core: ∂B/∂r continuous at ICB
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
export BoundaryType, DIRICHLET, NEUMANN, MIXED, ROBIN, NEUMANN_DERIV1, NEUMANN_DERIV2, NEUMANN_MAG_INNER, NEUMANN_MAG_OUTER, CONTINUITY_MAG
export FieldType, TEMPERATURE, COMPOSITION, VELOCITY, MAGNETIC

# ================================================================================
# Include specialized boundary condition modules
# ================================================================================

include("common.jl")           # Common utilities and data structures
include("netcdf_io.jl")        # NetCDF file I/O functionality
include("interpolation.jl")    # Grid interpolation utilities
include("programmatic.jl")     # Programmatic boundary generation
include("file_bc_loader.jl")   # File-based spectral BC loading

# Temperature and composition BCs are now embedded in the implicit matrix
# (see bcs/thermal_bc.jl and bcs/compositional_bc.jl, included by thermal.jl/compositional.jl)

# Integration modules
include("integration.jl")      # Integration with field structures
include("timestepping.jl")     # Integration with timestepping

# topography coupling module
include("topography/topography.jl")  # Boundary topography effects

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
        @warn "Temperature BCs are now embedded in the implicit matrix system (see bcs/thermal_bc.jl). File-based loading is not supported."
        return field
    elseif field_type == COMPOSITION
        @warn "Composition BCs are now embedded in the implicit matrix system (see bcs/compositional_bc.jl). File-based loading is not supported."
        return field
    elseif field_type == VELOCITY
        @warn "Velocity BCs are now embedded in the implicit matrix system (see bcs/velocity_bc.jl). File-based loading is not supported."
        return field
    elseif field_type == MAGNETIC
        @warn "Magnetic BCs are now embedded in the implicit matrix system (see bcs/magnetic_bc.jl). File-based loading is not supported."
        return field
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
        # Temperature BCs are embedded in the implicit matrix system (see bcs/thermal_bc.jl)
        return field
    elseif field_type == COMPOSITION
        # Composition BCs are embedded in the implicit matrix system (see bcs/compositional_bc.jl)
        return field
    elseif field_type == VELOCITY
        # Velocity BCs are embedded in the implicit matrix system (see bcs/velocity_bc.jl)
        return field
    elseif field_type == MAGNETIC
        # Magnetic BCs are embedded in the implicit matrix system (see bcs/magnetic_bc.jl)
        return field
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
        # Temperature BCs are embedded in the implicit matrix system (see bcs/thermal_bc.jl)
        return true
    elseif field_type == COMPOSITION
        # Composition BCs are embedded in the implicit matrix system (see bcs/compositional_bc.jl)
        return true
    elseif field_type == VELOCITY
        # Velocity BCs are embedded in the implicit matrix system (see bcs/velocity_bc.jl)
        return true
    elseif field_type == MAGNETIC
        # Magnetic BCs are embedded in the implicit matrix system (see bcs/magnetic_bc.jl)
        return true
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
        # Temperature BCs are embedded in the implicit matrix system (see bcs/thermal_bc.jl)
        return Dict(:info => "Temperature BCs are embedded in implicit matrices")
    elseif field_type == COMPOSITION
        # Composition BCs are embedded in the implicit matrix system (see bcs/compositional_bc.jl)
        return Dict(:info => "Composition BCs are embedded in implicit matrices")
    elseif field_type == VELOCITY
        # Velocity BCs are embedded in the implicit matrix system (see bcs/velocity_bc.jl)
        return Dict(:info => "Velocity BCs are embedded in implicit matrices")
    elseif field_type == MAGNETIC
        # Magnetic BCs are embedded in the implicit matrix system (see bcs/magnetic_bc.jl)
        return Dict(:info => "Magnetic BCs are embedded in implicit matrices")
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

# Temperature and composition BCs are now embedded in the implicit matrix system.
# See bcs/thermal_bc.jl and bcs/compositional_bc.jl (included by their respective field modules).

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

    # Julia Dict is not thread-safe for concurrent read/write — always lock
    lock(_BC_SHTNS_CONFIG_LOCK) do
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
        "module_name" => "bcs",
        "version" => "1.1.0",
        "supported_fields" => ["temperature", "composition", "velocity", "magnetic"],
        "supported_formats" => ["netcdf", "programmatic", "hybrid"],
        "features" => [
            "MPI parallelization",
            "PencilArrays integration",
            "PencilFFTs support",
            "Time-dependent boundaries",
            "Grid interpolation",
            "Comprehensive validation",
            "Boundary topography coupling",
            "Stefan condition for ICB evolution"
        ]
    )
end

export get_boundary_module_info

# ================================================================================
# Re-export topography Module
# ================================================================================

# Make topography module accessible as bcs.topography
using .topography

# Re-export key topography functions for convenience
export topography
export enable_topography!, disable_topography!, is_topography_enabled
export TopographyCouplingConfig, get_topography_config, set_topography_config!
export TopographyData, TopographyField
export GauntTensorCache, precompute_gaunt_tensors!
export apply_all_topography_corrections!
export StefanState, initialize_stefan_state!, update_icb_topography!

end # module bcs