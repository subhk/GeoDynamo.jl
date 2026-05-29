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
# ├── scalar_bc.jl            # Shared scalar matrix/RHS helpers (included by GeoDynamo.jl)
# ├── thermal_bc.jl           # Temperature BCs (included by physics/temperature/field.jl)
# ├── compositional_bc.jl     # Composition BCs (included by physics/composition/field.jl)
# ├── velocity_bc.jl          # Velocity BCs (included by physics/velocity/field.jl)
# └── magnetic_bc.jl          # Magnetic BCs (included by physics/magnetic/field.jl)
#
# ================================================================================
# USAGE PATTERN
# ================================================================================
#
# Boundary conditions are embedded in the implicit LHS matrices following
# the matrix-embedded approach. BC types are controlled by parameters:
#   - temperature_bcs: BoundaryConditions object for temperature
#   - composition_bcs: BoundaryConditions object for composition
#   - velocity_bcs: BoundaryConditions object for velocity
#   - magnetic_boundary: Magnetic BC type
#
# Matrix creation functions (in respective *_bc.jl files):
#   create_temperature_matrices(config, domain, diffusivity, dt; temperature_bc_code)
#   create_composition_matrices(config, domain, diffusivity, dt; composition_bc_code)
#   create_velocity_toroidal_matrices(config, domain, dt)
#   create_magnetic_toroidal_matrices(config, domain, dt)
#
# RHS boundary value functions:
#   set_temperature_rhs_bc!(rhs, temperature_bc_code; val_inner=0, val_outer=0)
#   set_composition_rhs_bc!(rhs, composition_bc_code; val_inner=0, val_outer=0)
#
# ================================================================================
# DEBUGGING TIPS
# ================================================================================
#
# 1. Check BC parameter: `println(state.parameters.temperature_bcs)`
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
export BoundaryType, DIRICHLET, NEUMANN, MIXED, ROBIN, NEUMANN_DERIV1, NEUMANN_DERIV2,
       NEUMANN_MAG_INNER, NEUMANN_MAG_OUTER, CONTINUITY_MAG
export FieldType, TEMPERATURE, COMPOSITION, VELOCITY, MAGNETIC

# ================================================================================
# Include specialized boundary condition modules
# ================================================================================

include("common.jl")           # Common utilities and data structures
include("netcdf_io.jl")        # NetCDF file I/O functionality
include("interpolation.jl")    # Grid interpolation utilities
include("programmatic.jl")     # Programmatic boundary generation
include("file_bc_loader.jl")   # File-based spectral BC loading

# Temperature and composition wrappers share the scalar matrix/RHS implementation
# in bcs/scalar_bc.jl.

# Integration modules
include("integration.jl")      # Integration with field structures
include("timestepping.jl")     # Integration with timestepping

# Forward declaration needed by topography submodule (defined fully below)
function shtns_spectral_to_physical end

# topography coupling module
include("topography/topography.jl")  # Boundary topography effects

# ================================================================================
# Unified Interface Functions
# ================================================================================

@doc raw"""
    load_boundary_conditions!(field, field_type::FieldType, boundary_specs::Dict)

Unified interface to load boundary conditions for any field type.

# Arguments
- `field`: Field structure to apply boundary conditions to
- `field_type`: Type of field (TEMPERATURE, COMPOSITION, VELOCITY, MAGNETIC)
- `boundary_specs`: Dictionary specifying boundary condition sources

# Current Behavior
- For `TEMPERATURE` and `COMPOSITION`, this high-level helper accepts
  programmatic tuple specs like `(:uniform, value)`, `(:dirichlet, value)`,
  and `(:neumann, flux)` and immediately embeds them into the field.
- File-based scalar loading is validated but not applied here because the
  matrix-embedded BC workflow needs an explicit field-side setup path.
- `VELOCITY` and `MAGNETIC` BCs remain matrix-embedded and are not configured
  through this helper.

# Examples
```julia
# Temperature boundaries
load_boundary_conditions!(temp_field, TEMPERATURE, Dict(
    :inner => (:uniform, 1.0),
    :outer => (:dirichlet, 0.0),
))

# Composition boundaries
load_boundary_conditions!(comp_field, COMPOSITION, Dict(
    :inner => (:neumann, 0.0),
    :outer => (:dirichlet, 0.0),
))
```
"""
function load_boundary_conditions!(field, field_type::FieldType, boundary_specs::Dict)
    config = hasfield(typeof(field), :config) ? field.config : nothing
    validate_boundary_files(field_type, boundary_specs, config)

    if field_type == TEMPERATURE || field_type == COMPOSITION
        inner_spec = get(boundary_specs, :inner, nothing)
        outer_spec = get(boundary_specs, :outer, nothing)

        if inner_spec isa Tuple && outer_spec isa Tuple
            if config === nothing
                throw(ArgumentError("Scalar boundary loading requires a field with a config"))
            end

            if field_type == TEMPERATURE
                pbs = create_programmatic_temperature_boundaries(inner_spec, outer_spec, config)
                return apply_temperature_boundaries!(field, pbs)
            else
                pbs = create_programmatic_composition_boundaries(inner_spec, outer_spec, config)
                return apply_composition_boundaries!(field, pbs)
            end
        end

        throw(ArgumentError(
            "File-based scalar BC loading through load_boundary_conditions! is not supported in the matrix-embedded BC workflow. " *
            "Use programmatic tuple specs here, or use the lower-level NetCDF/spectral loaders plus the field-specific BC setup path."))
    elseif field_type == VELOCITY
        throw(ArgumentError(
            "Velocity BCs are embedded in the implicit matrix system (see bcs/velocity_bc.jl); " *
            "load_boundary_conditions! does not support runtime velocity BC loading."))
    elseif field_type == MAGNETIC
        throw(ArgumentError(
            "Magnetic BCs are embedded in the implicit matrix system (see bcs/magnetic_bc.jl); " *
            "load_boundary_conditions! does not support runtime magnetic BC loading."))
    else
        throw(ArgumentError("Unknown field type: $field_type"))
    end
end

"""
    update_time_dependent_boundaries!(field, field_type::FieldType, current_time::Float64)

Update time-dependent boundary conditions for any field type.

For scalar fields, this refreshes the active boundary time index and rebuilds
the spectral boundary rows from the current physical-space boundary slice.
Vector BCs remain matrix-embedded and are therefore rejected if a
time-dependent boundary set is attached.
"""
function update_time_dependent_boundaries!(field, field_type::FieldType, current_time::Float64)
    if !(field_type == TEMPERATURE || field_type == COMPOSITION || field_type == VELOCITY ||
         field_type == MAGNETIC)
        throw(ArgumentError("Unknown field type: $field_type"))
    end

    if !hasfield(typeof(field), :boundary_condition_set) ||
       field.boundary_condition_set === nothing
        return field
    end

    boundary_set = field.boundary_condition_set
    is_time_dependent = boundary_set.inner_boundary.is_time_dependent ||
                        boundary_set.outer_boundary.is_time_dependent
    if !is_time_dependent
        return field
    end

    time_index = find_boundary_time_index(boundary_set, current_time)
    if hasfield(typeof(field), :boundary_time_index)
        field.boundary_time_index[] = time_index
    end

    if field_type == TEMPERATURE || field_type == COMPOSITION
        if !hasfield(typeof(field), :config) || !hasfield(typeof(field), :boundary_values)
            return field
        end

        config = field.config
        inner_slice = _extract_boundary_slice(boundary_set.inner_boundary, time_index)
        outer_slice = _extract_boundary_slice(boundary_set.outer_boundary, time_index)
        inner_coeffs = shtns_physical_to_spectral(inner_slice, config)
        outer_coeffs = shtns_physical_to_spectral(outer_slice, config)

        fill!(field.boundary_values, zero(eltype(field.boundary_values)))
        nlm = min(size(field.boundary_values, 2), length(inner_coeffs), length(outer_coeffs))
        for idx in 1:nlm
            field.boundary_values[1, idx] = inner_coeffs[idx]
            field.boundary_values[2, idx] = outer_coeffs[idx]
        end

        return field
    end

    throw(ArgumentError(
        "Time-dependent vector BC updates are not supported by update_time_dependent_boundaries!; " *
        "velocity and magnetic BCs are embedded in the implicit matrix system."))
end

"""
    validate_boundary_files(field_type::FieldType, boundary_specs::Dict, config)

Validate boundary condition specifications for any field type.

- NetCDF file specs are checked with `validate_netcdf_boundary_file`.
- Scalar tuple specs are validated via the programmatic boundary parser.
- Velocity and magnetic BCs reject tuple specs here because their supported
  configuration path is still the matrix-embedded timestep system.
"""
function validate_boundary_files(field_type::FieldType, boundary_specs::Dict, config)
    if !(field_type == TEMPERATURE || field_type == COMPOSITION || field_type == VELOCITY ||
         field_type == MAGNETIC)
        throw(ArgumentError("Unknown field type: $field_type"))
    end

    isempty(boundary_specs) && throw(ArgumentError("boundary_specs cannot be empty"))

    for side in keys(boundary_specs)
        if side != :inner && side != :outer
            throw(ArgumentError("Unsupported boundary key: $side. Supported keys are :inner and :outer."))
        end
    end

    for side in (:inner, :outer)
        haskey(boundary_specs, side) || continue
        spec = boundary_specs[side]

        if spec isa AbstractString
            validate_netcdf_boundary_file(spec)
        elseif spec isa Tuple
            if field_type == TEMPERATURE || field_type == COMPOSITION
                config === nothing &&
                    throw(ArgumentError("Programmatic scalar boundary validation requires a config"))
                _parse_boundary_spec(spec, config)
            else
                throw(ArgumentError(
                    "Tuple boundary specs are only supported for temperature and composition by validate_boundary_files; " *
                    "velocity/magnetic BCs are configured through the implicit matrix system."))
            end
        else
            throw(ArgumentError(
                "Unsupported boundary spec for $side: $(typeof(spec)). Expected a NetCDF filename or a programmatic tuple spec."))
        end
    end

    return true
end

function _extract_boundary_slice(boundary_data::BoundaryData, time_index::Int)
    values = boundary_data.values
    if !boundary_data.is_time_dependent
        return values
    elseif ndims(values) == 3
        return Matrix(@view values[:, :, time_index])
    elseif ndims(values) == 4 && boundary_data.ncomponents == 1
        return Matrix(@view values[:, :, time_index, 1])
    else
        throw(ArgumentError(
            "Only scalar time-dependent boundary data can be updated automatically; got array with size $(size(values))."))
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

# ================================================================================
# Common Utility Functions
# ================================================================================

# ================================================================================
# Cached SHTnsKit Configuration for Boundary Transforms (v1.1.15 optimization)
# ================================================================================
# This avoids recreating configs for each boundary transform call

const _BC_SHTNS_CONFIG_CACHE = Dict{Tuple{Int, Int, Int, Int}, Any}()
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
            _BC_SHTNS_CONFIG_CACHE[key] = SHTnsKit.create_gauss_config(lmax, nlat; mmax = mmax, nlon = nlon)
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
function shtns_physical_to_spectral(physical_data::Matrix{T}, config; return_complex::Bool = false) where {T}
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
                        coeffs[idx] = coeffs_matrix[l + 1, m + 1]
                    else
                        # Extract real part (boundary conditions typically use real values)
                        coeffs[idx] = real(coeffs_matrix[l + 1, m + 1])
                    end
                end
            end
        end

        return coeffs
    catch e
        error("SHTnsKit analysis failed: $e. " *
              "Cannot safely fall back to zeroed l>0 modes as this would produce " *
              "incorrect boundary conditions. Check SHTnsKit configuration and input data shape.")
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
function shtns_spectral_to_physical(coeffs::Vector{T}, config, nlat::Int, nlon::Int) where {T}
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
                    coeffs_matrix[l + 1, m + 1] = complex(coeffs[idx])
                end
            end
        end

        # Perform inverse transform
        physical_data = SHTnsKit.synthesis(shtconfig, coeffs_matrix; real_output = true)
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

@doc raw"""
    get_boundary_module_info()

Get information about the boundary-conditions module, including supported field
types, data formats, and major features.
"""
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
export gaunt_on_the_fly, gradient_gaunt_from_basic
export compute_gaunt_tensor
export evaluate_spherical_harmonics_grid, evaluate_spherical_harmonic_gradient_grid
export apply_all_topography_corrections!
export StefanState, initialize_stefan_state!, update_icb_topography!

end # module bcs
