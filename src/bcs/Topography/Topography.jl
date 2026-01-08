# ================================================================================
# Topography Module - Boundary Topography Coupling for Geodynamo Simulations
# ================================================================================
#
# This module implements boundary topography effects for geodynamo simulations,
# following the linearized perturbation approach for small topographic deviations:
#
#   r = r_b + ε * h_b(θ, φ),  where ε << 1
#
# The topography introduces coupling between different spherical harmonic modes
# through Gaunt-type tensors (integrals of products of spherical harmonics).
#
# Physical Fields Affected:
# - Velocity: impermeability, no-slip, stress-free conditions
# - Magnetic: CMB insulating, ICB insulating/conducting conditions
# - Temperature: Dirichlet (fixed T), Neumann (fixed flux) conditions
# - Phase change: Stefan condition at ICB (optional)
#
# References:
# - Topography-linearized boundary conditions in poloidal-toroidal form
# - Gaunt tensor formulation for spectral coupling
#
# ================================================================================

module Topography

using LinearAlgebra
using SHTnsKit
using MPI
using NCDatasets

# Import parent module types and utilities
import ..bcs: BoundaryLocation, INNER_BOUNDARY, OUTER_BOUNDARY
import ..bcs: BoundaryType, DIRICHLET, NEUMANN
import ..bcs: FieldType, TEMPERATURE, COMPOSITION, VELOCITY, MAGNETIC
import ..bcs: get_rank, get_comm

# These will be available when the module is loaded in the context of GeoDynamo
# Define an abstract type for spectral fields - actual implementations will be duck-typed
abstract type AbstractSpectralField{T} end

# Type alias for compatibility - functions accept Any to allow duck typing
const SHTnsSpectralField = Any

# ================================================================================
# Topography Coupling Enable/Disable Flags
# ================================================================================

Base.@kwdef mutable struct TopographyCouplingConfig
    enabled::Bool = false
    velocity_coupling::Bool = true
    magnetic_coupling::Bool = true
    thermal_coupling::Bool = true
    stefan_enabled::Bool = false
    include_shift_terms::Bool = true
    include_slope_terms::Bool = true
    epsilon::Float64 = 0.01
    lmax_topo::Int = -1  # -1 means use simulation lmax
end

@doc """
    TopographyCouplingConfig

Configuration structure for enabling/disabling topography coupling.

# Fields
- `enabled::Bool`: Master switch for all topography coupling (default: false)
- `velocity_coupling::Bool`: Enable velocity BC topography corrections
- `magnetic_coupling::Bool`: Enable magnetic BC topography corrections
- `thermal_coupling::Bool`: Enable thermal BC topography corrections
- `stefan_enabled::Bool`: Enable Stefan condition for ICB evolution
- `include_shift_terms::Bool`: Include O(εh) shift terms (more accurate but slower)
- `include_slope_terms::Bool`: Include O(ε∇h) slope terms (required for coupling)
- `epsilon::Float64`: Topography amplitude parameter ε (default: 0.01)
- `lmax_topo::Int`: Maximum degree for topography expansion (default: same as simulation)

# Example
```julia
config = TopographyCouplingConfig(
    enabled = true,
    velocity_coupling = true,
    magnetic_coupling = true,
    epsilon = 0.02
)
```

See also: [`enable_topography!`](@ref), [`get_topography_config`](@ref)
""" TopographyCouplingConfig

# Global configuration instance
const TOPOGRAPHY_CONFIG = Ref{TopographyCouplingConfig}(TopographyCouplingConfig())

"""
    get_topography_config() -> TopographyCouplingConfig

Get the current topography coupling configuration.
"""
function get_topography_config()
    return TOPOGRAPHY_CONFIG[]
end

"""
    set_topography_config!(config::TopographyCouplingConfig)

Set the topography coupling configuration.
"""
function set_topography_config!(config::TopographyCouplingConfig)
    TOPOGRAPHY_CONFIG[] = config
    return config
end

"""
    enable_topography!(; kwargs...)

Enable topography coupling with optional configuration.

# Arguments
- `epsilon::Float64=0.01`: Topography amplitude parameter
- `velocity::Bool=true`: Enable velocity coupling
- `magnetic::Bool=true`: Enable magnetic coupling
- `thermal::Bool=true`: Enable thermal coupling
- `stefan::Bool=false`: Enable Stefan condition
- `slope_terms::Bool=true`: Include slope terms
- `shift_terms::Bool=true`: Include shift terms
- `lmax_topo::Int=-1`: Maximum topography degree (-1 for auto)

# Example
```julia
enable_topography!(epsilon=0.05, stefan=true)
```
"""
function enable_topography!(;
    epsilon::Float64=0.01,
    velocity::Bool=true,
    magnetic::Bool=true,
    thermal::Bool=true,
    stefan::Bool=false,
    slope_terms::Bool=true,
    shift_terms::Bool=true,
    lmax_topo::Int=-1
)
    config = TopographyCouplingConfig(
        enabled=true,
        velocity_coupling=velocity,
        magnetic_coupling=magnetic,
        thermal_coupling=thermal,
        stefan_enabled=stefan,
        include_slope_terms=slope_terms,
        include_shift_terms=shift_terms,
        epsilon=epsilon,
        lmax_topo=lmax_topo
    )
    set_topography_config!(config)

    if get_rank() == 0
        @info "Topography coupling enabled" epsilon=epsilon velocity magnetic thermal stefan
    end

    return config
end

"""
    disable_topography!()

Disable all topography coupling.
"""
function disable_topography!()
    config = get_topography_config()
    config.enabled = false
    if get_rank() == 0
        @info "Topography coupling disabled"
    end
    return config
end

"""
    is_topography_enabled() -> Bool

Check if topography coupling is enabled.
"""
function is_topography_enabled()
    return get_topography_config().enabled
end

# ================================================================================
# Include submodules
# ================================================================================

include("gaunt_tensors.jl")
include("derivatives.jl")
include("topography_data.jl")
include("velocity_coupling.jl")
include("magnetic_coupling.jl")
include("thermal_coupling.jl")
include("stefan_condition.jl")

# ================================================================================
# Exports
# ================================================================================

# Configuration
export TopographyCouplingConfig
export get_topography_config, set_topography_config!
export enable_topography!, disable_topography!, is_topography_enabled

# Gaunt tensors
export GauntTensorCache
export compute_gaunt_tensor, compute_gradient_gaunt_tensor, compute_cross_gaunt_tensor
export precompute_gaunt_tensors!, get_gaunt_tensor, get_gradient_gaunt, get_cross_gaunt

# Topography data
export TopographyData, TopographyField
export create_topography_data, load_topography_from_file, load_topography_from_array
export get_topography_coefficients, set_topography_coefficients!
export create_uniform_topography, create_spherical_harmonic_topography

# Velocity coupling
export apply_velocity_topography_correction!
export compute_impermeability_correction
export compute_noslip_correction
export compute_stressfree_correction

# Magnetic coupling
export apply_magnetic_topography_correction!
export compute_cmb_insulating_correction
export compute_icb_insulating_correction

# Thermal coupling
export apply_thermal_topography_correction!
export compute_dirichlet_thermal_correction
export compute_neumann_thermal_correction

# Stefan condition
export StefanState
export initialize_stefan_state!, update_icb_topography!
export compute_stefan_flux

# High-level interface
export apply_all_topography_corrections!

# ================================================================================
# High-level Interface Functions
# ================================================================================

"""
    apply_all_topography_corrections!(fields, topography, config)

Apply all enabled topography corrections to the simulation fields.

This is the main entry point for applying topography coupling during
the simulation timestep.

# Arguments
- `fields`: Named tuple or struct containing velocity, magnetic, temperature fields
- `topography`: TopographyData containing ICB and CMB topography
- `config`: TopographyCouplingConfig (optional, uses global if not provided)
"""
function apply_all_topography_corrections!(fields, topography;
                                           config::TopographyCouplingConfig=get_topography_config())
    if !config.enabled
        return nothing
    end

    # Apply velocity corrections if enabled and field exists
    if config.velocity_coupling && hasfield(typeof(fields), :velocity)
        apply_velocity_topography_correction!(fields.velocity, topography, config)
    end

    # Apply magnetic corrections if enabled and field exists
    if config.magnetic_coupling && hasfield(typeof(fields), :magnetic)
        apply_magnetic_topography_correction!(fields.magnetic, topography, config)
    end

    # Apply thermal corrections if enabled and field exists
    if config.thermal_coupling && hasfield(typeof(fields), :temperature)
        apply_thermal_topography_correction!(fields.temperature, topography, config)
    end

    return nothing
end

"""
    print_topography_summary(topography::TopographyData)

Print a summary of the current topography configuration.
"""
function print_topography_summary(topography::TopographyData)
    config = get_topography_config()

    println("╔═══════════════════════════════════════════════════════════════╗")
    println("║              TOPOGRAPHY COUPLING SUMMARY                       ║")
    println("╠═══════════════════════════════════════════════════════════════╣")
    println("║ Status: $(config.enabled ? "ENABLED" : "DISABLED")                                          ║")
    println("║ Epsilon (ε): $(config.epsilon)                                      ║")
    println("╠───────────────────────────────────────────────────────────────╣")
    println("║ Coupling Flags:                                               ║")
    println("║   Velocity: $(config.velocity_coupling)                                        ║")
    println("║   Magnetic: $(config.magnetic_coupling)                                        ║")
    println("║   Thermal:  $(config.thermal_coupling)                                        ║")
    println("║   Stefan:   $(config.stefan_enabled)                                        ║")
    println("╠───────────────────────────────────────────────────────────────╣")
    println("║ Term Inclusion:                                               ║")
    println("║   Slope terms (∇h): $(config.include_slope_terms)                             ║")
    println("║   Shift terms (h):  $(config.include_shift_terms)                             ║")
    println("╠───────────────────────────────────────────────────────────────╣")

    if topography.icb !== nothing
        icb = topography.icb
        println("║ ICB Topography:                                               ║")
        println("║   L_max: $(icb.lmax)                                                 ║")
        println("║   RMS amplitude: $(round(icb.rms_amplitude, digits=6))                        ║")
    end

    if topography.cmb !== nothing
        cmb = topography.cmb
        println("║ CMB Topography:                                               ║")
        println("║   L_max: $(cmb.lmax)                                                 ║")
        println("║   RMS amplitude: $(round(cmb.rms_amplitude, digits=6))                        ║")
    end

    println("╚═══════════════════════════════════════════════════════════════╝")
end

end # module Topography
