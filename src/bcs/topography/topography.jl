# ================================================================================
# topography Module - Boundary Topography Coupling for Geodynamo Simulations
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

module topography

using LinearAlgebra
using SHTnsKit
using MPI
using NCDatasets

# Import parent module types and utilities
import ..bcs: BoundaryLocation, INNER_BOUNDARY, OUTER_BOUNDARY
import ..bcs: BoundaryType, DIRICHLET, NEUMANN
import ..bcs: FieldType, TEMPERATURE, COMPOSITION, VELOCITY, MAGNETIC
import ..bcs: get_rank, get_comm
import ..bcs: shtns_spectral_to_physical
# The corrections below must land in whichever boundary-value arrays the implicit
# solve actually reads — the interpolation cache wins once a spectral BC file is
# loaded, and writing to `field.boundary_values` regardless made every correction a
# silent no-op in exactly those runs.
import ..bcs: active_boundary_arrays

# `local_spectral_storage_slot` and `get_mode_index` are defined at the GeoDynamo
# top level (two modules up: topography -> bcs -> GeoDynamo). Forward to them
# lazily so this submodule does not depend on package include order.
const _GEODYNAMO_TOP = parentmodule(parentmodule(@__MODULE__))
@inline local_spectral_storage_slot(config, lm_idx::Int) =
    _GEODYNAMO_TOP.local_spectral_storage_slot(config, lm_idx)
@inline get_mode_index(config, l::Int, m::Int) = _GEODYNAMO_TOP.get_mode_index(config, l, m)
@inline gather_local_radial_profile!(output_real, output_imag, data_real, data_imag,
    slot, r_range) = _GEODYNAMO_TOP.gather_local_radial_profile!(
    output_real, output_imag, data_real, data_imag, slot, r_range)

# These will be available when the module is loaded in the context of GeoDynamo
# Define an abstract type for spectral fields - actual implementations will be duck-typed
abstract type AbstractSpectralField{T} end

# Compatibility alias and structural predicate for spectral field inputs.
# GeoDynamo's concrete spectral type is defined after the bcs module loads,
# so topography code must rely on structure rather than a direct subtype relation.
const SHTnsSpecField = AbstractSpectralField

@inline function _is_spectral_field_like(field)
    return hasfield(typeof(field), :data_real) &&
           hasfield(typeof(field), :data_imag) &&
           hasfield(typeof(field), :boundary_values) &&
           hasfield(typeof(field), :config) &&
           hasfield(typeof(field), :nlm)
end

# ----------------------------------------------------------------------------
# Per-step boundary-value base reset
#
# Topography corrections are a LAGGED function of the current field state and are
# re-applied every timestep (apply_solver_topography! runs each step/substage).
# They write the corrected boundary row in place (`bv -= ε·corr`). Without
# re-establishing the un-corrected base each step, the correction compounds:
# step n holds `base - n·ε·corr` instead of `base - ε·corr`, an unbounded drift.
#
# The base is captured on the FIRST call for a given boundary_values array (before
# any correction has been written) and restored before each subsequent application.
# Keyed by array identity so each field (temperature/composition/velocity poloidal
# & toroidal/magnetic) gets its own base snapshot. Velocity/magnetic bases are the
# all-zero initial rows; temperature/composition carry the parameter mean-mode base.
# ----------------------------------------------------------------------------
mutable struct BoundaryValueBase{A <: AbstractMatrix}
    target::WeakRef
    snapshot::A
    # What the topography correction LEFT in the array last time round, or `nothing`
    # before the first correction. `reset_boundary_to_base!` rolls the array back only
    # when it still matches this: the array has other owners
    # (`update_time_dependent_boundaries!`, `apply_temperature_boundaries!`) and
    # restoring a stale snapshot over their writes froze a time-dependent BC at its
    # t = 0 value for the rest of the run.
    applied::Union{A, Nothing}
end

BoundaryValueBase(target::WeakRef, snapshot::A) where {A <: AbstractMatrix} =
    BoundaryValueBase{A}(target, snapshot, nothing)

const _BOUNDARY_VALUE_BASE = Dict{UInt, BoundaryValueBase}()
const _BOUNDARY_VALUE_BASE_LOCK = ReentrantLock()

@inline function _restore_boundary_to_base!(bv, entry::BoundaryValueBase)
    copyto!(bv, entry.snapshot)
    return bv
end

function _prune_boundary_value_base_cache!()
    filter!(entry -> entry.second.target.value !== nothing, _BOUNDARY_VALUE_BASE)
    return nothing
end

function _finalize_boundary_value_base!(key::UInt, target)
    # A finalizer must not block if it interrupts code already holding this lock.
    trylock(_BOUNDARY_VALUE_BASE_LOCK) || return nothing
    try
        entry = get(_BOUNDARY_VALUE_BASE, key, nothing)
        if entry !== nothing
            cached_target = entry.target.value
            if cached_target === nothing || cached_target === target
                delete!(_BOUNDARY_VALUE_BASE, key)
            end
        end
    finally
        unlock(_BOUNDARY_VALUE_BASE_LOCK)
    end
    return nothing
end

function reset_boundary_to_base!(bv::AbstractMatrix)
    entry = lock(_BOUNDARY_VALUE_BASE_LOCK) do
        _prune_boundary_value_base_cache!()
        key = objectid(bv)
        entry = get(_BOUNDARY_VALUE_BASE, key, nothing)
        if entry === nothing || entry.target.value !== bv
            entry = BoundaryValueBase(WeakRef(bv), copy(bv))
            _BOUNDARY_VALUE_BASE[key] = entry
            if Base.ismutable(bv)
                finalizer(bv) do target
                    _finalize_boundary_value_base!(key, target)
                end
            end
            entry
        else
            entry
        end
    end
    # Roll back only OUR own correction. If the array no longer matches what the last
    # correction left, some other owner has written to it since — that write is the
    # new base, not something to undo.
    if entry.applied !== nothing && bv == entry.applied
        _restore_boundary_to_base!(bv, entry)
    else
        entry.snapshot = copy(bv)
        entry.applied = nothing
    end
    return bv
end

"""
    _active_boundary_array_list(fields...) -> Vector

The distinct, non-`nothing` boundary-value arrays (real and imaginary) that the solver
reads for `fields`. Deduplicated by identity, because two fields can legitimately share
one array and resetting it twice in a pass would roll the second reset back over the
first field's base.
"""
function _active_boundary_array_list(fields...)
    out = AbstractMatrix[]
    for f in fields
        for a in active_boundary_arrays(f)
            a === nothing && continue
            any(x -> x === a, out) && continue
            push!(out, a)
        end
    end
    return out
end

"""
    mark_boundary_applied!(bv)

Record `bv` as the state the topography correction just left behind.

`reset_boundary_to_base!` uses it to tell its own correction (safe to roll back) from
an update made by another owner of the same array (must be kept, and becomes the new
base). Call it once per boundary array after a correction pass has finished writing.
"""
function mark_boundary_applied!(bv::AbstractMatrix)
    lock(_BOUNDARY_VALUE_BASE_LOCK) do
        entry = get(_BOUNDARY_VALUE_BASE, objectid(bv), nothing)
        if entry !== nothing && entry.target.value === bv
            entry.applied = copy(bv)
        end
    end
    return bv
end

function clear_boundary_value_base_cache!()
    lock(_BOUNDARY_VALUE_BASE_LOCK) do
        empty!(_BOUNDARY_VALUE_BASE)
    end
    return nothing
end

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
        epsilon::Float64 = 0.01,
        velocity::Bool = true,
        magnetic::Bool = true,
        thermal::Bool = true,
        stefan::Bool = false,
        slope_terms::Bool = true,
        shift_terms::Bool = true,
        lmax_topo::Int = -1
)
    config = TopographyCouplingConfig(
        enabled = true,
        velocity_coupling = velocity,
        magnetic_coupling = magnetic,
        thermal_coupling = thermal,
        stefan_enabled = stefan,
        include_slope_terms = slope_terms,
        include_shift_terms = shift_terms,
        epsilon = epsilon,
        lmax_topo = lmax_topo
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
export gaunt_on_the_fly, gradient_gaunt_from_basic
export evaluate_spherical_harmonics_grid, evaluate_spherical_harmonic_gradient_grid

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
export clear_boundary_value_base_cache!
export mark_boundary_applied!

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
        config::TopographyCouplingConfig = get_topography_config())
    if !config.enabled
        return nothing
    end

    # Apply velocity corrections if enabled and field exists
    if config.velocity_coupling && _field_present(fields, :velocity)
        apply_velocity_topography_correction!(fields.velocity, topography, config)
    end

    # Apply magnetic corrections if enabled and field exists
    if config.magnetic_coupling && _field_present(fields, :magnetic)
        apply_magnetic_topography_correction!(fields.magnetic, topography, config)
    end

    # Apply thermal corrections if enabled and field exists
    if config.thermal_coupling && _field_present(fields, :temperature)
        apply_thermal_topography_correction!(fields.temperature, topography, config)
    end

    # Composition obeys the same advection-diffusion boundary algebra as
    # temperature and has its own correction, but was never dispatched: a run with
    # `include_composition = true` and topography enabled silently got no
    # compositional boundary correction at all.
    if config.thermal_coupling && _field_present(fields, :composition)
        apply_composition_topography_correction!(fields.composition, topography, config)
    end

    return nothing
end

"""
    _field_present(fields, name::Symbol) -> Bool

Whether `fields` actually CARRIES a field named `name`.

`hasfield` answers a question about the TYPE, and `SolverFields` declares its optional
slots as `magnetic::M where M <: Union{MagneticFieldsType,Nothing}` — so `hasfield` is
true on a hydro-only run and the magnetic correction was invoked with `nothing` every
single step, warning unthrottled from inside. Presence is a property of the value.
"""
_field_present(fields, name::Symbol) =
    hasfield(typeof(fields), name) && getfield(fields, name) !== nothing

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

end # module topography
