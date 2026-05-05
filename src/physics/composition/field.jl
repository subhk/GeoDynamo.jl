# ================================================================================
# Composition Field Module with SHTns
# ================================================================================
#
# This module implements the composition field evolution for geodynamo simulations
# using spherical harmonic transforms (SHTnsKit).
#
# ================================================================================
# GOVERNING EQUATION
# ================================================================================
# The non-dimensional composition equation in magnetic diffusion time scaling:
#
#   ∂C/∂t + u·∇C = (Pm/Sc) ∇²C
#
# where:
#   C            : Composition (light element concentration perturbation)
#   u            : Velocity field
#   Pm = ν/η     : Magnetic Prandtl number
#   Sc = ν/κ_c   : Schmidt number (viscous/compositional diffusivity)
#   Pm/Sc = κ_c/η: Ratio of compositional to magnetic diffusivity
#
# PHYSICAL INTERPRETATION:
# ========================
# - Left side: Time rate of change + advection by flow
# - Right side: Compositional diffusion
#
# The advection term -u·∇C represents light element transport by the convecting
# fluid. This is the EXPLICIT part computed in physical space.
#
# The diffusion term (Pm/Sc)∇²C is treated IMPLICITLY for numerical stability.
# The diffusivity coefficient passed to the time-stepper is Pm/Sc.
#
# ================================================================================
# BOUNDARY CONDITIONS
# ================================================================================
#
# Common configurations for compositional convection:
#   - No-flux (Neumann): ∂C/∂r = 0 at both boundaries (default)
#     This represents impermeable boundaries with no mass flux
#   - Fixed composition (Dirichlet): C = C_boundary at r = r_i, r_o
#   - Mixed: Different types at inner/outer boundaries
#
# For typical inner core boundary evolution:
#   bc_type_inner[1] = NEUMANN   # No-flux for l=0, m=0 at ICB
#   bc_type_outer[1] = NEUMANN   # No-flux for l=0, m=0 at CMB
#   (All modes typically use no-flux with zero boundary values)
#
# ================================================================================
# MPI PARALLELIZATION
# ================================================================================
#
# Data distribution follows the same pattern as thermal field:
#   - Spectral data: distributed over (l,m) modes via lm_range
#   - Radial data: distributed over radial points via r_range
#   - Physical data: distributed using pencil decomposition
#
# See `fields/scalar_operators.jl` for shared utility functions and MPI patterns.
#
# ================================================================================

using PencilArrays
using SHTnsKit
using LinearAlgebra
using SparseArrays

import .bcs
import .bcs: BoundaryType, DIRICHLET, NEUMANN

# `fields/scalar_operators.jl` is included in the main module, so its shared
# scalar-field helpers are available here.

"""
    SHTnsCompositionField{T}

Composition field state used by the compositional transport equation.

This mirrors the temperature-field layout so scalar transport code can share
operators and diagnostics while keeping composition-specific boundary/source
data separate.
"""
mutable struct SHTnsCompositionField{
    T,
    C<:SHTnsKitConfig,
    PF<:SHTnsPhysField{T},
    VF<:SHTnsVectorField{T},
    SF<:SHTnsSpecField{T},
} <: AbstractScalarField{T}
    # Physical space composition
    composition::PF
    gradient::VF

    # Spectral representation
    spectral::SF

    # Nonlinear terms (advection)
    nonlinear::SF
    prev_nonlinear::SF

    # Work arrays for efficient computation
    work_spectral::SF
    work_physical::PF
    advection_physical::PF

    # Boundary conditions
    boundary_values::Matrix{T}         # [2, nlm] for ICB and CMB
    bc_type_inner::Vector{Int}         # BC type for each mode at inner
    bc_type_outer::Vector{Int}         # BC type for each mode at outer

    # File-based boundary condition support (matching physics/temperature/field.jl field order)
    boundary_condition_set::Union{bcs.BoundaryConditionSet{T}, Nothing}
    boundary_interpolation_cache::bcs.BoundaryInterpolationCache{T}
    boundary_time_index::Ref{Int}

    # Pre-computed coefficients
    ℓ_factors::Vector{T}               # l(l+1) values for diffusion

    # Internal sources (radial profile, matching thermal field)
    internal_sources::Vector{T}

    # Configuration (SHTnsKit)
    config::C

    # Radial derivative matrices
    ∂r::BandedMatrix{T}         # First derivative d/dr
    ∂²r::BandedMatrix{T}        # Second derivative d²/dr²

    # Spectral derivative operators (matching physics/temperature/field.jl types)
    ∂θ::SparseMatrixCSC{T,Int}  # Pre-computed θ-derivative
    theta_recurrence_coeffs::Matrix{T}               # Recurrence coefficients

    # Performance tracking
    computation_time::Ref{Float64}
    transform_time::Ref{Float64}
    comm_time::Ref{Float64}
    spectral_time::Ref{Float64}

    # Geometry
    domain::RadialDomain
end

# Specialization for composition field
get_main_physical_field(𝔽::SHTnsCompositionField{T}) where T = 𝔽.composition

# ================================================================================
# Field Creation
# ================================================================================

"""
    create_shtns_composition_field(::Type{T}, config,
                                   𝒟ᵒᶜ::RadialDomain,
                                   pencils=nothing, pencil_spec=nothing) where T

Create a composition field structure with all necessary arrays for spectral computation.

# Arguments
- `T`: Numeric type (e.g., Float64)
- `config`: SHTnsKitConfig with transform parameters (lmax, mmax, nlat, nlon)
- `𝒟ᵒᶜ`: Radial domain information for outer core
- `pencils`: Optional pencil decomposition (defaults to config.pencils)
- `pencil_spec`: Optional spectral pencil (defaults to pencils.spec)

# Returns
- `SHTnsCompositionField{T}`: Fully initialized composition field structure

# Example
```julia
config = create_shtns_config(lmax=32, mmax=32, nlat=64, nlon=128)
domain = create_radial_domain(nr=64, ri=0.35, ro=1.0)
𝔽 = create_shtns_composition_field(Float64, config, domain)
```

# Notes
Default boundary conditions are no-flux (NEUMANN) at both boundaries,
appropriate for typical compositional convection problems.
"""
function create_shtns_composition_field(::Type{T}, config::C,
                                        𝒟ᵒᶜ::RadialDomain,
                                        pencils=nothing, pencil_spec=nothing) where {T,C<:SHTnsKitConfig}
    # Use config's pencils by default (consistent with velocity/magnetic creators)
    if pencils === nothing
        pencils = config.pencils
    end
    if pencil_spec === nothing
        pencil_spec = pencils.spec
    end

    # Composition field in r-pencil for efficient radial operations
    composition = create_shtns_physical_field(T, config, 𝒟ᵒᶜ, pencils.r)

    # Gradient components
    gradient = create_shtns_vector_field(T, config, 𝒟ᵒᶜ,
                                        (pencils.θ, pencils.φ, pencils.r))

    # Spectral representation using spectral pencil
    spectral       = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    nonlinear      = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    prev_nonlinear = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)

    # Work arrays
    work_spectral      = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    work_physical      = create_shtns_physical_field(T, config, 𝒟ᵒᶜ, pencils.r)
    advection_physical = create_shtns_physical_field(T, config, 𝒟ᵒᶜ, pencils.r)

    # Boundary conditions
    boundary_values  = zeros(T, 2, config.nlm)

    # Default BC types (DIRICHLET = fixed value, NEUMANN = fixed flux)
    # For composition: typically no-flux at both boundaries
    bc_type_inner = fill(Int(NEUMANN), config.nlm)  # No-flux at inner boundary
    bc_type_outer = fill(Int(NEUMANN), config.nlm)  # No-flux at outer boundary

    # Pre-compute l(l+1) factors (matching physics/temperature/field.jl pattern)
    ℓ_factors = T[l * (l + 1) for l in config.l_values]

    # Create radial derivative matrices
    ∂r  = create_derivative_matrix(T, 1, 𝒟ᵒᶜ)
    ∂²r = create_derivative_matrix(T, 2, 𝒟ᵒᶜ)

    # Pre-compute spectral derivative operators
    ∂θ = build_∂θ(T, config)
    theta_recurrence_coeffs = compute_theta_recurrence_coefficients(T, config)

    # Internal sources (radial profile, e.g. secular cooling)
    internal_sources = zeros(T, 𝒟ᵒᶜ.N)

    return SHTnsCompositionField(
        composition, gradient, spectral, nonlinear, prev_nonlinear,
        work_spectral, work_physical, advection_physical,
        boundary_values, bc_type_inner, bc_type_outer,
        nothing, bcs.BoundaryInterpolationCache(T), Ref(1),  # boundary condition fields
        ℓ_factors, internal_sources, config,
        ∂r, ∂²r,
        ∂θ, theta_recurrence_coeffs,
        Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0),
        𝒟ᵒᶜ
    )
end

# Matrix-embedded composition BC functions
include("../../bcs/compositional_bc.jl")

# ================================================================================
# Main Nonlinear Computation
# ================================================================================

"""
    compute_composition_nonlinear!(𝔽::SHTnsCompositionField{T},
                                   vel_fields, 𝒟ᵒᶜ::RadialDomain,
                                   ws::GradientWorkspace{T};
                                   geometry::Symbol) where T

Compute the nonlinear advection term -u·∇C for the composition equation.

This function implements the EXPLICIT part of the composition equation:
    ∂C/∂t + u·∇C = (Pm/Sc) ∇²C

The advection term -u·∇C is computed using the following optimized workflow:
1. Compute ∇C in spectral space (no MPI communication required)
2. Batched transform of C and ∇C to physical space
3. Compute advection -u·∇C locally in physical space
4. Transform result back to spectral space
5. Apply boundary conditions

# Arguments
- `𝔽`: Composition field structure to update
- `vel_fields`: Velocity field structure (provides u for advection)
- `𝒟ᵒᶜ`: Radial domain information
- `geometry`: Geometry type (:shell or :ball) for transform selection

# Side Effects
- Updates `𝔽.nonlinear` with the computed advection term
- Updates `𝔽.gradient` with ∇C in physical space
- Applies boundary conditions to spectral representation

# Performance Notes
- Gradients computed in spectral space avoid MPI communication
- Batched transforms minimize data transfer overhead
- Physical space operations are embarrassingly parallel
"""
function compute_composition_nonlinear!(𝔽::SHTnsCompositionField{T},
                                        vel_fields, 𝒟ᵒᶜ::RadialDomain,
                                        ws::GradientWorkspace{T};
                                        geometry::Symbol) where T
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0

    # Zero work arrays and gradient workspace
    zero_scalar_work_arrays!(𝔽)
    zero_gradient_workspace!(ws)

    # Step 1: Compute ALL gradients in spectral space (NO COMMUNICATION!)
    if ENABLE_TIMING[]
        t_spectral = MPI.Wtime()
    end
    compute_all_gradients_spectral!(𝔽, 𝒟ᵒᶜ, ws)
    if ENABLE_TIMING[]
        𝔽.spectral_time[] += MPI.Wtime() - t_spectral
    end

    # Step 2: Single batched transform of composition and gradients to physical
    if ENABLE_TIMING[]
        t_transform = MPI.Wtime()
    end
    transform_field_and_gradients_to_physical!(𝔽, ws)
    if ENABLE_TIMING[]
        𝔽.transform_time[] += MPI.Wtime() - t_transform
    end

    # Step 3: Compute advection term -u·∇C in physical space (local operation)
    if vel_fields !== nothing
        compute_scalar_advection_local!(𝔽, vel_fields)
    end

    # Step 4: Transform advection back to spectral space
    if ENABLE_TIMING[]
        t_transform = MPI.Wtime()
    end
    if geometry === :ball
        ball_physical_to_spectral!(𝔽.advection_physical, 𝔽.nonlinear)
    else
        shtnskit_physical_to_spectral!(𝔽.advection_physical, 𝔽.nonlinear)
    end
    if ENABLE_TIMING[]
        𝔽.transform_time[] += MPI.Wtime() - t_transform
        𝔽.computation_time[] += MPI.Wtime() - t_start
    end
end

# Backwards compatibility wrapper using shared implementation from
# `fields/scalar_operators.jl`.
zero_composition_work_arrays!(𝔽::SHTnsCompositionField{T}) where T = zero_scalar_work_arrays!(𝔽)

# ================================================================================
# NOTE: Gradient computation functions moved to `fields/scalar_operators.jl`.
# NOTE: Batched transform operations moved to `fields/scalar_operators.jl`.
# ================================================================================

# ================================================================================
# Validation and Testing
# ================================================================================

function validate_composition_field(𝔽::SHTnsCompositionField{T}, domain::RadialDomain) where T
    """
    Validate composition field consistency and boundary conditions
    """
    errors = String[]
    
    # Check spectral field dimensions
    spec_real = parent(𝔽.spectral.data_real)
    local_slot_capacity = size(spec_real, 1) * size(spec_real, 2)
    local_mode_count = length(local_spectral_mode_indices(𝔽.config))
    if local_mode_count > local_slot_capacity
        push!(errors, "Spectral field local slot capacity is smaller than owned spectral mode count")
    end
    
    # Check boundary condition arrays
    if length(𝔽.bc_type_inner) != 𝔽.config.nlm
        push!(errors, "Inner BC array size mismatch")
    end
    
    if length(𝔽.bc_type_outer) != 𝔽.config.nlm
        push!(errors, "Outer BC array size mismatch")
    end

    if !isempty(errors)
        error("Composition field validation failed:\n" * join(errors, "\n"))
    end
    
    return true
end

# ================================================================================
# Diagnostic functions
# ================================================================================

"""
    compute_composition_rms(composition_field, domain)

Compute a global RMS measure of the composition field from its spectral
coefficients.
"""
function compute_composition_rms(𝔽::SHTnsCompositionField{T}, 𝒟ᵒᶜ::RadialDomain) where T
    # Compute RMS composition
    spec_real = parent(𝔽.spectral.data_real)
    spec_imag = parent(𝔽.spectral.data_imag)
    
    local_sum = zero(T)
    
    lm_range = local_spectral_mode_indices(𝔽.config)
    r_range = range_local(𝔽.config.pencils.spec, 3)

    # Sum over local spectral modes
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        local_r <= size(spec_real, 3) || continue

        for lm_idx in lm_range
            slot = local_spectral_storage_slot(𝔽.config, lm_idx)
            slot === nothing && continue
            val_real = local_spectral_value(spec_real, slot, local_r)
            val_imag = local_spectral_value(spec_imag, slot, local_r)
            local_sum += val_real^2 + val_imag^2
        end
    end
    
    # Global reduction
    comm = get_comm()
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, comm)
    
    return sqrt(global_sum / (𝒟ᵒᶜ.N * 𝔽.config.nlm))
end

"""
    compute_composition_energy(composition_field, domain)

Compute a volume-averaged compositional energy from the current physical field.
"""
function compute_composition_energy(𝔽::SHTnsCompositionField{T}, 𝒟ᵒᶜ::RadialDomain) where T
    # Compute compositional energy ∫ C² dV
    
    # Transform to physical space first
    shtnskit_spectral_to_physical!(𝔽.spectral, 𝔽.work_physical)
    
    comp_data = parent(𝔽.work_physical.data)
    local_energy = zero(T)
    
    # Integrate over local physical space
    nlat, nlon, nr = size(comp_data)
    
    for r_idx in 1:nr
        for φ_idx in 1:nlon  
            for θ_idx in 1:nlat
                C = comp_data[θ_idx, φ_idx, r_idx]
                local_energy += C^2
            end
        end
    end
    
    # Global reduction
    comm = get_comm()
    global_energy = MPI.Allreduce(local_energy, MPI.SUM, comm)
    
    return global_energy / (𝔽.config.nlat * 𝔽.config.nlon * 𝒟ᵒᶜ.N)
end

# ================================================================================
# Performance monitoring and statistics
# ================================================================================

"""
    get_composition_statistics(composition_field, domain)

Return a named collection of basic composition statistics such as extrema, RMS
level, and integrated energy.
"""
function get_composition_statistics(𝔽::SHTnsCompositionField{T}, 
                                   domain::RadialDomain) where T
    """
    Compute various composition field statistics
    """
    # Min/max composition
    comp_data = parent(𝔽.composition.data)
    local_min = minimum(comp_data)
    local_max = maximum(comp_data)

    global_min = MPI.Allreduce(local_min, MPI.MIN, get_comm())
    global_max = MPI.Allreduce(local_max, MPI.MAX, get_comm())

    # RMS composition
    local_sum = sum(abs2, comp_data)
    local_count = length(comp_data)

    global_sum = MPI.Allreduce(local_sum, MPI.SUM, get_comm())
    global_count = MPI.Allreduce(local_count, MPI.SUM, get_comm())
    
    rms_comp = sqrt(global_sum / global_count)
    
    # Total energy
    energy = compute_composition_energy(𝔽, domain)
    
    return (min = global_min,
            max = global_max,
            rms = rms_comp,
            energy = energy)
end

# ================================================================================
# Utility functions
# ================================================================================

"""
    set_composition_ic!(field, ic_type, domain)

Initialize the composition field from one of the built-in analytic profiles.
The helper is intended for tests and simple setups that want a uniform, linear,
or randomly perturbed compositional state without reading a restart file.
"""
function set_composition_ic!(𝔽::SHTnsCompositionField{T}, 
                             ic_type::Symbol, 𝒟ᵒᶜ::RadialDomain) where T
    # Set initial conditions for composition field
    
    spec_real = parent(𝔽.spectral.data_real)
    spec_imag = parent(𝔽.spectral.data_imag)

    fill!(spec_real, zero(T))
    fill!(spec_imag, zero(T))
    lm_range = local_spectral_mode_indices(𝔽.config)
    r_range = range_local(𝔽.config.pencils.spec, 3)
    
    if ic_type == :uniform
        # Uniform composition (only l=0, m=0 mode)
        # Set uniform value in l=0, m=0 mode
        slot = local_spectral_storage_slot(𝔽.config, 1)
        if slot !== nothing
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                set_local_spectral_value!(spec_real, slot, local_r, one(T))
            end
        end
        
    elseif ic_type == :linear
        # Linear profile in radius
        slot = local_spectral_storage_slot(𝔽.config, 1)
        if slot !== nothing
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                r = 𝒟ᵒᶜ.r[r_idx, 4]  # Normalized radius
                set_local_spectral_value!(spec_real, slot, local_r, T(r))
            end
        end
        
    elseif ic_type == :random_perturbation
        # Small random perturbation on all modes
        amplitude = 1e-3
        for r_idx in r_range
            local_r = r_idx - first(r_range) + 1
            for lm_idx in lm_range
                if lm_idx > 1  # Don't perturb l=0, m=0 mode
                    slot = local_spectral_storage_slot(𝔽.config, lm_idx)
                    slot === nothing && continue
                    set_local_spectral_value!(spec_real, slot, local_r, amplitude * (rand(T) - T(0.5)))
                    set_local_spectral_value!(spec_imag, slot, local_r, amplitude * (rand(T) - T(0.5)))
                end
            end
        end
    end
end

"""
    set_composition_boundary_conditions!(field, bc_inner, bc_outer,
                                         value_inner=zero(T), value_outer=zero(T))

Assign simple programmatic boundary conditions to the composition field. Use
`:fixed` for Dirichlet concentration values or `:no_flux` for impermeable
Neumann boundaries at the inner and outer shell surfaces.
"""
function set_composition_boundary_conditions!(𝔽::SHTnsCompositionField{T}, 
                                              bc_inner::Symbol, bc_outer::Symbol,
                                              value_inner::T = zero(T), value_outer::T = zero(T)) where T
    # Set boundary condition types and values
    
    if bc_inner == :fixed
        fill!(𝔽.bc_type_inner, Int(DIRICHLET))
        𝔽.boundary_values[1, :] .= value_inner
    elseif bc_inner == :no_flux
        fill!(𝔽.bc_type_inner, Int(NEUMANN))
    end

    if bc_outer == :fixed
        fill!(𝔽.bc_type_outer, Int(DIRICHLET))
        𝔽.boundary_values[2, :] .= value_outer
    elseif bc_outer == :no_flux
        fill!(𝔽.bc_type_outer, Int(NEUMANN))
    end
end

# Note: NetCDF boundary condition functions moved to src/bcs/composition.jl

# ================================================================================
# Export functions
# ================================================================================
# export SHTnsCompositionField, create_shtns_composition_field
# export compute_composition_nonlinear!
# export compute_composition_rms, compute_composition_energy
# export get_composition_statistics
# export zero_composition_work_arrays!
# export set_composition_ic!, set_composition_boundary_conditions!

# Note: File-based boundary condition functions moved to src/bcs/composition.jl

# Note: Boundary condition exports moved to src/bcs/composition.jl
