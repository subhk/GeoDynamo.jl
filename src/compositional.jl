# ================================================================================
# Composition Field Module with SHTns
# ================================================================================
#
# This module implements the composition field evolution for geodynamo simulations
# using spherical harmonic transforms (SHTnsKit).
#
# REFERENCE: Sreenivasan & Kar (2018), Phys. Rev. Fluids 3, 093801
#            Equation: Composition evolution (analogous to thermal)
#
# ================================================================================
# GOVERNING EQUATION
# ================================================================================
#
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
# The diffusivity coefficient passed to the time-stepper is (Pm/Sc) = d_Pm/d_Sc.
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
# See scalar_field_common.jl for shared utility functions and MPI patterns.
#
# ================================================================================

using PencilArrays
using SHTnsKit
using LinearAlgebra
using SparseArrays

import .bcs
import .bcs: BoundaryType, DIRICHLET, NEUMANN

# scalar_field_common.jl is included in main module - functions are available here

mutable struct SHTnsCompositionField{T} <: AbstractScalarField{T}
    # Physical space composition
    composition::SHTnsPhysicalField{T}
    gradient::SHTnsVectorField{T}

    # Spectral representation
    spectral::SHTnsSpectralField{T}

    # Nonlinear terms (advection)
    nonlinear::SHTnsSpectralField{T}
    prev_nonlinear::SHTnsSpectralField{T}

    # Work arrays for efficient computation
    work_spectral::SHTnsSpectralField{T}
    work_physical::SHTnsPhysicalField{T}
    advection_physical::SHTnsPhysicalField{T}

    # Gradient spectral components for efficiency
    grad_theta_spec::SHTnsSpectralField{T}
    grad_phi_spec::SHTnsSpectralField{T}
    grad_r_spec::SHTnsSpectralField{T}

    # Sources and boundary conditions
    internal_sources::Vector{T}        # Radial profile of compositional sources
    boundary_values::Matrix{T}         # [2, nlm] for ICB and CMB
    bc_type_inner::Vector{Int}         # BC type for each mode at inner
    bc_type_outer::Vector{Int}         # BC type for each mode at outer

    # File-based boundary condition support (matching thermal.jl field order)
    boundary_condition_set::Union{bcs.BoundaryConditionSet{T}, Nothing}
    boundary_interpolation_cache::Dict{String, Any}
    boundary_time_index::Ref{Int}

    # Pre-computed coefficients
    ℓ_factors::Vector{Float64}         # l(l+1) values for diffusion

    # Configuration (SHTnsKit)
    config::SHTnsKitConfig

    # Radial derivative matrices
    ∂r::BandedMatrix{T}         # First derivative d/dr
    ∂²r::BandedMatrix{T}        # Second derivative d²/dr²

    # Spectral derivative operators (matching thermal.jl types)
    theta_derivative_matrix::SparseMatrixCSC{T,Int}  # Pre-computed θ-derivative
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
get_main_physical_field(field::SHTnsCompositionField{T}) where T = field.composition

# ================================================================================
# Field Creation
# ================================================================================

"""
    create_shtns_composition_field(::Type{T}, config::SHTnsKitConfig,
                                   Dᵒᶜ::RadialDomain,
                                   pencils=nothing, pencil_spec=nothing) where T

Create a composition field structure with all necessary arrays for spectral computation.

# Arguments
- `T`: Numeric type (e.g., Float64)
- `config`: SHTnsKitConfig with transform parameters (lmax, mmax, nlat, nlon)
- `Dᵒᶜ`: Radial domain information for outer core
- `pencils`: Optional pencil decomposition (defaults to config.pencils)
- `pencil_spec`: Optional spectral pencil (defaults to pencils.spec)

# Returns
- `SHTnsCompositionField{T}`: Fully initialized composition field structure

# Example
```julia
config = create_shtns_config(lmax=32, mmax=32, nlat=64, nlon=128)
domain = create_radial_domain(nr=64, ri=0.35, ro=1.0)
comp_field = create_shtns_composition_field(Float64, config, domain)
```

# Notes
Default boundary conditions are no-flux (NEUMANN) at both boundaries,
appropriate for typical compositional convection problems.
"""
function create_shtns_composition_field(::Type{T}, config::SHTnsKitConfig,
                                        Dᵒᶜ::RadialDomain,
                                        pencils=nothing, pencil_spec=nothing) where T
    # Use config's pencils by default (consistent with velocity/magnetic creators)
    if pencils === nothing
        pencils = config.pencils
    end
    if pencil_spec === nothing
        pencil_spec = pencils.spec
    end

    # Composition field in r-pencil for efficient radial operations
    composition = create_shtns_physical_field(T, config, Dᵒᶜ, pencils.r)

    # Gradient components
    gradient = create_shtns_vector_field(T, config, Dᵒᶜ,
                                        (pencils.θ, pencils.φ, pencils.r))

    # Spectral representation using spectral pencil
    spectral       = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    nonlinear      = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    prev_nonlinear = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)

    # Work arrays
    work_spectral      = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    work_physical      = create_shtns_physical_field(T, config, Dᵒᶜ, pencils.r)
    advection_physical = create_shtns_physical_field(T, config, Dᵒᶜ, pencils.r)

    # Gradient spectral components
    grad_theta_spec = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    grad_phi_spec   = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    grad_r_spec     = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)

    # Sources and boundary conditions
    internal_sources = zeros(T, Dᵒᶜ.N)
    boundary_values  = zeros(T, 2, config.nlm)

    # Default BC types (DIRICHLET = fixed value, NEUMANN = fixed flux)
    # For composition: typically no-flux at both boundaries
    bc_type_inner = fill(Int(NEUMANN), config.nlm)  # No-flux at inner boundary
    bc_type_outer = fill(Int(NEUMANN), config.nlm)  # No-flux at outer boundary

    # Storage for file-based boundary conditions
    boundary_data_cache = Dict{String, Any}()

    # Pre-compute l(l+1) factors (matching thermal.jl pattern)
    ℓ_factors = Float64[l * (l + 1) for l in config.l_values]

    # Create radial derivative matrices
    ∂r  = create_derivative_matrix(1, Dᵒᶜ)
    ∂²r = create_derivative_matrix(2, Dᵒᶜ)

    # Pre-compute spectral derivative operators
    theta_derivative_matrix = build_theta_derivative_matrix(T, config)
    theta_recurrence_coeffs = compute_theta_recurrence_coefficients(T, config)

    return SHTnsCompositionField{T}(
        composition, gradient, spectral, nonlinear, prev_nonlinear,
        work_spectral, work_physical, advection_physical,
        grad_theta_spec, grad_phi_spec, grad_r_spec,
        internal_sources, boundary_values,
        bc_type_inner, bc_type_outer,
        nothing, Dict{String, Any}(), Ref(1),  # boundary condition fields
        ℓ_factors, config,
        ∂r, ∂²r,
        theta_derivative_matrix, theta_recurrence_coeffs,
        Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0),
        Dᵒᶜ
    )
end

"""
    enforce_composition_boundary_values!(field)

Anchor spectral boundary modes to stored Dirichlet values when applicable.
"""
function enforce_composition_boundary_values!(field::SHTnsCompositionField{T}) where T
    spec_real = parent(field.spectral.data_real)
    spec_imag = parent(field.spectral.data_imag)

    lm_range = range_local(field.config.pencils.spec, 1)
    r_range  = range_local(field.config.pencils.spec, 3)

    has_inner = 1 in r_range && field.domain.r[1, 4] > 0
    has_outer = field.domain.N in r_range

    inner_idx = has_inner ? (1 - first(r_range) + 1) : 0
    outer_idx = has_outer ? (field.domain.N - first(r_range) + 1) : 0

    dirichlet = Int(bcs.DIRICHLET)

    for lm_idx in lm_range
        if lm_idx <= field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1

            if has_inner && 1 <= inner_idx <= size(spec_real, 3) && field.bc_type_inner[lm_idx] == dirichlet
                spec_real[local_lm, 1, inner_idx] = field.boundary_values[1, lm_idx]
                spec_imag[local_lm, 1, inner_idx] = zero(T)
            end

            if has_outer && 1 <= outer_idx <= size(spec_real, 3) && field.bc_type_outer[lm_idx] == dirichlet
                spec_real[local_lm, 1, outer_idx] = field.boundary_values[2, lm_idx]
                spec_imag[local_lm, 1, outer_idx] = zero(T)
            end
        end
    end

    return field
end

"""
    apply_composition_boundary_conditions!(field; time_index=nothing)

Refresh composition boundary values from the bcs subsystem and
enforce Dirichlet data in spectral space.
"""
function apply_composition_boundary_conditions!(field::SHTnsCompositionField{T};
                                                 time_index::Union{Nothing,Int}=nothing) where T
    if field.boundary_condition_set === nothing
        return field
    end

    if time_index === nothing
        bcs.apply_composition_boundary_conditions!(field)
    else
        bcs.apply_composition_boundary_conditions!(field, time_index)
    end

    enforce_composition_boundary_values!(field)

    if field.domain.r[1, 4] == 0.0
        apply_ball_composition_regularity!(field)
    end

    return field
end

# ================================================================================
# Main Nonlinear Computation
# ================================================================================

"""
    compute_composition_nonlinear!(comp_field::SHTnsCompositionField{T},
                                   vel_fields, Dᵒᶜ::RadialDomain;
                                   geometry::Symbol = get_parameters().geometry) where T

Compute the nonlinear advection term -u·∇C for the composition equation.

This function implements the EXPLICIT part of the composition equation:
    ∂C/∂t + u·∇C = (Pm/Sc) ∇²C

The advection term -u·∇C is computed using the following optimized workflow:
1. Compute ∇C in spectral space (no MPI communication required)
2. Batched transform of C and ∇C to physical space
3. Compute advection -u·∇C locally in physical space
4. Add internal compositional sources
5. Transform result back to spectral space
6. Apply boundary conditions

# Arguments
- `comp_field`: Composition field structure to update
- `vel_fields`: Velocity field structure (provides u for advection)
- `Dᵒᶜ`: Radial domain information
- `geometry`: Geometry type (:shell or :ball) for transform selection

# Side Effects
- Updates `comp_field.nonlinear` with the computed advection term
- Updates `comp_field.gradient` with ∇C in physical space
- Applies boundary conditions to spectral representation

# Performance Notes
- Gradients computed in spectral space avoid MPI communication
- Batched transforms minimize data transfer overhead
- Physical space operations are embarrassingly parallel
"""
function compute_composition_nonlinear!(comp_field::SHTnsCompositionField{T},
                                        vel_fields, Dᵒᶜ::RadialDomain;
                                        geometry::Symbol = get_parameters().geometry) where T
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0

    # Zero work arrays
    zero_scalar_work_arrays!(comp_field)

    # Step 1: Compute ALL gradients in spectral space (NO COMMUNICATION!)
    t_spectral = MPI.Wtime()
    compute_all_gradients_spectral!(comp_field, Dᵒᶜ)
    comp_field.spectral_time[] += MPI.Wtime() - t_spectral

    # Step 2: Single batched transform of composition and gradients to physical
    t_transform = MPI.Wtime()
    transform_field_and_gradients_to_physical!(comp_field)
    comp_field.transform_time[] += MPI.Wtime() - t_transform

    # Step 3: Compute advection term -u·∇C in physical space (local operation)
    if vel_fields !== nothing
        compute_scalar_advection_local!(comp_field, vel_fields)
    end

    # Step 4: Add internal compositional sources (local operation)
    add_internal_sources_local!(comp_field, Dᵒᶜ)

    # Step 5: Transform advection + sources back to spectral space
    t_transform = MPI.Wtime()
    if geometry === :ball
        ball_physical_to_spectral!(comp_field.advection_physical, comp_field.nonlinear)
    else
        shtnskit_physical_to_spectral!(comp_field.advection_physical, comp_field.nonlinear)
    end
    comp_field.transform_time[] += MPI.Wtime() - t_transform

    # Step 6: Apply boundary conditions in spectral space
    apply_composition_boundary_conditions!(comp_field)
    apply_composition_boundary_conditions_spectral!(comp_field, Dᵒᶜ)

    if ENABLE_TIMING[]
        comp_field.computation_time[] += MPI.Wtime() - t_start
    end
end

# Backwards compatibility wrapper - uses shared implementation from scalar_field_common.jl
zero_composition_work_arrays!(comp_field::SHTnsCompositionField{T}) where T = zero_scalar_work_arrays!(comp_field)

# ================================================================================
# NOTE: Gradient computation functions moved to scalar_field_common.jl
# NOTE: Batched transform operations moved to scalar_field_common.jl
# NOTE: Internal source functions use add_internal_sources_local! from scalar_field_common.jl
# ================================================================================

# ================================================================================
# Boundary conditions in spectral space
# ================================================================================

function apply_composition_boundary_conditions_spectral!(comp_field::SHTnsCompositionField{T}, 
                                                         domain::RadialDomain) where T
    # Apply boundary conditions in spectral space
    
    spec_real = parent(comp_field.spectral.data_real)
    spec_imag = parent(comp_field.spectral.data_imag)
    
    lm_range = range_local(comp_field.config.pencils.spec, 1)
    r_range  = range_local(comp_field.config.pencils.spec, 3)
    
    # Check which boundaries are local
    has_inner = 1 in r_range
    has_outer = domain.N in r_range
    
    if has_inner || has_outer
        for (local_lm, lm_idx) in enumerate(lm_range)
            bc_inner = comp_field.bc_type_inner[lm_idx]
            bc_outer = comp_field.bc_type_outer[lm_idx]
            
            # Apply inner boundary condition (skip at r=0 for ball)
            if has_inner && domain.r[1,4] > 0
                r_local = 1
                if bc_inner == Int(DIRICHLET)  # Fixed composition
                    spec_real[local_lm, 1, r_local] = comp_field.boundary_values[1, lm_idx]
                    spec_imag[local_lm, 1, r_local] = 0.0  # Fixed values are real
                elseif bc_inner == Int(NEUMANN)  # No-flux (zero gradient)
                    # Defer to full no-flux enforcement after loop
                end
            end
            
            # Apply outer boundary condition
            if has_outer
                r_local = domain.N - first(r_range) + 1
                if r_local <= size(spec_real, 3)
                    if bc_outer == Int(DIRICHLET)  # Fixed composition
                        spec_real[local_lm, 1, r_local] = comp_field.boundary_values[2, lm_idx]
                        spec_imag[local_lm, 1, r_local] = 0.0  # Fixed values are real
                    elseif bc_outer == Int(NEUMANN)  # No-flux (zero gradient)
                        # Defer to full no-flux enforcement after loop
                    end
                end
            end
        end
    end
    # Apply flux boundary conditions using the common scalar field implementation
    if any(comp_field.bc_type_inner .== Int(NEUMANN)) || any(comp_field.bc_type_outer .== Int(NEUMANN))
        # Use the robust tau method for flux boundary conditions
        apply_scalar_flux_bc_spectral!(comp_field, domain; method=:tau)
    end

    if domain.r[1, 4] == 0.0
        apply_ball_composition_regularity!(comp_field)
    end
end

# ================================================================================
# Flux Boundary Conditions - NOW USING scalar_field_common.jl
# ================================================================================
# The compositional field now uses the same robust boundary condition methods
# as the thermal field (tau method, influence matrix, direct) via the common 
# apply_scalar_flux_bc_spectral! function.
#
# This provides:
# - Prescribed flux values (not just zero-flux) 
# - Multiple robust enforcement methods (tau, influence matrix, direct)
# - Consistent implementation across all scalar fields
# - Superior numerical accuracy compared to the old matrix-based approach

# ================================================================================
# Validation and Testing
# ================================================================================

function validate_composition_field(comp_field::SHTnsCompositionField{T}, domain::RadialDomain) where T
    """
    Validate composition field consistency and boundary conditions
    """
    errors = String[]
    
    # Check spectral field dimensions
    spec_real = parent(comp_field.spectral.data_real)
    if size(spec_real, 1) != comp_field.config.nlm
        push!(errors, "Spectral field nlm dimension mismatch")
    end
    
    # Check boundary condition arrays
    if length(comp_field.bc_type_inner) != comp_field.config.nlm
        push!(errors, "Inner BC array size mismatch")
    end
    
    if length(comp_field.bc_type_outer) != comp_field.config.nlm
        push!(errors, "Outer BC array size mismatch")
    end
    
    # Check internal sources
    if length(comp_field.internal_sources) != domain.N
        push!(errors, "Internal sources array size mismatch")
    end
    
    if !isempty(errors)
        error("Composition field validation failed:\n" * join(errors, "\n"))
    end
    
    return true
end

# ================================================================================
# Diagnostic functions
# ================================================================================

function compute_composition_rms(comp_field::SHTnsCompositionField{T}, Dᵒᶜ::RadialDomain) where T
    # Compute RMS composition
    spec_real = parent(comp_field.spectral.data_real)
    spec_imag = parent(comp_field.spectral.data_imag)
    
    local_sum = zero(T)
    
    # Sum over local spectral modes
    for r_idx in axes(spec_real, 3)
        for lm_idx in axes(spec_real, 1)
            val_real = spec_real[lm_idx, 1, r_idx]
            val_imag = spec_imag[lm_idx, 1, r_idx]
            local_sum += val_real^2 + val_imag^2
        end
    end
    
    # Global reduction
    comm = get_comm()
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, comm)
    
    return sqrt(global_sum / (Dᵒᶜ.N * comp_field.config.nlm))
end

function compute_composition_energy(comp_field::SHTnsCompositionField{T}, Dᵒᶜ::RadialDomain) where T
    # Compute compositional energy ∫ C² dV
    
    # Transform to physical space first
    shtnskit_spectral_to_physical!(comp_field.spectral, comp_field.work_physical)
    
    comp_data = parent(comp_field.work_physical.data)
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
    
    return global_energy / (comp_field.config.nlat * comp_field.config.nlon * Dᵒᶜ.N)
end

# ================================================================================
# Performance monitoring and statistics
# ================================================================================

function get_composition_statistics(comp_field::SHTnsCompositionField{T}, 
                                   domain::RadialDomain) where T
    """
    Compute various composition field statistics
    """
    # Min/max composition
    comp_data = parent(comp_field.composition.data)
    local_min = minimum(comp_data)
    local_max = maximum(comp_data)

    global_min = MPI.Allreduce(local_min, MPI.MIN, get_comm())
    global_max = MPI.Allreduce(local_max, MPI.MAX, get_comm())

    # RMS composition
    local_sum = sum(comp_data.^2)
    local_count = length(comp_data)

    global_sum = MPI.Allreduce(local_sum, MPI.SUM, get_comm())
    global_count = MPI.Allreduce(local_count, MPI.SUM, get_comm())
    
    rms_comp = sqrt(global_sum / global_count)
    
    # Total energy
    energy = compute_composition_energy(comp_field, domain)
    
    return (min = global_min,
            max = global_max,
            rms = rms_comp,
            energy = energy)
end

# ================================================================================
# Utility functions
# ================================================================================

function set_composition_ic!(comp_field::SHTnsCompositionField{T}, 
                             ic_type::Symbol, Dᵒᶜ::RadialDomain) where T
    # Set initial conditions for composition field
    
    spec_real = parent(comp_field.spectral.data_real)
    spec_imag = parent(comp_field.spectral.data_imag)
    
    if ic_type == :uniform
        # Uniform composition (only l=0, m=0 mode)
        fill!(spec_real, zero(T))
        fill!(spec_imag, zero(T))
        
        # Set uniform value in l=0, m=0 mode
        for r_idx in axes(spec_real, 3)
            spec_real[1, 1, r_idx] = 1.0  # Uniform composition = 1
        end
        
    elseif ic_type == :linear
        # Linear profile in radius
        for r_idx in axes(spec_real, 3)
            r = Dᵒᶜ.r[r_idx, 4]  # Normalized radius
            spec_real[1, 1, r_idx] = r  # Linear in radius
        end
        
    elseif ic_type == :random_perturbation
        # Small random perturbation on all modes
        fill!(spec_real, zero(T))
        fill!(spec_imag, zero(T))
        
        amplitude = 1e-3
        for r_idx in axes(spec_real, 3)
            for lm_idx in axes(spec_real, 1)
                if lm_idx > 1  # Don't perturb l=0, m=0 mode
                    spec_real[lm_idx, 1, r_idx] = amplitude * (rand(T) - 0.5)
                    spec_imag[lm_idx, 1, r_idx] = amplitude * (rand(T) - 0.5)
                end
            end
        end
    end
end

# Boundary condition setup
function set_composition_boundary_conditions!(comp_field::SHTnsCompositionField{T}, 
                                              bc_inner::Symbol, bc_outer::Symbol,
                                              value_inner::T = zero(T), value_outer::T = zero(T)) where T
    # Set boundary condition types and values
    
    if bc_inner == :fixed
        fill!(comp_field.bc_type_inner, Int(DIRICHLET))
        comp_field.boundary_values[1, :] .= value_inner
    elseif bc_inner == :no_flux
        fill!(comp_field.bc_type_inner, Int(NEUMANN))
    end

    if bc_outer == :fixed
        fill!(comp_field.bc_type_outer, Int(DIRICHLET))
        comp_field.boundary_values[2, :] .= value_outer
    elseif bc_outer == :no_flux
        fill!(comp_field.bc_type_outer, Int(NEUMANN))
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
