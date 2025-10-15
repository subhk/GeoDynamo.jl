# ================================================================================
# Composition field components with enhanced SHTns transforms
# ================================================================================
# For no-flux boundary conditions (typical for composition):
# bc_type_inner = fill(2, nlm)  # No-flux at inner boundary  
# bc_type_outer = fill(2, nlm)  # No-flux at outer boundary
# apply_composition_boundary_conditions_spectral!(comp_field, domain)
# ================================================================================

# ================================================================================
# Compositional field components with full spectral optimization
# ================================================================================

using PencilArrays
using SHTnsKit
using LinearAlgebra
using SparseArrays

import .BoundaryConditions
import .GeodynamoBall

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
    
    # Pre-computed coefficients
    l_factors::Vector{Float64}         # l(l+1) values for diffusion
    config::SHTnsKitConfig             # SHTnsKit configuration
    
    # Radial derivative matrices
    dr_matrix::BandedMatrix{T}        # First derivative d/dr
    d2r_matrix::BandedMatrix{T}       # Second derivative d²/dr²
    
    # Spectral derivative operators (pre-computed)
    theta_derivative_matrix::Matrix{T}     # θ derivative coupling
    theta_recurrence_coeffs::Matrix{T}     # Recurrence relations
    
    # Performance tracking
    computation_time::Ref{Float64}
    transform_time::Ref{Float64}
    boundary_condition_set::Union{BoundaryConditions.BoundaryConditionSet{T}, Nothing}
    boundary_interpolation_cache::Dict{String, Any}
    boundary_time_index::Ref{Int}
    domain::RadialDomain
end

# Specialization for composition field
get_main_physical_field(field::SHTnsCompositionField{T}) where T = field.composition

# ================================================================================
# Main nonlinear computation with full spectral optimization  
# ================================================================================
# NOTE: Pre-computation functions moved to scalar_field_common.jl

function create_shtns_composition_field(::Type{T}, config::SHTnsKitConfig, 
                                        oc_domain::RadialDomain) where T
    # Use config's pencils directly
    pencils = config.pencils
    
    # Composition field in r-pencil for efficient radial operations
    composition = create_shtns_physical_field(T, config, oc_domain, pencils.r)
    
    # Gradient components
    gradient = create_shtns_vector_field(T, config, oc_domain, 
                                        (pencils.θ, pencils.φ, pencils.r))
    
    # Spectral representation in spectral pencil for efficient transforms
    spectral = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    nonlinear = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    prev_nonlinear = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    
    # Work arrays
    work_spectral      = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    work_physical      = create_shtns_physical_field(T, config, oc_domain, pencils.r)
    advection_physical = create_shtns_physical_field(T, config, oc_domain, pencils.r)
    
    # Gradient spectral components
    grad_theta_spec = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    grad_phi_spec   = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    grad_r_spec     = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    
    # Sources and boundary conditions
    internal_sources = zeros(T, oc_domain.N)
    boundary_values  = zeros(T, 2, config.nlm)
    
    # Default boundary conditions (1 = fixed value, 2 = flux)
    # For composition: typically no-flux at both boundaries
    bc_type_inner = fill(2, config.nlm)  # No-flux at inner boundary
    bc_type_outer = fill(2, config.nlm)  # No-flux at outer boundary
    
    # Pre-compute l(l+1) factors for diffusion operator
    l_factors = zeros(Float64, config.nlm)
    for lm_idx in 1:config.nlm
        l = config.l_values[lm_idx]
        l_factors[lm_idx] = Float64(l * (l + 1))
    end
    
    # Transform manager removed in SHTnsKit migration
    
    # Create radial derivative matrices
    dr_matrix  = create_derivative_matrix(1, oc_domain)
    d2r_matrix = create_derivative_matrix(2, oc_domain)
    
    # Pre-compute spectral derivative operators
    theta_derivative_matrix = build_theta_derivative_matrix(T, config)
    theta_recurrence_coeffs = compute_theta_recurrence_coefficients(T, config)
    
    boundary_condition_set = nothing
    boundary_cache = Dict{String, Any}()
    boundary_time_index = Ref{Int}(1)

    return SHTnsCompositionField{T}(
        composition, gradient, spectral, nonlinear, prev_nonlinear,
        work_spectral, work_physical, advection_physical,
        grad_theta_spec, grad_phi_spec, grad_r_spec,
        internal_sources, boundary_values,
        bc_type_inner, bc_type_outer,
        l_factors, config,
        dr_matrix, d2r_matrix,
        theta_derivative_matrix, theta_recurrence_coeffs,
        Ref{Float64}(0.0), Ref{Float64}(0.0),
        boundary_condition_set, boundary_cache, boundary_time_index,
        oc_domain
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

    dirichlet = Int(BoundaryConditions.DIRICHLET)

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

Refresh composition boundary values from the BoundaryConditions subsystem and
enforce Dirichlet data in spectral space.
"""
function apply_composition_boundary_conditions!(field::SHTnsCompositionField{T};
                                                 time_index::Union{Nothing,Int}=nothing) where T
    if field.boundary_condition_set === nothing
        return field
    end

    if time_index === nothing
        BoundaryConditions.apply_composition_boundary_conditions!(field)
    else
        BoundaryConditions.apply_composition_boundary_conditions!(field, time_index)
    end

    enforce_composition_boundary_values!(field)

    if field.domain.r[1, 4] == 0.0
        GeodynamoBall.apply_ball_composition_regularity!(field)
    end

    return field
end

function compute_composition_nonlinear!(comp_field::SHTnsCompositionField{T}, 
                                        vel_fields, oc_domain::RadialDomain; 
                                        geometry::Symbol = get_parameters().geometry) where T
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0
    
    # Zero work arrays
    zero_scalar_work_arrays!(comp_field)
    
    # Step 1: Transform composition to physical space for advection
    t_transform = MPI.Wtime()
    shtnskit_spectral_to_physical!(comp_field.spectral, comp_field.composition)
    comp_field.transform_time[] += MPI.Wtime() - t_transform
    
    # Step 2: Compute gradient in physical space if needed for diffusion
    compute_composition_gradient_local!(comp_field, oc_domain)
    
    # Step 3: Compute advection term -u·∇C in physical space (local operation)
    if vel_fields !== nothing
        compute_scalar_advection_local!(comp_field, vel_fields)
    end
    
    # Step 4: Add internal compositional sources (local operation)
    add_internal_sources_local!(comp_field, oc_domain)
    
    # Step 5: Transform advection + sources back to spectral space
    t_transform = MPI.Wtime()
    if geometry === :ball
        GeodynamoBall.ball_physical_to_spectral!(comp_field.advection_physical, comp_field.nonlinear)
    else
        shtnskit_physical_to_spectral!(comp_field.advection_physical, comp_field.nonlinear)
    end
    comp_field.transform_time[] += MPI.Wtime() - t_transform
    
    # Step 6: Apply boundary conditions in spectral space
    apply_composition_boundary_conditions!(comp_field)
    apply_composition_boundary_conditions_spectral!(comp_field, oc_domain)
    
    if ENABLE_TIMING[]
        comp_field.computation_time[] += MPI.Wtime() - t_start
    end
end

function zero_composition_work_arrays!(comp_field::SHTnsCompositionField{T}) where T
    fill!(parent(comp_field.work_spectral.data_real), zero(T))
    fill!(parent(comp_field.work_spectral.data_imag), zero(T))
    fill!(parent(comp_field.work_physical.data), zero(T))
    fill!(parent(comp_field.advection_physical.data), zero(T))
    fill!(parent(comp_field.nonlinear.data_real), zero(T))
    fill!(parent(comp_field.nonlinear.data_imag), zero(T))
end

function compute_composition_gradient_local!(comp_field::SHTnsCompositionField{T}, oc_domain::RadialDomain) where T
    """
    Complete composition gradient computation (θ, φ, r) in spectral space
    This is completely local - no MPI communication required
    """
    # 1. Compute spectral gradients (local operation)
    compute_all_gradients_spectral!(comp_field, oc_domain)
    
    # 2. Transform all components to physical space (batched operation)
    transform_field_and_gradients_to_physical!(comp_field)
end

# ================================================================================
# NOTE: Gradient computation functions moved to scalar_field_common.jl
# ================================================================================
# NOTE: Batched transform operations moved to scalar_field_common.jl
# ================================================================================
# Local physical space operations (no communication)
# ================================================================================


function add_compositional_sources_local!(comp_field::SHTnsCompositionField{T}, 
                                          oc_domain::RadialDomain) where T
    # Add volumetric compositional sources (completely local operation)
    
    adv_data = parent(comp_field.advection_physical.data)
    
    if !all(iszero, comp_field.internal_sources)
        # Get local physical dimensions
        nlat, nlon, nr_local = size(adv_data)
        
        # Add sources (assumes sources are radially symmetric)
        for r_local in 1:nr_local
            for φ_idx in 1:nlon
                for θ_idx in 1:nlat
                    if r_local <= length(comp_field.internal_sources)
                        adv_data[θ_idx, φ_idx, r_local] += comp_field.internal_sources[r_local]
                    end
                end
            end
        end
    end
end

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
                if bc_inner == 1  # Fixed composition
                    spec_real[local_lm, 1, r_local] = comp_field.boundary_values[1, lm_idx]
                    spec_imag[local_lm, 1, r_local] = 0.0  # Fixed values are real
                elseif bc_inner == 2  # No-flux (zero gradient)
                    # Defer to full no-flux enforcement after loop
                end
            end
            
            # Apply outer boundary condition
            if has_outer
                r_local = domain.N - first(r_range) + 1
                if r_local <= size(spec_real, 3)
                    if bc_outer == 1  # Fixed composition
                        spec_real[local_lm, 1, r_local] = comp_field.boundary_values[2, lm_idx]
                        spec_imag[local_lm, 1, r_local] = 0.0  # Fixed values are real
                    elseif bc_outer == 2  # No-flux (zero gradient)
                        # Defer to full no-flux enforcement after loop
                    end
                end
            end
        end
    end
    # Apply flux boundary conditions using the common scalar field implementation
    if any(comp_field.bc_type_inner .== 2) || any(comp_field.bc_type_outer .== 2)
        # Use the robust tau method for flux boundary conditions
        apply_scalar_flux_bc_spectral!(comp_field, domain; method=:tau)
    end

    if domain.r[1, 4] == 0.0
        GeodynamoBall.apply_ball_composition_regularity!(comp_field)
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

function compute_composition_rms(comp_field::SHTnsCompositionField{T}, oc_domain::RadialDomain) where T
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
    
    return sqrt(global_sum / (oc_domain.N * comp_field.config.nlm))
end

function compute_composition_energy(comp_field::SHTnsCompositionField{T}, oc_domain::RadialDomain) where T
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
    
    return global_energy / (comp_field.config.nlat * comp_field.config.nlon * oc_domain.N)
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
                             ic_type::Symbol, oc_domain::RadialDomain) where T
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
            r = oc_domain.r[r_idx, 4]  # Normalized radius
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
        fill!(comp_field.bc_type_inner, 1)
        comp_field.boundary_values[1, :] .= value_inner
    elseif bc_inner == :no_flux
        fill!(comp_field.bc_type_inner, 2)
    end
    
    if bc_outer == :fixed  
        fill!(comp_field.bc_type_outer, 1)
        comp_field.boundary_values[2, :] .= value_outer
    elseif bc_outer == :no_flux
        fill!(comp_field.bc_type_outer, 2)
    end
end

# Note: NetCDF boundary condition functions moved to src/BoundaryConditions/composition.jl

# ================================================================================
# Export functions
# ================================================================================
# export SHTnsCompositionField, create_shtns_composition_field
# export compute_composition_nonlinear!
# export compute_composition_rms, compute_composition_energy
# export get_composition_statistics
# export zero_composition_work_arrays!
# export set_composition_ic!, set_composition_boundary_conditions!

# Note: File-based boundary condition functions moved to src/BoundaryConditions/composition.jl

# Note: Boundary condition exports moved to src/BoundaryConditions/composition.jl
