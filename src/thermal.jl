# ================================================================================
# Temperature field components with enhanced SHTns transforms
# ================================================================================
# For uniform heating from below (l=0, m=0 mode has flux BC)
# Other modes have fixed temperature
# bc_type_inner = ones(Int, nlm)
# bc_type_outer = ones(Int, nlm)
# bc_type_inner[1] = 2  # Flux BC for l=0, m=0 at inner boundary
# bc_type_outer[1] = 2  # Flux BC for l=0, m=0 at outer boundary
# apply_mixed_boundary_conditions!(temp_field, domain, bc_type_inner, bc_type_outer)
# ================================================================================

# ================================================================================
# Temperature/Thermal field components with full spectral optimization
# ================================================================================

using PencilArrays
using SHTnsKit
using LinearAlgebra
using SparseArrays

# scalar_field_common.jl is included in main module - functions are available here

struct SHTnsTemperatureField{T} <: AbstractScalarField{T}
    # Physical space temperature
    temperature::SHTnsPhysicalField{T}
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
    internal_sources::Vector{T}        # Radial profile of heating
    boundary_values::Matrix{T}         # [2, nlm] for ICB and CMB
    bc_type_inner::Vector{Int}         # BC type for each mode at inner
    bc_type_outer::Vector{Int}         # BC type for each mode at outer
    
    # File-based boundary condition support
    boundary_condition_set::Union{Dict{String, Any}, Nothing}        # Loaded boundary conditions
    boundary_interpolation_cache::Dict{String, Any}                  # Cached interpolated data
    boundary_time_index::Ref{Int}                                    # Current time index for time-dependent BCs
    
    # Pre-computed coefficients
    l_factors::Vector{Float64}         # l(l+1) values
    
    # Configuration (SHTnsKit)
    config::SHTnsKitConfig
    
    # Radial derivative matrices
    dr_matrix::BandedMatrix{T}
    d2r_matrix::BandedMatrix{T}
    
    # Spectral derivative operators
    theta_derivative_matrix::SparseMatrixCSC{T,Int}  # Pre-computed θ-derivative
    theta_recurrence_coeffs::Matrix{T}               # Recurrence coefficients
    
    # Performance tracking
    computation_time::Ref{Float64}
    transform_time::Ref{Float64}
    comm_time::Ref{Float64}
    spectral_time::Ref{Float64}
end

# Specialization for temperature field (moved after struct definition)
get_main_physical_field(field::SHTnsTemperatureField{T}) where T = field.temperature

function create_shtns_temperature_field(::Type{T}, config::SHTnsKitConfig, 
                                        oc_domain::RadialDomain) where T
    # Use config's pencils directly
    pencils = config.pencils
    
    # Temperature field in r-pencil for efficient radial operations
    temperature = create_shtns_physical_field(T, config, oc_domain, pencils.r)
    
    # Gradient components
    gradient = create_shtns_vector_field(T, config, oc_domain, 
                                        (pencils.θ, pencils.φ, pencils.r))
    
    # Spectral representation using spectral pencil
    spectral  = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
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
    
    # Default BC types (1 = Dirichlet, 2 = Neumann)
    bc_type_inner = ones(Int, config.nlm)  # Default to fixed temperature
    bc_type_outer = ones(Int, config.nlm)
    
    # Storage for file-based boundary conditions
    boundary_data_cache = Dict{String, Any}()
    
    # Pre-compute l(l+1) factors
    l_factors = Float64[l * (l + 1) for l in config.l_values]
    
    # Transform manager removed in SHTnsKit migration
    
    # Create radial derivative matrices
    dr_matrix  = create_derivative_matrix(1, oc_domain)
    d2r_matrix = create_derivative_matrix(2, oc_domain)
    
    # Pre-compute spectral derivative operators
    theta_derivative_matrix = build_theta_derivative_matrix(T, config)
    theta_recurrence_coeffs = compute_theta_recurrence_coefficients(T, config)
    
    return SHTnsTemperatureField{T}(
        temperature, gradient, spectral, nonlinear, prev_nonlinear,
        work_spectral, work_physical, advection_physical,
        grad_theta_spec, grad_phi_spec, grad_r_spec,
        internal_sources, boundary_values,
        bc_type_inner, bc_type_outer,
        nothing, Dict{String, Any}(), Ref(1),  # boundary condition fields
        l_factors, config,
        dr_matrix, d2r_matrix,
        theta_derivative_matrix, theta_recurrence_coeffs,
        Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0)
    )
end

# ================================================================================
# Main nonlinear computation with full spectral optimization
# ================================================================================
function compute_temperature_nonlinear!(temp_field::SHTnsTemperatureField{T}, 
                                        vel_fields, oc_domain::RadialDomain; 
                                        geometry::Symbol = get_parameters().geometry) where T
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0
    
    # Zero work arrays
    zero_scalar_work_arrays!(temp_field)
    
    # Step 1: Compute ALL gradients in spectral space (NO COMMUNICATION!)
    t_spectral = MPI.Wtime()
    compute_all_gradients_spectral!(temp_field, oc_domain)
    temp_field.spectral_time[] += MPI.Wtime() - t_spectral
    
    # Step 2: Single batched transform of temperature and gradients to physical
    t_transform = MPI.Wtime()
    transform_field_and_gradients_to_physical!(temp_field)
    temp_field.transform_time[] += MPI.Wtime() - t_transform
    
    # Step 3: Compute advection term -u·∇T in physical space (local operation)
    if vel_fields !== nothing
        compute_scalar_advection_local!(temp_field, vel_fields)
    end
    
    # Step 4: Add internal heat sources (local operation)
    add_internal_sources_local!(temp_field, oc_domain)
    
    # Step 5: Transform advection + sources back to spectral space
    t_transform = MPI.Wtime()
    if geometry === :ball
        GeodynamoBall.ball_physical_to_spectral!(temp_field.advection_physical, temp_field.nonlinear)
    else
        shtnskit_physical_to_spectral!(temp_field.advection_physical, temp_field.nonlinear)
    end
    temp_field.transform_time[] += MPI.Wtime() - t_transform
    
    # Step 6: Apply boundary conditions in spectral space
    apply_temperature_boundary_conditions_spectral!(temp_field, oc_domain)
    
    if ENABLE_TIMING[]
        temp_field.computation_time[] += MPI.Wtime() - t_start
    end
end

# ================================================================================
# Fully spectral gradient computation (NO COMMUNICATION!)
# ================================================================================
# NOTE: Gradient computation functions moved to scalar_field_common.jl
# NOTE: Batched transform operations moved to scalar_field_common.jl
# ================================================================================

# ================================================================================
# Local physical space operations (no communication)
# ================================================================================
function compute_temperature_advection_local!(temp_field::SHTnsTemperatureField{T}, 
                                             vel_fields) where T
    """
    Compute -u·∇T in physical space (completely local operation)
    """
    u_r = parent(vel_fields.velocity.r_component.data)
    u_θ = parent(vel_fields.velocity.θ_component.data)
    u_φ = parent(vel_fields.velocity.φ_component.data)
    
    grad_r = parent(temp_field.gradient.r_component.data)
    grad_θ = parent(temp_field.gradient.θ_component.data)
    grad_φ = parent(temp_field.gradient.φ_component.data)
    
    advection = parent(temp_field.advection_physical.data)
    
    @inbounds @simd for idx in eachindex(advection)
        if idx <= length(u_r) && idx <= length(grad_r)
            advection[idx] = -(u_r[idx] * grad_r[idx] + 
                              u_θ[idx] * grad_θ[idx] + 
                              u_φ[idx] * grad_φ[idx])
        end
    end
end

function add_internal_sources_local!(temp_field::SHTnsTemperatureField{T}, 
                                    domain::RadialDomain) where T
    """
    Add volumetric heating (completely local operation)
    """
    advection = parent(temp_field.advection_physical.data)
    
    if !all(iszero, temp_field.internal_sources)
        # Get local physical dimensions
        local_shape = size(temp_field.advection_physical.data)
        nlat_local, nlon_local, nr_local = local_shape
        
        r_range = range_local(temp_field.config.pencils.r, 3)
        
        @inbounds for k in 1:nr_local
            r_idx = k + first(r_range) - 1
            if r_idx <= length(temp_field.internal_sources)
                source_value = temp_field.internal_sources[r_idx]
                
                # Add uniformly at this radius
                @simd for j in 1:nlon_local
                    for i in 1:nlat_local
                        idx = i + (j-1)*nlat_local + (k-1)*nlat_local*nlon_local
                        if idx <= length(advection)
                            advection[idx] += source_value
                        end
                    end
                end
            end
        end
    end
end

# ================================================================================
# Boundary conditions in spectral space
# ================================================================================
function apply_temperature_boundary_conditions_spectral!(temp_field::SHTnsTemperatureField{T}, 
                                                        domain::RadialDomain) where T
    """
    Apply boundary conditions in spectral space
    """
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    r_range  = range_local(temp_field.config.pencils.spec, 3)
    
    # Check which boundaries are local
    has_inner = 1 in r_range
    has_outer = domain.N in r_range
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Inner boundary (skip at r=0 for ball geometry)
            if has_inner && domain.r[1, 4] > 0
                if temp_field.bc_type_inner[lm_idx] == 1    # Dirichlet
                    local_r = 1 - first(r_range) + 1
                    spec_real[local_lm, 1, local_r] = temp_field.boundary_values[1, lm_idx]
                    spec_imag[local_lm, 1, local_r] = 0.0
                elseif temp_field.bc_type_inner[lm_idx] == 2  # Neumann
                    # Defer to full flux BC application after loop
                    # (handled by apply_flux_bc_spectral!(temp_field, domain))
                end
            end
            
            # Outer boundary
            if has_outer
                if temp_field.bc_type_outer[lm_idx] == 1      # Dirichlet
                    local_r = domain.N - first(r_range) + 1
                    spec_real[local_lm, 1, local_r] = temp_field.boundary_values[2, lm_idx]
                    spec_imag[local_lm, 1, local_r] = 0.0
                elseif temp_field.bc_type_outer[lm_idx] == 2  # Neumann
                    # Defer to full flux BC application after loop
                end
            end
        end
    end
    # If any Neumann BCs present, apply the complete spectral flux BC correction
    if any(temp_field.bc_type_inner .== 2) || any(temp_field.bc_type_outer .== 2)
        apply_flux_bc_spectral!(temp_field, domain)
    end

    if domain.r[1, 4] == 0.0
        GeodynamoBall.apply_ball_temperature_regularity!(temp_field)
    end
end


# ================================================================================
# Complete Flux Boundary Condition Implementation for Spectral Methods
# ================================================================================

"""
    apply_flux_bc_spectral_complete!(temp_field, domain)

Complete implementation of flux boundary conditions in spectral space.
This modifies the spectral coefficients to satisfy ∂T/∂r = prescribed_flux.
"""
function apply_flux_bc_spectral!(temp_field::SHTnsTemperatureField{T}, 
                                         domain::RadialDomain) where T
    # Use the common flux BC implementation with tau method (most robust)
    apply_scalar_flux_bc_spectral!(temp_field, domain; method=:tau)
end

# ================================================================================
# Boundary Condition Implementation - MOVED TO scalar_field_common.jl
# ================================================================================
# All flux boundary condition methods (tau, influence matrix, direct) have been
# moved to scalar_field_common.jl to be shared between thermal and compositional fields.
# The functions are now generic and work with AbstractScalarField.

# Validation and Testing
# ================================================================================

function validate_flux_bc(temp_field, domain)
    """
    Check if flux boundary conditions are satisfied within tolerance.
    """
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    
    max_error = 0.0
    
    for lm_idx in lm_range
        if lm_idx <= temp_field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Check inner boundary
            if temp_field.bc_type_inner[lm_idx] == 2
                prescribed = get_flux_value(lm_idx, 1, temp_field)
                actual = compute_flux_at_boundary(spec_real, spec_imag, local_lm, 
                                                 1, temp_field, domain)
                error = abs(prescribed - actual)
                max_error = max(max_error, error)
            end
            
            # Check outer boundary
            if temp_field.bc_type_outer[lm_idx] == 2
                prescribed = get_flux_value(lm_idx, 2, temp_field)
                actual = compute_flux_at_boundary(spec_real, spec_imag, local_lm,
                                                 domain.N, temp_field, domain)
                error = abs(prescribed - actual)
                max_error = max(max_error, error)
            end
        end
    end
    
    # Global maximum error
    global_max_error = MPI.Allreduce(max_error, MPI.MAX, get_comm())
    
    if get_rank() == 0
        println("Maximum flux BC error: $(global_max_error)")
        if global_max_error > 1e-6
            println("Warning: Flux BC error exceeds tolerance")
        end
    end
    
    return global_max_error
end


# ================================================================================
# Diagnostic functions
# ================================================================================
function compute_nusselt_number(temp_field::SHTnsTemperatureField{T}, 
                               domain::RadialDomain) where T
    """
    Compute Nusselt number from heat flux at boundaries
    """
    # Compute heat flux from radial gradient
    grad_r = temp_field.gradient.r_component
    
    # Get flux at boundaries (requires communication)
    flux_inner = compute_surface_flux(grad_r, 1, temp_field.config)
    flux_outer = compute_surface_flux(grad_r, oc_domain.N, temp_field.config)
    
    # Nusselt number
    conductive_flux = 4π * oc_domain.r[1, 4]^2
    Nu = abs(flux_outer) / conductive_flux
    
    return Nu
end


function compute_thermal_energy(temp_field::SHTnsTemperatureField{T}) where T
    """
    Compute total thermal energy in spectral space
    """
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    # Local energy computation
    local_energy = 0.0
    
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    r_range  = range_local(temp_field.config.pencils.spec, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(spec_real, 3)
                    local_energy += (spec_real[local_lm, 1, local_r]^2 + 
                                   spec_imag[local_lm, 1, local_r]^2)
                end
            end
        end
    end
    
    # Global sum across all processes
    return 0.5 * MPI.Allreduce(local_energy, MPI.SUM, get_comm())
end


function compute_surface_flux(field::SHTnsPhysicalField{T}, r_level::Int, 
                             config::SHTnsKitConfig) where T
    """
    Compute surface integral of flux at given radial level
    """
    data = parent(field.data)
    
    # Local contribution
    local_flux = 0.0
    
    # Get local range
    local_range = range_local(config.pencils.r)
    θ_range, φ_range, r_range = local_range
    
    if r_level in r_range
        local_r = r_level - first(r_range) + 1
        
        for φ_idx in φ_range, θ_idx in θ_range
            if θ_idx <= config.nlat && φ_idx <= config.nlon
                local_θ = θ_idx - first(θ_range) + 1
                local_φ = φ_idx - first(φ_range) + 1
                
                idx = local_θ + (local_φ-1)*length(θ_range) + (local_r-1)*length(θ_range)*length(φ_range)
                
                if idx <= length(data)
                    # Use Gaussian quadrature weights
                    weight = config.gauss_weights[θ_idx] * (2π / config.nlon) * sin(config.theta_grid[θ_idx])
                    local_flux += data[idx] * weight
                end
            end
        end
    end
    
    # Global reduction
    return MPI.Allreduce(local_flux, MPI.SUM, get_comm())
end


# ================================================================================
# Performance monitoring and statistics
# ================================================================================
function get_temperature_statistics(temp_field::SHTnsTemperatureField{T}, 
                                   domain::RadialDomain) where T
    """
    Compute various temperature field statistics
    """
    # Min/max temperature
    temp_data = parent(temp_field.temperature.data)
    local_min = minimum(temp_data)
    local_max = maximum(temp_data)
    
    global_min = MPI.Allreduce(local_min, MPI.MIN, get_comm())
    global_max = MPI.Allreduce(local_max, MPI.MAX, get_comm())
    
    # RMS temperature
    local_sum = sum(temp_data.^2)
    local_count = length(temp_data)
    
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, get_comm())
    global_count = MPI.Allreduce(local_count, MPI.SUM, get_comm())
    
    rms_temp = sqrt(global_sum / global_count)
    
    # Nusselt number
    Nu = compute_nusselt_number(temp_field, domain)
    
    # Total energy
    energy = compute_thermal_energy(temp_field)
    
    return (min = global_min,
            max = global_max,
            rms = rms_temp,
            nusselt = Nu,
            energy = energy)
end

# ================================================================================
# Utility functions
# ================================================================================
function zero_temperature_work_arrays!(temp_field::SHTnsTemperatureField{T}) where T
    """
    Efficiently zero all work arrays
    """
    fill!(parent(temp_field.work_spectral.data_real), zero(T))
    fill!(parent(temp_field.work_spectral.data_imag), zero(T))
    fill!(parent(temp_field.work_physical.data), zero(T))
    fill!(parent(temp_field.advection_physical.data), zero(T))
    fill!(parent(temp_field.grad_theta_spec.data_real), zero(T))
    fill!(parent(temp_field.grad_theta_spec.data_imag), zero(T))
    fill!(parent(temp_field.grad_phi_spec.data_real), zero(T))
    fill!(parent(temp_field.grad_phi_spec.data_imag), zero(T))
    fill!(parent(temp_field.grad_r_spec.data_real), zero(T))
    fill!(parent(temp_field.grad_r_spec.data_imag), zero(T))
end

function set_temperature_ic!(temp_field::SHTnsTemperatureField{T}, 
                            domain::RadialDomain;
                            perturbation_amplitude::T = T(1e-3)) where T
    """
    Set initial condition for temperature field
    """
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    r_range = range_local(temp_field.config.pencils.spec, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l = temp_field.config.l_values[lm_idx]
            m = temp_field.config.m_values[lm_idx]
            
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(spec_real, 3)
                    r = domain.r[r_idx, 4]
                    
                    if l == 0 && m == 0
                        # Conductive profile for l=0, m=0
                        spec_real[local_lm, 1, local_r] = 1.0 - r
                        spec_imag[local_lm, 1, local_r] = 0.0
                    elseif l <= 4
                        # Small perturbation for low modes
                        spec_real[local_lm, 1, local_r] = perturbation_amplitude * randn(T)
                        if m > 0
                            spec_imag[local_lm, 1, local_r] = perturbation_amplitude * randn(T)
                        else
                            spec_imag[local_lm, 1, local_r] = 0.0
                        end
                    else
                        # Zero for high modes
                        spec_real[local_lm, 1, local_r] = 0.0
                        spec_imag[local_lm, 1, local_r] = 0.0
                    end
                end
            end
        end
    end
end

function set_boundary_conditions!(temp_field::SHTnsTemperatureField{T};
                                 inner_bc_type::Int = 1,
                                 outer_bc_type::Int = 1,
                                 inner_value::T = T(1.0),
                                 outer_value::T = T(0.0)) where T
    """
    Set boundary condition types and values
    """
    # Set BC types for all modes
    fill!(temp_field.bc_type_inner, inner_bc_type)
    fill!(temp_field.bc_type_outer, outer_bc_type)
    
    # Set boundary values for l=0, m=0 mode (mean temperature)
    l0m0_idx = get_mode_index(temp_field.config, 0, 0)
    if l0m0_idx > 0
        temp_field.boundary_values[1, l0m0_idx] = inner_value
        temp_field.boundary_values[2, l0m0_idx] = outer_value
    end
    
    # Other modes have zero boundary values by default
    for lm_idx in 2:temp_field.config.nlm
        temp_field.boundary_values[1, lm_idx] = T(0.0)
        temp_field.boundary_values[2, lm_idx] = T(0.0)
    end
end

function set_internal_heating!(temp_field::SHTnsTemperatureField{T}, 
                              domain::RadialDomain;
                              heating_type::Symbol = :uniform,
                              amplitude::T = T(1.0)) where T
    """
    Set internal heating profile
    """
    if heating_type == :uniform
        # Uniform volumetric heating
        fill!(temp_field.internal_sources, amplitude)
    elseif heating_type == :gaussian
        # Gaussian heating profile centered at mid-radius
        r_mid = 0.5 * (domain.r[1, 4] + domain.r[end, 4])
        sigma = 0.1 * (domain.r[end, 4] - domain.r[1, 4])
        
        for i in 1:domain.N
            r = domain.r[i, 4]
            temp_field.internal_sources[i] = amplitude * exp(-((r - r_mid)/sigma)^2)
        end
    elseif heating_type == :bottom
        # Heating concentrated near bottom
        for i in 1:domain.N
            r = domain.r[i, 4]
            r_norm = (r - domain.r[1, 4]) / (domain.r[end, 4] - domain.r[1, 4])
            temp_field.internal_sources[i] = amplitude * exp(-5.0 * r_norm)
        end
    else
        # No heating
        fill!(temp_field.internal_sources, zero(T))
    end
end

# Note: NetCDF boundary condition functions moved to src/BoundaryConditions/thermal.jl

# ================================================================================
# Export functions
# ================================================================================
# export SHTnsTemperatureField, create_shtns_temperature_field
# export compute_temperature_nonlinear!
# export compute_nusselt_number, compute_thermal_energy
# export compute_surface_flux, get_temperature_statistics
# export zero_temperature_work_arrays!
# export set_temperature_ic!, set_boundary_conditions!, set_internal_heating!


#export print_temperature_performance

# # Export functions
# export SHTnsTemperatureField, create_shtns_temperature_field
# export compute_temperature_nonlinear!, compute_temperature_batch!
# export zero_work_arrays!

# Note: File-based boundary condition functions moved to src/BoundaryConditions/thermal.jl

# Note: Boundary condition exports moved to src/BoundaryConditions/thermal.jl
