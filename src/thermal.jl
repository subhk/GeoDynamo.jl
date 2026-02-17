# ================================================================================
# Temperature Field Module with SHTns
# ================================================================================
#
# This module implements the temperature field evolution for geodynamo simulations
# using spherical harmonic transforms (SHTnsKit).
#
# REFERENCE: Sreenivasan & Kar (2018), Phys. Rev. Fluids 3, 093801
#            Equation (2): Temperature evolution
#
# ================================================================================
# GOVERNING EQUATION
# ================================================================================
#
# The non-dimensional temperature equation in magnetic diffusion time scaling:
#
#   ∂T/∂t + u·∇T = (Pm/Pr) ∇²T
#
# where:
#   T            : Temperature perturbation from conductive profile
#   u            : Velocity field
#   Pm = ν/η     : Magnetic Prandtl number
#   Pr = ν/κ     : Prandtl number
#   Pm/Pr = κ/η  : Ratio of thermal to magnetic diffusivity
#
# PHYSICAL INTERPRETATION:
# ========================
# - Left side: Time rate of change + advection by flow
# - Right side: Thermal diffusion
#
# The advection term -u·∇T represents heat transport by the convecting fluid.
# This is the EXPLICIT part computed in physical space.
#
# The diffusion term (Pm/Pr)∇²T is treated IMPLICITLY for numerical stability.
# The diffusivity coefficient passed to the time-stepper is (Pm/Pr) = d_Pm/d_Pr.
#
# ================================================================================
# BOUNDARY CONDITIONS
# ================================================================================
#
# Common configurations:
#   - Fixed temperature (Dirichlet): T = T_boundary at r = r_i, r_o
#   - Fixed flux (Neumann): ∂T/∂r = prescribed_flux at boundaries
#   - Mixed: Different types at inner/outer boundaries
#
# For uniform heating from below:
#   bc_type_inner[1] = NEUMANN   # Flux BC for l=0, m=0 at inner boundary
#   bc_type_outer[1] = NEUMANN   # Flux BC for l=0, m=0 at outer boundary
#   (Other modes typically use Dirichlet with zero boundary values)
#
# ================================================================================

using PencilArrays
using SHTnsKit
using LinearAlgebra
using SparseArrays

import .bcs
import .bcs: BoundaryType, DIRICHLET, NEUMANN

# scalar_field_common.jl is included in main module - functions are available here

mutable struct SHTnsTemperatureField{T} <: AbstractScalarField{T}
    # Physical space temperature
    temperature::SHTnsPhysField{T}
    gradient::SHTnsVectorField{T}

    # Spectral representation
    spectral::SHTnsSpecField{T}

    # Nonlinear terms (advection)
    nonlinear::SHTnsSpecField{T}
    prev_nonlinear::SHTnsSpecField{T}

    # Work arrays for efficient computation
    work_spectral::SHTnsSpecField{T}
    work_physical::SHTnsPhysField{T}
    advection_physical::SHTnsPhysField{T}

    # Sources and boundary conditions
    internal_sources::Vector{T}        # Radial profile of heating
    boundary_values::Matrix{T}         # [2, nlm] for ICB and CMB
    bc_type_inner::Vector{Int}         # BC type for each mode at inner
    bc_type_outer::Vector{Int}         # BC type for each mode at outer

    # File-based boundary condition support
    boundary_condition_set::Union{bcs.BoundaryConditionSet{T}, Nothing}  # Loaded boundary conditions
    boundary_interpolation_cache::Dict{String, Any}                  # Cached interpolated data
    boundary_time_index::Ref{Int}                                    # Current time index for time-dependent BCs
    
    # Pre-computed coefficients
    ℓ_factors::Vector{Float64}         # l(l+1) values
    
    # Configuration (SHTnsKit)
    config::SHTnsKitConfig
    
    # Radial derivative matrices
    ∂r::BandedMatrix{T}
    ∂²r::BandedMatrix{T}
    
    # Spectral derivative operators
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

# Specialization for temperature field (moved after struct definition)
get_main_physical_field(𝔽::SHTnsTemperatureField{T}) where T = 𝔽.temperature

function create_shtns_temperature_field(::Type{T}, config::SHTnsKitConfig,
                                        𝒟ᵒᶜ::RadialDomain,
                                        pencils=nothing, pencil_spec=nothing) where T
    # Use config's pencils by default (consistent with velocity/magnetic creators)
    if pencils === nothing
        pencils = config.pencils
    end
    if pencil_spec === nothing
        pencil_spec = pencils.spec
    end
    
    # Temperature field in r-pencil for efficient radial operations
    temperature = create_shtns_physical_field(T, config, 𝒟ᵒᶜ, pencils.r)
    
    # Gradient components
    gradient = create_shtns_vector_field(T, config, 𝒟ᵒᶜ, 
                                        (pencils.θ, pencils.φ, pencils.r))
    
    # Spectral representation using spectral pencil
    spectral  = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    nonlinear = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    prev_nonlinear = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)

    # Work arrays
    work_spectral      = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    work_physical      = create_shtns_physical_field(T, config, 𝒟ᵒᶜ, pencils.r)
    advection_physical = create_shtns_physical_field(T, config, 𝒟ᵒᶜ, pencils.r)

    # Sources and boundary conditions
    internal_sources = zeros(T, 𝒟ᵒᶜ.N)
    boundary_values  = zeros(T, 2, config.nlm)
    
    # Default BC types (DIRICHLET = fixed temperature, NEUMANN = fixed flux)
    bc_type_inner = fill(Int(DIRICHLET), config.nlm)  # Default to fixed temperature
    bc_type_outer = fill(Int(DIRICHLET), config.nlm)
    
    # Storage for file-based boundary conditions
    boundary_data_cache = Dict{String, Any}()
    
    # Pre-compute l(l+1) factors
    ℓ_factors = Float64[l * (l + 1) for l in config.l_values]
    
    # Transform manager removed in SHTnsKit migration
    
    # Create radial derivative matrices
    ∂r  = create_derivative_matrix(1, 𝒟ᵒᶜ)
    ∂²r = create_derivative_matrix(2, 𝒟ᵒᶜ)
    
    # Pre-compute spectral derivative operators
    theta_derivative_matrix = build_∂θ(T, config)
    theta_recurrence_coeffs = compute_theta_recurrence_coefficients(T, config)
    
    return SHTnsTemperatureField{T}(
        temperature, gradient, spectral, nonlinear, prev_nonlinear,
        work_spectral, work_physical, advection_physical,
        internal_sources, boundary_values,
        bc_type_inner, bc_type_outer,
        nothing, Dict{String, Any}(), Ref(1),  # boundary condition fields
        ℓ_factors, config,
        ∂r, ∂²r,
        theta_derivative_matrix, theta_recurrence_coeffs,
        Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0),
        𝒟ᵒᶜ
    )
end

# Matrix-embedded temperature BC functions
include("bcs/thermal_bc.jl")

# ================================================================================
# Main nonlinear computation with full spectral optimization
# ================================================================================
function compute_temperature_nonlinear!(temp_𝔽::SHTnsTemperatureField{T},
                                        vel_fields, 𝒟ᵒᶜ::RadialDomain,
                                        ws::GradientWorkspace{T};
                                        geometry::Symbol = get_parameters().geometry) where T
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0

    # Zero work arrays and gradient workspace
    zero_scalar_work_arrays!(temp_𝔽)
    zero_gradient_workspace!(ws)

    # Step 1: Compute ALL gradients in spectral space (NO COMMUNICATION!)
    t_spectral = MPI.Wtime()
    compute_all_gradients_spectral!(temp_𝔽, 𝒟ᵒᶜ, ws)
    temp_𝔽.spectral_time[] += MPI.Wtime() - t_spectral

    # Step 2: Single batched transform of temperature and gradients to physical
    t_transform = MPI.Wtime()
    transform_field_and_gradients_to_physical!(temp_𝔽, ws)
    temp_𝔽.transform_time[] += MPI.Wtime() - t_transform

    # Step 3: Compute advection term -u·∇T in physical space (local operation)
    if vel_fields !== nothing
        compute_scalar_advection_local!(temp_𝔽, vel_fields)
    end

    # Step 4: Add internal heat sources (local operation)
    add_internal_sources_local!(temp_𝔽, 𝒟ᵒᶜ)

    # Step 5: Transform advection + sources back to spectral space
    t_transform = MPI.Wtime()
    if geometry === :ball
        ball_physical_to_spectral!(temp_𝔽.advection_physical, temp_𝔽.nonlinear)
    else
        shtnskit_physical_to_spectral!(temp_𝔽.advection_physical, temp_𝔽.nonlinear)
    end
    temp_𝔽.transform_time[] += MPI.Wtime() - t_transform

    if ENABLE_TIMING[]
        temp_𝔽.computation_time[] += MPI.Wtime() - t_start
    end
end

# ================================================================================
# Fully spectral gradient computation (NO COMMUNICATION!)
# ================================================================================
# NOTE: Gradient computation functions moved to scalar_field_common.jl
# NOTE: Batched transform operations moved to scalar_field_common.jl
# ================================================================================

# ================================================================================
# Local Physical Space Operations (no MPI communication)
# ================================================================================
#
# These functions compute the advection term in physical space where
# point-wise products are straightforward.
#
# ================================================================================

function compute_temperature_advection_local!(temp_𝔽::SHTnsTemperatureField{T},
                                             vel_fields) where T
    # =========================================================================
    # Compute the advection term: -u·∇T
    # =========================================================================
    #
    # This is the EXPLICIT part of the temperature equation (Eq. 2):
    #   ∂T/∂t + u·∇T = (Pm/Pr) ∇²T
    #
    # In spherical coordinates:
    #   u·∇T = uᵣ ∂T/∂r + (uθ/r) ∂T/∂θ + (uφ/(r sin θ)) ∂T/∂φ
    #
    # The gradients (∂T/∂r, ∂T/∂θ, ∂T/∂φ) are pre-computed in spectral space
    # and transformed to physical space before this function is called.
    #
    # This operation is COMPLETELY LOCAL - no MPI communication needed.
    # =========================================================================
    uᵣ = parent(vel_fields.velocity.r_component.data)
    uθ = parent(vel_fields.velocity.θ_component.data)
    uφ = parent(vel_fields.velocity.φ_component.data)

    ∇r = parent(temp_𝔽.gradient.r_component.data)
    ∇θ = parent(temp_𝔽.gradient.θ_component.data)
    ∇φ = parent(temp_𝔽.gradient.φ_component.data)

    advection = parent(temp_𝔽.advection_physical.data)

    @inbounds @simd for idx in eachindex(advection)
        if idx <= length(uᵣ) && idx <= length(∇r)
            advection[idx] = -(uᵣ[idx] * ∇r[idx] +
                              uθ[idx] * ∇θ[idx] +
                              uφ[idx] * ∇φ[idx])
        end
    end
end

function add_internal_sources_local!(temp_𝔽::SHTnsTemperatureField{T},
                                    domain::RadialDomain) where T
    advection = parent(temp_𝔽.advection_physical.data)

    if !all(iszero, temp_𝔽.internal_sources)
        # Get local physical dimensions
        local_shape = size(temp_𝔽.advection_physical.data)
        nlat_local, nlon_local, nr_local = local_shape

        r_range = range_local(temp_𝔽.config.pencils.r, 3)

        @inbounds for k in 1:nr_local
            r_idx = k + first(r_range) - 1
            if r_idx <= length(temp_𝔽.internal_sources)
                source_value = temp_𝔽.internal_sources[r_idx]

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
            if temp_field.bc_type_inner[lm_idx] == Int(NEUMANN)
                prescribed = get_flux_value(lm_idx, 1, temp_field)
                actual = compute_flux_at_boundary(spec_real, spec_imag, local_lm,
                                                 1, temp_field, domain)
                error = abs(prescribed - actual)
                max_error = max(max_error, error)
            end

            # Check outer boundary
            if temp_field.bc_type_outer[lm_idx] == Int(NEUMANN)
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
function compute_nusselt_number(temp_𝔽::SHTnsTemperatureField{T},
                               domain::RadialDomain) where T
    # Compute heat flux from radial gradient
    ∇r = temp_𝔽.gradient.r_component

    # Get flux at boundaries (requires communication)
    flux_inner = compute_surface_flux(∇r, 1, temp_𝔽.config)
    flux_outer = compute_surface_flux(∇r, domain.N, temp_𝔽.config)

    # Nusselt number: ratio of actual to conductive heat flux
    # For spherical shell with ΔT=1: Q_cond = 4π·r_i·r_o / (r_o - r_i)
    r_inner = domain.r[1, 4]
    r_outer = domain.r[domain.N, 4]
    conductive_flux = 4π * r_inner * r_outer / (r_outer - r_inner)
    Nu = abs(flux_outer) / max(conductive_flux, eps(Float64))

    return Nu
end


function compute_thermal_energy(temp_𝔽::SHTnsTemperatureField{T}) where T
    spec_real = parent(temp_𝔽.spectral.data_real)
    spec_imag = parent(temp_𝔽.spectral.data_imag)

    # Local energy computation
    local_energy = 0.0

    lm_range = range_local(temp_𝔽.config.pencils.spec, 1)
    r_range  = range_local(temp_𝔽.config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_𝔽.config.nlm
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


function compute_surface_flux(field::SHTnsPhysField{T}, r_level::Int, 
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
                    # Use Gaussian quadrature weights (already account for sin(θ) via Gauss-Legendre)
                    weight = config.gauss_weights[θ_idx] * (2π / config.nlon)
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
function get_temperature_statistics(temp_𝔽::SHTnsTemperatureField{T},
                                   domain::RadialDomain) where T
    # Min/max temperature
    temp_data = parent(temp_𝔽.temperature.data)
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
    Nu = compute_nusselt_number(temp_𝔽, domain)

    # Total energy
    energy = compute_thermal_energy(temp_𝔽)

    return (min = global_min,
            max = global_max,
            rms = rms_temp,
            nusselt = Nu,
            energy = energy)
end

# ================================================================================
# Utility functions
# ================================================================================
function zero_temperature_work_arrays!(temp_𝔽::SHTnsTemperatureField{T}) where T
    fill!(parent(temp_𝔽.work_spectral.data_real), zero(T))
    fill!(parent(temp_𝔽.work_spectral.data_imag), zero(T))
    fill!(parent(temp_𝔽.work_physical.data), zero(T))
    fill!(parent(temp_𝔽.advection_physical.data), zero(T))
end

function set_temperature_ic!(temp_𝔽::SHTnsTemperatureField{T},
                            domain::RadialDomain;
                            perturbation_amplitude::T = T(1e-3)) where T
    spec_real = parent(temp_𝔽.spectral.data_real)
    spec_imag = parent(temp_𝔽.spectral.data_imag)

    lm_range = range_local(temp_𝔽.config.pencils.spec, 1)
    r_range = range_local(temp_𝔽.config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_𝔽.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l = temp_𝔽.config.l_values[lm_idx]
            m = temp_𝔽.config.m_values[lm_idx]
            
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

function set_boundary_conditions!(temp_𝔽::SHTnsTemperatureField{T};
                                 inner_bc_type::Int = Int(DIRICHLET),
                                 outer_bc_type::Int = Int(DIRICHLET),
                                 inner_value::T = T(1.0),
                                 outer_value::T = T(0.0)) where T
    """
    Set boundary condition types and values
    """
    # Set BC types for all modes
    fill!(temp_𝔽.bc_type_inner, inner_bc_type)
    fill!(temp_𝔽.bc_type_outer, outer_bc_type)
    
    # Set boundary values for l=0, m=0 mode (mean temperature)
    l0m0_idx = get_mode_index(temp_𝔽.config, 0, 0)
    if l0m0_idx > 0
        temp_𝔽.boundary_values[1, l0m0_idx] = inner_value
        temp_𝔽.boundary_values[2, l0m0_idx] = outer_value
    end
    
    # Other modes have zero boundary values by default
    for lm_idx in 2:temp_𝔽.config.nlm
        temp_𝔽.boundary_values[1, lm_idx] = T(0.0)
        temp_𝔽.boundary_values[2, lm_idx] = T(0.0)
    end
end

function set_internal_heating!(temp_𝔽::SHTnsTemperatureField{T},
                              domain::RadialDomain;
                              heating_type::Symbol = :uniform,
                              amplitude::T = T(1.0)) where T
    if heating_type == :uniform
        # Uniform volumetric heating
        fill!(temp_𝔽.internal_sources, amplitude)
    elseif heating_type == :gaussian
        # Gaussian heating profile centered at mid-radius
        r_mid = 0.5 * (domain.r[1, 4] + domain.r[end, 4])
        sigma = 0.1 * (domain.r[end, 4] - domain.r[1, 4])

        for i in 1:domain.N
            r = domain.r[i, 4]
            temp_𝔽.internal_sources[i] = amplitude * exp(-((r - r_mid)/sigma)^2)
        end
    elseif heating_type == :bottom
        # Heating concentrated near bottom
        for i in 1:domain.N
            r = domain.r[i, 4]
            r_norm = (r - domain.r[1, 4]) / (domain.r[end, 4] - domain.r[1, 4])
            temp_𝔽.internal_sources[i] = amplitude * exp(-5.0 * r_norm)
        end
    else
        # No heating
        fill!(temp_𝔽.internal_sources, zero(T))
    end
end

# Note: NetCDF boundary condition functions moved to src/bcs/thermal.jl

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

# Note: File-based boundary condition functions moved to src/bcs/thermal.jl

# Note: Boundary condition exports moved to src/bcs/thermal.jl
