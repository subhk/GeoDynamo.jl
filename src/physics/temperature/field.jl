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
# The diffusivity coefficient passed to the time-stepper is (Pm/Pr) = Pm/Pr.
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

# `fields/scalar_operators.jl` is included in the main module, so its shared
# scalar-field helpers are available here.

"""
    SHTnsTemperatureField{T}

Temperature field state used by GeoDynamo’s thermal evolution equations.

It contains the physical temperature, its gradient, spectral coefficients,
nonlinear work arrays, boundary/source data, and cached radial/spectral
operators.
"""
mutable struct SHTnsTemperatureField{
    T,
    C <: SHTnsKitConfig,
    PF <: SHTnsPhysField{T},
    VF <: SHTnsVectorField{T},
    SF <: SHTnsSpecField{T}
} <: AbstractScalarField{T}
    # Physical space temperature
    temperature::PF
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

    # Sources and boundary conditions
    internal_sources::Vector{T}        # Radial profile of heating
    boundary_values::Matrix{T}         # [2, nlm] for ICB and CMB
    bc_type_inner::Vector{Int}         # BC type for each mode at inner
    bc_type_outer::Vector{Int}         # BC type for each mode at outer

    # File-based boundary condition support
    boundary_condition_set::Union{bcs.BoundaryConditionSet{T}, Nothing}  # Loaded boundary conditions
    boundary_interpolation_cache::bcs.BoundaryInterpolationCache{T}  # Cached interpolated data
    boundary_time_index::Ref{Int}                                    # Current time index for time-dependent BCs

    # Pre-computed coefficients
    l_factors::Vector{T}               # l(l+1) values

    # Configuration (SHTnsKit)
    config::C

    # Radial derivative matrices
    ∂r::BandedMatrix{T}
    ∂²r::BandedMatrix{T}

    # Spectral derivative operators
    theta_derivative_matrix::SparseMatrixCSC{T, Int}  # Pre-computed θ-derivative
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
get_main_physical_field(𝔽::SHTnsTemperatureField{T}) where {T} = 𝔽.temperature

"""
    create_shtns_temperature_field(T, config, domain; pencils=nothing, pencil_spec=nothing)

Allocate and initialize the temperature field container for a simulation.
"""
function create_shtns_temperature_field(::Type{T}, config::C,
        outer_core_domain::RadialDomain,
        pencils = nothing, pencil_spec = nothing) where {T, C <: SHTnsKitConfig}
    # Use config's pencils by default (consistent with velocity/magnetic creators)
    if pencils === nothing
        pencils = config.pencils
    end
    if pencil_spec === nothing
        pencil_spec = pencils.spec
    end

    # Temperature field in r-pencil for efficient radial operations
    temperature = create_shtns_physical_field(T, config, outer_core_domain, pencils.r)

    # Gradient components
    gradient = create_shtns_vector_field(T, config, outer_core_domain,
        (pencils.θ, pencils.φ, pencils.r))

    # Spectral representation using spectral pencil
    spectral = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    nonlinear = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    prev_nonlinear = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)

    # Work arrays
    work_spectral = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    work_physical = create_shtns_physical_field(T, config, outer_core_domain, pencils.r)
    advection_physical = create_shtns_physical_field(T, config, outer_core_domain, pencils.r)

    # Sources and boundary conditions
    internal_sources = zeros(T, outer_core_domain.N)
    boundary_values = zeros(T, 2, config.nlm)

    # Default BC types (DIRICHLET = fixed temperature, NEUMANN = fixed flux)
    bc_type_inner = fill(Int(DIRICHLET), config.nlm)  # Default to fixed temperature
    bc_type_outer = fill(Int(DIRICHLET), config.nlm)

    # Pre-compute l(l+1) factors
    l_factors = T[l * (l + 1) for l in config.l_values]

    # Transform manager removed in SHTnsKit migration

    # Create radial derivative matrices
    ∂r = create_derivative_matrix(T, 1, outer_core_domain)
    ∂²r = create_derivative_matrix(T, 2, outer_core_domain)

    # Pre-compute spectral derivative operators
    theta_derivative_matrix = build_∂θ(T, config)
    theta_recurrence_coeffs = compute_theta_recurrence_coefficients(T, config)

    return SHTnsTemperatureField(
        temperature, gradient, spectral, nonlinear, prev_nonlinear,
        work_spectral, work_physical, advection_physical,
        internal_sources, boundary_values,
        bc_type_inner, bc_type_outer,
        nothing, bcs.BoundaryInterpolationCache(T), Ref(1),  # boundary condition fields
        l_factors, config,
        ∂r, ∂²r,
        theta_derivative_matrix, theta_recurrence_coeffs,
        Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0),
        outer_core_domain
    )
end

# Matrix-embedded temperature BC functions
include("../../bcs/thermal_bc.jl")

# ================================================================================
# Main nonlinear computation with full spectral optimization
# ================================================================================
function compute_temperature_nonlinear!(temp_𝔽::SHTnsTemperatureField{T},
        vel_fields, outer_core_domain::RadialDomain,
        ws::GradientWorkspace{T};
        geometry::Symbol) where {T}
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0

    # Zero work arrays and gradient workspace
    zero_scalar_work_arrays!(temp_𝔽)
    zero_gradient_workspace!(ws)

    # Step 1: Compute ALL gradients in spectral space (NO COMMUNICATION!)
    if ENABLE_TIMING[]
        t_spectral = MPI.Wtime()
    end
    compute_all_gradients_spectral!(temp_𝔽, outer_core_domain, ws)
    if ENABLE_TIMING[]
        temp_𝔽.spectral_time[] += MPI.Wtime() - t_spectral
    end

    # Step 2: Single batched transform of temperature and gradients to physical
    if ENABLE_TIMING[]
        t_transform = MPI.Wtime()
    end
    transform_field_and_gradients_to_physical!(temp_𝔽, ws)
    if ENABLE_TIMING[]
        temp_𝔽.transform_time[] += MPI.Wtime() - t_transform
    end

    # Step 3: Compute advection term -u·∇T in physical space (local operation)
    if vel_fields !== nothing
        compute_scalar_advection_local!(temp_𝔽, vel_fields)
    end

    # Step 4: Add internal heat sources (local operation)
    add_internal_sources_local!(temp_𝔽, outer_core_domain)

    # Step 5: Transform advection + sources back to spectral space
    if ENABLE_TIMING[]
        t_transform = MPI.Wtime()
    end
    # geometry-blind since the ball grid has no r=0 node (off-center grid)
    shtnskit_physical_to_spectral!(temp_𝔽.advection_physical, temp_𝔽.nonlinear)
    if ENABLE_TIMING[]
        temp_𝔽.transform_time[] += MPI.Wtime() - t_transform
        temp_𝔽.computation_time[] += MPI.Wtime() - t_start
    end
end

# ================================================================================
# Fully spectral gradient computation (NO COMMUNICATION!)
# ================================================================================
# NOTE: Gradient computation functions moved to `fields/scalar_operators.jl`.
# NOTE: Batched transform operations moved to `fields/scalar_operators.jl`.
# ================================================================================

# ================================================================================
# Local Physical Space Operations (no MPI communication)
# ================================================================================
#
# These functions compute the advection term in physical space where
# point-wise products are straightforward.
#
# ================================================================================

function add_internal_sources_local!(temp_𝔽::SHTnsTemperatureField{T},
        domain::RadialDomain) where {T}
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
# Boundary condition implementation moved to `fields/scalar_operators.jl`.
# ================================================================================
# All flux boundary condition methods (tau, influence matrix, direct) have been
# moved to `fields/scalar_operators.jl` to be shared between temperature and
# composition fields.
# The functions are now generic and work with AbstractScalarField.

# Validation and Testing
# ================================================================================

function validate_flux_bc(temp_field, domain)
    """
    Check if flux boundary conditions are satisfied within tolerance.
    """
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)

    lm_range = local_spectral_mode_indices(temp_field.config)

    max_error = 0.0

    for lm_idx in lm_range
        if lm_idx <= temp_field.config.nlm
            slot = local_spectral_storage_slot(temp_field.config, lm_idx)
            slot === nothing && continue

            # Check inner boundary
            if temp_field.bc_type_inner[lm_idx] == Int(NEUMANN)
                prescribed = get_flux_value(lm_idx, 1, temp_field)
                actual = compute_flux_at_boundary(spec_real, spec_imag, slot,
                    1, temp_field, domain)
                error = abs(prescribed - actual)
                max_error = max(max_error, error)
            end

            # Check outer boundary
            if temp_field.bc_type_outer[lm_idx] == Int(NEUMANN)
                prescribed = get_flux_value(lm_idx, 2, temp_field)
                actual = compute_flux_at_boundary(spec_real, spec_imag, slot,
                    2, temp_field, domain)
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
"""
    compute_nusselt_number(temperature_field, domain)

Compute a shell-averaged Nusselt number from the radial heat flux at the inner
and outer boundaries.
"""
function compute_nusselt_number(temp_𝔽::SHTnsTemperatureField{T},
        domain::RadialDomain) where {T}
    ∇r = temp_𝔽.gradient.r_component
    config = temp_𝔽.config

    r_inner = domain.r[1, 4]
    r_outer = domain.r[domain.N, 4]
    if r_outer <= r_inner
        return T(NaN)
    end

    # Nu = ⟨∂T/∂r⟩_outer / ⟨∂T/∂r⟩_conductive_outer — a ratio of solid-angle means.
    # compute_surface_flux returns the quadrature sum ∮(·)dΩ; dividing by the same
    # quadrature on a unit field (surface_solid_angle) gives the physical mean, so
    # both the Gauss-weight normalization and the r² area factor cancel in the
    # ratio. Conductive reference: T_cond(r)=ri·ro/(ro−ri)·(1/r−1/ro) for the ΔT=1
    # spherical-shell solution, so dT/dr|_ro = −ri/((ro−ri)·ro).
    flux_outer = compute_surface_flux(∇r, domain.N, config)
    norm_outer = surface_solid_angle(domain.N, config)
    if norm_outer <= eps(T)
        return T(NaN)
    end
    mean_grad_outer = flux_outer / norm_outer
    cond_grad_outer = -r_inner / ((r_outer - r_inner) * r_outer)
    if abs(cond_grad_outer) <= eps(T)
        return T(NaN)
    end
    return abs(mean_grad_outer / cond_grad_outer)
end

"""
    compute_thermal_energy(temperature_field)

Compute the global thermal energy from the spectral temperature coefficients.
"""
function compute_thermal_energy(temp_𝔽::SHTnsTemperatureField{T}) where {T}
    spec_real = parent(temp_𝔽.spectral.data_real)
    spec_imag = parent(temp_𝔽.spectral.data_imag)

    # Local energy computation
    local_energy = 0.0

    lm_range = local_spectral_mode_indices(temp_𝔽.config)
    r_range = range_local(temp_𝔽.config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_𝔽.config.nlm
            slot = local_spectral_storage_slot(temp_𝔽.config, lm_idx)
            slot === nothing && continue

            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(spec_real, 3)
                    local_energy += (local_spectral_value(spec_real, slot, local_r)^2 +
                                     local_spectral_value(spec_imag, slot, local_r)^2)
                end
            end
        end
    end

    # Global sum across all processes
    return 0.5 * MPI.Allreduce(local_energy, MPI.SUM, get_comm())
end

"""
    compute_surface_flux(field, r_level, config)

Compute the horizontally integrated scalar flux through the radial shell at
`r_level`. This is the low-level thermal diagnostic used by Nusselt-number and
boundary-flux post-processing.
"""
function compute_surface_flux(field::SHTnsPhysField{T}, r_level::Int,
        config::C) where {T, C <: SHTnsKitConfig}
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

                idx = local_θ + (local_φ-1)*length(θ_range) +
                      (local_r-1)*length(θ_range)*length(φ_range)

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

# Quadrature norm ∮dΩ at radial level r_level, using the same Gauss weights and
# summation as compute_surface_flux. compute_nusselt_number divides by this to
# form the solid-angle mean, so the weight-normalization convention cancels.
function surface_solid_angle(r_level::Int, config::C) where {C <: SHTnsKitConfig}
    local_norm = 0.0
    θ_range, φ_range, r_range = range_local(config.pencils.r)
    if r_level in r_range
        for φ_idx in φ_range, θ_idx in θ_range

            if θ_idx <= config.nlat && φ_idx <= config.nlon
                local_norm += config.gauss_weights[θ_idx] * (2π / config.nlon)
            end
        end
    end
    return MPI.Allreduce(local_norm, MPI.SUM, get_comm())
end

# ================================================================================
# Performance monitoring and statistics
# ================================================================================
"""
    get_temperature_statistics(temperature_field, domain)

Return a named set of basic temperature statistics such as extrema, RMS level,
surface fluxes, and Nusselt number.
"""
function get_temperature_statistics(temp_𝔽::SHTnsTemperatureField{T},
        domain::RadialDomain) where {T}
    # Min/max temperature
    temp_data = parent(temp_𝔽.temperature.data)
    local_min = minimum(temp_data)
    local_max = maximum(temp_data)

    global_min = MPI.Allreduce(local_min, MPI.MIN, get_comm())
    global_max = MPI.Allreduce(local_max, MPI.MAX, get_comm())

    # RMS temperature
    local_sum = sum(abs2, temp_data)
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
function zero_temperature_work_arrays!(temp_𝔽::SHTnsTemperatureField{T}) where {T}
    fill!(parent(temp_𝔽.work_spectral.data_real), zero(T))
    fill!(parent(temp_𝔽.work_spectral.data_imag), zero(T))
    fill!(parent(temp_𝔽.work_physical.data), zero(T))
    fill!(parent(temp_𝔽.advection_physical.data), zero(T))
end

"""
    set_temperature_ic!(temp_field, domain; perturbation_amplitude=1e-3)

Seed the temperature field with the standard conductive mean profile plus
small low-degree perturbations. This is the default programmatic initializer
used by examples and tests that want a nontrivial thermal state without
loading an external restart.
"""
function set_temperature_ic!(temp_𝔽::SHTnsTemperatureField{T},
        domain::RadialDomain;
        perturbation_amplitude::T = T(1e-3)) where {T}
    spec_real = parent(temp_𝔽.spectral.data_real)
    spec_imag = parent(temp_𝔽.spectral.data_imag)

    lm_range = local_spectral_mode_indices(temp_𝔽.config)
    r_range = range_local(temp_𝔽.config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_𝔽.config.nlm
            slot = local_spectral_storage_slot(temp_𝔽.config, lm_idx)
            slot === nothing && continue
            l = temp_𝔽.config.l_values[lm_idx]
            m = temp_𝔽.config.m_values[lm_idx]

            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(spec_real, 3)
                    r = domain.r[r_idx, 4]

                    if l == 0 && m == 0
                        # Conductive profile for spherical shell: T(r) = ri*ro/(ro-ri) * (1/r - 1/ro)
                        ri = domain.r[1, 4]
                        ro = domain.r[domain.N, 4]
                        set_local_spectral_value!(spec_real, slot, local_r,
                            ri * ro / (ro - ri) * (1.0 / r - 1.0 / ro))
                        set_local_spectral_value!(spec_imag, slot, local_r, 0.0)
                    elseif l <= 4
                        # Small perturbation for low modes
                        set_local_spectral_value!(spec_real, slot, local_r,
                            perturbation_amplitude * randn(T))
                        if m > 0
                            set_local_spectral_value!(spec_imag, slot, local_r,
                                perturbation_amplitude * randn(T))
                        else
                            set_local_spectral_value!(spec_imag, slot, local_r, 0.0)
                        end
                    else
                        # Zero for high modes
                        set_local_spectral_value!(spec_real, slot, local_r, 0.0)
                        set_local_spectral_value!(spec_imag, slot, local_r, 0.0)
                    end
                end
            end
        end
    end
end

"""
    set_boundary_conditions!(temp_field; inner_bc_type=DIRICHLET,
                             outer_bc_type=DIRICHLET,
                             inner_value=one(T), outer_value=zero(T))

Assign uniform thermal boundary-condition types and the mean-mode boundary
values for a temperature field. Higher modes are reset to zero boundary values
so the thermal boundary state stays consistent with a spherically symmetric
prescribed profile.
"""
function set_boundary_conditions!(temp_𝔽::SHTnsTemperatureField{T};
        inner_bc_type::Int = Int(DIRICHLET),
        outer_bc_type::Int = Int(DIRICHLET),
        inner_value::T = T(1.0),
        outer_value::T = T(0.0)) where {T}
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

"""
    set_internal_heating!(temp_field, domain; heating_type=:uniform, amplitude=one(T))

Populate the radial internal-heating profile used by the temperature equation.
The helper provides a few common analytic profiles so tests and examples can
configure volumetric heating without constructing the source vector manually.
"""
function set_internal_heating!(temp_𝔽::SHTnsTemperatureField{T},
        domain::RadialDomain;
        heating_type::Symbol = :uniform,
        amplitude::T = T(1.0)) where {T}
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
