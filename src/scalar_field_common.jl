# ================================================================================
# Common Scalar Field Implementation for Thermal and Compositional Fields
# ================================================================================
#
# This file contains shared functionality between thermal and compositional
# fields to reduce code duplication and improve maintainability.
#
# ================================================================================
# SCALAR FIELD BOUNDARY CONDITIONS
# ================================================================================
#
# Scalar fields (temperature, composition) support two BC types:
#
# 1. DIRICHLET (fixed value): f = f₀ at boundary
#    - Common for prescribed temperature/composition
#    - Implemented by directly setting spectral coefficients
#
# 2. NEUMANN (fixed flux): ∂f/∂r = q at boundary
#    - For prescribed heat/mass flux
#    - More complex: requires flux enforcement methods
#
# Unlike velocity (which has ∂T/∂r = T/r for stress-free), scalar fields use
# SIMPLE Neumann: ∂f/∂r = q where q is the prescribed flux value.
#
# ================================================================================
# FLUX BC ENFORCEMENT METHODS
# ================================================================================
#
# Three methods are provided for enforcing Neumann (flux) BCs:
#
# 1. TAU METHOD (recommended)
#    - Uses Chebyshev polynomial tau correction
#    - Computes: flux_error = prescribed_flux - current_flux
#    - Adds correction polynomial that exactly fixes flux at boundaries
#    - Highest accuracy, preserves spectral convergence
#    - Function: apply_flux_bc_tau!()
#
# 2. INFLUENCE MATRIX METHOD
#    - Precomputes influence functions G_inner, G_outer
#    - Solves 2x2 system for correction amplitudes
#    - Good accuracy, slightly faster after initialization
#    - Function: apply_flux_bc_influence_matrix!()
#
# 3. DIRECT METHOD
#    - Simple first-order finite difference
#    - Fast but lower accuracy
#    - Use only for testing/debugging
#    - Function: apply_flux_bc_direct!()
#
# ================================================================================
# MPI SAFETY - CRITICAL PATTERN
# ================================================================================
#
# The main entry point apply_scalar_flux_bc_spectral!() uses GLOBAL loop bounds
# to ensure MPI safety:
#
#   for lm_idx in 1:nlm_total  # ALL processes iterate ALL modes
#       owns_mode = lm_idx in lm_range
#
#       if needs_bc  # Check BC type (consistent across processes)
#           apply_flux_bc_tau!(..., owns_mode)  # Contains Allreduce
#       end
#   end
#
# The owns_mode parameter propagates to inner functions, which:
#   - Fill data only if owns_mode == true
#   - Call Allreduce unconditionally (all processes participate)
#   - Store results only if owns_mode == true
#
# See timestep.jl header for detailed explanation of this pattern.
#
# ================================================================================
# USAGE EXAMPLE
# ================================================================================
#
#   # Set BC types for a temperature field
#   fill!(temp_field.bc_type_inner, Int(DIRICHLET))  # Fixed T at inner
#   fill!(temp_field.bc_type_outer, Int(NEUMANN))    # Fixed flux at outer
#
#   # Set boundary values
#   temp_field.boundary_values[1, :] .= T_inner      # Dirichlet value
#   temp_field.boundary_values[2, :] .= q_outer      # Neumann flux value
#
#   # Apply BCs (automatically uses appropriate method per mode)
#   apply_scalar_flux_bc_spectral!(temp_field, domain; method=:tau)
#
# ================================================================================

using PencilArrays
using SHTnsKit
using LinearAlgebra
using SparseArrays

import .bcs: BoundaryType, DIRICHLET, NEUMANN

# BandedMatrix type is defined in linear_algebra.jl and available in parent module scope

# ================================================================================
# Generic scalar field struct definition
# ================================================================================

"""
Abstract type for scalar fields (temperature, composition, etc.)
"""
abstract type AbstractScalarField{T} end

# Helper to map (l,m) to linear index
function get_mode_index(config::SHTnsKitConfig, l::Int, m::Int)
    @inbounds for idx in 1:config.nlm
        if config.l_values[idx] == l && config.m_values[idx] == m
            return idx
        end
    end
    return 0
end

# ================================================================================
# Shared Gradient Workspace (memory optimization)
# ================================================================================

"""
    GradientWorkspace{T}

Shared workspace for transient gradient spectral fields.
Reused across scalar field computations that run sequentially (temperature, composition).
These arrays are zeroed and recomputed every timestep, so sharing is safe.
"""
struct GradientWorkspace{T}
    ∇θ_spec::SHTnsSpecField{T}
    ∇φ_spec::SHTnsSpecField{T}
    ∇r_spec::SHTnsSpecField{T}
end

function create_gradient_workspace(::Type{T}, config::SHTnsKitConfig,
                                   domain::RadialDomain, pencil_spec=nothing) where T
    if pencil_spec === nothing
        pencil_spec = config.pencils.spec
    end
    GradientWorkspace{T}(
        create_shtns_spectral_field(T, config, domain, pencil_spec),
        create_shtns_spectral_field(T, config, domain, pencil_spec),
        create_shtns_spectral_field(T, config, domain, pencil_spec),
    )
end

function zero_gradient_workspace!(ws::GradientWorkspace{T}) where T
    fill!(parent(ws.∇θ_spec.data_real), zero(T))
    fill!(parent(ws.∇θ_spec.data_imag), zero(T))
    fill!(parent(ws.∇φ_spec.data_real), zero(T))
    fill!(parent(ws.∇φ_spec.data_imag), zero(T))
    fill!(parent(ws.∇r_spec.data_real), zero(T))
    fill!(parent(ws.∇r_spec.data_imag), zero(T))
end

# ================================================================================
# Pre-computation of spectral derivative operators (SHARED)
# ================================================================================

"""
    build_∂θ(::Type{T}, config::SHTnsKitConfig) where T

Build sparse matrix for θ-derivatives in spectral space.
This matrix couples different l modes with the same m.
"""
function build_∂θ(::Type{T}, config::SHTnsKitConfig) where T
    nlm = config.nlm
    
    # Build sparse matrix for derivative operator
    I = Int[]
    J = Int[]
    V = T[]
    
    for lm_idx in 1:nlm
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        abs_m = abs(m)
        
        # Forward coupling (l,m) → (l+1,m)
        if l < config.lmax
            lp1m_idx = get_mode_index(config, l+1, m)
            if lp1m_idx > 0
                coeff = sqrt((l + abs_m + 1) * (l - abs_m + 1) / 
                           ((2*l + 1) * (2*l + 3))) * l
                push!(I, lm_idx)
                push!(J, lp1m_idx)
                push!(V, T(coeff))
            end
        end
        
        # Backward coupling (l,m) → (l-1,m)
        if l > abs_m
            lm1m_idx = get_mode_index(config, l-1, m)
            if lm1m_idx > 0
                coeff = -sqrt((l + abs_m) * (l - abs_m) / 
                            ((2*l - 1) * (2*l + 1))) * (l + 1)
                push!(I, lm_idx)
                push!(J, lm1m_idx)
                push!(V, T(coeff))
            end
        end
    end
    
    return sparse(I, J, V, nlm, nlm)
end

"""
    compute_theta_recurrence_coefficients(::Type{T}, config::SHTnsKitConfig) where T

Pre-compute recurrence coefficients for θ-derivatives.
Store as [nlm, 2] matrix: [:, 1] for l-1 coupling, [:, 2] for l+1 coupling
"""
function compute_theta_recurrence_coefficients(::Type{T}, config::SHTnsKitConfig) where T
    coeffs = zeros(T, config.nlm, 2)
    
    for lm_idx in 1:config.nlm
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        abs_m = abs(m)
        
        # Backward coupling coefficient (l-1,m)
        if l > abs_m
            coeffs[lm_idx, 1] = -sqrt((l + abs_m) * (l - abs_m) / 
                                     ((2*l - 1) * (2*l + 1))) * (l + 1)
        end
        
        # Forward coupling coefficient (l+1,m)
        if l < config.lmax
            coeffs[lm_idx, 2] = sqrt((l + abs_m + 1) * (l - abs_m + 1) / 
                                    ((2*l + 1) * (2*l + 3))) * l
        end
    end
    
    return coeffs
end

# ================================================================================
# Spectral gradient computation (SHARED)
# ================================================================================

"""
    compute_theta_gradient_spectral!(𝔽::AbstractScalarField{T}, ws::GradientWorkspace{T}) where T

Compute ∂field/∂θ using spherical harmonic recurrence relations (local operation).
This is generic and works for any scalar field.
"""
function compute_theta_gradient_spectral!(𝔽::AbstractScalarField{T}, ws::GradientWorkspace{T}) where T
    spec_real   = parent(𝔽.spectral.data_real)
    spec_imag   = parent(𝔽.spectral.data_imag)
    ∇θ_real = parent(ws.∇θ_spec.data_real)
    ∇θ_imag = parent(ws.∇θ_spec.data_imag)

    lm_range = range_local(𝔽.config.pencils.spec, 1)
    r_range  = range_local(𝔽.config.pencils.spec, 3)
    nlm = 𝔽.config.nlm

    # MPI setup: theta gradient couples (l,m) with (l±1,m) which may reside
    # on different MPI ranks. Gather full spectral data via Allreduce.
    comm = get_comm()
    multi = MPI.Initialized() && MPI.Comm_size(comm) > 1

    # Pre-allocate full spectral vectors for gathering across ranks
    full_real = zeros(T, nlm)
    full_imag = zeros(T, nlm)

    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r > size(∇θ_real, 3)
            continue
        end

        # Gather full spectral data for this radial level
        fill!(full_real, zero(T))
        fill!(full_imag, zero(T))
        for lm_idx in lm_range
            if lm_idx <= nlm
                local_lm = lm_idx - first(lm_range) + 1
                if local_lm <= size(spec_real, 1)
                    full_real[lm_idx] = spec_real[local_lm, 1, local_r]
                    full_imag[lm_idx] = spec_imag[local_lm, 1, local_r]
                end
            end
        end
        if multi
            MPI.Allreduce!(full_real, MPI.SUM, comm)
            MPI.Allreduce!(full_imag, MPI.SUM, comm)
        end

        # Compute theta gradient using gathered full spectral data
        for lm_idx in lm_range
            if lm_idx <= nlm
                local_lm = lm_idx - first(lm_range) + 1
                if local_lm > size(∇θ_real, 1)
                    continue
                end

                l = 𝔽.config.l_values[lm_idx]
                m = 𝔽.config.m_values[lm_idx]
                abs_m = abs(m)

                # Initialize gradient to zero
                dtheta_real = zero(T)
                dtheta_imag = zero(T)

                # Recurrence relations for ∂/∂θ (4π-normalized SH)
                # ∂Y_l^m/∂θ = A_+^{l,m} Y_{l+1}^m + A_-^{l,m} Y_{l-1}^m

                # Contribution from Y_{l+1}^m (forward coupling)
                if l < 𝔽.config.lmax
                    lm_plus = get_mode_index(𝔽.config, l+1, m)
                    if lm_plus > 0 && lm_plus <= nlm
                        A_plus = T(l) * sqrt(T((l + abs_m + 1) * (l - abs_m + 1)) /
                                             T((2*l + 1) * (2*l + 3)))
                        dtheta_real += A_plus * full_real[lm_plus]
                        dtheta_imag += A_plus * full_imag[lm_plus]
                    end
                end

                # Contribution from Y_{l-1}^m (backward coupling)
                if l > abs_m
                    lm_minus = get_mode_index(𝔽.config, l-1, m)
                    if lm_minus > 0 && lm_minus <= nlm
                        A_minus = -T(l + 1) * sqrt(T((l + abs_m) * (l - abs_m)) /
                                                   T((2*l - 1) * (2*l + 1)))
                        dtheta_real += A_minus * full_real[lm_minus]
                        dtheta_imag += A_minus * full_imag[lm_minus]
                    end
                end

                ∇θ_real[local_lm, 1, local_r] = dtheta_real
                ∇θ_imag[local_lm, 1, local_r] = dtheta_imag
            end
        end
    end
end

"""
    compute_phi_gradient_spectral!(𝔽::AbstractScalarField{T}, ws::GradientWorkspace{T}) where T

Compute ∂field/∂φ using spherical harmonic properties (local operation).
This is generic and works for any scalar field.
"""
function compute_phi_gradient_spectral!(𝔽::AbstractScalarField{T}, ws::GradientWorkspace{T}) where T
    spec_real   = parent(𝔽.spectral.data_real)
    spec_imag   = parent(𝔽.spectral.data_imag)
    ∇φ_real = parent(ws.∇φ_spec.data_real)
    ∇φ_imag = parent(ws.∇φ_spec.data_imag)
    
    lm_range = range_local(𝔽.config.pencils.spec, 1)
    r_range  = range_local(𝔽.config.pencils.spec, 3)
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(∇φ_real, 3)
            
            for lm_idx in lm_range
                if lm_idx <= 𝔽.config.nlm
                    local_lm = lm_idx - first(lm_range) + 1

                    m = 𝔽.config.m_values[lm_idx]

                    if local_lm <= size(∇φ_real, 1)
                        # ∂Y_l^m/∂φ = im * m * Y_l^m
                        # For real/imag decomposition:
                        # ∂(Real)/∂φ = -m * Imag
                        # ∂(Imag)/∂φ = m * Real

                        ∇φ_real[local_lm, 1, local_r] = -T(m) * spec_imag[local_lm, 1, local_r]
                        ∇φ_imag[local_lm, 1, local_r] =  T(m) * spec_real[local_lm, 1, local_r]
                    end
                end
            end
        end
    end
end

"""
    compute_radial_gradient_spectral!(𝔽::AbstractScalarField{T}, domain::RadialDomain, ws::GradientWorkspace{T}) where T

Compute ∂field/∂r using banded matrix derivative operator in spectral space.
This uses the pre-computed derivative matrices from the field for optimal accuracy and efficiency.

Note: Radial data is always local (not MPI distributed). Only spectral (lm) modes
are distributed across MPI processes.
"""
function compute_radial_gradient_spectral!(𝔽::AbstractScalarField{T}, domain::RadialDomain, ws::GradientWorkspace{T}) where T
    spec_real = parent(𝔽.spectral.data_real)
    spec_imag = parent(𝔽.spectral.data_imag)
    ∇r_real   = parent(ws.∇r_spec.data_real)
    ∇r_imag   = parent(ws.∇r_spec.data_imag)

    lm_range = range_local(𝔽.config.pencils.spec, 1)
    r_range  = range_local(𝔽.config.pencils.spec, 3)
    nr       = domain.N
    bandwidth = 𝔽.∂r.bandwidth

    @inbounds for lm_idx in lm_range
        if lm_idx <= 𝔽.config.nlm
            local_lm = lm_idx - first(lm_range) + 1

            # Apply banded matrix to radial profile
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(∇r_real, 3)

                    dr_real = zero(T)
                    dr_imag = zero(T)

                    # Banded matrix multiplication (all data is local)
                    for j in max(1, r_idx - bandwidth):min(nr, r_idx + bandwidth)
                        if j in r_range
                            local_j = j - first(r_range) + 1
                            band_row = bandwidth + 1 + r_idx - j
                            if 1 <= band_row <= 2*bandwidth + 1
                                coeff = 𝔽.∂r.data[band_row, j]
                                dr_real += coeff * spec_real[local_lm, 1, local_j]
                                dr_imag += coeff * spec_imag[local_lm, 1, local_j]
                            end
                        end
                    end

                    ∇r_real[local_lm, 1, local_r] = dr_real
                    ∇r_imag[local_lm, 1, local_r] = dr_imag
                end
            end
        end
    end
end

"""
    apply_geometric_factors_spectral!(ws::GradientWorkspace{T}, 𝔽::AbstractScalarField{T}, domain::RadialDomain) where T

Apply geometric factors (1/r, 1/(r sin θ)) in spectral space.
For gradients in spherical coordinates. This is generic.
"""
function apply_geometric_factors_spectral!(ws::GradientWorkspace{T}, 𝔽::AbstractScalarField{T}, domain::RadialDomain) where T
    ∇θ_real = parent(ws.∇θ_spec.data_real)
    ∇θ_imag = parent(ws.∇θ_spec.data_imag)
    ∇φ_real = parent(ws.∇φ_spec.data_real)
    ∇φ_imag = parent(ws.∇φ_spec.data_imag)
    
    r_range  = range_local(𝔽.config.pencils.spec, 3)
    lm_range = range_local(𝔽.config.pencils.spec, 1)
    
    # Use provided domain information
    
    @inbounds for r_idx in r_range
        if r_idx <= domain.N
            local_r = r_idx - first(r_range) + 1
            r_val = domain.r[r_idx, 4]
            # Avoid 1/0 at r=0 (ball geometry). In that case, leave gradients zero.
            if r_val == 0.0
                @simd for lm_idx in lm_range
                    local_lm = lm_idx - first(lm_range) + 1
                    if local_lm <= size(∇θ_real, 1) && local_r <= size(∇θ_real, 3)
                        ∇θ_real[local_lm, 1, local_r] = 0
                        ∇θ_imag[local_lm, 1, local_r] = 0
                        ∇φ_real[local_lm, 1, local_r] = 0
                        ∇φ_imag[local_lm, 1, local_r] = 0
                    end
                end
            else
                r⁻¹ = domain.r[r_idx, 3]  # 1/r
                @simd for lm_idx in lm_range
                    local_lm = lm_idx - first(lm_range) + 1
                    if local_lm <= size(∇θ_real, 1) && local_r <= size(∇θ_real, 3)
                        # ∇_θ field = (1/r) ∂field/∂θ
                        ∇θ_real[local_lm, 1, local_r] *= r⁻¹
                        ∇θ_imag[local_lm, 1, local_r] *= r⁻¹
                        # ∇_φ field = (1/(r sin θ)) ∂field/∂φ; sinθ handled by SH basis
                        ∇φ_real[local_lm, 1, local_r] *= r⁻¹
                        ∇φ_imag[local_lm, 1, local_r] *= r⁻¹
                    end
                end
            end
        end
    end
end

"""
    compute_all_gradients_spectral!(𝔽::AbstractScalarField{T}, domain::RadialDomain, ws::GradientWorkspace{T}) where T

Compute all gradient components (θ, φ, r) in spectral space.
This is the main driver function that works for any scalar field.
"""
function compute_all_gradients_spectral!(𝔽::AbstractScalarField{T}, domain::RadialDomain, ws::GradientWorkspace{T}) where T
    # Compute θ and φ gradients using SH derivatives (local operation)
    compute_theta_gradient_spectral!(𝔽, ws)
    compute_phi_gradient_spectral!(𝔽, ws)

    # Compute radial gradient using banded matrix derivative operator (local operation)
    compute_radial_gradient_spectral!(𝔽, domain, ws)

    # Apply geometric factors for spherical coordinates (local operation)
    apply_geometric_factors_spectral!(ws, 𝔽, domain)
end

# ================================================================================
# Batched transform operations (SHARED)
# ================================================================================

"""
    transform_field_and_gradients_to_physical!(𝔽::AbstractScalarField{T}, ws::GradientWorkspace{T}) where T

Transform scalar field and all gradient components to physical space
in a single batched operation to minimize communication.
"""
function transform_field_and_gradients_to_physical!(𝔽::AbstractScalarField{T}, ws::GradientWorkspace{T}) where T
    # Create arrays of fields to transform
    spectral_fields = [𝔽.spectral,
                       ws.∇θ_spec,
                       ws.∇φ_spec,
                       ws.∇r_spec]

    # Determine physical field based on field type
    main_physical_field = get_main_physical_field(𝔽)

    physical_fields = [main_physical_field,
                       𝔽.gradient.θ_component,
                       𝔽.gradient.φ_component,
                       𝔽.gradient.r_component]

    # Single batched transform with one MPI communication
    batch_spectral_to_physical!(spectral_fields, physical_fields)
end

# Helper function to get the appropriate main physical field
# This needs to be specialized for each field type
function get_main_physical_field end

# ================================================================================
# Local physical space operations (SHARED)
# ================================================================================

"""
    compute_scalar_advection_local!(𝔽::AbstractScalarField{T}, vel_fields) where T

Compute -u·∇field in physical space (completely local operation).
This works for any scalar field.
"""
function compute_scalar_advection_local!(𝔽::AbstractScalarField{T}, vel_fields) where T
    u_r = parent(vel_fields.velocity.r_component.data)
    u_θ = parent(vel_fields.velocity.θ_component.data)
    u_φ = parent(vel_fields.velocity.φ_component.data)
    
    ∇r = parent(𝔽.gradient.r_component.data)
    ∇θ = parent(𝔽.gradient.θ_component.data)
    ∇φ = parent(𝔽.gradient.φ_component.data)
    
    advection = parent(𝔽.advection_physical.data)
    
    @inbounds @simd for idx in eachindex(advection)
        if idx <= length(u_r) && idx <= length(∇r)
            advection[idx] = -(u_r[idx] * ∇r[idx] + 
                              u_θ[idx] * ∇θ[idx] + 
                              u_φ[idx] * ∇φ[idx])
        end
    end
end

"""
    add_internal_sources_local!(𝔽::AbstractScalarField{T}, domain::RadialDomain) where T

Add volumetric sources (completely local operation).
This works for any scalar field with radial source profile.
"""
function add_internal_sources_local!(𝔽::AbstractScalarField{T}, domain::RadialDomain) where T
    advection = parent(𝔽.advection_physical.data)
    
    if !all(iszero, 𝔽.internal_sources)
        # Get local physical dimensions
        local_shape = size(𝔽.advection_physical.data)
        nlat_local, nlon_local, nr_local = local_shape
        
        r_range = range_local(𝔽.config.pencils.r, 3)
        
        @inbounds for k in 1:nr_local
            r_idx = k + first(r_range) - 1
            if r_idx <= length(𝔽.internal_sources)
                source_value = 𝔽.internal_sources[r_idx]
                
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
# Diagnostic functions (SHARED)
# ================================================================================

"""
    compute_scalar_rms(𝔽::AbstractScalarField{T}, 𝒟ᵒᶜ::RadialDomain) where T

Compute RMS value of scalar field.
"""
function compute_scalar_rms(𝔽::AbstractScalarField{T}, 𝒟ᵒᶜ::RadialDomain) where T
    spec_real = parent(𝔽.spectral.data_real)
    spec_imag = parent(𝔽.spectral.data_imag)
    
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
    global_sum = Allreduce(local_sum, MPI.SUM, comm)
    
    return sqrt(global_sum / (𝒟ᵒᶜ.N * 𝔽.config.nlm))
end

"""
    compute_scalar_energy(𝔽::AbstractScalarField{T}, 𝒟ᵒᶜ::RadialDomain) where T

Compute energy ∫ field² dV
"""
function compute_scalar_energy(𝔽::AbstractScalarField{T}, 𝒟ᵒᶜ::RadialDomain) where T
    spec_real = parent(𝔽.spectral.data_real)
    spec_imag = parent(𝔽.spectral.data_imag)
    
    local_energy = zero(T)
    r_range = range_local(𝔽.config.pencils.spec, 3)
    
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3) && r_idx <= 𝒟ᵒᶜ.N
            # Volume element for this radius: r² (column 6 = r^(6-4) = r^2)
            vol_element = 𝒟ᵒᶜ.r[r_idx, 6]
            
            for lm_idx in axes(spec_real, 1)
                if lm_idx <= 𝔽.config.nlm
                    val_real = spec_real[lm_idx, 1, local_r]
                    val_imag = spec_imag[lm_idx, 1, local_r]
                    local_energy += vol_element * (val_real^2 + val_imag^2)
                end
            end
        end
    end
    
    # Global reduction
    comm = get_comm()
    global_energy = Allreduce(local_energy, MPI.SUM, comm)
    
    return global_energy / (𝔽.config.nlat * 𝔽.config.nlon * 𝒟ᵒᶜ.N)
end

# ================================================================================
# Utility functions (SHARED)
# ================================================================================

"""
    zero_scalar_work_arrays!(𝔽::AbstractScalarField{T}) where T

Efficiently zero all work arrays for a scalar field.
"""
function zero_scalar_work_arrays!(𝔽::AbstractScalarField{T}) where T
    fill!(parent(𝔽.work_spectral.data_real), zero(T))
    fill!(parent(𝔽.work_spectral.data_imag), zero(T))
    fill!(parent(𝔽.work_physical.data), zero(T))
    fill!(parent(𝔽.advection_physical.data), zero(T))
    fill!(parent(𝔽.nonlinear.data_real), zero(T))
    fill!(parent(𝔽.nonlinear.data_imag), zero(T))
end

# ================================================================================
# Boundary Condition Utilities (SHARED)
# ================================================================================

# Tau method cache structure for efficient boundary condition enforcement
mutable struct _TauCache
    nr::Int
    tau1::Vector{Float64}
    tau2::Vector{Float64}
    dtau1_inner::Float64
    dtau1_outer::Float64
    dtau2_inner::Float64
    dtau2_outer::Float64
end

# Global tau cache dictionary
const _TAU_CACHE = IdDict{RadialDomain, _TauCache}()

# Influence matrix cache structure
mutable struct _InfluenceCache
    nr::Int
    G_inner::Vector{Float64}
    G_outer::Vector{Float64}
    influence_matrix::Matrix{Float64}
end

# Global influence cache dictionary
const _INFLUENCE_CACHE = IdDict{RadialDomain, _InfluenceCache}()

# ================================================================================
# Chebyshev Polynomial Utilities
# ================================================================================

"""
    compute_chebyshev_polynomial(n::Int, domain::RadialDomain)

Compute Chebyshev polynomial T_n on the radial grid.
"""
function compute_chebyshev_polynomial(n::Int, domain::RadialDomain)
    nr = domain.N
    poly = zeros(nr)
    
    ri = domain.r[1, 4]
    ro = domain.r[nr, 4]
    
    for i in 1:nr
        r = domain.r[i, 4]
        # Map to [-1, 1]
        x = 2.0 * (r - ri) / (ro - ri) - 1.0
        # T_n(x) = cos(n * acos(x))
        poly[i] = cos(n * acos(clamp(x, -1.0, 1.0)))
    end
    
    return poly
end

"""
    evaluate_chebyshev_derivative(n::Int, r::T, domain::RadialDomain) where T

Evaluate derivative of Chebyshev polynomial at a point.
"""
function evaluate_chebyshev_derivative(n::Int, r::T, domain::RadialDomain) where T
    ri = domain.r[1, 4]
    ro = domain.r[domain.N, 4]
    
    # Map to [-1, 1]
    x = 2.0 * (r - ri) / (ro - ri) - 1.0
    
    # dT_n/dx = n * U_{n-1}(x) where U is Chebyshev polynomial of second kind
    # U_{n-1}(x) = sin(n * acos(x)) / sin(acos(x))
    
    if abs(abs(x) - 1.0) < 1e-12
        # Special case at boundaries
        dTn_dx = n^2 * sign(x)^(n+1)
    else
        theta = acos(clamp(x, -1.0, 1.0))
        dTn_dx = n * sin(n * theta) / sin(theta)
    end
    
    # Chain rule: dT_n/dr = dT_n/dx * dx/dr
    dx_dr = 2.0 / (ro - ri)
    
    return dTn_dx * dx_dr
end

# ================================================================================
# Tau Method Cache Management
# ================================================================================

"""
    _get_tau_cache(domain::RadialDomain)

Get or create cached tau polynomials and derivatives for given domain.
"""
function _get_tau_cache(domain::RadialDomain)
    nr = domain.N
    cache = get(_TAU_CACHE, domain, nothing)
    if cache === nothing || cache.nr != nr || length(cache.tau1) != nr
        # Recompute cache for current domain
        tau1 = compute_chebyshev_polynomial(nr-1, domain)
        tau2 = compute_chebyshev_polynomial(nr, domain)
        dt1i = evaluate_chebyshev_derivative(nr-1, domain.r[1, 4], domain)
        dt1o = evaluate_chebyshev_derivative(nr-1, domain.r[nr, 4], domain)
        dt2i = evaluate_chebyshev_derivative(nr,   domain.r[1, 4], domain)
        dt2o = evaluate_chebyshev_derivative(nr,   domain.r[nr, 4], domain)
        cache = _TauCache(nr, tau1, tau2, dt1i, dt1o, dt2i, dt2o)
        _TAU_CACHE[domain] = cache
    end
    return cache
end

# ================================================================================
# Tau Method Implementation
# ================================================================================

"""
    compute_tau_coefficients_both(flux_error_inner::T, flux_error_outer::T, domain::RadialDomain) where T

Compute tau polynomial coefficients for both boundaries.
Uses highest two Chebyshev modes as tau functions.
"""
function compute_tau_coefficients_both(flux_error_inner::T, flux_error_outer::T,
                                      domain::RadialDomain) where T
    nr = domain.N
    # Use cached tau polynomials/derivatives for this nr
    tau_cache = _get_tau_cache(domain)
    tau1 = tau_cache.tau1
    tau2 = tau_cache.tau2
    dtau1_inner = tau_cache.dtau1_inner
    dtau1_outer = tau_cache.dtau1_outer
    dtau2_inner = tau_cache.dtau2_inner
    dtau2_outer = tau_cache.dtau2_outer
    
    # Solve 2x2 system for tau coefficients
    # [dtau1_inner  dtau2_inner] [c1]   [flux_error_inner]
    # [dtau1_outer  dtau2_outer] [c2] = [flux_error_outer]
    
    det = dtau1_inner * dtau2_outer - dtau1_outer * dtau2_inner
    
    if abs(det) > 1e-12
        c1 = (flux_error_inner * dtau2_outer - flux_error_outer * dtau2_inner) / det
        c2 = (flux_error_outer * dtau1_inner - flux_error_inner * dtau1_outer) / det
    else
        c1 = c2 = zero(T)
    end
    
    return (c1, c2, tau1, tau2)
end

"""
    compute_tau_coefficients_inner(flux_error::T, domain::RadialDomain) where T

Compute tau coefficient for inner boundary only.
Uses a single tau function that doesn't affect outer boundary.
"""
function compute_tau_coefficients_inner(flux_error::T, domain::RadialDomain) where T
    nr = domain.N
    
    # Use a polynomial that has zero derivative at outer boundary
    # This is a linear combination of Chebyshev polynomials
    tau = compute_inner_tau_function(domain)
    
    # Derivative at inner boundary
    dtau_inner = evaluate_tau_derivative_inner(tau, domain)
    
    # Tau coefficient
    c = flux_error / dtau_inner
    
    return (c, tau)
end

"""
    compute_tau_coefficients_outer(flux_error::T, domain::RadialDomain) where T

Compute tau coefficient for outer boundary only.
Uses a single tau function that doesn't affect inner boundary.
"""
function compute_tau_coefficients_outer(flux_error::T, domain::RadialDomain) where T
    nr = domain.N
    
    # Use a polynomial that has zero derivative at inner boundary
    # Similar to inner case but with roles reversed
    tau = compute_outer_tau_function(domain)
    
    # Derivative at outer boundary
    dtau_outer = evaluate_tau_derivative_outer(tau, domain)
    
    # Tau coefficient
    c = flux_error / dtau_outer
    
    return (c, tau)
end

"""
    compute_inner_tau_function(domain::RadialDomain)

Compute tau function for inner boundary only (zero derivative at outer).
"""
function compute_inner_tau_function(domain::RadialDomain)
    nr = domain.N
    # Use T_{nr-1} - α*T_{nr} where α chosen to make derivative zero at outer
    tau1 = compute_chebyshev_polynomial(nr-1, domain)
    tau2 = compute_chebyshev_polynomial(nr, domain)
    
    dt1_outer = evaluate_chebyshev_derivative(nr-1, domain.r[nr, 4], domain)
    dt2_outer = evaluate_chebyshev_derivative(nr, domain.r[nr, 4], domain)
    
    α = abs(dt2_outer) > 1e-12 ? dt1_outer / dt2_outer : 0.0
    
    return tau1 - α * tau2
end

"""
    compute_outer_tau_function(domain::RadialDomain)

Compute tau function for outer boundary only (zero derivative at inner).
"""
function compute_outer_tau_function(domain::RadialDomain)
    nr = domain.N
    # Use T_{nr-1} - β*T_{nr} where β chosen to make derivative zero at inner
    tau1 = compute_chebyshev_polynomial(nr-1, domain)
    tau2 = compute_chebyshev_polynomial(nr, domain)
    
    dt1_inner = evaluate_chebyshev_derivative(nr-1, domain.r[1, 4], domain)
    dt2_inner = evaluate_chebyshev_derivative(nr, domain.r[1, 4], domain)
    
    β = abs(dt2_inner) > 1e-12 ? dt1_inner / dt2_inner : 0.0
    
    return tau1 - β * tau2
end

"""
    evaluate_tau_derivative_inner(tau::Vector, domain::RadialDomain)

Evaluate tau function derivative at inner boundary.
"""
function evaluate_tau_derivative_inner(tau::Vector, domain::RadialDomain)
    ∂r = create_derivative_matrix(1, domain)
    dtau = ∂r * tau
    return dtau[1]
end

"""
    evaluate_tau_derivative_outer(tau::Vector, domain::RadialDomain)

Evaluate tau function derivative at outer boundary.
"""
function evaluate_tau_derivative_outer(tau::Vector, domain::RadialDomain)
    ∂r = create_derivative_matrix(1, domain)
    dtau = ∂r * tau
    return dtau[end]
end

"""
    apply_tau_correction!(profile::Vector{T}, tau_coeffs, domain::RadialDomain) where T

Add tau correction to the radial profile.
"""
function apply_tau_correction!(profile::Vector{T}, tau_coeffs, domain::RadialDomain) where T
    if length(tau_coeffs) == 4  # Both boundaries
        c1, c2, tau1, tau2 = tau_coeffs
        @. profile += c1 * tau1 + c2 * tau2
    elseif length(tau_coeffs) == 2  # Single boundary
        c, tau = tau_coeffs
        @. profile += c * tau
    end
end

# ================================================================================
# Boundary Condition Utility Functions
# ================================================================================

"""
    compute_boundary_fluxes(profile::Vector{T}, ∂r, domain::RadialDomain) where T

Compute flux (dT/dr) at both boundaries using the derivative matrix.
Note: ∂r should be a BandedMatrix{T} (defined in linear_algebra.jl)
"""
function compute_boundary_fluxes(profile::Vector{T}, ∂r,
                                domain::RadialDomain) where T
    nr = domain.N
    dprofile = ∂r * profile
    
    return dprofile[1], dprofile[nr]
end

"""
    get_flux_value(lm_idx::Int, boundary::Int, 𝔽::AbstractScalarField)

Get prescribed flux value for given mode and boundary from scalar field.
This is a generic version that works with any scalar field.
"""
function get_flux_value(lm_idx::Int, boundary::Int, 𝔽::AbstractScalarField)
    # Check if field has boundary_values matrix
    if hasfield(typeof(𝔽), :boundary_values)
        # Return flux value from boundary_values matrix
        # boundary_values[boundary, lm_idx] where boundary: 1=inner, 2=outer
        return 𝔽.boundary_values[boundary, lm_idx]
    else
        # Fallback: get from l,m values if available
        if hasfield(typeof(𝔽), :config) &&
           hasfield(typeof(𝔽.config), :l_values) &&
           hasfield(typeof(𝔽.config), :m_values)
            
            l = 𝔽.config.l_values[lm_idx]
            m = 𝔽.config.m_values[lm_idx]
            
            # Default example: uniform heating/cooling for l=0,m=0
            if l == 0 && m == 0
                if boundary == 1
                    return 1.0   # Heating from below
                else
                    return -1.0  # Cooling from above
                end
            else
                return 0.0  # No flux for other modes
            end
        else
            return 0.0  # No flux available
        end
    end
end

"""
    apply_flux_bc_tau!(spec_real, spec_imag, local_lm, lm_idx,
                       apply_inner, apply_outer, 𝔽::AbstractScalarField,
                       domain, r_range, owns_mode::Bool)

Apply flux boundary conditions using the tau method.
This is the generalized version that works with any scalar field.

# MPI Safety
The `owns_mode` parameter indicates whether this process owns the lm mode.
All processes must call this function for each mode to ensure Allreduce is called
the same number of times by all processes (prevents deadlock).
"""
function apply_flux_bc_tau!(spec_real, spec_imag, local_lm, lm_idx,
                           apply_inner, apply_outer,
                           𝔽::AbstractScalarField, domain, r_range, owns_mode::Bool)
    T = eltype(spec_real)
    nr = domain.N

    # Extract radial profile for this mode (only if this process owns the mode)
    profile_real = zeros(T, nr)
    profile_imag = zeros(T, nr)

    if owns_mode
        for r_idx in r_range
            local_r = r_idx - first(r_range) + 1
            if local_r <= size(spec_real, 3)
                profile_real[r_idx] = spec_real[local_lm, 1, local_r]
                profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
            end
        end
    end

    # MPI gather (ALL processes call this for synchronization)
    comm = get_comm()
    if comm !== nothing && MPI.Comm_size(comm) > 1
        Allreduce!(profile_real, MPI.SUM, comm)
        Allreduce!(profile_imag, MPI.SUM, comm)
    end

    # Get prescribed flux values
    flux_inner = apply_inner ? get_flux_value(lm_idx, 1, 𝔽) : T(0)
    flux_outer = apply_outer ? get_flux_value(lm_idx, 2, 𝔽) : T(0)

    # Compute current fluxes at boundaries
    current_flux_inner, current_flux_outer = compute_boundary_fluxes(
        profile_real, 𝔽.∂r, domain)
    
    # Compute tau corrections
    if apply_inner && apply_outer
        # Both boundaries have flux BCs
        tau_coeffs = compute_tau_coefficients_both(
            flux_inner - current_flux_inner,
            flux_outer - current_flux_outer,
            domain)
    elseif apply_inner
        # Only inner boundary
        tau_coeffs = compute_tau_coefficients_inner(
            flux_inner - current_flux_inner, domain)
    else
        # Only outer boundary
        tau_coeffs = compute_tau_coefficients_outer(
            flux_outer - current_flux_outer, domain)
    end
    
    # Apply tau correction to profile
    apply_tau_correction!(profile_real, tau_coeffs, domain)
    if any(x -> abs(x) > 1e-12, profile_imag)
        apply_tau_correction!(profile_imag, tau_coeffs, domain)
    end

    # Store corrected profile back (only if this process owns the mode)
    if owns_mode
        for r_idx in r_range
            local_r = r_idx - first(r_range) + 1
            if local_r <= size(spec_real, 3)
                spec_real[local_lm, 1, local_r] = profile_real[r_idx]
                spec_imag[local_lm, 1, local_r] = profile_imag[r_idx]
            end
        end
    end
end

# ================================================================================
# Influence Matrix Method Implementation
# ================================================================================

"""
    compute_influence_functions_flux(𝒟ᵒᶜ::RadialDomain)

Compute influence functions for flux BCs.
These are solutions to the homogeneous equation with specific BCs.
"""
function compute_influence_functions_flux(𝒟ᵒᶜ::RadialDomain)
    nr = 𝒟ᵒᶜ.N
    ri = 𝒟ᵒᶜ.r[1, 4]
    ro = 𝒟ᵒᶜ.r[nr, 4]
    
    G_inner = zeros(nr)
    G_outer = zeros(nr)
    
    for i in 1:nr
        r = 𝒟ᵒᶜ.r[i, 4]
        ξ = (r - ri) / (ro - ri)
        
        # Inner influence: strong at inner, weak at outer
        G_inner[i] = exp(-3.0 * ξ) * (1.0 - ξ)
        
        # Outer influence: weak at inner, strong at outer
        G_outer[i] = exp(-3.0 * (1.0 - ξ)) * ξ
    end
    
    # Normalize to have unit flux contribution
    normalize_influence_function!(G_inner, 𝒟ᵒᶜ, 1)
    normalize_influence_function!(G_outer, 𝒟ᵒᶜ, 2)
    
    return G_inner, G_outer
end

"""
    normalize_influence_function!(G::Vector{T}, domain::RadialDomain, which_boundary::Int) where T

Normalize influence function to have unit flux at the specified boundary.
"""
function normalize_influence_function!(G::Vector{T}, domain::RadialDomain, which_boundary::Int) where T
    dr = create_derivative_matrix(1, domain)
    dG = dr * G
    if which_boundary == 1
        scale = dG[1]
    else
        scale = dG[end]
    end
    if abs(scale) > eps(T)
        @. G = G / scale
    end
    return G
end

"""
    build_influence_matrix(G_inner, G_outer, ∂r, domain)

Construct a 2×2 matrix mapping influence amplitudes to boundary flux errors.
Rows correspond to (inner, outer) boundaries; columns to (inner, outer) influence functions.
"""
function build_influence_matrix(G_inner::Vector{T}, G_outer::Vector{T},
                               ∂r, domain::RadialDomain) where T
    dGi = ∂r * G_inner
    dGo = ∂r * G_outer
    return [dGi[1]  dGo[1];
            dGi[end] dGo[end]]
end

"""
    _get_influence_cache(domain::RadialDomain, ∂r)

Get or create cached influence functions and matrix for given domain.
Note: ∂r should be a BandedMatrix (defined in linear_algebra.jl)
"""
function _get_influence_cache(domain::RadialDomain, ∂r)
    cache = get(_INFLUENCE_CACHE, domain, nothing)
    if cache === nothing || cache.nr != domain.N
        Gi, Go = compute_influence_functions_flux(domain)
        M = build_influence_matrix(Gi, Go, ∂r, domain)
        cache = _InfluenceCache(domain.N, Gi, Go, M)
        _INFLUENCE_CACHE[domain] = cache
    end
    return cache
end

"""
    apply_flux_bc_influence_matrix!(spec_real, spec_imag, local_lm, lm_idx,
                                   apply_inner, apply_outer, 𝔽::AbstractScalarField,
                                   domain, r_range, owns_mode::Bool)

Apply flux boundary conditions using the influence matrix method.

# MPI Safety
The `owns_mode` parameter indicates whether this process owns the lm mode.
All processes must call this function for each mode to ensure Allreduce is called
the same number of times by all processes (prevents deadlock).
"""
function apply_flux_bc_influence_matrix!(spec_real, spec_imag, local_lm, lm_idx,
                                       apply_inner, apply_outer,
                                       𝔽::AbstractScalarField, domain, r_range, owns_mode::Bool)
    T = eltype(spec_real)
    infl = _get_influence_cache(domain, 𝔽.∂r)

    # Get prescribed and current flux values
    flux_prescribed = [get_flux_value(lm_idx, 1, 𝔽),
                      get_flux_value(lm_idx, 2, 𝔽)]

    # Extract and gather radial profile (only if this process owns the mode)
    nr = domain.N
    profile_real = zeros(T, nr)
    profile_imag = zeros(T, nr)

    if owns_mode
        for r_idx in r_range
            local_r = r_idx - first(r_range) + 1
            if local_r <= size(spec_real, 3)
                profile_real[r_idx] = spec_real[local_lm, 1, local_r]
                profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
            end
        end
    end

    # MPI gather (ALL processes call this for synchronization)
    comm = get_comm()
    if comm !== nothing && MPI.Comm_size(comm) > 1
        Allreduce!(profile_real, MPI.SUM, comm)
        Allreduce!(profile_imag, MPI.SUM, comm)
    end

    # Compute current flux at boundaries
    current_flux_inner, current_flux_outer = compute_boundary_fluxes(
        profile_real, 𝔽.∂r, domain)
    flux_current = [current_flux_inner, current_flux_outer]

    # Solve for influence amplitudes
    flux_error = flux_prescribed - flux_current
    amplitudes = infl.influence_matrix \ flux_error

    # Apply influence correction
    @. profile_real += amplitudes[1] * infl.G_inner + amplitudes[2] * infl.G_outer
    if any(x -> abs(x) > 1e-12, profile_imag)
        @. profile_imag += amplitudes[1] * infl.G_inner + amplitudes[2] * infl.G_outer
    end

    # Store corrected profile back (only if this process owns the mode)
    if owns_mode
        for r_idx in r_range
            local_r = r_idx - first(r_range) + 1
            if local_r <= size(spec_real, 3)
                spec_real[local_lm, 1, local_r] = profile_real[r_idx]
                spec_imag[local_lm, 1, local_r] = profile_imag[r_idx]
            end
        end
    end
end

# ================================================================================
# Direct Method Implementation (Simple but less accurate)
# ================================================================================

"""
    apply_flux_bc_direct!(spec_real, spec_imag, local_lm, lm_idx,
                         𝔽::AbstractScalarField, domain, r_range)

Direct modification of boundary values to approximately satisfy flux BC.
This is the simplest but least accurate method.
"""
function apply_flux_bc_direct!(spec_real, spec_imag, local_lm, lm_idx,
                              𝔽::AbstractScalarField, domain, r_range)
    T = eltype(spec_real)
    
    # Get prescribed flux
    if 1 in r_range
        flux_inner = get_flux_value(lm_idx, 1, 𝔽)
        modify_for_flux_inner!(spec_real, spec_imag, local_lm, flux_inner,
                              𝔽.∂r, domain, r_range)
    end

    if domain.N in r_range
        flux_outer = get_flux_value(lm_idx, 2, 𝔽)
        modify_for_flux_outer!(spec_real, spec_imag, local_lm, flux_outer,
                              𝔽.∂r, domain, r_range)
    end
end

"""
    modify_for_flux_inner!(spec_real, spec_imag, local_lm, prescribed_flux,
                          ∂r, domain, r_range)

Modify coefficients near inner boundary to approximate flux condition.
Uses low-order extrapolation.
"""
function modify_for_flux_inner!(spec_real, spec_imag, local_lm, prescribed_flux,
                               ∂r, domain, r_range)
    T = eltype(spec_real)
    
    if 1 in r_range && 2 in r_range
        # Use linear extrapolation based on prescribed flux
        local_1 = 1 - first(r_range) + 1
        local_2 = 2 - first(r_range) + 1
        
        dr = domain.r[2, 4] - domain.r[1, 4]
        
        # T(r1) ≈ T(r2) - prescribed_flux * dr
        spec_real[local_lm, 1, local_1] = spec_real[local_lm, 1, local_2] - prescribed_flux * dr
        spec_imag[local_lm, 1, local_1] = spec_imag[local_lm, 1, local_2]
    end
end

"""
    modify_for_flux_outer!(spec_real, spec_imag, local_lm, prescribed_flux,
                          ∂r, domain, r_range)

Modify coefficients near outer boundary to approximate flux condition.
Uses low-order extrapolation.
"""
function modify_for_flux_outer!(spec_real, spec_imag, local_lm, prescribed_flux,
                               ∂r, domain, r_range)
    T = eltype(spec_real)
    nr = domain.N
    
    if (nr-1) in r_range && nr in r_range
        # Use linear extrapolation based on prescribed flux
        local_nr_1 = (nr-1) - first(r_range) + 1
        local_nr = nr - first(r_range) + 1
        
        dr = domain.r[nr, 4] - domain.r[nr-1, 4]
        
        # T(r_outer) ≈ T(r_inner) + prescribed_flux * dr
        spec_real[local_lm, 1, local_nr] = spec_real[local_lm, 1, local_nr_1] + prescribed_flux * dr
        spec_imag[local_lm, 1, local_nr] = spec_imag[local_lm, 1, local_nr_1]
    end
end

# ================================================================================
# High-level boundary condition application
# ================================================================================

"""
    apply_scalar_flux_bc_spectral!(𝔽::AbstractScalarField{T}, domain::RadialDomain;
                                   method::Symbol=:tau) where T

Apply flux boundary conditions to a scalar field in spectral space.
Methods available: :tau (most robust), :influence_matrix, :direct (simplest).

# MPI Safety
Uses global loop bounds (1:nlm) to ensure all processes call MPI collectives
the same number of times, preventing deadlock with uneven lm distribution.
"""
function apply_scalar_flux_bc_spectral!(𝔽::AbstractScalarField{T}, domain::RadialDomain;
                                       method::Symbol=:tau) where T
    spec_real = parent(𝔽.spectral.data_real)
    spec_imag = parent(𝔽.spectral.data_imag)

    lm_range = range_local(𝔽.config.pencils.spec, 1)
    r_range  = range_local(𝔽.config.pencils.spec, 3)
    nlm_total = 𝔽.config.nlm

    # Use GLOBAL loop bounds to ensure all processes call MPI collectives same number of times
    for lm_idx in 1:nlm_total
        # Check if this process owns this lm mode
        owns_mode = lm_idx in lm_range
        local_lm = owns_mode ? (lm_idx - first(lm_range) + 1) : 0

        # Check if this mode needs flux BC (BC type arrays are consistent across processes)
        needs_inner_bc = 𝔽.bc_type_inner[lm_idx] == Int(NEUMANN)
        needs_outer_bc = 𝔽.bc_type_outer[lm_idx] == Int(NEUMANN)

        # If any boundary needs BC, all processes must participate for MPI synchronization
        if needs_inner_bc || needs_outer_bc
            # Determine which boundaries this process can apply (owns boundary points)
            apply_inner = needs_inner_bc && (1 in r_range)
            apply_outer = needs_outer_bc && (domain.N in r_range)

            # Apply flux BC using specified method
            # ALL processes call these functions for MPI synchronization (Allreduce inside)
            if method == :tau
                apply_flux_bc_tau!(spec_real, spec_imag, local_lm, lm_idx,
                                  apply_inner, apply_outer, 𝔽, domain, r_range, owns_mode)
            elseif method == :influence_matrix
                apply_flux_bc_influence_matrix!(spec_real, spec_imag, local_lm, lm_idx,
                                               apply_inner, apply_outer, 𝔽, domain, r_range, owns_mode)
            elseif method == :direct
                # Direct method doesn't use MPI collectives, only apply if owns_mode
                if owns_mode
                    apply_flux_bc_direct!(spec_real, spec_imag, local_lm, lm_idx,
                                         𝔽, domain, r_range)
                end
            else
                error("Unknown flux BC method: $method. Use :tau, :influence_matrix, or :direct")
            end
        end
    end
end

# ================================================================================
# MPI utilities (SHARED)
# ================================================================================

# get_comm() function provided by pencil_decomps.jl - removed duplicate
