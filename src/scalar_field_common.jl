# ================================================================================
# Common scalar field implementation for thermal and compositional fields
# ================================================================================
# This file contains shared functionality between thermal and compositional
# fields to reduce code duplication and improve maintainability.
# ================================================================================

using PencilArrays
using SHTnsKit
using LinearAlgebra
using SparseArrays

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
# Pre-computation of spectral derivative operators (SHARED)
# ================================================================================

"""
    build_theta_derivative_matrix(::Type{T}, config::SHTnsKitConfig) where T

Build sparse matrix for θ-derivatives in spectral space.
This matrix couples different l modes with the same m.
"""
function build_theta_derivative_matrix(::Type{T}, config::SHTnsKitConfig) where T
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
    compute_theta_gradient_spectral!(field::AbstractScalarField{T}) where T

Compute ∂field/∂θ using spherical harmonic recurrence relations (local operation).
This is generic and works for any scalar field.
"""
function compute_theta_gradient_spectral!(field::AbstractScalarField{T}) where T
    spec_real   = parent(field.spectral.data_real)
    spec_imag   = parent(field.spectral.data_imag)
    grad_θ_real = parent(field.grad_theta_spec.data_real) 
    grad_θ_imag = parent(field.grad_theta_spec.data_imag)
    
    lm_range = range_local(field.config.pencils.spec, 1)
    r_range  = range_local(field.config.pencils.spec, 3)
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(grad_θ_real, 3)
            
            for lm_idx in lm_range
                if lm_idx <= field.config.nlm
                    local_lm = lm_idx - first(lm_range) + 1
                    
                    l, m = field.config.lm_mapping[lm_idx]
                    
                    if local_lm <= size(grad_θ_real, 1)
                        # Initialize gradient to zero
                        dtheta_real = zero(T)
                        dtheta_imag = zero(T)
                        
                        # Recurrence relations for ∂/∂θ
                        # ∂Y_l^m/∂θ = A_+^{l,m} Y_{l+1}^m + A_-^{l,m} Y_{l-1}^m
                        
                        # Contribution from Y_{l+1}^m (if exists)
                        if l+1 <= field.config.lmax
                            lm_plus = field.config.get_lm_index(l+1, m)
                            if lm_plus !== nothing && lm_plus in lm_range
                                local_lm_plus = lm_plus - first(lm_range) + 1
                                if local_lm_plus <= size(spec_real, 1)
                                    A_plus = sqrt((l+1)^2 - m^2) * sqrt((l+1+m)*(l+1-m)) / (2*l+1)
                                    dtheta_real += A_plus * spec_real[local_lm_plus, 1, local_r]
                                    dtheta_imag += A_plus * spec_imag[local_lm_plus, 1, local_r]
                                end
                            end
                        end
                        
                        # Contribution from Y_{l-1}^m (if exists)
                        if l >= 1
                            lm_minus = field.config.get_lm_index(l-1, m)
                            if lm_minus !== nothing && lm_minus in lm_range
                                local_lm_minus = lm_minus - first(lm_range) + 1
                                if local_lm_minus <= size(spec_real, 1)
                                    A_minus = -sqrt(l^2 - m^2) * sqrt((l+m)*(l-m)) / (2*l-1)
                                    dtheta_real += A_minus * spec_real[local_lm_minus, 1, local_r]
                                    dtheta_imag += A_minus * spec_imag[local_lm_minus, 1, local_r]
                                end
                            end
                        end
                        
                        grad_θ_real[local_lm, 1, local_r] = dtheta_real
                        grad_θ_imag[local_lm, 1, local_r] = dtheta_imag
                    end
                end
            end
        end
    end
end

"""
    compute_phi_gradient_spectral!(field::AbstractScalarField{T}) where T

Compute ∂field/∂φ using spherical harmonic properties (local operation).
This is generic and works for any scalar field.
"""
function compute_phi_gradient_spectral!(field::AbstractScalarField{T}) where T
    spec_real   = parent(field.spectral.data_real)
    spec_imag   = parent(field.spectral.data_imag)
    grad_φ_real = parent(field.grad_phi_spec.data_real)
    grad_φ_imag = parent(field.grad_phi_spec.data_imag)
    
    lm_range = range_local(field.config.pencils.spec, 1)
    r_range  = range_local(field.config.pencils.spec, 3)
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(grad_φ_real, 3)
            
            for lm_idx in lm_range
                if lm_idx <= field.config.nlm
                    local_lm = lm_idx - first(lm_range) + 1
                    
                    l, m = field.config.lm_mapping[lm_idx]
                    
                    if local_lm <= size(grad_φ_real, 1)
                        # ∂Y_l^m/∂φ = im * m * Y_l^m
                        # For real/imag decomposition: 
                        # ∂(Real)/∂φ = -m * Imag
                        # ∂(Imag)/∂φ = m * Real
                        
                        grad_φ_real[local_lm, 1, local_r] = -T(m) * spec_imag[local_lm, 1, local_r]
                        grad_φ_imag[local_lm, 1, local_r] =  T(m) * spec_real[local_lm, 1, local_r]
                    end
                end
            end
        end
    end
end

"""
    compute_radial_gradient_spectral!(field::AbstractScalarField{T}, domain::RadialDomain) where T

Compute ∂field/∂r using banded matrix derivative operator in spectral space (local operation).
This uses the pre-computed derivative matrices from the field for optimal accuracy and efficiency.
"""
function compute_radial_gradient_spectral!(field::AbstractScalarField{T}, domain::RadialDomain) where T
    spec_real   = parent(field.spectral.data_real)
    spec_imag   = parent(field.spectral.data_imag)
    grad_r_real = parent(field.grad_r_spec.data_real)
    grad_r_imag = parent(field.grad_r_spec.data_imag)
    
    lm_range = range_local(field.config.pencils.spec, 1)
    r_range  = range_local(field.config.pencils.spec, 3)
    
    # Use the banded matrix derivative operator from the field
    bandwidth = field.dr_matrix.bandwidth
    nr = domain.N
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Apply banded matrix to radial profile
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(grad_r_real, 3)
                    
                    dr_real = zero(T)
                    dr_imag = zero(T)
                    
                    # Banded matrix multiplication
                    for j in max(1, r_idx - bandwidth):min(nr, r_idx + bandwidth)
                        if j in r_range
                            local_j = j - first(r_range) + 1
                            band_row = bandwidth + 1 + r_idx - j
                            if 1 <= band_row <= 2*bandwidth + 1
                                coeff = field.dr_matrix.data[band_row, j]
                                dr_real += coeff * spec_real[local_lm, 1, local_j]
                                dr_imag += coeff * spec_imag[local_lm, 1, local_j]
                            end
                        end
                    end
                    
                    grad_r_real[local_lm, 1, local_r] = dr_real
                    grad_r_imag[local_lm, 1, local_r] = dr_imag
                end
            end
        end
    end
end

"""
    apply_geometric_factors_spectral!(field::AbstractScalarField{T}, domain::RadialDomain) where T

Apply geometric factors (1/r, 1/(r sin θ)) in spectral space.
For gradients in spherical coordinates. This is generic.
"""
function apply_geometric_factors_spectral!(field::AbstractScalarField{T}, domain::RadialDomain) where T
    grad_θ_real = parent(field.grad_theta_spec.data_real)
    grad_θ_imag = parent(field.grad_theta_spec.data_imag)
    grad_φ_real = parent(field.grad_phi_spec.data_real)
    grad_φ_imag = parent(field.grad_phi_spec.data_imag)
    
    r_range  = range_local(field.config.pencils.spec, 3)
    lm_range = range_local(field.config.pencils.spec, 1)
    
    # Use provided domain information
    
    @inbounds for r_idx in r_range
        if r_idx <= domain.N
            local_r = r_idx - first(r_range) + 1
            r_val = domain.r[r_idx, 4]
            # Avoid 1/0 at r=0 (ball geometry). In that case, leave gradients zero.
            if r_val == 0.0
                @simd for lm_idx in lm_range
                    local_lm = lm_idx - first(lm_range) + 1
                    if local_lm <= size(grad_θ_real, 1) && local_r <= size(grad_θ_real, 3)
                        grad_θ_real[local_lm, 1, local_r] = 0
                        grad_θ_imag[local_lm, 1, local_r] = 0
                        grad_φ_real[local_lm, 1, local_r] = 0
                        grad_φ_imag[local_lm, 1, local_r] = 0
                    end
                end
            else
                r_inv = domain.r[r_idx, 3]  # 1/r
                @simd for lm_idx in lm_range
                    local_lm = lm_idx - first(lm_range) + 1
                    if local_lm <= size(grad_θ_real, 1) && local_r <= size(grad_θ_real, 3)
                        # ∇_θ field = (1/r) ∂field/∂θ
                        grad_θ_real[local_lm, 1, local_r] *= r_inv
                        grad_θ_imag[local_lm, 1, local_r] *= r_inv
                        # ∇_φ field = (1/(r sin θ)) ∂field/∂φ; sinθ handled by SH basis
                        grad_φ_real[local_lm, 1, local_r] *= r_inv
                        grad_φ_imag[local_lm, 1, local_r] *= r_inv
                    end
                end
            end
        end
    end
end

"""
    compute_all_gradients_spectral!(field::AbstractScalarField{T}, domain::RadialDomain) where T

Compute all gradient components (θ, φ, r) in spectral space.
This is the main driver function that works for any scalar field.
"""
function compute_all_gradients_spectral!(field::AbstractScalarField{T}, domain::RadialDomain) where T
    # Compute θ and φ gradients using SH derivatives (local operation)  
    compute_theta_gradient_spectral!(field)
    compute_phi_gradient_spectral!(field)
    
    # Compute radial gradient using banded matrix derivative operator (local operation)
    compute_radial_gradient_spectral!(field, domain)
    
    # Apply geometric factors for spherical coordinates (local operation)
    apply_geometric_factors_spectral!(field, domain)
end

# ================================================================================
# Batched transform operations (SHARED)
# ================================================================================

"""
    transform_field_and_gradients_to_physical!(field::AbstractScalarField{T}) where T

Transform scalar field and all gradient components to physical space
in a single batched operation to minimize communication.
"""
function transform_field_and_gradients_to_physical!(field::AbstractScalarField{T}) where T
    # Create arrays of fields to transform
    spectral_fields = [field.spectral,
                       field.grad_theta_spec,
                       field.grad_phi_spec,
                       field.grad_r_spec]
    
    # Determine physical field based on field type
    main_physical_field = get_main_physical_field(field)
    
    physical_fields = [main_physical_field,
                       field.gradient.θ_component,
                       field.gradient.φ_component,
                       field.gradient.r_component]
    
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
    compute_scalar_advection_local!(field::AbstractScalarField{T}, vel_fields) where T

Compute -u·∇field in physical space (completely local operation).
This works for any scalar field.
"""
function compute_scalar_advection_local!(field::AbstractScalarField{T}, vel_fields) where T
    u_r = parent(vel_fields.velocity.r_component.data)
    u_θ = parent(vel_fields.velocity.θ_component.data)
    u_φ = parent(vel_fields.velocity.φ_component.data)
    
    grad_r = parent(field.gradient.r_component.data)
    grad_θ = parent(field.gradient.θ_component.data)
    grad_φ = parent(field.gradient.φ_component.data)
    
    advection = parent(field.advection_physical.data)
    
    @inbounds @simd for idx in eachindex(advection)
        if idx <= length(u_r) && idx <= length(grad_r)
            advection[idx] = -(u_r[idx] * grad_r[idx] + 
                              u_θ[idx] * grad_θ[idx] + 
                              u_φ[idx] * grad_φ[idx])
        end
    end
end

"""
    add_internal_sources_local!(field::AbstractScalarField{T}, domain::RadialDomain) where T

Add volumetric sources (completely local operation).
This works for any scalar field with radial source profile.
"""
function add_internal_sources_local!(field::AbstractScalarField{T}, domain::RadialDomain) where T
    advection = parent(field.advection_physical.data)
    
    if !all(iszero, field.internal_sources)
        # Get local physical dimensions
        local_shape = size(field.advection_physical.data)
        nlat_local, nlon_local, nr_local = local_shape
        
        r_range = range_local(field.config.pencils.r, 3)
        
        @inbounds for k in 1:nr_local
            r_idx = k + first(r_range) - 1
            if r_idx <= length(field.internal_sources)
                source_value = field.internal_sources[r_idx]
                
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
    compute_scalar_rms(field::AbstractScalarField{T}, oc_domain::RadialDomain) where T

Compute RMS value of scalar field.
"""
function compute_scalar_rms(field::AbstractScalarField{T}, oc_domain::RadialDomain) where T
    spec_real = parent(field.spectral.data_real)
    spec_imag = parent(field.spectral.data_imag)
    
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
    
    return sqrt(global_sum / (oc_domain.N * field.config.nlm))
end

"""
    compute_scalar_energy(field::AbstractScalarField{T}, oc_domain::RadialDomain) where T

Compute energy ∫ field² dV
"""
function compute_scalar_energy(field::AbstractScalarField{T}, oc_domain::RadialDomain) where T
    spec_real = parent(field.spectral.data_real)
    spec_imag = parent(field.spectral.data_imag)
    
    local_energy = zero(T)
    r_range = range_local(field.config.pencils.spec, 3)
    
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3) && r_idx <= oc_domain.N
            # Volume element for this radius
            vol_element = oc_domain.r[r_idx, 2]  # 4π r² dr weight
            
            for lm_idx in axes(spec_real, 1)
                if lm_idx <= field.config.nlm
                    val_real = spec_real[lm_idx, 1, local_r]
                    val_imag = spec_imag[lm_idx, 1, local_r]
                    local_energy += vol_element * (val_real^2 + val_imag^2)
                end
            end
        end
    end
    
    # Global reduction
    comm = get_comm()
    global_energy = MPI.Allreduce(local_energy, MPI.SUM, comm)
    
    return global_energy / (field.config.nlat * field.config.nlon * oc_domain.N)
end

# ================================================================================
# Utility functions (SHARED)
# ================================================================================

"""
    zero_scalar_work_arrays!(field::AbstractScalarField{T}) where T

Efficiently zero all work arrays for a scalar field.
"""
function zero_scalar_work_arrays!(field::AbstractScalarField{T}) where T
    fill!(parent(field.work_spectral.data_real), zero(T))
    fill!(parent(field.work_spectral.data_imag), zero(T))
    fill!(parent(field.work_physical.data), zero(T))
    fill!(parent(field.advection_physical.data), zero(T))
    fill!(parent(field.grad_theta_spec.data_real), zero(T))
    fill!(parent(field.grad_theta_spec.data_imag), zero(T))
    fill!(parent(field.grad_phi_spec.data_real), zero(T))
    fill!(parent(field.grad_phi_spec.data_imag), zero(T))
    fill!(parent(field.grad_r_spec.data_real), zero(T))
    fill!(parent(field.grad_r_spec.data_imag), zero(T))
    fill!(parent(field.nonlinear.data_real), zero(T))
    fill!(parent(field.nonlinear.data_imag), zero(T))
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
    dr_matrix = create_derivative_matrix(1, domain)
    dtau = dr_matrix * tau
    return dtau[1]
end

"""
    evaluate_tau_derivative_outer(tau::Vector, domain::RadialDomain)

Evaluate tau function derivative at outer boundary.
"""
function evaluate_tau_derivative_outer(tau::Vector, domain::RadialDomain)
    dr_matrix = create_derivative_matrix(1, domain)
    dtau = dr_matrix * tau
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
    compute_boundary_fluxes(profile::Vector{T}, dr_matrix, domain::RadialDomain) where T

Compute flux (dT/dr) at both boundaries using the derivative matrix.
Note: dr_matrix should be a BandedMatrix{T} (defined in linear_algebra.jl)
"""
function compute_boundary_fluxes(profile::Vector{T}, dr_matrix,
                                domain::RadialDomain) where T
    nr = domain.N
    dprofile = dr_matrix * profile
    
    return dprofile[1], dprofile[nr]
end

"""
    get_flux_value(lm_idx::Int, boundary::Int, field::AbstractScalarField)

Get prescribed flux value for given mode and boundary from scalar field.
This is a generic version that works with any scalar field.
"""
function get_flux_value(lm_idx::Int, boundary::Int, field::AbstractScalarField)
    # Check if field has boundary_values matrix
    if hasfield(typeof(field), :boundary_values)
        # Return flux value from boundary_values matrix
        # boundary_values[boundary, lm_idx] where boundary: 1=inner, 2=outer
        return field.boundary_values[boundary, lm_idx]
    else
        # Fallback: get from l,m values if available
        if hasfield(typeof(field), :config) && 
           hasfield(typeof(field.config), :l_values) && 
           hasfield(typeof(field.config), :m_values)
            
            l = field.config.l_values[lm_idx]
            m = field.config.m_values[lm_idx]
            
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
                       apply_inner, apply_outer, field::AbstractScalarField, 
                       domain, r_range)

Apply flux boundary conditions using the tau method.
This is the generalized version that works with any scalar field.
"""
function apply_flux_bc_tau!(spec_real, spec_imag, local_lm, lm_idx,
                           apply_inner, apply_outer,
                           field::AbstractScalarField, domain, r_range)
    T = eltype(spec_real)
    nr = domain.N
    
    # Extract radial profile for this mode
    profile_real = zeros(T, nr)
    profile_imag = zeros(T, nr)
    
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            profile_real[r_idx] = spec_real[local_lm, 1, local_r]
            profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
        end
    end
    
    # MPI gather to get complete profile (needed for BC application)
    if MPI.Comm_size(get_comm()) > 1
        MPI.Allreduce!(profile_real, MPI.SUM, get_comm())
        MPI.Allreduce!(profile_imag, MPI.SUM, get_comm())
    end
    
    # Get prescribed flux values
    flux_inner = apply_inner ? get_flux_value(lm_idx, 1, field) : T(0)
    flux_outer = apply_outer ? get_flux_value(lm_idx, 2, field) : T(0)
    
    # Compute current fluxes at boundaries
    current_flux_inner, current_flux_outer = compute_boundary_fluxes(
        profile_real, field.dr_matrix, domain)
    
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
    
    # Store corrected profile back
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            spec_real[local_lm, 1, local_r] = profile_real[r_idx]
            spec_imag[local_lm, 1, local_r] = profile_imag[r_idx]
        end
    end
end

# ================================================================================
# Influence Matrix Method Implementation
# ================================================================================

"""
    compute_influence_functions_flux(oc_domain::RadialDomain)

Compute influence functions for flux BCs.
These are solutions to the homogeneous equation with specific BCs.
"""
function compute_influence_functions_flux(oc_domain::RadialDomain)
    nr = oc_domain.N
    ri = oc_domain.r[1, 4]
    ro = oc_domain.r[nr, 4]
    
    G_inner = zeros(nr)
    G_outer = zeros(nr)
    
    for i in 1:nr
        r = oc_domain.r[i, 4]
        ξ = (r - ri) / (ro - ri)
        
        # Inner influence: strong at inner, weak at outer
        G_inner[i] = exp(-3.0 * ξ) * (1.0 - ξ)
        
        # Outer influence: weak at inner, strong at outer
        G_outer[i] = exp(-3.0 * (1.0 - ξ)) * ξ
    end
    
    # Normalize to have unit flux contribution
    normalize_influence_function!(G_inner, oc_domain, 1)
    normalize_influence_function!(G_outer, oc_domain, 2)
    
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
    build_influence_matrix(G_inner, G_outer, dr_matrix, domain)

Construct a 2×2 matrix mapping influence amplitudes to boundary flux errors.
Rows correspond to (inner, outer) boundaries; columns to (inner, outer) influence functions.
"""
function build_influence_matrix(G_inner::Vector{T}, G_outer::Vector{T},
                               dr_matrix, domain::RadialDomain) where T
    dGi = dr_matrix * G_inner
    dGo = dr_matrix * G_outer
    return [dGi[1]  dGo[1];
            dGi[end] dGo[end]]
end

"""
    _get_influence_cache(domain::RadialDomain, dr_matrix)

Get or create cached influence functions and matrix for given domain.
Note: dr_matrix should be a BandedMatrix (defined in linear_algebra.jl)
"""
function _get_influence_cache(domain::RadialDomain, dr_matrix)
    cache = get(_INFLUENCE_CACHE, domain, nothing)
    if cache === nothing || cache.nr != domain.N
        Gi, Go = compute_influence_functions_flux(domain)
        M = build_influence_matrix(Gi, Go, dr_matrix, domain)
        cache = _InfluenceCache(domain.N, Gi, Go, M)
        _INFLUENCE_CACHE[domain] = cache
    end
    return cache
end

"""
    apply_flux_bc_influence_matrix!(spec_real, spec_imag, local_lm, lm_idx,
                                   apply_inner, apply_outer, field::AbstractScalarField,
                                   domain, r_range)

Apply flux boundary conditions using the influence matrix method.
"""
function apply_flux_bc_influence_matrix!(spec_real, spec_imag, local_lm, lm_idx,
                                       apply_inner, apply_outer,
                                       field::AbstractScalarField, domain, r_range)
    T = eltype(spec_real)
    infl = _get_influence_cache(domain, field.dr_matrix)
    
    # Get prescribed and current flux values
    flux_prescribed = [get_flux_value(lm_idx, 1, field),
                      get_flux_value(lm_idx, 2, field)]
    
    # Extract and gather radial profile
    nr = domain.N
    profile_real = zeros(T, nr)
    profile_imag = zeros(T, nr)
    
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            profile_real[r_idx] = spec_real[local_lm, 1, local_r]
            profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
        end
    end
    
    if MPI.Comm_size(get_comm()) > 1
        MPI.Allreduce!(profile_real, MPI.SUM, get_comm())
        MPI.Allreduce!(profile_imag, MPI.SUM, get_comm())
    end
    
    # Compute current flux at boundaries
    current_flux_inner, current_flux_outer = compute_boundary_fluxes(
        profile_real, field.dr_matrix, domain)
    flux_current = [current_flux_inner, current_flux_outer]
    
    # Solve for influence amplitudes
    flux_error = flux_prescribed - flux_current
    amplitudes = infl.influence_matrix \ flux_error
    
    # Apply influence correction
    @. profile_real += amplitudes[1] * infl.G_inner + amplitudes[2] * infl.G_outer
    if any(x -> abs(x) > 1e-12, profile_imag)
        @. profile_imag += amplitudes[1] * infl.G_inner + amplitudes[2] * infl.G_outer
    end
    
    # Store corrected profile back
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            spec_real[local_lm, 1, local_r] = profile_real[r_idx]
            spec_imag[local_lm, 1, local_r] = profile_imag[r_idx]
        end
    end
end

# ================================================================================
# Direct Method Implementation (Simple but less accurate)
# ================================================================================

"""
    apply_flux_bc_direct!(spec_real, spec_imag, local_lm, lm_idx,
                         field::AbstractScalarField, domain, r_range)

Direct modification of boundary values to approximately satisfy flux BC.
This is the simplest but least accurate method.
"""
function apply_flux_bc_direct!(spec_real, spec_imag, local_lm, lm_idx,
                              field::AbstractScalarField, domain, r_range)
    T = eltype(spec_real)
    
    # Get prescribed flux
    if 1 in r_range
        flux_inner = get_flux_value(lm_idx, 1, field)
        modify_for_flux_inner!(spec_real, spec_imag, local_lm, flux_inner, 
                              field.dr_matrix, domain, r_range)
    end
    
    if domain.N in r_range
        flux_outer = get_flux_value(lm_idx, 2, field)
        modify_for_flux_outer!(spec_real, spec_imag, local_lm, flux_outer,
                              field.dr_matrix, domain, r_range)
    end
end

"""
    modify_for_flux_inner!(spec_real, spec_imag, local_lm, prescribed_flux,
                          dr_matrix, domain, r_range)

Modify coefficients near inner boundary to approximate flux condition.
Uses low-order extrapolation.
"""
function modify_for_flux_inner!(spec_real, spec_imag, local_lm, prescribed_flux,
                               dr_matrix, domain, r_range)
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
                          dr_matrix, domain, r_range)

Modify coefficients near outer boundary to approximate flux condition.
Uses low-order extrapolation.
"""
function modify_for_flux_outer!(spec_real, spec_imag, local_lm, prescribed_flux,
                               dr_matrix, domain, r_range)
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
    apply_scalar_flux_bc_spectral!(field::AbstractScalarField{T}, domain::RadialDomain;
                                   method::Symbol=:tau) where T

Apply flux boundary conditions to a scalar field in spectral space.
Methods available: :tau (most robust), :influence_matrix, :direct (simplest).
"""
function apply_scalar_flux_bc_spectral!(field::AbstractScalarField{T}, domain::RadialDomain;
                                       method::Symbol=:tau) where T
    spec_real = parent(field.spectral.data_real)
    spec_imag = parent(field.spectral.data_imag)
    
    lm_range = range_local(field.config.pencils.spec, 1)
    r_range  = range_local(field.config.pencils.spec, 3)
    
    for lm_idx in lm_range
        if lm_idx <= field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Check if this mode needs flux BC
            apply_inner = (field.bc_type_inner[lm_idx] == 2) && (1 in r_range)
            apply_outer = (field.bc_type_outer[lm_idx] == 2) && (domain.N in r_range)
            
            if apply_inner || apply_outer
                # Apply flux BC using specified method
                if method == :tau
                    apply_flux_bc_tau!(spec_real, spec_imag, local_lm, lm_idx,
                                      apply_inner, apply_outer, field, domain, r_range)
                elseif method == :influence_matrix
                    apply_flux_bc_influence_matrix!(spec_real, spec_imag, local_lm, lm_idx,
                                                   apply_inner, apply_outer, field, domain, r_range)
                elseif method == :direct
                    apply_flux_bc_direct!(spec_real, spec_imag, local_lm, lm_idx,
                                         field, domain, r_range)
                else
                    error("Unknown flux BC method: $method. Use :tau, :influence_matrix, or :direct")
                end
            end
        end
    end
end

# ================================================================================
# MPI utilities (SHARED)
# ================================================================================

# get_comm() function provided by pencil_decomps.jl - removed duplicate
