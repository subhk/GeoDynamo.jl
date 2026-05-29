# ================================================================================
# Velocity Boundary Condition Topography Corrections
# ================================================================================
#
# This file implements topography corrections for velocity boundary conditions:
#
# 1. Impermeability (u·n̂ = 0):
#    uᵣ - ε(∇_H h)·uₜ + εh ∂_r uᵣ = 0
#
# 2. No-slip (uₜ = 0):
#    uₜ + εh ∂_r uₜ = U_b,t (typically 0 at CMB, Ω_ic × r at ICB)
#
# 3. Stress-free (∂_r(uₜ/r) = 0):
#    ∂_r(uₜ/r) = (ε/r_b)(∇_H h) ∂_r uᵣ
#
# In spectral (l,m) space, the impermeability condition becomes:
#    l(l+1)/r_b² P_{u,lm} + ε Σ h_{LM} [α ∂_r P + β T_u + γ P] = 0
#
# ================================================================================

# Local helper for banded matrix multiplication (avoids import cycle)
function _apply_banded_matrix!(output::Vector{T}, matrix, input::Vector{T}) where {T}
    N = size(matrix.data, 2)
    bandwidth = (size(matrix.data, 1) - 1) ÷ 2

    fill!(output, zero(T))

    @inbounds for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            if 1 <= band_row <= size(matrix.data, 1)
                output[i] += matrix.data[band_row, j] * input[j]
            end
        end
    end

    return output
end

# ================================================================================
# Main Interface
# ================================================================================

"""
    apply_velocity_topography_correction!(velocity_field, topography::TopographyData,
                                          config::TopographyCouplingConfig)

Apply topography corrections to velocity boundary conditions.

This modifies the boundary values of the velocity field's poloidal and toroidal
components to account for topographic coupling.

# Arguments
- `velocity_field`: SHTnsVelocityFields or similar structure with toroidal/poloidal fields
- `topography`: TopographyData containing ICB and/or CMB topography
- `config`: TopographyCouplingConfig with coupling parameters
"""
function apply_velocity_topography_correction!(velocity_field, topography::TopographyData,
        config::TopographyCouplingConfig)
    if !config.velocity_coupling || !config.enabled
        return nothing
    end

    ε = config.epsilon
    gaunt = topography.gaunt_cache

    if gaunt === nothing
        @warn "Gaunt cache not initialized for velocity topography correction"
        return nothing
    end

    # Get poloidal and toroidal components
    # Support different field structure conventions
    if hasfield(typeof(velocity_field), :poloidal)
        poloidal = velocity_field.poloidal
        toroidal = velocity_field.toroidal
    elseif hasfield(typeof(velocity_field), :P)
        poloidal = velocity_field.P
        toroidal = velocity_field.T
    else
        @warn "Cannot identify poloidal/toroidal components in velocity field"
        return nothing
    end

    # The expensive radial traces are staged once per field, then reused for
    # both boundaries and every coupled (l,m) mode. That keeps the topography
    # correction focused on mode coupling rather than repeated radial algebra.
    # Precompute boundary value/derivative caches once for this field
    p_cache = compute_boundary_derivative_cache(poloidal,
        velocity_field.∂r,
        velocity_field.∂²r,
        velocity_field.domain)
    t_cache = compute_boundary_derivative_cache(toroidal,
        velocity_field.∂r,
        velocity_field.∂²r,
        velocity_field.domain)

    # Apply corrections at ICB if topography defined
    if topography.icb !== nothing
        apply_velocity_correction_at_boundary!(
            poloidal, toroidal, p_cache, t_cache,
            topography.icb, gaunt, ε, config, INNER_BOUNDARY
        )
    end

    # Apply corrections at CMB if topography defined
    if topography.cmb !== nothing
        apply_velocity_correction_at_boundary!(
            poloidal, toroidal, p_cache, t_cache,
            topography.cmb, gaunt, ε, config, OUTER_BOUNDARY
        )
    end

    return nothing
end

# ================================================================================
# Boundary-Specific Corrections
# ================================================================================

"""
    apply_velocity_correction_at_boundary!(poloidal, toroidal, topo_field,
                                           gaunt, ε, config, location)

Apply velocity topography corrections at a specific boundary.
"""
function apply_velocity_correction_at_boundary!(poloidal,
        toroidal,
        p_cache::BoundaryDerivativeCache{T},
        t_cache::BoundaryDerivativeCache{T},
        topo_field::TopographyField{T},
        gaunt::GauntTensorCache{T},
        ε::T,
        config::TopographyCouplingConfig,
        location::BoundaryLocation) where {T}
    rb = topo_field.radius
    lmax = min(poloidal.config.lmax, gaunt.lmax)
    mmax = min(poloidal.config.mmax, gaunt.lmax)

    # Get boundary row index (1 for inner, 2 for outer)
    bc_row = location == INNER_BOUNDARY ? 1 : 2

    # Get poloidal/toroidal boundary values
    P_bv = poloidal.boundary_values
    T_bv = toroidal.boundary_values

    # Boundary values are updated mode-by-mode. Each target mode gathers all
    # topography and field couplings that project back onto that same (l,m).
    # Compute corrections for each (l, m) mode
    for l in 1:lmax  # Start from l=1 (l=0 has no velocity)
        for m in -l:l
            if abs(m) > mmax
                continue
            end

            # Compute impermeability correction
            imp_corr = compute_impermeability_correction(
                l, m, p_cache, t_cache, topo_field, gaunt, rb, location, config
            )

            # Apply correction to poloidal BC
            # The flat-sphere BC is: l(l+1)/r² P = 0
            # With topography: l(l+1)/r² P + ε * correction = 0
            lm_idx = lm_to_spectral_index(l, m, poloidal.config)
            if lm_idx > 0 && lm_idx <= size(P_bv, 2)
                P_bv[bc_row, lm_idx] -= ε * real(imp_corr) * rb^2 / (l * (l + 1))
            end

            # Apply toroidal correction if stress-free
            if is_stress_free_boundary(toroidal, location)
                sf_corr = compute_stressfree_correction(
                    l, m, p_cache, topo_field, gaunt, rb, location, config
                )
                if lm_idx > 0 && lm_idx <= size(T_bv, 2)
                    T_bv[bc_row, lm_idx] += ε * real(sf_corr)
                end
            end

            if is_no_slip_boundary(toroidal, location)
                _,
                ns_t = compute_noslip_correction(
                    l, m, p_cache, t_cache, topo_field, gaunt, rb, location, config
                )
                if lm_idx > 0 && lm_idx <= size(T_bv, 2)
                    T_bv[bc_row, lm_idx] -= ε * real(ns_t)
                end
            end
        end
    end

    return nothing
end

# ================================================================================
# Impermeability Correction
# ================================================================================

"""
    compute_impermeability_correction(l, m, poloidal, toroidal, topo,
                                      gaunt, rb, config) -> Complex

Compute the topography correction to the impermeability condition for mode (l, m).

The flat-sphere condition is: uᵣ = l(l+1)/r² P_{lm} = 0

With topography:
uᵣ - ε(∇_H h)·uₜ + εh ∂_r uᵣ = 0

In spectral space (schematic):
l(l+1)/r² P_{lm} + ε Σ_{LM,l'm'} h_{LM} [
    G · ∂_r(l'(l'+1)/r² P_{l'm'})           # shift term
    - G^{(∇)} · ∂_r P_{l'm'}                   # slope term (poloidal)
    - G^{(×)} · T_{l'm'}                       # slope term (toroidal)
] = 0
"""
function compute_impermeability_correction(l::Int, m::Int,
        p_cache::BoundaryDerivativeCache{T},
        t_cache::BoundaryDerivativeCache{T},
        topo::TopographyField{T},
        gaunt::GauntTensorCache{T},
        rb::T,
        location::BoundaryLocation,
        config::TopographyCouplingConfig) where {T}
    correction = zero(Complex{T})

    lmax = min(p_cache.lmax, gaunt.lmax)
    lmax_t = topo.lmax

    # This is the actual mode-coupling core: topography mode (L,M) mixes every
    # compatible field mode (l',m') back into the target impermeability row.
    # Sum over topography modes (L, M)
    for L in 0:lmax_t
        for M in -L:L
            h_LM = get_coefficient(topo, L, M)
            if abs(h_LM) < 1e-15
                continue
            end

            # Sum over field modes (l', m')
            for lp in 1:lmax
                for mp in -lp:lp
                    # m' + M must equal m (azimuthal selection rule)
                    if mp + M != m
                        continue
                    end

                    # Get Gaunt tensors
                    G = get_gaunt_tensor(gaunt, l, m, lp, mp, L, M)
                    G_grad = get_gradient_gaunt(gaunt, l, m, lp, mp, L, M)
                    G_cross = get_cross_gaunt(gaunt, l, m, lp, mp, L, M)

                    # Get field values at boundary
                    P_lpm = get_cache_value(p_cache, lp, mp, location)
                    T_lpm = get_cache_value(t_cache, lp, mp, location)
                    dP_dr = get_cache_d1(p_cache, lp, mp, location)

                    ll_factor = T(lp * (lp + 1))

                    # Shift term: h · ∂_r uᵣ
                    if config.include_shift_terms && abs(G) > 1e-15
                        # ∂_r(l'(l'+1)/r² P) evaluated at r = rb
                        shift_contrib = G * ll_factor * (dP_dr / rb^2 - 2 * P_lpm / rb^3)
                        correction += h_LM * shift_contrib
                    end

                    # Slope terms: -∇_H h · uₜ
                    if config.include_slope_terms
                        # Poloidal contribution: -G^{(∇)} · ∂_r P
                        if abs(G_grad) > 1e-15
                            slope_P = -G_grad * (dP_dr / rb^2)
                            correction += h_LM * slope_P
                        end

                        # Toroidal contribution: -G^{(×)} · T
                        if abs(G_cross) > 1e-15
                            slope_T = -G_cross * T_lpm / rb^2
                            correction += h_LM * slope_T
                        end
                    end
                end
            end
        end
    end

    return correction
end

# ================================================================================
# No-Slip Correction
# ================================================================================

"""
    compute_noslip_correction(l, m, poloidal, toroidal, topo,
                              gaunt, rb, config) -> (Complex, Complex)

Compute topography correction to no-slip condition for mode (l, m).

The flat-sphere condition is: uₜ = 0

With topography:
uₜ + εh ∂_r uₜ = U_{b,t}

Returns (poloidal_correction, toroidal_correction).
"""
function compute_noslip_correction(l::Int, m::Int,
        p_cache::BoundaryDerivativeCache{T},
        t_cache::BoundaryDerivativeCache{T},
        topo::TopographyField{T},
        gaunt::GauntTensorCache{T},
        rb::T,
        location::BoundaryLocation,
        config::TopographyCouplingConfig) where {T}
    P_corr = zero(Complex{T})
    T_corr = zero(Complex{T})

    if !config.include_shift_terms
        return (P_corr, T_corr)
    end

    lmax = min(p_cache.lmax, gaunt.lmax)
    lmax_t = topo.lmax

    # For no-slip: uₜ + εh ∂_r uₜ = 0
    # We only correct the toroidal Dirichlet condition via T + εh ∂_r T = 0.

    for L in 0:lmax_t
        for M in -L:L
            h_LM = get_coefficient(topo, L, M)
            if abs(h_LM) < 1e-15
                continue
            end

            for lp in 1:lmax
                for mp in -lp:lp
                    if mp + M != m
                        continue
                    end

                    G = get_gaunt_tensor(gaunt, l, m, lp, mp, L, M)
                    if abs(G) < 1e-15
                        continue
                    end

                    # Toroidal shift: T + ε h ∂_r T = 0
                    dT_dr = get_cache_d1(t_cache, lp, mp, location)
                    T_corr += h_LM * G * dT_dr
                end
            end
        end
    end

    return (P_corr, T_corr)
end

# ================================================================================
# Stress-Free Correction
# ================================================================================

"""
    compute_stressfree_correction(l, m, poloidal, toroidal, topo,
                                  gaunt, rb, config) -> Complex

Compute topography correction to stress-free condition for mode (l, m).

The flat-sphere condition is: ∂_r(uₜ/r) = 0

With topography:
∂_r(uₜ/r) = (ε/r_b)(∇_H h) ∂_r uᵣ

The RHS involves the slope of topography coupling with radial velocity gradient.
"""
function compute_stressfree_correction(l::Int, m::Int,
        p_cache::BoundaryDerivativeCache{T},
        topo::TopographyField{T},
        gaunt::GauntTensorCache{T},
        rb::T,
        location::BoundaryLocation,
        config::TopographyCouplingConfig) where {T}
    correction = zero(Complex{T})

    if !config.include_slope_terms
        return correction
    end

    lmax = min(p_cache.lmax, gaunt.lmax)
    lmax_t = topo.lmax

    for L in 0:lmax_t
        for M in -L:L
            h_LM = get_coefficient(topo, L, M)
            if abs(h_LM) < 1e-15
                continue
            end

            for lp in 1:lmax
                for mp in -lp:lp
                    if mp + M != m
                        continue
                    end

                    # Use gradient Gaunt for slope term
                    G_grad = get_gradient_gaunt(gaunt, l, m, lp, mp, L, M)
                    if abs(G_grad) < 1e-15
                        continue
                    end

                    # Get ∂_r uᵣ at boundary
                    # uᵣ = l'(l'+1)/r² P, so ∂_r uᵣ involves ∂_r P and P
                    dP_dr = get_cache_d1(p_cache, lp, mp, location)
                    P_val = get_cache_value(p_cache, lp, mp, location)

                    ll_factor = T(lp * (lp + 1))
                    # ∂_r uᵣ = ∂_r[l'(l'+1)/r² P] = l'(l'+1)[∂_r P/r² - 2P/r³]
                    duᵣ_dr = ll_factor * (dP_dr / rb^2 - 2 * P_val / rb^3)

                    # Stress-free correction: (1/r_b) ∇_H h · (∂_r uᵣ)
                    correction += h_LM * G_grad * duᵣ_dr / (rb^2)
                end
            end
        end
    end

    return correction
end

# ================================================================================
# Utility Functions
# ================================================================================

"""
    lm_to_spectral_index(l::Int, m::Int, config) -> Int

Convert (l, m) to spectral array index using SHTnsKit convention.
"""
function lm_to_spectral_index(l::Int, m::Int, config)
    # Use the canonical (l,m)->index lookup via config.l_values/m_values
    # to match the convention used throughout the simulation
    abs_m = abs(m)
    return get_mode_index(config, l, abs_m)
end

"""
    get_spectral_boundary_value(field, l::Int, m::Int, location::BoundaryLocation=OUTER_BOUNDARY)

Get the boundary value of spectral coefficient (l, m).
"""
function get_spectral_boundary_value(field, l::Int, m::Int,
        location::BoundaryLocation = OUTER_BOUNDARY)
    idx = lm_to_spectral_index(l, abs(m), field.config)
    T = eltype(field.boundary_values)
    if idx <= 0 || idx > field.nlm
        return zero(Complex{T})
    end

    # Get from boundary_values (row 1 = inner, row 2 = outer)
    bc_row = location == INNER_BOUNDARY ? 1 : 2
    val_real = field.boundary_values[bc_row, idx]
    val_imag = zero(T)  # Boundary values are typically real

    if m >= 0
        return complex(val_real, val_imag)
    else
        # Conjugate relation for negative m
        phase = iseven(-m) ? one(T) : -one(T)
        return phase * conj(complex(val_real, val_imag))
    end
end

"""
    get_spectral_radial_derivative(field, l::Int, m::Int, r,
                                   location::BoundaryLocation=OUTER_BOUNDARY;
                                   ∂r, domain)

Compute the radial derivative of spectral coefficient (l, m) at a boundary.
Requires access to the radial derivative matrix and domain.
"""
function get_spectral_radial_derivative(field, l::Int, m::Int, r,
        location::BoundaryLocation = OUTER_BOUNDARY;
        ∂r = nothing, domain = nothing)
    if ∂r === nothing || domain === nothing
        throw(ArgumentError("get_spectral_radial_derivative requires ∂r and domain; use compute_boundary_derivative_cache and get_cache_d1"))
    end

    # Infer element type from field data
    T = eltype(parent(field.data_real))

    idx = lm_to_spectral_index(l, abs(m), field.config)
    if idx <= 0 || idx > field.nlm
        return zero(Complex{T})
    end

    nr = domain.N
    profile_real = zeros(T, nr)
    profile_imag = zeros(T, nr)
    data_real = parent(field.data_real)
    data_imag = parent(field.data_imag)
    lm_range = field.pencil.axes_local[1]
    r_range = field.pencil.axes_local[3]

    if idx in lm_range
        slot = local_spectral_storage_slot(field.config, idx)
        slot !== nothing && gather_local_radial_profile!(profile_real, profile_imag,
            data_real, data_imag, slot, r_range)
    end

    comm = get_comm()
    if comm !== nothing && MPI.Comm_size(comm) > 1
        MPI.Allreduce!(profile_real, MPI.SUM, comm)
        MPI.Allreduce!(profile_imag, MPI.SUM, comm)
    end

    dprofile_real = zeros(T, nr)
    dprofile_imag = zeros(T, nr)
    _apply_banded_matrix!(dprofile_real, ∂r, profile_real)
    _apply_banded_matrix!(dprofile_imag, ∂r, profile_imag)

    idx_r = location == INNER_BOUNDARY ? 1 : nr
    val = complex(dprofile_real[idx_r], dprofile_imag[idx_r])
    if m < 0
        phase = iseven(-m) ? one(T) : -one(T)
        val = phase * conj(val)
    end
    return val
end

"""
    is_stress_free_boundary(field, location::BoundaryLocation) -> Bool

Check if the field has stress-free boundary conditions at the given location.
"""
function is_stress_free_boundary(field, location::BoundaryLocation)
    # Check BC type - NEUMANN for toroidal typically means stress-free
    bc_array = location == INNER_BOUNDARY ? field.bc_type_inner : field.bc_type_outer

    # If any mode has NEUMANN BC, consider it stress-free
    return any(bc -> bc == Int(NEUMANN), bc_array)
end

"""
    is_no_slip_boundary(field, location::BoundaryLocation) -> Bool

Check if the field uses Dirichlet tangential conditions at the boundary.
"""
function is_no_slip_boundary(field, location::BoundaryLocation)
    bc_array = location == INNER_BOUNDARY ? field.bc_type_inner : field.bc_type_outer
    has_dirichlet = any(bc -> bc == Int(DIRICHLET), bc_array)
    has_neumann = any(bc -> bc == Int(NEUMANN), bc_array)
    return has_dirichlet && !has_neumann
end

# ================================================================================
# Boundary Operator Assembly
# ================================================================================

"""
    assemble_velocity_boundary_operator(l::Int, topo::TopographyField{T},
                                        gaunt::GauntTensorCache{T},
                                        config::TopographyCouplingConfig,
                                        bc_type::Symbol) where T

Assemble the boundary condition operator matrix row for mode l.

Returns a sparse representation of coupling coefficients to other modes.

# Arguments
- `l`: Target spherical harmonic degree
- `topo`: Topography field at boundary
- `gaunt`: Gaunt tensor cache
- `config`: Coupling configuration
- `bc_type`: :impermeability, :noslip, or :stressfree
"""
function assemble_velocity_boundary_operator(l::Int, topo::TopographyField{T},
        gaunt::GauntTensorCache{T},
        config::TopographyCouplingConfig,
        bc_type::Symbol) where {T}
    rb = topo.radius
    ε = config.epsilon
    lmax = gaunt.lmax
    lmax_t = topo.lmax

    # Storage for sparse operator entries
    # Format: Dict[(l', m', field)] => coefficient
    # where field is :P for poloidal or :T for toroidal
    operator = Dict{Tuple{Int, Int, Symbol}, Complex{T}}()

    # Flat-sphere contribution (diagonal)
    if bc_type == :impermeability
        # l(l+1)/r² P = 0
        for m in -l:l
            operator[(l, m, :P)] = T(l * (l + 1)) / rb^2
        end
    end

    # Topography coupling (off-diagonal)
    for m in -l:l
        for L in 0:lmax_t
            for M in -L:L
                h_LM = get_coefficient(topo, L, M)
                if abs(h_LM) < 1e-15
                    continue
                end

                for lp in 1:lmax
                    mp = m - M  # Selection rule
                    if abs(mp) > lp
                        continue
                    end

                    G = get_gaunt_tensor(gaunt, l, m, lp, mp, L, M)
                    G_grad = get_gradient_gaunt(gaunt, l, m, lp, mp, L, M)
                    G_cross = get_cross_gaunt(gaunt, l, m, lp, mp, L, M)

                    if bc_type == :impermeability
                        # Add topography corrections
                        if config.include_slope_terms
                            if abs(G_grad) > 1e-15
                                key = (lp, mp, :dP)  # Poloidal derivative
                                coeff = get(operator, key, zero(Complex{T}))
                                operator[key] = coeff + ε * h_LM * (-G_grad / rb^2)
                            end
                            if abs(G_cross) > 1e-15
                                key = (lp, mp, :T)   # Toroidal
                                coeff = get(operator, key, zero(Complex{T}))
                                operator[key] = coeff + ε * h_LM * (-G_cross / rb^2)
                            end
                        end
                    end
                end
            end
        end
    end

    return operator
end
