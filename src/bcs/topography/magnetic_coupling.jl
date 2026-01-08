# ================================================================================
# Magnetic Boundary Condition Topography Corrections
# ================================================================================
#
# This file implements topography corrections for magnetic boundary conditions:
#
# 1. CMB with insulating mantle:
#    - Toroidal: T_b(r_o) + εh_o ∂_r T_b(r_o) = 0
#    - Poloidal: ∂_r P_{b,ℓm} + (ℓ+1)/r_o P_{b,ℓm} + topography_couplings = 0
#
# 2. ICB with insulating inner core:
#    - Toroidal: T_b(r_i) + εh_i ∂_r T_b(r_i) = 0
#    - Poloidal: ∂_r P_{b,ℓm} - ℓ/r_i P_{b,ℓm} + topography_couplings = 0
#
# The flat-sphere magnetic BC conditions (insulating):
#    CMB: ∂_r P + (ℓ+1)/r_o P = 0,  T = 0
#    ICB: ∂_r P - ℓ/r_i P = 0,      T = 0
#
# ================================================================================

# ================================================================================
# Main Interface
# ================================================================================

"""
    apply_magnetic_topography_correction!(magnetic_field, topography::TopographyData,
                                          config::TopographyCouplingConfig)

Apply topography corrections to magnetic boundary conditions.

This modifies the boundary values of the magnetic field's poloidal and toroidal
components to account for topographic coupling.

# Arguments
- `magnetic_field`: SHTnsMagneticFields or similar structure with toroidal/poloidal fields
- `topography`: TopographyData containing ICB and/or CMB topography
- `config`: TopographyCouplingConfig with coupling parameters
"""
function apply_magnetic_topography_correction!(magnetic_field, topography::TopographyData,
                                               config::TopographyCouplingConfig)
    if !config.magnetic_coupling || !config.enabled
        return nothing
    end

    ε = config.epsilon
    gaunt = topography.gaunt_cache

    if gaunt === nothing
        @warn "Gaunt cache not initialized for magnetic topography correction"
        return nothing
    end

    # Get poloidal and toroidal components
    if hasfield(typeof(magnetic_field), :poloidal)
        poloidal = magnetic_field.poloidal
        toroidal = magnetic_field.toroidal
    elseif hasfield(typeof(magnetic_field), :P)
        poloidal = magnetic_field.P
        toroidal = magnetic_field.T
    else
        @warn "Cannot identify poloidal/toroidal components in magnetic field"
        return nothing
    end

    # Precompute boundary value/derivative caches once for this field
    p_cache = compute_boundary_derivative_cache(poloidal,
                                                magnetic_field.dr_matrix,
                                                magnetic_field.d2r_matrix,
                                                magnetic_field.outer_domain)
    t_cache = compute_boundary_derivative_cache(toroidal,
                                                magnetic_field.dr_matrix,
                                                magnetic_field.d2r_matrix,
                                                magnetic_field.outer_domain)

    # Apply corrections at ICB if topography defined
    if topography.icb !== nothing
        apply_magnetic_correction_at_boundary!(
            poloidal, toroidal, p_cache, t_cache,
            topography.icb, gaunt, ε, config, INNER_BOUNDARY, :insulating_inner
        )
    end

    # Apply corrections at CMB if topography defined
    if topography.cmb !== nothing
        apply_magnetic_correction_at_boundary!(
            poloidal, toroidal, p_cache, t_cache,
            topography.cmb, gaunt, ε, config, OUTER_BOUNDARY, :insulating_outer
        )
    end

    return nothing
end

# ================================================================================
# Boundary-Specific Corrections
# ================================================================================

"""
    apply_magnetic_correction_at_boundary!(poloidal, toroidal, topo_field,
                                           gaunt, ε, config, location, bc_type)

Apply magnetic topography corrections at a specific boundary.

# Arguments
- `bc_type`: :insulating_inner, :insulating_outer, or :conducting_inner
"""
function apply_magnetic_correction_at_boundary!(poloidal,
                                                toroidal,
                                                p_cache::BoundaryDerivativeCache{T},
                                                t_cache::BoundaryDerivativeCache{T},
                                                topo_field::TopographyField{T},
                                                gaunt::GauntTensorCache{T},
                                                ε::T,
                                                config::TopographyCouplingConfig,
                                                location::BoundaryLocation,
                                                bc_type::Symbol) where T
    rb = topo_field.radius
    lmax = min(poloidal.config.lmax, gaunt.lmax)
    mmax = min(poloidal.config.mmax, gaunt.lmax)

    # Get boundary row index (1 for inner, 2 for outer)
    bc_row = location == INNER_BOUNDARY ? 1 : 2

    # Get poloidal/toroidal boundary values
    P_bv = poloidal.boundary_values
    T_bv = toroidal.boundary_values

    # Compute corrections for each (ℓ, m) mode
    for l in 1:lmax  # Start from l=1 (l=0 has no magnetic dipole)
        for m in -l:l
            if abs(m) > mmax
                continue
            end

            lm_idx = lm_to_spectral_index(l, m, poloidal.config)
            if lm_idx <= 0 || lm_idx > size(P_bv, 2)
                continue
            end

            if bc_type == :insulating_outer
                # CMB insulating: ∂_r P + (ℓ+1)/r_o P = 0, T = 0
                P_corr, T_corr = compute_cmb_insulating_correction(
                    l, m, p_cache, t_cache, topo_field, gaunt, rb, location, config
                )
                P_bv[bc_row, lm_idx] -= ε * real(P_corr)
                T_bv[bc_row, lm_idx] -= ε * real(T_corr)

            elseif bc_type == :insulating_inner
                # ICB insulating: ∂_r P - ℓ/r_i P = 0, T = 0
                P_corr, T_corr = compute_icb_insulating_correction(
                    l, m, p_cache, t_cache, topo_field, gaunt, rb, location, config
                )
                P_bv[bc_row, lm_idx] -= ε * real(P_corr)
                T_bv[bc_row, lm_idx] -= ε * real(T_corr)
            end
        end
    end

    return nothing
end

# ================================================================================
# CMB Insulating Correction
# ================================================================================

"""
    compute_cmb_insulating_correction(l, m, poloidal, toroidal, topo,
                                      gaunt, ro, config) -> (P_corr, T_corr)

Compute topography correction to CMB insulating boundary condition.

Flat-sphere conditions:
- ∂_r P_{b,ℓm} + (ℓ+1)/r_o P_{b,ℓm} = 0
- T_{b,ℓm} = 0

With topography, the poloidal condition becomes:
[∂_r P + (ℓ+1)/r_o P]_{ℓm} + ε Σ h^o_{LM} (α^o ∂_r P + β^o T + γ^o P) = 0

and toroidal:
T + εh_o ∂_r T = 0
"""
function compute_cmb_insulating_correction(l::Int, m::Int,
                                           p_cache::BoundaryDerivativeCache{T},
                                           t_cache::BoundaryDerivativeCache{T},
                                           topo::TopographyField{T},
                                           gaunt::GauntTensorCache{T},
                                           ro::T,
                                           location::BoundaryLocation,
                                           config::TopographyCouplingConfig) where T
    P_correction = zero(Complex{T})
    T_correction = zero(Complex{T})

    lmax = min(p_cache.lmax, gaunt.lmax)
    lmax_t = topo.lmax

    # Toroidal shift correction: T + εh ∂_r T = 0
    # For flat T = 0, the correction is εh ∂_r T
    if config.include_shift_terms
        for L in 0:lmax_t
            for M in -L:L
                h_LM = get_coefficient(topo, L, M)
                if abs(h_LM) < 1e-15
                    continue
                end

                for lp in 1:lmax
                    mp = m - M
                    if abs(mp) > lp
                        continue
                    end

                    G = get_gaunt_tensor(gaunt, l, m, lp, mp, L, M)
                    if abs(G) < 1e-15
                        continue
                    end

                    # Get ∂_r T at CMB
                    dT_dr = get_cache_d1(t_cache, lp, mp, location)
                    T_correction += h_LM * G * dT_dr
                end
            end
        end
    end

    # Poloidal coupling through topography
    for L in 0:lmax_t
        for M in -L:L
            h_LM = get_coefficient(topo, L, M)
            if abs(h_LM) < 1e-15
                continue
            end

            for lp in 1:lmax
                mp = m - M
                if abs(mp) > lp
                    continue
                end

                G = get_gaunt_tensor(gaunt, l, m, lp, mp, L, M)
                G_grad = get_gradient_gaunt(gaunt, l, m, lp, mp, L, M)
                G_cross = get_cross_gaunt(gaunt, l, m, lp, mp, L, M)

                # Get field values
                P_val = get_cache_value(p_cache, lp, mp, location)
                T_val = get_cache_value(t_cache, lp, mp, location)
                dP_dr = get_cache_d1(p_cache, lp, mp, location)
                d2P_dr2 = get_cache_d2(p_cache, lp, mp, location)

                if config.include_shift_terms && abs(G) > 1e-15
                    # Shift term: h · ∂r(∂r P + (ℓ+1)P/r)
                    shift_term = d2P_dr2 + (lp + 1) * dP_dr / ro - (lp + 1) * P_val / ro^2
                    P_correction += h_LM * G * shift_term
                end

                if config.include_slope_terms
                    # Poloidal slope coupling: (∇_H h)·∇_H(∂_r P)
                    # Uses G^{(∇)} with surface gradient scaling 1/r²
                    if abs(G_grad) > 1e-15
                        P_correction -= h_LM * G_grad * dP_dr / ro^2
                    end

                    # Toroidal coupling from tangential matching: (∇_H h)·(r̂ × ∇_H T)
                    if abs(G_cross) > 1e-15
                        P_correction -= h_LM * G_cross * T_val / ro^2
                    end
                end
            end
        end
    end

    return (P_correction, T_correction)
end

# ================================================================================
# ICB Insulating Correction
# ================================================================================

"""
    compute_icb_insulating_correction(l, m, poloidal, toroidal, topo,
                                      gaunt, ri, config) -> (P_corr, T_corr)

Compute topography correction to ICB insulating boundary condition.

Flat-sphere conditions:
- ∂_r P_{b,ℓm} - ℓ/r_i P_{b,ℓm} = 0
- T_{b,ℓm} = 0

With topography, the poloidal condition becomes:
[∂_r P - ℓ/r_i P]_{ℓm} + ε Σ h^i_{LM} (α^i ∂_r P + β^i T + γ^i P) = 0
"""
function compute_icb_insulating_correction(l::Int, m::Int,
                                           p_cache::BoundaryDerivativeCache{T},
                                           t_cache::BoundaryDerivativeCache{T},
                                           topo::TopographyField{T},
                                           gaunt::GauntTensorCache{T},
                                           ri::T,
                                           location::BoundaryLocation,
                                           config::TopographyCouplingConfig) where T
    P_correction = zero(Complex{T})
    T_correction = zero(Complex{T})

    lmax = min(p_cache.lmax, gaunt.lmax)
    lmax_t = topo.lmax

    # Toroidal shift correction (same as CMB)
    if config.include_shift_terms
        for L in 0:lmax_t
            for M in -L:L
                h_LM = get_coefficient(topo, L, M)
                if abs(h_LM) < 1e-15
                    continue
                end

                for lp in 1:lmax
                    mp = m - M
                    if abs(mp) > lp
                        continue
                    end

                    G = get_gaunt_tensor(gaunt, l, m, lp, mp, L, M)
                    if abs(G) < 1e-15
                        continue
                    end

                    dT_dr = get_cache_d1(t_cache, lp, mp, location)
                    T_correction += h_LM * G * dT_dr
                end
            end
        end
    end

    # Poloidal coupling
    for L in 0:lmax_t
        for M in -L:L
            h_LM = get_coefficient(topo, L, M)
            if abs(h_LM) < 1e-15
                continue
            end

            for lp in 1:lmax
                mp = m - M
                if abs(mp) > lp
                    continue
                end

                G = get_gaunt_tensor(gaunt, l, m, lp, mp, L, M)
                G_grad = get_gradient_gaunt(gaunt, l, m, lp, mp, L, M)
                G_cross = get_cross_gaunt(gaunt, l, m, lp, mp, L, M)

                P_val = get_cache_value(p_cache, lp, mp, location)
                T_val = get_cache_value(t_cache, lp, mp, location)
                dP_dr = get_cache_d1(p_cache, lp, mp, location)
                d2P_dr2 = get_cache_d2(p_cache, lp, mp, location)

                if config.include_shift_terms && abs(G) > 1e-15
                    # Shift term: h · ∂r(∂r P - ℓ P/r)
                    shift_term = d2P_dr2 - lp * dP_dr / ri + lp * P_val / ri^2
                    P_correction += h_LM * G * shift_term
                end

                if config.include_slope_terms
                    # Poloidal slope coupling: (∇_H h)·∇_H(∂_r P)
                    # Uses G^{(∇)} with surface gradient scaling 1/r²
                    if abs(G_grad) > 1e-15
                        P_correction -= h_LM * G_grad * dP_dr / ri^2
                    end

                    # Toroidal coupling from tangential matching
                    if abs(G_cross) > 1e-15
                        P_correction -= h_LM * G_cross * T_val / ri^2
                    end
                end
            end
        end
    end

    return (P_correction, T_correction)
end

# ================================================================================
# Boundary Operator Assembly for Magnetic Field
# ================================================================================

"""
    assemble_magnetic_boundary_operator(l::Int, topo::TopographyField{T},
                                        gaunt::GauntTensorCache{T},
                                        config::TopographyCouplingConfig,
                                        location::BoundaryLocation) where T

Assemble the magnetic boundary condition operator matrix row for mode l.

Returns a sparse representation of coupling coefficients.

# Arguments
- `l`: Target spherical harmonic degree
- `topo`: Topography field at boundary
- `gaunt`: Gaunt tensor cache
- `config`: Coupling configuration
- `location`: INNER_BOUNDARY (ICB) or OUTER_BOUNDARY (CMB)
"""
function assemble_magnetic_boundary_operator(l::Int, topo::TopographyField{T},
                                             gaunt::GauntTensorCache{T},
                                             config::TopographyCouplingConfig,
                                             location::BoundaryLocation) where T
    rb = topo.radius
    ε = config.epsilon
    lmax = gaunt.lmax
    lmax_t = topo.lmax

    # Storage for sparse operator entries
    operator = Dict{Tuple{Int, Int, Symbol}, Complex{T}}()

    # Flat-sphere contribution (diagonal)
    for m in -l:l
        if location == OUTER_BOUNDARY
            # CMB: ∂_r P + (ℓ+1)/r_o P = 0
            operator[(l, m, :dP)] = one(Complex{T})
            operator[(l, m, :P)] = Complex{T}((l + 1) / rb)
        else
            # ICB: ∂_r P - ℓ/r_i P = 0
            operator[(l, m, :dP)] = one(Complex{T})
            operator[(l, m, :P)] = Complex{T}(-l / rb)
        end
        # Toroidal: T = 0
        operator[(l, m, :T)] = one(Complex{T})
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
                    mp = m - M
                    if abs(mp) > lp
                        continue
                    end

                    G = get_gaunt_tensor(gaunt, l, m, lp, mp, L, M)
                    G_grad = get_gradient_gaunt(gaunt, l, m, lp, mp, L, M)
                    G_cross = get_cross_gaunt(gaunt, l, m, lp, mp, L, M)

                    if config.include_shift_terms && abs(G) > 1e-15
                        # Add shift term corrections
                        key_dP = (lp, mp, :dP)
                        coeff = get(operator, key_dP, zero(Complex{T}))
                        operator[key_dP] = coeff + ε * h_LM * G

                        key_P = (lp, mp, :P)
                        coeff = get(operator, key_P, zero(Complex{T}))
                        sign = location == OUTER_BOUNDARY ? (lp + 1) : -lp
                        operator[key_P] = coeff + ε * h_LM * G * T(sign) / rb
                    end

                    if config.include_slope_terms
                        # Add poloidal slope term: (∇_H h)·∇_H(∂_r P)
                        if abs(G_grad) > 1e-15
                            key_dP = (lp, mp, :dP)
                            coeff = get(operator, key_dP, zero(Complex{T}))
                            operator[key_dP] = coeff - ε * h_LM * G_grad / rb^2
                        end

                        # Add toroidal slope term: (∇_H h)·(r̂ × ∇_H T)
                        if abs(G_cross) > 1e-15
                            key_T = (lp, mp, :T)
                            coeff = get(operator, key_T, zero(Complex{T}))
                            operator[key_T] = coeff - ε * h_LM * G_cross / rb^2
                        end
                    end
                end
            end
        end
    end

    return operator
end

# ================================================================================
# Magnetic Potential Matching with Topography
# ================================================================================

"""
    compute_potential_matching_coefficients(l::Int, topo::TopographyField{T},
                                           gaunt::GauntTensorCache{T},
                                           location::BoundaryLocation) where T

Compute the coefficients for matching the core magnetic field to an
external potential field across a topographic boundary.

For an insulating exterior, B = -∇Φ where:
- Φ_ext = Σ a_{ℓm} r^{-(ℓ+1)} Y_ℓ^m  (exterior, r > r_o)
- Φ_int = Σ b_{ℓm} r^ℓ Y_ℓ^m         (interior, r < r_i)

Continuity of B_t at the true surface introduces topography coupling.
"""
function compute_potential_matching_coefficients(l::Int, topo::TopographyField{T},
                                                 gaunt::GauntTensorCache{T},
                                                 location::BoundaryLocation) where T
    rb = topo.radius
    ε = one(T) * 0.01  # Default epsilon
    lmax_t = topo.lmax

    # Matching coefficients for each (ℓ', m')
    coeffs = Dict{Tuple{Int, Int}, Complex{T}}()

    # The flat matching condition at CMB (exterior insulating):
    # a_ℓm = P_ℓm / r_o^{ℓ+2}  (relates to poloidal field)

    # With topography, modes couple through Gaunt tensors
    for L in 0:lmax_t
        for M in -L:L
            h_LM = get_coefficient(topo, L, M)
            if abs(h_LM) < 1e-15
                continue
            end

            for m in -l:l
                mp = m - M
                if abs(mp) > l
                    continue
                end

                G = get_gaunt_tensor(gaunt, l, m, l, mp, L, M)
                if abs(G) > 1e-15
                    key = (l, mp)
                    coeff = get(coeffs, key, zero(Complex{T}))
                    coeffs[key] = coeff + ε * h_LM * G
                end
            end
        end
    end

    return coeffs
end
