# ================================================================================
# Velocity Boundary Condition Topography Corrections
# ================================================================================
#
# This file implements topography corrections for velocity boundary conditions:
#
# 1. Impermeability (u·n̂ = 0):
#    u_r - ε(∇_H h)·u_t + εh ∂_r u_r = 0
#
# 2. No-slip (u_t = 0):
#    u_t + εh ∂_r u_t = U_b,t (typically 0 at CMB, Ω_ic × r at ICB)
#
# 3. Stress-free (∂_r(u_t/r) = 0):
#    ∂_r(u_t/r) = (ε/r_b)(∇_H h) ∂_r u_r
#
# In spectral (ℓ,m) space, the impermeability condition becomes:
#    ℓ(ℓ+1)/r_b² P_{u,ℓm} + ε Σ h_{LM} [α ∂_r P + β T_u + γ P] = 0
#
# ================================================================================

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

    # Precompute boundary value/derivative caches once for this field
    p_cache = compute_boundary_derivative_cache(poloidal,
                                                velocity_field.dr_matrix,
                                                velocity_field.d2r_matrix,
                                                velocity_field.domain)
    t_cache = compute_boundary_derivative_cache(toroidal,
                                                velocity_field.dr_matrix,
                                                velocity_field.d2r_matrix,
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
                                                location::BoundaryLocation) where T
    rb = topo_field.radius
    lmax = min(poloidal.config.lmax, gaunt.lmax)
    mmax = min(poloidal.config.mmax, gaunt.lmax)

    # Get boundary row index (1 for inner, 2 for outer)
    bc_row = location == INNER_BOUNDARY ? 1 : 2

    # Get poloidal/toroidal boundary values
    P_bv = poloidal.boundary_values
    T_bv = toroidal.boundary_values

    # Compute corrections for each (ℓ, m) mode
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
            # The flat-sphere BC is: ℓ(ℓ+1)/r² P = 0
            # With topography: ℓ(ℓ+1)/r² P + ε * correction = 0
            lm_idx = lm_to_spectral_index(l, m, poloidal.config)
            if lm_idx > 0 && lm_idx <= size(P_bv, 2)
                P_bv[bc_row, lm_idx] -= ε * real(imp_corr)
            end

            # Apply toroidal correction if stress-free
            if is_stress_free_boundary(toroidal, location)
                sf_corr = compute_stressfree_correction(
                    l, m, p_cache, topo_field, gaunt, rb, location, config
                )
                if lm_idx > 0 && lm_idx <= size(T_bv, 2)
                    T_bv[bc_row, lm_idx] -= ε * real(sf_corr)
                end
            end

            if is_no_slip_boundary(toroidal, location)
                _, ns_t = compute_noslip_correction(
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

The flat-sphere condition is: u_r = ℓ(ℓ+1)/r² P_{ℓm} = 0

With topography:
u_r - ε(∇_H h)·u_t + εh ∂_r u_r = 0

In spectral space (schematic):
ℓ(ℓ+1)/r² P_{ℓm} + ε Σ_{LM,ℓ'm'} h_{LM} [
    G · ∂_r(ℓ'(ℓ'+1)/r² P_{ℓ'm'})           # shift term
    - G^{(∇)} · ∂_r P_{ℓ'm'}                   # slope term (poloidal)
    - G^{(×)} · T_{ℓ'm'}                       # slope term (toroidal)
] = 0
"""
function compute_impermeability_correction(l::Int, m::Int,
                                           p_cache::BoundaryDerivativeCache{T},
                                           t_cache::BoundaryDerivativeCache{T},
                                           topo::TopographyField{T},
                                           gaunt::GauntTensorCache{T},
                                           rb::T,
                                           location::BoundaryLocation,
                                           config::TopographyCouplingConfig) where T
    correction = zero(Complex{T})

    lmax = min(p_cache.lmax, gaunt.lmax)
    lmax_t = topo.lmax

    # Sum over topography modes (L, M)
    for L in 0:lmax_t
        for M in -L:L
            h_LM = get_coefficient(topo, L, M)
            if abs(h_LM) < 1e-15
                continue
            end

            # Sum over field modes (ℓ', m')
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

                    # Shift term: h · ∂_r u_r
                    if config.include_shift_terms && abs(G) > 1e-15
                        # ∂_r(ℓ'(ℓ'+1)/r² P) evaluated at r = rb
                        shift_contrib = G * ll_factor * (dP_dr / rb^2 - 2 * P_lpm / rb^3)
                        correction += h_LM * shift_contrib
                    end

                    # Slope terms: -∇_H h · u_t
                    if config.include_slope_terms
                        # Poloidal contribution: -G^{(∇)} · ∂_r P
                        if abs(G_grad) > 1e-15
                            slope_P = -G_grad * (dP_dr / rb^2 + P_lpm / rb^3)
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

The flat-sphere condition is: u_t = 0

With topography:
u_t + εh ∂_r u_t = U_{b,t}

Returns (poloidal_correction, toroidal_correction).
"""
function compute_noslip_correction(l::Int, m::Int,
                                   p_cache::BoundaryDerivativeCache{T},
                                   t_cache::BoundaryDerivativeCache{T},
                                   topo::TopographyField{T},
                                   gaunt::GauntTensorCache{T},
                                   rb::T,
                                   location::BoundaryLocation,
                                   config::TopographyCouplingConfig) where T
    P_corr = zero(Complex{T})
    T_corr = zero(Complex{T})

    if !config.include_shift_terms
        return (P_corr, T_corr)
    end

    lmax = min(p_cache.lmax, gaunt.lmax)
    lmax_t = topo.lmax

    # For no-slip: u_t + εh ∂_r u_t = 0
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

The flat-sphere condition is: ∂_r(u_t/r) = 0

With topography:
∂_r(u_t/r) = (ε/r_b)(∇_H h) ∂_r u_r

The RHS involves the slope of topography coupling with radial velocity gradient.
"""
function compute_stressfree_correction(l::Int, m::Int,
                                       p_cache::BoundaryDerivativeCache{T},
                                       topo::TopographyField{T},
                                       gaunt::GauntTensorCache{T},
                                       rb::T,
                                       location::BoundaryLocation,
                                       config::TopographyCouplingConfig) where T
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

                    # Get ∂_r u_r at boundary
                    # u_r = ℓ'(ℓ'+1)/r² P, so ∂_r u_r involves ∂_r P and P
                    dP_dr = get_cache_d1(p_cache, lp, mp, location)
                    P_val = get_cache_value(p_cache, lp, mp, location)

                    ll_factor = T(lp * (lp + 1))
                    # ∂_r u_r = ∂_r[ℓ'(ℓ'+1)/r² P] = ℓ'(ℓ'+1)[∂_r P/r² - 2P/r³]
                    dur_dr = ll_factor * (dP_dr / rb^2 - 2 * P_val / rb^3)

                    # Stress-free correction: (1/r_b) ∇_H h · (∂_r u_r)
                    correction += h_LM * G_grad * dur_dr / (rb^2)
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
    # SHTnsKit uses a specific indexing convention
    # This should match the convention used in the main simulation
    if m >= 0
        # For m >= 0: standard triangular storage
        idx = l * (l + 1) ÷ 2 + m + 1
        return min(idx, config.nlm)
    else
        # For m < 0: map to positive m (conjugate relationship)
        return lm_to_spectral_index(l, -m, config)
    end
end

"""
    get_spectral_boundary_value(field, l::Int, m::Int, location::BoundaryLocation=OUTER_BOUNDARY)

Get the boundary value of spectral coefficient (l, m).
"""
function get_spectral_boundary_value(field, l::Int, m::Int,
                                     location::BoundaryLocation=OUTER_BOUNDARY)
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
    get_spectral_radial_derivative(field, l::Int, m::Int, r)

Estimate the radial derivative of spectral coefficient (l, m) at radius r.

This is an approximation using finite differences from the field data.
"""
function get_spectral_radial_derivative(field, l::Int, m::Int, r,
                                        location::BoundaryLocation=OUTER_BOUNDARY)
    # This is a placeholder - actual implementation needs access to radial grid
    # For now, return a simple estimate based on boundary value
    bv = get_spectral_boundary_value(field, l, m, location)

    # Simple estimate: derivative ~ value / shell_thickness
    # This should be replaced with proper finite difference from radial data
    return bv / 0.65  # Assuming typical shell thickness
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
                                             bc_type::Symbol) where T
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
        # ℓ(ℓ+1)/r² P = 0
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
