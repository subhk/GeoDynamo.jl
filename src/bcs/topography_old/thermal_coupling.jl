# ================================================================================
# Thermal Boundary Condition Topography Corrections
# ================================================================================
#
# This file implements topography corrections for thermal boundary conditions:
#
# 1. Dirichlet (fixed temperature):
#    Θ(r_b) + εh_b ∂_r Θ(r_b) = T_b(θ,φ,t) - T_cond(r_b) - εh_b ∂_r T_cond(r_b)
#
# 2. Neumann (fixed flux -k ∂_n T = q_b):
#    -k[∂_r Θ - ε∇_H h·∇_H Θ + εh ∂_rr Θ] = q_b + k[∂_r T_cond + εh ∂_rr T_cond]
#
# In spectral form (from PDF equations 38-39):
#
# Dirichlet:
#   Θ_{ℓm} + ε Σ h^b_{LM} G_{ℓm,ℓ'm',LM} ∂_r Θ_{ℓ'm'}
#       = [T_b - T_cond(r_b)]_{ℓm} - ε Σ h^b_{LM} G_{ℓm,00,LM} ∂_r T_cond
#
# Neumann:
#   ∂_r Θ_{ℓm} - ε Σ h^b_{LM} G^{(∇)}_{ℓm,ℓ'm',LM} Θ_{ℓ'm'}
#              + ε Σ h^b_{LM} G_{ℓm,ℓ'm',LM} ∂_rr Θ_{ℓ'm'}
#       = -q_{b,ℓm}/k - ∂_r T_cond δ_{ℓ0}δ_{m0} - ε Σ h^b_{LM} G_{ℓm,00,LM} ∂_rr T_cond
#
# ================================================================================

# ================================================================================
# Main Interface
# ================================================================================

"""
    apply_thermal_topography_correction!(temperature_field, topography::TopographyData,
                                         config::TopographyCouplingConfig;
                                         T_cond::Union{Function, Nothing}=nothing)

Apply topography corrections to thermal boundary conditions.

# Arguments
- `temperature_field`: SHTnsTemperatureField or similar scalar field structure
- `topography`: TopographyData containing ICB and/or CMB topography
- `config`: TopographyCouplingConfig with coupling parameters
- `T_cond`: Optional conductive temperature profile function T_cond(r)
"""
function apply_thermal_topography_correction!(temperature_field, topography::TopographyData,
                                              config::TopographyCouplingConfig;
                                              T_cond::Union{Function, Nothing}=nothing)
    if !config.thermal_coupling || !config.enabled
        return nothing
    end

    ε = config.epsilon
    gaunt = topography.gaunt_cache

    if gaunt === nothing
        @warn "Gaunt cache not initialized for thermal topography correction"
        return nothing
    end

    # Get the spectral field (temperature is stored as perturbation Θ = T - T_cond)
    if hasfield(typeof(temperature_field), :spectral)
        spectral = temperature_field.spectral
    elseif temperature_field isa SHTnsSpectralField
        spectral = temperature_field
    else
        @warn "Cannot identify spectral component in temperature field"
        return nothing
    end

    # Prefer BC metadata from the parent scalar field when available
    bc_inner = hasfield(typeof(temperature_field), :bc_type_inner) ?
        temperature_field.bc_type_inner : spectral.bc_type_inner
    bc_outer = hasfield(typeof(temperature_field), :bc_type_outer) ?
        temperature_field.bc_type_outer : spectral.bc_type_outer
    bv = hasfield(typeof(temperature_field), :boundary_values) ?
        temperature_field.boundary_values : spectral.boundary_values

    # Radial derivative metadata for accurate coupling terms
    dr_matrix = hasfield(typeof(temperature_field), :dr_matrix) ?
        temperature_field.dr_matrix : nothing
    d2r_matrix = hasfield(typeof(temperature_field), :d2r_matrix) ?
        temperature_field.d2r_matrix : nothing
    domain = hasfield(typeof(temperature_field), :domain) ?
        temperature_field.domain : nothing

    cache = nothing
    if dr_matrix !== nothing && domain !== nothing
        cache = compute_boundary_derivative_cache(spectral, dr_matrix, d2r_matrix, domain)
    else
        @warn "Missing radial derivative metadata; skipping thermal topography coupling"
        return nothing
    end

    # Apply corrections at ICB if topography defined
    if topography.icb !== nothing
        apply_thermal_correction_at_boundary!(
            spectral, cache, bv, bc_inner, bc_outer,
            topography.icb, gaunt, ε, config, INNER_BOUNDARY, T_cond
        )
    end

    # Apply corrections at CMB if topography defined
    if topography.cmb !== nothing
        apply_thermal_correction_at_boundary!(
            spectral, cache, bv, bc_inner, bc_outer,
            topography.cmb, gaunt, ε, config, OUTER_BOUNDARY, T_cond
        )
    end

    return nothing
end

# ================================================================================
# Boundary-Specific Corrections
# ================================================================================

"""
    apply_thermal_correction_at_boundary!(spectral, cache, boundary_values,
                                          bc_inner, bc_outer, topo_field,
                                          gaunt, ε, config, location, T_cond)

Apply thermal topography corrections at a specific boundary.
"""
function apply_thermal_correction_at_boundary!(spectral,
                                               cache::Union{BoundaryDerivativeCache{T}, Nothing},
                                               boundary_values::AbstractMatrix{T},
                                               bc_inner::AbstractVector{Int},
                                               bc_outer::AbstractVector{Int},
                                               topo_field::TopographyField{T},
                                               gaunt::GauntTensorCache{T},
                                               ε::T,
                                               config::TopographyCouplingConfig,
                                               location::BoundaryLocation,
                                               T_cond::Union{Function, Nothing}) where T
    rb = topo_field.radius
    lmax = min(spectral.config.lmax, gaunt.lmax)
    mmax = min(spectral.config.mmax, gaunt.lmax)

    # Get boundary row index (1 for inner, 2 for outer)
    bc_row = location == INNER_BOUNDARY ? 1 : 2

    # Determine BC type from field
    bc_type_array = location == INNER_BOUNDARY ? bc_inner : bc_outer

    # Boundary values storage
    bv = boundary_values

    # Conductive profile contributions (if provided)
    dTcond_dr = zero(T)
    d2Tcond_dr2 = zero(T)
    if T_cond !== nothing
        # Estimate derivatives numerically
        dr = T(0.001)
        Tcond_rb = T_cond(rb)
        dTcond_dr = (T_cond(rb + dr) - T_cond(rb - dr)) / (2dr)
        d2Tcond_dr2 = (T_cond(rb + dr) - 2Tcond_rb + T_cond(rb - dr)) / dr^2
    end

    # Compute corrections for each (ℓ, m) mode
    for l in 0:lmax
        for m in -l:l
            if abs(m) > mmax
                continue
            end

            lm_idx = lm_to_spectral_index(l, m, spectral.config)
            if lm_idx <= 0 || lm_idx > size(bv, 2)
                continue
            end

            # Determine if Dirichlet or Neumann
            is_dirichlet = bc_type_array[lm_idx] == Int(DIRICHLET)

            if is_dirichlet
                correction = compute_dirichlet_thermal_correction(
                    l, m, spectral, cache, topo_field, gaunt, rb, location,
                    config, dTcond_dr
                )
            else
                correction = compute_neumann_thermal_correction(
                    l, m, spectral, cache, topo_field, gaunt, rb, location,
                    config, dTcond_dr, d2Tcond_dr2
                )
            end

            bv[bc_row, lm_idx] -= ε * real(correction)
        end
    end

    return nothing
end

# ================================================================================
# Dirichlet Thermal Correction
# ================================================================================

"""
    compute_dirichlet_thermal_correction(l, m, spectral, topo, gaunt, rb, config, dTcond_dr)

Compute topography correction to Dirichlet thermal BC for mode (l, m).

From PDF equation 38:
Θ_{ℓm} + ε Σ h^b_{LM} G_{ℓm,ℓ'm',LM} ∂_r Θ_{ℓ'm'}
    = [T_b - T_cond(r_b)]_{ℓm} - ε Σ h^b_{LM} G_{ℓm,00,LM} ∂_r T_cond

The correction to the RHS involves:
1. Shift term: h · ∂_r Θ (couples modes)
2. Conductive correction: h · ∂_r T_cond (modifies boundary value)
"""
function compute_dirichlet_thermal_correction(l::Int, m::Int,
                                              spectral,
                                              cache::Union{BoundaryDerivativeCache{T}, Nothing},
                                              topo::TopographyField{T},
                                              gaunt::GauntTensorCache{T},
                                              rb::T,
                                              location::BoundaryLocation,
                                              config::TopographyCouplingConfig,
                                              dTcond_dr::T) where T
    correction = zero(Complex{T})

    lmax = min(spectral.config.lmax, gaunt.lmax)
    lmax_t = topo.lmax

    if !config.include_shift_terms
        return correction
    end

    if cache === nothing
        @warn "Missing boundary derivative cache; skipping thermal shift term"
    end

    # Sum over topography modes
    for L in 0:lmax_t
        for M in -L:L
            h_LM = get_coefficient(topo, L, M)
            if abs(h_LM) < 1e-15
                continue
            end

            # Mode coupling: h · ∂_r Θ
            for lp in 0:lmax
                for mp in -lp:lp
                    if mp + M != m
                        continue
                    end

                    G = get_gaunt_tensor(gaunt, l, m, lp, mp, L, M)
                    if abs(G) < 1e-15
                        continue
                    end

                    # Get ∂_r Θ at boundary
                    cache === nothing && continue
                    dTheta_dr = get_cache_d1(cache, lp, mp, location)
                    correction += h_LM * G * dTheta_dr
                end
            end

            # Conductive profile correction
            G_00 = get_gaunt_tensor(gaunt, l, m, 0, 0, L, M)
            if abs(G_00) > 1e-15
                correction += h_LM * G_00 * dTcond_dr
            end
        end
    end

    return correction
end

# ================================================================================
# Neumann Thermal Correction
# ================================================================================

"""
    compute_neumann_thermal_correction(l, m, spectral, topo, gaunt, rb, config,
                                       dTcond_dr, d2Tcond_dr2)

Compute topography correction to Neumann thermal BC for mode (l, m).

From PDF equation 39:
∂_r Θ_{ℓm} - ε Σ h^b_{LM} G^{(∇)}_{ℓm,ℓ'm',LM} Θ_{ℓ'm'}
           + ε Σ h^b_{LM} G_{ℓm,ℓ'm',LM} ∂_rr Θ_{ℓ'm'}
    = -q_{b,ℓm}/k - ∂_r T_cond δ_{ℓ0}δ_{m0} - ε Σ h^b_{LM} G_{ℓm,00,LM} ∂_rr T_cond

The correction involves:
1. Slope term: -∇_H h · ∇_H Θ (gradient coupling)
2. Shift term: h · ∂_rr Θ (second derivative coupling)
3. Conductive correction: h · ∂_rr T_cond
"""
function compute_neumann_thermal_correction(l::Int, m::Int,
                                            spectral,
                                            cache::Union{BoundaryDerivativeCache{T}, Nothing},
                                            topo::TopographyField{T},
                                            gaunt::GauntTensorCache{T},
                                            rb::T,
                                            location::BoundaryLocation,
                                            config::TopographyCouplingConfig,
                                            dTcond_dr::T,
                                            d2Tcond_dr2::T) where T
    correction = zero(Complex{T})

    lmax = min(spectral.config.lmax, gaunt.lmax)
    lmax_t = topo.lmax
    warned_missing_d2 = false

    # Sum over topography modes
    for L in 0:lmax_t
        for M in -L:L
            h_LM = get_coefficient(topo, L, M)
            if abs(h_LM) < 1e-15
                continue
            end

            for lp in 0:lmax
                for mp in -lp:lp
                    if mp + M != m
                        continue
                    end

                    G = get_gaunt_tensor(gaunt, l, m, lp, mp, L, M)
                    G_grad = get_gradient_gaunt(gaunt, l, m, lp, mp, L, M)

                    if cache === nothing
                        Theta_val = get_spectral_boundary_value(spectral, lp, mp, location)
                    else
                        Theta_val = get_cache_value(cache, lp, mp, location)
                    end

                    # Slope term: -G^{(∇)} · Θ
                    if config.include_slope_terms && abs(G_grad) > 1e-15
                        correction -= h_LM * G_grad * Theta_val / rb^2
                    end

                    # Shift term: G · ∂_rr Θ (requires second-derivative cache)
                    if config.include_shift_terms && abs(G) > 1e-15
                        if cache === nothing || cache.d2_inner === nothing
                            if !warned_missing_d2
                                @warn "Missing second-derivative cache; skipping thermal shift term"
                                warned_missing_d2 = true
                            end
                        else
                            d2Theta_dr2 = get_cache_d2(cache, lp, mp, location)
                            correction += h_LM * G * d2Theta_dr2
                        end
                    end
                end
            end

            # Conductive profile correction
            G_00 = get_gaunt_tensor(gaunt, l, m, 0, 0, L, M)
            if abs(G_00) > 1e-15 && config.include_shift_terms
                correction += h_LM * G_00 * d2Tcond_dr2
            end
        end
    end

    return correction
end

# ================================================================================
# Utility Functions
# ================================================================================


# ================================================================================
# Compositional Field Support
# ================================================================================

"""
    apply_composition_topography_correction!(composition_field, topography::TopographyData,
                                             config::TopographyCouplingConfig)

Apply topography corrections to compositional boundary conditions.

Composition follows the same mathematical structure as temperature,
so we can reuse the thermal correction functions.
"""
function apply_composition_topography_correction!(composition_field, topography::TopographyData,
                                                  config::TopographyCouplingConfig)
    # Composition uses the same equations as temperature (advection-diffusion)
    # Default: no conductive profile for composition
    apply_thermal_topography_correction!(composition_field, topography, config;
                                         T_cond=nothing)
end

# ================================================================================
# Boundary Operator Assembly for Thermal Field
# ================================================================================

"""
    assemble_thermal_boundary_operator(l::Int, topo::TopographyField{T},
                                       gaunt::GauntTensorCache{T},
                                       config::TopographyCouplingConfig,
                                       bc_type::Symbol) where T

Assemble the thermal boundary condition operator matrix row for mode l.

# Arguments
- `l`: Target spherical harmonic degree
- `topo`: Topography field at boundary
- `gaunt`: Gaunt tensor cache
- `config`: Coupling configuration
- `bc_type`: :dirichlet or :neumann
"""
function assemble_thermal_boundary_operator(l::Int, topo::TopographyField{T},
                                            gaunt::GauntTensorCache{T},
                                            config::TopographyCouplingConfig,
                                            bc_type::Symbol) where T
    rb = topo.radius
    ε = config.epsilon
    lmax = gaunt.lmax
    lmax_t = topo.lmax

    # Storage for sparse operator entries
    # Format: Dict[(l', m', field)] => coefficient
    # where field is :Θ for temperature or :dΘ for derivative
    operator = Dict{Tuple{Int, Int, Symbol}, Complex{T}}()

    # Flat-sphere contribution (diagonal)
    for m in -l:l
        if bc_type == :dirichlet
            # Θ = T_b at boundary
            operator[(l, m, :Θ)] = one(Complex{T})
        else
            # ∂_r Θ = -q_b/k at boundary
            operator[(l, m, :dΘ)] = one(Complex{T})
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

                for lp in 0:lmax
                    mp = m - M
                    if abs(mp) > lp
                        continue
                    end

                    G = get_gaunt_tensor(gaunt, l, m, lp, mp, L, M)
                    G_grad = get_gradient_gaunt(gaunt, l, m, lp, mp, L, M)

                    if bc_type == :dirichlet
                        # Add shift term: G · ∂_r Θ
                        if config.include_shift_terms && abs(G) > 1e-15
                            key = (lp, mp, :dΘ)
                            coeff = get(operator, key, zero(Complex{T}))
                            operator[key] = coeff + ε * h_LM * G
                        end
                    else  # Neumann
                        # Add slope term: -G^{(∇)} · Θ
                        if config.include_slope_terms && abs(G_grad) > 1e-15
                            key = (lp, mp, :Θ)
                            coeff = get(operator, key, zero(Complex{T}))
                            operator[key] = coeff - ε * h_LM * G_grad / rb^2
                        end
                        # Add shift term: G · ∂_rr Θ
                        if config.include_shift_terms && abs(G) > 1e-15
                            key = (lp, mp, :d2Θ)
                            coeff = get(operator, key, zero(Complex{T}))
                            operator[key] = coeff + ε * h_LM * G
                        end
                    end
                end
            end
        end
    end

    return operator
end

# ================================================================================
# Heat Flux Computation with Topography
# ================================================================================

"""
    compute_boundary_heat_flux(temperature_field, topography::TopographyData,
                               config::TopographyCouplingConfig,
                               location::BoundaryLocation;
                               k::Float64=1.0) -> Matrix

Compute the heat flux at a topographic boundary.

The normal heat flux on the true surface (to first order) is:
q_n = -k [∂_r T - ε ∇_H h · ∇_H T + εh ∂_rr T]

Returns heat flux on the (θ, φ) grid.
"""
function compute_boundary_heat_flux(temperature_field, topography::TopographyData,
                                    config::TopographyCouplingConfig,
                                    location::BoundaryLocation;
                                    k::Float64=1.0)
    ε = config.epsilon
    topo_field = location == INNER_BOUNDARY ? topography.icb : topography.cmb

    if topo_field === nothing
        @warn "No topography defined at $(location)"
        return nothing
    end

    # Get spectral field
    if hasfield(typeof(temperature_field), :spectral)
        spectral = temperature_field.spectral
    else
        spectral = temperature_field
    end

    rb = topo_field.radius
    config_sht = spectral.config
    nlat, nlon = config_sht.nlat, config_sht.nlon

    dr_matrix = hasfield(typeof(temperature_field), :dr_matrix) ? temperature_field.dr_matrix : nothing
    d2r_matrix = hasfield(typeof(temperature_field), :d2r_matrix) ? temperature_field.d2r_matrix : nothing
    domain = hasfield(typeof(temperature_field), :domain) ? temperature_field.domain : nothing
    if dr_matrix === nothing || domain === nothing
        @warn "Missing radial derivative metadata; cannot compute boundary heat flux"
        return nothing
    end

    cache = compute_boundary_derivative_cache(spectral, dr_matrix, d2r_matrix, domain)

    gaunt = topography.gaunt_cache
    if gaunt === nothing && (config.include_slope_terms || config.include_shift_terms)
        @warn "Gaunt cache not initialized; computing flat-sphere heat flux only"
    end

    lmax = spectral.config.lmax
    mmax = spectral.config.mmax
    lmax_coupling = gaunt === nothing ? -1 : min(lmax, gaunt.lmax)
    lmax_t = gaunt === nothing ? -1 : topo_field.lmax

    T = typeof(real(zero(eltype(cache.value_inner))))
    coeffs = zeros(Complex{T}, spectral.config.nlm)
    warned_missing_d2 = false

    for l in 0:lmax
        for m in 0:min(l, mmax)
            dT_dr = get_cache_d1(cache, l, m, location)
            slope_sum = zero(Complex{T})
            shift_sum = zero(Complex{T})

            if gaunt !== nothing && l <= lmax_coupling &&
                (config.include_slope_terms || config.include_shift_terms)
                for L in 0:lmax_t
                    for M in -L:L
                        h_LM = get_coefficient(topo_field, L, M)
                        if abs(h_LM) < 1e-15
                            continue
                        end

                        mp = m - M
                        for lp in 0:lmax_coupling
                            if abs(mp) > lp
                                continue
                            end

                            if config.include_slope_terms
                                G_grad = get_gradient_gaunt(gaunt, l, m, lp, mp, L, M)
                                if abs(G_grad) > 1e-15
                                    Theta_val = get_cache_value(cache, lp, mp, location)
                                    slope_sum += h_LM * G_grad * Theta_val / rb^2
                                end
                            end

                            if config.include_shift_terms
                                if cache.d2_inner === nothing
                                    if !warned_missing_d2
                                        @warn "Missing second-derivative matrix; skipping thermal shift term in heat flux"
                                        warned_missing_d2 = true
                                    end
                                else
                                    G = get_gaunt_tensor(gaunt, l, m, lp, mp, L, M)
                                    if abs(G) > 1e-15
                                        d2Theta_dr2 = get_cache_d2(cache, lp, mp, location)
                                        shift_sum += h_LM * G * d2Theta_dr2
                                    end
                                end
                            end
                        end
                    end
                end
            end

            qn_lm = -k * (dT_dr - ε * slope_sum + ε * shift_sum)
            idx = lm_to_spectral_index(l, m, spectral.config)
            if idx > 0 && idx <= length(coeffs)
                coeffs[idx] = qn_lm
            end
        end
    end

    return shtns_spectral_to_physical(coeffs, spectral.config, nlat, nlon)
end
