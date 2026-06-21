# ================================================================================
# Stefan Condition for ICB Topography Evolution
# ================================================================================
#
# This file implements the Stefan condition at the inner-core boundary (ICB)
# for modeling phase change and topography evolution:
#
#   k_ic ∂_n T_ic - k ∂_n T = ρ L (V_b - uₙ)      (Eq. 21 from PDF)
#
# where:
# - k_ic, k = thermal conductivities (inner core, outer core)
# - T_ic, T = temperatures (inner core, outer core)
# - ρ = density
# - L = latent heat of fusion
# - V_b = boundary velocity (= ε ∂_t h_i for moving boundary)
# - uₙ = normal fluid velocity
#
# The implementable update for topography evolution is (Eq. 22):
#   ε ∂_t h_i = uₙ + (k_ic ∂_n T_ic - k ∂_n T) / (ρ L)
#
# In spectral form (Eq. 31):
#   ε ∂_t h^i_{lm} = u_{n,lm} + (1/ρL) F_{lm}
#
# where F_{lm} collects the heat flux contributions.
#
# ================================================================================

# ================================================================================
# Stefan State Structure
# ================================================================================

mutable struct StefanState{T <: AbstractFloat}
    # Physical parameters
    k_ic::T                                  # Inner core conductivity
    k_oc::T                                  # Outer core conductivity
    rho::T                                   # ICB density
    L::T                                     # Latent heat
    Stefan::T                                # Stefan number
    lambda::T                                # k_ic / k_oc

    # State variables (spectral)
    topography::TopographyField{T}           # h_i(θ, φ)
    topography_rate::TopographyField{T}      # ∂_t h_i
    heat_flux_ic::Vector{Complex{T}}         # Inner core flux
    heat_flux_oc::Vector{Complex{T}}         # Outer core flux
    normal_velocity::Vector{Complex{T}}      # uₙ at ICB

    # Configuration
    is_initialized::Bool
    use_clapeyron::Bool
    clapeyron_slope::T
end

"""
    StefanState{T}(; lmax=32, ri=0.35, kwargs...) where T

Create an uninitialized Stefan state with default parameters.

# Arguments
- `lmax::Int=32`: Maximum spherical harmonic degree
- `ri::Float64`: Inner core radius
- `k_ic::Float64=1.0`: Inner core thermal conductivity (nondimensional)
- `k_oc::Float64=1.0`: Outer core thermal conductivity (nondimensional)
- `rho::Float64=1.0`: Density (nondimensional)
- `L::Float64=1.0`: Latent heat (nondimensional)
- `use_clapeyron::Bool=false`: Include Clapeyron correction
- `clapeyron_slope::Float64=0.0`: Clapeyron slope dT_m/dP
"""
function StefanState{T}(;
        lmax::Int = 32,
        ri::T = T(0.35),
        k_ic::T = one(T),
        k_oc::T = one(T),
        rho::T = one(T),
        L::T = one(T),
        use_clapeyron::Bool = false,
        clapeyron_slope::T = zero(T)
) where {T}
    # Create empty topography fields
    topo = TopographyField{T}(lmax, lmax, ri, INNER_BOUNDARY)
    topo_rate = TopographyField{T}(lmax, lmax, ri, INNER_BOUNDARY)

    nlm = topo.nlm
    Stefan = one(T)  # Will be computed from temperature scale
    lambda = k_ic / k_oc

    return StefanState{T}(
        k_ic, k_oc, rho, L, Stefan, lambda,
        topo, topo_rate,
        zeros(Complex{T}, nlm),
        zeros(Complex{T}, nlm),
        zeros(Complex{T}, nlm),
        false,
        use_clapeyron,
        clapeyron_slope
    )
end

@doc """
    StefanState(; kwargs...)
    StefanState{T}

Stefan condition state for tracking ICB (inner core boundary) evolution.

The Stefan condition relates the rate of phase boundary motion to the heat flux
discontinuity across the interface:
```math
\\rho L \\frac{\\partial h_i}{\\partial t} = k_{oc} \\frac{\\partial T}{\\partial n}\\bigg|_{oc} - k_{ic} \\frac{\\partial T}{\\partial n}\\bigg|_{ic}
```

This structure tracks the ICB topography evolution including latent heat release
and optional Clapeyron slope corrections.

# Keyword Arguments
- `lmax::Int=32`: Maximum spherical harmonic degree
- `ri::Float64=0.35`: Inner core radius (nondimensional)
- `k_ic::Float64=1.0`: Inner core thermal conductivity
- `k_oc::Float64=1.0`: Outer core thermal conductivity
- `rho::Float64=1.0`: Density at ICB
- `L::Float64=1.0`: Latent heat of fusion
- `use_clapeyron::Bool=false`: Include Clapeyron slope correction
- `clapeyron_slope::Float64=0.0`: Clapeyron slope dT_m/dP (K/Pa)

# Example
```julia
state = StefanState(lmax=64, ri=0.35, k_ic=2.0, k_oc=1.0)
```

See also: [`initialize_stefan_state!`](@ref), [`update_icb_topography!`](@ref)
""" StefanState

StefanState(; kwargs...) = StefanState{Float64}(; kwargs...)

# ================================================================================
# Initialization
# ================================================================================

"""
    initialize_stefan_state!(state::StefanState{T}, temperature_ic, temperature_oc,
                             velocity_field; gaunt_cache=nothing) where T

Initialize the Stefan state from current field values.

# Arguments
- `state`: StefanState to initialize
- `temperature_ic`: Inner core temperature field (or spectral coefficients)
- `temperature_oc`: Outer core temperature field
- `velocity_field`: Velocity field for computing uₙ
- `gaunt_cache`: Optional pre-computed Gaunt tensors
"""
function initialize_stefan_state!(state::StefanState{T}, temperature_ic, temperature_oc,
        velocity_field; gaunt_cache = nothing) where {T}
    ri = state.topography.radius
    lmax = state.topography.lmax

    # Compute normal derivatives at ICB (used in Stefan flux)
    state.heat_flux_ic = compute_boundary_heat_flux_spectral(temperature_ic, ri, :inner)
    state.heat_flux_oc = compute_boundary_heat_flux_spectral(temperature_oc, ri, :outer)

    # Compute normal velocity at ICB
    state.normal_velocity = compute_normal_velocity_spectral(velocity_field, ri, INNER_BOUNDARY)

    # Compute Stefan number if temperature scale is available
    # St = c_p ΔT / L  (typically from simulation parameters)

    state.is_initialized = true

    if get_rank() == 0
        @info "Stefan state initialized at r_i=$ri, lmax=$lmax"
    end

    return state
end

# ================================================================================
# Stefan Flux Computation
# ================================================================================

"""
    compute_stefan_flux(state::StefanState{T}) -> Vector{Complex{T}}

Compute the Stefan flux contribution F_{lm} for topography evolution.

F_{lm} = k_ic ∂_n Θ^{ic}_{lm} - k ∂_n Θ_{lm}

Returns spectral coefficients of the net heat flux imbalance.
"""
function compute_stefan_flux(state::StefanState{T}) where {T}
    nlm = state.topography.nlm

    # Net flux: k_ic ∂_n T_ic - k ∂_n T
    flux = zeros(Complex{T}, nlm)
    for i in 1:nlm
        flux[i] = state.k_ic * state.heat_flux_ic[i] - state.k_oc * state.heat_flux_oc[i]
    end

    return flux
end

"""
    compute_stefan_flux_with_topography(state::StefanState{T}, temperature_ic,
                                        temperature_oc, topo_data::TopographyData,
                                        gaunt::GauntTensorCache{T},
                                        config::TopographyCouplingConfig) where T

Compute Stefan flux including topography corrections to the normal derivative.

The linearized normal derivative at r = r_i is:
∂_n ≈ ∂_r - ε ∇_H h_i · ∇_H + εh_i ∂_rr

This adds coupling between modes through Gaunt tensors.
"""
function compute_stefan_flux_with_topography(state::StefanState{T}, temperature_ic,
        temperature_oc, topo_data::TopographyData,
        gaunt::GauntTensorCache{T},
        config::TopographyCouplingConfig) where {T}
    ri = state.topography.radius
    ε = config.epsilon
    lmax = state.topography.lmax
    mmax = state.topography.mmax
    nlm = state.topography.nlm

    # Base flux (flat-sphere)
    flux = compute_stefan_flux(state)

    if !config.include_slope_terms && !config.include_shift_terms
        return flux
    end

    topo_field = topo_data.icb
    if topo_field === nothing
        return flux
    end

    # Optional derivative caches for accurate shift terms
    cache_ic = nothing
    cache_oc = nothing
    if hasfield(typeof(temperature_ic), :spectral) &&
       hasfield(typeof(temperature_ic), :∂r) &&
       hasfield(typeof(temperature_ic), :domain)
        cache_ic = compute_boundary_derivative_cache(temperature_ic.spectral,
            temperature_ic.∂r,
            hasfield(typeof(temperature_ic), :∂²r) ?
            temperature_ic.∂²r : nothing,
            temperature_ic.domain)
    elseif _is_spectral_field_like(temperature_ic)
        cache_ic = nothing
    end

    if hasfield(typeof(temperature_oc), :spectral) &&
       hasfield(typeof(temperature_oc), :∂r) &&
       hasfield(typeof(temperature_oc), :domain)
        cache_oc = compute_boundary_derivative_cache(temperature_oc.spectral,
            temperature_oc.∂r,
            hasfield(typeof(temperature_oc), :∂²r) ?
            temperature_oc.∂²r : nothing,
            temperature_oc.domain)
    elseif _is_spectral_field_like(temperature_oc)
        cache_oc = nothing
    end

    warned_missing_d2 = false

    # Add topography corrections to normal derivative.
    # Iterate the target order m ≥ 0 ONLY (matching the thermal/velocity/magnetic
    # coupling siblings): the flux storage keeps m ≥ 0 modes, and the inner sum
    # over (lp, mp = m − M) already gathers every coupled source. A −m target pass
    # would write the SAME abs(m) slot again — a spurious double application.
    for l in 0:lmax
        for m in 0:min(l, mmax)
            lm_idx = lm_to_index(l, m, lmax, mmax)
            if lm_idx > nlm
                continue
            end

            correction = zero(Complex{T})

            # Sum over topography modes
            for L in 0:topo_field.lmax
                for M in -L:L
                    h_LM = get_coefficient(topo_field, L, M)
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

                        lp_idx = lm_to_index(lp, abs(mp), lmax, mmax)
                        if lp_idx > nlm
                            continue
                        end

                        # Slope term: -∇_H h · ∇_H T
                        if config.include_slope_terms && abs(G_grad) > 1e-15
                            # For outer core
                            if cache_oc === nothing
                                Theta_oc = get_spectral_coefficient(temperature_oc, lp, mp, INNER_BOUNDARY)
                            else
                                Theta_oc = get_cache_value(cache_oc, lp, mp, INNER_BOUNDARY)
                            end
                            correction += h_LM * state.k_oc * G_grad / ri^2 * Theta_oc

                            # For inner core
                            if cache_ic === nothing
                                Theta_ic = get_spectral_coefficient(temperature_ic, lp, mp, INNER_BOUNDARY)
                            else
                                Theta_ic = get_cache_value(cache_ic, lp, mp, INNER_BOUNDARY)
                            end
                            correction -= h_LM * state.k_ic * G_grad / ri^2 * Theta_ic
                        end

                        # Shift term: h · ∂_rr T
                        if config.include_shift_terms && abs(G) > 1e-15
                            if cache_oc === nothing || cache_oc.d2_inner === nothing ||
                               cache_ic === nothing || cache_ic.d2_inner === nothing
                                if !warned_missing_d2
                                    @warn "Missing second-derivative cache; skipping Stefan shift term"
                                    warned_missing_d2 = true
                                end
                            else
                                d2T_oc = get_cache_d2(cache_oc, lp, mp, INNER_BOUNDARY)
                                d2T_ic = get_cache_d2(cache_ic, lp, mp, INNER_BOUNDARY)

                                correction -= h_LM * state.k_oc * G * d2T_oc
                                correction += h_LM * state.k_ic * G * d2T_ic
                            end
                        end
                    end
                end
            end

            flux[lm_idx] += ε * correction
        end
    end

    return flux
end

# ================================================================================
# Topography Evolution
# ================================================================================

"""
    update_icb_topography!(state::StefanState{T}, dt::T, velocity_field,
                           temperature_ic, temperature_oc;
                           topo_data::Union{TopographyData, Nothing}=nothing,
                           gaunt::Union{GauntTensorCache{T}, Nothing}=nothing,
                           config::Union{TopographyCouplingConfig, Nothing}=nothing) where T

Update the ICB topography based on the Stefan condition.

From Eq. 22/40:
ε ∂_t h_i = uₙ + (k_ic ∂_n T_ic - k ∂_n T) / (ρ L)

or equivalently with Stefan number:
ε ∂_t h_i = uₙ + (1/St) (λ ∂_n Θ_ic - ∂_n Θ)

# Arguments
- `state`: StefanState to update
- `dt`: Time step
- `velocity_field`: Current velocity field
- `temperature_ic`: Inner core temperature
- `temperature_oc`: Outer core temperature
- `topo_data`: Optional TopographyData for coupled evolution
- `gaunt`: Optional Gaunt cache for topography coupling
- `config`: Optional coupling configuration
"""
function update_icb_topography!(state::StefanState{T}, dt::T, velocity_field,
        temperature_ic, temperature_oc;
        topo_data::Union{TopographyData, Nothing} = nothing,
        gaunt::Union{GauntTensorCache{T}, Nothing} = nothing,
        config::Union{TopographyCouplingConfig, Nothing} = nothing) where {T}
    ri = state.topography.radius
    rho_L = state.rho * state.L
    nlm = state.topography.nlm

    # Update normal velocity
    state.normal_velocity = compute_normal_velocity_spectral(velocity_field, ri, INNER_BOUNDARY)

    # Update heat fluxes
    state.heat_flux_ic = compute_boundary_heat_flux_spectral(temperature_ic, ri, :inner)
    state.heat_flux_oc = compute_boundary_heat_flux_spectral(temperature_oc, ri, :outer)

    # Compute Stefan flux
    if topo_data !== nothing && gaunt !== nothing && config !== nothing
        stefan_flux = compute_stefan_flux_with_topography(
            state, temperature_ic, temperature_oc, topo_data, gaunt, config
        )
    else
        stefan_flux = compute_stefan_flux(state)
    end

    # Get epsilon from config or use default
    ε = config !== nothing ? config.epsilon : T(0.01)

    # Compute topography rate: ε ∂_t h = uₙ + F/(ρL)
    for i in 1:nlm
        dh_dt = (state.normal_velocity[i] + stefan_flux[i] / rho_L) / ε
        state.topography_rate.coeffs_real[i] = real(dh_dt)
        state.topography_rate.coeffs_imag[i] = imag(dh_dt)
    end

    # Update topography (forward Euler - can be replaced with better scheme)
    # Warn if the topography growth rate may exceed stability bounds
    max_rate = zero(T)
    for i in 1:nlm
        rate_mag = abs(state.topography_rate.coeffs_real[i]) +
                   abs(state.topography_rate.coeffs_imag[i])
        max_rate = max(max_rate, rate_mag)
    end
    if max_rate * dt > T(0.1)
        @warn "Stefan condition: large topography update detected (max |dh/dt| * dt = $(max_rate * dt)). " *
              "Consider reducing the timestep for stability of the forward Euler topography update."
    end
    for i in 1:nlm
        state.topography.coeffs_real[i] += dt * state.topography_rate.coeffs_real[i]
        state.topography.coeffs_imag[i] += dt * state.topography_rate.coeffs_imag[i]
    end

    # Update statistics
    update_topography_statistics!(state.topography)
    update_topography_statistics!(state.topography_rate)

    return state
end

"""
    update_icb_topography_semiimplicit!(state::StefanState{T}, dt::T, velocity_field,
                                        temperature_ic, temperature_oc;
                                        theta::T=0.5) where T

Update ICB topography using a semi-implicit (θ-scheme) time integration.

h^{n+1} = h^n + dt [θ RHS^{n+1} + (1-θ) RHS^n]

where RHS = (1/ε) [uₙ + F/(ρL)]

# Arguments
- `theta::T=0.5`: Implicitness parameter (0.5 = Crank-Nicolson, 1.0 = backward Euler)
"""
function update_icb_topography_semiimplicit!(state::StefanState{T}, dt::T, velocity_field,
        temperature_ic, temperature_oc;
        theta::T = T(0.5),
        topo_data::Union{TopographyData, Nothing} = nothing,
        gaunt::Union{GauntTensorCache{T}, Nothing} = nothing,
        config::Union{TopographyCouplingConfig, Nothing} = nothing) where {T}
    rho_L = state.rho * state.L
    nlm = state.topography.nlm
    ε = config !== nothing ? config.epsilon : T(0.01)

    # Store old RHS
    rhs_old = zeros(Complex{T}, nlm)
    for i in 1:nlm
        rhs_old[i] = complex(state.topography_rate.coeffs_real[i],
            state.topography_rate.coeffs_imag[i])
    end

    # Compute new fluxes and velocities
    state.normal_velocity = compute_normal_velocity_spectral(velocity_field,
        state.topography.radius,
        INNER_BOUNDARY)
    state.heat_flux_ic = compute_boundary_heat_flux_spectral(temperature_ic,
        state.topography.radius, :inner)
    state.heat_flux_oc = compute_boundary_heat_flux_spectral(temperature_oc,
        state.topography.radius, :outer)

    # Compute Stefan flux at new time level
    if topo_data !== nothing && gaunt !== nothing && config !== nothing
        stefan_flux = compute_stefan_flux_with_topography(
            state, temperature_ic, temperature_oc, topo_data, gaunt, config
        )
    else
        stefan_flux = compute_stefan_flux(state)
    end

    # Compute new RHS
    rhs_new = zeros(Complex{T}, nlm)
    for i in 1:nlm
        rhs_new[i] = (state.normal_velocity[i] + stefan_flux[i] / rho_L) / ε
    end

    # Semi-implicit update: h^{n+1} = h^n + dt [θ RHS^{n+1} + (1-θ) RHS^n]
    for i in 1:nlm
        dh = dt * (theta * rhs_new[i] + (one(T) - theta) * rhs_old[i])
        state.topography.coeffs_real[i] += real(dh)
        state.topography.coeffs_imag[i] += imag(dh)

        # Store current rate
        state.topography_rate.coeffs_real[i] = real(rhs_new[i])
        state.topography_rate.coeffs_imag[i] = imag(rhs_new[i])
    end

    update_topography_statistics!(state.topography)

    return state
end

# ================================================================================
# Utility Functions
# ================================================================================

"""
    compute_boundary_heat_flux_spectral(temperature_field, r::T, side::Symbol) where T

Compute spectral coefficients of heat flux at a boundary.

# Arguments
- `temperature_field`: Temperature field
- `r`: Boundary radius
- `side`: :inner or :outer

Returns vector of spectral coefficients for ∂_r T at the boundary.
"""
function compute_boundary_heat_flux_spectral(temperature_field, r::T, side::Symbol) where {T}
    # Get spectral field
    if hasfield(typeof(temperature_field), :spectral)
        spectral = temperature_field.spectral
    elseif _is_spectral_field_like(temperature_field)
        spectral = temperature_field
    else
        return Complex{T}[]
    end

    nlm = spectral.nlm
    flux = zeros(Complex{T}, nlm)
    location = side == :inner ? INNER_BOUNDARY : OUTER_BOUNDARY

    cache = nothing
    if hasfield(typeof(temperature_field), :∂r) &&
       hasfield(typeof(temperature_field), :domain)
        cache = compute_boundary_derivative_cache(spectral,
            temperature_field.∂r,
            nothing,
            temperature_field.domain)
    end

    # Store ∂_r T (k absorbed into coefficients elsewhere)
    for lm_idx in 1:nlm
        l, m = index_to_lm(lm_idx, spectral.config.lmax)

        # Get radial derivative at boundary
        if cache === nothing
            ∂r = hasfield(typeof(temperature_field), :∂r) ? temperature_field.∂r : nothing
            domain = hasfield(typeof(temperature_field), :domain) ?
                     temperature_field.domain : nothing
            dT_dr = get_spectral_radial_derivative(spectral, l, m, r, location;
                ∂r = ∂r, domain = domain)
        else
            dT_dr = get_cache_d1(cache, l, m, location)
        end
        flux[lm_idx] = dT_dr
    end

    return flux
end

"""
    compute_normal_velocity_spectral(velocity_field, r::T,
                                     location::BoundaryLocation=OUTER_BOUNDARY) where T

Compute spectral coefficients of normal (radial) velocity at a boundary.

uₙ = uᵣ = l(l+1)/r² P at the boundary

Returns vector of spectral coefficients for uᵣ.
"""
function compute_normal_velocity_spectral(velocity_field, r::T,
        location::BoundaryLocation = OUTER_BOUNDARY) where {T}
    # Get poloidal component
    if hasfield(typeof(velocity_field), :poloidal)
        poloidal = velocity_field.poloidal
    elseif hasfield(typeof(velocity_field), :P)
        poloidal = velocity_field.P
    else
        return Complex{T}[]
    end

    nlm = poloidal.nlm
    un = zeros(Complex{T}, nlm)

    # uᵣ = l(l+1)/r² P
    for lm_idx in 1:nlm
        l, m = index_to_lm(lm_idx, poloidal.config.lmax)

        if l == 0
            continue  # l=0 has no radial velocity
        end

        P_val = get_spectral_boundary_value(poloidal, l, m, location)
        ll_factor = T(l * (l + 1))
        un[lm_idx] = ll_factor / r^2 * P_val
    end

    return un
end

"""
    get_spectral_coefficient(field, l::Int, m::Int)

Get spectral coefficient (l, m) from a field.
"""
function get_spectral_coefficient(field, l::Int, m::Int,
        location::BoundaryLocation = OUTER_BOUNDARY)
    if hasfield(typeof(field), :spectral)
        spectral = field.spectral
    elseif _is_spectral_field_like(field)
        spectral = field
    else
        return zero(Complex{Float64})
    end

    return get_spectral_boundary_value(spectral, l, m, location)
end

# ================================================================================
# Clapeyron Correction (Optional)
# ================================================================================

"""
    apply_clapeyron_correction!(state::StefanState{T}, pressure_field) where T

Apply Clapeyron slope correction to the melting temperature at ICB.

The melting temperature varies with pressure:
T_m(P) = T_m0 + (dT_m/dP) ΔP

This modifies the effective temperature boundary condition at the ICB.
"""
function apply_clapeyron_correction!(state::StefanState{T}, pressure_field) where {T}
    if !state.use_clapeyron || state.clapeyron_slope ≈ 0
        return state
    end

    ri = state.topography.radius
    gamma = state.clapeyron_slope

    # Get pressure perturbation at ICB
    # ΔP ~ ρ g h for topography-induced pressure

    # This would modify the effective melting temperature
    # T_m_eff = T_m0 + γ ρ g h

    # For now, this is a placeholder for future implementation

    return state
end

# ================================================================================
# Diagnostics
# ================================================================================

"""
    get_stefan_diagnostics(state::StefanState{T}) where T

Get diagnostic information about the Stefan condition state.
"""
function get_stefan_diagnostics(state::StefanState{T}) where {T}
    return Dict{String, Any}(
        "topography_rms" => state.topography.rms_amplitude,
        "topography_max" => state.topography.max_amplitude,
        "growth_rate_rms" => state.topography_rate.rms_amplitude,
        "heat_flux_rms" => sqrt(sum(abs2, compute_stefan_flux(state))),
        "normal_velocity_rms" => sqrt(sum(abs2, state.normal_velocity)),
        "stefan_number" => state.Stefan,
        "lambda" => state.lambda,
        "is_initialized" => state.is_initialized
    )
end

"""
    print_stefan_summary(state::StefanState)

Print a summary of the Stefan condition state.
"""
function print_stefan_summary(state::StefanState)
    diag = get_stefan_diagnostics(state)

    println("╔═══════════════════════════════════════════════════════════════╗")
    println("║              STEFAN CONDITION STATE SUMMARY                   ║")
    println("╠═══════════════════════════════════════════════════════════════╣")
    println("║ Initialized: $(diag["is_initialized"])                                        ║")
    println("║ Stefan number St = $(round(diag["stefan_number"], digits=4))                          ║")
    println("║ Conductivity ratio λ = $(round(diag["lambda"], digits=4))                       ║")
    println("╠───────────────────────────────────────────────────────────────╣")
    println("║ ICB Topography:                                               ║")
    println("║   RMS amplitude: $(round(diag["topography_rms"], sigdigits=4))                        ║")
    println("║   Max amplitude: $(round(diag["topography_max"], sigdigits=4))                        ║")
    println("║   Growth rate RMS: $(round(diag["growth_rate_rms"], sigdigits=4))                     ║")
    println("╠───────────────────────────────────────────────────────────────╣")
    println("║ Fluxes:                                                       ║")
    println("║   Heat flux RMS: $(round(diag["heat_flux_rms"], sigdigits=4))                         ║")
    println("║   Normal velocity RMS: $(round(diag["normal_velocity_rms"], sigdigits=4))             ║")
    println("╚═══════════════════════════════════════════════════════════════╝")
end
