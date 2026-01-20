# ================================================================================
# Magnetic Field Module with SHTns
# ================================================================================
#
# This module implements the magnetic field representation and induction equation
# for geodynamo simulations using spherical harmonic transforms (SHTnsKit).
#
# REFERENCE: Sreenivasan & Kar (2018), Phys. Rev. Fluids 3, 093801
#            Equation (3): Induction equation
#
# ================================================================================
# GOVERNING EQUATION
# ================================================================================
#
# The non-dimensional induction equation in magnetic diffusion time scaling:
#
#   ∂B/∂t = ∇×(u×B) + ∇²B                    (Equation 3 in paper)
#
# where:
#   B            : Magnetic field
#   u            : Velocity field
#   ∇×(u×B)      : Induction term (field generation/stretching by flow)
#   ∇²B          : Magnetic diffusion (Ohmic decay)
#
# PHYSICAL INTERPRETATION:
# ========================
# - ∇×(u×B): The "dynamo" term - fluid motion stretches and twists field lines,
#            converting kinetic energy to magnetic energy
# - ∇²B: Ohmic dissipation - resistive decay of magnetic field
#
# In magnetic diffusion time scaling (τ = L²/η), the diffusion coefficient is 1.0.
# This is why the time-stepper uses diffusivity = 1.0 for magnetic fields.
#
# ================================================================================
# TOROIDAL-POLOIDAL DECOMPOSITION
# ================================================================================
#
# For solenoidal fields (∇·B = 0), the magnetic field is decomposed as:
#
#   B = ∇×(T r̂) + ∇×∇×(P r̂)
#
# where:
#   T(r,θ,φ) = Toroidal scalar potential (purely tangential field)
#   P(r,θ,φ) = Poloidal scalar potential (has radial component)
#
# Physical interpretation:
#   - Toroidal: azimuthal magnetic field (like field from ring currents)
#   - Poloidal: meridional magnetic field (like Earth's dipole field)
#
# This decomposition AUTOMATICALLY satisfies ∇·B = 0 (Eq. 4 in paper).
#
# ================================================================================
# INDUCTION EQUATION IMPLEMENTATION
# ================================================================================
#
# The induction term ∇×(u×B) is computed in three steps:
#
# Step 1: Compute u×B in PHYSICAL space (point-wise cross product)
#         This is done in compute_velocity_cross_magnetic!()
#
# Step 2: Transform u×B to SPECTRAL space (SHTns vector analysis)
#         This gives (u×B)_toroidal and (u×B)_poloidal coefficients
#
# Step 3: Compute ∇×(u×B) in SPECTRAL space using curl operator:
#         [∇×(u×B)]_tor = [l(l+1)/r² - d²/dr² - 2/r d/dr] (u×B)_pol
#         [∇×(u×B)]_pol = -l(l+1)/r² (u×B)_tor
#
# The diffusion term ∇²B is handled IMPLICITLY by the time-stepper.
#
# Transform Flow:
# ===============
#  Spectral B(tor/pol) → [shtns_vector_synthesis!] → Physical B
#                             ↓
#                   Compute j = ∇×B in spectral
#                             ↓
#               Compute u×B in physical space
#                             ↓
# Physical (u×B) → [shtns_vector_analysis!] → Spectral (u×B)
#                             ↓
#               Compute ∇×(u×B) in spectral
#                             ↓
#                   Add to nonlinear terms
#
# ================================================================================
# BOUNDARY CONDITIONS
# ================================================================================
#
# Three main magnetic BC types:
#
# 1. INSULATING (σ = 0 outside):
#    - Physical: No current can cross boundary (J_n = 0)
#    - Mathematical: (∇×B)_r = 0 at boundary
#    - Implementation: Match potential field (Dirichlet on both T and P)
#
# 2. PERFECT CONDUCTOR (σ → ∞ outside):
#    - Physical: Tangential B excluded from conductor
#    - Mathematical: B_tangential = 0 at boundary
#    - Implementation: T = 0 (Dirichlet), P from ∇·B = 0
#
# 3. POTENTIAL FIELD:
#    - Physical: Field matches external potential field
#    - Mathematical: B = -∇V where ∇²V = 0
#    - Implementation: Dirichlet matching external field
#
# NOTE: Unlike velocity, magnetic BCs use Dirichlet for all types.
# The distinction is in WHAT values are specified, not the BC type.
#
# ================================================================================
# CURRENT DENSITY COMPUTATION
# ================================================================================
#
# Current density j = ∇×B in spectral space:
#
# From toroidal-poloidal decomposition:
#   jᵀoidal = [l(l+1)/r² - d²/dr² - 2/r d/dr] P^{lm}
#   jᴾoidal = -[l(l+1)/r²] T^{lm}
#
# This is computed mode-by-mode in spectral space for efficiency.
#
# ================================================================================
# MPI PARALLELIZATION
# ================================================================================
#
# Data distribution follows the same pattern as velocity:
#   - Spectral data: distributed over (l,m) modes via lm_range
#   - Radial data: distributed over radial points via r_range
#
# MPI collectives (Allreduce) in diagnostic functions are called AFTER loops,
# not inside them, so no special owns_mode pattern is needed here.
#
# ================================================================================

import .bcs
import .bcs: BoundaryType, DIRICHLET, NEUMANN

mutable struct SHTnsMagneticFields{T}
    # Physical space magnetic field
    magnetic::SHTnsVectorField{T}
    current::SHTnsVectorField{T}

    # Spectral representation
    𝒯::SHTnsSpecField{T}
    𝒫::SHTnsSpecField{T}

    # Inner core fields
    ic_toroidal::SHTnsSpecField{T}
    ic_poloidal::SHTnsSpecField{T}

    # Nonlinear terms (induction)
    nlᵀ::SHTnsSpecField{T}
    nlᴾ::SHTnsSpecField{T}
    prev_nlᵀ::SHTnsSpecField{T}
    prev_nlᴾ::SHTnsSpecField{T}

    # Work arrays
    work_tor::SHTnsSpecField{T}
    work_pol::SHTnsSpecField{T}
    work_physical::SHTnsVectorField{T}
    induction_physical::SHTnsVectorField{T}  # Added missing field for u×B

    # Pre-computed coefficients
    ℓ_factors::Vector{Float64}  # l(l+1) values

    # Radial derivative matrices (cached for performance)
    ∂r::BandedMatrix{T}          # First derivative d/dr
    ∂²r::BandedMatrix{T}         # Second derivative d²/dr²

    # Transform manager removed; SHTnsKit transforms are used directly

    # Imposed field (if any)
    imposed_field::Union{SHTnsVectorField{T}, Nothing}
    config::SHTnsKitConfig
    outer_domain::RadialDomain
    boundary_condition_set::Union{bcs.BoundaryConditionSet{T}, Nothing}
    boundary_interpolation_cache::Dict{String, Any}
    boundary_time_index::Ref{Int}
end


function create_shtns_magnetic_fields(::Type{T}, config::SHTnsKitConfig, 
                                      𝒟ᵒᶜ::RadialDomain, 
                                      𝒟ⁱᶜ::RadialDomain, 
                                      pencils=nothing, pencil_spec=nothing) where T

    # Use enhanced pencil topology from config if not provided
    if pencils === nothing
        pencils = config.pencils
    end
    pencil_θ, pencil_φ, pencil_r = pencils.θ, pencils.φ, pencils.r
    
    # Use spectral pencil from topology if not provided
    if pencil_spec === nothing
        pencil_spec = pencils.spec
    end
    
    # Physical space fields
    magnetic = create_shtns_vector_field(T, config, 𝒟ᵒᶜ, pencils)
    current  = create_shtns_vector_field(T, config, 𝒟ᵒᶜ, pencils)
    
    # Spectral fields
    𝒯 = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    𝒫 = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    
    # Inner core fields (different domain)
    ic_toroidal = create_shtns_spectral_field(T, config, 𝒟ⁱᶜ, pencil_spec)
    ic_poloidal = create_shtns_spectral_field(T, config, 𝒟ⁱᶜ, pencil_spec)
    
    # Nonlinear terms
    nlᵀ = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    nlᴾ = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    prev_nlᵀ = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    prev_nlᴾ = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    
    # Work arrays
    work_tor = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    work_pol = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    work_physical = create_shtns_vector_field(T, config, 𝒟ᵒᶜ, pencils)
    induction_physical = create_shtns_vector_field(T, config, 𝒟ᵒᶜ, pencils)
    
    # Pre-compute l(l+1) factors
    ℓ_factors = Float64[l * (l + 1) for l in config.l_values]

    # Create radial derivative matrices (cached for performance)
    ∂r  = create_derivative_matrix(1, 𝒟ᵒᶜ)
    ∂²r = create_derivative_matrix(2, 𝒟ᵒᶜ)

    # Create transpose plans for efficient data movement
    transpose_plans = create_transpose_plans(pencils)
    
    imposed_field = nothing
    boundary_condition_set = nothing
    boundary_cache = Dict{String, Any}()
    boundary_time_index = Ref{Int}(1)
    
    return SHTnsMagneticFields{T}(magnetic, current,
                                𝒯, 𝒫,
                                ic_toroidal, ic_poloidal,
                                nlᵀ, nlᴾ, prev_nlᵀ, prev_nlᴾ,
                                work_tor, work_pol, work_physical,
                                induction_physical,
                                ℓ_factors,
                                ∂r, ∂²r,
                                imposed_field,
                                config,
                                𝒟ᵒᶜ,
                                boundary_condition_set, boundary_cache, boundary_time_index)
end


# ========================================================
# Main nonlinear computation using enhanced transforms
# ========================================================
function compute_magnetic_nonlinear!(ℬ::SHTnsMagneticFields{T},
                                    𝒰, 𝒟ᵒᶜ::RadialDomain, 𝒟ⁱᶜ::RadialDomain,
                                    rotation_rate::Float64=0.0;
                                    geometry::Symbol = get_parameters().geometry) where T
    # Zero work arrays
    zero_magnetic_work_arrays!(ℬ)
    
    # Step 1: Convert spectral B to physical space using enhanced transforms
    shtnskit_vector_synthesis!(ℬ.𝒯, ℬ.𝒫,
                               ℬ.magnetic; domain=𝒟ᵒᶜ)

    # Step 2: Compute current density j = ∇ × B in spectral space
    compute_current_density_spectral!(ℬ, 𝒟ᵒᶜ)

    # Step 3: Transform current to physical space
    shtnskit_vector_synthesis!(ℬ.work_tor, ℬ.work_pol,
                               ℬ.current; domain=𝒟ᵒᶜ)
    
    # Step 4: Compute induction equation: ∂B/∂t = ∇ × (u × B) + η∇²B
    if 𝒰 !== nothing
        compute_induction_term!(ℬ, 𝒰; geometry)
    end
    
    # Step 5: Inner core rotation effects
    if rotation_rate != 0.0
        add_inner_core_rotation!(ℬ, rotation_rate)
    end
    
    # Note: The nonlinear terms are now in ℬ.nlᵀ/poloidal
end

"""
    enforce_magnetic_boundary_values!(fields)

Anchor magnetic toroidal/poloidal spectral data to cached Dirichlet boundary
values on the inner and outer radial surfaces.
"""
function enforce_magnetic_boundary_values!(fields::SHTnsMagneticFields{T}) where T
    domain = fields.outer_domain

    tor_real = parent(fields.𝒯.data_real)
    tor_imag = parent(fields.𝒯.data_imag)
    pol_real = parent(fields.𝒫.data_real)
    pol_imag = parent(fields.𝒫.data_imag)

    tor_bc = fields.𝒯.boundary_values
    pol_bc = fields.𝒫.boundary_values

    lm_range = get_local_range(fields.𝒯.pencil, 1)
    r_range = get_local_range(fields.𝒯.pencil, 3)

    has_inner = 1 in r_range && domain.r[1, 4] > 0
    has_outer = domain.N in r_range

    inner_idx = has_inner ? (1 - first(r_range) + 1) : 0
    outer_idx = has_outer ? (domain.N - first(r_range) + 1) : 0

    dirichlet_code = Int(bcs.DIRICHLET)

    for lm_idx in lm_range
        if lm_idx <= fields.𝒯.nlm
            local_lm = lm_idx - first(lm_range) + 1

            if has_inner && 1 <= inner_idx <= size(tor_real, 3)
                if fields.𝒯.bc_type_inner[lm_idx] == dirichlet_code
                    tor_real[local_lm, 1, inner_idx] = tor_bc[1, lm_idx]
                    tor_imag[local_lm, 1, inner_idx] = zero(T)
                end
                if fields.𝒫.bc_type_inner[lm_idx] == dirichlet_code
                    pol_real[local_lm, 1, inner_idx] = pol_bc[1, lm_idx]
                    pol_imag[local_lm, 1, inner_idx] = zero(T)
                end
            end

            if has_outer && 1 <= outer_idx <= size(tor_real, 3)
                if fields.𝒯.bc_type_outer[lm_idx] == dirichlet_code
                    tor_real[local_lm, 1, outer_idx] = tor_bc[2, lm_idx]
                    tor_imag[local_lm, 1, outer_idx] = zero(T)
                end
                if fields.𝒫.bc_type_outer[lm_idx] == dirichlet_code
                    pol_real[local_lm, 1, outer_idx] = pol_bc[2, lm_idx]
                    pol_imag[local_lm, 1, outer_idx] = zero(T)
                end
            end
        end
    end

    return fields
end

"""
    apply_magnetic_boundary_conditions!(fields; time_index=nothing)

Refresh magnetic boundary data from the bcs subsystem and
enforce the corresponding Dirichlet values in spectral space.
"""
function apply_magnetic_boundary_conditions!(fields::SHTnsMagneticFields{T};
                                              time_index::Union{Nothing,Int}=nothing) where T
    boundary_set, _ = bcs.get_magnetic_boundary_data(fields)
    if boundary_set === nothing
        return fields
    end

    if time_index === nothing
        bcs.apply_magnetic_boundary_conditions!(fields)
    else
        bcs.apply_magnetic_boundary_conditions!(fields, time_index)
    end

    enforce_magnetic_boundary_values!(fields)

    if fields.outer_domain.r[1, 4] == 0.0
        enforce_ball_vector_regularity!(fields.𝒯, fields.𝒫)
    end
    return fields
end


# ========================================================
# Current density computation in spectral space
# ========================================================
# ========================================================
# Current density computation in spectral space
# ========================================================
function compute_current_density_spectral!(ℬ::SHTnsMagneticFields{T}, 
                                          𝒟ᵒᶜ::RadialDomain) where T
    # Compute j = ∇ × B using spectral relationships with full radial derivatives
    # For toroidal-poloidal decomposition:
    # B = Bᵀ + Bᴾ where:
    #   Bᵀ = ∇ × (T(r,θ,φ) r̂)
    #   Bᴾ = ∇ × ∇ × (P(r,θ,φ) r̂)
    #
    # Current density j = ∇ × B:
    #   jᵀ = [l(l+1)/r² - d²/dr² - 2/r d/dr] P^{lm}
    #   jᴾ = -[l(l+1)/r²] T^{lm}
    
    # Get local data views
    𝒯ᵇ_real = parent(ℬ.𝒯.data_real)
    𝒯ᵇ_imag = parent(ℬ.𝒯.data_imag)
    𝒫ᵇ_real = parent(ℬ.𝒫.data_real)
    𝒫ᵇ_imag = parent(ℬ.𝒫.data_imag)
    
    jᵀ_real = parent(ℬ.work_tor.data_real)
    jᵀ_imag = parent(ℬ.work_tor.data_imag)
    jᴾ_real = parent(ℬ.work_pol.data_real)
    jᴾ_imag = parent(ℬ.work_pol.data_imag)
    
    # Get local ranges using config-aware pencil topology
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    config = ℬ.𝒯.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range  = range_local(config.pencils.spec, 3)
    total_nlm = config.nlm  # Total number of (l,m) modes

    # Use cached radial derivative matrices for performance
    d1_matrix = ℬ.∂r   # First derivative d/dr
    d²_matrix = ℬ.∂²r  # Second derivative d²/dr²

    # Pre-allocate work arrays for radial profiles
    # Note: Radial data is always local (not MPI distributed)
    nr = 𝒟ᵒᶜ.N
    Pᴾ_profile_real = zeros(T, nr)
    Pᴾ_profile_imag = zeros(T, nr)
    dᴾ_dr_real      = zeros(T, nr)
    dᴾ_dr_imag      = zeros(T, nr)
    d²ᴾ_dr²_real    = zeros(T, nr)
    d²ᴾ_dr²_imag    = zeros(T, nr)

    # Process each (l,m) mode - only owned modes (radial data is local)
    @inbounds for lm_idx in lm_range
        if lm_idx <= length(ℬ.ℓ_factors)
            local_lm = lm_idx - first(lm_range) + 1
            ℓ_factor = ℬ.ℓ_factors[lm_idx]

            # Extract radial profile (all radial data is local)
            fill!(Pᴾ_profile_real, zero(T))
            fill!(Pᴾ_profile_imag, zero(T))
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(𝒫ᵇ_real, 3)
                    Pᴾ_profile_real[r_idx] = 𝒫ᵇ_real[local_lm, 1, local_r]
                    Pᴾ_profile_imag[r_idx] = 𝒫ᵇ_imag[local_lm, 1, local_r]
                end
            end

            # Compute radial derivatives
            apply_∂r!(dᴾ_dr_real, d1_matrix, Pᴾ_profile_real)
            apply_∂r!(dᴾ_dr_imag, d1_matrix, Pᴾ_profile_imag)
            apply_∂r!(d²ᴾ_dr²_real, d²_matrix, Pᴾ_profile_real)
            apply_∂r!(d²ᴾ_dr²_imag, d²_matrix, Pᴾ_profile_imag)

            # Compute current density components
            r_first = first(r_range)
            r_last = min(last(r_range), nr)
            if r_last < r_first
                continue
            end
            @simd for r_idx in r_first:r_last
                local_r = r_idx - r_first + 1
                if local_r <= size(jᵀ_real, 3)
                    r_val = 𝒟ᵒᶜ.r[r_idx, 4]
                    if r_val == 0.0
                        jᵀ_real[local_lm, 1, local_r] = zero(T)
                        jᵀ_imag[local_lm, 1, local_r] = zero(T)
                        jᴾ_real[local_lm, 1, local_r] = zero(T)
                        jᴾ_imag[local_lm, 1, local_r] = zero(T)
                    else
                        r⁻¹ = 𝒟ᵒᶜ.r[r_idx, 3]
                        r⁻² = 𝒟ᵒᶜ.r[r_idx, 2]
                        jᵀ_real[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_profile_real[r_idx]
                                                            - d²ᴾ_dr²_real[r_idx]
                                                            - 2.0 * r⁻¹ * dᴾ_dr_real[r_idx])
                        jᵀ_imag[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_profile_imag[r_idx]
                                                            - d²ᴾ_dr²_imag[r_idx]
                                                            - 2.0 * r⁻¹ * dᴾ_dr_imag[r_idx])
                        jᴾ_real[local_lm, 1, local_r] = -ℓ_factor * r⁻² * 𝒯ᵇ_real[local_lm, 1, local_r]
                        jᴾ_imag[local_lm, 1, local_r] = -ℓ_factor * r⁻² * 𝒯ᵇ_imag[local_lm, 1, local_r]
                    end
                end
            end
        end
    end
end


# ================================================================================
# Induction Term Computation: ∇×(u×B)
# ================================================================================
#
# This implements the EXPLICIT part of the induction equation (Eq. 3):
#   ∂B/∂t = ∇×(u×B) + ∇²B
#
# The induction term ∇×(u×B) represents how fluid motion generates and
# modifies the magnetic field. This is the essence of the dynamo mechanism.
#
# ================================================================================

function compute_induction_term!(ℬ::SHTnsMagneticFields{T}, 𝒰; geometry::Symbol = get_parameters().geometry) where T
    # =========================================================================
    # Compute ∇×(u×B) for the induction equation in three steps
    # =========================================================================

    # Step 1: Compute u×B in PHYSICAL space
    # -------------------------------------
    # Cross product is simple point-wise operation in physical space:
    #   (u×B)_r = uθ Bφ - uφ Bθ
    #   (u×B)_θ = uφ Bᵣ - uᵣ Bφ
    #   (u×B)_φ = uᵣ Bθ - uθ Bᵣ
    compute_velocity_cross_magnetic!(ℬ, 𝒰)

    # Step 2: Transform u×B to SPECTRAL space
    # ----------------------------------------
    # SHTns vector analysis decomposes (u×B) into toroidal and poloidal parts
    if geometry === :ball
        ball_vector_analysis!(ℬ.induction_physical,
                             ℬ.work_tor, ℬ.work_pol)
    else
        shtnskit_vector_analysis!(ℬ.induction_physical,
                                  ℬ.work_tor, ℬ.work_pol)
    end

    # Step 3: Compute ∇×(u×B) in SPECTRAL space
    # ------------------------------------------
    # Uses the spectral curl operator for toroidal-poloidal decomposition
    compute_curl_of_induction!(ℬ)
end


function compute_velocity_cross_magnetic!(ℬ::SHTnsMagneticFields{T}, 𝒰) where T
    # Compute u × B in physical space with enhanced memory access
    
    # Get local data views
    uᵣ = parent(𝒰.velocity.r_component.data)
    uθ = parent(𝒰.velocity.θ_component.data)
    uφ = parent(𝒰.velocity.φ_component.data)
    
    Bᵣ = parent(ℬ.magnetic.r_component.data)
    Bθ = parent(ℬ.magnetic.θ_component.data)
    Bφ = parent(ℬ.magnetic.φ_component.data)
    
    # Output: u × B
    uBᵣ = parent(ℬ.induction_physical.r_component.data)
    uBθ = parent(ℬ.induction_physical.θ_component.data)
    uBφ = parent(ℬ.induction_physical.φ_component.data)
    
    # Get configuration for enhanced access patterns
    config = ℬ.magnetic.r_component.config
    
    # Compute cross product with vectorization
    @inbounds @simd for idx in eachindex(uᵣ)
        if idx <= length(Bᵣ)
            # u × B = (uθ Bφ - uφ Bθ, uφ Bᵣ - uᵣ Bφ, uᵣ Bθ - uθ Bᵣ)
            uBᵣ[idx] = uθ[idx] * Bφ[idx] - uφ[idx] * Bθ[idx]
            uBθ[idx] = uφ[idx] * Bᵣ[idx] - uᵣ[idx] * Bφ[idx]
            uBφ[idx] = uᵣ[idx] * Bθ[idx] - uθ[idx] * Bᵣ[idx]
        end
    end
end


function compute_curl_of_induction!(ℬ::SHTnsMagneticFields{T}) where T
    # Compute ∇ × (u × B) in spectral space
    # This becomes the nonlinear term for the induction equation
    #
    # For toroidal-poloidal decomposition, curl satisfies:
    #   (∇×V)_tor = [l(l+1)/r² - d²/dr² - 2/r d/dr] V_pol
    #   (∇×V)_pol = -l(l+1)/r² V_tor
    #
    # This matches the vorticity and current density computations.

    # Get local data views
    𝒯ᵘᵇ_real = parent(ℬ.work_tor.data_real)
    𝒯ᵘᵇ_imag = parent(ℬ.work_tor.data_imag)
    𝒫ᵘᵇ_real = parent(ℬ.work_pol.data_real)
    𝒫ᵘᵇ_imag = parent(ℬ.work_pol.data_imag)

    NLᵀ_real = parent(ℬ.nlᵀ.data_real)
    NLᵀ_imag = parent(ℬ.nlᵀ.data_imag)
    NLᴾ_real = parent(ℬ.nlᴾ.data_real)
    NLᴾ_imag = parent(ℬ.nlᴾ.data_imag)

    # Get local ranges using config-aware pencil topology
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    config = ℬ.𝒯.config
    domain = ℬ.outer_domain
    lm_range = range_local(config.pencils.spec, 1)
    r_range  = range_local(config.pencils.spec, 3)
    total_nlm = config.nlm  # Total number of (l,m) modes

    nr = domain.N

    # Use cached radial derivative matrices for performance
    d1_matrix = ℬ.∂r   # First derivative d/dr
    d²_matrix = ℬ.∂²r  # Second derivative d²/dr²

    # Pre-allocate work arrays for radial profiles
    # Note: Radial data is always local (not MPI distributed)
    Pᴾ_profile_real = zeros(T, nr)
    Pᴾ_profile_imag = zeros(T, nr)
    dᴾ_dr_real     = zeros(T, nr)
    dᴾ_dr_imag     = zeros(T, nr)
    d²ᴾ_dr²_real   = zeros(T, nr)
    d²ᴾ_dr²_imag   = zeros(T, nr)

    # Process each (l,m) mode
    @inbounds for lm_idx in lm_range
        if lm_idx <= length(ℬ.ℓ_factors)
            local_lm = lm_idx - first(lm_range) + 1
            ℓ_factor = ℬ.ℓ_factors[lm_idx]

            # Extract radial profile (all radial data is local)
            fill!(Pᴾ_profile_real, zero(T))
            fill!(Pᴾ_profile_imag, zero(T))
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(𝒫ᵘᵇ_real, 3)
                    Pᴾ_profile_real[r_idx] = 𝒫ᵘᵇ_real[local_lm, 1, local_r]
                    Pᴾ_profile_imag[r_idx] = 𝒫ᵘᵇ_imag[local_lm, 1, local_r]
                end
            end

            # Compute radial derivatives
            apply_∂r!(dᴾ_dr_real, d1_matrix, Pᴾ_profile_real)
            apply_∂r!(dᴾ_dr_imag, d1_matrix, Pᴾ_profile_imag)
            apply_∂r!(d²ᴾ_dr²_real, d²_matrix, Pᴾ_profile_real)
            apply_∂r!(d²ᴾ_dr²_imag, d²_matrix, Pᴾ_profile_imag)

            # Compute curl components
            r_first = first(r_range)
            r_last = min(last(r_range), nr)
            if r_last < r_first
                continue
            end
            @simd for r_idx in r_first:r_last
                local_r = r_idx - r_first + 1
                if local_r <= size(NLᵀ_real, 3)
                    r_val = domain.r[r_idx, 4]
                    if r_val == 0.0
                        NLᵀ_real[local_lm, 1, local_r] = zero(T)
                        NLᵀ_imag[local_lm, 1, local_r] = zero(T)
                        NLᴾ_real[local_lm, 1, local_r] = zero(T)
                        NLᴾ_imag[local_lm, 1, local_r] = zero(T)
                    else
                        r⁻¹ = domain.r[r_idx, 3]
                        r⁻² = domain.r[r_idx, 2]
                        NLᵀ_real[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_profile_real[r_idx]
                                                             - d²ᴾ_dr²_real[r_idx]
                                                             - 2.0 * r⁻¹ * dᴾ_dr_real[r_idx])
                        NLᵀ_imag[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_profile_imag[r_idx]
                                                             - d²ᴾ_dr²_imag[r_idx]
                                                             - 2.0 * r⁻¹ * dᴾ_dr_imag[r_idx])
                        NLᴾ_real[local_lm, 1, local_r] = -ℓ_factor * r⁻² * 𝒯ᵘᵇ_real[local_lm, 1, local_r]
                        NLᴾ_imag[local_lm, 1, local_r] = -ℓ_factor * r⁻² * 𝒯ᵘᵇ_imag[local_lm, 1, local_r]
                    end
                end
            end
        end
    end
end


# ========================================================
# Inner core rotation effects
# ========================================================
function add_inner_core_rotation!(ℬ::SHTnsMagneticFields{T}, Ω::Float64) where T
    # Inner core rotation: affects boundary coupling
    # This modifies the nonlinear terms based on inner core rotation
    
    # Get local data views
    ic_tor_real = parent(ℬ.ic_toroidal.data_real)
    ic_tor_imag = parent(ℬ.ic_toroidal.data_imag)
    ic_pol_real = parent(ℬ.ic_poloidal.data_real)
    ic_pol_imag = parent(ℬ.ic_poloidal.data_imag)
    
    NLᵀ_real = parent(ℬ.nlᵀ.data_real)
    NLᵀ_imag = parent(ℬ.nlᵀ.data_imag)
    NLᴾ_real = parent(ℬ.nlᴾ.data_real)
    NLᴾ_imag = parent(ℬ.nlᴾ.data_imag)
    
    # Get local ranges
    lm_range = get_local_range(ℬ.ic_toroidal.pencil, 1)
    r_range  = get_local_range(ℬ.ic_toroidal.pencil, 3)
    
    # Rotation factor for inner core coupling
    rotation_factor = Ω * 1e-3  # Scaled rotation rate
    
    # Add rotation effects to nonlinear terms at inner core boundary
    @inbounds for lm_idx in lm_range
        if lm_idx <= ℬ.ic_toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            m = ℬ.𝒯.config.m_values[lm_idx]
            
            # Only affects m ≠ 0 modes (azimuthal dependence)
            if m != 0
                # Apply at inner core boundary (first radial point)
                if 1 in r_range
                    local_r = 1 - first(r_range) + 1
                    if local_r <= size(NLᵀ_real, 3)
                        # Add rotation-induced coupling
                        coupling_factor = rotation_factor * Float64(m)
                        
                        # Cross-coupling between toroidal and poloidal due to rotation
                        NLᵀ_real[local_lm, 1, local_r] += coupling_factor * ic_pol_imag[local_lm, 1, local_r]
                        NLᵀ_imag[local_lm, 1, local_r] -= coupling_factor * ic_pol_real[local_lm, 1, local_r]
                        NLᴾ_real[local_lm, 1, local_r] -= coupling_factor * ic_tor_imag[local_lm, 1, local_r]
                        NLᴾ_imag[local_lm, 1, local_r] += coupling_factor * ic_tor_real[local_lm, 1, local_r]
                    end
                end
            end
        end
    end
end


# =======================
# Diagnostic functions
# =======================
function compute_magnetic_energy(ℬ::SHTnsMagneticFields{T}) where T
    # Compute magnetic energy in spectral space
    
    tor_real = parent(ℬ.𝒯.data_real)
    tor_imag = parent(ℬ.𝒯.data_imag)
    pol_real = parent(ℬ.𝒫.data_real)
    pol_imag = parent(ℬ.𝒫.data_imag)
    
    local_energy = zero(Float64)

    # Get local ranges using config-aware pencil topology
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    config = ℬ.𝒯.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range  = range_local(config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= ℬ.𝒯.nlm
            local_lm = lm_idx - first(lm_range) + 1
            ℓ_factor = ℬ.ℓ_factors[lm_idx]
            
            # Weight by l(l+1) for proper spectral integration
            weight = 1.0 / max(ℓ_factor, 1.0)
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    local_energy += weight * (
                        tor_real[local_lm, 1, local_r]^2 + 
                        tor_imag[local_lm, 1, local_r]^2 + 
                        pol_real[local_lm, 1, local_r]^2 + 
                        pol_imag[local_lm, 1, local_r]^2
                    )
                end
            end
        end
    end
    
    # Global sum across all processes
    return 0.5 * Allreduce(local_energy, MPI.SUM, get_comm())
end

function compute_ohmic_dissipation(ℬ::SHTnsMagneticFields{T}) where T
    # Compute Ohmic dissipation: η |∇ × B|²
    
    # Current density already computed in work arrays
    jᵀ_real = parent(ℬ.work_tor.data_real)
    jᵀ_imag = parent(ℬ.work_tor.data_imag)
    jᴾ_real = parent(ℬ.work_pol.data_real)
    jᴾ_imag = parent(ℬ.work_pol.data_imag)
    
    local_dissipation = zero(Float64)
    
    lm_range = get_local_range(ℬ.work_tor.pencil, 1)
    r_range  = get_local_range(ℬ.work_tor.pencil, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= ℬ.work_tor.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(jᵀ_real, 3)
                    local_dissipation += (
                        jᵀ_real[local_lm, 1, local_r]^2 + 
                        jᵀ_imag[local_lm, 1, local_r]^2 + 
                        jᴾ_real[local_lm, 1, local_r]^2 + 
                        jᴾ_imag[local_lm, 1, local_r]^2
                    )
                end
            end
        end
    end
    
    # Scale by diffusivity (unity in magnetic-diffusion units) and global sum
    return Allreduce(local_dissipation, MPI.SUM, get_comm())
end


# =======================
# Utility functions
# =======================
function zero_magnetic_work_arrays!(ℬ::SHTnsMagneticFields{T}) where T
    # Efficiently zero all work arrays with batch operations
    # Use threaded operations for better performance on large arrays
    Threads.@threads for arr in [
        parent(ℬ.work_tor.data_real),
        parent(ℬ.work_tor.data_imag),
        parent(ℬ.work_pol.data_real),
        parent(ℬ.work_pol.data_imag),
        parent(ℬ.work_physical.r_component.data),
        parent(ℬ.work_physical.θ_component.data),
        parent(ℬ.work_physical.φ_component.data),
        parent(ℬ.induction_physical.r_component.data),
        parent(ℬ.induction_physical.θ_component.data),
        parent(ℬ.induction_physical.φ_component.data)
    ]
        fill!(arr, zero(T))
    end
end


# ================================================================================
# Enhanced utility functions using pencil decomposition and SHTns integration
# ================================================================================

"""
    batch_magnetic_transforms!(ℬ::SHTnsMagneticFields{T}) where T
    
Perform batched transforms for better cache efficiency using shtnskit_transforms.jl
"""
function batch_magnetic_transforms!(ℬ::SHTnsMagneticFields{T}) where T
    # Use batched operations from shtnskit_transforms.jl for better performance
    specs = [ℬ.𝒯, ℬ.𝒫, ℬ.ic_toroidal, ℬ.ic_poloidal]
    physs = [ℬ.work_physical.r_component, ℬ.work_physical.θ_component, 
             ℬ.work_physical.φ_component, ℬ.magnetic.r_component]
    
    # Only transform if specs and physs have compatible lengths
    n_transform = min(length(specs), length(physs))
    if n_transform > 0
        batch_spectral_to_physical!(specs[1:n_transform], physs[1:n_transform])
    end
end


"""
    optimize_magnetic_memory_layout!(ℬ::SHTnsMagneticFields{T}) where T
    
Optimize memory layout for better cache performance using pencil topology
"""
function optimize_magnetic_memory_layout!(ℬ::SHTnsMagneticFields{T}) where T
    # Use transpose plans for optimal data layout based on upcoming operations
    config = ℬ.𝒯.config
    
    # Use transpose plans if available
    plans = config.transpose_plans
    if !isempty(plans) && haskey(plans, :r_to_spec)
        transpose_with_timer!(ℬ.work_tor.data_real, ℬ.𝒯.data_real, 
                              plans[:r_to_spec], "magnetic_toroidal_layout_opt")
        transpose_with_timer!(ℬ.work_pol.data_real, ℬ.𝒫.data_real, 
                              plans[:r_to_spec], "magnetic_poloidal_layout_opt")
    end
end


"""
    validate_magnetic_configuration(ℬ::SHTnsMagneticFields{T}, config::SHTnsKitConfig) where T
    
Validate magnetic field configuration consistency with SHTns setup
"""
function validate_magnetic_configuration(ℬ::SHTnsMagneticFields{T}, config::SHTnsKitConfig) where T
    errors = String[]
    
    # Check field dimensions match config
    if size(ℬ.𝒯.data_real, 1) != config.nlm
        push!(errors, "Toroidal magnetic field size mismatch with config.nlm")
    end
    
    # Check that ℓ_factors are consistent
    if length(ℬ.ℓ_factors) != config.nlm
        push!(errors, "ℓ_factors length mismatch with config.nlm")
    end
    
    # Validate pencil topology consistency
    spec_range = range_local(config.pencils.spec, 1)
    if !isempty(spec_range) && maximum(spec_range) > config.nlm
        push!(errors, "Spectral pencil range exceeds config.nlm")
    end
    
    # Note: Transform manager checks removed - now handled by SHTnsKit directly
    
    # Check inner core field consistency
    if size(ℬ.ic_toroidal.data_real, 1) != config.nlm
        push!(errors, "Inner core toroidal field size mismatch with config.nlm")
    end
    
    if !isempty(errors)
        @warn "Magnetic configuration validation failed:\n" * join(errors, "\n")
        return false
    end
    
    return true
end


"""
    compute_magnetic_helicity(ℬ::SHTnsMagneticFields{T}) where T
    
Compute magnetic helicity using enhanced spectral integration
"""
function compute_magnetic_helicity(ℬ::SHTnsMagneticFields{T}) where T
    # Compute helicity H = ∫ A · B dV in spectral space
    # This requires the magnetic vector potential A
    
    # Get local data views
    tor_real = parent(ℬ.𝒯.data_real)
    tor_imag = parent(ℬ.𝒯.data_imag)
    pol_real = parent(ℬ.𝒫.data_real)
    pol_imag = parent(ℬ.𝒫.data_imag)
    
    local_helicity = zero(Float64)

    # Use configuration pencils for consistent range access
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    config = ℬ.𝒯.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range = range_local(config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= ℬ.𝒯.nlm
            local_lm = lm_idx - first(lm_range) + 1
            ℓ_factor = ℬ.ℓ_factors[lm_idx]
            
            # Weight for helicity calculation
            weight = 1.0 / max(sqrt(ℓ_factor), 1.0)
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    # Simplified helicity contribution (A·B ~ T²+P²/l)
                    local_helicity += weight * (
                        tor_real[local_lm, 1, local_r]^2 + 
                        tor_imag[local_lm, 1, local_r]^2 + 
                        (pol_real[local_lm, 1, local_r]^2 + 
                         pol_imag[local_lm, 1, local_r]^2) / max(ℓ_factor, 1.0)
                    )
                end
            end
        end
    end
    
    # Global sum across all processes
    return Allreduce(local_helicity, MPI.SUM, get_comm())
end


# Note: Boundary condition functions moved to src/bcs/magnetic.jl
