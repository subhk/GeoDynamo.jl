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
#    - Implementation: T = 0 and l-dependent Robin rows on P
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
# NOTE: The standard insulating magnetic path embeds T = 0 and
# l-dependent poloidal potential-matching rows in the implicit matrices.
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
import .bcs: BoundaryType, DIRICHLET, NEUMANN, NEUMANN_MAG_INNER, NEUMANN_MAG_OUTER, CONTINUITY_MAG

"""
    SHTnsMagneticFields{T}

Magnetic-field state for the GeoDynamo MHD solver.

This stores the physical magnetic/current fields, toroidal-poloidal spectral
coefficients for the outer and inner core, induction-term work arrays, and the
radial derivative operators used by magnetic update kernels.
"""
mutable struct SHTnsMagneticFields{
    T,
    C<:SHTnsKitConfig,
    VF<:SHTnsVectorField{T},
    SF<:SHTnsSpecField{T},
}
    # Physical space magnetic field
    magnetic::VF
    current::VF

    # Spectral representation
    toroidal::SF
    poloidal::SF

    # Inner core fields
    toroidal_ic::SF
    poloidal_ic::SF

    # Nonlinear terms (induction)
    nl_toroidal::SF
    nl_poloidal::SF
    prev_nl_toroidal::SF
    prev_nl_poloidal::SF

    # Work arrays
    work_tor::SF
    work_pol::SF
    work_physical::VF
    induction_physical::VF  # Added missing field for u×B

    # Pre-computed coefficients
    ℓ_factors::Vector{T}        # l(l+1) values

    # Radial derivative matrices (cached for performance)
    ∂r::BandedMatrix{T}          # First derivative d/dr
    ∂²r::BandedMatrix{T}         # Second derivative d²/dr²
    curl_work::NTuple{6,Vector{T}}

    # Transform manager removed; SHTnsKit transforms are used directly

    # Imposed field (if any)
    imposed_field::Union{VF, Nothing}
    config::C
    outer_domain::RadialDomain
    boundary_condition_set::Union{bcs.BoundaryConditionSet{T}, Nothing}
    boundary_interpolation_cache::bcs.BoundaryInterpolationCache{T}
    boundary_time_index::Ref{Int}
end


"""
    create_shtns_magnetic_fields(T, config, outer_domain, inner_domain; pencils=nothing, pencil_spec=nothing)

Allocate and initialize the magnetic field container used by GeoDynamo.
"""
function create_shtns_magnetic_fields(::Type{T}, config::C,
                                      outer_core_domain::RadialDomain, 
                                      inner_core_domain::RadialDomain, 
                                      pencils=nothing, pencil_spec=nothing) where {T,C<:SHTnsKitConfig}

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
    magnetic = create_shtns_vector_field(T, config, outer_core_domain, pencils)
    current  = create_shtns_vector_field(T, config, outer_core_domain, pencils)
    
    # Spectral fields
    toroidal = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    poloidal = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    
    # Inner core fields (different domain). Give them an inner-core-sized spectral
    # pencil (nr_inner radial points) instead of padding to the outer-core nr; the
    # (l, m) topology/slot layout is identical to the outer-core fields.
    pencil_spec_ic = create_inner_core_spectral_pencil(config, pencil_spec, inner_core_domain.N)
    toroidal_ic = create_shtns_spectral_field(T, config, inner_core_domain, pencil_spec_ic)
    poloidal_ic = create_shtns_spectral_field(T, config, inner_core_domain, pencil_spec_ic)
    
    # Nonlinear terms
    nl_toroidal = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    nl_poloidal = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    prev_nl_toroidal = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    prev_nl_poloidal = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    
    # Work arrays
    work_tor = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    work_pol = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    work_physical = create_shtns_vector_field(T, config, outer_core_domain, pencils)
    induction_physical = create_shtns_vector_field(T, config, outer_core_domain, pencils)
    
    # Pre-compute l(l+1) factors
    ℓ_factors = T[l * (l + 1) for l in config.l_values]

    # Create radial derivative matrices (cached for performance)
    ∂r  = create_derivative_matrix(T, 1, outer_core_domain)
    ∂²r = create_derivative_matrix(T, 2, outer_core_domain)
    curl_work = ntuple(_ -> zeros(T, outer_core_domain.N), 6)

    # Create transpose plans for efficient data movement
    transpose_plans = create_transpose_plans(pencils)
    
    imposed_field = nothing
    boundary_condition_set = nothing
    boundary_cache = bcs.BoundaryInterpolationCache(T)
    boundary_time_index = Ref{Int}(1)
    
    return SHTnsMagneticFields(magnetic, current,
                               toroidal, poloidal,
                               toroidal_ic, poloidal_ic,
                               nl_toroidal, nl_poloidal, prev_nl_toroidal, prev_nl_poloidal,
                               work_tor, work_pol, work_physical,
                               induction_physical,
                               ℓ_factors,
                               ∂r, ∂²r, curl_work,
                               imposed_field,
                               config,
                               outer_core_domain,
                               boundary_condition_set, boundary_cache, boundary_time_index)
end

# Include matrix-embedded magnetic BC functions
include("../../bcs/magnetic_bc.jl")

# Include conducting inner-core admittance module
include("inner_core.jl")

# ========================================================
# Main nonlinear computation using enhanced transforms
# ========================================================
function compute_magnetic_nonlinear!(ℬ::SHTnsMagneticFields{T},
                                    𝒰, outer_core_domain::RadialDomain, inner_core_domain::RadialDomain,
                                    rotation_rate::Float64=0.0;
                                    geometry::Symbol) where T
    # Zero work arrays
    zero_magnetic_work_arrays!(ℬ)
    
    # The magnetic explicit path is a three-stage pipeline:
    #   B -> j = curl(B) -> u×B -> curl(u×B),
    # with transforms only at the points where products must be formed in
    # physical space.

    # Step 1: Convert spectral B to physical space using enhanced transforms
    shtnskit_vector_synthesis!(ℬ.toroidal, ℬ.poloidal,
                               ℬ.magnetic; domain=outer_core_domain)

    # Step 2: Compute current density j = ∇ × B in spectral space
    compute_current_density_spectral!(ℬ, outer_core_domain)

    # Step 3: Transform current to physical space
    shtnskit_vector_synthesis!(ℬ.work_tor, ℬ.work_pol,
                               ℬ.current; domain=outer_core_domain)
    
    # Step 4: Compute induction equation: ∂B/∂t = ∇ × (u × B) + η∇²B
    if 𝒰 !== nothing
        compute_induction_term!(ℬ, 𝒰; geometry)
    end
    
    # Step 5: Inner core rotation effects
    if rotation_rate != 0.0
        add_inner_core_rotation!(ℬ, rotation_rate)
    end
    
    # Note: The nonlinear terms are now in ℬ.nl_toroidal/poloidal
end


# ========================================================
# Current density computation in spectral space
# ========================================================
# ========================================================
# Current density computation in spectral space
# ========================================================
"""
    _spectral_curl_torpol!(dst_tor_r, dst_tor_i, dst_pol_r, dst_pol_i,
                           src_tor_r, src_tor_i, src_pol_r, src_pol_i,
                           ℓ_factors, d1_matrix, d²_matrix, domain, config, T)

Shared spectral curl for toroidal-poloidal fields:
    (∇×V)_tor = [l(l+1)/r² - d²/dr² - 2/r d/dr] V_pol
    (∇×V)_pol = -l(l+1)/r² V_tor

Used by both current density (j = ∇×B) and induction curl (∇×(u×B)).
"""
function _spectral_curl_torpol!(
    dst_tor_r, dst_tor_i, dst_pol_r, dst_pol_i,
    src_tor_r, src_tor_i, src_pol_r, src_pol_i,
    ℓ_factors, d1_matrix, d²_matrix, domain::RadialDomain, config::C, ::Type{T};
    _work::Union{Nothing, NTuple{6, Vector{T}}}=nothing
) where {T,C<:SHTnsKitConfig}
    lm_range = local_spectral_mode_indices(config)
    r_range  = range_local(config.pencils.spec, 3)
    nr = domain.N

    if _work !== nothing
        Pᴾ_profile_real, Pᴾ_profile_imag, dᴾ_dr_real, dᴾ_dr_imag, d²ᴾ_dr²_real, d²ᴾ_dr²_imag = _work
    else
        Pᴾ_profile_real = zeros(T, nr)
        Pᴾ_profile_imag = zeros(T, nr)
        dᴾ_dr_real      = zeros(T, nr)
        dᴾ_dr_imag      = zeros(T, nr)
        d²ᴾ_dr²_real    = zeros(T, nr)
        d²ᴾ_dr²_imag    = zeros(T, nr)
    end

    # Curl is evaluated mode-by-mode: gather one poloidal radial profile,
    # differentiate it, and combine it with the toroidal coefficient at that
    # same mode. That shared kernel is reused for both current density and the
    # curl of the induction term.
    for lm_idx in lm_range
        if lm_idx > length(ℓ_factors)
            continue
        end
        slot = local_spectral_storage_slot(config, lm_idx)
        if slot === nothing
            continue
        end
        ℓ_factor = ℓ_factors[lm_idx]

        fill!(Pᴾ_profile_real, zero(T))
        fill!(Pᴾ_profile_imag, zero(T))
        gather_local_radial_profile!(Pᴾ_profile_real, Pᴾ_profile_imag, src_pol_r, src_pol_i, slot, r_range)

        apply_∂r!(dᴾ_dr_real, d1_matrix, Pᴾ_profile_real)
        apply_∂r!(dᴾ_dr_imag, d1_matrix, Pᴾ_profile_imag)
        apply_∂r!(d²ᴾ_dr²_real, d²_matrix, Pᴾ_profile_real)
        apply_∂r!(d²ᴾ_dr²_imag, d²_matrix, Pᴾ_profile_imag)

        r_first = first(r_range)
        r_last = min(last(r_range), nr)
        r_last < r_first && continue

        @inbounds @simd for r_idx in r_first:r_last
            local_r = r_idx - r_first + 1
            if local_r <= size(dst_tor_r, 3)
                r_val = domain.r[r_idx, 4]
                if r_val == 0.0
                    set_local_spectral_value!(dst_tor_r, slot, local_r, zero(T))
                    set_local_spectral_value!(dst_tor_i, slot, local_r, zero(T))
                    set_local_spectral_value!(dst_pol_r, slot, local_r, zero(T))
                    set_local_spectral_value!(dst_pol_i, slot, local_r, zero(T))
                else
                    r⁻¹ = domain.r[r_idx, 3]
                    r⁻² = domain.r[r_idx, 2]
                    set_local_spectral_value!(dst_tor_r, slot, local_r,
                                              ℓ_factor * r⁻² * Pᴾ_profile_real[r_idx] -
                                              d²ᴾ_dr²_real[r_idx] -
                                              2.0 * r⁻¹ * dᴾ_dr_real[r_idx])
                    set_local_spectral_value!(dst_tor_i, slot, local_r,
                                              ℓ_factor * r⁻² * Pᴾ_profile_imag[r_idx] -
                                              d²ᴾ_dr²_imag[r_idx] -
                                              2.0 * r⁻¹ * dᴾ_dr_imag[r_idx])
                    set_local_spectral_value!(dst_pol_r, slot, local_r,
                                              -ℓ_factor * r⁻² * local_spectral_value(src_tor_r, slot, local_r))
                    set_local_spectral_value!(dst_pol_i, slot, local_r,
                                              -ℓ_factor * r⁻² * local_spectral_value(src_tor_i, slot, local_r))
                end
            end
        end
    end
end

function compute_current_density_spectral!(ℬ::SHTnsMagneticFields{T},
                                          outer_core_domain::RadialDomain) where T
    # j = ∇ × B: source is B (toroidal,poloidal), destination is work arrays
    _spectral_curl_torpol!(
        parent(ℬ.work_tor.data_real), parent(ℬ.work_tor.data_imag),
        parent(ℬ.work_pol.data_real), parent(ℬ.work_pol.data_imag),
        parent(ℬ.toroidal.data_real), parent(ℬ.toroidal.data_imag),
        parent(ℬ.poloidal.data_real), parent(ℬ.poloidal.data_imag),
        ℬ.ℓ_factors, ℬ.∂r, ℬ.∂²r, outer_core_domain, ℬ.toroidal.config, T;
        _work=ℬ.curl_work,
    )
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

function compute_induction_term!(ℬ::SHTnsMagneticFields{T}, 𝒰; geometry::Symbol) where T
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
    # SHTns vector analysis decomposes the physical cross product into
    # toroidal/poloidal coefficients so the curl operator can be applied with
    # the same spectral machinery used elsewhere.
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
    # All physical-space arrays share the same pencil layout and must be equal in size
    @assert length(uᵣ) == length(Bᵣ) "Velocity and magnetic field arrays must have the same size"
    @inbounds @simd for idx in eachindex(uᵣ)
        # u × B = (uθ Bφ - uφ Bθ, uφ Bᵣ - uᵣ Bφ, uᵣ Bθ - uθ Bᵣ)
        uBᵣ[idx] = uθ[idx] * Bφ[idx] - uφ[idx] * Bθ[idx]
        uBθ[idx] = uφ[idx] * Bᵣ[idx] - uᵣ[idx] * Bφ[idx]
        uBφ[idx] = uᵣ[idx] * Bθ[idx] - uθ[idx] * Bᵣ[idx]
    end
end


function compute_curl_of_induction!(ℬ::SHTnsMagneticFields{T}) where T
    # ∇ × (u × B): source is work arrays, destination is NL arrays
    _spectral_curl_torpol!(
        parent(ℬ.nl_toroidal.data_real), parent(ℬ.nl_toroidal.data_imag),
        parent(ℬ.nl_poloidal.data_real), parent(ℬ.nl_poloidal.data_imag),
        parent(ℬ.work_tor.data_real), parent(ℬ.work_tor.data_imag),
        parent(ℬ.work_pol.data_real), parent(ℬ.work_pol.data_imag),
        ℬ.ℓ_factors, ℬ.∂r, ℬ.∂²r, ℬ.outer_domain, ℬ.toroidal.config, T;
        _work=ℬ.curl_work,
    )
end


# ========================================================
# Inner core rotation effects
# ========================================================
function add_inner_core_rotation!(ℬ::SHTnsMagneticFields{T}, Ω::Float64) where T
    # Inner core rotation: affects boundary coupling
    # This modifies the nonlinear terms based on inner core rotation
    
    # Get local data views
    ic_tor_real = parent(ℬ.toroidal_ic.data_real)
    ic_tor_imag = parent(ℬ.toroidal_ic.data_imag)
    ic_pol_real = parent(ℬ.poloidal_ic.data_real)
    ic_pol_imag = parent(ℬ.poloidal_ic.data_imag)
    
    NLᵀ_real = parent(ℬ.nl_toroidal.data_real)
    NLᵀ_imag = parent(ℬ.nl_toroidal.data_imag)
    NLᴾ_real = parent(ℬ.nl_poloidal.data_real)
    NLᴾ_imag = parent(ℬ.nl_poloidal.data_imag)
    
    # Get local ranges
    lm_range = get_local_range(ℬ.toroidal_ic.pencil, 1)
    r_range  = get_local_range(ℬ.toroidal_ic.pencil, 3)
    
    # Rotation factor for inner core coupling (direct rotation rate, no arbitrary scaling)
    rotation_factor = Ω
    
    # This correction is intentionally localized to the inner boundary row: it
    # models boundary coupling from rotating inner-core data rather than a bulk
    # volumetric forcing.
    # Add rotation effects to nonlinear terms at inner core boundary
    @inbounds for lm_idx in lm_range
        if lm_idx <= ℬ.toroidal_ic.nlm
            slot = local_spectral_storage_slot(ℬ.toroidal.config, lm_idx)
            slot === nothing && continue
            m = ℬ.toroidal.config.m_values[lm_idx]
            
            # Only affects m ≠ 0 modes (azimuthal dependence)
            if m != 0
                # Apply at inner core boundary (first radial point)
                if 1 in r_range
                    local_r = 1 - first(r_range) + 1
                    if local_r <= size(NLᵀ_real, 3)
                        # Add rotation-induced coupling
                        coupling_factor = rotation_factor * Float64(m)
                        
                        # Cross-coupling between toroidal and poloidal due to rotation
                        set_local_spectral_value!(
                            NLᵀ_real, slot, local_r,
                            local_spectral_value(NLᵀ_real, slot, local_r) +
                            coupling_factor * local_spectral_value(ic_pol_imag, slot, local_r),
                        )
                        set_local_spectral_value!(
                            NLᵀ_imag, slot, local_r,
                            local_spectral_value(NLᵀ_imag, slot, local_r) -
                            coupling_factor * local_spectral_value(ic_pol_real, slot, local_r),
                        )
                        set_local_spectral_value!(
                            NLᴾ_real, slot, local_r,
                            local_spectral_value(NLᴾ_real, slot, local_r) -
                            coupling_factor * local_spectral_value(ic_tor_imag, slot, local_r),
                        )
                        set_local_spectral_value!(
                            NLᴾ_imag, slot, local_r,
                            local_spectral_value(NLᴾ_imag, slot, local_r) +
                            coupling_factor * local_spectral_value(ic_tor_real, slot, local_r),
                        )
                    end
                end
            end
        end
    end
end


# =======================
# Diagnostic functions
# =======================
"""
    compute_magnetic_energy(magnetic_fields, domain)

Compute the global magnetic energy from the spectral toroidal-poloidal
representation.
"""
function compute_magnetic_energy(ℬ::SHTnsMagneticFields{T}, domain::RadialDomain) where T
    # Compute magnetic energy in spectral space
    
    tor_real = parent(ℬ.toroidal.data_real)
    tor_imag = parent(ℬ.toroidal.data_imag)
    pol_real = parent(ℬ.poloidal.data_real)
    pol_imag = parent(ℬ.poloidal.data_imag)
    
    local_energy = zero(Float64)

    # Get local ranges using config-aware pencil topology
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    config = ℬ.toroidal.config
    lm_range = local_spectral_mode_indices(config)
    r_range  = range_local(config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= ℬ.toroidal.nlm
            slot = local_spectral_storage_slot(config, lm_idx)
            slot === nothing && continue
            ℓ_factor = ℬ.ℓ_factors[lm_idx]
            
            # Spectral magnetic energy weight: l(l+1) for toroidal-poloidal decomposition
            weight = Float64(ℓ_factor)

            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    # Include radial weight for spherical volume integration
                    r = domain.r[r_idx, 4]
                    r_weight = r^2 * domain.integration_weights[r_idx]

                    local_energy += weight * r_weight * (
                        local_spectral_value(tor_real, slot, local_r)^2 +
                        local_spectral_value(tor_imag, slot, local_r)^2 +
                        local_spectral_value(pol_real, slot, local_r)^2 +
                        local_spectral_value(pol_imag, slot, local_r)^2
                    )
                end
            end
        end
    end
    
    # Global sum across all processes
    return 0.5 * Allreduce(local_energy, MPI.SUM, get_comm())
end

"""
    compute_ohmic_dissipation(magnetic_fields)

Estimate the global Ohmic dissipation from the current-density work arrays.
"""
function compute_ohmic_dissipation(ℬ::SHTnsMagneticFields{T}) where T
    # Compute Ohmic dissipation: η |∇ × B|²
    
    # Current density already computed in work arrays
    jᵀ_real = parent(ℬ.work_tor.data_real)
    jᵀ_imag = parent(ℬ.work_tor.data_imag)
    jᴾ_real = parent(ℬ.work_pol.data_real)
    jᴾ_imag = parent(ℬ.work_pol.data_imag)
    
    local_dissipation = zero(Float64)
    
    config = ℬ.toroidal.config
    lm_range = local_spectral_mode_indices(config)
    r_range  = get_local_range(ℬ.work_tor.pencil, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= ℬ.work_tor.nlm
            slot = local_spectral_storage_slot(config, lm_idx)
            slot === nothing && continue
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(jᵀ_real, 3)
                    local_dissipation += (
                        local_spectral_value(jᵀ_real, slot, local_r)^2 +
                        local_spectral_value(jᵀ_imag, slot, local_r)^2 +
                        local_spectral_value(jᴾ_real, slot, local_r)^2 +
                        local_spectral_value(jᴾ_imag, slot, local_r)^2
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
    z = zero(T)
    fill!(parent(ℬ.work_tor.data_real), z)
    fill!(parent(ℬ.work_tor.data_imag), z)
    fill!(parent(ℬ.work_pol.data_real), z)
    fill!(parent(ℬ.work_pol.data_imag), z)
    fill!(parent(ℬ.work_physical.r_component.data), z)
    fill!(parent(ℬ.work_physical.θ_component.data), z)
    fill!(parent(ℬ.work_physical.φ_component.data), z)
    fill!(parent(ℬ.induction_physical.r_component.data), z)
    fill!(parent(ℬ.induction_physical.θ_component.data), z)
    fill!(parent(ℬ.induction_physical.φ_component.data), z)
end


# ================================================================================
# Enhanced utility functions using pencil decomposition and SHTns integration
# ================================================================================

"""
    batch_magnetic_transforms!(ℬ::SHTnsMagneticFields{T}) where T
    
Perform batched transforms for better cache efficiency using `transforms/spectral.jl`.
"""
function batch_magnetic_transforms!(ℬ::SHTnsMagneticFields{T}) where T
    # Use batched operations from transforms/spectral.jl for better performance.
    specs = [ℬ.toroidal, ℬ.poloidal, ℬ.toroidal_ic, ℬ.poloidal_ic]
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
    config = ℬ.toroidal.config
    
    # Use transpose plans if available
    plans = config.transpose_plans
    if !isempty(plans) && haskey(plans, :r_to_spec)
        transpose_with_timer!(ℬ.work_tor.data_real, ℬ.toroidal.data_real,
                              plans[:r_to_spec], "magnetic_toroidal_layout_opt")
        transpose_with_timer!(ℬ.work_pol.data_real, ℬ.poloidal.data_real, 
                              plans[:r_to_spec], "magnetic_poloidal_layout_opt")
    end
end


"""
    validate_magnetic_configuration(ℬ::SHTnsMagneticFields{T}, config::C) where {T,C<:SHTnsKitConfig}
    
Validate magnetic field configuration consistency with SHTns setup
"""
function validate_magnetic_configuration(ℬ::SHTnsMagneticFields{T}, config::C) where {T,C<:SHTnsKitConfig}
    errors = String[]
    
    # Check field dimensions match config
    local_slot_capacity = size(ℬ.toroidal.data_real, 1) * size(ℬ.toroidal.data_real, 2)
    local_mode_count = length(local_spectral_mode_indices(config))
    if local_mode_count > local_slot_capacity
        push!(errors, "Toroidal magnetic field local slot capacity is smaller than owned spectral mode count")
    end
    
    # Check that ℓ_factors are consistent
    if length(ℬ.ℓ_factors) != config.nlm
        push!(errors, "ℓ_factors length mismatch with config.nlm")
    end
    
    # Validate pencil topology consistency
    local_modes = local_spectral_mode_indices(config)
    if !isempty(local_modes) && maximum(local_modes) > config.nlm
        push!(errors, "Owned spectral mode index exceeds config.nlm")
    end
    
    # Note: Transform manager checks removed - now handled by SHTnsKit directly
    
    # Check inner core field consistency
    if local_mode_count > size(ℬ.toroidal_ic.data_real, 1) * size(ℬ.toroidal_ic.data_real, 2)
        push!(errors, "Inner core toroidal field local slot capacity is smaller than owned spectral mode count")
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
    tor_real = parent(ℬ.toroidal.data_real)
    tor_imag = parent(ℬ.toroidal.data_imag)
    pol_real = parent(ℬ.poloidal.data_real)
    pol_imag = parent(ℬ.poloidal.data_imag)
    
    local_helicity = zero(Float64)

    # Use configuration pencils for consistent range access
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    config = ℬ.toroidal.config
    lm_range = local_spectral_mode_indices(config)
    r_range = range_local(config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= ℬ.toroidal.nlm
            slot = local_spectral_storage_slot(config, lm_idx)
            slot === nothing && continue
            ℓ_factor = ℬ.ℓ_factors[lm_idx]
            
            # Weight for helicity calculation
            weight = 1.0 / max(sqrt(ℓ_factor), 1.0)
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    # Simplified helicity contribution (A·B ~ T²+P²/l)
                    local_helicity += weight * (
                        local_spectral_value(tor_real, slot, local_r)^2 +
                        local_spectral_value(tor_imag, slot, local_r)^2 +
                        (local_spectral_value(pol_real, slot, local_r)^2 +
                         local_spectral_value(pol_imag, slot, local_r)^2) / max(ℓ_factor, 1.0)
                    )
                end
            end
        end
    end
    
    # Global sum across all processes
    return Allreduce(local_helicity, MPI.SUM, get_comm())
end


# Note: Boundary condition functions moved to src/bcs/magnetic_bc.jl
