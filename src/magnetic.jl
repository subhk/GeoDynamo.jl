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
    toroidal::SHTnsSpectralField{T}
    poloidal::SHTnsSpectralField{T}

    # Inner core fields
    ic_toroidal::SHTnsSpectralField{T}
    ic_poloidal::SHTnsSpectralField{T}

    # Nonlinear terms (induction)
    nlᵀ::SHTnsSpectralField{T}
    nlᴾ::SHTnsSpectralField{T}
    prev_nlᵀ::SHTnsSpectralField{T}
    prev_nlᴾ::SHTnsSpectralField{T}

    # Work arrays
    work_tor::SHTnsSpectralField{T}
    work_pol::SHTnsSpectralField{T}
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
                                      Dᵒᶜ::RadialDomain, 
                                      Dⁱᶜ::RadialDomain, 
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
    magnetic = create_shtns_vector_field(T, config, Dᵒᶜ, pencils)
    current  = create_shtns_vector_field(T, config, Dᵒᶜ, pencils)
    
    # Spectral fields
    toroidal = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    poloidal = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    
    # Inner core fields (different domain)
    ic_toroidal = create_shtns_spectral_field(T, config, Dⁱᶜ, pencil_spec)
    ic_poloidal = create_shtns_spectral_field(T, config, Dⁱᶜ, pencil_spec)
    
    # Nonlinear terms
    nlᵀ = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    nlᴾ = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    prev_nlᵀ = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    prev_nlᴾ = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    
    # Work arrays
    work_tor = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    work_pol = create_shtns_spectral_field(T, config, Dᵒᶜ, pencil_spec)
    work_physical = create_shtns_vector_field(T, config, Dᵒᶜ, pencils)
    induction_physical = create_shtns_vector_field(T, config, Dᵒᶜ, pencils)
    
    # Pre-compute l(l+1) factors
    ℓ_factors = Float64[l * (l + 1) for l in config.l_values]

    # Create radial derivative matrices (cached for performance)
    ∂r  = create_derivative_matrix(1, Dᵒᶜ)
    ∂²r = create_derivative_matrix(2, Dᵒᶜ)

    # Create transpose plans for efficient data movement
    transpose_plans = create_transpose_plans(pencils)
    
    imposed_field = nothing
    boundary_condition_set = nothing
    boundary_cache = Dict{String, Any}()
    boundary_time_index = Ref{Int}(1)
    
    return SHTnsMagneticFields{T}(magnetic, current,
                                toroidal, poloidal,
                                ic_toroidal, ic_poloidal,
                                nlᵀ, nlᴾ, prev_nlᵀ, prev_nlᴾ,
                                work_tor, work_pol, work_physical,
                                induction_physical,
                                ℓ_factors,
                                ∂r, ∂²r,
                                imposed_field,
                                config,
                                Dᵒᶜ,
                                boundary_condition_set, boundary_cache, boundary_time_index)
end


# ========================================================
# Main nonlinear computation using enhanced transforms
# ========================================================
function compute_magnetic_nonlinear!(mag_fields::SHTnsMagneticFields{T},
                                    vel_fields, Dᵒᶜ::RadialDomain, Dⁱᶜ::RadialDomain,
                                    rotation_rate::Float64=0.0;
                                    geometry::Symbol = get_parameters().geometry) where T
    # Zero work arrays
    zero_magnetic_work_arrays!(mag_fields)
    
    # Step 1: Convert spectral B to physical space using enhanced transforms
    shtnskit_vector_synthesis!(mag_fields.toroidal, mag_fields.poloidal,
                               mag_fields.magnetic; domain=Dᵒᶜ)

    # Step 2: Compute current density j = ∇ × B in spectral space
    compute_current_density_spectral!(mag_fields, Dᵒᶜ)

    # Step 3: Transform current to physical space
    shtnskit_vector_synthesis!(mag_fields.work_tor, mag_fields.work_pol,
                               mag_fields.current; domain=Dᵒᶜ)
    
    # Step 4: Compute induction equation: ∂B/∂t = ∇ × (u × B) + η∇²B
    if vel_fields !== nothing
        compute_induction_term!(mag_fields, vel_fields; geometry)
    end
    
    # Step 5: Inner core rotation effects
    if rotation_rate != 0.0
        add_inner_core_rotation!(mag_fields, rotation_rate)
    end
    
    # Note: The nonlinear terms are now in mag_fields.nlᵀ/poloidal
end

"""
    enforce_magnetic_boundary_values!(fields)

Anchor magnetic toroidal/poloidal spectral data to cached Dirichlet boundary
values on the inner and outer radial surfaces.
"""
function enforce_magnetic_boundary_values!(fields::SHTnsMagneticFields{T}) where T
    domain = fields.outer_domain

    tor_real = parent(fields.toroidal.data_real)
    tor_imag = parent(fields.toroidal.data_imag)
    pol_real = parent(fields.poloidal.data_real)
    pol_imag = parent(fields.poloidal.data_imag)

    tor_bc = fields.toroidal.boundary_values
    pol_bc = fields.poloidal.boundary_values

    lm_range = get_local_range(fields.toroidal.pencil, 1)
    r_range = get_local_range(fields.toroidal.pencil, 3)

    has_inner = 1 in r_range && domain.r[1, 4] > 0
    has_outer = domain.N in r_range

    inner_idx = has_inner ? (1 - first(r_range) + 1) : 0
    outer_idx = has_outer ? (domain.N - first(r_range) + 1) : 0

    dirichlet_code = Int(bcs.DIRICHLET)

    for lm_idx in lm_range
        if lm_idx <= fields.toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1

            if has_inner && 1 <= inner_idx <= size(tor_real, 3)
                if fields.toroidal.bc_type_inner[lm_idx] == dirichlet_code
                    tor_real[local_lm, 1, inner_idx] = tor_bc[1, lm_idx]
                    tor_imag[local_lm, 1, inner_idx] = zero(T)
                end
                if fields.poloidal.bc_type_inner[lm_idx] == dirichlet_code
                    pol_real[local_lm, 1, inner_idx] = pol_bc[1, lm_idx]
                    pol_imag[local_lm, 1, inner_idx] = zero(T)
                end
            end

            if has_outer && 1 <= outer_idx <= size(tor_real, 3)
                if fields.toroidal.bc_type_outer[lm_idx] == dirichlet_code
                    tor_real[local_lm, 1, outer_idx] = tor_bc[2, lm_idx]
                    tor_imag[local_lm, 1, outer_idx] = zero(T)
                end
                if fields.poloidal.bc_type_outer[lm_idx] == dirichlet_code
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
        enforce_ball_vector_regularity!(fields.toroidal, fields.poloidal)
    end
    return fields
end


# ========================================================
# Current density computation in spectral space
# ========================================================
# ========================================================
# Current density computation in spectral space
# ========================================================
function compute_current_density_spectral!(mag_fields::SHTnsMagneticFields{T}, 
                                          Dᵒᶜ::RadialDomain) where T
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
    Bᵀ_real = parent(mag_fields.toroidal.data_real)
    Bᵀ_imag = parent(mag_fields.toroidal.data_imag)
    Bᴾ_real = parent(mag_fields.poloidal.data_real)
    Bᴾ_imag = parent(mag_fields.poloidal.data_imag)
    
    jᵀ_real = parent(mag_fields.work_tor.data_real)
    jᵀ_imag = parent(mag_fields.work_tor.data_imag)
    jᴾ_real = parent(mag_fields.work_pol.data_real)
    jᴾ_imag = parent(mag_fields.work_pol.data_imag)
    
    # Get local ranges using config-aware pencil topology
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    config = mag_fields.toroidal.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range  = range_local(config.pencils.spec, 3)
    total_nlm = config.nlm  # Total number of (l,m) modes

    # Use cached radial derivative matrices for performance
    d1_matrix = mag_fields.∂r   # First derivative d/dr
    d²_matrix = mag_fields.∂²r  # Second derivative d²/dr²

    # Pre-allocate work arrays for radial profiles
    nr = Dᵒᶜ.N
    Pᴾ_profile_real = zeros(T, nr)
    Pᴾ_profile_imag = zeros(T, nr)
    Pᴾ_gathered_real = zeros(T, nr)
    Pᴾ_gathered_imag = zeros(T, nr)
    dᴾ_dr_real    = zeros(T, nr)
    dᴾ_dr_imag    = zeros(T, nr)
    d²ᴾ_dr²_real  = zeros(T, nr)
    d²ᴾ_dr²_imag  = zeros(T, nr)

    # Check if we need MPI communication (radial dimension distributed)
    comm = get_comm()
    nprocs = MPI.Comm_size(comm)
    radial_distributed = length(r_range) < nr

    # CRITICAL MPI SYNCHRONIZATION:
    # When radial dimension is distributed, ALL processes must call Allreduce
    # for the SAME lm mode at the SAME time. Different processes may own different
    # lm_range, so we must loop over ALL lm modes to ensure synchronization.
    # Processes that don't own a mode contribute zeros.
    if radial_distributed && nprocs > 1
        # MPI path: All processes loop over ALL lm modes for proper synchronization
        @inbounds for lm_idx in 1:total_nlm
            i_own_this_mode = lm_idx in lm_range
            ℓ_factor = mag_fields.ℓ_factors[lm_idx]

            # Extract radial profile (owners contribute data, non-owners contribute zeros)
            fill!(Pᴾ_profile_real, zero(T))
            fill!(Pᴾ_profile_imag, zero(T))
            if i_own_this_mode
                local_lm = lm_idx - first(lm_range) + 1
                for r_idx in r_range
                    local_r = r_idx - first(r_range) + 1
                    if local_r <= size(Bᴾ_real, 3)
                        Pᴾ_profile_real[r_idx] = Bᴾ_real[local_lm, 1, local_r]
                        Pᴾ_profile_imag[r_idx] = Bᴾ_imag[local_lm, 1, local_r]
                    end
                end
            end

            # ALL processes call Allreduce together for this lm mode
            MPI.Allreduce!(Pᴾ_profile_real, Pᴾ_gathered_real, MPI.SUM, comm)
            MPI.Allreduce!(Pᴾ_profile_imag, Pᴾ_gathered_imag, MPI.SUM, comm)

            # Only mode owners compute derivatives and store results
            if i_own_this_mode
                local_lm = lm_idx - first(lm_range) + 1

                # Compute radial derivatives using complete profile
                apply_∂r!(dᴾ_dr_real, d1_matrix, Pᴾ_gathered_real)
                apply_∂r!(dᴾ_dr_imag, d1_matrix, Pᴾ_gathered_imag)
                apply_∂r!(d²ᴾ_dr²_real, d²_matrix, Pᴾ_gathered_real)
                apply_∂r!(d²ᴾ_dr²_imag, d²_matrix, Pᴾ_gathered_imag)

                # Compute current density components
                r_first = first(r_range)
                r_last = min(last(r_range), nr)
                @simd for r_idx in r_first:r_last
                    local_r = r_idx - r_first + 1
                    if local_r <= size(jᵀ_real, 3)
                        r_val = Dᵒᶜ.r[r_idx, 4]
                        if r_val == 0.0
                            jᵀ_real[local_lm, 1, local_r] = zero(T)
                            jᵀ_imag[local_lm, 1, local_r] = zero(T)
                            jᴾ_real[local_lm, 1, local_r] = zero(T)
                            jᴾ_imag[local_lm, 1, local_r] = zero(T)
                        else
                            r⁻¹ = Dᵒᶜ.r[r_idx, 3]
                            r⁻² = Dᵒᶜ.r[r_idx, 2]
                            jᵀ_real[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_gathered_real[r_idx]
                                                                - d²ᴾ_dr²_real[r_idx]
                                                                - 2.0 * r⁻¹ * dᴾ_dr_real[r_idx])
                            jᵀ_imag[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_gathered_imag[r_idx]
                                                                - d²ᴾ_dr²_imag[r_idx]
                                                                - 2.0 * r⁻¹ * dᴾ_dr_imag[r_idx])
                            jᴾ_real[local_lm, 1, local_r] = -ℓ_factor * r⁻² * Bᵀ_real[local_lm, 1, local_r]
                            jᴾ_imag[local_lm, 1, local_r] = -ℓ_factor * r⁻² * Bᵀ_imag[local_lm, 1, local_r]
                        end
                    end
                end
            end
        end
    else
        # Serial/local-radial path: Only process owned modes (no MPI communication needed)
        @inbounds for lm_idx in lm_range
            if lm_idx <= length(mag_fields.ℓ_factors)
                local_lm = lm_idx - first(lm_range) + 1
                ℓ_factor = mag_fields.ℓ_factors[lm_idx]

                # Extract radial profile (all radial data is local)
                fill!(Pᴾ_profile_real, zero(T))
                fill!(Pᴾ_profile_imag, zero(T))
                for r_idx in r_range
                    local_r = r_idx - first(r_range) + 1
                    if local_r <= size(Bᴾ_real, 3)
                        Pᴾ_profile_real[r_idx] = Bᴾ_real[local_lm, 1, local_r]
                        Pᴾ_profile_imag[r_idx] = Bᴾ_imag[local_lm, 1, local_r]
                    end
                end

                # No MPI needed - use profile directly
                copyto!(Pᴾ_gathered_real, Pᴾ_profile_real)
                copyto!(Pᴾ_gathered_imag, Pᴾ_profile_imag)

                # Compute radial derivatives
                apply_∂r!(dᴾ_dr_real, d1_matrix, Pᴾ_gathered_real)
                apply_∂r!(dᴾ_dr_imag, d1_matrix, Pᴾ_gathered_imag)
                apply_∂r!(d²ᴾ_dr²_real, d²_matrix, Pᴾ_gathered_real)
                apply_∂r!(d²ᴾ_dr²_imag, d²_matrix, Pᴾ_gathered_imag)

                # Compute current density components
                r_first = first(r_range)
                r_last = min(last(r_range), nr)
                if r_last < r_first
                    continue
                end
                @simd for r_idx in r_first:r_last
                    local_r = r_idx - r_first + 1
                    if local_r <= size(jᵀ_real, 3)
                        r_val = Dᵒᶜ.r[r_idx, 4]
                        if r_val == 0.0
                            jᵀ_real[local_lm, 1, local_r] = zero(T)
                            jᵀ_imag[local_lm, 1, local_r] = zero(T)
                            jᴾ_real[local_lm, 1, local_r] = zero(T)
                            jᴾ_imag[local_lm, 1, local_r] = zero(T)
                        else
                            r⁻¹ = Dᵒᶜ.r[r_idx, 3]
                            r⁻² = Dᵒᶜ.r[r_idx, 2]
                            jᵀ_real[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_gathered_real[r_idx]
                                                                - d²ᴾ_dr²_real[r_idx]
                                                                - 2.0 * r⁻¹ * dᴾ_dr_real[r_idx])
                            jᵀ_imag[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_gathered_imag[r_idx]
                                                                - d²ᴾ_dr²_imag[r_idx]
                                                                - 2.0 * r⁻¹ * dᴾ_dr_imag[r_idx])
                            jᴾ_real[local_lm, 1, local_r] = -ℓ_factor * r⁻² * Bᵀ_real[local_lm, 1, local_r]
                            jᴾ_imag[local_lm, 1, local_r] = -ℓ_factor * r⁻² * Bᵀ_imag[local_lm, 1, local_r]
                        end
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

function compute_induction_term!(mag_fields::SHTnsMagneticFields{T}, vel_fields; geometry::Symbol = get_parameters().geometry) where T
    # =========================================================================
    # Compute ∇×(u×B) for the induction equation in three steps
    # =========================================================================

    # Step 1: Compute u×B in PHYSICAL space
    # -------------------------------------
    # Cross product is simple point-wise operation in physical space:
    #   (u×B)_r = uθ Bφ - uφ Bθ
    #   (u×B)_θ = uφ Bᵣ - uᵣ Bφ
    #   (u×B)_φ = uᵣ Bθ - uθ Bᵣ
    compute_velocity_cross_magnetic!(mag_fields, vel_fields)

    # Step 2: Transform u×B to SPECTRAL space
    # ----------------------------------------
    # SHTns vector analysis decomposes (u×B) into toroidal and poloidal parts
    if geometry === :ball
        ball_vector_analysis!(mag_fields.induction_physical,
                             mag_fields.work_tor, mag_fields.work_pol)
    else
        shtnskit_vector_analysis!(mag_fields.induction_physical,
                                  mag_fields.work_tor, mag_fields.work_pol)
    end

    # Step 3: Compute ∇×(u×B) in SPECTRAL space
    # ------------------------------------------
    # Uses the spectral curl operator for toroidal-poloidal decomposition
    compute_curl_of_induction!(mag_fields)
end


function compute_velocity_cross_magnetic!(mag_fields::SHTnsMagneticFields{T}, vel_fields) where T
    # Compute u × B in physical space with enhanced memory access
    
    # Get local data views
    uᵣ = parent(vel_fields.velocity.r_component.data)
    uθ = parent(vel_fields.velocity.θ_component.data)
    uφ = parent(vel_fields.velocity.φ_component.data)
    
    Bᵣ = parent(mag_fields.magnetic.r_component.data)
    Bθ = parent(mag_fields.magnetic.θ_component.data)
    Bφ = parent(mag_fields.magnetic.φ_component.data)
    
    # Output: u × B
    uBᵣ = parent(mag_fields.induction_physical.r_component.data)
    uBθ = parent(mag_fields.induction_physical.θ_component.data)
    uBφ = parent(mag_fields.induction_physical.φ_component.data)
    
    # Get configuration for enhanced access patterns
    config = mag_fields.magnetic.r_component.config
    
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


function compute_curl_of_induction!(mag_fields::SHTnsMagneticFields{T}) where T
    # Compute ∇ × (u × B) in spectral space
    # This becomes the nonlinear term for the induction equation
    #
    # For toroidal-poloidal decomposition, curl satisfies:
    #   (∇×V)_tor = [l(l+1)/r² - d²/dr² - 2/r d/dr] V_pol
    #   (∇×V)_pol = -l(l+1)/r² V_tor
    #
    # This matches the vorticity and current density computations.

    # Get local data views
    uBᵀ_real = parent(mag_fields.work_tor.data_real)
    uBᵀ_imag = parent(mag_fields.work_tor.data_imag)
    uBᴾ_real = parent(mag_fields.work_pol.data_real)
    uBᴾ_imag = parent(mag_fields.work_pol.data_imag)

    NLᵀ_real = parent(mag_fields.nlᵀ.data_real)
    NLᵀ_imag = parent(mag_fields.nlᵀ.data_imag)
    NLᴾ_real = parent(mag_fields.nlᴾ.data_real)
    NLᴾ_imag = parent(mag_fields.nlᴾ.data_imag)

    # Get local ranges using config-aware pencil topology
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    config = mag_fields.toroidal.config
    domain = mag_fields.outer_domain
    lm_range = range_local(config.pencils.spec, 1)
    r_range  = range_local(config.pencils.spec, 3)
    total_nlm = config.nlm  # Total number of (l,m) modes

    nr = domain.N

    # Use cached radial derivative matrices for performance
    d1_matrix = mag_fields.∂r   # First derivative d/dr
    d²_matrix = mag_fields.∂²r  # Second derivative d²/dr²

    # Pre-allocate work arrays for radial profiles
    Pᴾ_profile_real = zeros(T, nr)
    Pᴾ_profile_imag = zeros(T, nr)
    Pᴾ_gathered_real = zeros(T, nr)
    Pᴾ_gathered_imag = zeros(T, nr)
    dᴾ_dr_real     = zeros(T, nr)
    dᴾ_dr_imag     = zeros(T, nr)
    d²ᴾ_dr²_real   = zeros(T, nr)
    d²ᴾ_dr²_imag   = zeros(T, nr)

    # Check if we need MPI communication (radial dimension distributed)
    comm = get_comm()
    nprocs = MPI.Comm_size(comm)
    radial_distributed = length(r_range) < nr

    # CRITICAL MPI SYNCHRONIZATION:
    # When radial dimension is distributed, ALL processes must call Allreduce
    # for the SAME lm mode at the SAME time. Different processes may own different
    # lm_range, so we must loop over ALL lm modes to ensure synchronization.
    if radial_distributed && nprocs > 1
        # MPI path: All processes loop over ALL lm modes for proper synchronization
        @inbounds for lm_idx in 1:total_nlm
            i_own_this_mode = lm_idx in lm_range
            ℓ_factor = mag_fields.ℓ_factors[lm_idx]

            # Extract radial profile (owners contribute data, non-owners contribute zeros)
            fill!(Pᴾ_profile_real, zero(T))
            fill!(Pᴾ_profile_imag, zero(T))
            if i_own_this_mode
                local_lm = lm_idx - first(lm_range) + 1
                for r_idx in r_range
                    local_r = r_idx - first(r_range) + 1
                    if local_r <= size(uBᴾ_real, 3)
                        Pᴾ_profile_real[r_idx] = uBᴾ_real[local_lm, 1, local_r]
                        Pᴾ_profile_imag[r_idx] = uBᴾ_imag[local_lm, 1, local_r]
                    end
                end
            end

            # ALL processes call Allreduce together for this lm mode
            MPI.Allreduce!(Pᴾ_profile_real, Pᴾ_gathered_real, MPI.SUM, comm)
            MPI.Allreduce!(Pᴾ_profile_imag, Pᴾ_gathered_imag, MPI.SUM, comm)

            # Only mode owners compute derivatives and store results
            if i_own_this_mode
                local_lm = lm_idx - first(lm_range) + 1

                # Compute radial derivatives using complete profile
                apply_∂r!(dᴾ_dr_real, d1_matrix, Pᴾ_gathered_real)
                apply_∂r!(dᴾ_dr_imag, d1_matrix, Pᴾ_gathered_imag)
                apply_∂r!(d²ᴾ_dr²_real, d²_matrix, Pᴾ_gathered_real)
                apply_∂r!(d²ᴾ_dr²_imag, d²_matrix, Pᴾ_gathered_imag)

                # Compute curl components
                r_first = first(r_range)
                r_last = min(last(r_range), nr)
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
                            NLᵀ_real[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_gathered_real[r_idx]
                                                                 - d²ᴾ_dr²_real[r_idx]
                                                                 - 2.0 * r⁻¹ * dᴾ_dr_real[r_idx])
                            NLᵀ_imag[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_gathered_imag[r_idx]
                                                                 - d²ᴾ_dr²_imag[r_idx]
                                                                 - 2.0 * r⁻¹ * dᴾ_dr_imag[r_idx])
                            NLᴾ_real[local_lm, 1, local_r] = -ℓ_factor * r⁻² * uBᵀ_real[local_lm, 1, local_r]
                            NLᴾ_imag[local_lm, 1, local_r] = -ℓ_factor * r⁻² * uBᵀ_imag[local_lm, 1, local_r]
                        end
                    end
                end
            end
        end
    else
        # Serial/local-radial path: Only process owned modes
        @inbounds for lm_idx in lm_range
            if lm_idx <= length(mag_fields.ℓ_factors)
                local_lm = lm_idx - first(lm_range) + 1
                ℓ_factor = mag_fields.ℓ_factors[lm_idx]

                # Extract radial profile (all radial data is local)
                fill!(Pᴾ_profile_real, zero(T))
                fill!(Pᴾ_profile_imag, zero(T))
                for r_idx in r_range
                    local_r = r_idx - first(r_range) + 1
                    if local_r <= size(uBᴾ_real, 3)
                        Pᴾ_profile_real[r_idx] = uBᴾ_real[local_lm, 1, local_r]
                        Pᴾ_profile_imag[r_idx] = uBᴾ_imag[local_lm, 1, local_r]
                    end
                end

                copyto!(Pᴾ_gathered_real, Pᴾ_profile_real)
                copyto!(Pᴾ_gathered_imag, Pᴾ_profile_imag)

                # Compute radial derivatives
                apply_∂r!(dᴾ_dr_real, d1_matrix, Pᴾ_gathered_real)
                apply_∂r!(dᴾ_dr_imag, d1_matrix, Pᴾ_gathered_imag)
                apply_∂r!(d²ᴾ_dr²_real, d²_matrix, Pᴾ_gathered_real)
                apply_∂r!(d²ᴾ_dr²_imag, d²_matrix, Pᴾ_gathered_imag)

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
                            NLᵀ_real[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_gathered_real[r_idx]
                                                                 - d²ᴾ_dr²_real[r_idx]
                                                                 - 2.0 * r⁻¹ * dᴾ_dr_real[r_idx])
                            NLᵀ_imag[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_gathered_imag[r_idx]
                                                                 - d²ᴾ_dr²_imag[r_idx]
                                                                 - 2.0 * r⁻¹ * dᴾ_dr_imag[r_idx])
                            NLᴾ_real[local_lm, 1, local_r] = -ℓ_factor * r⁻² * uBᵀ_real[local_lm, 1, local_r]
                            NLᴾ_imag[local_lm, 1, local_r] = -ℓ_factor * r⁻² * uBᵀ_imag[local_lm, 1, local_r]
                        end
                    end
                end
            end
        end
    end
end


# ========================================================
# Inner core rotation effects
# ========================================================
function add_inner_core_rotation!(mag_fields::SHTnsMagneticFields{T}, Ω::Float64) where T
    # Inner core rotation: affects boundary coupling
    # This modifies the nonlinear terms based on inner core rotation
    
    # Get local data views
    ic_tor_real = parent(mag_fields.ic_toroidal.data_real)
    ic_tor_imag = parent(mag_fields.ic_toroidal.data_imag)
    ic_pol_real = parent(mag_fields.ic_poloidal.data_real)
    ic_pol_imag = parent(mag_fields.ic_poloidal.data_imag)
    
    NLᵀ_real = parent(mag_fields.nlᵀ.data_real)
    NLᵀ_imag = parent(mag_fields.nlᵀ.data_imag)
    NLᴾ_real = parent(mag_fields.nlᴾ.data_real)
    NLᴾ_imag = parent(mag_fields.nlᴾ.data_imag)
    
    # Get local ranges
    lm_range = get_local_range(mag_fields.ic_toroidal.pencil, 1)
    r_range  = get_local_range(mag_fields.ic_toroidal.pencil, 3)
    
    # Rotation factor for inner core coupling
    rotation_factor = Ω * 1e-3  # Scaled rotation rate
    
    # Add rotation effects to nonlinear terms at inner core boundary
    @inbounds for lm_idx in lm_range
        if lm_idx <= mag_fields.ic_toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            m = mag_fields.toroidal.config.m_values[lm_idx]
            
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
function compute_magnetic_energy(mag_fields::SHTnsMagneticFields{T}) where T
    # Compute magnetic energy in spectral space
    
    tor_real = parent(mag_fields.toroidal.data_real)
    tor_imag = parent(mag_fields.toroidal.data_imag)
    pol_real = parent(mag_fields.poloidal.data_real)
    pol_imag = parent(mag_fields.poloidal.data_imag)
    
    local_energy = zero(Float64)

    # Get local ranges using config-aware pencil topology
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    config = mag_fields.toroidal.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range  = range_local(config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= mag_fields.toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            ℓ_factor = mag_fields.ℓ_factors[lm_idx]
            
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

function compute_ohmic_dissipation(mag_fields::SHTnsMagneticFields{T}) where T
    # Compute Ohmic dissipation: η |∇ × B|²
    
    # Current density already computed in work arrays
    jᵀ_real = parent(mag_fields.work_tor.data_real)
    jᵀ_imag = parent(mag_fields.work_tor.data_imag)
    jᴾ_real = parent(mag_fields.work_pol.data_real)
    jᴾ_imag = parent(mag_fields.work_pol.data_imag)
    
    local_dissipation = zero(Float64)
    
    lm_range = get_local_range(mag_fields.work_tor.pencil, 1)
    r_range  = get_local_range(mag_fields.work_tor.pencil, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= mag_fields.work_tor.nlm
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
function zero_magnetic_work_arrays!(mag_fields::SHTnsMagneticFields{T}) where T
    # Efficiently zero all work arrays with batch operations
    # Use threaded operations for better performance on large arrays
    Threads.@threads for arr in [
        parent(mag_fields.work_tor.data_real),
        parent(mag_fields.work_tor.data_imag),
        parent(mag_fields.work_pol.data_real),
        parent(mag_fields.work_pol.data_imag),
        parent(mag_fields.work_physical.r_component.data),
        parent(mag_fields.work_physical.θ_component.data),
        parent(mag_fields.work_physical.φ_component.data),
        parent(mag_fields.induction_physical.r_component.data),
        parent(mag_fields.induction_physical.θ_component.data),
        parent(mag_fields.induction_physical.φ_component.data)
    ]
        fill!(arr, zero(T))
    end
end


# ================================================================================
# Enhanced utility functions using pencil decomposition and SHTns integration
# ================================================================================

"""
    batch_magnetic_transforms!(mag_fields::SHTnsMagneticFields{T}) where T
    
Perform batched transforms for better cache efficiency using shtnskit_transforms.jl
"""
function batch_magnetic_transforms!(mag_fields::SHTnsMagneticFields{T}) where T
    # Use batched operations from shtnskit_transforms.jl for better performance
    specs = [mag_fields.toroidal, mag_fields.poloidal, mag_fields.ic_toroidal, mag_fields.ic_poloidal]
    physs = [mag_fields.work_physical.r_component, mag_fields.work_physical.θ_component, 
             mag_fields.work_physical.φ_component, mag_fields.magnetic.r_component]
    
    # Only transform if specs and physs have compatible lengths
    n_transform = min(length(specs), length(physs))
    if n_transform > 0
        batch_spectral_to_physical!(specs[1:n_transform], physs[1:n_transform])
    end
end


"""
    optimize_magnetic_memory_layout!(mag_fields::SHTnsMagneticFields{T}) where T
    
Optimize memory layout for better cache performance using pencil topology
"""
function optimize_magnetic_memory_layout!(mag_fields::SHTnsMagneticFields{T}) where T
    # Use transpose plans for optimal data layout based on upcoming operations
    config = mag_fields.toroidal.config
    
    # Use transpose plans if available
    plans = config.transpose_plans
    if !isempty(plans) && haskey(plans, :r_to_spec)
        transpose_with_timer!(mag_fields.work_tor.data_real, mag_fields.toroidal.data_real, 
                              plans[:r_to_spec], "magnetic_toroidal_layout_opt")
        transpose_with_timer!(mag_fields.work_pol.data_real, mag_fields.poloidal.data_real, 
                              plans[:r_to_spec], "magnetic_poloidal_layout_opt")
    end
end


"""
    validate_magnetic_configuration(mag_fields::SHTnsMagneticFields{T}, config::SHTnsKitConfig) where T
    
Validate magnetic field configuration consistency with SHTns setup
"""
function validate_magnetic_configuration(mag_fields::SHTnsMagneticFields{T}, config::SHTnsKitConfig) where T
    errors = String[]
    
    # Check field dimensions match config
    if size(mag_fields.toroidal.data_real, 1) != config.nlm
        push!(errors, "Toroidal magnetic field size mismatch with config.nlm")
    end
    
    # Check that ℓ_factors are consistent
    if length(mag_fields.ℓ_factors) != config.nlm
        push!(errors, "ℓ_factors length mismatch with config.nlm")
    end
    
    # Validate pencil topology consistency
    spec_range = range_local(config.pencils.spec, 1)
    if !isempty(spec_range) && maximum(spec_range) > config.nlm
        push!(errors, "Spectral pencil range exceeds config.nlm")
    end
    
    # Note: Transform manager checks removed - now handled by SHTnsKit directly
    
    # Check inner core field consistency
    if size(mag_fields.ic_toroidal.data_real, 1) != config.nlm
        push!(errors, "Inner core toroidal field size mismatch with config.nlm")
    end
    
    if !isempty(errors)
        @warn "Magnetic configuration validation failed:\n" * join(errors, "\n")
        return false
    end
    
    return true
end


"""
    compute_magnetic_helicity(mag_fields::SHTnsMagneticFields{T}) where T
    
Compute magnetic helicity using enhanced spectral integration
"""
function compute_magnetic_helicity(mag_fields::SHTnsMagneticFields{T}) where T
    # Compute helicity H = ∫ A · B dV in spectral space
    # This requires the magnetic vector potential A
    
    # Get local data views
    tor_real = parent(mag_fields.toroidal.data_real)
    tor_imag = parent(mag_fields.toroidal.data_imag)
    pol_real = parent(mag_fields.poloidal.data_real)
    pol_imag = parent(mag_fields.poloidal.data_imag)
    
    local_helicity = zero(Float64)

    # Use configuration pencils for consistent range access
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    config = mag_fields.toroidal.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range = range_local(config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= mag_fields.toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            ℓ_factor = mag_fields.ℓ_factors[lm_idx]
            
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
