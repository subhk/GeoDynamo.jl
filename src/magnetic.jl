# ================================================================================
#  Magnetic field components with SHTns
# ================================================================================
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
#
import .BoundaryConditions
import .GeodynamoBall

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
    nl_toroidal::SHTnsSpectralField{T}
    nl_poloidal::SHTnsSpectralField{T}
    prev_nl_toroidal::SHTnsSpectralField{T}
    prev_nl_poloidal::SHTnsSpectralField{T}

    # Work arrays
    work_tor::SHTnsSpectralField{T}
    work_pol::SHTnsSpectralField{T}
    work_physical::SHTnsVectorField{T}
    induction_physical::SHTnsVectorField{T}  # Added missing field for u×B

    # Pre-computed coefficients
    l_factors::Vector{Float64}  # l(l+1) values

    # Transform manager removed; SHTnsKit transforms are used directly

    # Imposed field (if any)
    imposed_field::Union{SHTnsVectorField{T}, Nothing}
    config::SHTnsKitConfig
    outer_domain::RadialDomain
    boundary_condition_set::Union{BoundaryConditions.BoundaryConditionSet{T}, Nothing}
    boundary_interpolation_cache::Dict{String, Any}
    boundary_time_index::Ref{Int}
end


function create_shtns_magnetic_fields(::Type{T}, config::SHTnsKitConfig, 
                                      domain_oc::RadialDomain, 
                                      domain_ic::RadialDomain, 
                                      pencils=nothing, pencil_spec=nothing) where T

    # Use enhanced pencil topology from config if not provided
    if pencils === nothing
        pencils = create_pencil_topology(config, optimize=true)
    end
    pencil_θ, pencil_φ, pencil_r = pencils.θ, pencils.φ, pencils.r
    
    # Use spectral pencil from topology if not provided
    if pencil_spec === nothing
        pencil_spec = pencils.spec
    end
    
    # Physical space fields
    magnetic = create_shtns_vector_field(T, config, domain_oc, pencils)
    current  = create_shtns_vector_field(T, config, domain_oc, pencils)
    
    # Spectral fields
    toroidal = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    poloidal = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    
    # Inner core fields (different domain)
    ic_toroidal = create_shtns_spectral_field(T, config, domain_ic, pencil_spec)
    ic_poloidal = create_shtns_spectral_field(T, config, domain_ic, pencil_spec)
    
    # Nonlinear terms
    nl_toroidal = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    nl_poloidal = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    prev_nl_toroidal = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    prev_nl_poloidal = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    
    # Work arrays
    work_tor = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    work_pol = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    work_physical = create_shtns_vector_field(T, config, domain_oc, pencils)
    induction_physical = create_shtns_vector_field(T, config, domain_oc, pencils)
    
    # Pre-compute l(l+1) factors
    l_factors = Float64[l * (l + 1) for l in config.l_values]
    
    # Create transpose plans for efficient data movement
    transpose_plans = create_transpose_plans(pencils)
    
    imposed_field = nothing
    boundary_condition_set = nothing
    boundary_cache = Dict{String, Any}()
    boundary_time_index = Ref{Int}(1)
    
    return SHTnsMagneticFields{T}(magnetic, current, 
                                toroidal, poloidal,
                                ic_toroidal, ic_poloidal, 
                                nl_toroidal, nl_poloidal, prev_nl_toroidal, prev_nl_poloidal,
                                work_tor, work_pol, work_physical,
                                induction_physical,
                                l_factors,
                                imposed_field,
                                config,
                                domain_oc,
                                boundary_condition_set, boundary_cache, boundary_time_index)
end


# ========================================================
# Main nonlinear computation using enhanced transforms
# ========================================================
function compute_magnetic_nonlinear!(mag_fields::SHTnsMagneticFields{T}, 
                                    vel_fields, oc_domain::RadialDomain, ic_domain::RadialDomain,
                                    rotation_rate::Float64=0.0; geometry::Symbol = :shell) where T
    # Zero work arrays
    zero_magnetic_work_arrays!(mag_fields)
    
    # Step 1: Convert spectral B to physical space using enhanced transforms
    shtnskit_vector_synthesis!(mag_fields.toroidal, mag_fields.poloidal, 
                               mag_fields.magnetic)
    
    # Step 2: Compute current density j = ∇ × B in spectral space
    compute_current_density_spectral!(mag_fields, oc_domain)
    
    # Step 3: Transform current to physical space
    shtnskit_vector_synthesis!(mag_fields.work_tor, mag_fields.work_pol, 
                               mag_fields.current)
    
    # Step 4: Compute induction equation: ∂B/∂t = ∇ × (u × B) + η∇²B
    if vel_fields !== nothing
        compute_induction_term!(mag_fields, vel_fields; geometry)
    end
    
    # Step 5: Inner core rotation effects
    if rotation_rate != 0.0
        add_inner_core_rotation!(mag_fields, rotation_rate)
    end
    
    # Note: The nonlinear terms are now in mag_fields.nl_toroidal/poloidal
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

    dirichlet_code = Int(BoundaryConditions.DIRICHLET)

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

Refresh magnetic boundary data from the BoundaryConditions subsystem and
enforce the corresponding Dirichlet values in spectral space.
"""
function apply_magnetic_boundary_conditions!(fields::SHTnsMagneticFields{T};
                                              time_index::Union{Nothing,Int}=nothing) where T
    if fields.boundary_condition_set === nothing
        return fields
    end

    if time_index === nothing
        BoundaryConditions.apply_magnetic_boundary_conditions!(fields)
    else
        BoundaryConditions.apply_magnetic_boundary_conditions!(fields, time_index)
    end

    enforce_magnetic_boundary_values!(fields)

    if fields.outer_domain.r[1, 4] == 0.0
        GeodynamoBall.enforce_ball_vector_regularity!(fields.toroidal, fields.poloidal)
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
                                          oc_domain::RadialDomain) where T
    # Compute j = ∇ × B using spectral relationships with full radial derivatives
    # For toroidal-poloidal decomposition:
    # B = B_T + B_P where:
    #   B_T = ∇ × (T(r,θ,φ) r̂)
    #   B_P = ∇ × ∇ × (P(r,θ,φ) r̂)
    #
    # Current density j = ∇ × B:
    #   j_T = [l(l+1)/r² - d²/dr² - 2/r d/dr] P^{lm}
    #   j_P = -[l(l+1)/r²] T^{lm}
    
    # Get local data views
    B_tor_real = parent(mag_fields.toroidal.data_real)
    B_tor_imag = parent(mag_fields.toroidal.data_imag)
    B_pol_real = parent(mag_fields.poloidal.data_real)
    B_pol_imag = parent(mag_fields.poloidal.data_imag)
    
    j_tor_real = parent(mag_fields.work_tor.data_real)
    j_tor_imag = parent(mag_fields.work_tor.data_imag)
    j_pol_real = parent(mag_fields.work_pol.data_real)
    j_pol_imag = parent(mag_fields.work_pol.data_imag)
    
    # Get local ranges using config-aware pencil topology
    config = mag_fields.toroidal.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range  = range_local(config.pencils.r, 3)
    
    # Create radial derivative matrices
    d1_matrix = create_derivative_matrix(1, oc_domain)  # First derivative d/dr
    d2_matrix = create_derivative_matrix(2, oc_domain)  # Second derivative d²/dr²
    
    # Pre-allocate work arrays for radial profiles
    nr = oc_domain.N
    pol_profile_real = zeros(T, nr)
    pol_profile_imag = zeros(T, nr)
    dpol_dr_real    = zeros(T, nr)
    dpol_dr_imag    = zeros(T, nr)
    d2pol_dr2_real  = zeros(T, nr)
    d2pol_dr2_imag  = zeros(T, nr)
    
    # Process each (l,m) mode
    @inbounds for lm_idx in lm_range
        if lm_idx <= length(mag_fields.l_factors)
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = mag_fields.l_factors[lm_idx]  # l(l+1)
            
            # Extract radial profiles for poloidal field
            for r_idx in 1:nr
                if r_idx in r_range
                    local_r = r_idx - first(r_range) + 1
                    if local_r <= size(B_pol_real, 3)
                        pol_profile_real[r_idx] = B_pol_real[local_lm, 1, local_r]
                        pol_profile_imag[r_idx] = B_pol_imag[local_lm, 1, local_r]
                    end
                else
                    pol_profile_real[r_idx] = zero(T)
                    pol_profile_imag[r_idx] = zero(T)
                end
            end
            
            # Compute radial derivatives for poloidal field
            apply_derivative_matrix!(dpol_dr_real, d1_matrix, pol_profile_real)
            apply_derivative_matrix!(dpol_dr_imag, d1_matrix, pol_profile_imag)
            apply_derivative_matrix!(d2pol_dr2_real, d2_matrix, pol_profile_real)
            apply_derivative_matrix!(d2pol_dr2_imag, d2_matrix, pol_profile_imag)
            
            # Compute current density components
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(j_tor_real, 3)
                    r_val = oc_domain.r[r_idx, 4]
                    if r_val == 0.0
                        # At r=0 (ball geometry), enforce finite current density
                        j_tor_real[local_lm, 1, local_r] = 0
                        j_tor_imag[local_lm, 1, local_r] = 0
                        j_pol_real[local_lm, 1, local_r] = 0
                        j_pol_imag[local_lm, 1, local_r] = 0
                    else
                        r_inv = oc_domain.r[r_idx, 3]   # 1/r
                        r_inv2 = oc_domain.r[r_idx, 2]  # 1/r²
                        # Toroidal current from poloidal field (with full derivatives)
                        j_tor_real[local_lm, 1, local_r] = (l_factor * r_inv2 * pol_profile_real[r_idx] 
                                                            - d2pol_dr2_real[r_idx] 
                                                            - 2.0 * r_inv * dpol_dr_real[r_idx])
                        j_tor_imag[local_lm, 1, local_r] = (l_factor * r_inv2 * pol_profile_imag[r_idx] 
                                                            - d2pol_dr2_imag[r_idx] 
                                                            - 2.0 * r_inv * dpol_dr_imag[r_idx])
                        # Poloidal current from toroidal field (simpler - no radial derivatives)
                        j_pol_real[local_lm, 1, local_r] = -l_factor * r_inv2 * B_tor_real[local_lm, 1, local_r]
                        j_pol_imag[local_lm, 1, local_r] = -l_factor * r_inv2 * B_tor_imag[local_lm, 1, local_r]
                    end
                end
            end
        end
    end
end


# ==============================
# Induction term computation
# ==============================
function compute_induction_term!(mag_fields::SHTnsMagneticFields{T}, vel_fields; geometry::Symbol = get_parameters().geometry) where T
    # Compute ∇ × (u × B) for the induction equation
    
    # Step 1: Compute u × B in physical space
    compute_velocity_cross_magnetic!(mag_fields, vel_fields)
    
    # Step 2: Transform u × B to spectral space
    if geometry === :ball
        GeodynamoBall.ball_vector_analysis!(mag_fields.induction_physical, 
                                            mag_fields.work_tor, mag_fields.work_pol)
    else
        shtnskit_vector_analysis!(mag_fields.induction_physical, 
                                  mag_fields.work_tor, mag_fields.work_pol)
    end
    
    # Step 3: Compute curl of (u × B) in spectral space
    compute_curl_of_induction!(mag_fields)
end


function compute_velocity_cross_magnetic!(mag_fields::SHTnsMagneticFields{T}, vel_fields) where T
    # Compute u × B in physical space with enhanced memory access
    
    # Get local data views
    u_r = parent(vel_fields.velocity.r_component.data)
    u_θ = parent(vel_fields.velocity.θ_component.data)
    u_φ = parent(vel_fields.velocity.φ_component.data)
    
    B_r = parent(mag_fields.magnetic.r_component.data)
    B_θ = parent(mag_fields.magnetic.θ_component.data)
    B_φ = parent(mag_fields.magnetic.φ_component.data)
    
    # Output: u × B
    uxB_r = parent(mag_fields.induction_physical.r_component.data)
    uxB_θ = parent(mag_fields.induction_physical.θ_component.data)
    uxB_φ = parent(mag_fields.induction_physical.φ_component.data)
    
    # Get configuration for enhanced access patterns
    config = mag_fields.magnetic.r_component.config
    
    # Compute cross product with vectorization
    @inbounds @simd for idx in eachindex(u_r)
        if idx <= length(B_r)
            # u × B = (u_θ B_φ - u_φ B_θ, u_φ B_r - u_r B_φ, u_r B_θ - u_θ B_r)
            uxB_r[idx] = u_θ[idx] * B_φ[idx] - u_φ[idx] * B_θ[idx]
            uxB_θ[idx] = u_φ[idx] * B_r[idx] - u_r[idx] * B_φ[idx]
            uxB_φ[idx] = u_r[idx] * B_θ[idx] - u_θ[idx] * B_r[idx]
        end
    end
end


function compute_curl_of_induction!(mag_fields::SHTnsMagneticFields{T}) where T
    # Compute ∇ × (u × B) in spectral space
    # This becomes the nonlinear term for the induction equation
    
    # Get local data views
    uxB_tor_real = parent(mag_fields.work_tor.data_real)
    uxB_tor_imag = parent(mag_fields.work_tor.data_imag)
    uxB_pol_real = parent(mag_fields.work_pol.data_real)
    uxB_pol_imag = parent(mag_fields.work_pol.data_imag)
    
    nl_tor_real = parent(mag_fields.nl_toroidal.data_real)
    nl_tor_imag = parent(mag_fields.nl_toroidal.data_imag)
    nl_pol_real = parent(mag_fields.nl_poloidal.data_real)
    nl_pol_imag = parent(mag_fields.nl_poloidal.data_imag)
    
    # Get local ranges using config-aware pencil topology
    config = mag_fields.toroidal.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range  = range_local(config.pencils.r, 3)
    
    # Apply curl in spectral space
    @inbounds for lm_idx in lm_range
        if lm_idx <= length(mag_fields.l_factors)
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = mag_fields.l_factors[lm_idx]
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(uxB_tor_real, 3)
                    # Curl of (u × B)
                    nl_tor_real[local_lm, 1, local_r] = l_factor * uxB_pol_real[local_lm, 1, local_r]
                    nl_tor_imag[local_lm, 1, local_r] = l_factor * uxB_pol_imag[local_lm, 1, local_r]
                    nl_pol_real[local_lm, 1, local_r] = -l_factor * uxB_tor_real[local_lm, 1, local_r]
                    nl_pol_imag[local_lm, 1, local_r] = -l_factor * uxB_tor_imag[local_lm, 1, local_r]
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
    
    nl_tor_real = parent(mag_fields.nl_toroidal.data_real)
    nl_tor_imag = parent(mag_fields.nl_toroidal.data_imag)
    nl_pol_real = parent(mag_fields.nl_poloidal.data_real)
    nl_pol_imag = parent(mag_fields.nl_poloidal.data_imag)
    
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
                    if local_r <= size(nl_tor_real, 3)
                        # Add rotation-induced coupling
                        coupling_factor = rotation_factor * Float64(m)
                        
                        # Cross-coupling between toroidal and poloidal due to rotation
                        nl_tor_real[local_lm, 1, local_r] += coupling_factor * ic_pol_imag[local_lm, 1, local_r]
                        nl_tor_imag[local_lm, 1, local_r] -= coupling_factor * ic_pol_real[local_lm, 1, local_r]
                        nl_pol_real[local_lm, 1, local_r] -= coupling_factor * ic_tor_imag[local_lm, 1, local_r]
                        nl_pol_imag[local_lm, 1, local_r] += coupling_factor * ic_tor_real[local_lm, 1, local_r]
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
    config = mag_fields.toroidal.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range  = range_local(config.pencils.r, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= mag_fields.toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = mag_fields.l_factors[lm_idx]
            
            # Weight by l(l+1) for proper spectral integration
            weight = 1.0 / max(l_factor, 1.0)
            
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
    return 0.5 * MPI.Allreduce(local_energy, MPI.SUM, get_comm())
end

function compute_ohmic_dissipation(mag_fields::SHTnsMagneticFields{T}) where T
    # Compute Ohmic dissipation: η |∇ × B|²
    
    # Current density already computed in work arrays
    j_tor_real = parent(mag_fields.work_tor.data_real)
    j_tor_imag = parent(mag_fields.work_tor.data_imag)
    j_pol_real = parent(mag_fields.work_pol.data_real)
    j_pol_imag = parent(mag_fields.work_pol.data_imag)
    
    local_dissipation = zero(Float64)
    
    lm_range = get_local_range(mag_fields.work_tor.pencil, 1)
    r_range  = get_local_range(mag_fields.work_tor.pencil, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= mag_fields.work_tor.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(j_tor_real, 3)
                    local_dissipation += (
                        j_tor_real[local_lm, 1, local_r]^2 + 
                        j_tor_imag[local_lm, 1, local_r]^2 + 
                        j_pol_real[local_lm, 1, local_r]^2 + 
                        j_pol_imag[local_lm, 1, local_r]^2
                    )
                end
            end
        end
    end
    
    # Scale by diffusivity (unity in magnetic-diffusion units) and global sum
    return MPI.Allreduce(local_dissipation, MPI.SUM, get_comm())
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
    
    # Check that l_factors are consistent
    if length(mag_fields.l_factors) != config.nlm
        push!(errors, "l_factors length mismatch with config.nlm")
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
    config = mag_fields.toroidal.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range = range_local(config.pencils.r, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= mag_fields.toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = mag_fields.l_factors[lm_idx]
            
            # Weight for helicity calculation
            weight = 1.0 / max(sqrt(l_factor), 1.0)
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    # Simplified helicity contribution (A·B ~ T²+P²/l)
                    local_helicity += weight * (
                        tor_real[local_lm, 1, local_r]^2 + 
                        tor_imag[local_lm, 1, local_r]^2 + 
                        (pol_real[local_lm, 1, local_r]^2 + 
                         pol_imag[local_lm, 1, local_r]^2) / max(l_factor, 1.0)
                    )
                end
            end
        end
    end
    
    # Global sum across all processes
    return MPI.Allreduce(local_helicity, MPI.SUM, get_comm())
end


# Note: Boundary condition functions moved to src/BoundaryConditions/magnetic.jl
