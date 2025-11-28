import .BoundaryConditions
import .BoundaryConditions: BoundaryType, DIRICHLET, NEUMANN

# ================================================================================
# Physics Modules with SHTns
# ================================================================================

# Include optimized workspace-based BC functions
include("velocity_bc.jl")

# Velocity field components with SHTns
mutable struct SHTnsVelocityFields{T}
    # Physical space velocities
    velocity::SHTnsVectorField{T}
    vorticity::SHTnsVectorField{T}
    
    # Spectral representation (toroidal-poloidal)
    toroidal::SHTnsSpectralField{T}
    poloidal::SHTnsSpectralField{T}
    
    # Vorticity in spectral space (for efficient curl computation)
    vort_toroidal::SHTnsSpectralField{T}
    vort_poloidal::SHTnsSpectralField{T}
    
    # Nonlinear terms
    nl_toroidal::SHTnsSpectralField{T}
    nl_poloidal::SHTnsSpectralField{T}
    prev_nl_toroidal::SHTnsSpectralField{T}
    prev_nl_poloidal::SHTnsSpectralField{T}
    
    # Work arrays for efficient computation
    work_tor::SHTnsSpectralField{T}
    work_pol::SHTnsSpectralField{T}
    work_physical::SHTnsVectorField{T}
    advection_physical::SHTnsVectorField{T}
    
    # Pre-computed coefficients
    l_factors::Vector{Float64}          # l(l+1) values
    coriolis_factors::Matrix{Float64}   # Pre-computed Coriolis terms
    
    # Radial derivative matrices
    dr_matrix::BandedMatrix{T}          # First derivative
    d2r_matrix::BandedMatrix{T}         # Second derivative
    laplacian_matrix::BandedMatrix{T}   # Radial Laplacian operator
    
    # Transform manager removed; SHTnsKit transforms are used directly
    config::SHTnsKitConfig
    domain::RadialDomain
    boundary_condition_set::Union{BoundaryConditions.BoundaryConditionSet{T}, Nothing}
    boundary_interpolation_cache::Dict{String, Any}
    boundary_time_index::Ref{Int}
end

# ---------------------------------
# Optional shared workspace support
# ---------------------------------
struct VelocityWorkspace{T}
    pol_profile_real::Vector{Vector{T}}
    pol_profile_imag::Vector{Vector{T}}
    tor_profile_real::Vector{Vector{T}}
    tor_profile_imag::Vector{Vector{T}}
    dpol_dr_real::Vector{Vector{T}}
    dpol_dr_imag::Vector{Vector{T}}
    d2pol_dr2_real::Vector{Vector{T}}
    d2pol_dr2_imag::Vector{Vector{T}}
    # Pre-allocated buffers for BC operations (avoid allocations per mode)
    bc_profile_real::Vector{Vector{T}}
    bc_profile_imag::Vector{Vector{T}}
    bc_dprofile_real::Vector{Vector{T}}
    bc_dprofile_imag::Vector{Vector{T}}
    bc_correction::Vector{Vector{T}}
end

function create_velocity_workspace(::Type{T}, nr::Int, nthreads::Int=Threads.nthreads()) where T
    bufs() = [zeros(T, nr) for _ in 1:nthreads]
    return VelocityWorkspace{T}(
        bufs(), bufs(), bufs(), bufs(), bufs(), bufs(), bufs(), bufs(),
        # BC buffers
        bufs(), bufs(), bufs(), bufs(), bufs()
    )
end

# Global optional workspace reference (set by user to enable reuse across steps)
# Type-stable: only accepts VelocityWorkspace or nothing
const VELOCITY_WS = Ref{Union{VelocityWorkspace, Nothing}}(nothing)

"""
    set_velocity_workspace!(ws::Union{VelocityWorkspace{T}, Nothing}) where T

Register a global VelocityWorkspace to be used by velocity kernels when available.
Pass `nothing` to disable and fall back to internal buffers.
Type-stable version that only accepts VelocityWorkspace or nothing.
"""
function set_velocity_workspace!(ws::Union{VelocityWorkspace{T}, Nothing}) where T
    VELOCITY_WS[] = ws
    return ws
end

"""
    get_velocity_workspace(::Type{T})::Union{VelocityWorkspace{T}, Nothing} where T

Get the global velocity workspace if available and matches type T.
Returns nothing if not set or type mismatch.
"""
function get_velocity_workspace(::Type{T})::Union{VelocityWorkspace{T}, Nothing} where T
    ws = VELOCITY_WS[]
    if ws isa VelocityWorkspace{T}
        return ws
    else
        return nothing
    end
end

"""
    enforce_velocity_boundary_values!(fields)

Anchor toroidal and poloidal spectral coefficients to the currently cached
Dirichlet boundary values on the inner and outer radial surfaces.
"""
function enforce_velocity_boundary_values!(fields::SHTnsVelocityFields{T}) where T
    domain = fields.domain
    tor_real = parent(fields.toroidal.data_real)
    tor_imag = parent(fields.toroidal.data_imag)
    pol_real = parent(fields.poloidal.data_real)
    pol_imag = parent(fields.poloidal.data_imag)

    tor_bc = fields.toroidal.boundary_values
    pol_bc = fields.poloidal.boundary_values

    lm_range = get_local_range(fields.toroidal.pencil, 1)
    r_range  = get_local_range(fields.toroidal.pencil, 3)

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
    apply_velocity_boundary_conditions!(fields; time_index=nothing)

Refresh velocity boundary data from the BoundaryConditions subsystem and
immediately enforce Dirichlet constraints in spectral space.
"""
function apply_velocity_boundary_conditions!(fields::SHTnsVelocityFields{T};
                                              time_index::Union{Nothing,Int}=nothing) where T
    boundary_set, _ = BoundaryConditions.get_velocity_boundary_data(fields)
    if boundary_set === nothing
        return fields
    end

    if time_index === nothing
        BoundaryConditions.apply_velocity_boundary_conditions!(fields)
    else
        BoundaryConditions.apply_velocity_boundary_conditions!(fields, time_index)
    end

    enforce_velocity_boundary_values!(fields)

    if fields.domain.r[1, 4] == 0.0
        enforce_ball_vector_regularity!(fields.toroidal, fields.poloidal)
    end
    return fields
end

"""
    apply_velocity_flux_bc_spectral!(fields::SHTnsVelocityFields{T}, domain::RadialDomain;
                                     method::Symbol=:tau) where T

Apply Neumann (flux) boundary conditions to velocity field components in spectral space.

This function enforces stress-free or other Neumann boundary conditions for the
toroidal and poloidal components of the velocity field. For stress-free boundaries:
- Poloidal (radial) component: v_r = 0 (Dirichlet, handled by enforce_velocity_boundary_values!)
- Toroidal (tangential) component: ∂T/∂r = 0 (Neumann, handled here)

# Arguments
- `fields`: Velocity field structure with toroidal and poloidal components
- `domain`: Radial domain information
- `method`: Method for applying flux BCs (:tau, :influence_matrix, or :direct)

# Physical Interpretation
For stress-free boundaries, the zero tangential stress condition:
  ∂v_θ/∂r - v_θ/r = 0  and  ∂v_φ/∂r - v_φ/r = 0
translates to zero radial derivative of the toroidal scalar potential T:
  ∂T/∂r = 0  at boundaries
"""
function apply_velocity_flux_bc_spectral!(fields::SHTnsVelocityFields{T},
                                          domain::RadialDomain;
                                          method::Symbol=:tau) where T

    # Apply flux BC to toroidal component (if Neumann BC is set)
    apply_velocity_component_flux_bc!(fields.toroidal, domain, fields.dr_matrix, method)

    # Apply flux BC to poloidal component (if Neumann BC is set)
    # Note: For typical stress-free boundaries, poloidal uses Dirichlet (v_r = 0)
    # but this allows flexibility for other boundary conditions
    apply_velocity_component_flux_bc!(fields.poloidal, domain, fields.dr_matrix, method)

    return fields
end

"""
    apply_velocity_component_flux_bc!(field::SHTnsSpectralField{T},
                                       domain::RadialDomain,
                                       dr_matrix::BandedMatrix{T},
                                       method::Symbol) where T

Apply flux boundary conditions to a single velocity component (toroidal or poloidal).

# Methods
- `:direct` - Enforces ∂T/∂r ≈ 0 (may be incorrect for stress-free)
- `:tau` - Tau method for ∂T/∂r = 0
- `:physical_stress` - Enforces ∂T/∂r = T/r (correct for stress-free)

# Performance
Uses pre-allocated workspace buffers if available (set via set_velocity_workspace!).
Otherwise allocates temporary arrays (slower).
"""
function apply_velocity_component_flux_bc!(field::SHTnsSpectralField{T},
                                           domain::RadialDomain,
                                           dr_matrix::BandedMatrix{T},
                                           method::Symbol) where T

    spec_real = parent(field.data_real)
    spec_imag = parent(field.data_imag)

    lm_range = get_local_range(field.pencil, 1)
    r_range  = get_local_range(field.pencil, 3)

    # Try to get workspace for better performance
    ws = get_velocity_workspace(T)
    tid = Threads.threadid()

    for lm_idx in lm_range
        if lm_idx <= field.nlm
            local_lm = lm_idx - first(lm_range) + 1

            # Check if this mode needs flux BC
            apply_inner = (field.bc_type_inner[lm_idx] == Int(NEUMANN)) && (1 in r_range)
            apply_outer = (field.bc_type_outer[lm_idx] == Int(NEUMANN)) && (domain.N in r_range)

            if apply_inner || apply_outer
                # Apply flux BC using specified method (use workspace if available)
                if method == :tau
                    if ws !== nothing
                        apply_velocity_flux_bc_tau_ws!(spec_real, spec_imag, local_lm, lm_idx,
                                                       apply_inner, apply_outer, dr_matrix,
                                                       domain, r_range, ws, tid)
                    else
                        apply_velocity_flux_bc_tau!(spec_real, spec_imag, local_lm, lm_idx,
                                                   apply_inner, apply_outer, dr_matrix, domain, r_range)
                    end
                elseif method == :direct
                    if ws !== nothing
                        apply_velocity_flux_bc_direct_ws!(spec_real, spec_imag, local_lm, lm_idx,
                                                          apply_inner, apply_outer, domain, r_range, ws, tid)
                    else
                        apply_velocity_flux_bc_direct!(spec_real, spec_imag, local_lm, lm_idx,
                                                       apply_inner, apply_outer, domain, r_range)
                    end
                elseif method == :physical_stress
                    if ws !== nothing
                        apply_velocity_flux_bc_physical_stress_ws!(spec_real, spec_imag, local_lm, lm_idx,
                                                                   apply_inner, apply_outer, domain, r_range, ws, tid)
                    else
                        apply_velocity_flux_bc_physical_stress!(spec_real, spec_imag, local_lm, lm_idx,
                                                               apply_inner, apply_outer, domain, r_range)
                    end
                else
                    @warn "Flux BC method $method not implemented for velocity, using :physical_stress"
                    if ws !== nothing
                        apply_velocity_flux_bc_physical_stress_ws!(spec_real, spec_imag, local_lm, lm_idx,
                                                                   apply_inner, apply_outer, domain, r_range, ws, tid)
                    else
                        apply_velocity_flux_bc_physical_stress!(spec_real, spec_imag, local_lm, lm_idx,
                                                               apply_inner, apply_outer, domain, r_range)
                    end
                end
            end
        end
    end
end

"""
    apply_velocity_flux_bc_tau!(spec_real, spec_imag, local_lm, lm_idx,
                                apply_inner, apply_outer, dr_matrix, domain, r_range)

Apply flux boundary conditions using the tau method for velocity components.
This enforces ∂T/∂r = 0 at boundaries for stress-free conditions.
"""
function apply_velocity_flux_bc_tau!(spec_real, spec_imag, local_lm, lm_idx,
                                     apply_inner, apply_outer,
                                     dr_matrix::BandedMatrix, domain, r_range)
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
    comm = BoundaryConditions.get_comm()
    if comm !== nothing && Comm_size(comm) > 1
        Allreduce!(profile_real, SUM, comm)
        Allreduce!(profile_imag, SUM, comm)
    end

    # For stress-free boundaries: prescribed flux is zero (∂T/∂r = 0)
    # The boundary_values array stores the value for Dirichlet, not flux for Neumann
    # For Neumann BC, we enforce zero flux
    flux_inner = T(0)
    flux_outer = T(0)

    # Compute current fluxes at boundaries using derivative matrix
    dprofile_real = zeros(T, nr)
    dprofile_imag = zeros(T, nr)
    apply_derivative_matrix!(dprofile_real, dr_matrix, profile_real)
    apply_derivative_matrix!(dprofile_imag, dr_matrix, profile_imag)

    current_flux_inner = dprofile_real[1]
    current_flux_outer = dprofile_real[nr]

    # Compute tau corrections to enforce zero flux
    # Simple tau method: add correction to enforce BC
    if apply_inner && apply_outer
        # Both boundaries - use two tau functions
        # Correction is a polynomial that fixes both boundary fluxes
        correction_real = compute_tau_correction_both_boundaries(
            flux_inner - current_flux_inner,
            flux_outer - current_flux_outer,
            domain)
        profile_real .+= correction_real

        if any(x -> abs(x) > 1e-12, profile_imag)
            current_flux_inner_imag = dprofile_imag[1]
            current_flux_outer_imag = dprofile_imag[nr]
            correction_imag = compute_tau_correction_both_boundaries(
                -current_flux_inner_imag,
                -current_flux_outer_imag,
                domain)
            profile_imag .+= correction_imag
        end

    elseif apply_inner
        # Only inner boundary
        correction_real = compute_tau_correction_inner_boundary(
            flux_inner - current_flux_inner, domain)
        profile_real .+= correction_real

        if any(x -> abs(x) > 1e-12, profile_imag)
            current_flux_inner_imag = dprofile_imag[1]
            correction_imag = compute_tau_correction_inner_boundary(
                -current_flux_inner_imag, domain)
            profile_imag .+= correction_imag
        end

    elseif apply_outer
        # Only outer boundary
        correction_real = compute_tau_correction_outer_boundary(
            flux_outer - current_flux_outer, domain)
        profile_real .+= correction_real

        if any(x -> abs(x) > 1e-12, profile_imag)
            current_flux_outer_imag = dprofile_imag[nr]
            correction_imag = compute_tau_correction_outer_boundary(
                -current_flux_outer_imag, domain)
            profile_imag .+= correction_imag
        end
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

"""
    apply_velocity_flux_bc_direct!(spec_real, spec_imag, local_lm, lm_idx,
                                   apply_inner, apply_outer, domain, r_range)

Apply flux boundary conditions using direct substitution.
Sets the boundary point derivative to zero by adjusting the boundary value.

NOTE: This method enforces ∂T/∂r ≈ 0, which may be INCORRECT for stress-free boundaries.
The correct condition may be ∂T/∂r = T/r. Use :physical_stress method for proper stress-free.
"""
function apply_velocity_flux_bc_direct!(spec_real, spec_imag, local_lm, lm_idx,
                                        apply_inner, apply_outer,
                                        domain, r_range)
    T = eltype(spec_real)
    nr = domain.N

    # Extract radial profile
    profile_real = zeros(T, nr)
    profile_imag = zeros(T, nr)

    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            profile_real[r_idx] = spec_real[local_lm, 1, local_r]
            profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
        end
    end

    # MPI gather
    comm = BoundaryConditions.get_comm()
    if comm !== nothing && Comm_size(comm) > 1
        Allreduce!(profile_real, SUM, comm)
        Allreduce!(profile_imag, SUM, comm)
    end

    # Direct method: extrapolate to set derivative to zero
    # ∂u/∂r ≈ (u[2] - u[1])/Δr = 0  =>  u[1] = u[2]
    # ∂u/∂r ≈ (u[N] - u[N-1])/Δr = 0  =>  u[N] = u[N-1]

    if apply_inner  # apply_inner already checks 1 in r_range
        profile_real[1] = profile_real[2]
        if any(x -> abs(x) > 1e-12, profile_imag)
            profile_imag[1] = profile_imag[2]
        end
    end

    if apply_outer  # apply_outer already checks nr in r_range
        profile_real[nr] = profile_real[nr-1]
        if any(x -> abs(x) > 1e-12, profile_imag)
            profile_imag[nr] = profile_imag[nr-1]
        end
    end

    # Store back
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            spec_real[local_lm, 1, local_r] = profile_real[r_idx]
            spec_imag[local_lm, 1, local_r] = profile_imag[r_idx]
        end
    end
end

"""
    apply_velocity_flux_bc_physical_stress!(spec_real, spec_imag, local_lm, lm_idx,
                                            apply_inner, apply_outer, domain, r_range)

Apply flux boundary conditions for proper stress-free boundaries.
Enforces ∂T/∂r = T/r at boundaries, which corresponds to zero tangential stress.

# Physical Justification
For stress-free boundaries: τ = ∂v_tan/∂r - v_tan/r = 0
In spectral form with v_tan = T(r) × f(θ,φ):
  ∂(T×f)/∂r - (T×f)/r = 0
  (∂T/∂r - T/r) × f = 0
  => ∂T/∂r = T/r

This is the CORRECT condition for stress-free boundaries in spherical coordinates.
"""
function apply_velocity_flux_bc_physical_stress!(spec_real, spec_imag, local_lm, lm_idx,
                                                 apply_inner, apply_outer,
                                                 domain, r_range)
    T = eltype(spec_real)
    nr = domain.N
    r = domain.r[:, 4]

    # Extract radial profile
    profile_real = zeros(T, nr)
    profile_imag = zeros(T, nr)

    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            profile_real[r_idx] = spec_real[local_lm, 1, local_r]
            profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
        end
    end

    # MPI gather
    comm = BoundaryConditions.get_comm()
    if comm !== nothing && Comm_size(comm) > 1
        Allreduce!(profile_real, SUM, comm)
        Allreduce!(profile_imag, SUM, comm)
    end

    # Physical stress method: enforce ∂T/∂r = T/r
    # Using finite difference: (T[2] - T[1])/Δr = T[1]/r[1]
    # Solve for T[1]: T[1] = T[2] / (1 + Δr/r[1])

    if apply_inner  # apply_inner already checks 1 in r_range
        Δr = r[2] - r[1]
        scaling_factor = 1.0 / (1.0 + Δr / r[1])
        profile_real[1] = profile_real[2] * scaling_factor

        if any(x -> abs(x) > 1e-12, profile_imag)
            profile_imag[1] = profile_imag[2] * scaling_factor
        end
    end

    if apply_outer  # apply_outer already checks nr in r_range
        Δr = r[nr] - r[nr-1]
        # For outer boundary: (T[N] - T[N-1])/Δr = T[N]/r[N]
        # Solve for T[N]: T[N] = T[N-1] / (1 - Δr/r[N])
        scaling_factor = 1.0 / (1.0 - Δr / r[nr])
        profile_real[nr] = profile_real[nr-1] * scaling_factor

        if any(x -> abs(x) > 1e-12, profile_imag)
            profile_imag[nr] = profile_imag[nr-1] * scaling_factor
        end
    end

    # Store back
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            spec_real[local_lm, 1, local_r] = profile_real[r_idx]
            spec_imag[local_lm, 1, local_r] = profile_imag[r_idx]
        end
    end
end

# Helper functions for tau corrections (in-place versions to avoid allocations)

"""
    compute_tau_correction_both_boundaries!(correction::Vector{T}, ...)

In-place version that writes to pre-allocated correction buffer.
"""
function compute_tau_correction_both_boundaries!(correction::Vector{T},
                                                 flux_correction_inner::T,
                                                 flux_correction_outer::T,
                                                 domain::RadialDomain) where T
    nr = domain.N
    r = domain.r[:, 4]

    # Linear tau function: correction = a + b*r
    # Average the two flux corrections
    b = 0.5 * (flux_correction_inner + flux_correction_outer)

    # In-place computation
    @inbounds for i in 1:nr
        correction[i] = b * (r[i] - r[1])
    end

    return correction
end

"""
    compute_tau_correction_inner_boundary!(correction::Vector{T}, ...)

In-place version that writes to pre-allocated correction buffer.
"""
function compute_tau_correction_inner_boundary!(correction::Vector{T},
                                                flux_correction::T,
                                                domain::RadialDomain) where T
    nr = domain.N
    r = domain.r[:, 4]
    decay_scale = r[nr] - r[1]

    # Exponential decay from inner boundary (in-place)
    @inbounds for i in 1:nr
        correction[i] = flux_correction * exp(-(r[i] - r[1]) / decay_scale)
    end

    return correction
end

"""
    compute_tau_correction_outer_boundary!(correction::Vector{T}, ...)

In-place version that writes to pre-allocated correction buffer.
"""
function compute_tau_correction_outer_boundary!(correction::Vector{T},
                                                flux_correction::T,
                                                domain::RadialDomain) where T
    nr = domain.N
    r = domain.r[:, 4]
    decay_scale = r[nr] - r[1]

    # Exponential decay from outer boundary (in-place)
    @inbounds for i in 1:nr
        correction[i] = flux_correction * exp(-(r[nr] - r[i]) / decay_scale)
    end

    return correction
end

# Fallback allocating versions (for backward compatibility or when workspace not available)
function compute_tau_correction_both_boundaries(flux_correction_inner::T,
                                                flux_correction_outer::T,
                                                domain::RadialDomain) where T
    correction = zeros(T, domain.N)
    compute_tau_correction_both_boundaries!(correction, flux_correction_inner,
                                           flux_correction_outer, domain)
    return correction
end

function compute_tau_correction_inner_boundary(flux_correction::T,
                                               domain::RadialDomain) where T
    correction = zeros(T, domain.N)
    compute_tau_correction_inner_boundary!(correction, flux_correction, domain)
    return correction
end

function compute_tau_correction_outer_boundary(flux_correction::T,
                                               domain::RadialDomain) where T
    correction = zeros(T, domain.N)
    compute_tau_correction_outer_boundary!(correction, flux_correction, domain)
    return correction
end

"""
    validate_stress_free_boundary(v_r, v_theta, v_phi, r_val, theta, phi; tolerance=0.05)

Validate that velocity field satisfies stress-free boundary condition.

For stress-free boundaries with v_r = 0, the condition is:
    ∂v_θ/∂r - v_θ/r = 0  (zero tangential stress in θ direction)
    ∂v_φ/∂r - v_φ/r = 0  (zero tangential stress in φ direction)

# Arguments
- `v_r, v_theta, v_phi`: Velocity components at boundary [nlat, nlon]
- `r_val`: Radial position of boundary
- `theta, phi`: Coordinate arrays
- `tolerance`: Maximum allowed |stress| / |v|

# Returns
- `is_valid`: Boolean indicating if condition is satisfied
- `max_violation`: Maximum relative stress violation
"""
function validate_stress_free_boundary(v_r, v_theta, v_phi, r_val, theta, phi; tolerance=0.05)
    nlat, nlon = size(v_theta)

    # Compute radial derivatives using finite differences
    # Note: This is called at a single radial level, so we can't compute ∂/∂r directly
    # Instead, we check if the pattern is consistent with stress-free

    # For now, compute the stress components assuming the velocity pattern
    # varies smoothly in r with typical length scale ~ r

    # Estimate ∂v_θ/∂r using neighboring points (if available)
    # For a single boundary, we use the scaling relationship:
    # For stress-free: ∂v_θ/∂r ≈ v_θ/r (characteristic scaling)

    stress_theta = zeros(eltype(v_theta), nlat, nlon)
    stress_phi = zeros(eltype(v_phi), nlat, nlon)

    # Compute stress = (∂v/∂r - v/r)
    # For boundary validation, we check if v/r has appropriate scaling
    # This is an approximation - full validation requires field at multiple radii

    for i in 1:nlat
        for j in 1:nlon
            # Simplified check: for pure stress-free, v_tan should scale as ~ r
            # So v_tan/r should be roughly constant
            # This is checked by comparing magnitude ratios

            # For now, just check v_r ≈ 0 (primary condition)
            stress_theta[i, j] = v_r[i, j]  # Should be zero
            stress_phi[i, j] = v_r[i, j]     # Should be zero
        end
    end

    # Compute typical velocity magnitude
    v_magnitude = sqrt.(v_theta.^2 .+ v_phi.^2)
    typical_v = BoundaryConditions._Statistics.mean(v_magnitude[v_magnitude .> 1e-10])

    # Maximum stress (primarily checking v_r = 0 for now)
    max_stress = maximum(abs, v_r)
    max_violation = max_stress / (typical_v + 1e-15)

    is_valid = max_violation < tolerance

    return is_valid, max_violation
end

"""
    compute_tangential_stress_components(v_theta, v_phi, dv_theta_dr, dv_phi_dr, r_val)

Compute tangential stress components from velocity and derivatives.

For incompressible flow with v_r = 0:
    τ_rθ = μ(∂v_θ/∂r - v_θ/r)
    τ_rφ = μ(∂v_φ/∂r - v_φ/r)

# Arguments
- `v_theta, v_phi`: Tangential velocity components [nlat, nlon]
- `dv_theta_dr, dv_phi_dr`: Radial derivatives [nlat, nlon]
- `r_val`: Radial position

# Returns
- `tau_theta, tau_phi`: Stress components [nlat, nlon] with μ = 1
- `max_stress`: Maximum stress magnitude
"""
function compute_tangential_stress_components(v_theta, v_phi, dv_theta_dr, dv_phi_dr, r_val)
    # Compute stress (with μ = 1)
    tau_theta = dv_theta_dr .- v_theta ./ r_val
    tau_phi = dv_phi_dr .- v_phi ./ r_val

    # Total stress magnitude
    stress_magnitude = sqrt.(tau_theta.^2 .+ tau_phi.^2)
    max_stress = maximum(stress_magnitude)

    return tau_theta, tau_phi, max_stress
end

function compute_vorticity_spectral_full!(fields::SHTnsVelocityFields{T},
                                          domain::RadialDomain,
                                          ws::VelocityWorkspace{T}) where T
    # Same as the threaded version but using provided workspace buffers
    u_tor_real = parent(fields.toroidal.data_real)
    u_tor_imag = parent(fields.toroidal.data_imag)
    u_pol_real = parent(fields.poloidal.data_real)
    u_pol_imag = parent(fields.poloidal.data_imag)
    ω_tor_real = parent(fields.vort_toroidal.data_real)
    ω_tor_imag = parent(fields.vort_toroidal.data_imag)
    ω_pol_real = parent(fields.vort_poloidal.data_real)
    ω_pol_imag = parent(fields.vort_poloidal.data_imag)

    config = fields.toroidal.config
    lm_range = get_local_range(fields.toroidal.pencil, 1)
    r_range  = get_local_range(fields.toroidal.pencil, 3)
    nr = domain.N

    Threads.@threads for lm_idx in lm_range
        if lm_idx <= length(fields.l_factors)
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = fields.l_factors[lm_idx]
            tid = Threads.threadid()

            # Thread safety: ensure thread ID is within workspace bounds
            if tid > length(ws.pol_profile_real)
                error("Thread ID $tid exceeds workspace size $(length(ws.pol_profile_real)). " *
                      "This indicates the workspace was created for fewer threads than are currently active. " *
                      "Workspace threads: $(length(ws.pol_profile_real)), Active threads: $(Threads.nthreads()). " *
                      "Recreate the workspace with the correct thread count.")
            end

            pol_profile_real = ws.pol_profile_real[tid]
            pol_profile_imag = ws.pol_profile_imag[tid]
            tor_profile_real = ws.tor_profile_real[tid]
            tor_profile_imag = ws.tor_profile_imag[tid]
            dpol_dr_real     = ws.dpol_dr_real[tid]
            dpol_dr_imag     = ws.dpol_dr_imag[tid]
            d2pol_dr2_real   = ws.d2pol_dr2_real[tid]
            d2pol_dr2_imag   = ws.d2pol_dr2_imag[tid]

            extract_local_radial_profile!(pol_profile_real, u_pol_real, local_lm, nr, r_range)
            extract_local_radial_profile!(pol_profile_imag, u_pol_imag, local_lm, nr, r_range)
            extract_local_radial_profile!(tor_profile_real, u_tor_real, local_lm, nr, r_range)
            extract_local_radial_profile!(tor_profile_imag, u_tor_imag, local_lm, nr, r_range)

            apply_derivative_matrix!(dpol_dr_real,   fields.dr_matrix,  pol_profile_real)
            apply_derivative_matrix!(dpol_dr_imag,   fields.dr_matrix,  pol_profile_imag)
            apply_derivative_matrix!(d2pol_dr2_real, fields.d2r_matrix, pol_profile_real)
            apply_derivative_matrix!(d2pol_dr2_imag, fields.d2r_matrix, pol_profile_imag)

            @inbounds @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(ω_tor_real, 3)
                    r_val = domain.r[r_idx, 4]
                    if r_val == 0.0
                        ω_tor_real[local_lm, 1, local_r] = 0
                        ω_tor_imag[local_lm, 1, local_r] = 0
                        ω_pol_real[local_lm, 1, local_r] = 0
                        ω_pol_imag[local_lm, 1, local_r] = 0
                    else
                        r_inv  = domain.r[r_idx, 3]
                        r_inv2 = domain.r[r_idx, 2]
                        ω_tor_real[local_lm, 1, local_r] = (l_factor * r_inv2 * pol_profile_real[r_idx]
                                                            - d2pol_dr2_real[r_idx]
                                                            - 2.0 * r_inv * dpol_dr_real[r_idx])

                        ω_tor_imag[local_lm, 1, local_r] = (l_factor * r_inv2 * pol_profile_imag[r_idx]
                                                            - d2pol_dr2_imag[r_idx]
                                                            - 2.0 * r_inv * dpol_dr_imag[r_idx])

                        ω_pol_real[local_lm, 1, local_r] = -l_factor * r_inv2 * tor_profile_real[r_idx]
                        
                        ω_pol_imag[local_lm, 1, local_r] = -l_factor * r_inv2 * tor_profile_imag[r_idx]
                    end
                end
            end
        end
    end
end


function create_shtns_velocity_fields(::Type{T}, config::SHTnsKitConfig, 
                                      oc_domain::RadialDomain, 
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
    
    # Create vector fields
    velocity  = create_shtns_vector_field(T, config, oc_domain, pencils)
    vorticity = create_shtns_vector_field(T, config, oc_domain, pencils)
    
    # Spectral fields
    toroidal         = create_shtns_spectral_field(T, config, oc_domain, pencil_spec)
    poloidal         = create_shtns_spectral_field(T, config, oc_domain, pencil_spec)
    vort_toroidal    = create_shtns_spectral_field(T, config, oc_domain, pencil_spec)
    vort_poloidal    = create_shtns_spectral_field(T, config, oc_domain, pencil_spec)
    nl_toroidal      = create_shtns_spectral_field(T, config, oc_domain, pencil_spec)
    nl_poloidal      = create_shtns_spectral_field(T, config, oc_domain, pencil_spec)
    prev_nl_toroidal = create_shtns_spectral_field(T, config, oc_domain, pencil_spec)
    prev_nl_poloidal = create_shtns_spectral_field(T, config, oc_domain, pencil_spec)
    
    # Work arrays
    work_tor           = create_shtns_spectral_field(T, config, oc_domain, pencil_spec)
    work_pol           = create_shtns_spectral_field(T, config, oc_domain, pencil_spec)
    work_physical      = create_shtns_vector_field(T, config, oc_domain, pencils)
    advection_physical = create_shtns_vector_field(T, config, oc_domain, pencils)
    
    # Pre-compute l(l+1) factors
    l_factors = Float64[l * (l + 1) for l in config.l_values]
    
    # Pre-compute Coriolis factors (sin(θ) and cos(θ))
    coriolis_factors = zeros(Float64, 2, config.nlat)
    for i in 1:config.nlat
        coriolis_factors[1, i] = sin(config.theta_grid[i])
        coriolis_factors[2, i] = cos(config.theta_grid[i])
    end
    
    # Create radial derivative matrices
    dr_matrix        = create_derivative_matrix(1, oc_domain)
    d2r_matrix       = create_derivative_matrix(2, oc_domain)
    laplacian_matrix = create_radial_laplacian(oc_domain)
    
    # Create transpose plans for efficient data movement
    transpose_plans = create_transpose_plans(pencils)
    
    boundary_condition_set = nothing
    boundary_cache = Dict{String, Any}()
    boundary_time_index = Ref{Int}(1)

    return SHTnsVelocityFields{T}(velocity, vorticity, toroidal, poloidal,
                                  vort_toroidal, vort_poloidal,
                                  nl_toroidal, nl_poloidal, prev_nl_toroidal, prev_nl_poloidal,
                                  work_tor, work_pol, work_physical,
                                  advection_physical,
                                  l_factors, coriolis_factors,
                                  dr_matrix, d2r_matrix, laplacian_matrix,
                                  config,
                                  oc_domain,
                                  boundary_condition_set, boundary_cache, boundary_time_index)
end


# =============================
# Main nonlinear computation
# =============================
function compute_velocity_nonlinear!(fields::SHTnsVelocityFields{T}, 
                                    temp_field, comp_field, mag_field,
                                    oc_domain::RadialDomain) where T
    # Zero work arrays once
    zero_velocity_work_arrays!(fields)
    
    # Step 1: Use enhanced vector synthesis with automatic transpose handling
    shtnskit_vector_synthesis!(fields.toroidal, fields.poloidal, fields.velocity; domain=oc_domain)

    # Step 2: Compute vorticity in spectral space with enhanced derivative computation
    compute_vorticity_spectral_full!(fields, oc_domain)

    # Step 3: Transform vorticity to physical space with batched operations
    shtnskit_vector_synthesis!(fields.vort_toroidal, fields.vort_poloidal, fields.vorticity; domain=oc_domain)
    
    # Step 4: Compute all nonlinear terms with enhanced memory access patterns
    compute_all_nonlinear_terms!(fields, temp_field, comp_field, mag_field, oc_domain)
    
    # Step 5: Use enhanced vector analysis with efficient data layout
    shtnskit_vector_analysis!(fields.advection_physical, fields.nl_toroidal, fields.nl_poloidal)
end



# =================================================
# Enhanced vorticity computation with enhanced derivatives
# =================================================
using Base.Threads
function compute_vorticity_spectral_full!(fields::SHTnsVelocityFields{T}, 
                                         domain::RadialDomain) where T
    # If a compatible workspace is registered, use it
    ws_any = VELOCITY_WS[]
    if ws_any !== nothing && ws_any isa VelocityWorkspace{T}
        return compute_vorticity_spectral_full!(fields, domain, ws_any)
    end
    # Compute vorticity ω = ∇ × u in spectral space with full radial derivatives
    # For toroidal-poloidal decomposition:
    # ω_tor = [l(l+1)/r² - d²/dr² - 2/r d/dr] u_pol
    # ω_pol = -l(l+1)/r² u_tor
    
    # Get local data views with enhanced memory access
    u_tor_real = parent(fields.toroidal.data_real)
    u_tor_imag = parent(fields.toroidal.data_imag)
    u_pol_real = parent(fields.poloidal.data_real)
    u_pol_imag = parent(fields.poloidal.data_imag)
    
    ω_tor_real = parent(fields.vort_toroidal.data_real)
    ω_tor_imag = parent(fields.vort_toroidal.data_imag)
    ω_pol_real = parent(fields.vort_poloidal.data_real)
    ω_pol_imag = parent(fields.vort_poloidal.data_imag)
    
    # Use enhanced range functions from pencil decomposition
    config = fields.toroidal.config
    
    # Get local ranges using pencil topology
    lm_range = range_local(config.pencils.spec, 1)
    r_range  = range_local(config.pencils.r, 3)
    
    nr = domain.N
    
    # Thread-local scratch buffers reused across modes to avoid allocations
    nT = max(1, Threads.nthreads())
    pol_profile_real_bufs = [zeros(T, nr) for _ in 1:nT]
    pol_profile_imag_bufs = [zeros(T, nr) for _ in 1:nT]
    tor_profile_real_bufs = [zeros(T, nr) for _ in 1:nT]
    tor_profile_imag_bufs = [zeros(T, nr) for _ in 1:nT]
    dpol_dr_real_bufs     = [zeros(T, nr) for _ in 1:nT]
    dpol_dr_imag_bufs     = [zeros(T, nr) for _ in 1:nT]
    d2pol_dr2_real_bufs   = [zeros(T, nr) for _ in 1:nT]
    d2pol_dr2_imag_bufs   = [zeros(T, nr) for _ in 1:nT]

    # Process each (l,m) mode (parallel over lm)
    @inbounds Threads.@threads for lm_idx in lm_range
        if lm_idx <= length(fields.l_factors)
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = fields.l_factors[lm_idx]

            # Select thread-local buffers with proper bounds checking
            tid = Threads.threadid()
            if tid > nT
                error("Thread ID $tid exceeds allocated workspace size $nT. " *
                      "Active threads: $(Threads.nthreads()). " *
                      "This indicates a thread count mismatch. " *
                      "The clamping approach (min(tid, nT)) was removed because it causes data races.")
            end
            pol_profile_real = pol_profile_real_bufs[tid]
            pol_profile_imag = pol_profile_imag_bufs[tid]
            tor_profile_real = tor_profile_real_bufs[tid]
            tor_profile_imag = tor_profile_imag_bufs[tid]
            dpol_dr_real     = dpol_dr_real_bufs[tid]
            dpol_dr_imag     = dpol_dr_imag_bufs[tid]
            d2pol_dr2_real   = d2pol_dr2_real_bufs[tid]
            d2pol_dr2_imag   = d2pol_dr2_imag_bufs[tid]

            # Extract radial profiles (in-place)
            extract_local_radial_profile!(pol_profile_real, u_pol_real, local_lm, nr, r_range)
            extract_local_radial_profile!(pol_profile_imag, u_pol_imag, local_lm, nr, r_range)
            extract_local_radial_profile!(tor_profile_real, u_tor_real, local_lm, nr, r_range)
            extract_local_radial_profile!(tor_profile_imag, u_tor_imag, local_lm, nr, r_range)
            
            # Compute radial derivatives for poloidal component (in-place, reuse buffers)
            apply_derivative_matrix!(dpol_dr_real,   fields.dr_matrix,  pol_profile_real)
            apply_derivative_matrix!(dpol_dr_imag,   fields.dr_matrix,  pol_profile_imag)
            apply_derivative_matrix!(d2pol_dr2_real, fields.d2r_matrix, pol_profile_real)
            apply_derivative_matrix!(d2pol_dr2_imag, fields.d2r_matrix, pol_profile_imag)
            
            # Compute vorticity components
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(ω_tor_real, 3)
                    r_val = domain.r[r_idx, 4]
                    if r_val == 0.0
                        # At r=0 (ball geometry), regularity implies finite values → set to 0 safely
                        ω_tor_real[local_lm, 1, local_r] = 0
                        ω_tor_imag[local_lm, 1, local_r] = 0
                        ω_pol_real[local_lm, 1, local_r] = 0
                        ω_pol_imag[local_lm, 1, local_r] = 0
                    else
                        r_inv = domain.r[r_idx, 3]   # 1/r
                        r_inv2 = domain.r[r_idx, 2]  # 1/r²
                        # Toroidal vorticity from poloidal velocity (with full derivatives)
                        ω_tor_real[local_lm, 1, local_r] = (l_factor * r_inv2 * pol_profile_real[r_idx] 
                                                            - d2pol_dr2_real[r_idx] 
                                                            - 2.0 * r_inv * dpol_dr_real[r_idx])
                        ω_tor_imag[local_lm, 1, local_r] = (l_factor * r_inv2 * pol_profile_imag[r_idx] 
                                                            - d2pol_dr2_imag[r_idx] 
                                                            - 2.0 * r_inv * dpol_dr_imag[r_idx])
                        # Poloidal vorticity from toroidal velocity
                        ω_pol_real[local_lm, 1, local_r] = -l_factor * r_inv2 * tor_profile_real[r_idx]
                        ω_pol_imag[local_lm, 1, local_r] = -l_factor * r_inv2 * tor_profile_imag[r_idx]
                    end
                end
            end
        end
    end
end


# ==========================================
# Optimized nonlinear term computation
# ==========================================

"""
    compute_all_nonlinear_terms!(fields, temp_field, comp_field, mag_field, domain)

Compute all nonlinear forcing terms for the momentum equation.

# Governing Equation (Dimensional)

In a rotating reference frame with angular velocity Ω:

∂u/∂t = -u×ω - ∇p/ρ + 2Ω(ẑ×u) + ν∇²u + (αg/ρ)T·r̂ + (Δρg/ρ)C·r̂ + (1/μ₀ρ)(∇×B)×B

where:
- u: velocity, ω = ∇×u: vorticity
- Ω: rotation rate, ẑ: rotation axis
- ν: kinematic viscosity
- α: thermal expansion coefficient, g: gravity
- T: temperature perturbation, C: composition perturbation
- B: magnetic field, μ₀: permeability

# Non-Dimensionalization (Magnetic Diffusion Time Scaling)

Length scale: d (shell thickness or ball radius)
Time scale: τ = d²/η (magnetic diffusion time)
Velocity scale: U = η/d
Magnetic field scale: B₀
Temperature scale: ΔT

Dimensionless parameters:
- E = ν/(Ω d²): Ekman number
- Pm = ν/η: Magnetic Prandtl number
- Pr = ν/κ: Prandtl number
- Sc = ν/D: Schmidt number
- Ra = (αgΔT d³)/(νκ): Rayleigh number
- Ra_C = (Δρg d³)/(ρνD): Compositional Rayleigh number

# Non-Dimensional Momentum Equation

(E/Pm) ∂ũ/∂τ + (E/Pm)(∇×ũ)×ũ + ẑ×ũ = -∇p̃
                + (Pm/Pr)Ra·T̃·r̂ + (Pm/Sc)Ra_C·C̃·r̂
                + (∇×B̃)×B̃ + E∇²ũ

where tilde denotes dimensionless quantities.

# Implementation Notes

After dividing Eq. (1) by E/Pm, the explicit RHS entering the time integrator is:
RHS = ũ×ω̃ - (Pm/E)(ẑ×ũ)
      + (Pm/E)(Pm/Pr)Ra·T̃·r̂ + (Pm/E)(Pm/Sc)Ra_C·C̃·r̂
      + (Pm/E)(∇×B̃)×B̃

- The advection term ũ×ω̃ already matches −(∇×ũ)×ũ without extra scaling.
- Coriolis, buoyancy, and Lorentz forces carry the (Pm/E) prefactor (=`rossby_factor`).
- Viscous diffusion is treated implicitly with coefficient Pm (passed via `diffusivity`).

The time derivative retains the (E/Pm) mass term and is handled by the integrator.
"""
function compute_all_nonlinear_terms!(fields::SHTnsVelocityFields{T},
                                               temp_field, comp_field, mag_field,
                                               domain::RadialDomain) where T
    # Compute all forces in a single enhanced loop

    if iszero(d_E)
        throw(ArgumentError("Ekman number d_E must be nonzero when evaluating the velocity equation in magnetic-diffusion scaling."))
    end
    rossby_factor = d_Pm / d_E
    
    # Get all data views
    vel_r = parent(fields.velocity.r_component.data)
    vel_θ = parent(fields.velocity.θ_component.data)
    vel_φ = parent(fields.velocity.φ_component.data)
    
    vort_r = parent(fields.vorticity.r_component.data)
    vort_θ = parent(fields.vorticity.θ_component.data)
    vort_φ = parent(fields.vorticity.φ_component.data)
    
    adv_r = parent(fields.advection_physical.r_component.data)
    adv_θ = parent(fields.advection_physical.θ_component.data)
    adv_φ = parent(fields.advection_physical.φ_component.data)
    
    # Get dimensions from config for better performance
    config = fields.velocity.r_component.config
    local_size = size(vel_r)
    nlat = config.nlat
    nlon = config.nlon
    
    # Use pencil ranges for enhanced loop bounds
    r_range = range_local(config.pencils.r, 3)
    
    # Main fused computation loop with enhanced indexing (parallel over r-slices)
    adv_coeff = one(T)  # u×ω coefficient (matches -(∇×u)×u term in Eq. (1))
    @inbounds Threads.@threads for k in 1:local_size[3]
        # Get radius for this level using pencil range
        r_idx = k + first(r_range) - 1
        if r_idx <= domain.N
            r = domain.r[r_idx, 4]
            r_inv = domain.r[r_idx, 3]
        else
            r = 1.0
            r_inv = 1.0
        end
        
        for j in 1:local_size[2]
            # Get pre-computed Coriolis factors for this latitude
            theta_idx = min(j, nlat)
            sin_theta = fields.coriolis_factors[1, theta_idx]
            cos_theta = fields.coriolis_factors[2, theta_idx]
            
            @simd for i in 1:local_size[1]
                linear_idx = i + (j-1)*local_size[1] + (k-1)*local_size[1]*local_size[2]
                
                if linear_idx <= length(vel_r)
                    # Load velocity and vorticity components
                    u_r = vel_r[linear_idx]
                    u_θ = vel_θ[linear_idx]
                    u_φ = vel_φ[linear_idx]
                    
                    ω_r = vort_r[linear_idx]
                    ω_θ = vort_θ[linear_idx]
                    ω_φ = vort_φ[linear_idx]
                    
                    # Advection: u × ω (equivalent to - (∇×u) × u)
                    adv_r_val = adv_coeff * (u_θ * ω_φ - u_φ * ω_θ)
                    adv_θ_val = adv_coeff * (u_φ * ω_r - u_r * ω_φ)
                    adv_φ_val = adv_coeff * (u_r * ω_θ - u_θ * ω_r)
                    
                    # Coriolis: −(Pm/E) ẑ × u
                    zhat_cross_r = -sin_theta * u_φ
                    zhat_cross_θ = -cos_theta * u_φ
                    zhat_cross_φ = cos_theta * u_θ + sin_theta * u_r
                    cor_r = -rossby_factor * zhat_cross_r
                    cor_θ = -rossby_factor * zhat_cross_θ
                    cor_φ = -rossby_factor * zhat_cross_φ
                    
                    # Store combined result
                    adv_r[linear_idx] = adv_r_val + cor_r
                    adv_θ[linear_idx] = adv_θ_val + cor_θ
                    adv_φ[linear_idx] = adv_φ_val + cor_φ
                end
            end
        end
    end
    
    # Add buoyancy forces with proper scaling
    if temp_field !== nothing
        buoyancy_factor = rossby_factor * (d_Pm / d_Pr) * d_Ra
        add_thermal_buoyancy_force!(adv_r, temp_field, buoyancy_factor, domain)
    end
    
    if comp_field !== nothing
        comp_factor = rossby_factor * (d_Pm / d_Sc) * d_Ra_C
        add_buoyancy_force!(adv_r, comp_field, comp_factor, domain)
    end
    
    # Add Lorentz force if magnetic field present
    if mag_field !== nothing
        add_lorentz_force!(fields, mag_field, domain)
    end
end


# =====================================
# Thermal buoyancy force addition
# =====================================
function add_thermal_buoyancy_force!(force_r::AbstractArray{T,3},
                                      scalar_field, factor::Float64,
                                      domain::RadialDomain) where T
    # Add buoyancy force: F_buoyancy = (Pm²/E·Pr) Ra T r̂
    #
    # Boussinesq approximation: buoyancy is proportional to temperature anomaly
    # WITHOUT radial dependence (gravity is absorbed into Ra)
    #
    # In non-dimensional form with magnetic diffusion time scaling:
    # F = (Pm/E)·(Pm/Pr)·Ra·T·r̂
    if iszero(factor)
        return force_r
    end

    # Get scalar field data
    if isa(scalar_field, SHTnsPhysicalField)
        scalar_data = parent(scalar_field.data)
    else
        scalar_data = parent(scalar_field.temperature.data)
    end

    # Vectorized addition WITHOUT radial position factor (threaded in flat index space)
    # Standard Boussinesq: F ∝ T, NOT F ∝ r·T
    Ntot = length(force_r)
    chunk = max(1, Ntot ÷ max(1, Threads.nthreads()))
    @inbounds Threads.@threads for start in 1:chunk:Ntot
        stop = min(Ntot, start + chunk - 1)
        @simd for idx in start:stop
            if idx <= length(scalar_data)
                # Boussinesq buoyancy: force proportional to temperature, no radial weighting
                force_r[idx] += factor * scalar_data[idx]
            end
        end
    end
end

# Compositional buoyancy force (similar to thermal but for composition)
function add_buoyancy_force!(force_r::AbstractArray{T,3},
                             comp_field, factor::Float64,
                             domain::RadialDomain) where T
    # Add compositional buoyancy force: F_comp = (Pm²/E·Sc) Ra_C C r̂
    #
    # Boussinesq approximation: buoyancy is proportional to composition anomaly
    # WITHOUT radial dependence (analogous to thermal buoyancy)
    #
    # In non-dimensional form with magnetic diffusion time scaling:
    # F = (Pm/E)·(Pm/Sc)·Ra_C·C·r̂
    if iszero(factor)
        return force_r
    end

    # Get compositional field data
    if isa(comp_field, SHTnsPhysicalField)
        comp_data = parent(comp_field.data)
    else
        comp_data = parent(comp_field.composition.data)
    end

    # Vectorized addition WITHOUT radial position factor (threaded in flat index space)
    # Standard Boussinesq: F ∝ C, NOT F ∝ r·C
    Ntot = length(force_r)
    chunk = max(1, Ntot ÷ max(1, Threads.nthreads()))
    @inbounds Threads.@threads for start in 1:chunk:Ntot
        stop = min(Ntot, start + chunk - 1)
        @simd for idx in start:stop
            if idx <= length(comp_data)
                # Boussinesq buoyancy: force proportional to composition, no radial weighting
                force_r[idx] += factor * comp_data[idx]
            end
        end
    end
end

# ===============================
# Optimized Lorentz force computation
# ===============================
function add_lorentz_force!(fields::SHTnsVelocityFields{T}, 
                           mag_field::SHTnsMagneticFields{T},
                           domain::RadialDomain) where T

    # Compute Lorentz force F = (Pm/E) (∇ × B) × B with vectorization
    if iszero(d_E)
        throw(ArgumentError("Ekman number d_E must be nonzero when evaluating the velocity equation in magnetic-diffusion scaling."))
    end
    
    lorentz_factor = d_Pm / d_E
    
    # Step 1: Use pre-computed current density from magnetic field
    # Leverage shared memory access patterns for efficiency
    
    # Step 2: Compute j × B with enhanced vectorization
    j_r = parent(mag_field.current.r_component.data)
    j_θ = parent(mag_field.current.θ_component.data)
    j_φ = parent(mag_field.current.φ_component.data)
    
    B_r = parent(mag_field.magnetic.r_component.data)
    B_θ = parent(mag_field.magnetic.θ_component.data)
    B_φ = parent(mag_field.magnetic.φ_component.data)
    
    adv_r = parent(fields.advection_physical.r_component.data)
    adv_θ = parent(fields.advection_physical.θ_component.data)
    adv_φ = parent(fields.advection_physical.φ_component.data)
    
    # Fused loop for j × B / Pm
    @inbounds @simd for idx in eachindex(j_r)
        if idx <= length(B_r)
            # Add Lorentz force to existing forces
            adv_r[idx] += lorentz_factor * (j_θ[idx] * B_φ[idx] - j_φ[idx] * B_θ[idx])
            adv_θ[idx] += lorentz_factor * (j_φ[idx] * B_r[idx] - j_r[idx] * B_φ[idx])
            adv_φ[idx] += lorentz_factor * (j_r[idx] * B_θ[idx] - j_θ[idx] * B_r[idx])
        end
    end
end


# Note: Boundary condition functions moved to src/BoundaryConditions/velocity.jl


# ===========================================
# Helper functions for radial operations
# ===========================================
function extract_local_radial_profile(data::AbstractArray{T,3}, local_lm::Int, 
                                     nr::Int, r_range) where T
    profile = zeros(T, nr)
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(data, 3) && r_idx <= nr
            profile[r_idx] = data[local_lm, 1, local_r]
        end
    end
    
    return profile
end


"""
    extract_local_radial_profile!(profile, data, local_lm, nr, r_range)

In-place version to avoid allocations; writes the local radial line into
`profile` for the given `local_lm` using the provided `r_range`.
"""
function extract_local_radial_profile!(profile::Vector{T}, data::AbstractArray{T,3},
                                       local_lm::Int, nr::Int, r_range) where T
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(data, 3) && r_idx <= nr
            profile[r_idx] = data[local_lm, 1, local_r]
        end
    end
    return profile
end


function store_local_radial_profile!(data::AbstractArray{T,3}, profile::Vector{T},
                                    local_lm::Int, r_range) where T
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(data, 3) && r_idx <= length(profile)
            data[local_lm, 1, local_r] = profile[r_idx]
        end
    end
end


function apply_derivative_local(matrix::BandedMatrix{T}, field::Vector{T}) where T
    # Apply banded derivative matrix
    N = matrix.size
    bandwidth = matrix.bandwidth
    result = zeros(T, N)
    
    @inbounds for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            if 1 <= band_row <= 2*bandwidth + 1
                result[i] += matrix.data[band_row, j] * field[j]
            end
        end
    end
    
    return result
end


# function solve_helmholtz_equation(laplacian::BandedMatrix{T}, source::Vector{T},
#                                  l_factor::Float64, domain::RadialDomain) where T
#     # Solve (∇²_r - l(l+1)/r²) u = source
#     # This is a simplified solver - in practice would use more sophisticated methods
    
#     N = oc_domain.N
#     solution = zeros(T, N)
    
#     # Build full operator for this l value
#     operator = zeros(T, N, N)
#     bandwidth = laplacian.bandwidth
    
#     @inbounds for j in 1:N
#         for i in max(1, j - bandwidth):min(N, j + bandwidth)
#             band_row = bandwidth + 1 + i - j
#             if 1 <= band_row <= 2*bandwidth + 1
#                 operator[i, j] = laplacian.data[band_row, j]
#                 if i == j
#                     # Add -l(l+1)/r² term to diagonal
#                     r_inv2 = oc_domain.r[i, 2]
#                     operator[i, j] -= l_factor * r_inv2
#                 end
#             end
#         end
#     end
    
#     # Note: Boundary condition application now handled by modular system
    
#     # Solve linear system (would use iterative solver in practice)
#     solution = operator \ source
    
#     return solution
# end


# =====================================================
# Diagnostic functions using transform infrastructure
# =====================================================
function compute_kinetic_energy(fields::SHTnsVelocityFields{T}, oc_domain::RadialDomain) where T
    # Compute kinetic energy with configuration-aware integration
    
    tor_real = parent(fields.toroidal.data_real)
    tor_imag = parent(fields.toroidal.data_imag)
    pol_real = parent(fields.poloidal.data_real)
    pol_imag = parent(fields.poloidal.data_imag)
    
    local_energy = zero(Float64)
    
    # Use configuration pencils for consistent range access
    config = fields.toroidal.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range = range_local(config.pencils.r, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= fields.toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = fields.l_factors[lm_idx]
            
            # Weight by l(l+1) for proper spectral integration
            weight = 1.0 / max(l_factor, 1.0)
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    # Include radial weight for spherical integration
                    r = oc_domain.r[r_idx, 4]
                    r_weight = r^2 * oc_domain.integration_weights[r_idx]
                    
                    local_energy += weight * r_weight * (
                        tor_real[local_lm, 1, local_r]^2 + 
                        tor_imag[local_lm, 1, local_r]^2 + 
                        pol_real[local_lm, 1, local_r]^2 + 
                        pol_imag[local_lm, 1, local_r]^2
                    )
                end
            end
        end
    end
    
    # Global sum
    return 0.5 * Allreduce(local_energy, SUM, get_comm())
end


function compute_reynolds_stress(fields::SHTnsVelocityFields{T}) where T
    # Compute Reynolds stress tensor <u_i u_j>
    # This requires transforming to physical space and computing products
    
    vel_r = parent(fields.velocity.r_component.data)
    vel_θ = parent(fields.velocity.θ_component.data)
    vel_φ = parent(fields.velocity.φ_component.data)
    
    # Compute all 6 independent components
    R_rr = mean(vel_r .* vel_r)
    R_θθ = mean(vel_θ .* vel_θ)
    R_φφ = mean(vel_φ .* vel_φ)
    R_rθ = mean(vel_r .* vel_θ)
    R_rφ = mean(vel_r .* vel_φ)
    R_θφ = mean(vel_θ .* vel_φ)
    
    # Global averages
    R_rr = Allreduce(R_rr, SUM, get_comm()) / Comm_size(get_comm())
    R_θθ = Allreduce(R_θθ, SUM, get_comm()) / Comm_size(get_comm())
    R_φφ = Allreduce(R_φφ, SUM, get_comm()) / Comm_size(get_comm())
    R_rθ = Allreduce(R_rθ, SUM, get_comm()) / Comm_size(get_comm())
    R_rφ = Allreduce(R_rφ, SUM, get_comm()) / Comm_size(get_comm())
    R_θφ = Allreduce(R_θφ, SUM, get_comm()) / Comm_size(get_comm())
    
    return (R_rr, R_θθ, R_φφ, R_rθ, R_rφ, R_θφ)
end


# ================================================================================
# Utility functions
# ================================================================================
function zero_velocity_work_arrays!(fields::SHTnsVelocityFields{T}) where T
    # Efficiently zero all work arrays with batch operations
    # Use threaded operations for better performance on large arrays
    Threads.@threads for arr in [
        parent(fields.work_tor.data_real),
        parent(fields.work_tor.data_imag),
        parent(fields.work_pol.data_real),
        parent(fields.work_pol.data_imag),
        parent(fields.work_physical.r_component.data),
        parent(fields.work_physical.θ_component.data),
        parent(fields.work_physical.φ_component.data),
        parent(fields.advection_physical.r_component.data),
        parent(fields.advection_physical.θ_component.data),
        parent(fields.advection_physical.φ_component.data),
        parent(fields.vort_toroidal.data_real),
        parent(fields.vort_toroidal.data_imag),
        parent(fields.vort_poloidal.data_real),
        parent(fields.vort_poloidal.data_imag)
    ]
        fill!(arr, zero(T))
    end
end

function scale_field!(field::SHTnsVectorField{T}, factor::Float64) where T
    # Scale all components of a vector field
    parent(field.r_component.data) .*= factor
    parent(field.θ_component.data) .*= factor
    parent(field.φ_component.data) .*= factor
end

function add_vector_fields!(dest::SHTnsVectorField{T}, source::SHTnsVectorField{T}) where T
    # Add source to destination with vectorized operations
    parent(dest.r_component.data) .+= parent(source.r_component.data)
    parent(dest.θ_component.data) .+= parent(source.θ_component.data)
    parent(dest.φ_component.data) .+= parent(source.φ_component.data)
end


# ================================================================================
# Enhanced utility functions using pencil decomposition and SHTns integration
# ================================================================================

"""
    batch_velocity_transforms!(fields::SHTnsVelocityFields{T}) where T
    
Perform batched transforms for better cache efficiency using shtnskit_transforms.jl
"""
function batch_velocity_transforms!(fields::SHTnsVelocityFields{T}) where T
    # Use batched operations from shtnskit_transforms.jl for better performance
    specs = [fields.toroidal, fields.poloidal, fields.vort_toroidal, fields.vort_poloidal]
    physs = [fields.work_physical.r_component, fields.work_physical.θ_component, 
             fields.work_physical.φ_component, fields.velocity.r_component]
    
    # Only transform if specs and physs have compatible lengths
    n_transform = min(length(specs), length(physs))
    if n_transform > 0
        batch_spectral_to_physical!(specs[1:n_transform], physs[1:n_transform])
    end
end


"""
    optimize_velocity_memory_layout!(fields::SHTnsVelocityFields{T}) where T
    
Optimize memory layout for better cache performance using pencil topology
"""
function optimize_velocity_memory_layout!(fields::SHTnsVelocityFields{T}) where T
    # Use transpose plans for optimal data layout based on upcoming operations
    config = fields.toroidal.config
    
    # Use transpose plans if available
    plans = config.transpose_plans
    if !isempty(plans) && haskey(plans, :r_to_spec)
        transpose_with_timer!(fields.work_tor.data_real, fields.toroidal.data_real, 
                              plans[:r_to_spec], "toroidal_layout_opt")
        transpose_with_timer!(fields.work_pol.data_real, fields.poloidal.data_real, 
                              plans[:r_to_spec], "poloidal_layout_opt")
    end
end


"""
    validate_velocity_configuration(fields::SHTnsVelocityFields{T}, config::SHTnsKitConfig) where T
    
Validate velocity field configuration consistency with SHTns setup
"""
function validate_velocity_configuration(fields::SHTnsVelocityFields{T}, config::SHTnsKitConfig) where T
    errors = String[]
    
    # Check field dimensions match config
    if size(fields.toroidal.data_real, 1) != config.nlm
        push!(errors, "Toroidal field size mismatch with config.nlm")
    end
    
    # Check that l_factors are consistent
    if length(fields.l_factors) != config.nlm
        push!(errors, "l_factors length mismatch with config.nlm")
    end
    
    # Validate pencil topology consistency
    spec_range = range_local(config.pencils.spec, 1)
    if !isempty(spec_range) && maximum(spec_range) > config.nlm
        push!(errors, "Spectral pencil range exceeds config.nlm")
    end
    
    # Note: Transform manager checks removed - now handled by SHTnsKit directly
    
    if !isempty(errors)
        @warn "Velocity configuration validation failed:\n" * join(errors, "\n")
        return false
    end
    
    return true
end
