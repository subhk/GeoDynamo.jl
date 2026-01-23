# ================================================================================
# Velocity Field Module with SHTns
# ================================================================================
#
# This module implements the velocity field representation and momentum equation
# for geodynamo simulations using spherical harmonic transforms (SHTnsKit).
#
# REFERENCE: Sreenivasan & Kar (2018), Phys. Rev. Fluids 3, 093801
#            "Scale dependence of kinetic helicity and selection of the axial
#             dipole in rapidly rotating dynamos"
#
# ================================================================================
# GOVERNING EQUATION
# ================================================================================
#
# The non-dimensional momentum equation in magnetic diffusion time scaling:
#
#   (E/Pm) ∂u/∂t + (∇×u)×u + ẑ×u = -∇p* + (Pm/Pr)Ra·T·r̂ + (∇×B)×B + E∇²u
#
# where:
#   E  = ν/(2Ωd²)  : Ekman number (ratio of viscous to Coriolis forces)
#   Pm = ν/η       : Magnetic Prandtl number (viscous/magnetic diffusivity)
#   Pr = ν/κ       : Prandtl number (viscous/thermal diffusivity)
#   Ra             : Modified Rayleigh number (buoyancy driving)
#
# After dividing by (E/Pm) to get unit coefficient on ∂u/∂t:
#
#   ∂u/∂t = -(Pm/E)(∇×u)×u - (Pm/E)(ẑ×u) - (Pm/E)∇p*
#           + (Pm/E)(Pm/Pr)Ra·T·r̂ + (Pm/E)(∇×B)×B + Pm∇²u
#
# The factor (Pm/E) is called `rossby_factor` in the code.
#
# ================================================================================
# TOROIDAL-POLOIDAL DECOMPOSITION
# ================================================================================
#
# For incompressible flow (∇·u = 0), velocity is decomposed as:
#
#   u = ∇×(T r̂) + ∇×∇×(P r̂)
#
# where:
#   T(r,θ,φ) = Toroidal scalar potential (purely tangential flow)
#   P(r,θ,φ) = Poloidal scalar potential (has radial component)
#
# In spectral space, each (l,m) mode has independent T_lm(r) and P_lm(r).
#
# Physical interpretation:
#   - Toroidal: horizontal circulation (like trade winds, zonal jets)
#   - Poloidal: overturning circulation (like convection cells, meridional flow)
#
# This decomposition AUTOMATICALLY satisfies ∇·u = 0 (Eq. 4 in paper).
#
# ================================================================================
# BOUNDARY CONDITIONS
# ================================================================================
#
# Two main velocity BC types for geodynamo:
#
# 1. NO-SLIP (rigid boundary): u = 0 at boundary
#    - Poloidal: P = 0, ∂P/∂r = 0 (Dirichlet)
#    - Toroidal: T = 0 (Dirichlet)
#
# 2. STRESS-FREE (free-slip): No tangential stress at boundary
#    - Poloidal: P = 0 (v_r = 0, impermeable)
#    - Toroidal: ∂T/∂r = T/r (NOT simple Neumann!)
#
#    The stress-free condition on toroidal component comes from:
#      σ_rθ = μ * r * ∂/∂r(v_θ/r) = μ * (∂v_θ/∂r - v_θ/r) = 0
#
#    For toroidal flow where v_θ ∝ T:
#      ∂T/∂r = T/r
#
#    This is implemented using finite differences:
#      Inner: T[1] = T[2] / (1 + Δr/r[1])
#      Outer: T[nr] = T[nr-1] / (1 - Δr/r[nr])
#
# ================================================================================
# MPI PARALLELIZATION
# ================================================================================
#
# Data distribution (PencilArrays):
#   - Spectral data: distributed over (l,m) modes via `lm_range`
#   - Radial data: distributed over radial points via `r_range`
#
# CRITICAL: BC functions with MPI collectives use global loop bounds.
# See timestep.jl header for detailed explanation of the MPI safety pattern.
#
# The `owns_mode` parameter in BC functions indicates whether this MPI process
# owns the current (l,m) mode. All processes must call BC functions for each
# mode to ensure MPI collectives are balanced.
#
# ================================================================================
# WORKSPACE OPTIMIZATION
# ================================================================================
#
# VelocityWorkspace provides pre-allocated buffers to avoid allocation in
# hot loops. Create once and reuse:
#
#   ws = create_velocity_workspace(Float64, nr)
#   set_velocity_workspace!(ws)  # Register globally
#
# BC functions automatically use the workspace if available, providing
# ~10-100x speedup for BC application.
#
# ================================================================================

import .bcs
import .bcs: BoundaryType, DIRICHLET, NEUMANN

# ================================================================================
# Velocity Field Data Structures
# ================================================================================

# ---------------------------------
# Optional shared workspace support (defined before velocity_bc.jl needs it)
# ---------------------------------
struct VelocityWorkspace{T}
    Pᴾ_profile_real::Vector{Vector{T}}
    Pᴾ_profile_imag::Vector{Vector{T}}
    Tᵀ_profile_real::Vector{Vector{T}}
    Tᵀ_profile_imag::Vector{Vector{T}}
    ∂ᵣ𝒫_real::Vector{Vector{T}}
    ∂ᵣ𝒫_imag::Vector{Vector{T}}
    ∂ᵣᵣ𝒫_real::Vector{Vector{T}}
    ∂ᵣᵣ𝒫_imag::Vector{Vector{T}}
    # Pre-allocated buffers for BC operations (avoid allocations per mode)
    bc_profile_real::Vector{Vector{T}}
    bc_profile_imag::Vector{Vector{T}}
    bc_dprofile_real::Vector{Vector{T}}
    bc_dprofile_imag::Vector{Vector{T}}
    bc_correction::Vector{Vector{T}}
end

# Include matrix-embedded velocity BC functions (Fortran DD_2DCODE style)
include("bcs/velocity_bc.jl")

# Velocity field components with SHTns
mutable struct SHTnsVelocityFields{T}
    # Physical space velocities
    velocity::SHTnsVectorField{T}
    vorticity::SHTnsVectorField{T}

    # Spectral representation (toroidal-poloidal)
    𝒯::SHTnsSpecField{T}
    𝒫::SHTnsSpecField{T}
    
    # Vorticity in spectral space (for efficient curl computation)
    ζᵀ::SHTnsSpecField{T}
    ζᴾ::SHTnsSpecField{T}
    
    # Nonlinear terms
    nlᵀ::SHTnsSpecField{T}
    nlᴾ::SHTnsSpecField{T}
    prev_nlᵀ::SHTnsSpecField{T}
    prev_nlᴾ::SHTnsSpecField{T}
    
    # Work arrays for efficient computation
    work_tor::SHTnsSpecField{T}
    work_pol::SHTnsSpecField{T}
    work_physical::SHTnsVectorField{T}
    advection_physical::SHTnsVectorField{T}
    
    # Pre-computed coefficients
    ℓ_factors::Vector{Float64}          # l(l+1) values
    coriolis_factors::Matrix{Float64}   # Pre-computed Coriolis terms
    
    # Radial derivative matrices
    ∂r::BandedMatrix{T}          # First derivative
    ∂²r::BandedMatrix{T}         # Second derivative
    laplacian_matrix::BandedMatrix{T}   # Radial Laplacian operator
    
    # Transform manager removed; SHTnsKit transforms are used directly
    config::SHTnsKitConfig
    domain::RadialDomain
    boundary_condition_set::Union{bcs.BoundaryConditionSet{T}, Nothing}
    boundary_interpolation_cache::Dict{String, Any}
    boundary_time_index::Ref{Int}
end

# VelocityWorkspace creation function (struct defined at top of file)
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
    enforce_velocity_boundary_values!(𝒰)

Anchor toroidal and poloidal spectral coefficients to the currently cached
Dirichlet boundary values on the inner and outer radial surfaces.
"""
function enforce_velocity_boundary_values!(𝒰::SHTnsVelocityFields{T}) where T
    domain = 𝒰.domain
    tor_real = parent(𝒰.𝒯.data_real)
    tor_imag = parent(𝒰.𝒯.data_imag)
    pol_real = parent(𝒰.𝒫.data_real)
    pol_imag = parent(𝒰.𝒫.data_imag)

    tor_bc = 𝒰.𝒯.boundary_values
    pol_bc = 𝒰.𝒫.boundary_values

    lm_range = get_local_range(𝒰.𝒯.pencil, 1)
    r_range  = get_local_range(𝒰.𝒯.pencil, 3)

    has_inner = 1 in r_range && domain.r[1, 4] > 0
    has_outer = domain.N in r_range

    inner_idx = has_inner ? (1 - first(r_range) + 1) : 0
    outer_idx = has_outer ? (domain.N - first(r_range) + 1) : 0

    dirichlet_code = Int(bcs.DIRICHLET)

    for lm_idx in lm_range
        if lm_idx <= 𝒰.𝒯.nlm
            local_lm = lm_idx - first(lm_range) + 1

            if has_inner && 1 <= inner_idx <= size(tor_real, 3)
                if 𝒰.𝒯.bc_type_inner[lm_idx] == dirichlet_code
                    tor_real[local_lm, 1, inner_idx] = tor_bc[1, lm_idx]
                    tor_imag[local_lm, 1, inner_idx] = zero(T)
                end
                if 𝒰.𝒫.bc_type_inner[lm_idx] == dirichlet_code
                    pol_real[local_lm, 1, inner_idx] = pol_bc[1, lm_idx]
                    pol_imag[local_lm, 1, inner_idx] = zero(T)
                end
            end

            if has_outer && 1 <= outer_idx <= size(tor_real, 3)
                if 𝒰.𝒯.bc_type_outer[lm_idx] == dirichlet_code
                    tor_real[local_lm, 1, outer_idx] = tor_bc[2, lm_idx]
                    tor_imag[local_lm, 1, outer_idx] = zero(T)
                end
                if 𝒰.𝒫.bc_type_outer[lm_idx] == dirichlet_code
                    pol_real[local_lm, 1, outer_idx] = pol_bc[2, lm_idx]
                    pol_imag[local_lm, 1, outer_idx] = zero(T)
                end
            end
        end
    end

    return 𝒰
end


function compute_vorticity_spectral_full!(𝒰::SHTnsVelocityFields{T},
                                          domain::RadialDomain,
                                          ws::VelocityWorkspace{T}) where T
    # Same as the threaded version but using provided workspace buffers
    uᵀ_real = parent(𝒰.𝒯.data_real)
    uᵀ_imag = parent(𝒰.𝒯.data_imag)
    uᴾ_real = parent(𝒰.𝒫.data_real)
    uᴾ_imag = parent(𝒰.𝒫.data_imag)
    ζᵀ_real = parent(𝒰.ζᵀ.data_real)
    ζᵀ_imag = parent(𝒰.ζᵀ.data_imag)
    ζᴾ_real = parent(𝒰.ζᴾ.data_real)
    ζᴾ_imag = parent(𝒰.ζᴾ.data_imag)

    config = 𝒰.𝒯.config
    lm_range = get_local_range(𝒰.𝒯.pencil, 1)
    r_range  = get_local_range(𝒰.𝒯.pencil, 3)
    nr = domain.N

    Threads.@threads for lm_idx in lm_range
        if lm_idx <= length(𝒰.ℓ_factors)
            local_lm = lm_idx - first(lm_range) + 1
            ℓ_factor = 𝒰.ℓ_factors[lm_idx]
            tid = Threads.threadid()

            # Thread safety: ensure thread ID is within workspace bounds
            if tid > length(ws.Pᴾ_profile_real)
                error("Thread ID $tid exceeds workspace size $(length(ws.Pᴾ_profile_real)). " *
                      "This indicates the workspace was created for fewer threads than are currently active. " *
                      "Workspace threads: $(length(ws.Pᴾ_profile_real)), Active threads: $(Threads.nthreads()). " *
                      "Recreate the workspace with the correct thread count.")
            end

            Pᴾ_profile_real = ws.Pᴾ_profile_real[tid]
            Pᴾ_profile_imag = ws.Pᴾ_profile_imag[tid]
            Tᵀ_profile_real = ws.Tᵀ_profile_real[tid]
            Tᵀ_profile_imag = ws.Tᵀ_profile_imag[tid]
            ∂ᵣ𝒫_real     = ws.∂ᵣ𝒫_real[tid]
            ∂ᵣ𝒫_imag     = ws.∂ᵣ𝒫_imag[tid]
            ∂ᵣᵣ𝒫_real   = ws.∂ᵣᵣ𝒫_real[tid]
            ∂ᵣᵣ𝒫_imag   = ws.∂ᵣᵣ𝒫_imag[tid]

            extract_local_radial_profile!(Pᴾ_profile_real, uᴾ_real, local_lm, nr, r_range)
            extract_local_radial_profile!(Pᴾ_profile_imag, uᴾ_imag, local_lm, nr, r_range)
            extract_local_radial_profile!(Tᵀ_profile_real, uᵀ_real, local_lm, nr, r_range)
            extract_local_radial_profile!(Tᵀ_profile_imag, uᵀ_imag, local_lm, nr, r_range)

            apply_∂r!(∂ᵣ𝒫_real,   𝒰.∂r,  Pᴾ_profile_real)
            apply_∂r!(∂ᵣ𝒫_imag,   𝒰.∂r,  Pᴾ_profile_imag)
            apply_∂r!(∂ᵣᵣ𝒫_real, 𝒰.∂²r, Pᴾ_profile_real)
            apply_∂r!(∂ᵣᵣ𝒫_imag, 𝒰.∂²r, Pᴾ_profile_imag)

            r_first = first(r_range)
            r_last = min(last(r_range), nr)
            if r_last < r_first
                continue
            end
            @inbounds @simd for r_idx in r_first:r_last
                local_r = r_idx - r_first + 1
                if local_r <= size(ζᵀ_real, 3) && r_idx <= size(domain.r, 1)
                    r = domain.r[r_idx, 4]
                    if r == 0.0
                        ζᵀ_real[local_lm, 1, local_r] = 0
                        ζᵀ_imag[local_lm, 1, local_r] = 0
                        ζᴾ_real[local_lm, 1, local_r] = 0
                        ζᴾ_imag[local_lm, 1, local_r] = 0
                    else
                        r⁻¹  = domain.r[r_idx, 3]
                        r⁻² = domain.r[r_idx, 2]
                        ζᵀ_real[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_profile_real[r_idx]
                                                            - ∂ᵣᵣ𝒫_real[r_idx]
                                                            - 2.0 * r⁻¹ * ∂ᵣ𝒫_real[r_idx])

                        ζᵀ_imag[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_profile_imag[r_idx]
                                                            - ∂ᵣᵣ𝒫_imag[r_idx]
                                                            - 2.0 * r⁻¹ * ∂ᵣ𝒫_imag[r_idx])

                        ζᴾ_real[local_lm, 1, local_r] = -ℓ_factor * r⁻² * Tᵀ_profile_real[r_idx]
                        
                        ζᴾ_imag[local_lm, 1, local_r] = -ℓ_factor * r⁻² * Tᵀ_profile_imag[r_idx]
                    end
                end
            end
        end
    end
end


function create_shtns_velocity_fields(::Type{T}, config::SHTnsKitConfig, 
                                      𝒟ᵒᶜ::RadialDomain, 
                                      pencils=nothing, pencil_spec=nothing) where T
    # Use pencils from config by default (they already encode the correct nr)
    if pencils === nothing
        pencils = config.pencils
    end
    pencil_θ, pencil_φ, pencil_r = pencils.θ, pencils.φ, pencils.r
    
    # Use spectral pencil from topology if not provided
    if pencil_spec === nothing
        pencil_spec = pencils.spec
    end
    
    # Create vector fields
    velocity  = create_shtns_vector_field(T, config, 𝒟ᵒᶜ, pencils)
    vorticity = create_shtns_vector_field(T, config, 𝒟ᵒᶜ, pencils)
    
    # Spectral fields
    𝒯        = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    𝒫        = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    ζᵀ       = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    ζᴾ       = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    nlᵀ      = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    nlᴾ      = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    prev_nlᵀ = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    prev_nlᴾ = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    
    # Work arrays
    work_tor           = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    work_pol           = create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec)
    work_physical      = create_shtns_vector_field(T, config, 𝒟ᵒᶜ, pencils)
    advection_physical = create_shtns_vector_field(T, config, 𝒟ᵒᶜ, pencils)
    
    # Pre-compute l(l+1) factors
    ℓ_factors = Float64[l * (l + 1) for l in config.l_values]
    
    # Pre-compute Coriolis factors (sin(θ) and cos(θ))
    coriolis_factors = zeros(Float64, 2, config.nlat)
    for i in 1:config.nlat
        coriolis_factors[1, i] = sin(config.theta_grid[i])
        coriolis_factors[2, i] = cos(config.theta_grid[i])
    end
    
    # Create radial derivative matrices
    ∂r        = create_derivative_matrix(1, 𝒟ᵒᶜ)
    ∂²r       = create_derivative_matrix(2, 𝒟ᵒᶜ)
    laplacian_matrix = create_radial_laplacian(𝒟ᵒᶜ)
    
    # Create transpose plans for efficient data movement
    transpose_plans = create_transpose_plans(pencils)
    
    boundary_condition_set = nothing
    boundary_cache = Dict{String, Any}()
    boundary_time_index = Ref{Int}(1)

    return SHTnsVelocityFields{T}(velocity, vorticity, 𝒯, 𝒫,
                                  ζᵀ, ζᴾ,
                                  nlᵀ, nlᴾ, prev_nlᵀ, prev_nlᴾ,
                                  work_tor, work_pol, work_physical,
                                  advection_physical,
                                  ℓ_factors, coriolis_factors,
                                  ∂r, ∂²r, laplacian_matrix,
                                  config,
                                  𝒟ᵒᶜ,
                                  boundary_condition_set, boundary_cache, boundary_time_index)
end


# =============================
# Main nonlinear computation
# =============================
function compute_velocity_nonlinear!(𝒰::SHTnsVelocityFields{T},
                                    temp_field, comp_field, mag_field,
                                    𝒟ᵒᶜ::RadialDomain;
                                    geometry::Symbol = get_parameters().geometry) where T
    # Zero work arrays once
    zero_velocity_work_arrays!(𝒰)

    # Step 1: Use enhanced vector synthesis with automatic transpose handling
    shtnskit_vector_synthesis!(𝒰.𝒯, 𝒰.𝒫, 𝒰.velocity; domain=𝒟ᵒᶜ)

    # Step 2: Compute vorticity in spectral space with enhanced derivative computation
    compute_vorticity_spectral_full!(𝒰, 𝒟ᵒᶜ)

    # Step 3: Transform vorticity to physical space with batched operations
    shtnskit_vector_synthesis!(𝒰.ζᵀ, 𝒰.ζᴾ, 𝒰.vorticity; domain=𝒟ᵒᶜ)

    # Step 4: Compute all nonlinear terms with enhanced memory access patterns
    compute_all_nonlinear_terms!(𝒰, temp_field, comp_field, mag_field, 𝒟ᵒᶜ)

    # Step 5: Use enhanced vector analysis with efficient data layout
    if geometry === :ball
        ball_vector_analysis!(𝒰.advection_physical, 𝒰.nlᵀ, 𝒰.nlᴾ)
    else
        shtnskit_vector_analysis!(𝒰.advection_physical, 𝒰.nlᵀ, 𝒰.nlᴾ)
    end
end



# =================================================
# Vorticity Computation in Spectral Space
# =================================================
#
# MATHEMATICAL BACKGROUND:
# ========================
# Vorticity ζ = ∇×u is computed in spectral space using the curl operator
# for toroidal-poloidal decomposition.
#
# For a vector field V = ∇×(T r̂) + ∇×∇×(P r̂), the curl satisfies:
#
#   (∇×V)_toroidal = [l(l+1)/r² - d²/dr² - (2/r)d/dr] V_poloidal
#   (∇×V)_poloidal = -l(l+1)/r² V_toroidal
#
# This is the SAME formula used for:
#   - Vorticity: ζ = ∇×u (this function)
#   - Current density: j = ∇×B (in magnetic.jl)
#   - Induction curl: ∇×(u×B) (in magnetic.jl)
#
# The formula arises from the identity ∇×∇×A = ∇(∇·A) - ∇²A combined with
# the spherical harmonic eigenvalue -l(l+1)/r² for the angular Laplacian.
#
# =================================================
using Base.Threads
function compute_vorticity_spectral_full!(𝒰::SHTnsVelocityFields{T},
                                         domain::RadialDomain) where T
    # If a compatible workspace is registered, use it
    ws_any = VELOCITY_WS[]
    if ws_any !== nothing && ws_any isa VelocityWorkspace{T}
        return compute_vorticity_spectral_full!(𝒰, domain, ws_any)
    end

    # =========================================================================
    # Compute vorticity ζ = ∇×u in spectral space
    # =========================================================================
    # For toroidal-poloidal decomposition, the curl operator gives:
    #
    #   ζᵀoidal = [l(l+1)/r² - d²/dr² - (2/r)d/dr] uᴾoidal
    #   ζᴾoidal = -l(l+1)/r² uᵀoidal
    #
    # where l is the spherical harmonic degree.
    # =========================================================================

    # Get local data views with enhanced memory access
    uᵀ_real = parent(𝒰.𝒯.data_real)
    uᵀ_imag = parent(𝒰.𝒯.data_imag)
    uᴾ_real = parent(𝒰.𝒫.data_real)
    uᴾ_imag = parent(𝒰.𝒫.data_imag)

    ζᵀ_real = parent(𝒰.ζᵀ.data_real)
    ζᵀ_imag = parent(𝒰.ζᵀ.data_imag)
    ζᴾ_real = parent(𝒰.ζᴾ.data_real)
    ζᴾ_imag = parent(𝒰.ζᴾ.data_imag)

    # Use enhanced range functions from pencil decomposition
    config = 𝒰.𝒯.config

    # Get local ranges using pencil topology
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    lm_range = range_local(config.pencils.spec, 1)
    r_range  = range_local(config.pencils.spec, 3)

    nr = domain.N

    # Radial data is always local (not MPI distributed), use threaded version
    _compute_vorticity_spectral_threaded!(𝒰, domain, uᵀ_real, uᵀ_imag, uᴾ_real, uᴾ_imag,
                                           ζᵀ_real, ζᵀ_imag, ζᴾ_real, ζᴾ_imag,
                                           lm_range, r_range, nr)
end

# Threaded version: Radial data is local (no MPI communication needed)
function _compute_vorticity_spectral_threaded!(𝒰::SHTnsVelocityFields{T}, domain::RadialDomain,
                                                uᵀ_real, uᵀ_imag, uᴾ_real, uᴾ_imag,
                                                ζᵀ_real, ζᵀ_imag, ζᴾ_real, ζᴾ_imag,
                                                lm_range, r_range, nr) where T
    # Thread-local scratch buffers reused across modes to avoid allocations
    nT = max(1, Threads.nthreads())
    Pᴾ_profile_real_bufs = [zeros(T, nr) for _ in 1:nT]
    Pᴾ_profile_imag_bufs = [zeros(T, nr) for _ in 1:nT]
    Tᵀ_profile_real_bufs = [zeros(T, nr) for _ in 1:nT]
    Tᵀ_profile_imag_bufs = [zeros(T, nr) for _ in 1:nT]
    ∂ᵣ𝒫_real_bufs     = [zeros(T, nr) for _ in 1:nT]
    ∂ᵣ𝒫_imag_bufs     = [zeros(T, nr) for _ in 1:nT]
    ∂ᵣᵣ𝒫_real_bufs   = [zeros(T, nr) for _ in 1:nT]
    ∂ᵣᵣ𝒫_imag_bufs   = [zeros(T, nr) for _ in 1:nT]

    # Process each (l,m) mode (parallel over lm)
    @inbounds Threads.@threads for lm_idx in lm_range
        if lm_idx <= length(𝒰.ℓ_factors)
            local_lm = lm_idx - first(lm_range) + 1
            ℓ_factor = 𝒰.ℓ_factors[lm_idx]

            # Select thread-local buffers with proper bounds checking
            tid = Threads.threadid()
            if tid > nT
                error("Thread ID $tid exceeds allocated workspace size $nT. " *
                      "Active threads: $(Threads.nthreads()). " *
                      "This indicates a thread count mismatch. " *
                      "The clamping approach (min(tid, nT)) was removed because it causes data races.")
            end
            Pᴾ_profile_real = Pᴾ_profile_real_bufs[tid]
            Pᴾ_profile_imag = Pᴾ_profile_imag_bufs[tid]
            Tᵀ_profile_real = Tᵀ_profile_real_bufs[tid]
            Tᵀ_profile_imag = Tᵀ_profile_imag_bufs[tid]
            ∂ᵣ𝒫_real     = ∂ᵣ𝒫_real_bufs[tid]
            ∂ᵣ𝒫_imag     = ∂ᵣ𝒫_imag_bufs[tid]
            ∂ᵣᵣ𝒫_real   = ∂ᵣᵣ𝒫_real_bufs[tid]
            ∂ᵣᵣ𝒫_imag   = ∂ᵣᵣ𝒫_imag_bufs[tid]

            # Extract radial profiles (in-place)
            extract_local_radial_profile!(Pᴾ_profile_real, uᴾ_real, local_lm, nr, r_range)
            extract_local_radial_profile!(Pᴾ_profile_imag, uᴾ_imag, local_lm, nr, r_range)
            extract_local_radial_profile!(Tᵀ_profile_real, uᵀ_real, local_lm, nr, r_range)
            extract_local_radial_profile!(Tᵀ_profile_imag, uᵀ_imag, local_lm, nr, r_range)

            # Compute radial derivatives for poloidal component (in-place, reuse buffers)
            apply_∂r!(∂ᵣ𝒫_real,   𝒰.∂r,  Pᴾ_profile_real)
            apply_∂r!(∂ᵣ𝒫_imag,   𝒰.∂r,  Pᴾ_profile_imag)
            apply_∂r!(∂ᵣᵣ𝒫_real, 𝒰.∂²r, Pᴾ_profile_real)
            apply_∂r!(∂ᵣᵣ𝒫_imag, 𝒰.∂²r, Pᴾ_profile_imag)

            # Compute vorticity components
            r_first = first(r_range)
            r_last = min(last(r_range), nr)
            if r_last < r_first
                continue
            end
            @simd for r_idx in r_first:r_last
                local_r = r_idx - r_first + 1
                if local_r <= size(ζᵀ_real, 3)
                    r = domain.r[r_idx, 4]
                    if r == 0.0
                        # At r=0 (ball geometry), regularity implies finite values → set to 0 safely
                        ζᵀ_real[local_lm, 1, local_r] = 0
                        ζᵀ_imag[local_lm, 1, local_r] = 0
                        ζᴾ_real[local_lm, 1, local_r] = 0
                        ζᴾ_imag[local_lm, 1, local_r] = 0
                    else
                        r⁻¹ = domain.r[r_idx, 3]   # 1/r
                        r⁻² = domain.r[r_idx, 2]  # 1/r²
                        # Toroidal vorticity from poloidal velocity (with full derivatives)
                        ζᵀ_real[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_profile_real[r_idx]
                                                            - ∂ᵣᵣ𝒫_real[r_idx]
                                                            - 2.0 * r⁻¹ * ∂ᵣ𝒫_real[r_idx])
                        ζᵀ_imag[local_lm, 1, local_r] = (ℓ_factor * r⁻² * Pᴾ_profile_imag[r_idx]
                                                            - ∂ᵣᵣ𝒫_imag[r_idx]
                                                            - 2.0 * r⁻¹ * ∂ᵣ𝒫_imag[r_idx])
                        # Poloidal vorticity from toroidal velocity
                        ζᴾ_real[local_lm, 1, local_r] = -ℓ_factor * r⁻² * Tᵀ_profile_real[r_idx]
                        ζᴾ_imag[local_lm, 1, local_r] = -ℓ_factor * r⁻² * Tᵀ_profile_imag[r_idx]
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
    compute_all_nonlinear_terms!(𝒰, temp_field, comp_field, mag_field, domain)

Compute all nonlinear forcing terms for the momentum equation.

# Governing Equation (Dimensional)

In a rotating reference frame with angular velocity Ω:

∂u/∂t = -u×ω - ∇p/ρ + 2Ω(ẑ×u) + ν∇²u + (αg/ρ)T·r̂ + (Δρg/ρ)C·r̂ + (1/μ₀ρ)(∇×B)×B

where:
- u: velocity, ζ = ∇×u: vorticity
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

# Non-Dimensional Momentum Equation (magnetic diffusion time scaling)

E·Pm⁻¹[∂ũ/∂τ + (∇×ũ)×ũ] + ẑ×ũ = -∇p̃*
                + (Pm/Pr)·Ra·T̃·r·r̂ + (Pm/Sc)·Ra_C·C̃·r·r̂
                + (∇×B̃)×B̃ + E∇²ũ

where:
  - τ = L²/η (magnetic diffusion time scaling)
  - E = ν/(2ΩL²) is the Ekman number
  - r factor in buoyancy represents linear gravity profile g(r) ∝ r

# Implementation Notes

The explicit RHS entering the time integrator is:
RHS = -(E/Pm)·(∇×ũ)×ũ - (ẑ×ũ) + (Pm/Pr)·Ra·T̃·r·r̂ + (Pm/Sc)·Ra_C·C̃·r·r̂ + (∇×B̃)×B̃

Coefficients:
  - Advection: E/Pm (= E·Pm⁻¹)
  - Coriolis: 1 (no scaling)
  - Thermal buoyancy: (Pm/Pr) * Ra * r (with radial factor)
  - Compositional buoyancy: (Pm/Sc) * Ra_C * r (with radial factor)
  - Lorentz: 1

Viscous diffusion is treated implicitly with coefficient E (Ekman number).
"""
function compute_all_nonlinear_terms!(𝒰::SHTnsVelocityFields{T},
                                               temp_field, comp_field, mag_field,
                                               domain::RadialDomain) where T
    # Compute all forces in a single enhanced loop (magnetic diffusion time scaling)

    # Advection coefficient = E/Pm
    advection_coeff = d_E / d_Pm

    # Get all data views
    vᵣ = parent(𝒰.velocity.r_component.data)
    vθ = parent(𝒰.velocity.θ_component.data)
    vφ = parent(𝒰.velocity.φ_component.data)

    ζᵣ = parent(𝒰.vorticity.r_component.data)
    ζθ = parent(𝒰.vorticity.θ_component.data)
    ζφ = parent(𝒰.vorticity.φ_component.data)

    adv_r = parent(𝒰.advection_physical.r_component.data)
    adv_θ = parent(𝒰.advection_physical.θ_component.data)
    adv_φ = parent(𝒰.advection_physical.φ_component.data)

    # Get dimensions from config for better performance
    config = 𝒰.velocity.r_component.config
    local_size = size(vᵣ)
    nlat = config.nlat
    nlon = config.nlon

    # Use pencil ranges for enhanced loop bounds
    r_range = range_local(config.pencils.r, 3)

    # Main fused computation loop with enhanced indexing (parallel over r-slices)
    # Advection has coefficient E/Pm, Coriolis has coefficient 1
    adv_coeff = advection_coeff
    @inbounds Threads.@threads for k in 1:local_size[3]
        # Get radius for this level using pencil range
        r_idx = k + first(r_range) - 1
        if r_idx <= domain.N
            r = domain.r[r_idx, 4]
            r⁻¹ = domain.r[r_idx, 3]
        else
            r = 1.0
            r⁻¹ = 1.0
        end

        for j in 1:local_size[2]
            # Get pre-computed Coriolis factors for this latitude
            theta_idx = min(j, nlat)
            sin_theta = 𝒰.coriolis_factors[1, theta_idx]
            cos_theta = 𝒰.coriolis_factors[2, theta_idx]
            
            @simd for i in 1:local_size[1]
                linear_idx = i + (j-1)*local_size[1] + (k-1)*local_size[1]*local_size[2]
                
                if linear_idx <= length(vᵣ)
                    # Load velocity and vorticity components
                    u_r = vᵣ[linear_idx]
                    u_θ = vθ[linear_idx]
                    u_φ = vφ[linear_idx]
                    
                    ω_r = ζᵣ[linear_idx]
                    ω_θ = ζθ[linear_idx]
                    ω_φ = ζφ[linear_idx]
                    
                    # Advection: (E/Pm) * (u × ζ)
                    adv_r_val = adv_coeff * (u_θ * ω_φ - u_φ * ω_θ)
                    adv_θ_val = adv_coeff * (u_φ * ω_r - u_r * ω_φ)
                    adv_φ_val = adv_coeff * (u_r * ω_θ - u_θ * ω_r)
                    
                    # Coriolis: −(Pm/E) ẑ × u
                    zhat_cross_r = -sin_theta * u_φ
                    zhat_cross_θ = -cos_theta * u_φ
                    zhat_cross_φ = cos_theta * u_θ + sin_theta * u_r
                    cor_r = -zhat_cross_r  # Coriolis coefficient = 1
                    cor_θ = -zhat_cross_θ
                    cor_φ = -zhat_cross_φ
                    
                    # Store combined result
                    adv_r[linear_idx] = adv_r_val + cor_r
                    adv_θ[linear_idx] = adv_θ_val + cor_θ
                    adv_φ[linear_idx] = adv_φ_val + cor_φ
                end
            end
        end
    end
    
    # Add buoyancy forces: (Pm/Pr)*Ra*r (thermal), (Pm/Sc)*Ra_C*r (compositional)
    # Buoyancy coefficient is (Pm/Pr)·Ra (with radial factor r)
    if temp_field !== nothing
        buoyancy_factor = (d_Pm / d_Pr) * d_Ra
        add_thermal_buoyancy_force!(adv_r, temp_field, buoyancy_factor, domain)
    end

    if comp_field !== nothing
        comp_factor = (d_Pm / d_Sc) * d_Ra_C
        add_buoyancy_force!(adv_r, comp_field, comp_factor, domain)
    end
    
    # Add Lorentz force if magnetic field present
    if mag_field !== nothing
        add_lorentz_force!(𝒰, mag_field, domain)
    end
end


# =====================================
# Thermal buoyancy force addition
# =====================================
function add_thermal_buoyancy_force!(force_r::AbstractArray{T,3},
                                      scalar_field, factor::Float64,
                                      domain::RadialDomain) where T
    # Add buoyancy force: F_buoyancy = (Pm/Pr) · Ra · T · r · r̂
    #
    # Buoyancy includes radial factor for linear gravity profile
    # g(r) ∝ r in a spherical shell (gravity increases linearly with radius)
    #
    # In non-dimensional form with magnetic diffusion time scaling:
    # F = (Pm/Pr) · Ra · T · r · r̂
    if iszero(factor)
        return force_r
    end

    # Get scalar field data
    if isa(scalar_field, SHTnsPhysField)
        scalar_data = parent(scalar_field.data)
    else
        scalar_data = parent(scalar_field.temperature.data)
    end

    # Get pencil configuration to map linear indices to radial positions
    config = scalar_field isa SHTnsPhysField ? scalar_field.config : scalar_field.temperature.config
    r_range = range_local(config.pencils.r, 3)
    local_size = size(force_r)

    # Add buoyancy WITH radial factor r for linear gravity profile
    # Loop over radial levels to apply r-dependent scaling
    @inbounds Threads.@threads for k in 1:local_size[3]
        # Get radius for this level using pencil range
        r_idx = k + first(r_range) - 1
        if r_idx <= domain.N
            r = domain.r[r_idx, 4]  # r coordinate at this radial level
        else
            r = 1.0
        end

        # Apply factor * r * T at this radial level
        factor_r = factor * r
        for j in 1:local_size[2]
            @simd for i in 1:local_size[1]
                linear_idx = i + (j-1)*local_size[1] + (k-1)*local_size[1]*local_size[2]
                if linear_idx <= length(scalar_data)
                    force_r[linear_idx] += factor_r * scalar_data[linear_idx]
                end
            end
        end
    end
end

# Compositional buoyancy force (similar to thermal but for composition)
function add_buoyancy_force!(force_r::AbstractArray{T,3},
                             comp_field, factor::Float64,
                             domain::RadialDomain) where T
    # Add compositional buoyancy force: F_comp = (Pm/Sc) · Ra_C · C · r · r̂
    #
    # Buoyancy includes radial factor for linear gravity profile
    # g(r) ∝ r in a spherical shell (gravity increases linearly with radius)
    #
    # In non-dimensional form with magnetic diffusion time scaling:
    # F = (Pm/Sc) · Ra_C · C · r · r̂
    if iszero(factor)
        return force_r
    end

    # Get compositional field data
    if isa(comp_field, SHTnsPhysField)
        comp_data = parent(comp_field.data)
    else
        comp_data = parent(comp_field.composition.data)
    end

    # Get pencil configuration to map linear indices to radial positions
    config = comp_field isa SHTnsPhysField ? comp_field.config : comp_field.composition.config
    r_range = range_local(config.pencils.r, 3)
    local_size = size(force_r)

    # Add buoyancy WITH radial factor r for linear gravity profile
    # Loop over radial levels to apply r-dependent scaling
    @inbounds Threads.@threads for k in 1:local_size[3]
        # Get radius for this level using pencil range
        r_idx = k + first(r_range) - 1
        if r_idx <= domain.N
            r = domain.r[r_idx, 4]  # r coordinate at this radial level
        else
            r = 1.0
        end

        # Apply factor * r * C at this radial level
        factor_r = factor * r
        for j in 1:local_size[2]
            @simd for i in 1:local_size[1]
                linear_idx = i + (j-1)*local_size[1] + (k-1)*local_size[1]*local_size[2]
                if linear_idx <= length(comp_data)
                    force_r[linear_idx] += factor_r * comp_data[linear_idx]
                end
            end
        end
    end
end

# ===============================
# Optimized Lorentz force computation
# ===============================
function add_lorentz_force!(𝒰::SHTnsVelocityFields{T},
                           mag_field::SHTnsMagneticFields{T},
                           domain::RadialDomain) where T

    # Compute Lorentz force F = (∇ × B) × B with vectorization
    # Lorentz coefficient = 1 (magnetic diffusion time scaling)

    # Step 1: Use pre-computed current density from magnetic field (j = ∇×B)

    # Step 2: Compute j × B with enhanced vectorization
    j_r = parent(mag_field.current.r_component.data)
    j_θ = parent(mag_field.current.θ_component.data)
    j_φ = parent(mag_field.current.φ_component.data)

    B_r = parent(mag_field.magnetic.r_component.data)
    B_θ = parent(mag_field.magnetic.θ_component.data)
    B_φ = parent(mag_field.magnetic.φ_component.data)

    adv_r = parent(𝒰.advection_physical.r_component.data)
    adv_θ = parent(𝒰.advection_physical.θ_component.data)
    adv_φ = parent(𝒰.advection_physical.φ_component.data)

    # Fused loop for j × B (Lorentz coefficient = 1)
    @inbounds @simd for idx in eachindex(j_r)
        if idx <= length(B_r)
            # Add Lorentz force to existing forces
            adv_r[idx] += (j_θ[idx] * B_φ[idx] - j_φ[idx] * B_θ[idx])
            adv_θ[idx] += (j_φ[idx] * B_r[idx] - j_r[idx] * B_φ[idx])
            adv_φ[idx] += (j_r[idx] * B_θ[idx] - j_θ[idx] * B_r[idx])
        end
    end
end


# Note: Boundary condition functions moved to src/bcs/velocity.jl


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
#                                  ℓ_factor::Float64, domain::RadialDomain) where T
#     # Solve (∇²_r - l(l+1)/r²) u = source
#     # This is a simplified solver - in practice would use more sophisticated methods
    
#     N = 𝒟ᵒᶜ.N
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
#                     r⁻² = 𝒟ᵒᶜ.r[i, 2]
#                     operator[i, j] -= ℓ_factor * r⁻²
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
function compute_kinetic_energy(𝒰::SHTnsVelocityFields{T}, 𝒟ᵒᶜ::RadialDomain) where T
    # Compute kinetic energy with configuration-aware integration

    tor_real = parent(𝒰.𝒯.data_real)
    tor_imag = parent(𝒰.𝒯.data_imag)
    pol_real = parent(𝒰.𝒫.data_real)
    pol_imag = parent(𝒰.𝒫.data_imag)

    local_energy = zero(Float64)

    # Use configuration pencils for consistent range access
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    config = 𝒰.𝒯.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range = range_local(config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= 𝒰.𝒯.nlm
            local_lm = lm_idx - first(lm_range) + 1
            ℓ_factor = 𝒰.ℓ_factors[lm_idx]
            
            # Weight by l(l+1) for proper spectral integration
            weight = 1.0 / max(ℓ_factor, 1.0)
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    # Include radial weight for spherical integration
                    r = 𝒟ᵒᶜ.r[r_idx, 4]
                    r_weight = r^2 * 𝒟ᵒᶜ.integration_weights[r_idx]
                    
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
    return 0.5 * MPI.Allreduce(local_energy, MPI.SUM, get_comm())
end


function compute_reynolds_stress(𝒰::SHTnsVelocityFields{T}) where T
    # Compute Reynolds stress tensor <u_i u_j>
    # This requires transforming to physical space and computing products

    vᵣ = parent(𝒰.velocity.r_component.data)
    vθ = parent(𝒰.velocity.θ_component.data)
    vφ = parent(𝒰.velocity.φ_component.data)
    
    # Compute all 6 independent components
    R_rr = mean(vᵣ .* vᵣ)
    R_θθ = mean(vθ .* vθ)
    R_φφ = mean(vφ .* vφ)
    R_rθ = mean(vᵣ .* vθ)
    R_rφ = mean(vᵣ .* vφ)
    R_θφ = mean(vθ .* vφ)
    
    # Global averages
    R_rr = MPI.Allreduce(R_rr, MPI.SUM, get_comm()) / MPI.Comm_size(get_comm())
    R_θθ = MPI.Allreduce(R_θθ, MPI.SUM, get_comm()) / MPI.Comm_size(get_comm())
    R_φφ = MPI.Allreduce(R_φφ, MPI.SUM, get_comm()) / MPI.Comm_size(get_comm())
    R_rθ = MPI.Allreduce(R_rθ, MPI.SUM, get_comm()) / MPI.Comm_size(get_comm())
    R_rφ = MPI.Allreduce(R_rφ, MPI.SUM, get_comm()) / MPI.Comm_size(get_comm())
    R_θφ = MPI.Allreduce(R_θφ, MPI.SUM, get_comm()) / MPI.Comm_size(get_comm())
    
    return (R_rr, R_θθ, R_φφ, R_rθ, R_rφ, R_θφ)
end


# ================================================================================
# Utility functions
# ================================================================================
function zero_velocity_work_arrays!(𝒰::SHTnsVelocityFields{T}) where T
    # Efficiently zero all work arrays with batch operations
    # Use threaded operations for better performance on large arrays
    Threads.@threads for arr in [
        parent(𝒰.work_tor.data_real),
        parent(𝒰.work_tor.data_imag),
        parent(𝒰.work_pol.data_real),
        parent(𝒰.work_pol.data_imag),
        parent(𝒰.work_physical.r_component.data),
        parent(𝒰.work_physical.θ_component.data),
        parent(𝒰.work_physical.φ_component.data),
        parent(𝒰.advection_physical.r_component.data),
        parent(𝒰.advection_physical.θ_component.data),
        parent(𝒰.advection_physical.φ_component.data),
        parent(𝒰.ζᵀ.data_real),
        parent(𝒰.ζᵀ.data_imag),
        parent(𝒰.ζᴾ.data_real),
        parent(𝒰.ζᴾ.data_imag)
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
    batch_velocity_transforms!(𝒰::SHTnsVelocityFields{T}) where T

Perform batched transforms for better cache efficiency using shtnskit_transforms.jl
"""
function batch_velocity_transforms!(𝒰::SHTnsVelocityFields{T}) where T
    # Use batched operations from shtnskit_transforms.jl for better performance
    specs = [𝒰.𝒯, 𝒰.𝒫, 𝒰.ζᵀ, 𝒰.ζᴾ]
    physs = [𝒰.work_physical.r_component, 𝒰.work_physical.θ_component,
             𝒰.work_physical.φ_component, 𝒰.velocity.r_component]

    # Only transform if specs and physs have compatible lengths
    n_transform = min(length(specs), length(physs))
    if n_transform > 0
        batch_spectral_to_physical!(specs[1:n_transform], physs[1:n_transform])
    end
end


"""
    optimize_velocity_memory_layout!(𝒰::SHTnsVelocityFields{T}) where T

Optimize memory layout for better cache performance using pencil topology
"""
function optimize_velocity_memory_layout!(𝒰::SHTnsVelocityFields{T}) where T
    # Use transpose plans for optimal data layout based on upcoming operations
    config = 𝒰.𝒯.config

    # Use transpose plans if available
    plans = config.transpose_plans
    if !isempty(plans) && haskey(plans, :r_to_spec)
        transpose_with_timer!(𝒰.work_tor.data_real, 𝒰.𝒯.data_real,
                              plans[:r_to_spec], "toroidal_layout_opt")
        transpose_with_timer!(𝒰.work_pol.data_real, 𝒰.𝒫.data_real,
                              plans[:r_to_spec], "poloidal_layout_opt")
    end
end


"""
    validate_velocity_configuration(𝒰::SHTnsVelocityFields{T}, config::SHTnsKitConfig) where T

Validate velocity field configuration consistency with SHTns setup
"""
function validate_velocity_configuration(𝒰::SHTnsVelocityFields{T}, config::SHTnsKitConfig) where T
    errors = String[]

    # Check field dimensions match config
    if size(𝒰.𝒯.data_real, 1) != config.nlm
        push!(errors, "Toroidal field size mismatch with config.nlm")
    end

    # Check that ℓ_factors are consistent
    if length(𝒰.ℓ_factors) != config.nlm
        push!(errors, "ℓ_factors length mismatch with config.nlm")
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
