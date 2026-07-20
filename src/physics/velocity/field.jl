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
# The non-dimensional momentum equation advanced by the solver in magnetic
# diffusion time scaling:
#
#   E ∂u/∂t = E(u×ω) - ẑ×u - ∇p*
#             + (Pm/Pr)Ra·T·r·r̂ + (Pm/Sc)Ra_C·C·r·r̂
#             + (1/Pm)(∇×B)×B + E∇²u
#
# where:
#   E  = ν/(2Ωd²)  : Ekman number (ratio of viscous to Coriolis forces)
#   Pm = ν/η       : Magnetic Prandtl number (viscous/magnetic diffusivity)
#   Pr = ν/κ       : Prandtl number (viscous/thermal diffusivity)
#   Ra             : Modified Rayleigh number (buoyancy driving)
#
# The mass coefficient E is applied by the time-stepping matrices. The explicit
# force accumulator stores the non-diffusive right-hand side shown above.
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
#    - Poloidal: P = 0, P″ - (2/r)P′ = 0
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
# hot loops. Each SHTnsVelocityFields owns one (built lazily via
# `_get_or_build_velocity_workspace!`) — it is NOT shared across solver states.
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
"""
    VelocityWorkspace{T}

Reusable radial work buffers for velocity boundary-condition and derivative
operations. Keeping these arrays alive across timesteps avoids repeated
allocation in the hottest velocity-side boundary kernels.
"""
struct VelocityWorkspace{T}
    Pᴾ_profile_real::Vector{Vector{T}}
    Pᴾ_profile_imag::Vector{Vector{T}}
    Tᵀ_profile_real::Vector{Vector{T}}
    Tᵀ_profile_imag::Vector{Vector{T}}
    ∂ᵣpoloidal_real::Vector{Vector{T}}
    ∂ᵣpoloidal_imag::Vector{Vector{T}}
    ∂ᵣᵣpoloidal_real::Vector{Vector{T}}
    ∂ᵣᵣpoloidal_imag::Vector{Vector{T}}
    # Pre-allocated buffers for BC operations (avoid allocations per mode)
    bc_profile_real::Vector{Vector{T}}
    bc_profile_imag::Vector{Vector{T}}
    bc_dprofile_real::Vector{Vector{T}}
    bc_dprofile_imag::Vector{Vector{T}}
    bc_correction::Vector{Vector{T}}
    # Force-projection radial scratch (nl-3): rS and ∂r(rS) for
    # _poloidal_force_projection!, cached instead of allocated per call.
    force_proj_rS::Vector{Vector{T}}
    force_proj_drS::Vector{Vector{T}}
end

# Include matrix-embedded velocity BC functions
include("../../bcs/velocity_bc.jl")

"""
    SHTnsVelocityFields{T}

Velocity state in the toroidal-poloidal formulation used by GeoDynamo.

This bundles the physical velocity/vorticity fields, their spectral toroidal
and poloidal coefficients, nonlinear work arrays, and the radial derivative
operators needed by the velocity update kernels.
"""
mutable struct SHTnsVelocityFields{
    T,
    C <: SHTnsKitConfig,
    VF <: SHTnsVectorField{T},
    SF <: SHTnsSpecField{T}
}
    # Physical space velocities
    velocity::VF
    vorticity::VF

    # Spectral representation (toroidal-poloidal)
    toroidal::SF
    poloidal::SF

    # Vorticity in spectral space (for efficient curl computation)
    ζᵀ::SF
    ζᴾ::SF

    # Nonlinear terms
    nl_toroidal::SF
    nl_poloidal::SF
    prev_nl_toroidal::SF
    prev_nl_poloidal::SF

    # Work arrays for efficient computation
    work_tor::SF
    work_pol::SF
    work_physical::VF
    advection_physical::VF

    # Pre-computed coefficients
    l_factors::Vector{T}                # l(l+1) values
    coriolis_factors::Matrix{T}         # Pre-computed Coriolis terms

    # Radial derivative matrices
    ∂r::BandedMatrix{T}          # First derivative
    ∂²r::BandedMatrix{T}         # Second derivative
    laplacian_matrix::BandedMatrix{T}   # Radial Laplacian operator

    # Transform manager removed; SHTnsKit transforms are used directly
    config::C
    domain::RadialDomain
    parameters::SolverParameters
    boundary_condition_set::Union{bcs.BoundaryConditionSet{T}, Nothing}
    boundary_interpolation_cache::bcs.BoundaryInterpolationCache{T}
    boundary_time_index::Ref{Int}

    # State-local scratch workspace for the radial-profile velocity kernels.
    # Built lazily on first use. Owned by this field (NOT a module global) so it
    # is never shared across solver states — sharing previously leaked stale
    # values between states and made the magnetic-poloidal solve nondeterministic
    # under the full test suite (finite in isolation, occasionally non-finite in
    # CI). See `_get_or_build_velocity_workspace!`.
    velocity_workspace::Union{VelocityWorkspace{T}, Nothing}
end

"""
    create_velocity_workspace(::Type{T}, nr; nthreads=max(Threads.nthreads(), Threads.maxthreadid())) where T

Create a thread-local `VelocityWorkspace` sized for `nr` radial points. One
buffer set is allocated per thread so velocity boundary kernels can reuse
scratch storage without synchronization.
"""
function create_velocity_workspace(::Type{T}, nr::Int,
        nthreads::Int = max(Threads.nthreads(), Threads.maxthreadid())) where {T}
    bufs() = [zeros(T, nr) for _ in 1:nthreads]
    return VelocityWorkspace{T}(
        bufs(), bufs(), bufs(), bufs(), bufs(), bufs(), bufs(), bufs(),
        # BC buffers
        bufs(), bufs(), bufs(), bufs(), bufs(),
        # force-projection scratch (nl-3)
        bufs(), bufs()
    )
end

"""
    _get_or_build_velocity_workspace!(𝒰, nr; nthreads) -> VelocityWorkspace

Return the velocity field's own scratch workspace, building (and caching on the
field) a fresh zeroed one when absent or sized for fewer threads / a different
radial resolution. The workspace is owned by `𝒰` — never a module global — so
scratch is never shared between solver states. Cross-state sharing previously
let one state's leftover values leak into another's radial-profile kernels,
which made the lmax=2 magnetic-poloidal solve nondeterministic under the full
suite (finite alone, occasionally non-finite in CI).
"""
function _get_or_build_velocity_workspace!(𝒰::SHTnsVelocityFields{T}, nr::Int,
        nthreads::Int = max(Threads.nthreads(), Threads.maxthreadid())) where {T}
    ws = 𝒰.velocity_workspace
    if ws === nothing ||
       length(ws.Pᴾ_profile_real) < nthreads ||
       length(ws.Pᴾ_profile_real[1]) != nr
        ws = create_velocity_workspace(T, nr, nthreads)
        𝒰.velocity_workspace = ws
    end
    return ws
end

"""
    enforce_velocity_boundary_values!(𝒰)

Anchor toroidal and poloidal spectral coefficients to the currently cached
Dirichlet boundary values on the inner and outer radial surfaces.
"""
function enforce_velocity_boundary_values!(𝒰::SHTnsVelocityFields{T}) where {T}
    domain = 𝒰.domain
    config = 𝒰.toroidal.config
    tor_real = parent(𝒰.toroidal.data_real)
    tor_imag = parent(𝒰.toroidal.data_imag)
    pol_real = parent(𝒰.poloidal.data_real)
    pol_imag = parent(𝒰.poloidal.data_imag)

    tor_bc = 𝒰.toroidal.boundary_values
    pol_bc = 𝒰.poloidal.boundary_values

    # Axis 1 of the spectral pencil is the l-slot axis (length lmax+1), NOT the
    # flattened mode list — iterating it as a mode index would visit only the
    # m=0 block. Use the true local mode indices (1:nlm).
    lm_range = local_spectral_mode_indices(config)
    r_range = get_local_range(𝒰.toroidal.pencil, 3)

    has_inner = 1 in r_range && domain.r[1, 4] > 0
    has_outer = domain.N in r_range

    inner_idx = has_inner ? (1 - first(r_range) + 1) : 0
    outer_idx = has_outer ? (domain.N - first(r_range) + 1) : 0

    dirichlet_code = Int(bcs.DIRICHLET)

    for lm_idx in lm_range
        if lm_idx <= 𝒰.toroidal.nlm
            slot = local_spectral_storage_slot(config, lm_idx)
            slot === nothing && continue

            if has_inner && 1 <= inner_idx <= size(tor_real, 3)
                if 𝒰.toroidal.bc_type_inner[lm_idx] == dirichlet_code
                    set_local_spectral_value!(tor_real, slot, inner_idx, tor_bc[1, lm_idx])
                    set_local_spectral_value!(tor_imag, slot, inner_idx, zero(T))
                end
                if 𝒰.poloidal.bc_type_inner[lm_idx] == dirichlet_code
                    set_local_spectral_value!(pol_real, slot, inner_idx, pol_bc[1, lm_idx])
                    set_local_spectral_value!(pol_imag, slot, inner_idx, zero(T))
                end
            end

            if has_outer && 1 <= outer_idx <= size(tor_real, 3)
                if 𝒰.toroidal.bc_type_outer[lm_idx] == dirichlet_code
                    set_local_spectral_value!(tor_real, slot, outer_idx, tor_bc[2, lm_idx])
                    set_local_spectral_value!(tor_imag, slot, outer_idx, zero(T))
                end
                if 𝒰.poloidal.bc_type_outer[lm_idx] == dirichlet_code
                    set_local_spectral_value!(pol_real, slot, outer_idx, pol_bc[2, lm_idx])
                    set_local_spectral_value!(pol_imag, slot, outer_idx, zero(T))
                end
            end
        end
    end

    return 𝒰
end

function compute_vorticity_spectral_full!(𝒰::SHTnsVelocityFields{T},
        domain::RadialDomain,
        ws::VelocityWorkspace{T}) where {T}
    # Same as the threaded version but using provided workspace buffers
    uᵀ_real = parent(𝒰.toroidal.data_real)
    uᵀ_imag = parent(𝒰.toroidal.data_imag)
    uᴾ_real = parent(𝒰.poloidal.data_real)
    uᴾ_imag = parent(𝒰.poloidal.data_imag)
    ζᵀ_real = parent(𝒰.ζᵀ.data_real)
    ζᵀ_imag = parent(𝒰.ζᵀ.data_imag)
    ζᴾ_real = parent(𝒰.ζᴾ.data_real)
    ζᴾ_imag = parent(𝒰.ζᴾ.data_imag)

    config = 𝒰.toroidal.config
    # Axis 1 of the spectral pencil is the l-slot axis (length lmax+1), NOT the
    # flattened mode list — iterating it as a mode index would visit only the
    # m=0 block. Use the true local mode indices (1:nlm).
    lm_range = local_spectral_mode_indices(config)
    r_range = get_local_range(𝒰.toroidal.pencil, 3)
    nr = domain.N

    # The expensive part is radial differentiation, so the loop is organized by
    # spectral mode: gather one `(l,m)` radial profile, differentiate it, then
    # write back the corresponding vorticity profile for that same mode.
    Threads.@threads for lm_idx in lm_range
        if lm_idx <= length(𝒰.l_factors)
            slot = local_spectral_storage_slot(config, lm_idx)
            slot === nothing && continue
            l_factor = 𝒰.l_factors[lm_idx]
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
            ∂ᵣᵣpoloidal_real = ws.∂ᵣᵣpoloidal_real[tid]
            ∂ᵣᵣpoloidal_imag = ws.∂ᵣᵣpoloidal_imag[tid]

            extract_local_radial_profile!(Pᴾ_profile_real, uᴾ_real, slot, nr, r_range)
            extract_local_radial_profile!(Pᴾ_profile_imag, uᴾ_imag, slot, nr, r_range)
            extract_local_radial_profile!(Tᵀ_profile_real, uᵀ_real, slot, nr, r_range)
            extract_local_radial_profile!(Tᵀ_profile_imag, uᵀ_imag, slot, nr, r_range)

            apply_∂r!(∂ᵣᵣpoloidal_real, 𝒰.∂²r, Pᴾ_profile_real)
            apply_∂r!(∂ᵣᵣpoloidal_imag, 𝒰.∂²r, Pᴾ_profile_imag)

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
                        set_local_spectral_value!(ζᵀ_real, slot, local_r, zero(T))
                        set_local_spectral_value!(ζᵀ_imag, slot, local_r, zero(T))
                        set_local_spectral_value!(ζᴾ_real, slot, local_r, zero(T))
                        set_local_spectral_value!(ζᴾ_imag, slot, local_r, zero(T))
                    else
                        r⁻¹ = domain.r[r_idx, 3]
                        r⁻² = domain.r[r_idx, 2]
                        # Solenoidal convention (Stage 2): T_ω = (P'' − λP/r²)/r,
                        # P_ω = −r·T (verified vs the Stage-1 curl projections).
                        set_local_spectral_value!(ζᵀ_real, slot, local_r,
                            r⁻¹ * (∂ᵣᵣpoloidal_real[r_idx] -
                                   l_factor * r⁻² * Pᴾ_profile_real[r_idx]))
                        set_local_spectral_value!(ζᵀ_imag, slot, local_r,
                            r⁻¹ * (∂ᵣᵣpoloidal_imag[r_idx] -
                                   l_factor * r⁻² * Pᴾ_profile_imag[r_idx]))
                        set_local_spectral_value!(ζᴾ_real, slot, local_r,
                            -r * Tᵀ_profile_real[r_idx])
                        set_local_spectral_value!(ζᴾ_imag, slot, local_r,
                            -r * Tᵀ_profile_imag[r_idx])
                    end
                end
            end
        end
    end
end

function _default_velocity_parameters(config::C, domain::RadialDomain;
        geometry::Symbol = :shell) where {C <: SHTnsKitConfig}
    # Geometry CANNOT be inferred from the grid: the ball uses an off-center
    # radial grid whose innermost node r_1 > 0, so `iszero(r_inner)` would
    # always classify it as :shell. Callers that know the geometry must pass
    # it explicitly (GeoDynamoBall constructors pass :ball); the default
    # stays :shell.
    r_inner = domain.r[1, 4]
    r_outer = domain.r[domain.N, 4]
    radius_ratio = geometry === :ball ? 0.0 :
                   (iszero(r_outer) ? 0.0 : Float64(r_inner / r_outer))
    return SolverParameters(
        geometry = geometry,
        nr = domain.N,
        nr_inner = geometry === :ball ? 0 : min(16, domain.N),
        lmax = config.lmax,
        mmax = config.mmax,
        nlat = config.nlat,
        nlon = config.nlon,
        radial_bandwidth = radial_bandwidth(domain),
        radius_ratio = radius_ratio
    )
end

"""
    create_shtns_velocity_fields(T, config, domain, pencils=nothing, pencil_spec=nothing;
                                 geometry=:shell, params)

Allocate and initialize the velocity field container used by solver runtimes.

When `params` is not supplied, defaults are derived from `config`/`domain` —
but the geometry cannot be inferred from the grid (the ball's off-center
radial grid has r_1 > 0), so callers building ball fields without explicit
`params` must pass `geometry = :ball`.

The returned object includes physical-space velocity/vorticity fields, spectral
toroidal-poloidal coefficients, nonlinear history buffers, and cached radial
operators.
"""
function create_shtns_velocity_fields(::Type{T}, config::C,
        outer_core_domain::RadialDomain,
        pencils = nothing, pencil_spec = nothing;
        geometry::Symbol = :shell,   # used only when `params` is defaulted
        params::SolverParameters = _default_velocity_parameters(
            config, outer_core_domain; geometry)) where {
        T, C <: SHTnsKitConfig}
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
    velocity = create_shtns_vector_field(T, config, outer_core_domain, pencils)
    vorticity = create_shtns_vector_field(T, config, outer_core_domain, pencils)

    # Spectral fields
    toroidal = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    poloidal = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    ζᵀ = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    ζᴾ = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    nl_toroidal = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    nl_poloidal = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    prev_nl_toroidal = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    prev_nl_poloidal = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)

    # Work arrays
    work_tor = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    work_pol = create_shtns_spectral_field(T, config, outer_core_domain, pencil_spec)
    work_physical = create_shtns_vector_field(T, config, outer_core_domain, pencils)
    advection_physical = create_shtns_vector_field(T, config, outer_core_domain, pencils)

    # Pre-compute l(l+1) factors
    l_factors = T[l * (l + 1) for l in config.l_values]

    # Pre-compute Coriolis factors (sin(θ) and cos(θ))
    coriolis_factors = zeros(T, 2, config.nlat)
    for i in 1:config.nlat
        coriolis_factors[1, i] = sin(config.theta_grid[i])
        coriolis_factors[2, i] = cos(config.theta_grid[i])
    end

    # Create radial derivative matrices
    ∂r = create_derivative_matrix(T, 1, outer_core_domain)
    ∂²r = create_derivative_matrix(T, 2, outer_core_domain)
    laplacian_matrix = create_radial_laplacian(T, outer_core_domain)

    # Create transpose plans for efficient data movement
    transpose_plans = create_transpose_plans(pencils)

    params_snapshot = deepcopy(params)
    boundary_condition_set = nothing
    boundary_cache = bcs.BoundaryInterpolationCache(T)
    boundary_time_index = Ref{Int}(1)

    return SHTnsVelocityFields(velocity, vorticity, toroidal, poloidal,
        ζᵀ, ζᴾ,
        nl_toroidal, nl_poloidal, prev_nl_toroidal, prev_nl_poloidal,
        work_tor, work_pol, work_physical,
        advection_physical,
        l_factors, coriolis_factors,
        ∂r, ∂²r, laplacian_matrix,
        config,
        outer_core_domain,
        params_snapshot,
        boundary_condition_set, boundary_cache, boundary_time_index,
        nothing)  # velocity_workspace — built lazily, per-field (not shared)
end

# =============================
# Main nonlinear computation
# =============================
function compute_velocity_nonlinear!(𝒰::SHTnsVelocityFields{T},
        temp_field, comp_field, mag_field,
        outer_core_domain::RadialDomain;
        geometry::Symbol = 𝒰.parameters.geometry) where {T}
    # Zero work arrays once
    zero_velocity_work_arrays!(𝒰)

    # The velocity nonlinear path is staged to minimize layout thrash:
    # spectral velocity -> physical velocity, spectral curl -> physical
    # vorticity, physical nonlinear products, then one vector analysis back to
    # toroidal/poloidal nonlinear coefficients.

    # Step 1: Use enhanced vector synthesis with automatic transpose handling
    shtnskit_vector_synthesis!(𝒰.toroidal, 𝒰.poloidal, 𝒰.velocity; domain = outer_core_domain)

    # Step 2: Compute vorticity in spectral space with enhanced derivative computation
    compute_vorticity_spectral_full!(𝒰, outer_core_domain)

    # Step 3: Transform vorticity to physical space with batched operations
    shtnskit_vector_synthesis!(𝒰.ζᵀ, 𝒰.ζᴾ, 𝒰.vorticity; domain = outer_core_domain)

    # Step 4: Compute all nonlinear terms with enhanced memory access patterns
    compute_all_nonlinear_terms!(𝒰, temp_field, comp_field, mag_field, outer_core_domain)

    # Step 5: Use enhanced vector analysis with efficient data layout
    # (geometry-blind since the ball grid has no r=0 node — off-center grid).
    # Pass the domain so vector_physical_to_spectral! runs the Stage-2 solenoidal
    # recovery (poloidal from the RADIAL force component): without it the radial
    # buoyancy / Lorentz force is silently dropped and the poloidal RHS reduces to
    # the raw spheroidal scalar — the "convection-can't-start-from-rest" bug.
    shtnskit_vector_analysis!(𝒰.advection_physical, 𝒰.nl_toroidal, 𝒰.nl_poloidal;
        domain = outer_core_domain)
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
#   - Current density: j = ∇×B (in physics/magnetic/field.jl)
#   - Induction curl: ∇×(u×B) (in physics/magnetic/field.jl)
#
# The formula arises from the identity ∇×∇×A = ∇(∇·A) - ∇²A combined with
# the spherical harmonic eigenvalue -l(l+1)/r² for the angular Laplacian.
#
# =================================================
using Base.Threads
function compute_vorticity_spectral_full!(𝒰::SHTnsVelocityFields{T},
        domain::RadialDomain) where {T}
    # Use this field's own scratch workspace (built lazily, never shared across
    # states), so the radial-profile buffers can't carry another state's values.
    ws = _get_or_build_velocity_workspace!(𝒰, domain.N)
    return compute_vorticity_spectral_full!(𝒰, domain, ws)
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

E·∂ũ/∂τ = E(ũ×ω̃) - ẑ×ũ - ∇p̃*
          + (Pm/Pr)·Ra·T̃·r·r̂ + (Pm/Sc)·Ra_C·C̃·r·r̂
          + (1/Pm)(∇×B̃)×B̃ + E∇²ũ

where:
  - τ = L²/η (magnetic diffusion time scaling)
  - E = ν/(2ΩL²) is the Ekman number
  - r factor in buoyancy represents linear gravity profile g(r) ∝ r

# Implementation Notes

The explicit RHS entering the time integrator is:
RHS = E·(ũ×ω̃) - (ẑ×ũ) + (Pm/Pr)·Ra·T̃·r·r̂ + (Pm/Sc)·Ra_C·C̃·r·r̂ + (1/Pm)(∇×B̃)×B̃

Coefficients (magnetic diffusion time scaling, mass coeff = E on du/dt):
  - Advection: E
  - Coriolis: 1 (no scaling)
  - Thermal buoyancy: (Pm/Pr) * Ra * r (with radial factor)
  - Compositional buoyancy: (Pm/Sc) * Ra_C * r (with radial factor)
  - Lorentz: 1/Pm

Viscous diffusion is treated implicitly with coefficient E (Ekman number).
Mass coefficient E is applied in the time-stepping matrices.
"""
function compute_all_nonlinear_terms!(𝒰::SHTnsVelocityFields{T},
        temp_field, comp_field, mag_field,
        domain::RadialDomain) where {T}
    params = 𝒰.parameters
    Ek = params.Ek
    Pm = params.Pm
    Pr = params.Pr
    Sc = params.Sc
    Ra = params.Ra
    RaC = params.RaC

    # This fused kernel assembles every explicit momentum forcing term in one
    # physical-space sweep so advection, Coriolis, buoyancy, and Lorentz
    # contributions share the same cache-friendly field loads.
    # Compute all forces using magnetic diffusion time scaling.
    # The momentum equation (mass coeff Ek on du/dt):
    #   E·du/dt = Ek·(u×ω) - (z×u) + buoyancy + (1/Pm)·(j×B) + E·∇²u
    # Mass coefficient Ek is applied in the time-stepping matrices.

    advection_coeff = T(Ek)

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
    θ_range = range_local(config.pencils.r, 1)  # global theta indices for this rank

    # Main fused computation loop with enhanced indexing (parallel over r-slices)
    # Advection has coefficient E, Coriolis has coefficient 1
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
            @simd for i in 1:local_size[1]
                # Map local theta index to global for Coriolis factor lookup
                theta_idx_global = θ_range[i]
                sin_theta = 𝒰.coriolis_factors[1, theta_idx_global]
                cos_theta = 𝒰.coriolis_factors[2, theta_idx_global]
                linear_idx = i + (j-1)*local_size[1] + (k-1)*local_size[1]*local_size[2]

                if linear_idx <= length(vᵣ)
                    # Load velocity and vorticity components
                    u_r = vᵣ[linear_idx]
                    u_θ = vθ[linear_idx]
                    u_φ = vφ[linear_idx]

                    ω_r = ζᵣ[linear_idx]
                    ω_θ = ζθ[linear_idx]
                    ω_φ = ζφ[linear_idx]

                    # Advection: E * (u × ζ)
                    adv_r_val = adv_coeff * (u_θ * ω_φ - u_φ * ω_θ)
                    adv_θ_val = adv_coeff * (u_φ * ω_r - u_r * ω_φ)
                    adv_φ_val = adv_coeff * (u_r * ω_θ - u_θ * ω_r)

                    # Coriolis: -(z × u), coefficient = 1 (Fortran convention)
                    zhat_cross_r = -sin_theta * u_φ
                    zhat_cross_θ = -cos_theta * u_φ
                    zhat_cross_φ = cos_theta * u_θ + sin_theta * u_r
                    cor_r = -zhat_cross_r
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
    # Buoyancy: (Pm/Pr)·Ra (thermal), (Pm/Sc)·Ra_C (compositional)
    # Fortran coefficient form: PrT * qRaT * r, PrC * qRaC * r
    if temp_field !== nothing
        buoyancy_factor = (Pm / Pr) * Ra
        add_thermal_buoyancy_force!(adv_r, temp_field, buoyancy_factor, domain)
    end

    if comp_field !== nothing
        comp_factor = (Pm / Sc) * RaC
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
function add_thermal_buoyancy_force!(force_r::AbstractArray{T, 3},
        scalar_field, factor::Float64,
        domain::RadialDomain) where {T}
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
    config = scalar_field isa SHTnsPhysField ? scalar_field.config :
             scalar_field.temperature.config
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
function add_buoyancy_force!(force_r::AbstractArray{T, 3},
        comp_field, factor::Float64,
        domain::RadialDomain) where {T}
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
    config = comp_field isa SHTnsPhysField ? comp_field.config :
             comp_field.composition.config
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
        domain::RadialDomain) where {T}
    Pm = 𝒰.parameters.Pm

    # Compute Lorentz force F = (1/Pm)(∇ × B) × B with vectorization.

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

    # Fused loop for j × B (Lorentz coefficient = 1/Pm, Fortran convention)
    lorentz_coeff = T(1.0 / Pm)
    @inbounds @simd for idx in eachindex(j_r)
        if idx <= length(B_r)
            # Add Lorentz force to existing forces
            adv_r[idx] += lorentz_coeff * (j_θ[idx] * B_φ[idx] - j_φ[idx] * B_θ[idx])
            adv_θ[idx] += lorentz_coeff * (j_φ[idx] * B_r[idx] - j_r[idx] * B_φ[idx])
            adv_φ[idx] += lorentz_coeff * (j_r[idx] * B_θ[idx] - j_θ[idx] * B_r[idx])
        end
    end
end

# Note: Boundary condition functions moved to src/bcs/velocity_bc.jl

# ===========================================
# Helper functions for radial operations
# ===========================================
function extract_local_radial_profile(data::AbstractArray{T, 3}, slot::CartesianIndex{2},
        nr::Int, r_range) where {T}
    profile = zeros(T, nr)

    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(data, 3) && r_idx <= nr
            profile[r_idx] = local_spectral_value(data, slot, local_r)
        end
    end

    return profile
end

"""
    extract_local_radial_profile!(profile, data, slot, nr, r_range)

In-place version to avoid allocations; writes the local radial line into
`profile` for the given local spectral `slot` using the provided `r_range`.
"""
function extract_local_radial_profile!(profile::Vector{T}, data::AbstractArray{T, 3},
        slot::CartesianIndex{2}, nr::Int, r_range) where {T}
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(data, 3) && r_idx <= nr && r_idx <= length(profile)
            profile[r_idx] = local_spectral_value(data, slot, local_r)
        end
    end
    return profile
end

function store_local_radial_profile!(data::AbstractArray{T, 3}, profile::Vector{T},
        slot::CartesianIndex{2}, r_range) where {T}
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(data, 3) && r_idx <= length(profile)
            set_local_spectral_value!(data, slot, local_r, profile[r_idx])
        end
    end
end

function apply_derivative_local(matrix::BandedMatrix{T}, field::Vector{T}) where {T}
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

#     N = outer_core_domain.N
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
#                     r⁻² = outer_core_domain.r[i, 2]
#                     operator[i, j] -= l_factor * r⁻²
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
"""
    compute_kinetic_energy(velocity_fields, domain)

Compute the global kinetic energy of the current velocity state from its
spectral toroidal-poloidal coefficients.
"""
function compute_kinetic_energy(𝒰::SHTnsVelocityFields{T}, outer_core_domain::RadialDomain) where {T}
    # Compute kinetic energy with configuration-aware integration

    tor_real = parent(𝒰.toroidal.data_real)
    tor_imag = parent(𝒰.toroidal.data_imag)
    pol_real = parent(𝒰.poloidal.data_real)
    pol_imag = parent(𝒰.poloidal.data_imag)

    local_energy = zero(Float64)

    # Use configuration pencils for consistent range access
    # CRITICAL: Both lm_range and r_range must come from the SAME pencil (spec)
    # since spectral field data is distributed using pencils.spec
    config = 𝒰.toroidal.config
    lm_range = local_spectral_mode_indices(config)
    r_range = range_local(config.pencils.spec, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= 𝒰.toroidal.nlm
            slot = local_spectral_storage_slot(config, lm_idx)
            slot === nothing && continue
            l_factor = 𝒰.l_factors[lm_idx]

            # Spectral kinetic energy weight: l(l+1) for toroidal-poloidal decomposition.
            # Parseval factor: m>0 coefficients carry double the energy of m=0 (the
            # -m conjugate partner is not stored in the m>=0-only real-field layout).
            mweight = (config.m_values[lm_idx] == 0) ? 1.0 : 2.0
            weight = Float64(l_factor) * mweight

            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    # Include radial weight for spherical integration
                    r = outer_core_domain.r[r_idx, 4]
                    r_weight = r^2 * outer_core_domain.integration_weights[r_idx]

                    local_energy += weight * r_weight *
                                    (
                                        local_spectral_value(tor_real, slot, local_r)^2 +
                                        local_spectral_value(tor_imag, slot, local_r)^2 +
                                        local_spectral_value(pol_real, slot, local_r)^2 +
                                        local_spectral_value(pol_imag, slot, local_r)^2
                                    )
                end
            end
        end
    end

    # Global sum
    return 0.5 * MPI.Allreduce(local_energy, MPI.SUM, get_comm())
end

"""
    compute_reynolds_stress(velocity_fields)

Compute the volume-averaged Reynolds-stress tensor components from the current
physical velocity field.
"""
function compute_reynolds_stress(𝒰::SHTnsVelocityFields{T}) where {T}
    vᵣ = parent(𝒰.velocity.r_component.data)
    vθ = parent(𝒰.velocity.θ_component.data)
    vφ = parent(𝒰.velocity.φ_component.data)
    config = 𝒰.velocity.r_component.config
    local_size = size(vᵣ)
    θ_range = range_local(config.pencils.r, 1)
    r_range = range_local(config.pencils.r, 3)
    dφ = 2π / config.nlon

    local_weight = 0.0
    local_rr = 0.0
    local_θθ = 0.0
    local_φφ = 0.0
    local_rθ = 0.0
    local_rφ = 0.0
    local_θφ = 0.0

    @inbounds for k in 1:local_size[3]
        r_idx = first(r_range) + k - 1
        if r_idx > 𝒰.domain.N
            continue
        end
        radial_weight = 𝒰.domain.r[r_idx, 4]^2 * 𝒰.domain.integration_weights[r_idx]
        for j in 1:local_size[2]
            for i in 1:local_size[1]
                θ_idx = θ_range[i]
                weight = radial_weight * config.gauss_weights[θ_idx] * dφ
                linear_idx = i + (j - 1) * local_size[1] +
                             (k - 1) * local_size[1] * local_size[2]

                u_r = vᵣ[linear_idx]
                u_θ = vθ[linear_idx]
                u_φ = vφ[linear_idx]

                local_weight += weight
                local_rr += weight * u_r * u_r
                local_θθ += weight * u_θ * u_θ
                local_φφ += weight * u_φ * u_φ
                local_rθ += weight * u_r * u_θ
                local_rφ += weight * u_r * u_φ
                local_θφ += weight * u_θ * u_φ
            end
        end
    end

    global_weight = MPI.Allreduce(local_weight, MPI.SUM, get_comm())
    R_rr = MPI.Allreduce(local_rr, MPI.SUM, get_comm()) / global_weight
    R_θθ = MPI.Allreduce(local_θθ, MPI.SUM, get_comm()) / global_weight
    R_φφ = MPI.Allreduce(local_φφ, MPI.SUM, get_comm()) / global_weight
    R_rθ = MPI.Allreduce(local_rθ, MPI.SUM, get_comm()) / global_weight
    R_rφ = MPI.Allreduce(local_rφ, MPI.SUM, get_comm()) / global_weight
    R_θφ = MPI.Allreduce(local_θφ, MPI.SUM, get_comm()) / global_weight

    return (R_rr, R_θθ, R_φφ, R_rθ, R_rφ, R_θφ)
end

# ================================================================================
# Utility functions
# ================================================================================
function zero_velocity_work_arrays!(𝒰::SHTnsVelocityFields{T}) where {T}
    z = zero(T)
    fill!(parent(𝒰.work_tor.data_real), z)
    fill!(parent(𝒰.work_tor.data_imag), z)
    fill!(parent(𝒰.work_pol.data_real), z)
    fill!(parent(𝒰.work_pol.data_imag), z)
    fill!(parent(𝒰.work_physical.r_component.data), z)
    fill!(parent(𝒰.work_physical.θ_component.data), z)
    fill!(parent(𝒰.work_physical.φ_component.data), z)
    fill!(parent(𝒰.advection_physical.r_component.data), z)
    fill!(parent(𝒰.advection_physical.θ_component.data), z)
    fill!(parent(𝒰.advection_physical.φ_component.data), z)
    fill!(parent(𝒰.ζᵀ.data_real), z)
    fill!(parent(𝒰.ζᵀ.data_imag), z)
    fill!(parent(𝒰.ζᴾ.data_real), z)
    fill!(parent(𝒰.ζᴾ.data_imag), z)
end

function scale_field!(field::SHTnsVectorField{T}, factor::Float64) where {T}
    # Scale all components of a vector field
    parent(field.r_component.data) .*= factor
    parent(field.θ_component.data) .*= factor
    parent(field.φ_component.data) .*= factor
end

function add_vector_fields!(dest::SHTnsVectorField{T}, source::SHTnsVectorField{T}) where {T}
    # Add source to destination with vectorized operations
    parent(dest.r_component.data) .+= parent(source.r_component.data)
    parent(dest.θ_component.data) .+= parent(source.θ_component.data)
    parent(dest.φ_component.data) .+= parent(source.φ_component.data)
end

# ================================================================================
# Enhanced utility functions using pencil decomposition and SHTns integration
# ================================================================================

"""
    optimize_velocity_memory_layout!(𝒰::SHTnsVelocityFields{T}) where T

Optimize memory layout for better cache performance using pencil topology
"""
function optimize_velocity_memory_layout!(𝒰::SHTnsVelocityFields{T}) where {T}
    # Use transpose plans for optimal data layout based on upcoming operations
    config = 𝒰.toroidal.config

    # Use transpose plans if available
    plans = config.transpose_plans
    if !isempty(plans) && haskey(plans, :r_to_spec)
        transpose_with_timer!(𝒰.work_tor.data_real, 𝒰.toroidal.data_real,
            plans[:r_to_spec], "toroidal_layout_opt")
        transpose_with_timer!(𝒰.work_pol.data_real, 𝒰.poloidal.data_real,
            plans[:r_to_spec], "poloidal_layout_opt")
    end
end

"""
    validate_velocity_configuration(𝒰::SHTnsVelocityFields{T}, config::C) where {T,C<:SHTnsKitConfig}

Validate velocity field configuration consistency with SHTns setup
"""
function validate_velocity_configuration(𝒰::SHTnsVelocityFields{T}, config::C) where {
        T, C <: SHTnsKitConfig}
    errors = String[]

    # Check field dimensions match config
    local_slot_capacity = size(𝒰.toroidal.data_real, 1) * size(𝒰.toroidal.data_real, 2)
    local_mode_count = length(local_spectral_mode_indices(config))
    if local_mode_count > local_slot_capacity
        push!(errors, "Toroidal field local slot capacity is smaller than owned spectral mode count")
    end

    # Check that l_factors are consistent
    if length(𝒰.l_factors) != config.nlm
        push!(errors, "l_factors length mismatch with config.nlm")
    end

    # Validate pencil topology consistency
    local_modes = local_spectral_mode_indices(config)
    if !isempty(local_modes) && maximum(local_modes) > config.nlm
        push!(errors, "Owned spectral mode index exceeds config.nlm")
    end

    # Note: Transform manager checks removed - now handled by SHTnsKit directly

    if !isempty(errors)
        @warn "Velocity configuration validation failed:\n" * join(errors, "\n")
        return false
    end

    return true
end
