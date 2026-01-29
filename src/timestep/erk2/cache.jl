# ================================================================================
# ERK2 Cache Structure and Creation Functions
# ================================================================================

"""
    ERK2Cache{T}

Cached data structure for Exponential 2nd Order Runge-Kutta method.
Stores precomputed matrix exponentials and φ functions for each spherical harmonic mode.
"""
struct ERK2Cache{T}
    dt::Float64
    diffusivity::Float64
    nr::Int
    l_values::Vector{Int}

    # Matrix exponentials: exp(dt/2 * A_l) and exp(dt * A_l)
    E_half::Vector{Matrix{T}}     # exp(dt/2 * A_l) per l
    E_full::Vector{Matrix{T}}     # exp(dt * A_l) per l

    # φ functions for ERK2 (both φ1 and φ2 needed for correct formula)
    phi1_half::Vector{Matrix{T}}  # φ1(dt/2 * A_l) per l
    phi1_full::Vector{Matrix{T}}  # φ1(dt * A_l) per l
    phi2_full::Vector{Matrix{T}}  # φ2(dt * A_l) per l

    # Krylov method parameters
    use_krylov::Bool
    krylov_m::Int
    krylov_tol::Float64

    # MPI-aware caching for distributed operations
    mpi_consistent::Bool
end

# Backward-compatible constructor for legacy cache bundles (pre-diffusivity metadata)
function ERK2Cache{T}(dt::Float64, l_values::Vector{Int},
                      E_half::Vector{Matrix{T}}, E_full::Vector{Matrix{T}},
                      phi1_half::Vector{Matrix{T}}, phi1_full::Vector{Matrix{T}},
                      phi2_full::Vector{Matrix{T}}, use_krylov::Bool,
                      krylov_m::Int, krylov_tol::Float64, mpi_consistent::Bool) where T
    nr = isempty(E_half) ? 0 : size(E_half[1], 1)
    return ERK2Cache{T}(dt, NaN, nr, l_values, E_half, E_full,
                        phi1_half, phi1_full, phi2_full,
                        use_krylov, krylov_m, krylov_tol, mpi_consistent)
end

ERK2Cache(args...) = ERK2Cache{Float64}(args...)

const ERK2_DIAGNOSTICS_ENABLED = Ref(false)
const ERK2_DIAGNOSTICS_INTERVAL = Ref(1)

function set_erk2_diagnostics_interval!(interval::Int)
    interval <= 0 && error("ERK2 diagnostics interval must be positive, got $interval")
    ERK2_DIAGNOSTICS_INTERVAL[] = interval
    return interval
end

function enable_erk2_diagnostics!(; interval::Int=ERK2_DIAGNOSTICS_INTERVAL[])
    set_erk2_diagnostics_interval!(interval)
    ERK2_DIAGNOSTICS_ENABLED[] = true
    return nothing
end

disable_erk2_diagnostics!() = (ERK2_DIAGNOSTICS_ENABLED[] = false; nothing)
erk2_diagnostics_enabled() = ERK2_DIAGNOSTICS_ENABLED[]
erk2_diagnostics_interval() = ERK2_DIAGNOSTICS_INTERVAL[]

let env_val = get(ENV, "GEODYNAMO_ERK2_DIAGNOSTICS", nothing)
    if env_val !== nothing
        enable = startswith(lowercase(strip(env_val)), "t") ||
                 lowercase(strip(env_val)) in ("1", "yes", "on")
        if enable
            interval_val = get(ENV, "GEODYNAMO_ERK2_DIAGNOSTICS_INTERVAL", "")
            interval = try
                isempty(interval_val) ? 1 : parse(Int, strip(interval_val))
            catch
                1
            end
            try
                enable_erk2_diagnostics!(interval=interval)
            catch e
                @warn "Failed to enable ERK2 diagnostics from environment: $e"
            end
        end
    end
end

"""
    create_erk2_cache(config, domain, diffusivity, dt; use_krylov=false, m=20, tol=1e-8, bc_spec=nothing)

Create ERK2 cache with precomputed matrix functions for all spherical harmonic modes.

Boundary rows of A are zeroed only for l=0 (where the spherical Laplacian has a null
space, making A singular). For l≥1, the full operator is retained (non-singular due to
the -l(l+1)/r² term), enabling accurate LU-based phi1/phi2 computation and O(h³)
enforcement corrections per step.
"""
function create_erk2_cache(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                          diffusivity::Float64, dt::Float64;
                          use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8,
                          bc_spec::Union{ERK2BoundarySpec{T}, Nothing}=nothing) where T

    lap = create_radial_laplacian(domain)
    nr = domain.N
    r_inv2 = @views domain.r[1:nr, 2]
    lvals = unique(config.l_values)

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    if get_rank() == 0
        @info "Creating ERK2 cache for $(length(lvals)) l-modes with $(use_krylov ? "Krylov" : "dense") methods"
    end

    for l in lvals
        # Build A_l = diffusivity * (d²/dr² + (2/r)d/dr - l(l+1)/r²)
        Adata = diffusivity .* lap.data
        Adense = banded_to_dense(BandedMatrix(Adata, i_KL, nr))
        lfac = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            Adense[n, n] -= diffusivity * lfac * r_inv2[n]
        end

        # Zero boundary rows ONLY for l=0 where the spherical Laplacian
        # (d²/dr² + 2/r d/dr) has a null space (constant function), making A singular.
        # For l≥1, the -l(l+1)/r² term ensures A is non-singular, so the LU-based
        # phi1/phi2 computation works correctly and gives O(h³) enforcement corrections.
        # Zeroing for l≥1 would make A singular, forcing fallback to the Taylor series
        # and degrading boundary accuracy from O(h³) to O(h) per step.
        if l == 0
            Adense[1, :] .= zero(Float64)
            Adense[nr, :] .= zero(Float64)
        end

        if use_krylov
            # For large problems, we'll use Krylov methods during timestepping
            # Store only the operator for action-based computation
            push!(E_half, Adense)  # Store A for later Krylov action
            push!(E_full, Adense)
            push!(phi1_half, Adense)
            push!(phi1_full, Adense)
            push!(phi2_full, Adense)
        else
            # Dense computation of matrix functions
            Adt_half = (dt/2) .* Adense
            Adt_full = dt .* Adense

            # Compute exp(dt/2 * A) and exp(dt * A)
            E_half_l = exp(Adt_half)
            E_full_l = exp(Adt_full)
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))

            # Compute φ1 functions: φ1(z) = (exp(z) - I) / z
            phi1_half_l = compute_phi1_function(Adt_half, E_half_l)
            phi1_full_l = compute_phi1_function(Adt_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            # Compute φ2 function: φ2(z) = (exp(z) - I - z) / z²
            phi2_full_l = compute_phi2_function(Adt_full, E_full_l; l=l)
            push!(phi2_full, Matrix{T}(phi2_full_l))

        end
    end

    # Ensure MPI consistency
    MPI.Barrier(get_comm())

    return ERK2Cache{T}(dt, diffusivity, nr, lvals, E_half, E_full, phi1_half, phi1_full,
                       phi2_full, use_krylov, m, tol, true)
end

"""
    create_erk2_cache_magnetic_poloidal(T, config, domain, diffusivity, dt; use_krylov=false, m=20, tol=1e-8)

Create ERK2 cache for magnetic poloidal field with insulating BCs embedded in the matrix A.

This matches DD_2DCODE's approach where the matrix has:
- Inner boundary row: (∂/∂r - l/r) operator → insulating interior
- Outer boundary row: (∂/∂r + (l+1)/r) operator → insulating exterior

Embedding BCs in A ensures exp(dt*A) automatically preserves solutions satisfying the
insulating constraints, rather than requiring post-evolution correction.
"""
function create_erk2_cache_magnetic_poloidal(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                                              diffusivity::Float64, dt::Float64;
                                              use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    lap = create_radial_laplacian(domain)
    d1 = create_derivative_matrix(T, 1, domain)
    nr = domain.N
    bw = i_KL
    r_inv2 = @views domain.r[1:nr, 2]
    r_inv = @views domain.r[1:nr, 3]  # 1/r values
    lvals = unique(config.l_values)

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    if get_rank() == 0
        @info "Creating ERK2 cache for magnetic poloidal with insulating BCs embedded"
    end

    for l in lvals
        # Build A_l = diffusivity * (d²/dr² + (2/r)d/dr - l(l+1)/r²)
        Adata = diffusivity .* lap.data
        Adense = banded_to_dense(BandedMatrix(Adata, bw, nr))
        lfac = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            Adense[n, n] -= diffusivity * lfac * r_inv2[n]
        end

        # Zero boundary rows before embedding BCs
        Adense[1, :] .= zero(T)
        Adense[nr, :] .= zero(T)

        # Embed insulating BC at inner boundary: (∂/∂r - l/r)P = 0
        # Copy first derivative row and subtract l/r on diagonal
        for j in max(1, 1 - bw):min(nr, 1 + bw)
            band_idx = bw + 1 + 1 - j
            if 1 <= band_idx <= 2 * bw + 1
                Adense[1, j] = T(d1.data[band_idx, j])
            end
        end
        Adense[1, 1] -= T(l) * r_inv[1]

        # Embed insulating BC at outer boundary: (∂/∂r + (l+1)/r)P = 0
        # Copy first derivative row and add (l+1)/r on diagonal
        for j in max(1, nr - bw):min(nr, nr + bw)
            band_idx = bw + 1 + nr - j
            if 1 <= band_idx <= 2 * bw + 1
                Adense[nr, j] = T(d1.data[band_idx, j])
            end
        end
        Adense[nr, nr] += T(l + 1) * r_inv[nr]

        if use_krylov
            push!(E_half, Adense)
            push!(E_full, Adense)
            push!(phi1_half, Adense)
            push!(phi1_full, Adense)
            push!(phi2_full, Adense)
        else
            # Dense computation of matrix functions
            Adt_half = (dt / 2) .* Adense
            Adt_full = dt .* Adense

            # Compute exp(dt/2 * A) and exp(dt * A)
            E_half_l = exp(Adt_half)
            E_full_l = exp(Adt_full)
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))

            # Compute φ1 functions
            phi1_half_l = compute_phi1_function(Adt_half, E_half_l)
            phi1_full_l = compute_phi1_function(Adt_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            # Compute φ2 function
            phi2_full_l = compute_phi2_function(Adt_full, E_full_l; l=l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
        end
    end

    MPI.Barrier(get_comm())

    return ERK2Cache{T}(dt, diffusivity, nr, lvals, E_half, E_full, phi1_half, phi1_full,
                        phi2_full, use_krylov, m, tol, true)
end

"""
    create_erk2_cache_magnetic_toroidal(T, config, domain, diffusivity, dt; use_krylov=false, m=20, tol=1e-8)

Create ERK2 cache for magnetic toroidal field with insulating BCs embedded in the matrix A.

This matches DD_2DCODE's approach where the matrix has:
- Inner boundary row: identity → BT = 0
- Outer boundary row: identity → BT = 0

For Dirichlet BCs (BT = 0), we zero the boundary rows except for the diagonal (identity),
which makes exp(dt*A)|_boundary = identity, preserving BT = 0.
"""
function create_erk2_cache_magnetic_toroidal(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                                              diffusivity::Float64, dt::Float64;
                                              use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    lap = create_radial_laplacian(domain)
    nr = domain.N
    bw = i_KL
    r_inv2 = @views domain.r[1:nr, 2]
    lvals = unique(config.l_values)

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    if get_rank() == 0
        @info "Creating ERK2 cache for magnetic toroidal with Dirichlet BCs embedded"
    end

    for l in lvals
        # Build A_l = diffusivity * (d²/dr² + (2/r)d/dr - l(l+1)/r²)
        Adata = diffusivity .* lap.data
        Adense = banded_to_dense(BandedMatrix(Adata, bw, nr))
        lfac = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            Adense[n, n] -= diffusivity * lfac * r_inv2[n]
        end

        # Zero boundary rows (Dirichlet: BT = 0)
        # This makes exp(dt*A)|_boundary = identity, preserving BT = 0
        Adense[1, :] .= zero(T)
        Adense[nr, :] .= zero(T)

        if use_krylov
            push!(E_half, Adense)
            push!(E_full, Adense)
            push!(phi1_half, Adense)
            push!(phi1_full, Adense)
            push!(phi2_full, Adense)
        else
            Adt_half = (dt / 2) .* Adense
            Adt_full = dt .* Adense

            E_half_l = exp(Adt_half)
            E_full_l = exp(Adt_full)
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))

            phi1_half_l = compute_phi1_function(Adt_half, E_half_l)
            phi1_full_l = compute_phi1_function(Adt_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            phi2_full_l = compute_phi2_function(Adt_full, E_full_l; l=l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
        end
    end

    MPI.Barrier(get_comm())

    return ERK2Cache{T}(dt, diffusivity, nr, lvals, E_half, E_full, phi1_half, phi1_full,
                        phi2_full, use_krylov, m, tol, true)
end

"""
    create_erk2_cache_scalar(T, config, domain, diffusivity, dt, i_bc; use_krylov=false, m=20, tol=1e-8)

Create ERK2 cache for scalar fields (temperature or composition) with boundary rows
zeroed in the matrix A.

Boundary condition types (matching DD_2DCODE tmp_bc_T / cmp_bc_C):
- i_bc = 1: Dirichlet-Dirichlet (fixed value both boundaries)
- i_bc = 2: Dirichlet-Neumann (fixed value inner, fixed flux outer)
- i_bc = 3: Neumann-Dirichlet (fixed flux inner, fixed value outer)
- i_bc = 4: Neumann-Neumann (fixed flux both, l=0 inner uses Dirichlet)

For exponential integrators, boundary rows are always zeroed:
- exp(dt*A)[boundary,:] = [1, 0, ..., 0] → preserves boundary value
- Actual BC enforcement (Dirichlet/Neumann) is done via post-processing
  in erk2_prepare_field! and erk2_finalize_field! using enforce_erk2_bc!
"""
function create_erk2_cache_scalar(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                                   diffusivity::Float64, dt::Float64, i_bc::Int;
                                   use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    lap = create_radial_laplacian(domain)
    d1 = create_derivative_matrix(T, 1, domain)
    nr = domain.N
    bw = i_KL
    r_inv2 = @views domain.r[1:nr, 2]
    lvals = unique(config.l_values)

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    bc_desc = ["DD", "DN", "ND", "NN"][clamp(i_bc, 1, 4)]
    if get_rank() == 0
        @info "Creating ERK2 cache for scalar field with embedded BCs (type=$bc_desc, ν=$diffusivity)"
    end

    for l in lvals
        # Build A_l = diffusivity * (d²/dr² + (2/r)d/dr - l(l+1)/r²)
        Adata = diffusivity .* lap.data
        Adense = banded_to_dense(BandedMatrix(Adata, bw, nr))
        lfac = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            Adense[n, n] -= diffusivity * lfac * r_inv2[n]
        end

        # Zero boundary rows for exponential integrator
        # For exp(dt*A), zeroing the boundary row means:
        #   exp(dt*A)[boundary,:] = [1, 0, 0, ..., 0] (or [..., 0, 0, 1])
        # This preserves the boundary value during exponential evolution.
        # Actual BC enforcement is done via post-processing (enforce_erk2_bc!).
        #
        # This differs from implicit schemes where:
        #   Dirichlet → identity row (diagonal=1)
        #   Neumann → derivative row
        # For exponential integrators, we always zero boundary rows and rely
        # on post-processing to enforce the constraint.
        Adense[1, :] .= zero(T)
        Adense[nr, :] .= zero(T)

        # Note: BC type (Dirichlet/Neumann) and special l=0 handling for i_bc=4
        # are handled by the bc_spec post-processing in erk2_prepare_field! and
        # erk2_finalize_field! via enforce_erk2_bc!.

        if use_krylov
            push!(E_half, Adense)
            push!(E_full, Adense)
            push!(phi1_half, Adense)
            push!(phi1_full, Adense)
            push!(phi2_full, Adense)
        else
            Adt_half = (dt / 2) .* Adense
            Adt_full = dt .* Adense

            E_half_l = exp(Adt_half)
            E_full_l = exp(Adt_full)
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))

            phi1_half_l = compute_phi1_function(Adt_half, E_half_l)
            phi1_full_l = compute_phi1_function(Adt_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            phi2_full_l = compute_phi2_function(Adt_full, E_full_l; l=l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
        end
    end

    MPI.Barrier(get_comm())

    return ERK2Cache{T}(dt, diffusivity, nr, lvals, E_half, E_full, phi1_half, phi1_full,
                        phi2_full, use_krylov, m, tol, true)
end

# Convenience aliases for temperature and composition
"""
    create_erk2_cache_temperature(T, config, domain, diffusivity, dt, i_tmp_bc; kwargs...)

Create ERK2 cache for temperature field with embedded BCs.
Wrapper around create_erk2_cache_scalar with temperature-specific defaults.
"""
create_erk2_cache_temperature(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                               diffusivity::Float64, dt::Float64, i_tmp_bc::Int;
                               use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T =
    create_erk2_cache_scalar(T, config, domain, diffusivity, dt, i_tmp_bc;
                              use_krylov=use_krylov, m=m, tol=tol)

"""
    create_erk2_cache_composition(T, config, domain, diffusivity, dt, i_cmp_bc; kwargs...)

Create ERK2 cache for composition field with embedded BCs.
Wrapper around create_erk2_cache_scalar with composition-specific defaults.
"""
create_erk2_cache_composition(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                               diffusivity::Float64, dt::Float64, i_cmp_bc::Int;
                               use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T =
    create_erk2_cache_scalar(T, config, domain, diffusivity, dt, i_cmp_bc;
                              use_krylov=use_krylov, m=m, tol=tol)
