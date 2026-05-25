# ================================================================================
# Magnetic Boundary Conditions - Matrix-Embedded Approach
# ================================================================================
#
# This file implements magnetic boundary conditions by embedding them directly
# in the LHS implicit system matrix.
# This ensures the implicit solver satisfies BCs exactly, rather than applying
# them as post-processing.
#
# Boundary condition type: INSULATING (standard geodynamo)
#
# Toroidal BCs (insulating at both boundaries):
#   Inner: BT = 0   (identity row in matrix)
#   Outer: BT = 0   (identity row in matrix)
#
# Poloidal BCs (matching external potential field decay):
#   Inner: (∂/∂r - l/r) BP = 0     (field matches r^l interior solution)
#   Outer: (∂/∂r + (l+1)/r) BP = 0 (field matches r^{-(l+1)} exterior decay)
#
# Note: The poloidal BCs are l-dependent, so the boundary rows differ per degree.
#
# Included by: src/physics/magnetic/field.jl
#
# ================================================================================
# ALGORITHM (matching Fortran mag_bc_Tor / mag_bc_Pol / mag_setbc_Tor)
# ================================================================================
#
#   1. Construct LHS matrix: X = (1/dt)I - θ·L  for each harmonic degree l
#      where L = diffusivity * (∂²/∂r² + 2/r ∂/∂r - l(l+1)/r²)
#
#   2. Zero boundary rows of X (first and last rows of banded matrix)
#
#   3. Fill boundary rows with BC equations:
#      - Toroidal: identity row → BT[boundary] = 0
#      - Poloidal inner: (∂/∂r - l/r) row → decaying interior
#      - Poloidal outer: (∂/∂r + (l+1)/r) row → decaying exterior
#
#   4. LU factorize the modified matrix
#
#   5. Before solving, set RHS boundary values to 0 (homogeneous BCs)
#
#   6. Solve: X * solution = RHS
#      The solution automatically satisfies insulating BCs.
#
# ================================================================================

"""
    create_magnetic_toroidal_matrices(config, domain, diffusivity, dt;
                                      theta, T)

Create implicit time-stepping matrices for the toroidal magnetic component with
insulating boundary conditions embedded in the matrix rows (matching Fortran mag_bc_Tor).

Both boundary rows use identity (BT = 0) for insulating exterior/interior.
"""
function create_magnetic_toroidal_matrices(config::SHTnsKitConfig,
                                            domain::RadialDomain,
                                            diffusivity::Float64,
                                            dt::Float64;
                                            theta::Float64=0.5,
                                            T::Type{<:Number}=Float64)
    unique_l = unique(config.l_values)
    laplacian = create_radial_laplacian(domain)
    r_inv_sq = @views domain.r[1:domain.N, 2]

    base_data = T.(diffusivity .* laplacian.data)
    system_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    linear_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    factorizations = Vector{BandedLU{T}}(undef, length(unique_l))
    l_values = Vector{Int}(undef, length(unique_l))
    lookup = Dict{Int,Int}()

    bw = radial_bandwidth(domain)
    N = domain.N

    inv_dt = T(1 / dt)
    θ_T = T(theta)
    minus_θ = -θ_T

    for (idx, l) in enumerate(unique_l)
        l_values[idx] = l
        lookup[l] = idx

        # Build linear operator: L = diffusivity * (d²/dr² + 2/r d/dr - l(l+1)/r²)
        linear_data = copy(base_data)
        l_factor = Float64(l * (l + 1))
        @inbounds for n in 1:N
            linear_data[bw + 1, n] -= T(diffusivity * l_factor * r_inv_sq[n])
        end

        linear_matrix = BandedMatrix{T}(copy(linear_data), bw, N)

        # Build system matrix: X = (1/dt)I - θ*L
        system_data = copy(linear_data)
        system_data .*= minus_θ
        system_data[bw + 1, :] .+= inv_dt

        # Zero boundary rows (matching Fortran tim_lumesh_X / tim_iclumesh_X)
        # Inner boundary (row 1): zero columns 1 to 1+bw
        @inbounds for j in 1:(1 + bw)
            system_data[bw + 1 + 1 - j, j] = zero(T)
        end
        # Outer boundary (row N): zero columns N-bw to N
        @inbounds for j in (N - bw):N
            system_data[bw + 1 + N - j, j] = zero(T)
        end

        # Insulating BC: BT = 0 at both boundaries (identity rows)
        # Inner boundary
        system_data[bw + 1, 1] = one(T)
        # Outer boundary
        system_data[bw + 1, N] = one(T)

        system_matrix = BandedMatrix{T}(system_data, bw, N)
        system_matrices[idx] = system_matrix
        linear_matrices[idx] = linear_matrix
        factorizations[idx] = factorize_banded(system_matrix)
    end

    return SHTnsImplicitMatrices{T}(system_matrices, factorizations,
                                    linear_matrices, l_values, lookup, theta)
end

"""
    create_magnetic_poloidal_matrices(config, domain, diffusivity, dt;
                                      theta, T)

Create implicit time-stepping matrices for the poloidal magnetic component with
insulating boundary conditions embedded in the matrix rows (matching Fortran mag_bc_Pol).

Boundary conditions (l-dependent):
- Inner: (∂/∂r - l/r) BP = 0  (field matches r^l interior solution)
- Outer: (∂/∂r + (l+1)/r) BP = 0  (field matches r^{-(l+1)} exterior decay)
"""
function create_magnetic_poloidal_matrices(config::SHTnsKitConfig,
                                            domain::RadialDomain,
                                            diffusivity::Float64,
                                            dt::Float64;
                                            theta::Float64=0.5,
                                            T::Type{<:Number}=Float64)
    unique_l = unique(config.l_values)
    laplacian = create_radial_laplacian(domain)
    r_inv_sq = @views domain.r[1:domain.N, 2]

    base_data = T.(diffusivity .* laplacian.data)
    system_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    linear_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    factorizations = Vector{BandedLU{T}}(undef, length(unique_l))
    l_values = Vector{Int}(undef, length(unique_l))
    lookup = Dict{Int,Int}()

    # Create first derivative matrix for poloidal BCs
    d1_matrix = create_derivative_matrix(T, 1, domain)
    bw = radial_bandwidth(domain)
    N = domain.N

    inv_dt = T(1 / dt)
    θ_T = T(theta)
    minus_θ = -θ_T

    for (idx, l) in enumerate(unique_l)
        l_values[idx] = l
        lookup[l] = idx

        # Build linear operator: L = diffusivity * (d²/dr² + 2/r d/dr - l(l+1)/r²)
        linear_data = copy(base_data)
        l_factor = Float64(l * (l + 1))
        @inbounds for n in 1:N
            linear_data[bw + 1, n] -= T(diffusivity * l_factor * r_inv_sq[n])
        end

        linear_matrix = BandedMatrix{T}(copy(linear_data), bw, N)

        # Build system matrix: X = (1/dt)I - θ*L
        system_data = copy(linear_data)
        system_data .*= minus_θ
        system_data[bw + 1, :] .+= inv_dt

        # Zero boundary rows (matching Fortran)
        # Inner boundary (row 1)
        @inbounds for j in 1:(1 + bw)
            system_data[bw + 1 + 1 - j, j] = zero(T)
        end
        # Outer boundary (row N)
        @inbounds for j in (N - bw):N
            system_data[bw + 1 + N - j, j] = zero(T)
        end

        # Insulating poloidal BC at inner boundary:
        # (∂/∂r - l/r) BP = 0
        # Copy first derivative row and subtract l/r[1] on diagonal
        @inbounds for j in 1:(1 + bw)
            system_data[bw + 1 + 1 - j, j] = d1_matrix.data[bw + 1 + 1 - j, j]
        end
        system_data[bw + 1, 1] += T(l * domain.r[1, 3])  # subtract l/r[1]

        # Insulating poloidal BC at outer boundary:
        # (∂/∂r + (l+1)/r) BP = 0
        # Copy first derivative row and add (l+1)/r[N] on diagonal
        @inbounds for j in (N - bw):N
            system_data[bw + 1 + N - j, j] = d1_matrix.data[bw + 1 + N - j, j]
        end
        system_data[bw + 1, N] += T((l + 1) * domain.r[N, 3])  # add (l+1)/r[N]

        system_matrix = BandedMatrix{T}(system_data, bw, N)
        system_matrices[idx] = system_matrix
        linear_matrices[idx] = linear_matrix
        factorizations[idx] = factorize_banded(system_matrix)
    end

    return SHTnsImplicitMatrices{T}(system_matrices, factorizations,
                                    linear_matrices, l_values, lookup, theta)
end

"""
    set_magnetic_rhs_bc!(rhs_real, rhs_imag, slot, nr;
                          inner_value=0.0, outer_value=0.0)

Set boundary values in the RHS vector for the magnetic field solve.
Matches Fortran mag_setbc_Tor / tim_zerobc: sets RHS boundary rows to zero.

For insulating BCs, all boundary values are zero (homogeneous conditions).
"""
function set_magnetic_rhs_bc!(rhs_real::AbstractArray{T}, rhs_imag::AbstractArray{T},
                               slot::CartesianIndex{2}, nr::Int;
                               inner_value::T=zero(T),
                               outer_value::T=zero(T)) where T
    # Set boundary rows of RHS to prescribed values
    # Inner boundary (radial index 1)
    set_local_spectral_value!(rhs_real, slot, 1, inner_value)
    set_local_spectral_value!(rhs_imag, slot, 1, zero(T))
    # Outer boundary (radial index nr)
    set_local_spectral_value!(rhs_real, slot, nr, outer_value)
    set_local_spectral_value!(rhs_imag, slot, nr, zero(T))
end

"""
    solve_magnetic_implicit_step!(solution, rhs, matrices, component;
                                   mag_bc_inner=nothing, prev_bc_inner=nothing)

Solve the implicit magnetic system with boundary conditions embedded in the matrix.
Before solving, sets the RHS boundary values appropriately.

This matches the Fortran DD_2DCODE approach where each rank has full radial profiles
for its subset of (l,m) modes:
1. Loop over local lm modes
2. Set RHS boundary rows to 0 (insulating BCs)
3. Solve banded system with BC rows in matrix
4. Solution automatically satisfies BCs

# Arguments
- `solution`: Output spectral field
- `rhs`: Input RHS spectral field (modified in place for BCs)
- `matrices`: SHTnsImplicitMatrices with BCs embedded
- `component`: `:toroidal` or `:poloidal`
- `mag_bc_inner`: Optional inner boundary values for toroidal (conducting IC case)
- `prev_bc_inner`: Previous step inner BC values (for incremental form)
"""
function solve_magnetic_implicit_step!(solution::SHTnsSpecField{T},
                                        rhs::SHTnsSpecField{T},
                                        matrices::SHTnsImplicitMatrices{T},
                                        component::Symbol;
                                        mag_bc_inner::Union{Vector{T},Nothing}=nothing,
                                        prev_bc_inner::Union{Vector{T},Nothing}=nothing,
                                        mag_bc_inner_imag::Union{Vector{T},Nothing}=nothing,
                                        prev_bc_inner_imag::Union{Vector{T},Nothing}=nothing) where T
    sol_real = parent(solution.data_real)
    sol_imag = parent(solution.data_imag)
    rhs_real = parent(rhs.data_real)
    rhs_imag = parent(rhs.data_imag)

    lm_range = local_spectral_mode_indices(solution.config)
    nr = matrices.system_matrices[1].size  # Full radial size (local = global for spectral)

    # Allocate buffers for the radial profile
    tmp_r = Vector{T}(undef, nr)
    tmp_i = Vector{T}(undef, nr)

    # Loop over local lm modes only (radial is fully local, matching DD_2DCODE)
    @inbounds for lm_idx in lm_range
        slot = local_spectral_storage_slot(solution.config, lm_idx)
        slot === nothing && continue

        l = solution.config.l_values[lm_idx]
        idx = get(matrices.lookup, l, nothing)
        idx === nothing && continue

        # Copy RHS radial profile to work buffer
        for ir in 1:nr
            tmp_r[ir] = local_spectral_value(rhs_real, slot, ir)
            tmp_i[ir] = local_spectral_value(rhs_imag, slot, ir)
        end

        # Set RHS boundary values (matching Fortran mag_setbc_Tor / tim_zerobc)
        inner_val = zero(T)
        inner_imag = zero(T)
        outer_val = zero(T)

        # For conducting inner core case (toroidal only):
        # inner boundary gets mag_bc value (from nonlinear correction)
        if component == :toroidal && mag_bc_inner !== nothing
            if lm_idx <= length(mag_bc_inner)
                inner_val = mag_bc_inner[lm_idx]
                # Incremental form: subtract previous step's BC
                if prev_bc_inner !== nothing && lm_idx <= length(prev_bc_inner)
                    inner_val -= prev_bc_inner[lm_idx]
                end
            end
        end
        if component == :toroidal && mag_bc_inner_imag !== nothing
            if lm_idx <= length(mag_bc_inner_imag)
                inner_imag = mag_bc_inner_imag[lm_idx]
                if prev_bc_inner_imag !== nothing && lm_idx <= length(prev_bc_inner_imag)
                    inner_imag -= prev_bc_inner_imag[lm_idx]
                end
            end
        end

        # Set boundary values
        tmp_r[1] = inner_val
        tmp_i[1] = inner_imag
        tmp_r[nr] = outer_val
        tmp_i[nr] = zero(T)

        # Solve the banded system (matching Fortran tim_invX)
        solve_banded!(tmp_r, matrices.factorizations[idx], tmp_r)
        solve_banded!(tmp_i, matrices.factorizations[idx], tmp_i)

        # Store solution back
        for ir in 1:nr
            set_local_spectral_value!(sol_real, slot, ir, tmp_r[ir])
            set_local_spectral_value!(sol_imag, slot, ir, tmp_i[ir])
        end
    end

    return solution
end
