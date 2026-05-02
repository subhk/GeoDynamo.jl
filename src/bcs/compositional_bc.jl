# ================================================================================
# Composition Boundary Conditions - Matrix-Embedded Approach
# ================================================================================
#
# This file implements composition boundary conditions by embedding them directly
# in the LHS implicit system matrix.
#
# Boundary condition types controlled by composition_bc_code parameter:
#   1: Fixed C on both ICB and CMB (Dirichlet-Dirichlet)
#   2: Fixed C on ICB, fixed flux on CMB (Dirichlet-Neumann)
#   3: Fixed flux on ICB, fixed C on CMB (Neumann-Dirichlet)
#   4: Fixed flux on both ICB and CMB (Neumann-Neumann)
#      Special case: l=0 inner boundary uses Dirichlet to avoid underdetermined system
#
# Dirichlet: identity row in matrix (C = prescribed value)
# Neumann: first derivative row in matrix (∂C/∂r = prescribed flux)
#
# Included by: src/physics/composition/field.jl
#
# ================================================================================
# ALGORITHM (matching Fortran cmp_matrices / cmp_bc_C / cmp_setbc)
# ================================================================================
#
#   1. Construct LHS matrix: X = (c1/dt)I - c2*θ*L  for each harmonic degree l
#      where L = diffusivity * (∂²/∂r² + 2/r ∂/∂r - l(l+1)/r²)
#      and c1=1, c2=1/Sc (Schmidt number ratio)
#
#   2. Zero boundary rows of X (first and last rows of banded matrix)
#
#   3. Fill boundary rows based on composition_bc_code:
#      - Dirichlet: identity row (diagonal = 1)
#      - Neumann: copy first derivative operator row from D%dr(1)
#      - Special: if composition_bc_code=4 and l=0, override inner to Dirichlet
#
#   4. LU factorize the modified matrix
#
#   5. Before solving, set RHS boundary values (cmp_setbc: usually 0)
#
#   6. Solve: X * solution = RHS
#      The solution automatically satisfies prescribed BCs.
#
# ================================================================================

"""
    create_composition_matrices(config, domain, diffusivity, dt;
                                 composition_bc_code, theta, T)

Create implicit time-stepping matrices for the composition equation with
boundary conditions embedded in the matrix rows (matching Fortran cmp_bc_C).

# Arguments
- `config`: SHTnsKitConfig with l_values
- `domain`: RadialDomain with derivative matrices and grid
- `diffusivity`: Compositional diffusivity (Pm/Sc in magnetic diffusion time)
- `dt`: Timestep
- `composition_bc_code`: BC type (1=DD, 2=DN, 3=ND, 4=NN)
- `theta`: Implicit parameter
- `T`: Numeric type (default Float64)

# Returns
- `SHTnsImplicitMatrices{T}` with BCs embedded in system matrices
"""
function create_composition_matrices(config::SHTnsKitConfig,
                                      domain::RadialDomain,
                                      diffusivity::Float64,
                                      dt::Float64;
                                      composition_bc_code::Int,
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

    # First derivative matrix for Neumann BCs
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

        # Zero boundary rows (matching Fortran tim_lumesh_X)
        # Inner boundary (row 1): zero columns 1 to 1+bw
        @inbounds for j in 1:(1 + bw)
            system_data[bw + 1 + 1 - j, j] = zero(T)
        end
        # Outer boundary (row N): zero columns N-bw to N
        @inbounds for j in (N - bw):N
            system_data[bw + 1 + N - j, j] = zero(T)
        end

        # Apply boundary conditions based on composition_bc_code
        # (matching Fortran cmp_bc_C)

        # --- Inner boundary ---
        if composition_bc_code == 1 || composition_bc_code == 2
            # Dirichlet at inner: identity row
            system_data[bw + 1, 1] = one(T)
        else
            # Neumann at inner: copy first derivative row
            @inbounds for j in 1:(1 + bw)
                system_data[bw + 1 + 1 - j, j] = d1_matrix.data[bw + 1 + 1 - j, j]
            end
        end

        # --- Outer boundary ---
        if composition_bc_code == 1 || composition_bc_code == 3
            # Dirichlet at outer: identity row
            system_data[bw + 1, N] = one(T)
        else
            # Neumann at outer: copy first derivative row
            @inbounds for j in (N - bw):N
                system_data[bw + 1 + N - j, j] = d1_matrix.data[bw + 1 + N - j, j]
            end
        end

        # Special case: if both boundaries are Neumann (composition_bc_code=4) and l=0,
        # override inner boundary to Dirichlet to avoid underdetermined system.
        # (Matching Fortran: "Fix temperature so that the code knows what the
        #  temperature is when both boundaries are fixed flux")
        if composition_bc_code == 4 && l == 0
            # Zero inner boundary row again
            @inbounds for j in 1:(1 + bw)
                system_data[bw + 1 + 1 - j, j] = zero(T)
            end
            # Set identity at inner boundary
            system_data[bw + 1, 1] = one(T)
        end

        system_matrix = BandedMatrix{T}(system_data, bw, N)
        system_matrices[idx] = system_matrix
        linear_matrices[idx] = linear_matrix
        factorizations[idx] = factorize_banded(system_matrix)
    end

    return SHTnsImplicitMatrices{T}(system_matrices, factorizations,
                                    linear_matrices, l_values, lookup, theta)
end

"""
    set_composition_rhs_bc!(rhs_real, rhs_imag, slot, nr;
                             inner_value=0.0, outer_value=0.0,
                             inner_value_imag=0.0, outer_value_imag=0.0)

Set boundary values in the RHS vector for the composition solve.
Matches Fortran cmp_setbc: sets RHS boundary rows to prescribed values.

For standard compositional convection, boundary values are typically zero
(homogeneous Neumann: ∂C/∂r=0, or homogeneous Dirichlet: C=0).
Non-zero imaginary parts are used when file-based spectral BCs are loaded.
"""
function set_composition_rhs_bc!(rhs_real::AbstractArray{T}, rhs_imag::AbstractArray{T},
                                  slot::CartesianIndex{2}, nr::Int;
                                  inner_value::T=zero(T),
                                  outer_value::T=zero(T),
                                  inner_value_imag::T=zero(T),
                                  outer_value_imag::T=zero(T)) where T
    # Set boundary rows of RHS to prescribed values
    # Inner boundary (radial index 1)
    set_local_spectral_value!(rhs_real, slot, 1, inner_value)
    set_local_spectral_value!(rhs_imag, slot, 1, inner_value_imag)
    # Outer boundary (radial index nr)
    set_local_spectral_value!(rhs_real, slot, nr, outer_value)
    set_local_spectral_value!(rhs_imag, slot, nr, outer_value_imag)
end

"""
    solve_composition_implicit_step!(solution, rhs, matrices;
                                      bc_inner, bc_outer, bc_inner_imag, bc_outer_imag)

Solve the implicit composition system with boundary conditions embedded in the matrix.
Before solving, sets the RHS boundary values appropriately.

This matches the Fortran DD_2DCODE approach where each rank has full radial profiles
for its subset of (l,m) modes:
1. Loop over local lm modes
2. Set RHS boundary rows to prescribed values (cmp_setbc)
3. Solve banded system with BC rows in matrix
4. Solution automatically satisfies BCs

# Arguments
- `solution`: Output spectral field
- `rhs`: Input RHS spectral field (modified in place for BCs)
- `matrices`: SHTnsImplicitMatrices with composition BCs embedded
- `bc_inner`: Optional vector of inner boundary values (real part) per mode (default: zeros)
- `bc_outer`: Optional vector of outer boundary values (real part) per mode (default: zeros)
- `bc_inner_imag`: Optional vector of inner boundary values (imag part) per mode (default: zeros)
- `bc_outer_imag`: Optional vector of outer boundary values (imag part) per mode (default: zeros)
"""
function solve_composition_implicit_step!(solution::SHTnsSpecField{T},
                                           rhs::SHTnsSpecField{T},
                                           matrices::SHTnsImplicitMatrices{T};
                                           bc_inner::Union{Vector{T},Nothing}=nothing,
                                           bc_outer::Union{Vector{T},Nothing}=nothing,
                                           bc_inner_imag::Union{Vector{T},Nothing}=nothing,
                                           bc_outer_imag::Union{Vector{T},Nothing}=nothing) where T
    sol_real = parent(solution.data_real)
    sol_imag = parent(solution.data_imag)
    rhs_real = parent(rhs.data_real)
    rhs_imag = parent(rhs.data_imag)

    lm_range = get_local_range(solution.pencil, 1)
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

        # Set RHS boundary values (matching Fortran cmp_setbc)
        inner_val = bc_inner !== nothing && lm_idx <= length(bc_inner) ? bc_inner[lm_idx] : zero(T)
        outer_val = bc_outer !== nothing && lm_idx <= length(bc_outer) ? bc_outer[lm_idx] : zero(T)
        inner_val_i = bc_inner_imag !== nothing && lm_idx <= length(bc_inner_imag) ? bc_inner_imag[lm_idx] : zero(T)
        outer_val_i = bc_outer_imag !== nothing && lm_idx <= length(bc_outer_imag) ? bc_outer_imag[lm_idx] : zero(T)

        # Set boundary values
        tmp_r[1] = inner_val
        tmp_i[1] = inner_val_i
        tmp_r[nr] = outer_val
        tmp_i[nr] = outer_val_i

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
