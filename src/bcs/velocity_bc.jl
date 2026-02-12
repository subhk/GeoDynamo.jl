# ================================================================================
# Velocity Boundary Conditions - Matrix-Embedded Approach
# ================================================================================
#
# This file implements velocity boundary conditions by embedding them directly
# in the LHS implicit system matrix.
# This ensures the implicit solver satisfies BCs exactly, rather than applying
# them as post-processing.
#
# Boundary condition types (i_vel_bc):
#   1 = No-slip at both boundaries
#   2 = No-slip at inner, stress-free at outer
#   3 = Stress-free at inner, no-slip at outer
#   4 = Stress-free at both boundaries
#
# Toroidal BCs:
#   No-slip:     T = 0              (identity row in matrix)
#   Stress-free: ∂T/∂r - T/r = 0   (derivative row minus 1/r)
#
# Poloidal BCs:
#   No-slip:     ∂P/∂r = 0          (first derivative row)
#   Stress-free: ∂²P/∂r² = 0        (second derivative row)
#
# Included by: src/velocity.jl (after VelocityWorkspace is defined)
#
# ================================================================================
# ALGORITHM (matching Fortran vel_matrices / vel_bc_Tor / vel_bc_Pol)
# ================================================================================
#
#   1. Construct LHS matrix: X = (1/dt)I - θ·L  for each harmonic degree l
#      where L = diffusivity * (∂²/∂r² + 2/r ∂/∂r - l(l+1)/r²)
#
#   2. Zero boundary rows of X (first and last rows of banded matrix)
#
#   3. Fill boundary rows with BC equations:
#      - Toroidal no-slip:     identity row → T[boundary] = rhs
#      - Toroidal stress-free: ∂/∂r - 1/r row → (∂T/∂r - T/r)[boundary] = rhs
#      - Poloidal no-slip:     ∂/∂r row → (∂P/∂r)[boundary] = rhs
#      - Poloidal stress-free: ∂²/∂r² row → (∂²P/∂r²)[boundary] = rhs
#
#   4. LU factorize the modified matrix
#
#   5. Before solving, set RHS boundary values:
#      - Typically 0 (homogeneous BCs)
#      - For rotating inner core: T(l=1,m=0) = rot_omega * r[1]
#
#   6. Solve: X * solution = RHS
#      The solution automatically satisfies BCs.
#
# ================================================================================

"""
    create_velocity_toroidal_matrices(config, domain, diffusivity, dt;
                                      theta, i_vel_bc, T)

Create implicit time-stepping matrices for the toroidal velocity component with
boundary conditions embedded in the matrix rows.

The boundary rows of the system matrix are replaced with the BC equations:
- No-slip: identity row (T = value)
- Stress-free: ∂T/∂r - T/r = 0

This ensures the implicit solve enforces BCs exactly rather than applying them
as post-processing.
"""
function create_velocity_toroidal_matrices(config::SHTnsKitConfig,
                                            domain::RadialDomain,
                                            diffusivity::Float64,
                                            dt::Float64;
                                            theta::Float64=d_implicit,
                                            i_vel_bc::Int=get_parameters().i_vel_bc,
                                            mass_coeff::Float64=1.0,
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

    # Create first derivative matrix for stress-free BC
    d1_matrix = create_derivative_matrix(T, 1, domain)
    bw = i_KL
    N = domain.N

    # Mass coefficient: c1/dt (Fortran: c1=d_E for velocity)
    inv_dt = T(mass_coeff / dt)
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

        # Apply toroidal BC at inner boundary
        # i_vel_bc == 1 or 2: no-slip at inner (identity row)
        if i_vel_bc == 1 || i_vel_bc == 2
            system_data[bw + 1, 1] = one(T)  # T[1] = rhs
        else
            # Stress-free at inner: ∂T/∂r - T/r = 0
            # Copy first derivative row and subtract 1/r[1] on diagonal
            @inbounds for j in 1:(1 + bw)
                system_data[bw + 1 + 1 - j, j] = d1_matrix.data[bw + 1 + 1 - j, j]
            end
            system_data[bw + 1, 1] -= T(domain.r[1, 3])  # subtract 1/r[1]
        end

        # Apply toroidal BC at outer boundary
        # i_vel_bc == 1 or 3: no-slip at outer (identity row)
        if i_vel_bc == 1 || i_vel_bc == 3
            system_data[bw + 1, N] = one(T)  # T[N] = rhs
        else
            # Stress-free at outer: ∂T/∂r - T/r = 0
            # Copy first derivative row and subtract 1/r[N] on diagonal
            @inbounds for j in (N - bw):N
                system_data[bw + 1 + N - j, j] = d1_matrix.data[bw + 1 + N - j, j]
            end
            system_data[bw + 1, N] -= T(domain.r[N, 3])  # subtract 1/r[N]
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
    create_velocity_poloidal_matrices(config, domain, diffusivity, dt;
                                      theta, i_vel_bc, T)

Create implicit time-stepping matrices for the poloidal velocity component with
boundary conditions embedded in the matrix rows.

The boundary rows of the system matrix are replaced with the BC equations:
- No-slip: first derivative row (∂P/∂r = value)
- Stress-free: second derivative row (∂²P/∂r² = value)
"""
function create_velocity_poloidal_matrices(config::SHTnsKitConfig,
                                            domain::RadialDomain,
                                            diffusivity::Float64,
                                            dt::Float64;
                                            theta::Float64=d_implicit,
                                            i_vel_bc::Int=get_parameters().i_vel_bc,
                                            mass_coeff::Float64=1.0,
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

    # Create derivative matrices for BCs
    d1_matrix = create_derivative_matrix(T, 1, domain)
    d2_matrix = create_derivative_matrix(T, 2, domain)
    bw = i_KL
    N = domain.N

    # Mass coefficient: c1/dt (Fortran: c1=d_E for velocity)
    inv_dt = T(mass_coeff / dt)
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

        # Apply poloidal BC at inner boundary
        # i_vel_bc == 1 or 2: no-slip at inner (first derivative row: ∂P/∂r = value)
        if i_vel_bc == 1 || i_vel_bc == 2
            @inbounds for j in 1:(1 + bw)
                system_data[bw + 1 + 1 - j, j] = d1_matrix.data[bw + 1 + 1 - j, j]
            end
        else
            # Stress-free at inner: second derivative row (∂²P/∂r² = value)
            @inbounds for j in 1:(1 + bw)
                system_data[bw + 1 + 1 - j, j] = d2_matrix.data[bw + 1 + 1 - j, j]
            end
        end

        # Apply poloidal BC at outer boundary
        # i_vel_bc == 1 or 3: no-slip at outer (first derivative row: ∂P/∂r = value)
        if i_vel_bc == 1 || i_vel_bc == 3
            @inbounds for j in (N - bw):N
                system_data[bw + 1 + N - j, j] = d1_matrix.data[bw + 1 + N - j, j]
            end
        else
            # Stress-free at outer: second derivative row (∂²P/∂r² = value)
            @inbounds for j in (N - bw):N
                system_data[bw + 1 + N - j, j] = d2_matrix.data[bw + 1 + N - j, j]
            end
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
    create_velocity_green_matrices(config, domain, diffusivity;
                                    theta, T)

Create Green's function matrices for the influence matrix method (poloidal pressure).
These use Dirichlet BCs (identity rows) at both boundaries, matching Fortran vel_bc_Gre.
"""
function create_velocity_green_matrices(config::SHTnsKitConfig,
                                         domain::RadialDomain,
                                         diffusivity::Float64;
                                         theta::Float64=d_implicit,
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

    bw = i_KL
    N = domain.N

    # For Green's functions: c1=0, c2=1/d_implicit → system = -L (pure diffusion)
    inv_impl = T(theta > 0 ? 1.0 / theta : 1.0)

    for (idx, l) in enumerate(unique_l)
        l_values[idx] = l
        lookup[l] = idx

        # Build operator: -diffusivity * (d²/dr² + 2/r d/dr - l(l+1)/r²)
        linear_data = copy(base_data)
        l_factor = Float64(l * (l + 1))
        @inbounds for n in 1:N
            linear_data[bw + 1, n] -= T(diffusivity * l_factor * r_inv_sq[n])
        end
        linear_matrix = BandedMatrix{T}(copy(linear_data), bw, N)

        # System matrix for Green's function: -inv_impl * L
        system_data = copy(linear_data)
        system_data .*= T(-inv_impl)

        # Zero boundary rows
        @inbounds for j in 1:(1 + bw)
            system_data[bw + 1 + 1 - j, j] = zero(T)
        end
        @inbounds for j in (N - bw):N
            system_data[bw + 1 + N - j, j] = zero(T)
        end

        # Dirichlet BCs at both boundaries (identity rows)
        system_data[bw + 1, 1] = one(T)
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
    set_velocity_rhs_bc_toroidal!(rhs_real, rhs_imag, local_lm, nr;
                                   inner_value=0.0, outer_value=0.0)

Set boundary values in the RHS vector for the toroidal velocity solve.
Matches Fortran vel_setbc_Tor: sets RHS boundary rows to zero (or prescribed value).

For no-slip: RHS boundary = 0 (or prescribed velocity, e.g., rotating inner core)
For stress-free: RHS boundary = 0 (homogeneous condition ∂T/∂r - T/r = 0)
"""
function set_velocity_rhs_bc_toroidal!(rhs_real::AbstractArray{T}, rhs_imag::AbstractArray{T},
                                        local_lm::Int, nr::Int;
                                        inner_value::T=zero(T),
                                        outer_value::T=zero(T)) where T
    # Set boundary rows of RHS to prescribed values
    # Inner boundary (radial index 1)
    @inbounds rhs_real[local_lm, 1, 1] = inner_value
    @inbounds rhs_imag[local_lm, 1, 1] = zero(T)
    # Outer boundary (radial index nr)
    @inbounds rhs_real[local_lm, 1, nr] = outer_value
    @inbounds rhs_imag[local_lm, 1, nr] = zero(T)
end

"""
    set_velocity_rhs_bc_poloidal!(rhs_real, rhs_imag, local_lm, nr;
                                   inner_value=0.0, outer_value=0.0)

Set boundary values in the RHS vector for the poloidal velocity solve.
Matches Fortran: sets RHS boundary rows to zero for homogeneous BCs.

For no-slip: RHS boundary = 0 (∂P/∂r = 0)
For stress-free: RHS boundary = 0 (∂²P/∂r² = 0)
"""
function set_velocity_rhs_bc_poloidal!(rhs_real::AbstractArray{T}, rhs_imag::AbstractArray{T},
                                        local_lm::Int, nr::Int;
                                        inner_value::T=zero(T),
                                        outer_value::T=zero(T)) where T
    # Set boundary rows of RHS to prescribed values
    @inbounds rhs_real[local_lm, 1, 1] = inner_value
    @inbounds rhs_imag[local_lm, 1, 1] = zero(T)
    @inbounds rhs_real[local_lm, 1, nr] = outer_value
    @inbounds rhs_imag[local_lm, 1, nr] = zero(T)
end

"""
    solve_velocity_implicit_step!(solution, rhs, matrices, component;
                                   i_vel_bc=1, domain=nothing,
                                   rot_omega=0.0, current_field=nothing)

Solve the implicit velocity system with boundary conditions embedded in the matrix.
Before solving, sets the RHS boundary values appropriately.

This matches the Fortran DD_2DCODE approach where each rank has full radial profiles
for its subset of (l,m) modes:
1. Loop over local lm modes
2. Set RHS boundary rows to BC values (typically 0)
3. Solve banded system with BC rows in matrix
4. Solution automatically satisfies BCs

# Arguments
- `solution`: Output spectral field
- `rhs`: Input RHS spectral field (modified in place for BCs)
- `matrices`: SHTnsImplicitMatrices with BCs embedded
- `component`: `:toroidal` or `:poloidal`
- `i_vel_bc`: Velocity BC type (1-4)
- `domain`: RadialDomain (needed for rotating IC boundary)
- `rot_omega`: Inner core rotation rate (for no-slip toroidal l=1,m=0)
- `current_field`: Current velocity field (for incremental form of rotating IC BC)
"""
function solve_velocity_implicit_step!(solution::SHTnsSpecField{T},
                                        rhs::SHTnsSpecField{T},
                                        matrices::SHTnsImplicitMatrices{T},
                                        component::Symbol;
                                        i_vel_bc::Int=1,
                                        domain::Union{RadialDomain,Nothing}=nothing,
                                        rot_omega::Float64=0.0,
                                        current_field::Union{SHTnsSpecField{T},Nothing}=nothing) where T
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
        local_lm = lm_idx - first(lm_range) + 1

        l = solution.config.l_values[lm_idx]
        m = solution.config.m_values[lm_idx]
        idx = get(matrices.lookup, l, nothing)
        idx === nothing && continue

        # Copy RHS radial profile to work buffer
        for ir in 1:nr
            tmp_r[ir] = rhs_real[local_lm, 1, ir]
            tmp_i[ir] = rhs_imag[local_lm, 1, ir]
        end

        # Set RHS boundary values (matching Fortran vel_setbc_Tor / tim_zerobc)
        if component == :toroidal
            inner_val = zero(T)
            outer_val = zero(T)

            # For no-slip inner (i_vel_bc ≤ 2) with rotating IC:
            # T(l=1,m=0) at inner boundary = rot_omega * r[1]
            if (i_vel_bc == 1 || i_vel_bc == 2) && l == 1 && m == 0 && domain !== nothing
                inner_val = T(rot_omega * domain.r[1, 4])
                # If not first step, subtract current field value (incremental form)
                if current_field !== nothing
                    cur_real = parent(current_field.data_real)
                    inner_val -= cur_real[local_lm, 1, 1]
                end
            end

            # Set boundary values
            tmp_r[1] = inner_val
            tmp_i[1] = zero(T)
            tmp_r[nr] = outer_val
            tmp_i[nr] = zero(T)
        else  # :poloidal
            # Poloidal BCs: zero at both boundaries (∂P/∂r = 0 or ∂²P/∂r² = 0)
            tmp_r[1] = zero(T)
            tmp_i[1] = zero(T)
            tmp_r[nr] = zero(T)
            tmp_i[nr] = zero(T)
        end

        # Solve the banded system (matching Fortran tim_invX)
        solve_banded!(tmp_r, matrices.factorizations[idx], tmp_r)
        solve_banded!(tmp_i, matrices.factorizations[idx], tmp_i)

        # Store solution back
        for ir in 1:nr
            sol_real[local_lm, 1, ir] = tmp_r[ir]
            sol_imag[local_lm, 1, ir] = tmp_i[ir]
        end
    end

    return solution
end
