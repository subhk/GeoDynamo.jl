# ================================================================================
# ERK2 INFLUENCE MATRIX FOR POLOIDAL VELOCITY
# ================================================================================
# The influence matrix method enforces P = 0 at boundaries while using derivative
# BCs (∂P/∂r = 0 for no-slip, ∂²P/∂r² = 0 for stress-free) in the exponential operator.
#
# Algorithm (matching DD_2DCODE vel_matrices):
# 1. Create "Green's function" problem with Dirichlet BCs at boundaries
# 2. Set RHS = [1,0,...,0] for inner Green's function, [0,...,0,1] for outer
# 3. Solve with Dirichlet system → get Green's function responses G₁(r), G₂(r)
# 4. Apply these through the derivative BC system → get final Green's functions
# 5. Construct 2×2 influence matrix: invG[i,j] = Gⱼ at boundary i
# 6. Invert the influence matrix
# 7. During timestepping: compute correction = invG * [P(r_inner), P(r_outer)]
#    and subtract Green's function * correction from the solution
# ================================================================================

"""
    ERK2InfluenceMatrix{T}

Precomputed influence matrix data for enforcing P = 0 at boundaries while
using derivative boundary conditions in the exponential operator.

This matches the Fortran DD_2DCODE influence matrix method for poloidal velocity.
"""
struct ERK2InfluenceMatrix{T}
    # Green's functions: Gre[:, 1] = response to inner BC, Gre[:, 2] = response to outer BC
    Gre::Matrix{T}           # (nr, 2) Green's function radial profiles
    invG::Matrix{T}          # (2, 2) inverse influence matrix
    l::Int                   # spherical harmonic degree
end

"""
    create_velocity_poloidal_influence_matrices(T, config, domain, diffusivity, dt, i_vel_bc; theta)

Create influence matrices for poloidal velocity to enforce P = 0 at boundaries
while using derivative BCs (∂P/∂r = 0 or ∂²P/∂r² = 0) in the implicit/exponential system.

This matches DD_2DCODE's vel_matrices routine.
"""
function create_velocity_poloidal_influence_matrices(::Type{T},
                                                      config::SHTnsKitConfig,
                                                      domain::RadialDomain,
                                                      diffusivity::Float64,
                                                      dt::Float64,
                                                      i_vel_bc::Int;
                                                      theta::Float64=d_implicit) where T
    unique_l = unique(config.l_values)
    nr = domain.N
    bw = i_KL

    # Create derivative matrices
    d1 = create_derivative_matrix(T, 1, domain)
    d2 = create_derivative_matrix(T, 2, domain)
    laplacian = create_radial_laplacian(domain)
    r_inv_sq = @views domain.r[1:nr, 2]

    # Diffusion operator base
    base_data = T.(diffusivity .* laplacian.data)

    influence_matrices = Dict{Int, ERK2InfluenceMatrix{T}}()

    for l in unique_l
        l == 0 && continue  # l=0 has no poloidal component

        # Build the diffusion operator for this l: A_l = ν * (∇² - l(l+1)/r²)
        A_data = copy(base_data)
        l_factor = Float64(l * (l + 1))
        @inbounds for n in 1:nr
            A_data[bw + 1, n] -= T(diffusivity * l_factor * r_inv_sq[n])
        end

        # ========================================
        # Step 1: Create Green's function system with Dirichlet BCs
        # ========================================
        # System matrix: X = (1/dt)I - θ*A
        inv_dt = T(1 / dt)
        θ_T = T(theta)

        Xgre_data = copy(A_data)
        Xgre_data .*= -θ_T
        Xgre_data[bw + 1, :] .+= inv_dt

        # Zero boundary rows
        @inbounds for j in 1:(1 + bw)
            Xgre_data[bw + 1 + 1 - j, j] = zero(T)
        end
        @inbounds for j in (nr - bw):nr
            Xgre_data[bw + 1 + nr - j, j] = zero(T)
        end

        # Dirichlet BCs (identity rows) at both boundaries
        Xgre_data[bw + 1, 1] = one(T)
        Xgre_data[bw + 1, nr] = one(T)

        Xgre = BandedMatrix{T}(Xgre_data, bw, nr)
        Xgre_lu = factorize_banded(Xgre)

        # ========================================
        # Step 2: Create physical BC system (derivative BCs)
        # ========================================
        Xpol_data = copy(A_data)
        Xpol_data .*= -θ_T
        Xpol_data[bw + 1, :] .+= inv_dt

        # Zero boundary rows
        @inbounds for j in 1:(1 + bw)
            Xpol_data[bw + 1 + 1 - j, j] = zero(T)
        end
        @inbounds for j in (nr - bw):nr
            Xpol_data[bw + 1 + nr - j, j] = zero(T)
        end

        # Apply poloidal derivative BCs (matching DD_2DCODE vel_bc_Pol)
        # Inner boundary
        if i_vel_bc == 1 || i_vel_bc == 2  # No-slip: ∂P/∂r = 0
            @inbounds for j in 1:(1 + bw)
                Xpol_data[bw + 1 + 1 - j, j] = d1.data[bw + 1 + 1 - j, j]
            end
        else  # Stress-free: ∂²P/∂r² = 0
            @inbounds for j in 1:(1 + bw)
                Xpol_data[bw + 1 + 1 - j, j] = d2.data[bw + 1 + 1 - j, j]
            end
        end

        # Outer boundary
        if i_vel_bc == 1 || i_vel_bc == 3  # No-slip: ∂P/∂r = 0
            @inbounds for j in (nr - bw):nr
                Xpol_data[bw + 1 + nr - j, j] = d1.data[bw + 1 + nr - j, j]
            end
        else  # Stress-free: ∂²P/∂r² = 0
            @inbounds for j in (nr - bw):nr
                Xpol_data[bw + 1 + nr - j, j] = d2.data[bw + 1 + nr - j, j]
            end
        end

        Xpol = BandedMatrix{T}(Xpol_data, bw, nr)
        Xpol_lu = factorize_banded(Xpol)

        # ========================================
        # Step 3: Compute Green's functions
        # ========================================
        Gre = zeros(T, nr, 2)

        # Green's function for inner boundary perturbation
        rhs = zeros(T, nr)
        rhs[1] = one(T)   # Unit perturbation at inner
        solve_banded!(rhs, Xgre_lu, rhs)  # Solve Dirichlet problem

        # Zero the BCs (matching Fortran tim_zerobc)
        rhs[1] = zero(T)
        rhs[nr] = zero(T)

        # Solve with physical (derivative) BCs
        solve_banded!(rhs, Xpol_lu, rhs)
        Gre[:, 1] = rhs

        # Green's function for outer boundary perturbation
        fill!(rhs, zero(T))
        rhs[nr] = one(T)  # Unit perturbation at outer
        solve_banded!(rhs, Xgre_lu, rhs)  # Solve Dirichlet problem

        # Zero the BCs
        rhs[1] = zero(T)
        rhs[nr] = zero(T)

        # Solve with physical BCs
        solve_banded!(rhs, Xpol_lu, rhs)
        Gre[:, 2] = rhs

        # ========================================
        # Step 4: Build and invert influence matrix
        # ========================================
        invG = zeros(T, 2, 2)
        invG[1, 1] = Gre[1, 1]   # Inner response to inner perturbation
        invG[1, 2] = Gre[1, 2]   # Inner response to outer perturbation
        invG[2, 1] = Gre[nr, 1]  # Outer response to inner perturbation
        invG[2, 2] = Gre[nr, 2]  # Outer response to outer perturbation

        # Invert the 2×2 matrix
        det = invG[1, 1] * invG[2, 2] - invG[1, 2] * invG[2, 1]
        max_elem = max(abs(invG[1, 1]), abs(invG[2, 2]), abs(invG[1, 2]), abs(invG[2, 1]))
        rel_det = max_elem > zero(T) ? abs(det) / (max_elem^2) : abs(det)
        if rel_det > pivot_tol(T)
            inv_det = one(T) / det
            tmp = invG[1, 1]
            invG[1, 1] = invG[2, 2] * inv_det
            invG[2, 2] = tmp * inv_det
            invG[1, 2] = -invG[1, 2] * inv_det
            invG[2, 1] = -invG[2, 1] * inv_det
        else
            # Singular influence matrix — no-penetration BC cannot be enforced for this l
            @error "ERK2 influence matrix is near-singular for l=$l (det=$det, " *
                   "relative_det=$rel_det, max_elem=$max_elem). " *
                   "No-penetration boundary condition CANNOT be enforced. " *
                   "Check operator construction, timestep size, or boundary condition type."
            # Zero out correction rather than applying identity (which would add spurious corrections)
            invG[1, 1] = zero(T)
            invG[2, 2] = zero(T)
            invG[1, 2] = zero(T)
            invG[2, 1] = zero(T)
        end

        influence_matrices[l] = ERK2InfluenceMatrix{T}(Gre, invG, l)
    end

    return influence_matrices
end

"""
    apply_influence_matrix_correction!(result, influence, bc_inner_val, bc_outer_val)

Apply influence matrix correction to enforce P = 0 at boundaries.
This subtracts the Green's function response that would give non-zero boundary values.

After this correction, result[1] ≈ bc_inner_val and result[nr] ≈ bc_outer_val
(typically both zero for no-penetration).
"""
function apply_influence_matrix_correction!(result::AbstractVector{T},
                                             influence::ERK2InfluenceMatrix{T},
                                             bc_inner_val::T=zero(T),
                                             bc_outer_val::T=zero(T)) where T
    nr = length(result)

    # Current boundary values (before correction)
    P_inner = result[1]
    P_outer = result[nr]

    # Deviation from desired values
    delta_inner = P_inner - bc_inner_val
    delta_outer = P_outer - bc_outer_val

    # Compute correction coefficients: c = invG * [delta_inner; delta_outer]
    c1 = influence.invG[1, 1] * delta_inner + influence.invG[1, 2] * delta_outer
    c2 = influence.invG[2, 1] * delta_inner + influence.invG[2, 2] * delta_outer

    # Subtract Green's function response: result -= c1*G1 + c2*G2
    @inbounds for i in 1:nr
        result[i] -= c1 * influence.Gre[i, 1] + c2 * influence.Gre[i, 2]
    end

    return result
end

"""
    extract_dense_row(banded::BandedMatrix{T}, row::Int) -> Vector{T}

Extract a full dense row from a banded matrix.
"""
function extract_dense_row(data::Matrix{T}, bandwidth::Int, nr::Int, row::Int) where T
    result = zeros(T, nr)
    @inbounds for j in max(1, row - bandwidth):min(nr, row + bandwidth)
        band_idx = bandwidth + 1 + row - j
        if 1 <= band_idx <= 2*bandwidth + 1
            result[j] = data[band_idx, j]
        end
    end
    return result
end
