# ================================================================================
# Velocity Boundary Conditions - Matrix-Embedded Approach
# ================================================================================
#
# This file implements velocity boundary conditions by embedding them directly
# in the LHS implicit system matrix.
# This ensures the implicit solver satisfies BCs exactly, rather than applying
# them as post-processing.
#
# Boundary condition types (velocity_bc_code):
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
# Included by: src/physics/velocity/field.jl (after VelocityWorkspace is defined)
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
                                      theta, velocity_bc_code, mass_coeff, T,
                                      inner_regularity)

Create implicit time-stepping matrices for the toroidal velocity component with
boundary conditions embedded in the matrix rows.

The boundary rows of the system matrix are replaced with the BC equations:
- No-slip: identity row (T = value)
- Stress-free: ∂T/∂r - T/r = 0

When `inner_regularity = true` (ball / full-sphere geometry) the inner row
instead imposes the center regularity condition t′(r₁) = l·t(r₁)/r₁ (β = l,
raw sphtor toroidal scalar t ~ r^l); the outer row is still set by
`velocity_bc_code`.

This ensures the implicit solve enforces BCs exactly rather than applying them
as post-processing.
"""
function create_velocity_toroidal_matrices(config::SHTnsKitConfig,
        domain::RadialDomain,
        diffusivity::Float64,
        dt::Float64;
        velocity_bc_code::Int,
        theta::Float64 = 0.5,
        mass_coeff::Float64 = 1.0,
        T::Type{<:Number} = Float64,
        inner_regularity::Bool = false)
    unique_l = unique(config.l_values)
    laplacian = create_radial_laplacian(domain)
    r_inv_sq = @views domain.r[1:domain.N, 2]

    base_data = T.(diffusivity .* laplacian.data)
    system_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    linear_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    factorizations = Vector{BandedLU{T}}(undef, length(unique_l))
    l_values = Vector{Int}(undef, length(unique_l))
    lookup = Dict{Int, Int}()

    # Create first derivative matrix for stress-free BC
    d1_matrix = create_derivative_matrix(T, 1, domain)
    bw = radial_bandwidth(domain)
    N = domain.N

    # Mass coefficient: c1/dt (Fortran: c1=Ek for velocity)
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
        if inner_regularity
            # Ball center regularity for the raw sphtor toroidal scalar:
            # t ~ r^l ⇒ t′(r₁) = l·t(r₁)/r₁ (β = l).
            @inbounds for j in 1:(1 + bw)
                system_data[bw + 1 + 1 - j, j] = d1_matrix.data[bw + 1 + 1 - j, j]
            end
            system_data[bw + 1, 1] -= T(l * domain.r[1, 3])
        elseif velocity_bc_code == 1 || velocity_bc_code == 2
            # no-slip at inner (identity row)
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
        # velocity_bc_code == 1 or 3: no-slip at outer (identity row)
        if velocity_bc_code == 1 || velocity_bc_code == 3
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
                                      theta, velocity_bc_code, T)

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
        velocity_bc_code::Int,
        theta::Float64 = 0.5,
        mass_coeff::Float64 = 1.0,
        T::Type{<:Number} = Float64)
    unique_l = unique(config.l_values)
    d1_matrix = create_derivative_matrix(T, 1, domain)
    d2_matrix = create_derivative_matrix(T, 2, domain)
    r_inv_sq = @views domain.r[1:domain.N, 2]

    # Poloidal diffusion operator D_pol = ∂² − l(l+1)/r² — the radial part of the
    # double-curl, which has NO 2/r term (unlike the scalar Laplacian ∂²+2/r∂).
    # The base is the bare ∂² here; the −l(l+1)/r² shift is added per-l below.
    # (Matches the W-split builder; the full P=0 / P'=0 wall conditions are
    # completed by the influence-matrix correction, not by this 2nd-order solve.)
    base_data = T.(diffusivity .* d2_matrix.data)
    system_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    linear_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    factorizations = Vector{BandedLU{T}}(undef, length(unique_l))
    l_values = Vector{Int}(undef, length(unique_l))
    lookup = Dict{Int, Int}()

    bw = radial_bandwidth(domain)
    N = domain.N

    # Mass coefficient: c1/dt (Fortran: c1=Ek for velocity)
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
        # velocity_bc_code == 1 or 2: no-slip at inner (first derivative row: ∂P/∂r = value)
        if velocity_bc_code == 1 || velocity_bc_code == 2
            @inbounds for j in 1:(1 + bw)
                system_data[bw + 1 + 1 - j, j] = d1_matrix.data[bw + 1 + 1 - j, j]
            end
        else
            # Stress-free at inner: the no-tangential-stress row P″ − (2/r)P′ = 0
            # (from σ_rθ ∝ r·∂_r(u_θ/r) with u_θ ~ P′/r). The bare P″=0 row missed
            # the −2P′/r metric term.
            r2_in = T(2 / domain.r[1, 4])
            @inbounds for j in 1:(1 + bw)
                system_data[bw + 1 + 1 - j, j] =
                    d2_matrix.data[bw + 1 + 1 - j, j] - r2_in * d1_matrix.data[bw + 1 + 1 - j, j]
            end
        end

        # Apply poloidal BC at outer boundary
        # velocity_bc_code == 1 or 3: no-slip at outer (first derivative row: ∂P/∂r = value)
        if velocity_bc_code == 1 || velocity_bc_code == 3
            @inbounds for j in (N - bw):N
                system_data[bw + 1 + N - j, j] = d1_matrix.data[bw + 1 + N - j, j]
            end
        else
            # Stress-free at outer: P″ − (2/r)P′ = 0 (see inner comment).
            r2_out = T(2 / domain.r[N, 4])
            @inbounds for j in (N - bw):N
                system_data[bw + 1 + N - j, j] =
                    d2_matrix.data[bw + 1 + N - j, j] - r2_out * d1_matrix.data[bw + 1 + N - j, j]
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
        theta::Float64 = 0.5,
        T::Type{<:Number} = Float64)
    unique_l = unique(config.l_values)
    laplacian = create_radial_laplacian(domain)
    r_inv_sq = @views domain.r[1:domain.N, 2]

    base_data = T.(diffusivity .* laplacian.data)
    system_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    linear_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    factorizations = Vector{BandedLU{T}}(undef, length(unique_l))
    l_values = Vector{Int}(undef, length(unique_l))
    lookup = Dict{Int, Int}()

    bw = radial_bandwidth(domain)
    N = domain.N

    # For Green's functions: c1=0, c2=1/theta -> system = -L (pure diffusion)
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
    set_velocity_rhs_bc_toroidal!(rhs_real, rhs_imag, slot, nr;
                                   inner_value=0.0, outer_value=0.0)

Set boundary values in the RHS vector for the toroidal velocity solve.
Matches Fortran vel_setbc_Tor: sets RHS boundary rows to zero (or prescribed value).

For no-slip: RHS boundary = 0 (or prescribed velocity, e.g., rotating inner core)
For stress-free: RHS boundary = 0 (homogeneous condition ∂T/∂r - T/r = 0)
"""
function set_velocity_rhs_bc_toroidal!(
        rhs_real::AbstractArray{T}, rhs_imag::AbstractArray{T},
        slot::CartesianIndex{2}, nr::Int;
        inner_value::T = zero(T),
        outer_value::T = zero(T)) where {T}
    # Set boundary rows of RHS to prescribed values
    # Inner boundary (radial index 1)
    set_local_spectral_value!(rhs_real, slot, 1, inner_value)
    set_local_spectral_value!(rhs_imag, slot, 1, zero(T))
    # Outer boundary (radial index nr)
    set_local_spectral_value!(rhs_real, slot, nr, outer_value)
    set_local_spectral_value!(rhs_imag, slot, nr, zero(T))
end

"""
    set_velocity_rhs_bc_poloidal!(rhs_real, rhs_imag, slot, nr;
                                   inner_value=0.0, outer_value=0.0)

Set boundary values in the RHS vector for the poloidal velocity solve.
Matches Fortran: sets RHS boundary rows to zero for homogeneous BCs.

For no-slip: RHS boundary = 0 (∂P/∂r = 0)
For stress-free: RHS boundary = 0 (∂²P/∂r² = 0)
"""
function set_velocity_rhs_bc_poloidal!(
        rhs_real::AbstractArray{T}, rhs_imag::AbstractArray{T},
        slot::CartesianIndex{2}, nr::Int;
        inner_value::T = zero(T),
        outer_value::T = zero(T)) where {T}
    # Set boundary rows of RHS to prescribed values
    set_local_spectral_value!(rhs_real, slot, 1, inner_value)
    set_local_spectral_value!(rhs_imag, slot, 1, zero(T))
    set_local_spectral_value!(rhs_real, slot, nr, outer_value)
    set_local_spectral_value!(rhs_imag, slot, nr, zero(T))
end

"""
    solve_velocity_implicit_step!(solution, rhs, matrices, component;
                                   velocity_bc_code=1, domain=nothing,
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
- `velocity_bc_code`: Velocity BC type (1-4)
- `domain`: RadialDomain (needed for rotating IC boundary)
- `rot_omega`: Inner core rotation rate (for no-slip toroidal l=1,m=0)
- `current_field`: Current velocity field (for incremental form of rotating IC BC)
"""
function solve_velocity_implicit_step!(solution::SHTnsSpecField{T},
        rhs::SHTnsSpecField{T},
        matrices::SHTnsImplicitMatrices{T},
        component::Symbol;
        velocity_bc_code::Int = 1,
        domain::Union{RadialDomain, Nothing} = nothing,
        rot_omega::Float64 = 0.0,
        current_field::Union{SHTnsSpecField{T}, Nothing} = nothing) where {T}
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
        m = solution.config.m_values[lm_idx]
        idx = get(matrices.lookup, l, nothing)
        idx === nothing && continue

        # Copy RHS radial profile to work buffer
        for ir in 1:nr
            tmp_r[ir] = local_spectral_value(rhs_real, slot, ir)
            tmp_i[ir] = local_spectral_value(rhs_imag, slot, ir)
        end

        # Set RHS boundary values (matching Fortran vel_setbc_Tor / tim_zerobc)
        if component == :toroidal
            inner_val = zero(T)
            outer_val = zero(T)

            # For no-slip inner (velocity_bc_code ≤ 2) with rotating IC:
            # T(l=1,m=0) at inner boundary = rot_omega * r[1]
            if (velocity_bc_code == 1 || velocity_bc_code == 2) && l == 1 && m == 0 &&
               domain !== nothing
                inner_val = T(rot_omega * domain.r[1, 4])
                # If not first step, subtract current field value (incremental form)
                if current_field !== nothing
                    cur_real = parent(current_field.data_real)
                    inner_val -= local_spectral_value(cur_real, slot, 1)
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
            set_local_spectral_value!(sol_real, slot, ir, tmp_r[ir])
            set_local_spectral_value!(sol_imag, slot, ir, tmp_i[ir])
        end
    end

    return solution
end

"""
    create_velocity_poloidal_split_matrices(config, domain, Ek, dt; velocity_bc_code, theta=0.5, ball=false, T=Float64)

Stage-4B W-split operators for the pressure-free poloidal momentum equation
    Ek(∂t − D_pol)W = N_W,   W = D_pol·P,   D_pol = ∂_rr − l(l+1)/r²
with P-recovery using Dirichlet rows at both walls (shell) or a Robin
center-regularity inner row (`ball=true`), and cached no-slip influence
responses enforcing P′=0 (velocity_bc_code == 1). Other BC codes error loudly
until ported. See docs/superpowers/plans/2026-06-10-double-curl-stage4b-*.md.

Ball mode (`ball = true`, full sphere with the off-center grid r₁ > 0): the
inner wall conditions become center-regularity Robin rows exact for the
leading behaviors P ~ r^{l+1} and W ~ r^{l+1}. The P-recovery inner row is
the Robin P′(r₁) = (l+1)P(r₁)/r₁ (outer Dirichlet wall unchanged), and the
influence matrix mixes spaces: row 1 = the W-regularity residual
W′(r₁) − (l+1)W(r₁)/r₁ applied to the W-space Green columns g_i (the
condition lives on W = D_pol·P), row 2 = the outer wall residual on the
recovered P responses h_i (as shell). `d1_row_inner` then holds the PURE
first-derivative endpoint row; the l-dependent −(l+1)/r₁ correction is
applied at dot time via `reg_r_inv`.
"""
function create_velocity_poloidal_split_matrices(config::SHTnsKitConfig,
        domain::RadialDomain,
        Ek::Float64,
        dt::Float64;
        velocity_bc_code::Int,
        theta::Float64 = 0.5,
        ball::Bool = false,
        T::Type{<:Number} = Float64)
    velocity_bc_code in (1, 2, 3, 4) || error(
        "poloidal W-split: unknown velocity_bc_code $velocity_bc_code")

    unique_l = unique(config.l_values)
    nL = length(unique_l)
    d2 = create_derivative_matrix(T, 2, domain)
    d1 = create_derivative_matrix(T, 1, domain)
    r_inv_sq = @views domain.r[1:domain.N, 2]
    bw = radial_bandwidth(domain)
    N = domain.N
    reg_r_inv = ball ? domain.r[1, 3] : 0.0
    inv_dt = T(Ek / dt)
    θ_T = T(theta)

    dpol_op = Vector{BandedMatrix{T}}(undef, nL)
    w_factor = Vector{BandedLU{T}}(undef, nL)
    w_linear = Vector{BandedMatrix{T}}(undef, nL)
    p_factor = Vector{BandedLU{T}}(undef, nL)
    g1 = Vector{Vector{T}}(undef, nL); g2 = Vector{Vector{T}}(undef, nL)
    h1 = Vector{Vector{T}}(undef, nL); h2 = Vector{Vector{T}}(undef, nL)
    influence = Vector{Matrix{T}}(undef, nL)
    l_values = Vector{Int}(undef, nL)
    lookup = Dict{Int, Int}()

    # Endpoint residual rows for the influence corrections. No-slip walls
    # enforce P′ = 0 (first-derivative row); stress-free walls enforce
    # P″ − (2/r)·P′ = 0 (from σ_rθ ∝ r·∂_r(u_θ/r) with u_tan = (P′/r)∇₁Y).
    inner_noslip = velocity_bc_code in (1, 2)   # codes: 1 NS/NS, 2 NS/SF, 3 SF/NS, 4 SF/SF
    outer_noslip = velocity_bc_code in (1, 3)
    d1_row_inner = zeros(T, N)
    d1_row_outer = zeros(T, N)
    @inbounds for j in 1:N
        if 1 <= bw + 1 + 1 - j <= 2bw + 1
            v1 = d1.data[bw + 1 + 1 - j, j]
            d1_row_inner[j] = ball ? v1 :
                              (inner_noslip ? v1 :
                               d2.data[bw + 1 + 1 - j, j] - T(2 / domain.r[1, 4]) * v1)
        end
        if 1 <= bw + 1 + N - j <= 2bw + 1
            vN = d1.data[bw + 1 + N - j, j]
            d1_row_outer[j] = outer_noslip ? vN :
                              d2.data[bw + 1 + N - j, j] - T(2 / domain.r[N, 4]) * vN
        end
    end

    e1 = zeros(T, N); e1[1] = one(T)
    eN = zeros(T, N); eN[N] = one(T)
    gtmp = Vector{T}(undef, N)
    htmp = Vector{T}(undef, N)

    for (idx, l) in enumerate(unique_l)
        l_values[idx] = l
        lookup[l] = idx
        λ = Float64(l * (l + 1))

        # D_pol per l
        dpol_data = copy(d2.data)
        @inbounds for n in 1:N
            dpol_data[bw + 1, n] -= T(λ * r_inv_sq[n])
        end
        dpol = BandedMatrix{T}(dpol_data, bw, N)
        dpol_op[idx] = dpol

        # Ek·D_pol (explicit CN term)
        w_linear[idx] = BandedMatrix{T}(T(Ek) .* dpol_data, bw, N)

        # W-advance system: (Ek/dt)I − θ·Ek·D_pol  — PDE rows everywhere
        wsys_data = (-θ_T * T(Ek)) .* dpol_data
        wsys_data[bw + 1, :] .+= inv_dt
        wsys = BandedMatrix{T}(wsys_data, bw, N)
        w_factor[idx] = factorize_banded(wsys)

        # P-recovery: D_pol with Dirichlet rows (rows 1, N → identity).
        # Ball: the inner row is the center-regularity Robin instead.
        prec_data = copy(dpol_data)
        @inbounds for j in 1:(1 + bw)
            prec_data[bw + 1 + 1 - j, j] = zero(T)
        end
        @inbounds for j in (N - bw):N
            prec_data[bw + 1 + N - j, j] = zero(T)
        end
        if ball
            # Center regularity: P ~ r^{l+1} ⇒ P′(r₁) = (l+1)·P(r₁)/r₁.
            @inbounds for j in 1:(1 + bw)
                prec_data[bw + 1 + 1 - j, j] = d1.data[bw + 1 + 1 - j, j]
            end
            prec_data[bw + 1, 1] -= T((l + 1) * domain.r[1, 3])
        else
            prec_data[bw + 1, 1] = one(T)
        end
        prec_data[bw + 1, N] = one(T)
        prec = BandedMatrix{T}(prec_data, bw, N)
        p_factor[idx] = factorize_banded(prec)

        # Influence responses (no-slip): g_i = A_W⁻¹ e_i; h_i = A_P⁻¹ R(g_i)
        # with R zeroing the Dirichlet RHS rows; M[j,i] = endpoint-P′ of h_i.
        gv1 = Vector{T}(undef, N); gv2 = Vector{T}(undef, N)
        hv1 = Vector{T}(undef, N); hv2 = Vector{T}(undef, N)
        solve_banded!(gv1, w_factor[idx], e1)
        solve_banded!(gv2, w_factor[idx], eN)
        copyto!(gtmp, gv1); gtmp[1] = zero(T); gtmp[N] = zero(T)
        solve_banded!(hv1, p_factor[idx], gtmp)
        copyto!(htmp, gv2); htmp[1] = zero(T); htmp[N] = zero(T)
        solve_banded!(hv2, p_factor[idx], htmp)
        g1[idx] = gv1; g2[idx] = gv2; h1[idx] = hv1; h2[idx] = hv2
        M = Matrix{T}(undef, 2, 2)
        if ball
            # Mixed rows: row 1 = inner W-regularity residual evaluated on the
            # W-space Green columns g_i (the condition lives on W = D_pol·P);
            # row 2 = outer wall residual on the recovered P responses h_i.
            M[1, 1] = dot(d1_row_inner, gv1) - T((l + 1) * reg_r_inv) * gv1[1]
            M[1, 2] = dot(d1_row_inner, gv2) - T((l + 1) * reg_r_inv) * gv2[1]
        else
            M[1, 1] = dot(d1_row_inner, hv1)
            M[1, 2] = dot(d1_row_inner, hv2)
        end
        M[2, 1] = dot(d1_row_outer, hv1)
        M[2, 2] = dot(d1_row_outer, hv2)
        influence[idx] = M
    end

    return PoloidalSplitMatrices{T}(dpol_op, w_factor, w_linear, p_factor,
        g1, g2, h1, h2, influence, d1_row_inner, d1_row_outer,
        l_values, lookup, theta, Ek, ball, reg_r_inv,
        ntuple(_ -> Vector{T}(undef, N), 6))
end
