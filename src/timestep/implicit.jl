# ================================================================================
# CNAB2 (Crank-Nicolson Adams-Bashforth 2nd order) Scheme
# ================================================================================

"""
    SHTnsImplicitMatrices{T}

Precomputed linear operators for one implicit timestep solve family.

The matrices are stored per spherical-harmonic degree `l` so CNAB2/IMEX style
updates can reuse banded system matrices and their factorizations across many
timesteps.
"""
struct SHTnsImplicitMatrices{T}
    system_matrices::Vector{BandedMatrix{T}}  # (1/dt)I − θ·L per l
    factorizations::Vector{BandedLU{T}}       # Banded LU factorizations
    linear_matrices::Vector{BandedMatrix{T}}  # Linear operator L per l (scaled by diffusivity)
    l_values::Vector{Int}                     # l values for indexing
    lookup::Dict{Int, Int}                     # Map l → index into vectors above
    theta::Float64                            # Crank–Nicolson weight θ
end

"""
    create_shtns_timestepping_matrices(config, domain, diffusivity, dt; theta=0.5, mass_coeff=1.0, T=Float64)

Build the per-`l` implicit matrices used by the standard scalar/vector timestep
operators.

The result includes the linear operator, the shifted system matrix, and a
banded factorization for each spherical-harmonic degree present in `config`.
"""
function create_shtns_timestepping_matrices(config::SHTnsKitConfig,
        domain::RadialDomain,
        diffusivity::Float64,
        dt::Float64;
        theta::Float64 = 0.5,
        mass_coeff::Float64 = 1.0,
        T::Type{<:Number} = Float64)
    unique_l = unique(config.l_values)
    laplacian = create_radial_laplacian(domain)
    r_inv_sq = @views domain.r[1:domain.N, 2]
    bandwidth = radial_bandwidth(domain)

    base_data = T.(diffusivity .* laplacian.data)
    system_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    linear_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    factorizations = Vector{BandedLU{T}}(undef, length(unique_l))
    l_values = Vector{Int}(undef, length(unique_l))
    lookup = Dict{Int, Int}()

    # Mass coefficient: c1/dt (Fortran convention: c1=Ek for velocity, c1=1 for others)
    inv_dt = T(mass_coeff / dt)
    θ_T = T(theta)
    minus_θ = -θ_T

    for (idx, l) in enumerate(unique_l)
        l_values[idx] = l
        lookup[l] = idx

        linear_data = copy(base_data)
        l_factor = Float64(l * (l + 1))
        @inbounds for n in 1:domain.N
            linear_data[bandwidth + 1, n] -= T(diffusivity * l_factor * r_inv_sq[n])
        end

        linear_matrix = BandedMatrix{T}(copy(linear_data), bandwidth, domain.N)

        system_data = copy(linear_data)
        system_data .*= minus_θ
        system_data[bandwidth + 1, :] .+= inv_dt
        system_matrix = BandedMatrix{T}(system_data, bandwidth, domain.N)

        system_matrices[idx] = system_matrix
        linear_matrices[idx] = linear_matrix
        factorizations[idx] = factorize_banded(system_matrix)
    end

    return SHTnsImplicitMatrices{T}(system_matrices, factorizations,
        linear_matrices, l_values, lookup, theta)
end

# Velocity-specific matrix construction with embedded BCs is in src/bcs/velocity_bc.jl

"""
    banded_to_dense(matrix)

Materialize a `BandedMatrix` as a dense matrix.

This is mainly a debugging/compatibility helper; production timestep solves
prefer banded operators and factorizations.
"""
function banded_to_dense(matrix::BandedMatrix{T}) where {T}
    # Convert banded matrix to dense for LU factorization
    N = matrix.size
    bandwidth = matrix.bandwidth
    dense = zeros(T, N, N)

    for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            dense[i, j] = matrix.data[band_row, j]
        end
    end

    return dense
end

"""
    apply_explicit_operator!(output, input, nonlinear, domain, diffusivity, dt; nl_prev=nothing, matrices=nothing)

Assemble the explicit right-hand side for one timestep update.

When `nl_prev` and `matrices` are supplied this builds the CNAB2/IMEX right-hand
side. Otherwise it falls back to a simpler backward-Euler style explicit term
used by older paths and tests.
"""
function apply_explicit_operator!(output::SHTnsSpecField{T},
        input::SHTnsSpecField{T},
        nonlinear::SHTnsSpecField{T},
        domain::RadialDomain,
        diffusivity::Float64,
        dt::Float64;
        nl_prev::Union{SHTnsSpecField{T}, Nothing} = nothing,
        matrices::Union{SHTnsImplicitMatrices{T}, Nothing} = nothing) where {T}
    if nl_prev !== nothing && matrices !== nothing
        build_rhs_cnab2!(output, input, nonlinear, nl_prev, dt, matrices)
        return output
    end

    # Fallback: backward Euler style explicit operator without linear correction.
    out_real = parent(output.data_real)
    out_imag = parent(output.data_imag)
    in_real = parent(input.data_real)
    in_imag = parent(input.data_imag)
    nl_real = parent(nonlinear.data_real)
    nl_imag = parent(nonlinear.data_imag)

    lm_range = local_spectral_mode_indices(input.config)
    r_range = get_local_range(input.pencil, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= input.nlm
            slot = local_spectral_storage_slot(input.config, lm_idx)
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(out_real, 3)
                    set_local_spectral_value!(
                        out_real,
                        slot,
                        local_r,
                        local_spectral_value(in_real, slot, local_r) / dt +
                        local_spectral_value(nl_real, slot, local_r)
                    )
                    set_local_spectral_value!(
                        out_imag,
                        slot,
                        local_r,
                        local_spectral_value(in_imag, slot, local_r) / dt +
                        local_spectral_value(nl_imag, slot, local_r)
                    )
                end
            end
        end
    end

    return output
end

"""
    build_rhs_cnab2!(rhs, un, nl, nl_prev, dt, matrices)

Build RHS for CNAB2 IMEX: rhs = un/dt + (1-θ)·L·un + (3/2)·nl − (1/2)·nl_prev,
where θ = matrices.theta and L is the diffusivity-scaled linear operator.

# MPI Safety
Uses global loop bounds (1:nlm) to ensure all processes call Allreduce
the same number of times, preventing deadlock with uneven lm distribution.
"""
function build_rhs_cnab2!(rhs::SHTnsSpecField{T}, un::SHTnsSpecField{T},
        nl::SHTnsSpecField{T}, nl_prev::SHTnsSpecField{T},
        dt::Float64, matrices::SHTnsImplicitMatrices{T};
        mass_coeff::Float64 = 1.0) where {T}
    r_real = parent(rhs.data_real);
    r_imag = parent(rhs.data_imag)
    u_real = parent(un.data_real);
    u_imag = parent(un.data_imag)
    n_real = parent(nl.data_real);
    n_imag = parent(nl.data_imag)
    p_real = parent(nl_prev.data_real);
    p_imag = parent(nl_prev.data_imag)

    r_range = get_local_range(un.pencil, 3)

    # Mass coefficient: c1/dt (Fortran convention: c1=Ek for velocity, c1=1 for others)
    inv_dt = T(mass_coeff / dt)
    three_halves = T(1.5)
    half = T(0.5)

    θ_T = T(matrices.theta)
    linear_weight = one(T) - θ_T
    add_linear = !iszero(linear_weight)

    nr_global = add_linear ? matrices.system_matrices[1].size : 0
    ur = add_linear ? zeros(T, nr_global) : T[]
    ui = add_linear ? zeros(T, nr_global) : T[]
    lin_r = add_linear ? zeros(T, nr_global) : T[]
    lin_i = add_linear ? zeros(T, nr_global) : T[]

    nlm_total = un.nlm

    # Each (l,m) mode owns a COMPLETE radial profile on a single rank (the spectral
    # pencil keeps the radial dimension local), so the CNAB2 RHS is assembled with
    # no inter-rank communication; non-owners skip the mode entirely. Ownership is
    # read from the mode's storage slot — NOT a rectangular l-index range, which
    # silently dropped compact modes lm > lmax+1.
    @inbounds for lm_idx in 1:nlm_total
        slot = local_spectral_storage_slot(un.config, lm_idx)
        slot === nothing && continue

        l = un.config.l_values[lm_idx]

        if add_linear
            idx = get(matrices.lookup, l, nothing)
            idx === nothing && error("Missing implicit matrix for l=$l")
            fill!(ur, zero(T));
            fill!(ui, zero(T))
            gather_local_radial_profile!(ur, ui, u_real, u_imag, slot, r_range)
            fill!(lin_r, zero(T));
            fill!(lin_i, zero(T))
            apply_banded_full!(lin_r, matrices.linear_matrices[idx], ur)
            apply_banded_full!(lin_i, matrices.linear_matrices[idx], ui)
        end

        for r in r_range
            lr = r - first(r_range) + 1
            lr > size(r_real, 3) && continue

            value_real = inv_dt * local_spectral_value(u_real, slot, lr) +
                         three_halves * local_spectral_value(n_real, slot, lr) -
                         half * local_spectral_value(p_real, slot, lr)
            value_imag = inv_dt * local_spectral_value(u_imag, slot, lr) +
                         three_halves * local_spectral_value(n_imag, slot, lr) -
                         half * local_spectral_value(p_imag, slot, lr)

            if add_linear
                value_real += linear_weight * lin_r[r]
                value_imag += linear_weight * lin_i[r]
            end

            set_local_spectral_value!(r_real, slot, lr, value_real)
            set_local_spectral_value!(r_imag, slot, lr, value_imag)
        end
    end

    return rhs
end

"""
    solve_implicit_step!(solution, rhs, matrices)

Solve the implicit banded radial systems for every spherical-harmonic mode in
`rhs` and write the updated coefficients into `solution`.
"""
function solve_implicit_step!(solution::SHTnsSpecField{T},
        rhs::SHTnsSpecField{T},
        matrices::SHTnsImplicitMatrices{T}) where {T}
    sol_real = parent(solution.data_real)
    sol_imag = parent(solution.data_imag)
    rhs_real = parent(rhs.data_real)
    rhs_imag = parent(rhs.data_imag)

    lm_range = local_spectral_mode_indices(solution.config)
    r_range = get_local_range(solution.pencil, 3)
    nr = matrices.system_matrices[1].size  # Full radial size (local = global for spectral)

    # Allocate buffers for the radial profile
    tmp_r = Vector{T}(undef, nr)
    tmp_i = Vector{T}(undef, nr)

    # Loop over local lm modes only (radial is fully local, matching DD_2DCODE)
    @inbounds for lm_idx in lm_range
        slot = local_spectral_storage_slot(solution.config, lm_idx)

        l = solution.config.l_values[lm_idx]
        idx = get(matrices.lookup, l, nothing)
        idx === nothing && continue

        # Copy RHS radial profile to work buffer
        fill!(tmp_r, zero(T))
        fill!(tmp_i, zero(T))
        gather_local_radial_profile!(tmp_r, tmp_i, rhs_real, rhs_imag, slot, r_range)

        # Solve the banded system (matching Fortran tim_invX)
        solve_banded!(tmp_r, matrices.factorizations[idx], tmp_r)
        solve_banded!(tmp_i, matrices.factorizations[idx], tmp_i)

        # Store solution back
        scatter_local_radial_profile!(sol_real, sol_imag, tmp_r, tmp_i, slot, r_range)
    end

    return solution
end

"""
    compute_timestep_error(new_field, old_field)

Compute a simple relative update norm between two spectral fields.

This is primarily used by timestep-control and diagnostics code to quantify how
large one update was compared with the previous state.
"""
function compute_timestep_error(new_field::SHTnsSpecField{T},
        old_field::SHTnsSpecField{T}) where {T}
    error = zero(Float64)

    # Get local data
    new_real = parent(new_field.data_real)
    new_imag = parent(new_field.data_imag)
    old_real = parent(old_field.data_real)
    old_imag = parent(old_field.data_imag)

    # Compute local error with bounds checking for PencilArrays
    @inbounds for idx in eachindex(new_real, old_real)
        diff_real = new_real[idx] - old_real[idx]
        diff_imag = new_imag[idx] - old_imag[idx]
        error += diff_real^2 + diff_imag^2
    end

    # Global reduction across all MPI processes
    global_error = Allreduce(error, MPI.SUM, get_comm())
    return sqrt(global_error)
end

"""
    synchronize_pencil_transforms!(field::SHTnsSpecField{T}) where T

Ensure all pending PencilFFTs operations are completed and data is consistent across processes.
"""
function synchronize_pencil_transforms!(field::SHTnsSpecField{T}) where {T}
    # NOTE: MPI.Barrier only ensures all ranks reach this point; it does NOT
    # flush pending non-blocking communications. PencilFFTs operations should
    # be fully blocking by the time this is called. If async operations are
    # introduced in the future, replace this with proper completion (MPI.Waitall).
    MPI.Barrier(get_comm())
    return field
end

"""
    validate_mpi_consistency!(field::SHTnsSpecField{T}) where T

Check that spectral field data is valid (no NaN/Inf) across all MPI processes.
Returns the field after validation. Warns if any process has invalid data.
"""
function validate_mpi_consistency!(field::SHTnsSpecField{T}) where {T}
    comm = get_comm()
    rank = get_rank()

    # Check local data for NaN/Inf
    real_data = parent(field.data_real)
    imag_data = parent(field.data_imag)

    local_nan_count = count(isnan, real_data) + count(isnan, imag_data)
    local_inf_count = count(isinf, real_data) + count(isinf, imag_data)
    local_has_issues = (local_nan_count > 0 || local_inf_count > 0) ? 1 : 0

    # Check if any process has issues
    global_has_issues = Allreduce(local_has_issues, MPI.MAX, comm)

    if global_has_issues > 0
        # Gather counts from all processes for detailed reporting
        total_nan = Allreduce(local_nan_count, MPI.SUM, comm)
        total_inf = Allreduce(local_inf_count, MPI.SUM, comm)

        if rank == 0
            @warn "MPI data validation failed: $total_nan NaN values, $total_inf Inf values across all processes"
        end
    end

    return field
end
