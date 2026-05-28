"""
    solver_build_rhs_cnab2!(rhs, uₙ, nₙ, nₙ₋₁, Δt, matrices; mass_coeff=1.0)

Build the CNAB2 right-hand side inside the flattened solver layer in `src/`
so the solver step reads
as a self-contained algorithm instead of reaching back into the legacy
timestep entry points.
"""
function solver_build_rhs_cnab2!(
    rhs::SpectralFieldType{T},
    uₙ::SpectralFieldType{T},
    nₙ::SpectralFieldType{T},
    nₙ₋₁::SpectralFieldType{T},
    Δt::Float64,
    matrices::ImplicitMatrixSet{T};
    mass_coeff::Float64=1.0,
    work::Union{SolverRadialWork{T}, Nothing}=nothing,
) where T
    rhs_real = parent(rhs.data_real)
    rhs_imag = parent(rhs.data_imag)
    u_real = parent(uₙ.data_real)
    u_imag = parent(uₙ.data_imag)
    n_real = parent(nₙ.data_real)
    n_imag = parent(nₙ.data_imag)
    p_real = parent(nₙ₋₁.data_real)
    p_imag = parent(nₙ₋₁.data_imag)

    r_range = local_range(uₙ.pencil, 3)

    inv_Δt = T(mass_coeff / Δt)
    three_halves = T(1.5)
    one_half = T(0.5)
    θ = T(matrices.theta)
    linear_weight = one(T) - θ
    add_linear = !iszero(linear_weight)

    nr_global = add_linear ? matrices.system_matrices[1].size : 0
    work_ok = work !== nothing && length(work.u_real_global) == nr_global
    u_real_global = add_linear ? (work_ok ? work.u_real_global : zeros(T, nr_global)) : T[]
    u_imag_global = add_linear ? (work_ok ? work.u_imag_global : zeros(T, nr_global)) : T[]
    linear_real = add_linear ? (work_ok ? work.linear_real : zeros(T, nr_global)) : T[]
    linear_imag = add_linear ? (work_ok ? work.linear_imag : zeros(T, nr_global)) : T[]

    comm = mpi_comm()
    multi_rank = mpi_comm_size(comm) > 1

    @inbounds for lm_idx in 1:uₙ.nlm
        slot = local_spectral_storage_slot(uₙ.config, lm_idx)
        ℓ = uₙ.config.l_values[lm_idx]
        matrix_idx = add_linear ? get(matrices.lookup, ℓ, nothing) : nothing

        if add_linear
            # The implicit operator is assembled as a full radial system for each
            # degree ℓ, so local pencil data is gathered into a global radial view
            # before applying the linear contribution.
            matrix_idx === nothing && error("Missing implicit matrix for ℓ=$ℓ")
            fill!(u_real_global, zero(T))
            fill!(u_imag_global, zero(T))

            if slot !== nothing
                gather_local_radial_profile!(u_real_global, u_imag_global, u_real, u_imag, slot, r_range)
            end

            if multi_rank
                allreduce_sum_in_place!(u_real_global, comm)
                allreduce_sum_in_place!(u_imag_global, comm)
            end

            fill!(linear_real, zero(T))
            fill!(linear_imag, zero(T))
            apply_banded_full!(
                linear_real,
                matrices.linear_matrices[matrix_idx],
                u_real_global,
            )
            apply_banded_full!(
                linear_imag,
                matrices.linear_matrices[matrix_idx],
                u_imag_global,
            )
        end

        slot === nothing && continue

        # CNAB2 combines the current state, current nonlinear term, previous
        # nonlinear term, and optional explicit linear carry-over into the RHS
        # consumed by the implicit solve.
        for r_idx in r_range
            local_r = r_idx - first(r_range) + 1
            local_r <= size(rhs_real, 3) || continue

            rhs_value_real =
                inv_Δt * local_spectral_value(u_real, slot, local_r) +
                three_halves * local_spectral_value(n_real, slot, local_r) -
                one_half * local_spectral_value(p_real, slot, local_r)
            rhs_value_imag =
                inv_Δt * local_spectral_value(u_imag, slot, local_r) +
                three_halves * local_spectral_value(n_imag, slot, local_r) -
                one_half * local_spectral_value(p_imag, slot, local_r)

            if add_linear
                rhs_value_real += linear_weight * linear_real[r_idx]
                rhs_value_imag += linear_weight * linear_imag[r_idx]
            end

            set_local_spectral_value!(rhs_real, slot, local_r, rhs_value_real)
            set_local_spectral_value!(rhs_imag, slot, local_r, rhs_value_imag)
        end
    end

    return rhs
end

function get_radial_work!(
    caches::TimestepCaches{T},
    key::Symbol,
    nr::Int,
) where T
    # Each field/update family gets one scratch bundle sized for its radial
    # operator. Recreate it only when resolution or operator size changes.
    work = get(caches.radial_work, key, nothing)
    if work === nothing || length(work.tmp_real) != nr
        work = SolverRadialWork{T}(nr)
        caches.radial_work[key] = work
    end
    return work
end

@inline function boundary_mode_value(mode_values, lm_idx::Int)
    return mode_values !== nothing && lm_idx <= length(mode_values) ? mode_values[lm_idx] : nothing
end

struct SolverBandedAction{T}
    operator::BandedOperator{T}
end

@inline function (action::SolverBandedAction{T})(out::Vector{T}, v::AbstractVector{T}) where T
    apply_banded_full!(out, action.operator, v)
    return nothing
end

"""
    solver_get_eab2_alu_cache!(caches, key, ν, T, domain)

Typed EAB2 cache lookup for the new solver path.
"""
function solver_get_eab2_alu_cache!(
    caches::Dict{Symbol, EAB2CacheEntry{T}},
    key::Symbol,
    ν::Float64,
    ::Type{T},
    domain::RadialDomainType,
) where T
    entry = get(caches, key, nothing)
    nr = domain.N
    if entry === nothing || entry.ν != ν || entry.nr != nr
        entry = EAB2CacheEntry{T}(
            ν,
            nr,
            Dict{Int, Tuple{BandedOperator{T}, BandedFactorization{T}}}(),
        )
        caches[key] = entry
    end
    return entry.map
end

"""
    _ensure_etd_cache!(caches, field, ν, T, domain)

Ensure a typed EAB2/ETD cache entry exists for one solver field.

The cache is rebuilt when diffusivity or radial resolution changes; otherwise
the existing per-degree operator/factorization map is reused.
"""
function _ensure_etd_cache!(
    caches::TimestepCaches{T},
    field::Symbol,
    ν::Float64,
    ::Type{T},
    domain::RadialDomainType,
) where T
    entry = getfield(caches, field)
    nr = domain.N
    if entry === nothing || entry.ν != ν || entry.nr != nr
        entry = EAB2CacheEntry{T}(
            ν,
            nr,
            Dict{Int, Tuple{BandedOperator{T}, BandedFactorization{T}}}(),
        )
        setfield!(caches, field, entry)
    end
    return (entry::EAB2CacheEntry{T}).map
end

"""
    solver_eab2_update_krylov_cached!(u, nₙ, nₙ₋₁, alu_map, domain, ν, config, Δt; ...)

Solver-local EAB2 update so the new timestep code does not reach through the
legacy timestep API for its exponential update.

The solve gathers a full radial profile per spectral mode, applies the
exponential linear action plus the phi1 nonlinear correction, then re-applies
any endpoint BCs before scattering back to the local pencil storage.
"""
function solver_eab2_update_krylov_cached!(
    u::SpectralFieldType{T},
    nₙ::SpectralFieldType{T},
    nₙ₋₁::SpectralFieldType{T},
    alu_map::Dict{Int, Tuple{BandedOperator{T}, BandedFactorization{T}}},
    domain::RadialDomainType,
    diffusivity::Float64,
    config::SHTnsConfigType,
    Δt::Float64;
    m::Int=20,
    tol::Float64=1e-8,
    mass_coeff::Float64=1.0,
    bc_spec=nothing,
    krylov_work::Union{SolverRadialWork{T}, Nothing}=nothing,
) where T
    u_real = parent(u.data_real)
    u_imag = parent(u.data_imag)
    n_real = parent(nₙ.data_real)
    n_imag = parent(nₙ.data_imag)
    p_real = parent(nₙ₋₁.data_real)
    p_imag = parent(nₙ₋₁.data_imag)

    r_range = local_range(u.pencil, 3)
    nr = domain.N

    comm = mpi_comm()
    multi_rank = mpi_comm_size(comm) > 1

    work_ok = krylov_work !== nothing && length(krylov_work.u_real_global) == nr
    u_real_global = work_ok ? krylov_work.u_real_global : zeros(T, nr)
    u_imag_global = work_ok ? krylov_work.u_imag_global : zeros(T, nr)
    nl_real_global = work_ok ? krylov_work.linear_real : zeros(T, nr)
    nl_imag_global = work_ok ? krylov_work.linear_imag : zeros(T, nr)
    u_real_next = work_ok ? krylov_work.tmp_real : Vector{T}(undef, nr)
    u_imag_next = work_ok ? krylov_work.tmp_imag : Vector{T}(undef, nr)
    krylov_action_work = work_ok ? krylov_work.krylov : nothing
    inv_mass_coeff = T(inv(mass_coeff))

    for lm_idx in 1:u.nlm
        slot = local_spectral_storage_slot(u.config, lm_idx)
        ℓ = config.l_values[lm_idx]
        operator_entry = get(alu_map, ℓ, nothing)
        if operator_entry === nothing
            # EAB2 reuses the same per-degree radial operator across all modes
            # with the same ℓ, so cache the factorization once here.
            operator_matrix = solver_build_banded_A(T, domain, diffusivity / mass_coeff, ℓ)
            operator_lu = solver_factorize_banded(operator_matrix)
            operator_entry = (operator_matrix, operator_lu)
            alu_map[ℓ] = operator_entry
        end
        operator_matrix, operator_lu = operator_entry

        fill!(u_real_global, zero(T))
        fill!(u_imag_global, zero(T))
        fill!(nl_real_global, zero(T))
        fill!(nl_imag_global, zero(T))

        if slot !== nothing
            gather_local_radial_profile!(u_real_global, u_imag_global, u_real, u_imag, slot, r_range)
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                local_r <= size(n_real, 3) || continue
                nl_real_global[r_idx] =
                    inv_mass_coeff * (
                        T(1.5) * local_spectral_value(n_real, slot, local_r) -
                        T(0.5) * local_spectral_value(p_real, slot, local_r)
                    )
                nl_imag_global[r_idx] =
                    inv_mass_coeff * (
                        T(1.5) * local_spectral_value(n_imag, slot, local_r) -
                        T(0.5) * local_spectral_value(p_imag, slot, local_r)
                    )
            end
        end

        if multi_rank
            allreduce_sum_in_place!(u_real_global, comm)
            allreduce_sum_in_place!(u_imag_global, comm)
            allreduce_sum_in_place!(nl_real_global, comm)
            allreduce_sum_in_place!(nl_imag_global, comm)
        end

        # The exponential action advances the linear part, while phi₁(AΔt)
        # applies the matching correction to the Adams-Bashforth nonlinear term.
        Aop = SolverBandedAction(operator_matrix)

        exp_action_krylov!(u_real_next, Aop, u_real_global, Δt; m, tol, work=krylov_action_work)
        exp_action_krylov!(u_imag_next, Aop, u_imag_global, Δt; m, tol, work=krylov_action_work)
        phi1_action_krylov!(nl_real_global, Aop, operator_lu, nl_real_global, Δt; m, tol, work=krylov_action_work)
        phi1_action_krylov!(nl_imag_global, Aop, operator_lu, nl_imag_global, Δt; m, tol, work=krylov_action_work)
        @. u_real_next = u_real_next + Δt * nl_real_global
        @. u_imag_next = u_imag_next + Δt * nl_imag_global

        if bc_spec !== nothing
            # Krylov actions operate on the full radial vector and can move the
            # endpoint rows away from their matrix-embedded constraints, so the
            # accepted profile is projected back onto the requested BCs here.
            inner_val = boundary_mode_value(bc_spec.inner_mode_values, lm_idx)
            outer_val = boundary_mode_value(bc_spec.outer_mode_values, lm_idx)
            inner_val_i = boundary_mode_value(bc_spec.inner_mode_values_imag, lm_idx)
            outer_val_i = boundary_mode_value(bc_spec.outer_mode_values_imag, lm_idx)
            solver_enforce_erk2_bc!(u_real_next, bc_spec.inner, 1, ℓ, nr; value_override=inner_val)
            solver_enforce_erk2_bc!(u_real_next, bc_spec.outer, nr, ℓ, nr; value_override=outer_val)
            solver_enforce_erk2_bc!(u_imag_next, bc_spec.inner, 1, ℓ, nr; value_override=inner_val_i)
            solver_enforce_erk2_bc!(u_imag_next, bc_spec.outer, nr, ℓ, nr; value_override=outer_val_i)
        end

        slot === nothing && continue
        scatter_local_radial_profile!(u_real, u_imag, u_real_next, u_imag_next, slot, r_range)
    end

    return u
end

"""
    _solver_solve_scalar_implicit_step!(solution, rhs, matrices; bc_inner=nothing, bc_outer=nothing, ...)

Solve one scalar-field implicit radial system for each local spectral mode.

Boundary vectors are mode-indexed global arrays. Missing boundary vectors
default to homogeneous values for both real and imaginary parts.
"""
function _solver_solve_scalar_implicit_step!(
    solution::SpectralFieldType{T},
    rhs::SpectralFieldType{T},
    matrices::ImplicitMatrixSet{T};
    bc_inner::Union{AbstractVector{T}, Nothing}=nothing,
    bc_outer::Union{AbstractVector{T}, Nothing}=nothing,
    bc_inner_imag::Union{AbstractVector{T}, Nothing}=nothing,
    bc_outer_imag::Union{AbstractVector{T}, Nothing}=nothing,
    work::Union{SolverRadialWork{T}, Nothing}=nothing,
) where T
    sol_real = parent(solution.data_real)
    sol_imag = parent(solution.data_imag)
    rhs_real = parent(rhs.data_real)
    rhs_imag = parent(rhs.data_imag)

    lm_range = local_spectral_mode_indices(solution.config)
    r_range = local_range(solution.pencil, 3)
    nr = matrices.system_matrices[1].size
    work_ok = work !== nothing && length(work.tmp_real) == nr
    tmp_real = work_ok ? work.tmp_real : Vector{T}(undef, nr)
    tmp_imag = work_ok ? work.tmp_imag : Vector{T}(undef, nr)

    @inbounds for lm_idx in lm_range
        slot = local_spectral_storage_slot(solution.config, lm_idx)
        ℓ = solution.config.l_values[lm_idx]
        matrix_idx = get(matrices.lookup, ℓ, nothing)
        matrix_idx === nothing && continue

        fill!(tmp_real, zero(T))
        fill!(tmp_imag, zero(T))
        gather_local_radial_profile!(tmp_real, tmp_imag, rhs_real, rhs_imag, slot, r_range)

        # Boundary rows were embedded into `matrices`; these values are the RHS
        # targets for the inner/outer boundary equations for this spectral mode.
        inner_real =
            bc_inner !== nothing && lm_idx <= length(bc_inner) ? bc_inner[lm_idx] : zero(T)
        outer_real =
            bc_outer !== nothing && lm_idx <= length(bc_outer) ? bc_outer[lm_idx] : zero(T)
        inner_imag =
            bc_inner_imag !== nothing && lm_idx <= length(bc_inner_imag) ? bc_inner_imag[lm_idx] : zero(T)
        outer_imag =
            bc_outer_imag !== nothing && lm_idx <= length(bc_outer_imag) ? bc_outer_imag[lm_idx] : zero(T)

        tmp_real[1] = inner_real
        tmp_imag[1] = inner_imag
        tmp_real[nr] = outer_real
        tmp_imag[nr] = outer_imag

        solve_banded!(tmp_real, matrices.factorizations[matrix_idx], tmp_real)
        solve_banded!(tmp_imag, matrices.factorizations[matrix_idx], tmp_imag)

        scatter_local_radial_profile!(sol_real, sol_imag, tmp_real, tmp_imag, slot, r_range)
    end

    return solution
end

"""
    solver_solve_temperature_implicit_step!(solution, rhs, matrices; kwargs...)

Temperature-specific wrapper around the scalar implicit solve.
"""
function solver_solve_temperature_implicit_step!(
    solution::SpectralFieldType{T},
    rhs::SpectralFieldType{T},
    matrices::ImplicitMatrixSet{T};
    bc_inner::Union{AbstractVector{T}, Nothing}=nothing,
    bc_outer::Union{AbstractVector{T}, Nothing}=nothing,
    bc_inner_imag::Union{AbstractVector{T}, Nothing}=nothing,
    bc_outer_imag::Union{AbstractVector{T}, Nothing}=nothing,
    work::Union{SolverRadialWork{T}, Nothing}=nothing,
) where T
    return _solver_solve_scalar_implicit_step!(
        solution,
        rhs,
        matrices;
        bc_inner,
        bc_outer,
        bc_inner_imag,
        bc_outer_imag,
        work,
    )
end

"""
    solver_solve_composition_implicit_step!(solution, rhs, matrices; kwargs...)

Composition-specific wrapper around the scalar implicit solve.
"""
function solver_solve_composition_implicit_step!(
    solution::SpectralFieldType{T},
    rhs::SpectralFieldType{T},
    matrices::ImplicitMatrixSet{T};
    bc_inner::Union{AbstractVector{T}, Nothing}=nothing,
    bc_outer::Union{AbstractVector{T}, Nothing}=nothing,
    bc_inner_imag::Union{AbstractVector{T}, Nothing}=nothing,
    bc_outer_imag::Union{AbstractVector{T}, Nothing}=nothing,
    work::Union{SolverRadialWork{T}, Nothing}=nothing,
) where T
    return _solver_solve_scalar_implicit_step!(
        solution,
        rhs,
        matrices;
        bc_inner,
        bc_outer,
        bc_inner_imag,
        bc_outer_imag,
        work,
    )
end

"""
    solver_solve_velocity_implicit_step!(solution, rhs, matrices, component; ...)

Solve the velocity toroidal or poloidal implicit update with embedded boundary
conditions.

Toroidal velocity handles the optional inner-core rotation correction for the
`l=1, m=0` mode; poloidal velocity uses homogeneous endpoint values here and
is corrected later by the ERK2 influence operator when needed.
"""
function solver_solve_velocity_implicit_step!(
    solution::SpectralFieldType{T},
    rhs::SpectralFieldType{T},
    matrices::ImplicitMatrixSet{T},
    component::Symbol;
    velocity_bc_code::Int=1,
    domain::Union{RadialDomainType, Nothing}=nothing,
    rot_omega::Float64=0.0,
    current_field::Union{SpectralFieldType{T}, Nothing}=nothing,
    work::Union{SolverRadialWork{T}, Nothing}=nothing,
) where T
    sol_real = parent(solution.data_real)
    sol_imag = parent(solution.data_imag)
    rhs_real = parent(rhs.data_real)
    rhs_imag = parent(rhs.data_imag)

    lm_range = local_spectral_mode_indices(solution.config)
    r_range = local_range(solution.pencil, 3)
    nr = matrices.system_matrices[1].size
    work_ok = work !== nothing && length(work.tmp_real) == nr
    tmp_real = work_ok ? work.tmp_real : Vector{T}(undef, nr)
    tmp_imag = work_ok ? work.tmp_imag : Vector{T}(undef, nr)

    @inbounds for lm_idx in lm_range
        slot = local_spectral_storage_slot(solution.config, lm_idx)
        ℓ = solution.config.l_values[lm_idx]
        m = solution.config.m_values[lm_idx]
        matrix_idx = get(matrices.lookup, ℓ, nothing)
        matrix_idx === nothing && continue

        fill!(tmp_real, zero(T))
        fill!(tmp_imag, zero(T))
        gather_local_radial_profile!(tmp_real, tmp_imag, rhs_real, rhs_imag, slot, r_range)

        if component === :toroidal
            inner_real = zero(T)
            outer_real = zero(T)

            if (velocity_bc_code == 1 || velocity_bc_code == 2) && ℓ == 1 && m == 0 && domain !== nothing
                inner_real = T(rot_omega * domain.r[1, 4])
                if current_field !== nothing
                    current_real = parent(current_field.data_real)
                    inner_real -= local_spectral_value(current_real, slot, 1)
                end
            end

            tmp_real[1] = inner_real
            tmp_imag[1] = zero(T)
            tmp_real[nr] = outer_real
            tmp_imag[nr] = zero(T)
        else
            tmp_real[1] = zero(T)
            tmp_imag[1] = zero(T)
            tmp_real[nr] = zero(T)
            tmp_imag[nr] = zero(T)
        end

        solve_banded!(tmp_real, matrices.factorizations[matrix_idx], tmp_real)
        solve_banded!(tmp_imag, matrices.factorizations[matrix_idx], tmp_imag)

        scatter_local_radial_profile!(sol_real, sol_imag, tmp_real, tmp_imag, slot, r_range)
    end

    return solution
end

"""
    solver_solve_magnetic_implicit_step!(solution, rhs, matrices, component; ...)

Solve the magnetic toroidal or poloidal implicit update with embedded endpoint
conditions.

For toroidal magnetic fields the optional inner boundary vector is interpreted
as an imposed boundary increment relative to `prev_bc_inner`.
"""
function solver_solve_magnetic_implicit_step!(
    solution::SpectralFieldType{T},
    rhs::SpectralFieldType{T},
    matrices::ImplicitMatrixSet{T},
    component::Symbol;
    mag_bc_inner::Union{Vector{T}, Nothing}=nothing,
    prev_bc_inner::Union{Vector{T}, Nothing}=nothing,
    mag_bc_inner_imag::Union{Vector{T}, Nothing}=nothing,
    prev_bc_inner_imag::Union{Vector{T}, Nothing}=nothing,
    work::Union{SolverRadialWork{T}, Nothing}=nothing,
) where T
    sol_real = parent(solution.data_real)
    sol_imag = parent(solution.data_imag)
    rhs_real = parent(rhs.data_real)
    rhs_imag = parent(rhs.data_imag)

    lm_range = local_spectral_mode_indices(solution.config)
    r_range = local_range(solution.pencil, 3)
    nr = matrices.system_matrices[1].size
    work_ok = work !== nothing && length(work.tmp_real) == nr
    tmp_real = work_ok ? work.tmp_real : Vector{T}(undef, nr)
    tmp_imag = work_ok ? work.tmp_imag : Vector{T}(undef, nr)

    @inbounds for lm_idx in lm_range
        slot = local_spectral_storage_slot(solution.config, lm_idx)
        ℓ = solution.config.l_values[lm_idx]
        matrix_idx = get(matrices.lookup, ℓ, nothing)
        matrix_idx === nothing && continue

        fill!(tmp_real, zero(T))
        fill!(tmp_imag, zero(T))
        gather_local_radial_profile!(tmp_real, tmp_imag, rhs_real, rhs_imag, slot, r_range)

        # Inner-boundary RHS injection. Used by the conducting-inner-core path
        # for BOTH components (the Robin inner row consumes φ0 as its RHS); the
        # insulating default never supplies these, so it stays homogeneous.
        inner_real = zero(T)
        inner_imag = zero(T)
        if mag_bc_inner !== nothing && lm_idx <= length(mag_bc_inner)
            inner_real = mag_bc_inner[lm_idx]
            if prev_bc_inner !== nothing && lm_idx <= length(prev_bc_inner)
                inner_real -= prev_bc_inner[lm_idx]
            end
        end
        if mag_bc_inner_imag !== nothing && lm_idx <= length(mag_bc_inner_imag)
            inner_imag = mag_bc_inner_imag[lm_idx]
            if prev_bc_inner_imag !== nothing && lm_idx <= length(prev_bc_inner_imag)
                inner_imag -= prev_bc_inner_imag[lm_idx]
            end
        end

        tmp_real[1] = inner_real
        tmp_imag[1] = inner_imag
        tmp_real[nr] = zero(T)
        tmp_imag[nr] = zero(T)

        solve_banded!(tmp_real, matrices.factorizations[matrix_idx], tmp_real)
        solve_banded!(tmp_imag, matrices.factorizations[matrix_idx], tmp_imag)

        scatter_local_radial_profile!(sol_real, sol_imag, tmp_real, tmp_imag, slot, r_range)
    end

    return solution
end

"""
    LegacyDenseETDCache{T}

Dense matrix exponential cache kept for compatibility with older timestep
entry points and tests.
"""
struct LegacyDenseETDCache{T}
    dt::Float64
    l_values::Vector{Int}
    E::Vector{Matrix{T}}
    phi1::Vector{Matrix{T}}
end

"""
    legacy_banded_matrix(A)

Convert the solver-local banded operator representation to the older
`OldBandedMatrix` wrapper expected by legacy ETD helpers.
"""
@inline function legacy_banded_matrix(A::BandedOperator{T}) where {T}
    return OldBandedMatrix{T}(copy(A.data), A.bandwidth, A.size)
end

"""
    legacy_create_etd_cache(T, config, domain, diffusivity, dt)

Build dense exponential and phi1 matrices for the legacy EAB2 implementation.
"""
function legacy_create_etd_cache(
    ::Type{T},
    config::SHTnsConfigType,
    domain::RadialDomainType,
    diffusivity::Float64,
    dt::Float64,
) where {T}
    nr = domain.N
    r_inv2 = @views domain.r[1:nr, 2]
    l_values = unique(config.l_values)
    exp_matrices = Matrix{T}[]
    phi1_matrices = Matrix{T}[]
    radial_laplacian = build_radial_laplacian(T, domain)
    radial_dense = solver_banded_to_dense(radial_laplacian)
    identity_matrix = Matrix{T}(SOLVER_LINEAR_ALGEBRA.I, nr, nr)

    for l in l_values
        operator_dense = T(diffusivity) .* radial_dense
        l_factor = T(l * (l + 1))
        @inbounds for n in 1:nr
            operator_dense[n, n] -= T(diffusivity) * l_factor * T(r_inv2[n])
        end

        exp_matrix = exp(dt .* operator_dense)
        push!(exp_matrices, Matrix{T}(exp_matrix))
        phi1_rhs = (exp_matrix - identity_matrix) / dt
        push!(
            phi1_matrices,
            Matrix{T}(SOLVER_LINEAR_ALGEBRA.lu(operator_dense) \ phi1_rhs),
        )
    end

    return LegacyDenseETDCache{T}(dt, collect(l_values), exp_matrices, phi1_matrices)
end

@inline legacy_create_etd_cache(
    config::SHTnsConfigType,
    domain::RadialDomainType,
    diffusivity::Float64,
    dt::Float64,
) = legacy_create_etd_cache(Float64, config, domain, diffusivity, dt)

"""
    legacy_eab2_update_dense!(u, nl, nl_prev, etd, config, dt)

Apply the older dense EAB2 update using a precomputed `LegacyDenseETDCache`.
"""
function legacy_eab2_update_dense!(
    u::SpectralFieldType{T},
    nl::SpectralFieldType{T},
    nl_prev::SpectralFieldType{T},
    etd::LegacyDenseETDCache{T},
    config::SHTnsConfigType,
    dt::Float64,
) where {T}
    u_real = parent(u.data_real)
    u_imag = parent(u.data_imag)
    n_real = parent(nl.data_real)
    n_imag = parent(nl.data_imag)
    p_real = parent(nl_prev.data_real)
    p_imag = parent(nl_prev.data_imag)
    r_range = local_range(u.pencil, 3)
    nr_full = size(etd.E[1], 1)
    comm = mpi_comm()
    multi_rank = mpi_comm_size(comm) > 1
    linear_real = zeros(T, nr_full)
    linear_imag = similar(linear_real)
    phi_tmp = similar(linear_real)
    u_real_global = zeros(T, nr_full)
    u_imag_global = zeros(T, nr_full)
    nl_real_global = zeros(T, nr_full)
    nl_imag_global = zeros(T, nr_full)

    for lm_idx in 1:u.nlm
        slot = local_spectral_storage_slot(config, lm_idx)
        l = config.l_values[lm_idx]
        l_index = findfirst(==(l), etd.l_values)
        l_index === nothing && error("ETD cache missing l=$l")
        exp_matrix = etd.E[l_index]
        phi1_matrix = etd.phi1[l_index]

        fill!(u_real_global, zero(T))
        fill!(u_imag_global, zero(T))
        fill!(nl_real_global, zero(T))
        fill!(nl_imag_global, zero(T))

        if slot !== nothing
            gather_local_radial_profile!(u_real_global, u_imag_global, u_real, u_imag, slot, r_range)
            @inbounds for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                local_r <= size(n_real, 3) || continue
                nl_real_global[r_idx] =
                    T(1.5) * local_spectral_value(n_real, slot, local_r) -
                    T(0.5) * local_spectral_value(p_real, slot, local_r)
                nl_imag_global[r_idx] =
                    T(1.5) * local_spectral_value(n_imag, slot, local_r) -
                    T(0.5) * local_spectral_value(p_imag, slot, local_r)
            end
        end

        if multi_rank
            allreduce_sum_in_place!(u_real_global, comm)
            allreduce_sum_in_place!(u_imag_global, comm)
            allreduce_sum_in_place!(nl_real_global, comm)
            allreduce_sum_in_place!(nl_imag_global, comm)
        end

        SOLVER_LINEAR_ALGEBRA.mul!(linear_real, exp_matrix, u_real_global)
        SOLVER_LINEAR_ALGEBRA.mul!(phi_tmp, phi1_matrix, nl_real_global)
        @. linear_real = linear_real + dt * phi_tmp

        SOLVER_LINEAR_ALGEBRA.mul!(linear_imag, exp_matrix, u_imag_global)
        SOLVER_LINEAR_ALGEBRA.mul!(phi_tmp, phi1_matrix, nl_imag_global)
        @. linear_imag = linear_imag + dt * phi_tmp

        slot === nothing && continue
        scatter_local_radial_profile!(u_real, u_imag, linear_real, linear_imag, slot, r_range)
    end

    return u
end

@eval GeoDynamo begin
    function legacy_imex_depwarn(name::Symbol, replacement::AbstractString)
        Base.depwarn(
            string(
                name,
                " is deprecated and will be removed in a future release. ",
                replacement,
            ),
            name,
        )
        return nothing
    end

    """
        ETDCache

    Deprecated dense ETD/EAB2 cache alias.

    This compatibility type is retained for existing low-level callers while the
    supported timestep API moves through the solver and timestep drivers.
    """
    const ETDCache = $LegacyDenseETDCache

    """
        EAB2ALUCacheEntry

    Deprecated low-level cache alias for compatibility with older EAB2 code.
    """
    const EAB2ALUCacheEntry = $(EAB2CacheEntry)

    function phi1_action_krylov(
        Aop!,
        A_lu,
        v::Vector{T},
        dt::Float64;
        m::Int=20,
        tol::Float64=1e-8,
    ) where {T}
        legacy_imex_depwarn(:phi1_action_krylov, "Prefer the high-level timestep drivers or `exp_action_krylov` plus the solver banded solve helpers.")
        return solver_phi1_action_krylov(Aop!, A_lu, v, dt; m, tol)
    end

    function create_etd_cache(
        ::Type{T},
        config::$(SHTnsConfigType),
        domain::$(RadialDomainType),
        diffusivity::Float64,
        dt::Float64,
    ) where {T}
        legacy_imex_depwarn(:create_etd_cache, "Use the configured solver timestep drivers instead of constructing dense ETD caches manually.")
        return legacy_create_etd_cache(T, config, domain, diffusivity, dt)
    end

    function create_etd_cache(
        config::$(SHTnsConfigType),
        domain::$(RadialDomainType),
        diffusivity::Float64,
        dt::Float64,
    )
        legacy_imex_depwarn(:create_etd_cache, "Use the configured solver timestep drivers instead of constructing dense ETD caches manually.")
        return legacy_create_etd_cache(config, domain, diffusivity, dt)
    end

    function build_banded_A(
        ::Type{T},
        domain::$(RadialDomainType),
        diffusivity::Float64,
        l::Int,
    ) where {T}
        legacy_imex_depwarn(:build_banded_A, "Use the timestep matrix builders or solver-owned banded operators instead.")
        return legacy_banded_matrix(solver_build_banded_A(T, domain, diffusivity, l))
    end

    function eab2_update_krylov!(
        u::$(SpectralFieldType){T},
        nl::$(SpectralFieldType){T},
        nl_prev::$(SpectralFieldType){T},
        domain::$(RadialDomainType),
        diffusivity::Float64,
        config::$(SHTnsConfigType),
        dt::Float64;
        m::Int=20,
        tol::Float64=1e-8,
    ) where {T}
        legacy_imex_depwarn(:eab2_update_krylov!, "Use the high-level solver/timestep drivers instead of the legacy low-level EAB2 API.")
        alu_map = Dict{Int, Tuple{$(BandedOperator){T}, $(BandedFactorization){T}}}()
        return solver_eab2_update_krylov_cached!(u, nl, nl_prev, alu_map, domain, diffusivity, config, dt; m, tol)
    end

    function get_eab2_alu_cache!(
        caches::Dict{Symbol, $(EAB2CacheEntry){T}},
        key::Symbol,
        ν::Float64,
        ::Type{T},
        domain::$(RadialDomainType),
    ) where {T}
        legacy_imex_depwarn(:get_eab2_alu_cache!, "Use solver-managed timestep state instead of directly managing the EAB2 LU cache.")
        return solver_get_eab2_alu_cache!(caches, key, ν, T, domain)
    end

    function eab2_update_krylov_cached!(
        u::$(SpectralFieldType){T},
        nl::$(SpectralFieldType){T},
        nl_prev::$(SpectralFieldType){T},
        alu_map::Dict{Int, Tuple{$(BandedOperator){T}, $(BandedFactorization){T}}},
        domain::$(RadialDomainType),
        diffusivity::Float64,
        config::$(SHTnsConfigType),
        dt::Float64;
        m::Int=20,
        tol::Float64=1e-8,
        mass_coeff::Float64=1.0,
    ) where {T}
        legacy_imex_depwarn(:eab2_update_krylov_cached!, "Use the high-level solver/timestep drivers instead of the legacy low-level EAB2 API.")
        return solver_eab2_update_krylov_cached!(u, nl, nl_prev, alu_map, domain, diffusivity, config, dt; m, tol, mass_coeff)
    end

    function eab2_update!(
        u::$(SpectralFieldType){T},
        nl::$(SpectralFieldType){T},
        nl_prev::$(SpectralFieldType){T},
        etd::$(LegacyDenseETDCache){T},
        config::$(SHTnsConfigType),
        dt::Float64,
    ) where {T}
        legacy_imex_depwarn(:eab2_update!, "Use the configured solver timestep drivers instead of the dense legacy EAB2 update.")
        return legacy_eab2_update_dense!(u, nl, nl_prev, etd, config, dt)
    end
end
