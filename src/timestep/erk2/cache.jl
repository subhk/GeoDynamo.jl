# ERK2 cache lifecycle: cache builders, memoized accessors, and bundle persistence.

# ── boundary-structure fingerprint for the memoized stage caches ────────────────
# `solver_erk2_constrained_propagators` folds the two endpoint constraint rows into the
# generator, so E/phi1/phi2 depend on the row STRUCTURE — the side kinds, their stencils
# and l-corrections, plus which generator (dpol) and which inner row (regularity) was
# used. `solver_erk2_constraint_row` never reads the endpoint VALUE/target, and the
# mode-value slots are live views, so values are deliberately excluded: including them
# would force a rebuild on every step without changing a single propagator entry.
#
# Computed from the RAW getter inputs (the `bc_spec` as passed, not the spec derived
# inside a builder) so the getter and the builder always agree.
_erk2_side_signature(side::SolverERK2DirichletSide) = (:dirichlet,)
function _erk2_side_signature(side::SolverERK2StencilSide)
    return (:stencil, side.stencil, side.r_inv, side.l_sign,
        side.use_l_correction, side.fixed_correction)
end

function _erk2_bc_signature(bc_spec, boundary_condition::Int,
        inner_regularity::Bool, dpol_operator::Bool)
    bc_spec === nothing && return hash((:derived, boundary_condition,
        inner_regularity, dpol_operator))
    return hash((_erk2_side_signature(bc_spec.inner), _erk2_side_signature(bc_spec.outer),
        boundary_condition, inner_regularity, dpol_operator))
end

"""
    solver_erk2_constraint_row(T, side, boundary_idx, l, nr) -> Vector{T}

Dense length-`nr` row of the linear constraint that `solver_enforce_erk2_bc!`
imposes at `boundary_idx` for degree `l`.

The generator rows and the endpoint projection are both derived from this one
function, so the two can never drift apart — previously the cache builders
re-stamped the rows by hand and only source-text matching tied them together.
"""
function solver_erk2_constraint_row(
        ::Type{T},
        side::SolverERK2DirichletSide{T},
        boundary_idx::Int,
        l::Int,
        nr::Int
) where {T}
    row = zeros(T, nr)
    row[boundary_idx] = one(T)
    return row
end

function solver_erk2_constraint_row(
        ::Type{T},
        side::SolverERK2StencilSide{T},
        boundary_idx::Int,
        l::Int,
        nr::Int
) where {T}
    row = zeros(T, nr)
    copyto!(row, side.stencil)
    self_correction = side.fixed_correction
    if side.use_l_correction
        self_correction += side.l_sign * T(l) * side.r_inv
    end
    row[boundary_idx] += self_correction
    return row
end

"""
    solver_erk2_constrained_propagators(operator, row_inner, row_outer, dt, l)

Build the ERK2 propagator matrices for a radial generator subject to the two
endpoint constraints `row_inner · x = v_in`, `row_outer · x = v_out`.

Stamping a constraint row into a generator and then exponentiating it does *not*
impose the constraint: `exp(dt·A)` treats that row as an evolution equation for
the endpoint, so the interior only ever sees the boundary condition through the
post-hoc endpoint projection, one step late. For Dirichlet rows the two happen
to coincide, but for the derivative (Robin) rows used by the magnetic poloidal
insulating condition the lag is a genuine error in the diffusion operator: it
grows with `dt` *and* with radial resolution, and shows up directly as a
free-decay rate tens of percent below the analytic shell eigenvalue.

The constraint is instead eliminated. With `I` the interior indices and
`b = (1, nr)` the endpoints, the constraints give `x_b = g0 + G·x_I`, so the
interior obeys the reduced ODE

    d/dt x_I = Ã·x_I + A_Ib·g0 + N_I,      Ã = A_II + A_Ib·G

whose exponential propagators are exact. `g0` is recovered from the incoming
endpoint values (`g0 = x_b − G·x_I`, exact whenever the incoming state satisfies
the constraint), which keeps the inhomogeneous case correct without any change
to the caller. The results are embedded back into `nr × nr` matrices so the
staged integrator is untouched, and the trailing endpoint projection becomes a
no-op rather than the sole enforcement.

`homogeneous = true` drops the `A_Ib·g0` forcing (valid when the boundary data
is identically zero); this keeps the propagator's entries at O(1) instead of the
O(1/h⁴) that the forcing otherwise injects, which is what keeps a marginal
(σ = 0) mode stable at fine grids — see the note at the `propagators` closure.

Returns `(E_half, E_full, phi1_half, phi1_full, phi2_full)`.
"""
function solver_erk2_constrained_propagators(
        operator::Matrix{T},
        row_inner::Vector{T},
        row_outer::Vector{T},
        dt::Float64,
        l::Int;
        homogeneous::Bool = false
) where {T}
    nr = size(operator, 1)
    nr >= 5 || throw(ArgumentError(
        "ERK2 boundary elimination needs nr >= 5 radial points, got nr=$nr"))
    interior = 2:(nr - 1)
    edges = [1, nr]
    n_int = length(interior)

    # x_b = g0 + G·x_I from the two constraint rows.
    C_bb = T[row_inner[1] row_inner[nr]
             row_outer[1] row_outer[nr]]
    C_bI = Matrix{T}(undef, 2, n_int)
    @inbounds for (jj, j) in enumerate(interior)
        C_bI[1, jj] = row_inner[j]
        C_bI[2, jj] = row_outer[j]
    end
    # Relative test: the endpoint coefficients scale like 1/h, so an absolute
    # threshold would wave a singular block through on a fine grid.
    C_bb_scale = max(maximum(abs, C_bb), one(T))
    abs(LA.det(C_bb)) > eps(T) * C_bb_scale^2 || throw(ArgumentError(
        "ERK2 boundary constraints are not solvable for the endpoint values " *
        "(singular 2x2 endpoint block at l=$l); the radial stencil probably " *
        "spans the whole domain"))
    G = -(C_bb \ C_bI)

    A_II = operator[interior, interior]
    A_Ib = operator[interior, edges]
    A_tilde = A_II + A_Ib * G

    # Interior -> full expansion: endpoints follow the (homogeneous) constraint.
    # Any inhomogeneous offset is reinstated by the endpoint projection.
    M = zeros(T, nr, n_int)
    @inbounds for (jj, j) in enumerate(interior)
        M[j, jj] = one(T)
    end
    @inbounds for jj in 1:n_int
        M[1, jj] = G[1, jj]
        M[nr, jj] = G[2, jj]
    end

    function embed(interior_block::Matrix{T}, edge_block::Union{Matrix{T}, Nothing})
        out = zeros(T, nr, nr)
        out[:, interior] = M * interior_block
        if edge_block !== nothing
            out[:, edges] = M * edge_block
        end
        return out
    end

    # Scale s: x_I(s) = e^{sÃ}x_I + s·φ1(sÃ)·(A_Ib·g0 + N_I), and
    # g0 = x_b − G·x_I is linear in the incoming full vector.
    #
    # `homogeneous`: when the boundary data is identically zero (g0 ≡ 0 — e.g.
    # insulating magnetic, stress-free velocity, zero-flux scalar), the forcing
    # term contributes nothing, so it is dropped. This is not just an
    # optimisation: `forcing = s·φ1·A_Ib` carries A_Ib ~ 1/h² entries, and the
    # combined `Es − forcing·G` has O(1/h⁴) entries that must cancel on-manifold
    # to recover `Es·x_I`. For strictly-decaying modes the lost digits decay
    # away, but a MARGINAL mode (σ = 0, e.g. the velocity stress-free l=1
    # rigid-rotation mode) accumulates them and the projected step goes unstable
    # at fine grids (the spectral radius tracks 1/h⁴). Dropping the forcing when
    # it is provably zero keeps the interior propagator `M·Es` at O(1) entries,
    # so the marginal mode stays neutral (radius = 1) at every resolution. The
    # inhomogeneous path keeps the forcing (its fields have no marginal mode).
    function propagators(s::Float64)
        sT = T(s)
        sA = sT .* A_tilde
        Es = exp(sA)
        phi1s = solver_compute_phi1_function(sA, Es)
        if homogeneous
            return (embed(Es, nothing), embed(phi1s, nothing), sA, Es)
        end
        forcing = (sT .* phi1s) * A_Ib         # multiplies g0
        return (embed(Es - forcing * G, forcing), embed(phi1s, nothing), sA, Es)
    end

    E_half, phi1_half, _, _ = propagators(dt / 2)
    E_full, phi1_full, sA_full, Es_full = propagators(dt)
    phi2_full = embed(
        solver_compute_phi2_function(sA_full, Es_full; l = l), nothing)

    return (E_half, E_full, phi1_half, phi1_full, phi2_full)
end

# A boundary side carries no inhomogeneous datum when its scalar endpoint value
# is zero and it holds no per-mode value vector (or an all-zero one).
_erk2_side_homogeneous(v::Nothing) = true
_erk2_side_homogeneous(v::AbstractVector) = all(iszero, v)

"""
    solver_erk2_spec_homogeneous(spec) -> Bool

True when the ERK2 boundary spec imposes only zero endpoint data on both walls
(real and imaginary, scalar value and every per-mode value vector). Such a spec
has `g0 ≡ 0`, so `solver_erk2_constrained_propagators` may drop the forcing term
— see the note there on why that matters for marginal modes.
"""
function solver_erk2_spec_homogeneous(spec::SolverERK2BoundarySpec{T}) where {T}
    return iszero(erk2_endpoint_target(spec.inner)) &&
           iszero(erk2_endpoint_target(spec.outer)) &&
           _erk2_side_homogeneous(spec.inner_mode_values) &&
           _erk2_side_homogeneous(spec.outer_mode_values) &&
           _erk2_side_homogeneous(spec.inner_mode_values_imag) &&
           _erk2_side_homogeneous(spec.outer_mode_values_imag)
end

"""
    create_solver_erk2_scalar_cache(T, config, domain, diffusivity, dt, boundary_condition;
                                     bc_spec, inner_regularity, ...)

Precompute ERK2 propagators for scalar fields with the endpoint constraints
eliminated from the generator.

The cache stores one set of matrices per unique spherical-harmonic degree in
`config`. Dense matrices are precomputed unless `use_krylov=true`, in which
case operator matrices are stored for Krylov actions.

`bc_spec` supplies the endpoint descriptors; it defaults to the pair implied by
`boundary_condition` (DD/DN/ND/NN for codes 1–4) via `build_solver_erk2_scalar_bc`,
honouring `inner_regularity` for the ball center row. Neumann and regularity ends
are derivative rows — stamping them into the generator and exponentiating would
only lag-enforce them (the interior would see a spurious frozen-endpoint / Dirichlet
condition each sub-step, with the true derivative row applied one step late by the
endpoint projection), so they are eliminated instead — see
`solver_erk2_constrained_propagators`. Pass the SAME spec the staged integrator
enforces so the generator and the projection cannot drift.
"""
function create_solver_erk2_scalar_cache(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64,
        boundary_condition::Int;
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        inner_regularity::Bool = false,
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    laplacian = build_radial_laplacian(domain)
    nr = domain.N
    bandwidth = laplacian.bandwidth
    r_inv_sq = @views domain.r[1:nr, 2]
    l_values = unique(config.l_values)
    spec = bc_spec === nothing ?
           build_solver_erk2_scalar_bc(T, domain, boundary_condition; inner_regularity) :
           bc_spec

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    bc_desc = ["DD", "DN", "ND", "NN"][clamp(boundary_condition, 1, 4)]
    if mpi_rank() == 0
        @info "Creating solver ERK2 scalar cache (type=$bc_desc, ν=$diffusivity, eliminated BC rows)"
    end

    for l in l_values
        operator_data = diffusivity .* laplacian.data
        operator_dense = solver_banded_to_dense(BandedOperator(operator_data, bandwidth, nr))
        l_factor = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            operator_dense[n, n] -= diffusivity * l_factor * r_inv_sq[n]
        end

        if use_krylov
            operator_dense[1, :] .= zero(T)
            operator_dense[nr, :] .= zero(T)
            push!(E_half, operator_dense)
            push!(E_full, operator_dense)
            push!(phi1_half, operator_dense)
            push!(phi1_full, operator_dense)
            push!(phi2_full, operator_dense)
        else
            row_inner = solver_erk2_constraint_row(T, spec.inner, 1, l, nr)
            row_outer = solver_erk2_constraint_row(T, spec.outer, nr, l, nr)
            E_h, E_f, p1_h, p1_f, p2_f = solver_erk2_constrained_propagators(
                operator_dense, row_inner, row_outer, dt, l)
            push!(E_half, E_h)
            push!(E_full, E_f)
            push!(phi1_half, p1_h)
            push!(phi1_full, p1_f)
            push!(phi2_full, p2_f)
        end
    end

    mpi_barrier!()

    return ERK2StageCache{T}(
        dt,
        diffusivity,
        nr,
        l_values,
        E_half,
        E_full,
        phi1_half,
        phi1_full,
        phi2_full,
        use_krylov,
        m,
        tol,
        true,
        _erk2_bc_signature(bc_spec, boundary_condition, inner_regularity, false)
    )
end

"""
    create_solver_erk2_cache(T, config, domain, diffusivity, dt; bc_spec=nothing, ...)

Precompute generic ERK2 propagators for velocity-like spectral fields.

Two cases share this builder:

- **Toroidal** (`bc_spec !== nothing`, `dpol_operator = false`): the endpoint
  constraints are eliminated from the generator via
  `solver_erk2_constrained_propagators`, exactly like the magnetic/scalar caches,
  so the wall is embedded in the propagated operator and there is no stability
  ceiling. Homogeneous walls (stress-free, plain no-slip) drop the forcing term
  (`solver_erk2_spec_homogeneous`), which keeps the marginal velocity stress-free
  `l = 1` rigid-rotation mode (σ = 0: `Δ₁r = 0`, `(∂ᵣ − 1/r)r = 0` at both walls)
  neutral at every resolution; inhomogeneous walls (rotating inner core) keep the
  forcing (a Dirichlet row, no marginal mode).

- **Poloidal** (`dpol_operator = true`, `bc_spec === nothing`): the natural
  (un-constrained) boundary rows are exponentiated directly and the P = 0 /
  P′ = 0 (or stress-free) walls are imposed afterwards by the influence-matrix
  (Green's-function) W-split recovery (`src/timestep/erk2/influence.jl`,
  `_erk2_poloidal_recover!`). Elimination does not apply — the recovery needs the
  natural-row exponential.

Verified in `test/velocity_erk2_stability.jl`.
"""
function create_solver_erk2_cache(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8,
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        dpol_operator::Bool = false
) where {T}
    # dpol_operator: build on D_pol = d²/dr² − l(l+1)/r² (poloidal potentials
    # under the Stage-2 solenoidal convention) instead of the full scalar
    # Laplacian (Stage-4B ERK2 W-split port).
    laplacian = dpol_operator ? create_derivative_matrix(Float64, 2, domain) :
                build_radial_laplacian(domain)
    nr = domain.N
    r_inv_sq = @views domain.r[1:nr, 2]
    l_values = unique(config.l_values)

    # Toroidal path: eliminate the endpoint constraints (see the docstring). The
    # poloidal W-split (dpol_operator) keeps the natural-row exp for its influence
    # recovery, and the Krylov path stores the raw operator, so both opt out.
    eliminate_bc = (!use_krylov) && (bc_spec !== nothing) && (!dpol_operator)
    bc_homogeneous = eliminate_bc && solver_erk2_spec_homogeneous(bc_spec)

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    if mpi_rank() == 0
        method_name = use_krylov ? "Krylov" :
                      (eliminate_bc ? "dense, eliminated BC rows" : "dense")
        @info "Creating solver ERK2 cache for $(length(l_values)) l-modes with $method_name methods"
    end

    for l in l_values
        operator_data = diffusivity .* laplacian.data
        operator_dense = solver_banded_to_dense(
            BandedOperator(operator_data, laplacian.bandwidth, nr),
        )
        l_factor = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            operator_dense[n, n] -= diffusivity * l_factor * r_inv_sq[n]
        end

        if eliminate_bc
            row_inner = solver_erk2_constraint_row(T, bc_spec.inner, 1, l, nr)
            row_outer = solver_erk2_constraint_row(T, bc_spec.outer, nr, l, nr)
            E_h, E_f, p1_h, p1_f, p2_f = solver_erk2_constrained_propagators(
                operator_dense, row_inner, row_outer, dt, l;
                homogeneous = bc_homogeneous)
            push!(E_half, E_h)
            push!(E_full, E_f)
            push!(phi1_half, p1_h)
            push!(phi1_full, p1_f)
            push!(phi2_full, p2_f)
            continue
        end

        if l == 0
            operator_dense[1, :] .= zero(T)
            operator_dense[nr, :] .= zero(T)
        end

        if use_krylov
            push!(E_half, operator_dense)
            push!(E_full, operator_dense)
            push!(phi1_half, operator_dense)
            push!(phi1_full, operator_dense)
            push!(phi2_full, operator_dense)
        else
            operator_half = (dt / 2) .* operator_dense
            operator_full = dt .* operator_dense

            E_half_l = exp(operator_half)
            E_full_l = exp(operator_full)
            if !all(isfinite, E_half_l) || !all(isfinite, E_full_l)
                @error "Non-finite solver ERK2 matrix exponential for l=$l (dt=$dt, ||A||=$(opnorm(operator_dense)))"
            end
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))

            phi1_half_l = solver_compute_phi1_function(operator_half, E_half_l)
            phi1_full_l = solver_compute_phi1_function(operator_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            phi2_full_l = solver_compute_phi2_function(operator_full, E_full_l; l = l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
        end
    end

    mpi_barrier!()

    return ERK2StageCache{T}(
        dt,
        diffusivity,
        nr,
        l_values,
        E_half,
        E_full,
        phi1_half,
        phi1_full,
        phi2_full,
        use_krylov,
        m,
        tol,
        true,
        _erk2_bc_signature(bc_spec, 0, false, dpol_operator)
    )
end

"""
    create_solver_erk2_magnetic_toroidal_cache(T, config, domain, diffusivity, dt; bc_spec, ...)

Precompute ERK2 propagators for magnetic toroidal fields with the endpoint
constraints eliminated from the generator.

`bc_spec` supplies the endpoint descriptors; it defaults to the shell insulating
pair (homogeneous Dirichlet on both walls). Pass the ball spec built with
`inner_regularity = true` to get the center-regularity inner row. The spec must
be the same one the staged integrator enforces after each step — see
`solver_erk2_constrained_propagators`.
"""
function create_solver_erk2_magnetic_toroidal_cache(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64;
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    laplacian = build_radial_laplacian(domain)
    nr = domain.N
    bandwidth = laplacian.bandwidth
    r_inv_sq = @views domain.r[1:nr, 2]
    l_values = unique(config.l_values)
    spec = bc_spec === nothing ?
           build_solver_erk2_magnetic_tor_bc(T, domain) : bc_spec

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    if mpi_rank() == 0
        @info "Creating solver ERK2 cache for magnetic toroidal with eliminated BC rows"
    end

    for l in l_values
        operator_data = diffusivity .* laplacian.data
        operator_dense = solver_banded_to_dense(BandedOperator(operator_data, bandwidth, nr))
        l_factor = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            operator_dense[n, n] -= diffusivity * l_factor * r_inv_sq[n]
        end

        if use_krylov
            operator_dense[1, :] .= zero(T)
            operator_dense[nr, :] .= zero(T)
            push!(E_half, operator_dense)
            push!(E_full, operator_dense)
            push!(phi1_half, operator_dense)
            push!(phi1_full, operator_dense)
            push!(phi2_full, operator_dense)
        else
            row_inner = solver_erk2_constraint_row(T, spec.inner, 1, l, nr)
            row_outer = solver_erk2_constraint_row(T, spec.outer, nr, l, nr)
            E_h, E_f, p1_h, p1_f, p2_f = solver_erk2_constrained_propagators(
                operator_dense, row_inner, row_outer, dt, l)
            push!(E_half, E_h)
            push!(E_full, E_f)
            push!(phi1_half, p1_h)
            push!(phi1_full, p1_f)
            push!(phi2_full, p2_f)
        end
    end

    mpi_barrier!()

    return ERK2StageCache{T}(
        dt,
        diffusivity,
        nr,
        l_values,
        E_half,
        E_full,
        phi1_half,
        phi1_full,
        phi2_full,
        use_krylov,
        m,
        tol,
        true,
        _erk2_bc_signature(bc_spec, 0, false, false)
    )
end

"""
    create_solver_erk2_magnetic_poloidal_cache(T, config, domain, diffusivity, dt; bc_spec, ...)

Precompute ERK2 propagators for magnetic poloidal fields with the insulating
endpoint constraints eliminated from the generator.

`bc_spec` supplies the endpoint descriptors; it defaults to the insulating pair
(∂r − (l+1)/r)P = 0 inner and (∂r + l/r)P = 0 outer. Both are derivative rows,
so eliminating them is what makes the propagator actually satisfy them — see
`solver_erk2_constrained_propagators`. The ball center-regularity inner row is
algebraically identical to the insulating inner row, so `inner_regularity` does
not change this operator.
"""
function create_solver_erk2_magnetic_poloidal_cache(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64;
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    # Stage-4B solenoidal convention: magnetic POLOIDAL potentials diffuse with
    # D_pol = d²/dr² − l(l+1)/r² (no 2/r term) — same operator the CNAB2
    # magnetic-poloidal matrices use since the Stage-4A consistency fix.
    laplacian = create_derivative_matrix(Float64, 2, domain)
    nr = domain.N
    bandwidth = laplacian.bandwidth
    r_inv_sq = @views domain.r[1:nr, 2]
    l_values = unique(config.l_values)
    spec = bc_spec === nothing ?
           build_solver_erk2_magnetic_pol_bc(T, domain) : bc_spec

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    if mpi_rank() == 0
        @info "Creating solver ERK2 cache for magnetic poloidal (D_pol) with eliminated insulating BCs"
    end

    for l in l_values
        operator_data = diffusivity .* laplacian.data
        operator_dense = solver_banded_to_dense(BandedOperator(operator_data, bandwidth, nr))
        l_factor = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            operator_dense[n, n] -= diffusivity * l_factor * r_inv_sq[n]
        end

        if use_krylov
            operator_dense[1, :] .= zero(T)
            operator_dense[nr, :] .= zero(T)
            push!(E_half, operator_dense)
            push!(E_full, operator_dense)
            push!(phi1_half, operator_dense)
            push!(phi1_full, operator_dense)
            push!(phi2_full, operator_dense)
        else
            row_inner = solver_erk2_constraint_row(T, spec.inner, 1, l, nr)
            row_outer = solver_erk2_constraint_row(T, spec.outer, nr, l, nr)
            E_h, E_f, p1_h, p1_f, p2_f = solver_erk2_constrained_propagators(
                operator_dense, row_inner, row_outer, dt, l)
            push!(E_half, E_h)
            push!(E_full, E_f)
            push!(phi1_half, p1_h)
            push!(phi1_full, p1_f)
            push!(phi2_full, p2_f)
        end
    end

    mpi_barrier!()

    return ERK2StageCache{T}(
        dt,
        diffusivity,
        nr,
        l_values,
        E_half,
        E_full,
        phi1_half,
        phi1_full,
        phi2_full,
        use_krylov,
        m,
        tol,
        true,
        _erk2_bc_signature(bc_spec, 0, false, false)
    )
end

"""
    GeoDynamo.create_erk2_cache(T, config, domain, diffusivity, dt; ...)

Public wrapper for constructing a generic ERK2 stage cache.
"""
function GeoDynamo.create_erk2_cache(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8,
        bc_spec::Union{Nothing, GeoDynamo.ERK2BoundarySpec{T}} = nothing
) where {T}
    return create_solver_erk2_cache(
        T,
        config,
        domain,
        diffusivity,
        dt;
        use_krylov,
        m,
        tol,
        bc_spec
    )
end

"""
    GeoDynamo.create_erk2_cache_scalar(T, config, domain, diffusivity, dt, boundary_condition; ...)

Public wrapper for constructing scalar-field ERK2 caches with embedded
boundary conditions.
"""
function GeoDynamo.create_erk2_cache_scalar(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64,
        boundary_condition::Int;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    return create_solver_erk2_scalar_cache(
        T,
        config,
        domain,
        diffusivity,
        dt,
        boundary_condition;
        use_krylov,
        m,
        tol
    )
end

"""
    GeoDynamo.create_erk2_cache_temperature(T, config, domain, diffusivity, dt, temperature_bcs; ...)

Create the ERK2 cache used by temperature fields.
"""
function GeoDynamo.create_erk2_cache_temperature(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64,
        temperature_bcs::BoundaryConditions;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    GeoDynamo.create_erk2_cache_scalar(
        T,
        config,
        domain,
        diffusivity,
        dt,
        _thermal_bc_code(temperature_bcs);
        use_krylov,
        m,
        tol
    )
end

"""
    GeoDynamo.create_erk2_cache_composition(T, config, domain, diffusivity, dt, composition_bcs; ...)

Create the ERK2 cache used by composition fields.
"""
function GeoDynamo.create_erk2_cache_composition(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64,
        composition_bcs::BoundaryConditions;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    GeoDynamo.create_erk2_cache_scalar(
        T,
        config,
        domain,
        diffusivity,
        dt,
        _composition_bc_code(composition_bcs);
        use_krylov,
        m,
        tol
    )
end

"""
    GeoDynamo.create_erk2_cache_magnetic_toroidal(T, config, domain, diffusivity, dt; ...)

Create the ERK2 cache used by magnetic toroidal fields.
"""
function GeoDynamo.create_erk2_cache_magnetic_toroidal(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    return create_solver_erk2_magnetic_toroidal_cache(
        T,
        config,
        domain,
        diffusivity,
        dt;
        use_krylov,
        m,
        tol
    )
end

"""
    GeoDynamo.create_erk2_cache_magnetic_poloidal(T, config, domain, diffusivity, dt; ...)

Create the ERK2 cache used by magnetic poloidal fields.
"""
function GeoDynamo.create_erk2_cache_magnetic_poloidal(
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        diffusivity::Float64,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    return create_solver_erk2_magnetic_poloidal_cache(
        T,
        config,
        domain,
        diffusivity,
        dt;
        use_krylov,
        m,
        tol
    )
end

"""
    _get_or_build_erk2_cache(existing, label, diffusivity, T, config, domain, dt; ...)

Build or reuse an ERK2 stage cache for velocity-like fields.

Callers own the storage location; this helper only decides whether the existing
cache still matches the current grid, timestep, diffusivity, and method flags.
"""
function _get_or_build_erk2_cache(
        existing::Union{ERK2StageCache{T}, Nothing},
        label::AbstractString,
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8,
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        dpol_operator::Bool = false
)::ERK2StageCache{T} where {T}
    nr = domain.N
    bc_signature = _erk2_bc_signature(bc_spec, 0, false, dpol_operator)
    needs_rebuild = existing === nothing ||
                    existing.diffusivity != diffusivity ||
                    existing.nr != nr ||
                    existing.dt != dt ||
                    existing.use_krylov != use_krylov ||
                    !existing.mpi_consistent ||
                    existing.l_values != unique(config.l_values) ||
                    existing.bc_signature != bc_signature

    if needs_rebuild
        if mpi_rank() == 0
            @info "Creating solver $label ERK2 cache (ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        return create_solver_erk2_cache(
            T,
            config,
            domain,
            diffusivity,
            dt;
            use_krylov,
            m,
            tol,
            bc_spec,
            dpol_operator
        )
    end

    return existing::ERK2StageCache{T}
end

"""
    _get_or_build_erk2_scalar_cache(existing, label, diffusivity, T, config, domain, dt, boundary_condition; ...)

Build or reuse an ERK2 stage cache for scalar fields.
"""
function _get_or_build_erk2_scalar_cache(
        existing::Union{ERK2StageCache{T}, Nothing},
        label::AbstractString,
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64,
        boundary_condition::Int;
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        inner_regularity::Bool = false,
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
)::ERK2StageCache{T} where {T}
    nr = domain.N
    bc_signature = _erk2_bc_signature(bc_spec, boundary_condition, inner_regularity, false)
    needs_rebuild = existing === nothing ||
                    existing.diffusivity != diffusivity ||
                    existing.nr != nr ||
                    existing.dt != dt ||
                    existing.use_krylov != use_krylov ||
                    !existing.mpi_consistent ||
                    existing.l_values != unique(config.l_values) ||
                    existing.bc_signature != bc_signature

    if needs_rebuild
        bc_desc = ["DD", "DN", "ND", "NN"][clamp(boundary_condition, 1, 4)]
        if mpi_rank() == 0
            @info "Creating solver $label ERK2 cache (type=$bc_desc, ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        return create_solver_erk2_scalar_cache(
            T,
            config,
            domain,
            diffusivity,
            dt,
            boundary_condition;
            bc_spec,
            inner_regularity,
            use_krylov,
            m,
            tol
        )
    end

    return existing::ERK2StageCache{T}
end

"""
    get_solver_erk2_temperature_cache!(caches, diffusivity, T, config, domain, dt, temperature_bc_code; ...)

Return the solver-owned temperature ERK2 cache, rebuilding it when timestep,
grid, diffusivity, Krylov settings, or boundary conditions no longer match.
"""
function get_solver_erk2_temperature_cache!(
        caches::TimestepCaches{T},
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64,
        temperature_bc_code::Int;
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        inner_regularity::Bool = false,
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    caches.erk2_temperature = _get_or_build_erk2_scalar_cache(
        caches.erk2_temperature,
        "temperature",
        diffusivity,
        T,
        config,
        domain,
        dt,
        temperature_bc_code;
        bc_spec = bc_spec,
        inner_regularity = inner_regularity,
        use_krylov = use_krylov,
        m = m,
        tol = tol
    )
    return caches.erk2_temperature::ERK2StageCache{T}
end

"""
    get_solver_erk2_composition_cache!(caches, diffusivity, T, config, domain, dt, composition_bc_code; ...)

Return the solver-owned composition ERK2 cache with the same compatibility
checks used for the temperature scalar cache.
"""
function get_solver_erk2_composition_cache!(
        caches::TimestepCaches{T},
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64,
        composition_bc_code::Int;
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        inner_regularity::Bool = false,
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    caches.erk2_composition = _get_or_build_erk2_scalar_cache(
        caches.erk2_composition,
        "composition",
        diffusivity,
        T,
        config,
        domain,
        dt,
        composition_bc_code;
        bc_spec = bc_spec,
        inner_regularity = inner_regularity,
        use_krylov = use_krylov,
        m = m,
        tol = tol
    )
    return caches.erk2_composition::ERK2StageCache{T}
end

"""
    get_solver_erk2_cache!(caches, Val(:velocity_toroidal), diffusivity, T, config, domain, dt; ...)

Return the velocity-toroidal ERK2 cache from `TimestepCaches`.

This concrete overload avoids runtime `Symbol` dispatch in the main solver
step while still sharing the generic rebuild checks.
"""
function get_solver_erk2_cache!(
        caches::TimestepCaches{T},
        ::Val{:velocity_toroidal},
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8,
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing
) where {T}
    caches.erk2_velocity_toroidal = _get_or_build_erk2_cache(
        caches.erk2_velocity_toroidal,
        "velocity_toroidal",
        diffusivity,
        T,
        config,
        domain,
        dt;
        use_krylov = use_krylov,
        m = m,
        tol = tol,
        bc_spec = bc_spec
    )
    return caches.erk2_velocity_toroidal::ERK2StageCache{T}
end

"""
    get_solver_erk2_cache!(caches, Val(:velocity_poloidal), diffusivity, T, config, domain, dt; ...)

Return the velocity-poloidal ERK2 cache from `TimestepCaches`.
"""
function get_solver_erk2_cache!(
        caches::TimestepCaches{T},
        ::Val{:velocity_poloidal},
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8,
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        dpol_operator::Bool = false
) where {T}
    caches.erk2_velocity_poloidal = _get_or_build_erk2_cache(
        caches.erk2_velocity_poloidal,
        "velocity_poloidal",
        diffusivity,
        T,
        config,
        domain,
        dt;
        use_krylov = use_krylov,
        m = m,
        tol = tol,
        bc_spec = bc_spec,
        dpol_operator = dpol_operator
    )
    return caches.erk2_velocity_poloidal::ERK2StageCache{T}
end

"""
    get_solver_erk2_cache!(caches, key, diffusivity, T, config, domain, dt; ...)

Compatibility shim that dispatches legacy `Symbol` keys to the concrete
velocity cache overloads.
"""
function get_solver_erk2_cache!(
        caches::TimestepCaches{T},
        key::Symbol,
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64;
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8,
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        dpol_operator::Bool = false
) where {T}
    if key === :velocity_toroidal
        return get_solver_erk2_cache!(
            caches, Val(:velocity_toroidal), diffusivity, T, config, domain, dt;
            use_krylov = use_krylov, m = m, tol = tol, bc_spec = bc_spec
        )
    elseif key === :velocity_poloidal
        return get_solver_erk2_cache!(
            caches, Val(:velocity_poloidal), diffusivity, T, config, domain, dt;
            use_krylov = use_krylov, m = m, tol = tol, bc_spec = bc_spec,
            dpol_operator = dpol_operator
        )
    else
        error("get_solver_erk2_cache!: unsupported key $key for TimestepCaches")
    end
end

"""
    get_solver_erk2_magnetic_toroidal_cache!(caches, diffusivity, T, config, domain, dt; ...)

Return or rebuild the magnetic-toroidal ERK2 cache stored in `TimestepCaches`.
"""
function get_solver_erk2_magnetic_toroidal_cache!(
        caches::TimestepCaches{T},
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64;
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    nr = domain.N
    existing = caches.erk2_magnetic_toroidal
    needs_rebuild = existing === nothing ||
                    existing.diffusivity != diffusivity ||
                    existing.nr != nr ||
                    existing.dt != dt ||
                    existing.use_krylov != use_krylov ||
                    !existing.mpi_consistent ||
                    existing.l_values != unique(config.l_values)

    if needs_rebuild
        if mpi_rank() == 0
            @info "Creating solver magnetic toroidal ERK2 cache (ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        caches.erk2_magnetic_toroidal = create_solver_erk2_magnetic_toroidal_cache(
            T,
            config,
            domain,
            diffusivity,
            dt;
            bc_spec,
            use_krylov,
            m,
            tol
        )
    end

    return caches.erk2_magnetic_toroidal::ERK2StageCache{T}
end

"""
    get_solver_erk2_magnetic_poloidal_cache!(caches, diffusivity, T, config, domain, dt; ...)

Return or rebuild the magnetic-poloidal ERK2 cache stored in `TimestepCaches`.
"""
function get_solver_erk2_magnetic_poloidal_cache!(
        caches::TimestepCaches{T},
        diffusivity::Float64,
        ::Type{T},
        config::SHTnsConfigType,
        domain::RadialDomainType,
        dt::Float64;
        bc_spec::Union{SolverERK2BoundarySpec{T}, Nothing} = nothing,
        use_krylov::Bool = false,
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    nr = domain.N
    existing = caches.erk2_magnetic_poloidal
    needs_rebuild = existing === nothing ||
                    existing.diffusivity != diffusivity ||
                    existing.nr != nr ||
                    existing.dt != dt ||
                    existing.use_krylov != use_krylov ||
                    !existing.mpi_consistent ||
                    existing.l_values != unique(config.l_values)

    if needs_rebuild
        if mpi_rank() == 0
            @info "Creating solver magnetic poloidal ERK2 cache (ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        caches.erk2_magnetic_poloidal = create_solver_erk2_magnetic_poloidal_cache(
            T,
            config,
            domain,
            diffusivity,
            dt;
            bc_spec,
            use_krylov,
            m,
            tol
        )
    end

    return caches.erk2_magnetic_poloidal::ERK2StageCache{T}
end

"""
    GeoDynamo.save_erk2_cache_bundle(path, caches; metadata=Dict())

Persist compatible ERK2 stage caches and metadata to a JLD2 file.
"""
function GeoDynamo.save_erk2_cache_bundle(
        path::AbstractString,
        caches::AbstractDict{Symbol, <:Any};
        metadata::Dict{String, Any} = Dict{String, Any}()
)
    bundle = Dict{Symbol, Any}()
    for (key, value) in caches
        cache = compat_normalize_old_erk2_cache_entry(value)
        cache === nothing && continue
        bundle[key] = cache
    end

    meta = Dict{String, Any}(metadata)
    meta["created_at"] = get(meta, "created_at", string(GeoDynamo.now()))
    GeoDynamo.jldopen(path, "w") do file
        file["caches"] = bundle
        file["metadata"] = meta
    end
    return path
end

"""
    GeoDynamo.load_erk2_cache_bundle(path)

Load ERK2 cache bundle data and metadata from a JLD2 file.
"""
function GeoDynamo.load_erk2_cache_bundle(path::AbstractString)
    caches = Dict{Symbol, Any}()
    metadata = Dict{String, Any}()
    GeoDynamo.jldopen(path, "r") do file
        caches = Dict{Symbol, Any}(file["caches"])
        metadata = haskey(file, "metadata") ? Dict{String, Any}(file["metadata"]) :
                   Dict{String, Any}()
    end
    return caches, metadata
end

"""
    GeoDynamo.install_erk2_cache_bundle!(target, bundle)

Install cache entries from a loaded bundle into a target cache dictionary.
"""
function GeoDynamo.install_erk2_cache_bundle!(
        target::Dict{Symbol, Any},
        bundle::AbstractDict{Symbol, <:Any}
)
    for (key, value) in bundle
        cache = compat_normalize_old_erk2_cache_entry(value)
        cache === nothing && continue
        target[key] = cache
    end
    return target
end

"""
    GeoDynamo.install_erk2_cache_bundle!(target::Dict{Symbol, ERK2StageCache{T}}, bundle)

Typed cache-bundle installer used by solver-local cache dictionaries.
"""
function GeoDynamo.install_erk2_cache_bundle!(
        target::Dict{Symbol, ERK2StageCache{T}},
        bundle::AbstractDict{Symbol, <:Any}
) where {T}
    for (key, value) in bundle
        cache = compat_normalize_old_erk2_cache_entry(value)
        cache === nothing && continue
        target[key] = compat_solver_erk2_cache(cache)
    end
    return target
end

"""
    GeoDynamo.load_erk2_cache_bundle!(target, path)

Load a cache bundle from disk, install it into `target`, and return metadata.
"""
function GeoDynamo.load_erk2_cache_bundle!(
        target::Union{
            Dict{Symbol, Any},
            Dict{Symbol, ERK2StageCache{T}}
        },
        path::AbstractString
) where {T}
    bundle, metadata = GeoDynamo.load_erk2_cache_bundle(path)
    GeoDynamo.install_erk2_cache_bundle!(target, bundle)
    return metadata
end
