# ================================================================================
# Conducting Inner Core - Admittance Module
# ================================================================================
#
# This file builds the inner-core implicit diffusion operator and its
# inner-core-boundary (ICB) admittance for a conducting inner core, using a
# Schur-complement / admittance method.
#
# The inner core is a ball domain `[0, ri]`. The magnetic toroidal/poloidal
# scalars there evolve by pure diffusion. For each harmonic degree `l`, the
# implicit diffusion operator is
#
#     M_ic = (1/dt) I − θ · η · ∇²_l
#
# with the radial Laplacian
#
#     ∇²_l = ∂²/∂r² + (2/r) ∂/∂r − l(l+1)/r²
#
# Grid convention (ball domain): index 1 = r=0, index Nic = r=ri (the ICB).
#
# Boundary rows of M_ic:
#   - Inner (row 1, r=0):   identity row → regularity for l≥1 (scalar = 0 at r=0)
#   - Outer (row Nic, ICB): identity row → prescribed Dirichlet ICB value
#
# The ICB admittance α_l is the one-sided radial derivative at the ICB produced
# by a unit ICB value:
#
#     solve  M_ic x = e_{Nic}     (1 at the ICB row, 0 elsewhere)
#     α_l = (d1_ic top row) · x   (∂/∂r at r=ri, row Nic of the 1st-derivative matrix)
#
# For a diffusive interior the admittance is positive and increases with l.
#
# Magnetic has no l=0 mode, so l=0 is skipped.
#
# Included by: src/physics/magnetic/field.jl
# ================================================================================

"""
    InnerCoreAdmittance{T}

Precomputed conducting-inner-core implicit diffusion factorizations and ICB
admittances, per stored harmonic degree `l`.

# Fields
- `factor`: `M_ic` banded LU factorization per stored `l`.
- `alpha`: ICB admittance `α_l` per stored `l`.
- `d1_top`: dense one-sided `∂/∂r` row at `r=ri` (row `Nic` of the first-derivative
  matrix), length `Nic`.
- `lookup`: maps a harmonic degree `l` to its index in `factor`/`alpha`.
- `Nic`: number of inner-core radial points.
- `lin`: the `η·∇²_l` operator (`L`) per stored `l`, BEFORE forming
  `X = (1/dt)I − θL`. Full-band (no boundary rows cleared); used to assemble the
  CNAB2 history RHS `b_ic = (1/dt + (1−θ)η∇²_l) S_old`.
- `dt`: time step used to build `M_ic` (must match for consistent RHS assembly).
- `theta`: implicit weight `θ` used to build `M_ic`.
"""
struct InnerCoreAdmittance{T}
    factor::Vector{BandedLU{T}}    # M_ic LU per stored l
    alpha::Vector{T}               # ICB admittance per stored l
    d1_top::Vector{T}              # one-sided ∂/∂r row at r=ri (dense, length Nic)
    lookup::Dict{Int, Int}
    Nic::Int
    lin::Vector{BandedMatrix{T}}   # η∇²_l (L, before (1/dt)I − θL) per stored l
    dt::T                          # time step used to build M_ic
    theta::T                       # implicit weight θ used to build M_ic
end

"""
    inner_core_alpha(a::InnerCoreAdmittance, l::Int) -> T

ICB admittance `α_l` for harmonic degree `l`. Errors if `l` was not stored.
"""
inner_core_alpha(a::InnerCoreAdmittance, l::Int) = a.alpha[a.lookup[l]]

"""
    create_inner_core_admittance(T, l_values, ic_domain, diffusivity, dt; theta=0.5)
        -> InnerCoreAdmittance{T}

Build the conducting-inner-core implicit diffusion operator
`M_ic = (1/dt)I − θ·diffusivity·∇²_l` and its ICB admittance `α_l` for every
positive harmonic degree in `l_values`.

`ic_domain` is the inner-core ball domain (index 1 = r=0, index Nic = r=ri).
`l=0` is skipped (magnetic has no l=0 mode).
"""
function create_inner_core_admittance(::Type{T}, l_values, ic_domain,
        diffusivity::Float64, dt::Float64;
        theta::Float64 = 0.5, poloidal::Bool = false) where {T}
    uniq = sort(unique(l_values));
    filter!(>(0), uniq)
    Nic = ic_domain.N;
    bw = radial_bandwidth(ic_domain)
    # POLOIDAL magnetic potentials diffuse with D_pol = d²/dr² − l(l+1)/r²
    # (NO 2/r term), matching the outer-core `create_magnetic_poloidal_matrices`
    # operator the ICB admittance is paired with via the Robin row
    # `(∂/∂r − α_l)S = φ0`. TOROIDAL potentials keep the full scalar Laplacian
    # ∇²_l = d²/dr² + (2/r)d/dr − l(l+1)/r². The −l(l+1)/r² term is added per
    # degree in the loop below, so only the base radial operator differs here.
    lap = poloidal ? create_derivative_matrix(T, 2, ic_domain) :
                     create_radial_laplacian(T, ic_domain)
    d1 = create_derivative_matrix(T, 1, ic_domain)
    r_inv_sq = @views ic_domain.r[1:Nic, 2]
    d1_top = T[(1 <= bw+1+Nic-j <= 2bw+1) ? d1.data[bw + 1 + Nic - j, j] : zero(T)
               for j in 1:Nic]
    base = T.(diffusivity .* lap.data)
    facs = Vector{BandedLU{T}}();
    alphas = T[];
    lk = Dict{Int, Int}()
    lins = Vector{BandedMatrix{T}}()
    for (idx, l) in enumerate(uniq)
        data = copy(base);
        lf = Float64(l*(l+1))
        @inbounds for n in 1:Nic
            ;
            data[bw + 1, n] -= T(diffusivity*lf*r_inv_sq[n]);
        end
        # Keep a full-band copy of L = η∇²_l BEFORE forming X = (1/dt)I − θL.
        push!(lins, BandedMatrix{T}(copy(data), bw, Nic))
        data .*= -T(theta);
        data[bw + 1, :] .+= T(1/dt)
        @inbounds for j in 1:(1 + bw)
            ;
            data[bw + 1 + 1 - j, j] = zero(T);
        end   # row 1 cleared
        data[bw + 1, 1] = one(T)                                          # r=0 identity
        @inbounds for j in (Nic - bw):Nic
            ;
            data[bw + 1 + Nic - j, j] = zero(T);
        end  # row Nic cleared
        data[bw + 1, Nic] = one(T)                                        # ICB identity
        M = BandedMatrix{T}(data, bw, Nic);
        lu = factorize_banded(M)
        rhs = zeros(T, Nic);
        rhs[Nic] = one(T)
        x = similar(rhs);
        solve_banded!(x, lu, rhs)
        push!(facs, lu);
        push!(alphas, dot(d1_top, x));
        lk[l] = idx
    end
    return InnerCoreAdmittance{T}(facs, alphas, d1_top, lk, Nic,
        lins, T(dt), T(theta))
end

"""
    _ic_build_bic(a::InnerCoreAdmittance, l, S_old) -> Vector

Assemble the CNAB2 history RHS for the inner-core diffusion solve at degree `l`:

    b = (1/dt) S_old + (1−θ) · η∇²_l · S_old

using the SAME `η∇²_l` (`a.lin`), `dt`, and `θ` that built `M_ic`. Boundary rows
(`b[1]`, `b[Nic]`) are left as produced by the full-band Laplacian and are meant
to be overridden by the caller (regularity at `r=0`, prescribed ICB value).
"""
function _ic_build_bic(a::InnerCoreAdmittance{T}, l::Int, S_old::AbstractVector{T}) where {T}
    idx = a.lookup[l];
    L = a.lin[idx];
    Nic = a.Nic
    Lx = zeros(T, Nic);
    apply_∂r!(Lx, L, S_old)   # Lx = η∇²_l S_old
    b = (one(T)/a.dt) .* S_old .+ (one(T) - a.theta) .* Lx
    return b
end

"""
    inner_core_history_flux(a::InnerCoreAdmittance, l, S_old) -> T

ICB radial-derivative contribution `φ0` from the inner-core CNAB2 history with a
ZERO ICB value: solve `M_ic y = b_ic` with homogeneous boundary rows (`y(0)=0`,
`y(ri)=0`) and return `φ0 = d1_top · y`. Pairs with the admittance `α_l` to form
the outer-core inner Robin row `(∂/∂r − α_l) S = φ0`. For a zero history,
`φ0 = 0`.
"""
function inner_core_history_flux(a::InnerCoreAdmittance{T}, l::Int, S_old::AbstractVector{T}) where {T}
    b = _ic_build_bic(a, l, S_old)
    b[1] = zero(T);
    b[a.Nic] = zero(T)
    y = similar(b);
    solve_banded!(y, a.factor[a.lookup[l]], b)
    return dot(a.d1_top, y)
end

"""
    reconstruct_inner_core(a::InnerCoreAdmittance, l, g, S_old) -> Vector

Reconstruct the inner-core radial profile at degree `l` after the outer-core
solve: solve `M_ic S = b_ic` with regularity `S(0)=0` and ICB Dirichlet value
`S(ri)=g`. Returns the length-`Nic` profile. `g` is the outer-core value at the
ICB; `S_old` is the previous inner-core profile (its history enters `b_ic`).
"""
function reconstruct_inner_core(a::InnerCoreAdmittance{T}, l::Int, g::T, S_old::AbstractVector{T}) where {T}
    b = _ic_build_bic(a, l, S_old)
    b[1] = zero(T);
    b[a.Nic] = g
    S = similar(b);
    solve_banded!(S, a.factor[a.lookup[l]], b)
    return S
end
