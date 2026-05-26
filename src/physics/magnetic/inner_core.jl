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
"""
struct InnerCoreAdmittance{T}
    factor::Vector{BandedLU{T}}    # M_ic LU per stored l
    alpha::Vector{T}               # ICB admittance per stored l
    d1_top::Vector{T}              # one-sided ∂/∂r row at r=ri (dense, length Nic)
    lookup::Dict{Int,Int}
    Nic::Int
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
                                      theta::Float64=0.5) where T
    uniq = sort(unique(l_values)); filter!(>(0), uniq)
    Nic = ic_domain.N; bw = radial_bandwidth(ic_domain)
    lap = create_radial_laplacian(T, ic_domain)
    d1  = create_derivative_matrix(T, 1, ic_domain)
    r_inv_sq = @views ic_domain.r[1:Nic, 2]
    d1_top = T[ (1 <= bw+1+Nic-j <= 2bw+1) ? d1.data[bw+1+Nic-j, j] : zero(T) for j in 1:Nic ]
    base = T.(diffusivity .* lap.data)
    facs = Vector{BandedLU{T}}(); alphas = T[]; lk = Dict{Int,Int}()
    for (idx,l) in enumerate(uniq)
        data = copy(base); lf = Float64(l*(l+1))
        @inbounds for n in 1:Nic; data[bw+1, n] -= T(diffusivity*lf*r_inv_sq[n]); end
        data .*= -T(theta); data[bw+1, :] .+= T(1/dt)
        @inbounds for j in 1:(1+bw); data[bw+1+1-j, j] = zero(T); end   # row 1 cleared
        data[bw+1, 1] = one(T)                                          # r=0 identity
        @inbounds for j in (Nic-bw):Nic; data[bw+1+Nic-j, j] = zero(T); end  # row Nic cleared
        data[bw+1, Nic] = one(T)                                        # ICB identity
        M = BandedMatrix{T}(data, bw, Nic); lu = factorize_banded(M)
        rhs = zeros(T, Nic); rhs[Nic] = one(T)
        x = similar(rhs); solve_banded!(x, lu, rhs)
        push!(facs, lu); push!(alphas, dot(d1_top, x)); lk[l] = idx
    end
    return InnerCoreAdmittance{T}(facs, alphas, d1_top, lk, Nic)
end
