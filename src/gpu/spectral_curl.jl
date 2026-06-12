# =============================================================================
# GPU Phase 5a — batched banded radial mat-vec + spectral curl (vorticity ω=∇×u,
# current J=∇×B; the SAME operator on velocity vs magnetic (T,P)).  The derivative
# matrices d1=∂/∂r, d2=∂²/∂r² are banded (2bw+1,nr), l-independent.  A KA kernel
# applies one to each mode's radial profile (mirrors apply_radial_derivative!,
# numerics.jl:1026-1045); the curl is then a second-derivative mat-vec plus
# element-wise radius factors.
# Runs on Array (locally testable) and CuArray.  Curl is real-linear → real/imag
# handled independently.
# =============================================================================

# One workitem per (l,m). Y[li,mi,i] = Σ_{j∈[max(1,i-bw),min(nr,i+bw)]} mat[bw+1+i-j,j]·X[li,mi,j].
# Same ascending-j accumulation as apply_radial_derivative! (so exact == on CPU). Y ≠ X.
@kernel function _banded_matvec_kernel!(Y, @Const(X), @Const(mat), bw::Int, nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(Y)
    @inbounds for i in 1:nr
        s = zero(T)
        for j in max(1, i - bw):min(nr, i + bw)
            s += mat[bw + 1 + i - j, j] * X[li, mi, j]
        end
        Y[li, mi, i] = s
    end
end

"""
    gpu_batched_banded_matvec!(Y, X, mat, bw) -> Y

Apply the banded radial operator `mat` (`(2bw+1,nr)`) to every mode's radial
profile: `Y[l,m,:] = mat · X[l,m,:]`.  `Y`/`X` are `(nl,nm,nr)`.  `Y` must NOT
alias `X` (an output point reads input points at other radii).  Backend inferred
from `Y`.
"""
function gpu_batched_banded_matvec!(Y, X, mat, bw::Int)
    nl, nm, nr = size(Y)
    backend = KernelAbstractions.get_backend(Y)
    _banded_matvec_kernel!(backend)(Y, X, mat, bw, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)  # eager sync; gpu_spectral_curl! calls this 2×/curl — Phase-5c: hoist to caller
    return Y
end

"""
    gpu_spectral_curl!(dst_tor_r, dst_tor_i, dst_pol_r, dst_pol_i,
                       src_tor_r, src_tor_i, src_pol_r, src_pol_i,
                       d1, d2, lfac, rinv, rinv2, r_vec, bw) -> nothing

Spectral curl of a toroidal–poloidal field (vorticity `∇×u` from velocity, or
current `∇×B` from magnetic — the same operator):
  `dst_tor = rinv·(d2·P − lfac·rinv2·P)`,  `dst_pol = −r·T`,
with `P`=`src_pol`, `T`=`src_tor`.  `lfac[l+1]=l(l+1)` (length `nl`); `rinv`/`rinv2`
length `nr`; `r_vec` is `r`; `d1`/`d2` are banded `(2bw+1,nr)`. `d1` is retained
in the signature for operator-bundle symmetry but is not used by the Stage-2
formula.  All arrays on the same backend.  Real/imag handled independently
(curl is real-linear).  The `dst_*` arrays must NOT alias any `src_*` array.
"""
function gpu_spectral_curl!(dst_tor_r, dst_tor_i, dst_pol_r, dst_pol_i,
        src_tor_r, src_tor_i, src_pol_r, src_pol_i, d1, d2, lfac, rinv, rinv2, r_vec, bw::Int)
    # 2 mat-vec launches, each with its own synchronize (= 2 barriers/curl).
    # Phase-5c: accept caller-owned scratch (d2P*) + hoist a single barrier
    # after both launches (the outputs are independent, no inter-dependency).
    d2Pr = similar(src_pol_r); d2Pi = similar(src_pol_i)
    gpu_batched_banded_matvec!(d2Pr, src_pol_r, d2, bw)
    gpu_batched_banded_matvec!(d2Pi, src_pol_i, d2, bw)
    lf  = reshape(lfac, :, 1, 1)
    ri  = reshape(rinv, 1, 1, :)
    ri2 = reshape(rinv2, 1, 1, :)
    rr  = reshape(r_vec, 1, 1, :)
    @. dst_tor_r = ri * (d2Pr - lf * ri2 * src_pol_r)
    @. dst_tor_i = ri * (d2Pi - lf * ri2 * src_pol_i)
    @. dst_pol_r = -rr * src_tor_r
    @. dst_pol_i = -rr * src_tor_i
    return nothing
end
