# =============================================================================
# GPU Phase 5a — batched banded radial mat-vec + spectral curl (vorticity ω=∇×u,
# current J=∇×B; the SAME operator on velocity vs magnetic (T,P)).  Under the
# Stage-2 solenoidal convention, the curl potentials are
# T_curl = (P'' - l(l+1)P/r²)/r, P_curl = -rT.
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
    KernelAbstractions.synchronize(backend)  # eager sync; gpu_spectral_curl! calls this 4×/curl — Phase-5c: hoist to caller
    return Y
end

"""
    gpu_spectral_curl!(dst_tor_r, dst_tor_i, dst_pol_r, dst_pol_i,
                       src_tor_r, src_tor_i, src_pol_r, src_pol_i,
                       d1, d2, lfac, rinv, rinv2, bw) -> nothing

Spectral curl of a toroidal–poloidal field (vorticity `∇×u` from velocity, or
current `∇×B` from magnetic — the same operator):
  `dst_tor = rinv·(d2·P − lfac·rinv2·P)`,  `dst_pol = −T/rinv`,
with `P`=`src_pol`, `T`=`src_tor`.  `lfac[l+1]=l(l+1)` (length `nl`); `rinv`/`rinv2`
length `nr`; `d1`/`d2` banded `(2bw+1,nr)`.  All arrays on the same backend.
Real/imag handled independently (curl is real-linear).  The `dst_*` arrays must
NOT alias any `src_*` array.
"""
function gpu_spectral_curl!(dst_tor_r, dst_tor_i, dst_pol_r, dst_pol_i,
        src_tor_r, src_tor_i, src_pol_r, src_pol_i, d1, d2, lfac, rinv, rinv2, bw::Int)
    # 2 mat-vec launches, each with its own synchronize. The first-derivative
    # operator is kept in the signature for compatibility with existing nlops
    # bundles but is not used by the Stage-2 curl formula.
    d2Pr = similar(src_pol_r); d2Pi = similar(src_pol_i)
    gpu_batched_banded_matvec!(d2Pr, src_pol_r, d2, bw)
    gpu_batched_banded_matvec!(d2Pi, src_pol_i, d2, bw)
    lf  = reshape(lfac, :, 1, 1)
    ri  = reshape(rinv, 1, 1, :)
    ri2 = reshape(rinv2, 1, 1, :)
    @. dst_tor_r = ri * (d2Pr - lf * ri2 * src_pol_r)
    @. dst_tor_i = ri * (d2Pi - lf * ri2 * src_pol_i)
    @. dst_pol_r = -src_tor_r / ri
    @. dst_pol_i = -src_tor_i / ri
    return nothing
end
