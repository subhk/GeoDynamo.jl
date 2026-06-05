# =============================================================================
# GPU Phase 5c — CNAB2 implicit RHS assembly.  The linear term L_l·u uses a
# PER-DEGREE-l banded matrix (linear_matrices[l]), so this needs a per-l batched
# mat-vec (like Phase 5a's, but the matrix is indexed by l = dim-3 slice).
# Mirrors apply_banded_full! (numerics.jl:1765) + build_rhs_cnab2! (implicit.jl:216).
# KA + broadcast → runs on Array (locally testable) and CuArray.
# =============================================================================

# One workitem per (l,m). Y[li,mi,i] = Σ_{j∈[max(1,i-bw),min(nr,i+bw)]}
# mat_batched[bw+1+i-j, j, li] · X[li,mi,j].  Same ascending-j accumulation as
# apply_banded_full! → exact == on CPU.  Y ≠ X.
@kernel function _perl_matvec_kernel!(Y, @Const(X), @Const(mat_batched), bw::Int, nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(Y)
    @inbounds for i in 1:nr
        s = zero(T)
        for j in max(1, i - bw):min(nr, i + bw)
            s += mat_batched[bw + 1 + i - j, j, li] * X[li, mi, j]
        end
        Y[li, mi, i] = s
    end
end

"""
    gpu_batched_banded_matvec_perl!(Y, X, mat_batched, bw) -> Y

Per-degree-`l` banded mat-vec: `Y[l,m,:] = mat_batched[:,:,l] · X[l,m,:]`, where
`mat_batched` is `(2bw+1, nr, nl)` (degree `l` = dim-3 slice).  `Y`/`X` are
`(nl,nm,nr)`; `Y` must NOT alias `X`.  Backend inferred from `Y`.
"""
function gpu_batched_banded_matvec_perl!(Y, X, mat_batched, bw::Int)
    nl, nm, nr = size(Y)
    backend = KernelAbstractions.get_backend(Y)
    _perl_matvec_kernel!(backend)(Y, X, mat_batched, bw, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)  # eager sync; Phase-5c-step: hoist to caller
    return Y
end

"""
    gpu_build_rhs_cnab2!(rr, ri, ur, ui, nr_, ni_, pr, pi_, lin_batched, inv_dt, linear_weight, bw) -> nothing

Assemble the CNAB2 implicit RHS (split real/imag):
`rhs = inv_dt·u + 1.5·nl − 0.5·nl_prev + linear_weight·(lin·u)`, where `lin·u` is the
per-l banded mat-vec of the linear operator (`lin_batched` `(2bw+1,nr,nl)`).
`inv_dt = mass_coeff/dt`, `linear_weight = 1−θ`.  All arrays `(nl,nm,nr)` on the
same backend; outputs distinct from inputs.
"""
function gpu_build_rhs_cnab2!(rr, ri, ur, ui, nr_, ni_, pr, pi_, lin_batched, inv_dt, linear_weight, bw::Int)
    Lur = similar(ur); Lui = similar(ui)
    gpu_batched_banded_matvec_perl!(Lur, ur, lin_batched, bw)
    gpu_batched_banded_matvec_perl!(Lui, ui, lin_batched, bw)
    T = eltype(rr)
    a = T(inv_dt); lw = T(linear_weight); c32 = T(1.5); c12 = T(0.5)
    @. rr = a * ur + c32 * nr_ - c12 * pr + lw * Lur
    @. ri = a * ui + c32 * ni_ - c12 * pi_ + lw * Lui
    return nothing
end
