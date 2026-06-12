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
@kernel function _banded_matvec_perl_kernel!(Y, @Const(X), @Const(mat_batched), bw::Int, nr::Int)
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
    _banded_matvec_perl_kernel!(backend)(Y, X, mat_batched, bw, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)  # eager sync; Phase-5c-step: hoist to caller
    return Y
end

"""
    gpu_build_rhs_cnab2!(rr, ri, ur, ui, nr_, ni_, pr, pi_, lin_batched, inv_dt, linear_weight, bw) -> nothing

Assemble the CNAB2 implicit RHS (split real/imag):
`rhs = inv_dt·u + 1.5·nl − 0.5·nl_prev + linear_weight·(lin·u)`, where `lin·u` is the
per-l banded mat-vec of the linear operator (`lin_batched` `(2bw+1,nr,nl)`).
`inv_dt = mass_coeff/dt`, `linear_weight = 1−θ`.  All arrays `(nl,nm,nr)` on the
same backend; outputs distinct from inputs.  (`pi_` is the previous-step imaginary
nonlinear term — the trailing underscore avoids shadowing Julia's `pi` constant.)
"""
function gpu_build_rhs_cnab2!(rr, ri, ur, ui, nr_, ni_, pr, pi_, lin_batched, inv_dt, linear_weight, bw::Int;
        ws = nothing, tag::Symbol = :rhs)
    Lur = gpu_scratch!(ws, Symbol(tag, :_Lr), ur)
    Lui = gpu_scratch!(ws, Symbol(tag, :_Li), ui)
    gpu_batched_banded_matvec_perl!(Lur, ur, lin_batched, bw)
    gpu_batched_banded_matvec_perl!(Lui, ui, lin_batched, bw)
    T = eltype(rr)
    a = T(inv_dt); lw = T(linear_weight); three_halves = T(1.5); one_half = T(0.5)
    # GPU always runs both mat-vecs; when θ=1 (fully implicit) lw=0 and the linear
    # terms drop out (CPU short-circuits via `iszero(linear_weight)` — θ=0.5 CNAB2 is the norm).
    @. rr = a * ur + three_halves * nr_ - one_half * pr + lw * Lur
    @. ri = a * ui + three_halves * ni_ - one_half * pi_ + lw * Lui
    return nothing
end
