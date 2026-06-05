# =============================================================================
# GPU Phase 4 — batched radial banded solve (one banded LU per degree l, reused
# across all m; all (l,m) modes solved in parallel).  A KernelAbstractions kernel
# does the per-mode forward/back substitution, replicating the CPU solve_banded!
# (src/numerics/banded_operators.jl) exactly.  Runs on the CPU backend (Array,
# locally testable) and CUDA (CuArray).  The per-l LU factors are built on the
# host with factorize_banded and packed into a (2bw+1, nr, nl) batched array.
# =============================================================================

"""
    gpu_pack_banded_lu(lus, arch) -> (2bw+1, nr, nl) array on arch's backend

Stack the per-degree `BandedLU` factors (`lus[l]` for l-slot `l`) into a single
batched array `lu_batched[:, :, l] = lus[l].lu`, on `arch`'s backend.  All factors
must share the same bandwidth `bw` and size `nr`.
"""
function gpu_pack_banded_lu(lus::AbstractVector, arch::AbstractArchitecture)
    isempty(lus) && throw(ArgumentError("gpu_pack_banded_lu: lus must be non-empty"))
    nl = length(lus)
    bw = lus[1].bandwidth
    nr = lus[1].size
    host = Array{eltype(lus[1].lu)}(undef, 2bw + 1, nr, nl)
    for l in 1:nl
        (lus[l].bandwidth == bw && lus[l].size == nr) ||
            throw(ArgumentError("gpu_pack_banded_lu: all factors must share bandwidth=$bw size=$nr"))
        host[:, :, l] .= lus[l].lu
    end
    return on_architecture(arch, host)
end

# One workitem per (l,m) mode. Each does the sequential length-nr forward/back
# substitution along dim 3, reading its degree's LU factor lu_batched[:,:,li].
# Mirrors solve_banded! exactly (banded_operators.jl:84-125). The bounded j-ranges
# guarantee the band row index bw+1+i-j ∈ [1, 2bw+1], so no in-loop guard is needed.
@kernel function _banded_solve_kernel!(X, @Const(B), @Const(lu_batched), bw::Int, nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(X)
    # @Const(B) + in-place X===B safety: B[li,mi,i] is read ONCE at step i, BEFORE
    # X[li,mi,i] is written; the back sweep reads only X, never B. So no written
    # location is re-read through the B pointer → the read-only cache (__ldg on CUDA)
    # never sees a stale value, even when X===B. Do NOT move the B read after the X write.
    # Forward: L y = b  (unit diagonal)
    @inbounds for i in 1:nr
        s = zero(T)
        for j in max(1, i - bw):(i - 1)
            s += lu_batched[bw + 1 + i - j, j, li] * X[li, mi, j]
        end
        X[li, mi, i] = B[li, mi, i] - s
    end
    # Back: U x = y
    @inbounds for i in nr:-1:1
        s = zero(T)
        for j in (i + 1):min(nr, i + bw)
            s += lu_batched[bw + 1 + i - j, j, li] * X[li, mi, j]
        end
        X[li, mi, i] = (X[li, mi, i] - s) / lu_batched[bw + 1, i, li]
    end
end

"""
    gpu_batched_banded_solve!(X, B, lu_batched, bw) -> X

Solve `A_l · X[l,m,:] = B[l,m,:]` for every `(l,m)`, where `A_l`'s banded LU is
`lu_batched[:,:,l]` (degree `l` = dim-3 index; `lu_batched` is `(2bw+1,nr,nl)`).  `X`/`B` are `(nl,nm,nr)`; in-place
`X === B` is supported.  Backend (CPU/CUDA) is inferred from `X`.
"""
function gpu_batched_banded_solve!(X, B, lu_batched, bw::Int)
    nl, nm, nr = size(X)
    backend = KernelAbstractions.get_backend(X)
    _banded_solve_kernel!(backend)(X, B, lu_batched, bw, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)  # eager sync for correctness; Phase 5 may hoist to caller for pipelining
    return X
end
