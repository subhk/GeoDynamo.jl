# =============================================================================
# GPU Phase 5j — velocity poloidal influence-matrix 2×2 (Green's-function)
# endpoint correction. Post-solve, projects each poloidal radial profile back
# onto zero endpoints using a precomputed two-column influence operator per
# degree l. Mirrors apply_solver_influence_matrix_correction! (erk2.jl:1795) +
# apply_solver_velocity_poloidal_influence_correction! (erk2.jl:1821). The
# per-degree Gre (nr×2) / invG (2×2) are packed host-side into batched arrays;
# the KA kernel does one rank-2 subtract per (l,m) mode. Runs on Array + CuArray.
# =============================================================================

"""
    gpu_pack_influence(influence, nl, nr, arch) -> (Gre_b, invG_b)

Flatten the per-degree `ERK2InfluenceOp` correction operators into batched arrays
on `arch`'s backend: `Gre_b` is `(nr,2,nl)` and `invG_b` is `(2,2,nl)`, indexed by
dim-3 = dense degree slot `li` (degree `l = li-1`).  Degrees absent from
`influence` (including `l=0`) get all-zero columns, so the kernel applies an exact
no-op to those modes — matching the CPU path, which skips them.

`influence` is the `Dict{Int,ERK2InfluenceOp{T}}` keyed by degree `l` (0-based).
`nl` is the number of degree slots (`lmax+1`); `nr` the radial size.
"""
function gpu_pack_influence(influence::AbstractDict{Int, ERK2InfluenceOp{T}},
        nl::Int, nr::Int, arch::AbstractArchitecture) where {T}
    Gre_b  = zeros(T, nr, 2, nl)
    invG_b = zeros(T, 2, 2, nl)
    for (l, op) in influence
        slot = l + 1                      # degree l (0-based) → dim-3 slot
        (1 <= slot <= nl) || continue
        size(op.Gre, 1) == nr ||
            throw(ArgumentError("gpu_pack_influence: Gre has $(size(op.Gre,1)) rows, expected nr=$nr"))
        Gre_b[:, :, slot]  .= op.Gre
        invG_b[:, :, slot] .= op.invG
    end
    return on_architecture(arch, Gre_b), on_architecture(arch, invG_b)
end

# One workitem per (l,m) mode. Reads the two endpoints into registers, forms the
# two correction coefficients from the degree's 2×2 invG, then subtracts the
# rank-2 Green's combination along radius. Mirrors apply_solver_influence_matrix_
# correction! (erk2.jl:1808) with bc_inner=bc_outer=0.
#
# Aliasing: di/do_ and c1/c2 are captured in registers BEFORE the write loop, so
# the in-place writes to R[li,mi,1] and R[li,mi,nr] cannot corrupt the
# coefficients. Each workitem owns its full radial column → no cross-thread races.
@kernel function _influence_correction_kernel!(R, @Const(Gre_b), @Const(invG_b), nr::Int)
    li, mi = @index(Global, NTuple)
    @inbounds begin
        di  = R[li, mi, 1]             # delta_inner (bc_inner = 0)
        do_ = R[li, mi, nr]            # delta_outer (bc_outer = 0)
        c1 = invG_b[1, 1, li] * di + invG_b[1, 2, li] * do_
        c2 = invG_b[2, 1, li] * di + invG_b[2, 2, li] * do_
        for i in 1:nr
            R[li, mi, i] -= c1 * Gre_b[i, 1, li] + c2 * Gre_b[i, 2, li]
        end
    end
end

"""
    gpu_velocity_poloidal_influence_correction!(x_r, x_i, Gre_b, invG_b) -> nothing

Apply the velocity-poloidal endpoint influence correction in-place to the real
(`x_r`) and imaginary (`x_i`) parts of a dense `(nl,nm,nr)` spectral field, using
the batched operators from [`gpu_pack_influence`](@ref).  Backend (CPU/CUDA) is
inferred from `x_r`; `x_r`, `x_i`, `Gre_b`, `invG_b` must all be on the same backend.
"""
function gpu_velocity_poloidal_influence_correction!(x_r, x_i, Gre_b, invG_b)
    nl, nm, nr = size(x_r)
    backend = KernelAbstractions.get_backend(x_r)
    _influence_correction_kernel!(backend)(x_r, Gre_b, invG_b, nr; ndrange = (nl, nm))
    _influence_correction_kernel!(backend)(x_i, Gre_b, invG_b, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)  # eager sync; Phase-5n may hoist for pipelining
    return nothing
end
