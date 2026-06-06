# =============================================================================
# GPU Phase 5l — magnetic conducting-inner-core kernels: the CNAB2 history flux
# φ0 = d1_top·(M_ic⁻¹ b) and the inner-core profile reconstruction S = M_ic⁻¹ b,
# batched over all spectral modes. Mirrors inner_core_history_flux /
# reconstruct_inner_core (src/physics/magnetic/inner_core.jl:168-192). Each is
# a per-degree banded matvec (L·S_old, reusing 5c) + CNAB2 assembly + boundary
# rows + length-Nic banded solve (reusing 5d) [+ a d1_top·y reduction for the
# flux]. The per-degree InnerCoreAdmittance operators are packed into batched
# arrays; non-stored degrees (incl l=0) get a ZERO L and an IDENTITY LU so the
# batched pass over empty modes is a safe no-op (no divide-by-zero in the solve).
# Runs on Array + CuArray. (Per-call scratch — Phase-6 may cache.)
# =============================================================================

"""
    gpu_pack_inner_core(adm::InnerCoreAdmittance, nl, arch)
        -> (; lin_ic, lu_ic, d1_top, inv_dt, weight, Nic, bw)

Flatten the per-degree conducting-inner-core operators into batched arrays on
`arch`'s backend: `lin_ic` / `lu_ic` are `(2bw+1, Nic, nl)` indexed by dim-3 =
degree slot `li` (degree `l = li-1`).  Stored degrees carry `adm.lin[…].data` and
`adm.factor[…].lu`; non-stored degrees (including `l=0`) get a **zero** `L` and an
**identity** LU (diagonal row = 1) so the batched solve treats those empty modes as
`x = b` rather than dividing by a zero pivot.  Returns the operators plus the CNAB2
scalars `inv_dt = 1/dt`, `weight = 1−θ`, and `Nic`/`bw`.
"""
function gpu_pack_inner_core(adm::InnerCoreAdmittance{T}, nl::Int, arch::AbstractArchitecture) where {T}
    Nic = adm.Nic
    bw = isempty(adm.factor) ? 0 : adm.factor[1].bandwidth
    lin = zeros(T, 2bw + 1, Nic, nl)
    lu = zeros(T, 2bw + 1, Nic, nl)
    @inbounds for j in 1:Nic, li in 1:nl
        lu[bw + 1, j, li] = one(T)            # default = identity LU (overwritten for stored l)
    end
    for (l, idx) in adm.lookup
        slot = l + 1                          # degree l (0-based) → dim-3 slot
        (1 <= slot <= nl) || continue
        (adm.factor[idx].bandwidth == bw && adm.factor[idx].size == Nic) ||
            throw(ArgumentError("gpu_pack_inner_core: factor for l=$l has bw/size mismatch"))
        lin[:, :, slot] .= adm.lin[idx].data
        lu[:, :, slot]  .= adm.factor[idx].lu
    end
    return (; lin_ic = on_architecture(arch, lin), lu_ic = on_architecture(arch, lu),
              d1_top = on_architecture(arch, Vector{T}(adm.d1_top)),
              inv_dt = one(T) / adm.dt, weight = one(T) - adm.theta, Nic = Nic, bw = bw)
end

# One workitem per (l,m). φ0[li,mi] = Σ_i d1_top[i]·y[li,mi,i], ascending-i. The
# CPU reference reduces with `dot(d1_top, y)` (inner_core.jl:174), which for
# `Float64` dispatches to **BLAS dot** (a SIMD/blocked summation) — NOT a scalar
# ascending loop. So this kernel matches it only to a ULP, never bit-for-bit; it
# is therefore the GPU-backend path. On the CPU backend we instead call BLAS `dot`
# per mode (`_ic_flux_reduce_blas!`) to reproduce the reference operand-for-operand.
@kernel function _ic_flux_reduce_kernel!(φ0, @Const(y), @Const(d1_top), Nic::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(φ0)
    s = zero(T)
    @inbounds for i in 1:Nic
        s += d1_top[i] * y[li, mi, i]
    end
    @inbounds φ0[li, mi] = s
end

# CPU-backend reduction: per mode, `dot(d1_top, contiguous slice)` — the SAME
# BLAS `dot` on a materialized `Vector` that the reference uses, so the result is
# bit-exact (`==`) vs inner_core_history_flux. A strided dim-3 view would take a
# different BLAS path and round differently, so the slice is materialized.
function _ic_flux_reduce_blas!(φ0, y, d1_top)
    nl, nm, _ = size(y)
    @inbounds for mi in 1:nm, li in 1:nl
        φ0[li, mi] = LinearAlgebra.dot(d1_top, y[li, mi, :])
    end
    return nothing
end

# CNAB2 history assembly b = inv_dt·S_old + weight·(L·S_old), written into `b_*`.
# Mirrors _ic_build_bic (inner_core.jl:149-157): same op order, same scalars.
function _gpu_ic_build_bic!(b_r, b_i, S_old_r, S_old_i, ic)
    Lx_r = similar(S_old_r); Lx_i = similar(S_old_i)     # Phase-6: workspace
    gpu_batched_banded_matvec_perl!(Lx_r, S_old_r, ic.lin_ic, ic.bw)
    gpu_batched_banded_matvec_perl!(Lx_i, S_old_i, ic.lin_ic, ic.bw)
    b_r .= ic.inv_dt .* S_old_r .+ ic.weight .* Lx_r
    b_i .= ic.inv_dt .* S_old_i .+ ic.weight .* Lx_i
    return nothing
end

"""
    gpu_inner_core_history_flux!(φ0_r, φ0_i, S_old_r, S_old_i, ic) -> nothing

Per-mode conducting-inner-core CNAB2 history flux `φ0 = d1_top·y`, where
`M_ic y = b` with `b = inv_dt·S_old + weight·L·S_old` and homogeneous boundary
rows (`b[1]=b[Nic]=0`).  `S_old_*` are dense `(nl,nm,Nic)` inner-core spectra;
`φ0_*` are `(nl,nm)`.  `ic` is the bundle from [`gpu_pack_inner_core`](@ref).
Mirrors `inner_core_history_flux` (inner_core.jl:168-175).  All arrays on the
same backend.
"""
function gpu_inner_core_history_flux!(φ0_r, φ0_i, S_old_r, S_old_i, ic)
    nl, nm, _ = size(S_old_r)
    y_r = similar(S_old_r); y_i = similar(S_old_i)       # Phase-6: workspace
    _gpu_ic_build_bic!(y_r, y_i, S_old_r, S_old_i, ic)
    z = similar(φ0_r, nl, nm); fill!(z, zero(eltype(φ0_r)))   # zero BC rows (inner=outer=0)
    gpu_implicit_solve_field!(y_r, y_i, ic.lu_ic, z, z, z, z, ic.bw)
    backend = KernelAbstractions.get_backend(φ0_r)
    if backend isa KernelAbstractions.CPU
        # CPU: BLAS `dot` per mode → bit-exact `==` vs the CPU reference.
        _ic_flux_reduce_blas!(φ0_r, y_r, ic.d1_top)
        _ic_flux_reduce_blas!(φ0_i, y_i, ic.d1_top)
    else
        # GPU (or any non-CPU backend): the device reduction kernel (≈ to a ULP).
        _ic_flux_reduce_kernel!(backend)(φ0_r, y_r, ic.d1_top, ic.Nic; ndrange = (nl, nm))
        _ic_flux_reduce_kernel!(backend)(φ0_i, y_i, ic.d1_top, ic.Nic; ndrange = (nl, nm))
        KernelAbstractions.synchronize(backend)
    end
    return nothing
end

"""
    gpu_reconstruct_inner_core!(S_new_r, S_new_i, S_old_r, S_old_i, g_r, g_i, ic) -> nothing

Per-mode conducting-inner-core reconstruction: solve `M_ic S = b` with
`b = inv_dt·S_old + weight·L·S_old`, regularity `b[1]=0`, and ICB Dirichlet
`b[Nic]=g` (the outer-core value at the ICB).  `S_old_*` dense `(nl,nm,Nic)`,
`g_*` `(nl,nm)`; the solution is written to `S_new_*`.  Mirrors
`reconstruct_inner_core` (inner_core.jl:185-192).  `S_new_*` may not alias
`S_old_*`.  All arrays on the same backend.
"""
function gpu_reconstruct_inner_core!(S_new_r, S_new_i, S_old_r, S_old_i, g_r, g_i, ic)
    nl, nm, _ = size(S_old_r)
    _gpu_ic_build_bic!(S_new_r, S_new_i, S_old_r, S_old_i, ic)   # b into S_new
    z = similar(g_r); fill!(z, zero(eltype(g_r)))                # inner BC = 0
    gpu_implicit_solve_field!(S_new_r, S_new_i, ic.lu_ic, z, z, g_r, g_i, ic.bw)  # outer BC = g
    return nothing
end
