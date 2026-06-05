# =============================================================================
# GPU Phase 5b — scalar gradient ∇s (temperature/composition), matching the CPU
# (src/physics/nonlinear.jl:18-280).  Radial ∇r = d1·s (reuse Phase 5a mat-vec,
# no 1/r).  Longitudinal ∇φ = i·m·s (element-wise).  Latitudinal ∇θ = Legendre
# (l±1,m) recurrence (KA kernel).  Geometric 1/r on the tangential (∇θ,∇φ) only.
# Feeds the Phase-2 scalar advection.  Curls/derivatives reused from Phase 5a.
# =============================================================================

"""
    gpu_phi_gradient!(gφ_r, gφ_i, s_r, s_i, mvals) -> nothing

Longitudinal gradient `∂s/∂φ = i·m·s`: `gφ_r = −m·s_i`, `gφ_i = m·s_r`.
`mvals[m+1] = m` (length `nm`, m-slot index).
"""
function gpu_phi_gradient!(gφ_r, gφ_i, s_r, s_i, mvals)
    mm = reshape(mvals, 1, :, 1)            # (1, nm, 1) — m over the m-slot axis
    # mvals is typically Float64; the broadcast promotes to the output element type
    # (Float32-safe — no allocation beyond the lazy reshape).
    @. gφ_r = -mm * s_i
    @. gφ_i = mm * s_r
    return nothing
end

# One workitem per (l-slot li, m-slot mi). l=li-1, m=mi-1 (m≥0). Reads l±1 neighbors
# at [li±1, mi, r]. Mirrors compute_theta_gradient_spectral!:84-114 exactly.
# Empty slots (l<m) → 0 (the A₊ sqrt would be NaN there; guard skips them).
@kernel function _theta_grad_kernel!(gθr, gθi, @Const(sr), @Const(si), lmax::Int, nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(gθr)
    l = li - 1
    m = mi - 1
    @inbounds if l < m
        for r in 1:nr
            gθr[li, mi, r] = zero(T)
            gθi[li, mi, r] = zero(T)
        end
    else
        for r in 1:nr
            dtr = zero(T)
            dti = zero(T)
            if l < lmax
                ap = T(l) * sqrt(T((l + m + 1) * (l - m + 1)) / T((2l + 1) * (2l + 3)))
                dtr += ap * sr[li + 1, mi, r]
                dti += ap * si[li + 1, mi, r]
            end
            if l > m
                am = -T(l + 1) * sqrt(T((l + m) * (l - m)) / T((2l - 1) * (2l + 1)))
                dtr += am * sr[li - 1, mi, r]
                dti += am * si[li - 1, mi, r]
            end
            gθr[li, mi, r] = dtr
            gθi[li, mi, r] = dti
        end
    end
end

"""
    gpu_theta_gradient!(gθ_r, gθ_i, s_r, s_i, lmax) -> nothing

Latitudinal gradient `∂s/∂θ` via the Legendre `(l±1, m)` recurrence (matching the
CPU). Empty slots (`l < m`) are zeroed. Outputs must be distinct from inputs (the
recurrence reads neighbor l-slots). Backend inferred from `gθ_r`.
"""
function gpu_theta_gradient!(gθ_r, gθ_i, s_r, s_i, lmax::Int)
    nl, nm, nr = size(gθ_r)
    backend = KernelAbstractions.get_backend(gθ_r)
    _theta_grad_kernel!(backend)(gθ_r, gθ_i, s_r, s_i, lmax, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    gpu_scalar_gradient!(gr_r,gr_i, gθ_r,gθ_i, gφ_r,gφ_i, s_r,s_i,
                         d1, mvals, rinv, lmax, bw) -> nothing

Assemble the scalar gradient: `∇r = d1·s` (banded mat-vec, NOT scaled by 1/r),
`∇θ` via the Legendre recurrence, `∇φ = i·m·s`, then multiply the tangential
components (`∇θ`,`∇φ`) by `rinv = 1/r` (0 at r=0).  `d1` banded `(2bw+1,nr)`;
`mvals` length-`nm`; `rinv` length-`nr`; all on the same backend; outputs distinct
from `s_r`/`s_i`.
"""
function gpu_scalar_gradient!(gr_r, gr_i, gθ_r, gθ_i, gφ_r, gφ_i, s_r, s_i,
        d1, mvals, rinv, lmax::Int, bw::Int)
    gpu_batched_banded_matvec!(gr_r, s_r, d1, bw)     # ∇r real (no 1/r)
    gpu_batched_banded_matvec!(gr_i, s_i, d1, bw)     # ∇r imag
    gpu_theta_gradient!(gθ_r, gθ_i, s_r, s_i, lmax)   # ∇θ (pre-1/r)
    gpu_phi_gradient!(gφ_r, gφ_i, s_r, s_i, mvals)    # ∇φ (pre-1/r)
    ri = reshape(rinv, 1, 1, :)                        # geometric 1/r on tangential only
    @. gθ_r *= ri
    @. gθ_i *= ri
    @. gφ_r *= ri
    @. gφ_i *= ri
    return nothing
end
