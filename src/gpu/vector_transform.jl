# =============================================================================
# GPU Phase 3 — vector (velocity/magnetic, toroidal-poloidal) transform,
# Stage-2 SOLENOIDAL convention. Mirrors the CPU per-level vector transform
# (numerics.jl `vector_spectral_to_physical_disttranspose!` /
# `vector_physical_to_spectral!`):
#   synthesis:  tangential (v_θ,v_φ) = synthesis_sphtor(S, T) with S = (∂_r P)/r
#               radial     v_r       = scalar_synth(P · l(l+1)/r²)
#   analysis:   per-level analysis_sphtor → raw (S, T); toroidal ← T; poloidal ←
#               Q-based recovery P = r²·Q/λ with Q = scalar_anal(v_r), λ=l(l+1)
#               (l=0 slot zeroed) — the exact inverse of the synthesis.
# `raw_spheroidal = true` is the tangential-basis primitive: synthesis feeds the
# stored coefficients verbatim as S; analysis stores the raw sphtor S in the
# poloidal slot (used by the Stage-4 force/induction projections).
# The AbstractMatrix methods call the always-available serial CPU sphtor
# transform; the CUDA extension adds CuArray methods → gpu_*_sphtor. v_r reuses
# Phase 1's _scalar_synth/_scalar_anal; the radial coupling reuses Phase 5a's
# gpu_batched_banded_matvec!.
# =============================================================================

# CPU path (always available). Returns (vt, vp) / (S, T).
_vector_synth_sphtor(cfg_sht, S::AbstractMatrix, T::AbstractMatrix) =
    SHTnsKit.synthesis_sphtor(cfg_sht, S, T; real_output = true)
_vector_anal_sphtor(cfg_sht, vt::AbstractMatrix, vp::AbstractMatrix) =
    SHTnsKit.analysis_sphtor(cfg_sht, vt, vp)

"""
    gpu_vr_scale!(vr_alm_r, vr_alm_i, pol_r, pol_i, lfac, rscale) -> nothing

Scale the (split-complex) poloidal coefficients into the v_r source coefficients:
`vr_alm[l,m,r] = pol[l,m,r] · lfac[l] · rscale[r]`.  `lfac[l+1]=l(l+1)` (length
`lmax+1`); `rscale` is `1/r²` (the Stage-2 solenoidal convention u_r = l(l+1)·P/r²),
length `nr`.  `lfac`/`rscale` must reside on the same backend as the coefficient
arrays — mixing host and device arrays errors at broadcast time (use
`on_architecture`).
"""
function gpu_vr_scale!(vr_alm_r, vr_alm_i, pol_r, pol_i, lfac, rscale)
    lf = reshape(lfac, :, 1, 1)
    rs = reshape(rscale, 1, 1, :)
    fac = lf .* rs          # (lmax+1, 1, nr) — matches scalar f = lfac[l]*rscale[k] in loops
    @. vr_alm_r = pol_r * fac
    @. vr_alm_i = pol_i * fac
    return nothing
end

"""
    gpu_vector_spectral_to_physical!(vr, vθ, vφ, tor, pol, config, lfac, rscale,
                                     d1, rinv, bw; raw_spheroidal=false) -> nothing

Stage-2 solenoidal synthesis: tangential `(vθ, vφ)` per level via
`synthesis_sphtor(S, T)` with `S = (d1·P)·(1/r)`; `raw_spheroidal=true` passes
the stored poloidal coefficients directly as `S` (the tangential-basis
primitive).  Radial `v_r` per level via scalar synthesis of `P·lfac·rscale`
with `rscale = 1/r²` (`u_r = l(l+1)·P/r²`).  `d1` is the banded ∂r operator
(`(2bw+1, nr)`), `rinv = 1/r`.  Mirrors the CPU
`vector_spectral_to_physical_disttranspose!` convention.  All operator arrays
must be on the same backend as the fields (use `on_architecture`).
"""
function gpu_vector_spectral_to_physical!(vr::GPUPhysicalField, vθ::GPUPhysicalField,
        vφ::GPUPhysicalField, tor::GPUSpectralField, pol::GPUSpectralField, config,
        lfac, rscale, d1, rinv, bw::Int; raw_spheroidal::Bool = false)
    sht = config.sht_config
    nr = pol.nr
    # Spheroidal scalar S: raw mode = the stored coefficients; solenoidal mode
    # = (∂_r P)/r — batched banded d1 + 1/r broadcast (NOT in place: P is
    # still needed for v_r).
    S_r = similar(pol.data_real); S_i = similar(pol.data_imag)
    if raw_spheroidal
        S_r .= pol.data_real; S_i .= pol.data_imag
    else
        gpu_batched_banded_matvec!(S_r, pol.data_real, d1, bw)
        gpu_batched_banded_matvec!(S_i, pol.data_imag, d1, bw)
        ri = reshape(rinv, 1, 1, :)
        @. S_r *= ri
        @. S_i *= ri
    end
    # v_r source coefficients from the ORIGINAL P (whole field), then per-level
    # scalar synthesis.
    vr_alm_r = similar(pol.data_real); vr_alm_i = similar(pol.data_imag)
    gpu_vr_scale!(vr_alm_r, vr_alm_i, pol.data_real, pol.data_imag, lfac, rscale)
    for k in 1:nr
        # `complex.(view, view)` materializes a fresh (nl,nm) array on the field's
        # backend (a CuArray when on-device) → the ::CuArray sphtor/scalar method
        # fires, NOT the AbstractMatrix CPU fallback. (A bare @view would NOT.)
        S_k = complex.(view(S_r, :, :, k), view(S_i, :, :, k))
        T_k = complex.(view(tor.data_real, :, :, k), view(tor.data_imag, :, :, k))
        vt, vp = _vector_synth_sphtor(sht, S_k, T_k)
        vθ.data[:, :, k] .= vt
        vφ.data[:, :, k] .= vp
        vra_k = complex.(view(vr_alm_r, :, :, k), view(vr_alm_i, :, :, k))
        vr.data[:, :, k] .= _scalar_synth(sht, vra_k)
    end
    return nothing
end

"""
    gpu_vector_physical_to_spectral!(tor, pol, vθ, vφ, config;
                                     vr=nothing, lfac=nothing, r2=nothing,
                                     raw_spheroidal=false) -> nothing

Stage-2 analysis: per level `analysis_sphtor(vθ, vφ)` → raw `(S, T)`; toroidal
← `T`.  `raw_spheroidal=true` stores the raw `S` in the poloidal slot (the
tangential-basis primitive).  The default mode does the Q-based poloidal
recovery (requires `vr`, `lfac`, `r2`): `Q` = scalar analysis of `v_r`;
`P = r²·Q/λ` with `λ = l(l+1)` (`l=0` slot zeroed) — the exact inverse of the
solenoidal synthesis.  Mirrors CPU `vector_physical_to_spectral!` +
`_poloidal_from_radial_q!`.
"""
function gpu_vector_physical_to_spectral!(tor::GPUSpectralField, pol::GPUSpectralField,
        vθ::GPUPhysicalField, vφ::GPUPhysicalField, config;
        vr = nothing, lfac = nothing, r2 = nothing, raw_spheroidal::Bool = false)
    sht = config.sht_config
    nr = pol.nr
    for k in 1:nr
        # Plain indexing (NOT @view): a @view SubArray would miss the ::CuArray
        # sphtor method and silently run on CPU against device data (see Phase 1).
        vt_k = vθ.data[:, :, k]
        vp_k = vφ.data[:, :, k]
        S_k, T_k = _vector_anal_sphtor(sht, vt_k, vp_k)
        pol.data_real[:, :, k] .= real.(S_k)
        pol.data_imag[:, :, k] .= imag.(S_k)
        tor.data_real[:, :, k] .= real.(T_k)
        tor.data_imag[:, :, k] .= imag.(T_k)
    end
    raw_spheroidal && return nothing
    (vr === nothing || lfac === nothing || r2 === nothing) && throw(ArgumentError(
        "gpu_vector_physical_to_spectral!: default (solenoidal) mode requires vr, lfac, r2"))
    # Q-based poloidal recovery: Q = scalar analysis of v_r, then P = r²·Q/λ
    # (λ = l(l+1); the l=0 slot carries no poloidal content → zeroed via inv_lfac).
    for k in 1:nr
        q_k = _scalar_anal(sht, vr.data[:, :, k])
        pol.data_real[:, :, k] .= real.(q_k)
        pol.data_imag[:, :, k] .= imag.(q_k)
    end
    inv_lfac = similar(lfac)
    @. inv_lfac = ifelse(lfac > 0, 1 / lfac, zero(eltype(lfac)))
    lf = reshape(inv_lfac, :, 1, 1)
    rr2 = reshape(r2, 1, 1, :)
    @. pol.data_real *= lf * rr2
    @. pol.data_imag *= lf * rr2
    return nothing
end
