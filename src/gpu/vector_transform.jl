# =============================================================================
# GPU Phase 3 — vector (velocity/magnetic, toroidal-poloidal) transform.
# Mirrors the CPU transform (numerics.jl:846-966), which is PURELY ALGEBRAIC:
#   tangential (v_θ,v_φ) = synthesis_sphtor(S=poloidal, T=toroidal)   [no ∂/∂r]
#   radial     v_r       = scalar_synth(poloidal · l(l+1)/r)          [per-(l,r) factor]
# The AbstractMatrix methods call the always-available serial CPU sphtor transform;
# the CUDA extension adds CuArray methods → gpu_*_sphtor. v_r reuses Phase 1's
# _scalar_synth. Curls (vorticity/current) are a SEPARATE later phase — not here.
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
`lmax+1`); `rscale` is `1/r_val` (solver) or `1/r_val²` (MIE), length `nr`.
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
    gpu_vector_spectral_to_physical!(vr, vθ, vφ, tor, pol, config, lfac, rscale) -> nothing

Synthesize a toroidal–poloidal vector field to physical components.  Tangential
`(vθ, vφ)` per level via `synthesis_sphtor(poloidal, toroidal)`; radial `vr` per
level via scalar synthesis of `poloidal · lfac[l] · rscale[r]` (see `gpu_vr_scale!`).
"""
function gpu_vector_spectral_to_physical!(vr::GPUPhysicalField, vθ::GPUPhysicalField,
        vφ::GPUPhysicalField, tor::GPUSpectralField, pol::GPUSpectralField, config, lfac, rscale)
    sht = config.sht_config
    nr = pol.nr
    # v_r source coefficients (whole field), then per-level scalar synthesis.
    vr_alm_r = similar(pol.data_real); vr_alm_i = similar(pol.data_imag)
    gpu_vr_scale!(vr_alm_r, vr_alm_i, pol.data_real, pol.data_imag, lfac, rscale)
    for k in 1:nr
        S_k = complex.(view(pol.data_real, :, :, k), view(pol.data_imag, :, :, k))
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
    gpu_vector_physical_to_spectral!(tor, pol, vθ, vφ, config) -> nothing

Analyze the tangential physical components `(vθ, vφ)` into the toroidal `tor` and
poloidal `pol` spectral fields, per level, via `analysis_sphtor` (`S→pol`, `T→tor`).
`v_r` is not consumed (redundant for a solenoidal field), matching the CPU.
"""
function gpu_vector_physical_to_spectral!(tor::GPUSpectralField, pol::GPUSpectralField,
        vθ::GPUPhysicalField, vφ::GPUPhysicalField, config)
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
    return nothing
end
