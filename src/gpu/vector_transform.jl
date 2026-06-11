# =============================================================================
# GPU Phase 3 — vector (velocity/magnetic, toroidal-poloidal) transform.
# Dense single-device counterpart of the CPU Stage-2 solenoidal convention:
#   v_r       = scalar_synth(l(l+1)·P/r²)
#   v_θ, v_φ  = synthesis_sphtor(S=(∂r P)/r, T)
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
`lmax+1`); `rscale` is `1/r_val²` under the Stage-2 solenoidal convention.
`lfac`/`rscale` must reside on the same backend as the coefficient arrays —
mixing host and device arrays errors at broadcast time (use `on_architecture`).
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
    gpu_spheroidal_from_poloidal!(S_r, S_i, pol_r, pol_i, d1, rinv, bw) -> nothing

Compute the Stage-2 tangential spheroidal source `S = (∂r P)/r` for every
`(l,m)` radial profile in dense `(l,m,r)` storage.
"""
function gpu_spheroidal_from_poloidal!(S_r, S_i, pol_r, pol_i, d1, rinv, bw::Int)
    gpu_batched_banded_matvec!(S_r, pol_r, d1, bw)
    gpu_batched_banded_matvec!(S_i, pol_i, d1, bw)
    ri = reshape(rinv, 1, 1, :)
    @. S_r = S_r * ri
    @. S_i = S_i * ri
    return nothing
end

"""
    gpu_poloidal_from_radial_q!(pol_r, pol_i, lfac, rinv2) -> nothing

Convert scalar analysis of the radial component `Q` in-place to the stored
poloidal potential `P = Q / (l(l+1)/r²)`.  The `l=0` plane carries no poloidal
content and is forced to zero.
"""
function gpu_poloidal_from_radial_q!(pol_r, pol_i, lfac, rinv2)
    lf = reshape(lfac, :, 1, 1)
    ri2 = reshape(rinv2, 1, 1, :)
    lf_safe = max.(lf, one(eltype(pol_r)))
    @. pol_r = pol_r / (lf_safe * ri2)
    @. pol_i = pol_i / (lf_safe * ri2)
    pol_r[1, :, :] .= zero(eltype(pol_r))
    pol_i[1, :, :] .= zero(eltype(pol_i))
    return nothing
end

"""
    gpu_vector_spectral_to_physical!(vr, vθ, vφ, tor, pol, config, d1, lfac, rinv, rinv2, bw) -> nothing

Synthesize a solenoidal toroidal-poloidal vector field to physical components
under the Stage-2 convention: tangential `S = (∂r P)/r`, radial
`Q = l(l+1)P/r²`.  All operator arrays must live on the same backend as the
field arrays.
"""
function gpu_vector_spectral_to_physical!(vr::GPUPhysicalField, vθ::GPUPhysicalField,
        vφ::GPUPhysicalField, tor::GPUSpectralField, pol::GPUSpectralField, config,
        d1, lfac, rinv, rinv2, bw::Int)
    sht = config.sht_config
    nr = pol.nr
    S_r = similar(pol.data_real); S_i = similar(pol.data_imag)
    gpu_spheroidal_from_poloidal!(S_r, S_i, pol.data_real, pol.data_imag, d1, rinv, bw)
    vr_alm_r = similar(pol.data_real); vr_alm_i = similar(pol.data_imag)
    gpu_vr_scale!(vr_alm_r, vr_alm_i, pol.data_real, pol.data_imag, lfac, rinv2)
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
    gpu_vector_physical_to_spectral!(tor, pol, vθ, vφ, config) -> nothing

Raw tangential analysis of `(vθ, vφ)` via `analysis_sphtor`, storing `S` in
`pol` and `T` in `tor`. This is the force/QST primitive; it does not recover a
solenoidal poloidal potential from `v_r`.
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

"""
    gpu_vector_physical_to_spectral!(tor, pol, vr, vθ, vφ, config, lfac, rinv2) -> nothing

Analyze a solenoidal physical vector field under the Stage-2 convention.
Toroidal coefficients come from raw tangential sphtor analysis; poloidal
coefficients are recovered from radial scalar analysis,
`P = Q / (l(l+1)/r²)`.
"""
function gpu_vector_physical_to_spectral!(tor::GPUSpectralField, pol::GPUSpectralField,
        vr::GPUPhysicalField, vθ::GPUPhysicalField, vφ::GPUPhysicalField,
        config, lfac, rinv2)
    gpu_vector_physical_to_spectral!(tor, pol, vθ, vφ, config)
    gpu_scalar_physical_to_spectral!(pol, vr, config)
    gpu_poloidal_from_radial_q!(pol.data_real, pol.data_imag, lfac, rinv2)
    return nothing
end
