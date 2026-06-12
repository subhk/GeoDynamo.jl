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

# ── In-place per-level transforms (Array backend) ───────────────────────────
# The host (Array) path routes through a pooled SHTPlan → allocation-free and
# bit-identical to the functional calls (verified). Device arrays fall back to
# the functional SHTnsKit gpu path (no in-place GPU API; the CUDA pool manages
# those allocations). With no workspace the functional path is used as well —
# building an SHTPlan per call would cost more than it saves.

_get_sht_plan!(ws::GPUWorkspace, sht) =
    get!(() -> SHTnsKit.SHTPlan(sht), ws.pool, :sht_plan)

function _scalar_synth_into!(out, ws, sht, alm)
    if ws isa GPUWorkspace && out isa Matrix
        SHTnsKit.synthesis!(_get_sht_plan!(ws, sht), out, alm)
    else
        out .= _scalar_synth(sht, alm)
    end
    return out
end

function _scalar_anal_into!(alm_out, ws, sht, f)
    if ws isa GPUWorkspace && f isa Matrix
        SHTnsKit.analysis!(_get_sht_plan!(ws, sht), alm_out, f)
    else
        alm_out .= _scalar_anal(sht, f)
    end
    return alm_out
end

function _vector_synth_sphtor_into!(vt, vp, ws, sht, S, T)
    if ws isa GPUWorkspace && vt isa Matrix
        SHTnsKit.synthesis_sphtor!(_get_sht_plan!(ws, sht), vt, vp, S, T)
    else
        vt2, vp2 = _vector_synth_sphtor(sht, S, T)
        vt .= vt2; vp .= vp2
    end
    return vt, vp
end

function _vector_anal_sphtor_into!(S, T, ws, sht, vt, vp)
    if ws isa GPUWorkspace && vt isa Matrix
        SHTnsKit.analysis_sphtor!(_get_sht_plan!(ws, sht), S, T, vt, vp)
    else
        S2, T2 = _vector_anal_sphtor(sht, vt, vp)
        S .= S2; T .= T2
    end
    return S, T
end

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
        d1, lfac, rinv, rinv2, bw::Int; ws = nothing, tag::Symbol = :v_s2p)
    sht = config.sht_config
    nr = pol.nr
    nl, nm = size(pol.data_real, 1), size(pol.data_real, 2)
    S_r = gpu_scratch!(ws, Symbol(tag, :_Sr), pol.data_real)
    S_i = gpu_scratch!(ws, Symbol(tag, :_Si), pol.data_imag)
    gpu_spheroidal_from_poloidal!(S_r, S_i, pol.data_real, pol.data_imag, d1, rinv, bw)
    vr_alm_r = gpu_scratch!(ws, Symbol(tag, :_qr), pol.data_real)
    vr_alm_i = gpu_scratch!(ws, Symbol(tag, :_qi), pol.data_imag)
    gpu_vr_scale!(vr_alm_r, vr_alm_i, pol.data_real, pol.data_imag, lfac, rinv2)
    # Pooled per-level staging buffers (overwritten each level): concrete dense
    # arrays on the field's backend → the ::CuArray sphtor/scalar methods fire,
    # NOT the AbstractMatrix CPU fallback. (A bare @view would NOT.)
    S_k = gpu_scratch_complex!(ws, Symbol(tag, :_Sk), pol.data_real, (nl, nm))
    T_k = gpu_scratch_complex!(ws, Symbol(tag, :_Tk), pol.data_real, (nl, nm))
    Q_k = gpu_scratch_complex!(ws, Symbol(tag, :_Qk), pol.data_real, (nl, nm))
    nlat, nlon = size(vr.data, 1), size(vr.data, 2)
    vt_o = gpu_scratch!(ws, Symbol(tag, :_vto), vr.data, (nlat, nlon))
    vp_o = gpu_scratch!(ws, Symbol(tag, :_vpo), vr.data, (nlat, nlon))
    qo = gpu_scratch!(ws, Symbol(tag, :_qo), vr.data, (nlat, nlon))
    for k in 1:nr
        @. S_k = complex(@view(S_r[:, :, k]), @view(S_i[:, :, k]))
        @. T_k = complex(@view(tor.data_real[:, :, k]), @view(tor.data_imag[:, :, k]))
        _vector_synth_sphtor_into!(vt_o, vp_o, ws, sht, S_k, T_k)
        vθ.data[:, :, k] .= vt_o
        vφ.data[:, :, k] .= vp_o
        @. Q_k = complex(@view(vr_alm_r[:, :, k]), @view(vr_alm_i[:, :, k]))
        _scalar_synth_into!(qo, ws, sht, Q_k)
        vr.data[:, :, k] .= qo
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
        vθ::GPUPhysicalField, vφ::GPUPhysicalField, config; ws = nothing, tag::Symbol = :v_p2s)
    sht = config.sht_config
    nr = pol.nr
    nlat, nlon = size(vθ.data, 1), size(vθ.data, 2)
    # Pooled per-level staging copies (concrete dense arrays — a @view SubArray
    # would miss the ::CuArray sphtor method and silently run on CPU, see Phase 1).
    vt_k = gpu_scratch!(ws, Symbol(tag, :_vt), vθ.data, (nlat, nlon))
    vp_k = gpu_scratch!(ws, Symbol(tag, :_vp), vφ.data, (nlat, nlon))
    nl_, nm_ = size(pol.data_real, 1), size(pol.data_real, 2)
    S_k = gpu_scratch_complex!(ws, Symbol(tag, :_So), pol.data_real, (nl_, nm_))
    T_k = gpu_scratch_complex!(ws, Symbol(tag, :_To), pol.data_real, (nl_, nm_))
    for k in 1:nr
        vt_k .= @view vθ.data[:, :, k]
        vp_k .= @view vφ.data[:, :, k]
        _vector_anal_sphtor_into!(S_k, T_k, ws, sht, vt_k, vp_k)
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
