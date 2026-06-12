# =============================================================================
# GPU Phase 1 — scalar (T/C) spectral<->physical transform, per radial level.
# The transform itself is reused from SHTnsKit: the AbstractMatrix methods below
# call the always-available serial CPU transform; the CUDA extension adds CuArray
# methods that call SHTnsKit's gpu_synthesis/gpu_analysis. Both consume/produce a
# DENSE (lmax+1, mmax+1) coefficient matrix and an (nlat, nlon) spatial matrix.
# =============================================================================

# CPU path (always available; no CUDA needed). `cfg_sht` is a SHTnsKit.SHTConfig.
# (SHTnsKit is `using`d by the parent GeoDynamo module; we reference it qualified.)
_scalar_synth(cfg_sht, alm::AbstractMatrix) = SHTnsKit.synthesis(cfg_sht, alm; real_output = true)
_scalar_anal(cfg_sht, f::AbstractMatrix)    = SHTnsKit.analysis(cfg_sht, f)

"""
    gpu_scalar_spectral_to_physical!(phys, spec, config) -> phys

Synthesize each radial level of the dense spectral field `spec`
(`(lmax+1, mmax+1, nr)` split real/imag) into the physical field `phys`
(`(nlat, nlon, nr)`), via SHTnsKit's per-level transform (GPU on `CuArray`s,
CPU otherwise).  `config` is the GeoDynamo `SHTnsKitConfig`.
"""
function gpu_scalar_spectral_to_physical!(phys::GPUPhysicalField, spec::GPUSpectralField, config;
        ws = nothing, tag::Symbol = :s2p)
    sht = config.sht_config
    nr = spec.nr
    nl, nm = size(spec.data_real, 1), size(spec.data_real, 2)
    # Pooled per-level staging buffer (overwritten each level). The buffer is a
    # concrete dense array on the field's backend → the right _scalar_synth
    # method dispatches (a bare @view would not).
    alm = gpu_scratch_complex!(ws, Symbol(tag, :_alm), spec.data_real, (nl, nm))
    nlat, nlon = size(phys.data, 1), size(phys.data, 2)
    f_o = gpu_scratch!(ws, Symbol(tag, :_fo), phys.data, (nlat, nlon))
    for k in 1:nr
        @. alm = complex(@view(spec.data_real[:, :, k]), @view(spec.data_imag[:, :, k]))
        _scalar_synth_into!(f_o, ws, sht, alm)
        phys.data[:, :, k] .= f_o
    end
    return phys
end

"""
    gpu_scalar_physical_to_spectral!(spec, phys, config) -> spec

Analyze each radial level of the physical field `phys` (`(nlat, nlon, nr)`) into
the dense spectral field `spec` (`(lmax+1, mmax+1, nr)` split real/imag), via
SHTnsKit's per-level transform (GPU on `CuArray`s, CPU otherwise).
"""
function gpu_scalar_physical_to_spectral!(spec::GPUSpectralField, phys::GPUPhysicalField, config;
        ws = nothing, tag::Symbol = :p2s)
    sht = config.sht_config
    nr = spec.nr
    nlat, nlon = size(phys.data, 1), size(phys.data, 2)
    # Pooled per-level staging copy. A concrete dense array is REQUIRED: a
    # @view SubArray would not match the ::CuArray ext method, silently falling
    # through to the CPU transform while the field is on-device.
    f_buf = gpu_scratch!(ws, Symbol(tag, :_f), phys.data, (nlat, nlon))
    nl_, nm_ = size(spec.data_real, 1), size(spec.data_real, 2)
    alm_o = gpu_scratch_complex!(ws, Symbol(tag, :_ao), spec.data_real, (nl_, nm_))
    for k in 1:nr
        f_buf .= @view phys.data[:, :, k]
        _scalar_anal_into!(alm_o, ws, sht, f_buf)
        spec.data_real[:, :, k] .= real.(alm_o)
        spec.data_imag[:, :, k] .= imag.(alm_o)
    end
    return spec
end
