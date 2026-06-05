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
function gpu_scalar_spectral_to_physical!(phys::GPUPhysicalField, spec::GPUSpectralField, config)
    sht = config.sht_config
    nr = spec.nr
    for k in 1:nr
        # `complex.(view, view)` materializes a fresh (lmax+1,mmax+1) matrix on the
        # field's backend (a CuArray when the field is on-device) → dispatches to the
        # right _scalar_synth method. (Per-level alloc; preallocation is a later opt.)
        alm_k = complex.(view(spec.data_real, :, :, k), view(spec.data_imag, :, :, k))
        f_k = _scalar_synth(sht, alm_k)              # (nlat, nlon)
        phys.data[:, :, k] .= f_k
    end
    return phys
end

"""
    gpu_scalar_physical_to_spectral!(spec, phys, config) -> spec

Analyze each radial level of the physical field `phys` (`(nlat, nlon, nr)`) into
the dense spectral field `spec` (`(lmax+1, mmax+1, nr)` split real/imag), via
SHTnsKit's per-level transform (GPU on `CuArray`s, CPU otherwise).
"""
function gpu_scalar_physical_to_spectral!(spec::GPUSpectralField, phys::GPUPhysicalField, config)
    sht = config.sht_config
    nr = spec.nr
    for k in 1:nr
        # Plain indexing (NOT @view) is REQUIRED: a @view would be a SubArray that
        # does NOT match the ::CuArray ext method, silently falling through to the
        # CPU transform while the field is on-device → wrong results, no error.
        f_k = phys.data[:, :, k]
        alm_k = _scalar_anal(sht, f_k)                # (lmax+1, mmax+1) complex
        spec.data_real[:, :, k] .= real.(alm_k)
        spec.data_imag[:, :, k] .= imag.(alm_k)
    end
    return spec
end
