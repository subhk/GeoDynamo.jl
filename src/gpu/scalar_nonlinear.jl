# =============================================================================
# GPU Phase 5e — scalar field nonlinear term (explicit half), composing the
# verified kernels: gradient (5b) → transform ∇s to physical (1) → advection (2)
# → analyze (1).  Mirrors solver_compute_temperature_nonlinear! (temperature/
# solver.jl:71-119, pure-advection part).  Each sub-kernel is verified vs CPU per
# phase; this wires them.  Runs on Array (locally testable) and CuArray.
# =============================================================================

"""
    gpu_scalar_nonlinear!(nl_r, nl_i, s_r, s_i, u_r, u_θ, u_φ, config, d1, mvals, rinv, lmax, bw) -> nothing

Compute a scalar field's nonlinear term `nl = analyze( −(u·∇s) )`: gradient of
`s` (spectral) → transform the gradient components to physical → advection
`−(u_r·∇r + u_θ·∇θ + u_φ·∇φ)` against the supplied physical velocity → analyze the
product back to spectral.  `nl_*`/`s_*` are dense `(nl,nm,nr)`; `u_*` physical
`(nlat,nlon,nr)`; `d1`/`mvals`/`rinv` as in `gpu_scalar_gradient!`.  All on the
same backend; `nl_*` distinct from `s_*`.  (Per-call scratch — Phase-6 may cache.)
"""
function gpu_scalar_nonlinear!(nl_r, nl_i, s_r, s_i, u_r, u_θ, u_φ, config, d1, mvals, rinv, lmax::Int, bw::Int)
    spec_size = size(s_r)
    nr = spec_size[3]
    arch = arch_of(u_r)
    sht = config.sht_config
    spec(a, b) = GPUSpectralField{eltype(a), typeof(a)}(config, spec_size[1], spec_size[2], nr, a, b)
    # 1. radial gradient: ∇r = d1·s (spectral), then scalar synthesis
    gr_r = similar(s_r); gr_i = similar(s_i)
    gpu_batched_banded_matvec!(gr_r, s_r, d1, bw)
    gpu_batched_banded_matvec!(gr_i, s_i, d1, bw)
    grP = allocate_gpu_physical_field(eltype(u_r), arch, config, nr)
    gpu_scalar_spectral_to_physical!(grP, spec(gr_r, gr_i), config)
    # 2. EXACT tangential gradient via raw sphtor synthesis of the scalar's own
    #    coefficients: S = s, T = 0 gives (∂θf, (1/sinθ)∂φf) on the unit sphere;
    #    dividing by r yields the true physical components. Mirrors the CPU
    #    transform_field_and_gradients_to_physical! (the legacy θ/φ spectral
    #    recurrences are sinθ-weighted projections and were retired there).
    gtP = allocate_gpu_physical_field(eltype(u_r), arch, config, nr)
    gpP = allocate_gpu_physical_field(eltype(u_r), arch, config, nr)
    zero_r = fill!(similar(s_r), zero(eltype(s_r)))
    zero_i = fill!(similar(s_i), zero(eltype(s_i)))
    for k in 1:nr
        # `complex.(view, view)` materializes a fresh (nl,nm) array on the field's
        # backend → the ::CuArray sphtor method fires (see vector_transform.jl).
        S_k = complex.(view(s_r, :, :, k), view(s_i, :, :, k))
        T_k = complex.(view(zero_r, :, :, k), view(zero_i, :, :, k))
        gt, gp = _vector_synth_sphtor(sht, S_k, T_k)
        gtP.data[:, :, k] .= gt
        gpP.data[:, :, k] .= gp
    end
    ri = reshape(rinv, 1, 1, :)
    gtP.data .*= ri
    gpP.data .*= ri
    # 3. advection in physical space
    adv = allocate_gpu_physical_field(eltype(u_r), arch, config, nr)
    gpu_scalar_advection!(adv.data, u_r, u_θ, u_φ, grP.data, gtP.data, gpP.data)
    # 4. analyze the product back to spectral → nl
    gpu_scalar_physical_to_spectral!(spec(nl_r, nl_i), adv, config)
    return nothing
end
