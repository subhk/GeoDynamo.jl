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
    nl_size = size(s_r)
    nr = nl_size[3]
    arch = u_r isa Array ? CPU() : GPU()
    # 1. gradient (spectral)
    gr_r = similar(s_r); gr_i = similar(s_i)
    gt_r = similar(s_r); gt_i = similar(s_i)
    gp_r = similar(s_r); gp_i = similar(s_i)
    gpu_scalar_gradient!(gr_r, gr_i, gt_r, gt_i, gp_r, gp_i, s_r, s_i, d1, mvals, rinv, lmax, bw)
    # 2. transform each ∇ component to physical (wrap in Phase-0 containers)
    spec(a, b) = GPUSpectralField{eltype(a), typeof(a)}(config, nl_size[1], nl_size[2], nr, a, b)
    grP = allocate_gpu_physical_field(eltype(u_r), arch, config, nr)
    gtP = allocate_gpu_physical_field(eltype(u_r), arch, config, nr)
    gpP = allocate_gpu_physical_field(eltype(u_r), arch, config, nr)
    # Note: signature is gpu_scalar_spectral_to_physical!(phys, spec, config) — phys first
    gpu_scalar_spectral_to_physical!(grP, spec(gr_r, gr_i), config)
    gpu_scalar_spectral_to_physical!(gtP, spec(gt_r, gt_i), config)
    gpu_scalar_spectral_to_physical!(gpP, spec(gp_r, gp_i), config)
    # 3. advection in physical space
    adv = allocate_gpu_physical_field(eltype(u_r), arch, config, nr)
    gpu_scalar_advection!(adv.data, u_r, u_θ, u_φ, grP.data, gtP.data, gpP.data)
    # 4. analyze the product back to spectral → nl
    # Note: signature is gpu_scalar_physical_to_spectral!(spec, phys, config) — spec first
    gpu_scalar_physical_to_spectral!(spec(nl_r, nl_i), adv, config)
    return nothing
end
