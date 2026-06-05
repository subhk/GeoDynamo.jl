# =============================================================================
# GPU Phase 5g — velocity field nonlinear term (core: advection E·(u×ω) +
# Coriolis −ẑ×u), composing verified kernels: vector transform (3) → vorticity
# curl (5a) → vector transform (3) → cross + Coriolis (2) → vector analyze (3).
# Mirrors prepare_velocity_fields! + compute_velocity_body_forces! +
# finish_velocity_nonlinear! (velocity/solver.jl:10-50, numerics.jl:1160-1247),
# velocity-only part. Buoyancy (needs T) + Lorentz (needs J,B) accumulate before
# the analyze — added when those couplings are wired (5h). The force→(tor,pol)
# projection is the plain vector analysis (tangential only, no scaling, adv_r
# discarded — confirmed from finish_velocity_nonlinear!). Runs on Array + CuArray.
# =============================================================================

"""
    gpu_velocity_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, tor_r, tor_i, pol_r, pol_i,
                            config, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, lmax, bw) -> nothing

Velocity nonlinear term `nl = analyze( E·(u×ω) − ẑ×u )` (velocity-only).  `tor`/`pol`
are the velocity toroidal/poloidal spectral; `nl_tor`/`nl_pol` the toroidal/poloidal
nonlinear spectral.  `d1`/`d2` radial derivative ops, `lfac=l(l+1)`, `rinv=1/r`,
`rinv2=1/r²`, `rscale` the v_r scaling, `sinθ`/`cosθ` the Coriolis grid factors,
`E` the Ekman number.  All on the same backend; outputs distinct from inputs.
(Per-call scratch — Phase-6 may cache.)
"""
function gpu_velocity_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, tor_r, tor_i, pol_r, pol_i,
        config, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, lmax::Int, bw::Int)
    arch = arch_of(tor_r)
    sz = size(tor_r); nr = sz[3]
    spec(a, b) = GPUSpectralField{eltype(a), typeof(a)}(config, sz[1], sz[2], nr, a, b)
    ph() = allocate_gpu_physical_field(eltype(tor_r), arch, config, nr)
    # 1. velocity (tor,pol) → physical (u_r,u_θ,u_φ)
    ur = ph(); uθ = ph(); uφ = ph()
    gpu_vector_spectral_to_physical!(ur, uθ, uφ, spec(tor_r, tor_i), spec(pol_r, pol_i), config, lfac, rscale)
    # 2. vorticity ω = ∇×u (spectral)
    wtr = similar(tor_r); wti = similar(tor_i); wpr = similar(pol_r); wpi = similar(pol_i)
    gpu_spectral_curl!(wtr, wti, wpr, wpi, tor_r, tor_i, pol_r, pol_i, d1, d2, lfac, rinv, rinv2, bw)
    # 3. vorticity → physical (ω_r,ω_θ,ω_φ)
    wr = ph(); wθ = ph(); wφ = ph()
    gpu_vector_spectral_to_physical!(wr, wθ, wφ, spec(wtr, wti), spec(wpr, wpi), config, lfac, rscale)
    # 4. adv = E·(u×ω) − ẑ×u  (physical)
    ar = ph(); aθ = ph(); aφ = ph()
    gpu_cross!(ar.data, aθ.data, aφ.data, ur.data, uθ.data, uφ.data, wr.data, wθ.data, wφ.data, E)
    gpu_coriolis_sub!(ar.data, aθ.data, aφ.data, ur.data, uθ.data, uφ.data, sinθ, cosθ)
    # 5. analyze the tangential force → (nl_pol = S, nl_tor = T); adv_r discarded (CPU does the same)
    gpu_vector_physical_to_spectral!(spec(nl_tor_r, nl_tor_i), spec(nl_pol_r, nl_pol_i), aθ, aφ, config)
    return nothing
end
