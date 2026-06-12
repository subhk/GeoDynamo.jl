# =============================================================================
# GPU Phase 5g — velocity field nonlinear term (core: advection E·(u×ω) +
# Coriolis −ẑ×u), composing verified kernels: vector transform (3) → vorticity
# curl (5a) → vector transform (3) → cross + Coriolis (2) → vector analyze (3).
# Mirrors prepare_velocity_fields! + compute_velocity_body_forces! +
# finish_velocity_nonlinear! (velocity/solver.jl:10-50, numerics.jl:1160-1247),
# velocity-only part. Buoyancy (needs T) + Lorentz (needs J,B) accumulate before
# the analyze — added in Phase 5i via optional keyword inputs. The force→(tor,pol)
# projection is the plain vector analysis (tangential only, no scaling, adv_r
# discarded — confirmed from finish_velocity_nonlinear!). Runs on Array + CuArray.
# =============================================================================

"""
    gpu_velocity_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, tor_r, tor_i, pol_r, pol_i,
                            config, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, lmax, bw;
                            T_phys=nothing, thermal_factor=zero(eltype(tor_r)), r_vec=nothing,
                            C_phys=nothing, comp_factor=zero(eltype(tor_r)),
                            J_r=nothing, J_θ=nothing, J_φ=nothing,
                            B_r=nothing, B_θ=nothing, B_φ=nothing, lorentz_coeff=zero(eltype(tor_r))) -> nothing

Velocity nonlinear term: `nl = analyze( E·(u×ω) − ẑ×u [+ buoyancy + Lorentz] )`.

`tor`/`pol` are the velocity toroidal/poloidal spectral coefficients; `nl_tor`/`nl_pol`
receive the toroidal/poloidal nonlinear spectral output.  `d1`/`d2` are radial derivative
operators, `lfac=l(l+1)`, `rinv=1/r`, `rinv2=1/r²`, `rscale` the v_r scaling,
`sinθ`/`cosθ` the Coriolis grid factors, `E` the Ekman number.

**Optional coupled forcing** (Phase 5i) accumulated between the Coriolis and the analyze,
matching the CPU accumulation order `thermal → compositional → Lorentz`:

- **Thermal buoyancy:** supply `T_phys` (physical `(nlat,nlon,nr)`), `thermal_factor` (e.g.
  `(Pm/Pr)·Ra`), and `r_vec` (radial values, length `nr`).  Adds `thermal_factor·r·T_phys`
  to the radial component of the advection accumulator.
- **Compositional buoyancy:** supply `C_phys` and `comp_factor` (e.g. `(Pm/Sc)·Ra_C`).
  Adds `comp_factor·r·C_phys` to the radial component.
- **Lorentz force:** supply `J_r/J_θ/J_φ` (current density) and `B_r/B_θ/B_φ` (magnetic
  field) in physical space, plus `lorentz_coeff` (e.g. `1/Pm`).  Adds
  `lorentz_coeff·(J×B)` to all three components.

With no keywords (the defaults), the function is exactly the Phase-5g velocity-only path.
All arrays (including `r_vec`, `sinθ`, `cosθ`) must live on the same backend (Array or
CuArray); mixing host and device arrays errors at broadcast time.
(Per-call scratch — Phase-6 may cache.)
"""
function gpu_velocity_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, tor_r, tor_i, pol_r, pol_i,
        config, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, lmax::Int, bw::Int;
        T_phys = nothing, thermal_factor = zero(eltype(tor_r)), r_vec = nothing,
        C_phys = nothing, comp_factor = zero(eltype(tor_r)),
        J_r = nothing, J_θ = nothing, J_φ = nothing,
        B_r = nothing, B_θ = nothing, B_φ = nothing, lorentz_coeff = zero(eltype(tor_r)))
    # lmax kept for interface symmetry with the other field orchestrators; the spectral
    # bounds are encoded in lfac/rinv/config, so it isn't forwarded to the sub-calls here.
    arch = arch_of(tor_r)
    sz = size(tor_r); nr = sz[3]
    spec(a, b) = GPUSpectralField{eltype(a), typeof(a)}(config, sz[1], sz[2], nr, a, b)
    ph() = allocate_gpu_physical_field(eltype(tor_r), arch, config, nr)
    curl_r = r_vec === nothing ? inv.(rinv) : r_vec
    # 1. velocity (tor,pol) → physical (u_r,u_θ,u_φ)
    ur = ph(); uθ = ph(); uφ = ph()
    gpu_vector_spectral_to_physical!(ur, uθ, uφ, spec(tor_r, tor_i), spec(pol_r, pol_i), config,
        lfac, rscale, d1, rinv, bw)
    # 2. vorticity ω = ∇×u (spectral)
    wtr = similar(tor_r); wti = similar(tor_i); wpr = similar(pol_r); wpi = similar(pol_i)
    gpu_spectral_curl!(wtr, wti, wpr, wpi, tor_r, tor_i, pol_r, pol_i, d1, d2, lfac, rinv, rinv2, curl_r, bw)
    # 3. vorticity → physical (ω_r,ω_θ,ω_φ)
    wr = ph(); wθ = ph(); wφ = ph()
    gpu_vector_spectral_to_physical!(wr, wθ, wφ, spec(wtr, wti), spec(wpr, wpi), config,
        lfac, rscale, d1, rinv, bw)
    # 4. adv = E·(u×ω) − ẑ×u  (physical)
    ar = ph(); aθ = ph(); aφ = ph()
    gpu_cross!(ar.data, aθ.data, aφ.data, ur.data, uθ.data, uφ.data, wr.data, wθ.data, wφ.data, E)
    gpu_coriolis_sub!(ar.data, aθ.data, aφ.data, ur.data, uθ.data, uφ.data, sinθ, cosθ)
    # 4b. coupled forcing accumulated into adv (CPU order: thermal → compositional → Lorentz)
    if T_phys !== nothing && r_vec !== nothing
        gpu_buoyancy_add!(ar.data, T_phys, r_vec, thermal_factor)     # adv_r += thermal_factor·r·T
    end
    if C_phys !== nothing && r_vec !== nothing
        gpu_buoyancy_add!(ar.data, C_phys, r_vec, comp_factor)        # adv_r += comp_factor·r·C
    end
    # J_r !== nothing implies all six J_*/B_* components are supplied (all-or-nothing convention:
    # first component is the proxy guard, matching the codebase's first-component-as-proxy pattern).
    if J_r !== nothing
        gpu_cross_add!(ar.data, aθ.data, aφ.data, J_r, J_θ, J_φ, B_r, B_θ, B_φ, lorentz_coeff)  # adv += lorentz_coeff·(J×B)
    end
    # 5. analyze the tangential force → (nl_pol = S, nl_tor = T); adv_r discarded
    # TODO(Task 4): Stage-4B projection (N_W = ∂r(r·S_F) − Q_F) replaces this — raw
    # mode keeps the legacy shape compiling, results are WRONG until then.
    gpu_vector_physical_to_spectral!(spec(nl_tor_r, nl_tor_i), spec(nl_pol_r, nl_pol_i), aθ, aφ, config;
        raw_spheroidal = true)
    return nothing
end
