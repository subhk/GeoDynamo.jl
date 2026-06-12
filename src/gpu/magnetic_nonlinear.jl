# =============================================================================
# GPU Phase 5h — magnetic field nonlinear term (induction nl = ∇×(u×B)),
# composing: vector transform (3, B→physical) → cross u×B (2) → vector analyze
# (3, u×B → work_tor/work_pol) → spectral curl (5a, curl(work) → nl).  Mirrors
# apply_induction_nonlinear! (numerics.jl:1491-1603).  The extra curl (vs the
# velocity force projection) is the ∇× of the induction equation.  u physical is
# supplied (from the velocity nonlinear).  Runs on Array + CuArray.
# (Per-call scratch — Phase-6 may cache. Inner-core rotation coupling deferred.)
# =============================================================================

"""
    gpu_magnetic_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, B_tor_r, B_tor_i, B_pol_r, B_pol_i,
                            u_r, u_θ, u_φ, config, d1, d2, lfac, rinv, rinv2, rscale, lmax, bw;
                            r_vec=nothing) -> nothing

Magnetic induction nonlinear `nl = ∇×(u×B)`.  `B_tor`/`B_pol` the magnetic
toroidal/poloidal spectral; `u_*` the physical velocity (supplied); `nl_tor`/`nl_pol`
the toroidal/poloidal induction nonlinear.  `d1`/`d2`/`lfac`/`rinv`/`rinv2`/`rscale`
as in the curl/transform.  All on the same backend; outputs distinct from inputs.
"""
function gpu_magnetic_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, B_tor_r, B_tor_i, B_pol_r, B_pol_i,
        u_r, u_θ, u_φ, config, d1, d2, lfac, rinv, rinv2, rscale, lmax::Int, bw::Int;
        r_vec = nothing)
    # lmax kept for interface symmetry with the other field orchestrators; spectral
    # bounds are encoded in lfac/rinv/config, so it isn't forwarded to the sub-calls here.
    arch = arch_of(B_tor_r)
    sz = size(B_tor_r); nr = sz[3]
    spec(a, b) = GPUSpectralField{eltype(a), typeof(a)}(config, sz[1], sz[2], nr, a, b)
    ph() = allocate_gpu_physical_field(eltype(B_tor_r), arch, config, nr)
    curl_r = r_vec === nothing ? inv.(rinv) : r_vec
    # 1. B (tor,pol) → physical (B_r,B_θ,B_φ)
    Br = ph(); Bθ = ph(); Bφ = ph()
    gpu_vector_spectral_to_physical!(Br, Bθ, Bφ, spec(B_tor_r, B_tor_i), spec(B_pol_r, B_pol_i), config,
        lfac, rscale, d1, rinv, bw)
    # 2. uB = u×B (physical), coeff 1
    ubr = ph(); ubθ = ph(); ubφ = ph()
    gpu_cross!(ubr.data, ubθ.data, ubφ.data, u_r, u_θ, u_φ, Br.data, Bθ.data, Bφ.data, one(eltype(B_tor_r)))
    # 3. uB → spectral (work_tor = T, work_pol = S), tangential analyze
    # TODO(Task 5): Stage-4A curl potentials (P=−r·T_E, T=−(Q_E−∂r(r·S_E))/r) replace
    # this — raw mode keeps the legacy shape compiling, results are WRONG until then.
    wtr = similar(B_tor_r); wti = similar(B_tor_i); wpr = similar(B_pol_r); wpi = similar(B_pol_i)
    gpu_vector_physical_to_spectral!(spec(wtr, wti), spec(wpr, wpi), ubθ, ubφ, config;
        raw_spheroidal = true)
    # 4. curl(work) → nl  (∇× of the induction)
    gpu_spectral_curl!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, wtr, wti, wpr, wpi, d1, d2, lfac, rinv, rinv2, curl_r, bw)
    return nothing
end
