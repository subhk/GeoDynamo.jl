# =============================================================================
# GPU Phase 5h — magnetic field nonlinear term (induction nl = ∇×(u×B)).
# Mirrors the CPU Stage-4 induction projection: E = u×B is not solenoidal, so
# raw tangential analysis gives (S_E,T_E), scalar analysis of E_r gives Q_E, and
# the curl potentials are T = -(Q_E - ∂r(r*S_E))/r, P = -r*T_E.
# (Per-call scratch — Phase-6 may cache. Inner-core rotation coupling deferred.)
# =============================================================================

function gpu_induction_curl_potentials!(
        nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i,
        S_r, S_i, T_r, T_i, d1, rinv, bw::Int; ws = nothing, tag::Symbol = :icp)
    ri = reshape(rinv, 1, 1, :)
    rS_r = gpu_scratch!(ws, Symbol(tag, :_rsr), S_r)
    rS_i = gpu_scratch!(ws, Symbol(tag, :_rsi), S_i)
    @. rS_r = S_r / ri
    @. rS_i = S_i / ri
    drS_r = gpu_scratch!(ws, Symbol(tag, :_dsr), S_r)
    drS_i = gpu_scratch!(ws, Symbol(tag, :_dsi), S_i)
    gpu_batched_banded_matvec!(drS_r, rS_r, d1, bw)
    gpu_batched_banded_matvec!(drS_i, rS_i, d1, bw)
    @. nl_tor_r = -(nl_tor_r - drS_r) * ri
    @. nl_tor_i = -(nl_tor_i - drS_i) * ri
    @. nl_pol_r = -T_r / ri
    @. nl_pol_i = -T_i / ri
    return nothing
end

"""
    gpu_magnetic_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, B_tor_r, B_tor_i, B_pol_r, B_pol_i,
                            u_r, u_θ, u_φ, config, d1, d2, lfac, rinv, rinv2, rscale, lmax, bw) -> nothing

Magnetic induction nonlinear `nl = ∇×(u×B)`.  `B_tor`/`B_pol` the magnetic
toroidal/poloidal spectral; `u_*` the physical velocity (supplied); `nl_tor`/`nl_pol`
the toroidal/poloidal induction nonlinear.  `d1`/`d2`/`lfac`/`rinv`/`rinv2`
are the curl/transform operators; `rscale` is retained in the operator bundle for
interface compatibility.  All on the same backend; outputs distinct from inputs.
"""
function gpu_magnetic_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, B_tor_r, B_tor_i, B_pol_r, B_pol_i,
        u_r, u_θ, u_φ, config, d1, d2, lfac, rinv, rinv2, rscale, lmax::Int, bw::Int;
        ws = nothing, tag::Symbol = :mnl)
    # lmax kept for interface symmetry with the other field orchestrators; spectral
    # bounds are encoded in lfac/rinv/config, so it isn't forwarded to the sub-calls here.
    arch = arch_of(B_tor_r)
    sz = size(B_tor_r); nr = sz[3]
    T = eltype(B_tor_r)
    spec(a, b) = GPUSpectralField{eltype(a), typeof(a)}(config, sz[1], sz[2], nr, a, b)
    ph(k::Symbol) = gpu_scratch_phys!(ws, Symbol(tag, k), T, arch, config, nr)
    # 1. B (tor,pol) → physical (B_r,B_θ,B_φ)
    Br = ph(:_Br); Bθ = ph(:_Bt); Bφ = ph(:_Bp)
    gpu_vector_spectral_to_physical!(
        Br, Bθ, Bφ, spec(B_tor_r, B_tor_i), spec(B_pol_r, B_pol_i),
        config, d1, lfac, rinv, rinv2, bw; ws, tag = Symbol(tag, :_Bs))
    # 2. uB = u×B (physical), coeff 1
    ubr = ph(:_er); ubθ = ph(:_et); ubφ = ph(:_ep)
    gpu_cross!(ubr.data, ubθ.data, ubφ.data, u_r, u_θ, u_φ, Br.data, Bθ.data, Bφ.data, one(T))
    # 3. uB → raw QST scalars: work_tor = T_E, work_pol = S_E, nl_tor = Q_E.
    wtr = gpu_scratch!(ws, Symbol(tag, :_wtr), B_tor_r)
    wti = gpu_scratch!(ws, Symbol(tag, :_wti), B_tor_i)
    wpr = gpu_scratch!(ws, Symbol(tag, :_wpr), B_pol_r)
    wpi = gpu_scratch!(ws, Symbol(tag, :_wpi), B_pol_i)
    gpu_vector_physical_to_spectral!(spec(wtr, wti), spec(wpr, wpi), ubθ, ubφ, config;
        ws, tag = Symbol(tag, :_ea))
    gpu_scalar_physical_to_spectral!(spec(nl_tor_r, nl_tor_i), ubr, config;
        ws, tag = Symbol(tag, :_qa))
    # 4. QST curl projection → induction toroidal/poloidal nonlinear.
    gpu_induction_curl_potentials!(
        nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, wpr, wpi, wtr, wti, d1, rinv, bw;
        ws, tag = Symbol(tag, :_icp))
    return nothing
end
