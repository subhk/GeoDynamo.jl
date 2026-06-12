using Test
using GeoDynamo
using MPI
using Random
using LinearAlgebra

MPI.Initialized() || MPI.Init()

# =============================================================================
# Current-density curl convention (Stage-2 stored potentials)
#
# The code stores solenoidal fields as (T, P) with synthesis Q = l(l+1)P/r²,
# S = (∂r P)/r. Under this convention the curl of a (T, P) field has stored
# potentials
#     T_curl = (P″ − l(l+1)P/r²)/r = D_pol(P)/r,      P_curl = −r·T
# — the SAME rule the vorticity (compute_vorticity_spectral!) and the
# induction projections (_induction_curl_potentials!) use. The decisive
# consistency check is curl∘curl = −Δ: applying the rule twice must reproduce
# the code's own magnetic diffusion operators, −(Δ_l T, D_pol P). The legacy
# current formula (λP/r² − P″ − 2P′/r; −λT/r²) fails that identity.
# =============================================================================

@testset "current-density curl convention" begin
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 6, mmax = 6, nlat = 14, nlon = 28, nr = 8, nr_inner = 4,
        radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e5, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
        include_magnetic = true, include_composition = false)
    st = GeoDynamo.initialize_solver_state(Float64; params = params)
    mag = st.fields.magnetic
    cfg = st.backend.shtns_config
    dom = st.backend.outer_core_domain
    nr = dom.N
    nl = cfg.lmax + 1
    nm = cfg.mmax + 1

    rng = MersenneTwister(11)
    for f in (mag.toroidal, mag.poloidal)
        parent(f.data_real) .+= rand(rng, size(parent(f.data_real))...) .- 0.5
        parent(f.data_imag) .+= rand(rng, size(parent(f.data_imag))...) .- 0.5
    end

    T_r, T_i = GeoDynamo.cpu_spectral_to_dense(mag.toroidal, cfg, nr, Float64)
    P_r, P_i = GeoDynamo.cpu_spectral_to_dense(mag.poloidal, cfg, nr, Float64)

    # dense reference: vorticity-rule curl per (l, m) radial profile
    D2 = GeoDynamo.create_derivative_matrix(Float64, 2, dom)
    rinv = [dom.r[k, 3] for k in 1:nr]
    rval = [dom.r[k, 4] for k in 1:nr]
    function curl_dense(Td, Pd)
        jt = zeros(nl, nm, nr); jp = zeros(nl, nm, nr)
        prof = zeros(nr); d2p = zeros(nr)
        for mi in 1:nm, li in mi:nl
            l = li - 1
            λ = l * (l + 1)
            prof .= Pd[li, mi, :]
            mul!(d2p, D2, prof)
            for k in 1:nr
                jt[li, mi, k] = (d2p[k] - λ * rinv[k]^2 * prof[k]) * rinv[k]
                jp[li, mi, k] = -rval[k] * Td[li, mi, k]
            end
        end
        return jt, jp
    end
    expT_r, expP_r = curl_dense(T_r, P_r)
    expT_i, expP_i = curl_dense(T_i, P_i)

    @testset "J = ∇×B matches the Stage-2 (vorticity-rule) potentials" begin
        GeoDynamo.solver_compute_current_density_spectral!(mag, dom)
        jt_r, jt_i = GeoDynamo.cpu_spectral_to_dense(mag.work_tor, cfg, nr, Float64)
        jp_r, jp_i = GeoDynamo.cpu_spectral_to_dense(mag.work_pol, cfg, nr, Float64)
        @test isapprox(jt_r, expT_r; atol = 1e-12, rtol = 1e-10)
        @test isapprox(jt_i, expT_i; atol = 1e-12, rtol = 1e-10)
        @test isapprox(jp_r, expP_r; atol = 1e-12, rtol = 1e-10)
        @test isapprox(jp_i, expP_i; atol = 1e-12, rtol = 1e-10)
    end

    @testset "curl∘curl = −(Δ_l T, D_pol P) through the code path" begin
        # write the first curl's potentials back in as a field, curl again
        GeoDynamo.dense_to_cpu_spectral!(mag.toroidal, expT_r, expT_i, cfg, nr)
        GeoDynamo.dense_to_cpu_spectral!(mag.poloidal, expP_r, expP_i, cfg, nr)
        GeoDynamo.solver_compute_current_density_spectral!(mag, dom)
        jjt_r, _ = GeoDynamo.cpu_spectral_to_dense(mag.work_tor, cfg, nr, Float64)
        jjp_r, _ = GeoDynamo.cpu_spectral_to_dense(mag.work_pol, cfg, nr, Float64)

        # −Δ on the ORIGINAL field, in stored potentials. The POLOIDAL branch of
        # curl∘curl is DISCRETE-exact: P_jj = −r·T_j = −(D2·P − λP/r²) = −D_pol P
        # with the very same banded operator. (The toroidal branch −Δ_l T is exact
        # only analytically — the banded product rule D2(rT) ≠ r·D2T + 2·D1T at
        # finite bandwidth/boundary rows — so it is not asserted here; the
        # formula itself is already pinned by the first testset.)
        dpolP = zeros(nl, nm, nr)
        prof = zeros(nr); d2p = zeros(nr)
        for mi in 1:nm, li in mi:nl
            l = li - 1
            λ = l * (l + 1)
            prof .= P_r[li, mi, :]
            mul!(d2p, D2, prof)
            for k in 1:nr
                dpolP[li, mi, k] = d2p[k] - λ * rinv[k]^2 * prof[k]
            end
        end
        @test isapprox(jjp_r, -dpolP; atol = 1e-10, rtol = 1e-8)
        @test all(isfinite, jjt_r)
    end
end
