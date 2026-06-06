using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5m2 — Magnetic Conducting Inner Core Step" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 4)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 4
    Nic = 5
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    rng = MersenneTwister(17)
    dt = 5.0e-4; theta = 0.5

    function band(N, b; seed)
        r = MersenneTwister(seed); d = zeros(2b+1, N)
        for j in 1:N, i in max(1,j-b):min(N,j+b); d[b+1+i-j, j] = rand(r) - 0.5; end
        d
    end
    d1 = band(nr, bw; seed = 1); d2 = band(nr, bw; seed = 2)
    lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
    rinv = [1.0/(0.5 + 0.1k) for k in 1:nr]; rinv2 = rinv .^ 2; rscale = copy(rinv)

    function batched(N, seed)
        a = zeros(2bw+1, N, nl); r = MersenneTwister(seed)
        for li in 1:nl, j in 1:N, i in max(1,j-bw):min(N,j+bw); a[bw+1+i-j, j, li] = rand(r) - 0.5; end
        for li in 1:nl, j in 1:N; a[bw+1, j, li] += 5.0; end
        a
    end
    lin_tor = batched(nr, 10); lu_tor = batched(nr, 11); lin_pol = batched(nr, 20); lu_pol = batched(nr, 21)

    # synthetic InnerCoreAdmittance for tor + pol (degrees 1..lmax)
    function make_adm(seed0)
        facs = GeoDynamo.BandedLU{Float64}[]; lins = GeoDynamo.BandedMatrix{Float64}[]
        lookup = Dict{Int,Int}()
        for (idx, l) in enumerate(1:cfg.lmax)
            M = GeoDynamo.BandedMatrix{Float64}(
                (d = band(Nic, bw; seed = seed0 + l); for j in 1:Nic; d[bw+1, j] += 5.0; end; d), bw, Nic)
            push!(facs, GeoDynamo.factorize_banded(M))
            push!(lins, GeoDynamo.BandedMatrix{Float64}(band(Nic, bw; seed = seed0 + 100 + l), bw, Nic))
            lookup[l] = idx
        end
        GeoDynamo.InnerCoreAdmittance{Float64}(facs, rand(rng, cfg.lmax), rand(rng, Nic) .- 0.5,
                                               lookup, Nic, lins, dt, theta)
    end
    adm_tor = make_adm(300); adm_pol = make_adm(400)

    u_r = rand(rng, nlat, nlon, nr) .- 0.5; u_θ = rand(rng, nlat, nlon, nr) .- 0.5; u_φ = rand(rng, nlat, nlon, nr) .- 0.5
    inv_dt = 1.0 / dt; linear_weight = 1.0 - theta

    mk(N) = (a = zeros(nl, nm, N); for mi in 1:nm, li in mi:nl, r in 1:N; a[li,mi,r] = rand(rng) - 0.5; end; a)
    bt_r0 = mk(nr); bt_i0 = mk(nr); bp_r0 = mk(nr); bp_i0 = mk(nr)
    pnt_r0 = mk(nr); pnt_i0 = mk(nr); pnp_r0 = mk(nr); pnp_i0 = mk(nr)
    itr0 = mk(Nic); iti0 = mk(Nic); ipr0 = mk(Nic); ipi0 = mk(Nic)   # inner-core tor/pol state

    nlops = (; d1, d2, lfac, rinv, rinv2, rscale)

    @testset "conducting step == manual chain (exact) [LOCAL]" begin
        tor_adm = GeoDynamo.gpu_pack_inner_core(adm_tor, nl, CPU())
        pol_adm = GeoDynamo.gpu_pack_inner_core(adm_pol, nl, CPU())
        tor = (; spec_r = copy(bt_r0), spec_i = copy(bt_i0), prev_nl_r = copy(pnt_r0), prev_nl_i = copy(pnt_i0),
                 lin = lin_tor, lu = lu_tor)
        pol = (; spec_r = copy(bp_r0), spec_i = copy(bp_i0), prev_nl_r = copy(pnp_r0), prev_nl_i = copy(pnp_i0),
                 lin = lin_pol, lu = lu_pol)
        ic = (; tor_adm = tor_adm, pol_adm = pol_adm,
                tor_ic_r = copy(itr0), tor_ic_i = copy(iti0), pol_ic_r = copy(ipr0), pol_ic_i = copy(ipi0))
        GeoDynamo.gpu_magnetic_field_step!(tor, pol, copy(u_r), copy(u_θ), copy(u_φ), cfg, nlops,
                                           inv_dt, linear_weight, cfg.lmax, bw; ic = ic)

        # ---- manual chain ----
        bt_r = copy(bt_r0); bt_i = copy(bt_i0); bp_r = copy(bp_r0); bp_i = copy(bp_i0)
        pnt_r = copy(pnt_r0); pnt_i = copy(pnt_i0); pnp_r = copy(pnp_r0); pnp_i = copy(pnp_i0)
        nlt_r = similar(bt_r); nlt_i = similar(bt_i); nlp_r = similar(bp_r); nlp_i = similar(bp_i)
        GeoDynamo.gpu_magnetic_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i, bt_r, bt_i, bp_r, bp_i,
            copy(u_r), copy(u_θ), copy(u_φ), cfg, d1, d2, lfac, rinv, rinv2, rscale, cfg.lmax, bw)
        φt_r = zeros(nl, nm); φt_i = zeros(nl, nm); φp_r = zeros(nl, nm); φp_i = zeros(nl, nm)
        GeoDynamo.gpu_inner_core_history_flux!(φt_r, φt_i, copy(itr0), copy(iti0), tor_adm)
        GeoDynamo.gpu_inner_core_history_flux!(φp_r, φp_i, copy(ipr0), copy(ipi0), pol_adm)
        z = zeros(nl, nm)
        rt_r = similar(bt_r); rt_i = similar(bt_i); rp_r = similar(bp_r); rp_i = similar(bp_i)
        GeoDynamo.gpu_build_rhs_cnab2!(rt_r, rt_i, bt_r, bt_i, nlt_r, nlt_i, pnt_r, pnt_i, lin_tor, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rt_r, rt_i, lu_tor, φt_r, φt_i, z, z, bw)
        GeoDynamo.gpu_build_rhs_cnab2!(rp_r, rp_i, bp_r, bp_i, nlp_r, nlp_i, pnp_r, pnp_i, lin_pol, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rp_r, rp_i, lu_pol, φp_r, φp_i, z, z, bw)
        # reconstruct from the NEW outer-core ICB value (g = rt[:,:,1] / rp[:,:,1])
        ntr_r = similar(itr0); ntr_i = similar(iti0); npr_r = similar(ipr0); npr_i = similar(ipi0)
        GeoDynamo.gpu_reconstruct_inner_core!(ntr_r, ntr_i, copy(itr0), copy(iti0), rt_r[:, :, 1], rt_i[:, :, 1], tor_adm)
        GeoDynamo.gpu_reconstruct_inner_core!(npr_r, npr_i, copy(ipr0), copy(ipi0), rp_r[:, :, 1], rp_i[:, :, 1], pol_adm)

        @test tor.spec_r == rt_r && tor.spec_i == rt_i
        @test pol.spec_r == rp_r && pol.spec_i == rp_i
        @test tor.prev_nl_r == nlt_r && tor.prev_nl_i == nlt_i
        @test pol.prev_nl_r == nlp_r && pol.prev_nl_i == nlp_i
        @test ic.tor_ic_r == ntr_r && ic.tor_ic_i == ntr_i
        @test ic.pol_ic_r == npr_r && ic.pol_ic_i == npr_i
        @test all(isfinite, ic.tor_ic_r) && all(isfinite, ic.pol_ic_r)
    end

    @testset "insulating path unchanged when ic=nothing [LOCAL]" begin
        # ic=nothing must reproduce the Phase-5m insulating result exactly
        tor = (; spec_r = copy(bt_r0), spec_i = copy(bt_i0), prev_nl_r = copy(pnt_r0), prev_nl_i = copy(pnt_i0),
                 lin = lin_tor, lu = lu_tor)
        pol = (; spec_r = copy(bp_r0), spec_i = copy(bp_i0), prev_nl_r = copy(pnp_r0), prev_nl_i = copy(pnp_i0),
                 lin = lin_pol, lu = lu_pol)
        GeoDynamo.gpu_magnetic_field_step!(tor, pol, copy(u_r), copy(u_θ), copy(u_φ), cfg, nlops,
                                           inv_dt, linear_weight, cfg.lmax, bw)   # no ic, no continuity
        # manual insulating (no continuity, all-zero BC)
        bt_r = copy(bt_r0); bt_i = copy(bt_i0); bp_r = copy(bp_r0); bp_i = copy(bp_i0)
        nlt_r = similar(bt_r); nlt_i = similar(bt_i); nlp_r = similar(bp_r); nlp_i = similar(bp_i)
        GeoDynamo.gpu_magnetic_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i, bt_r, bt_i, bp_r, bp_i,
            copy(u_r), copy(u_θ), copy(u_φ), cfg, d1, d2, lfac, rinv, rinv2, rscale, cfg.lmax, bw)
        z = zeros(nl, nm)
        rt_r = similar(bt_r); rt_i = similar(bt_i); rp_r = similar(bp_r); rp_i = similar(bp_i)
        GeoDynamo.gpu_build_rhs_cnab2!(rt_r, rt_i, bt_r, bt_i, nlt_r, nlt_i, copy(pnt_r0), copy(pnt_i0), lin_tor, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rt_r, rt_i, lu_tor, z, z, z, z, bw)
        GeoDynamo.gpu_build_rhs_cnab2!(rp_r, rp_i, bp_r, bp_i, nlp_r, nlp_i, copy(pnp_r0), copy(pnp_i0), lin_pol, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rp_r, rp_i, lu_pol, z, z, z, z, bw)
        @test tor.spec_r == rt_r && pol.spec_r == rp_r
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5m2 gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            # CPU reference (conducting)
            ctor_adm = GeoDynamo.gpu_pack_inner_core(adm_tor, nl, CPU())
            cpol_adm = GeoDynamo.gpu_pack_inner_core(adm_pol, nl, CPU())
            ctor = (; spec_r = copy(bt_r0), spec_i = copy(bt_i0), prev_nl_r = copy(pnt_r0), prev_nl_i = copy(pnt_i0),
                      lin = lin_tor, lu = lu_tor)
            cpol = (; spec_r = copy(bp_r0), spec_i = copy(bp_i0), prev_nl_r = copy(pnp_r0), prev_nl_i = copy(pnp_i0),
                      lin = lin_pol, lu = lu_pol)
            cic = (; tor_adm = ctor_adm, pol_adm = cpol_adm,
                     tor_ic_r = copy(itr0), tor_ic_i = copy(iti0), pol_ic_r = copy(ipr0), pol_ic_i = copy(ipi0))
            GeoDynamo.gpu_magnetic_field_step!(ctor, cpol, copy(u_r), copy(u_θ), copy(u_φ), cfg, nlops,
                                               inv_dt, linear_weight, cfg.lmax, bw; ic = cic)
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            gtor_adm = GeoDynamo.gpu_pack_inner_core(adm_tor, nl, GPU())
            gpol_adm = GeoDynamo.gpu_pack_inner_core(adm_pol, nl, GPU())
            gnlops = (; d1 = d(d1), d2 = d(d2), lfac = d(lfac), rinv = d(rinv), rinv2 = d(rinv2), rscale = d(rscale))
            gtor = (; spec_r = d(copy(bt_r0)), spec_i = d(copy(bt_i0)), prev_nl_r = d(copy(pnt_r0)), prev_nl_i = d(copy(pnt_i0)),
                      lin = d(lin_tor), lu = d(lu_tor))
            gpol = (; spec_r = d(copy(bp_r0)), spec_i = d(copy(bp_i0)), prev_nl_r = d(copy(pnp_r0)), prev_nl_i = d(copy(pnp_i0)),
                      lin = d(lin_pol), lu = d(lu_pol))
            gic = (; tor_adm = gtor_adm, pol_adm = gpol_adm,
                     tor_ic_r = d(copy(itr0)), tor_ic_i = d(copy(iti0)), pol_ic_r = d(copy(ipr0)), pol_ic_i = d(copy(ipi0)))
            GeoDynamo.gpu_magnetic_field_step!(gtor, gpol, d(copy(u_r)), d(copy(u_θ)), d(copy(u_φ)), cfg, gnlops,
                                               inv_dt, linear_weight, cfg.lmax, bw; ic = gic)
            @test gtor.spec_r isa CUDA.CuArray
            @test gic.tor_ic_r isa CUDA.CuArray
            @test isapprox(Array(gtor.spec_r), ctor.spec_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.spec_i), cpol.spec_i; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gic.tor_ic_r), cic.tor_ic_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gic.pol_ic_i), cic.pol_ic_i; atol = 1e-9, rtol = 1e-8)
        end
    end
end
