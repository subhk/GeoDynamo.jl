using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5m — Magnetic Field CNAB2 Step (insulating, tor + pol)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 4)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 4
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    rng = MersenneTwister(13)

    function band(N, b; seed)
        r = MersenneTwister(seed); d = zeros(2b+1, N)
        for j in 1:N, i in max(1,j-b):min(N,j+b); d[b+1+i-j, j] = rand(r) - 0.5; end
        d
    end
    d1 = band(nr, bw; seed = 1); d2 = band(nr, bw; seed = 2)
    lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
    rinv = [1.0/(0.5 + 0.1k) for k in 1:nr]; rinv2 = rinv .^ 2; rscale = copy(rinv)

    function batched(seed)
        a = zeros(2bw+1, nr, nl); r = MersenneTwister(seed)
        for li in 1:nl, j in 1:nr, i in max(1,j-bw):min(nr,j+bw); a[bw+1+i-j, j, li] = rand(r) - 0.5; end
        for li in 1:nl, j in 1:nr; a[bw+1, j, li] += 5.0; end
        a
    end
    lin_tor = batched(10); lu_tor = batched(11); lin_pol = batched(20); lu_pol = batched(21)

    u_r = rand(rng, nlat, nlon, nr) .- 0.5
    u_θ = rand(rng, nlat, nlon, nr) .- 0.5
    u_φ = rand(rng, nlat, nlon, nr) .- 0.5

    inv_dt = 1.0 / 5.0e-4          # mass_coeff(1) / dt
    linear_weight = 0.5

    mk() = (a = zeros(nl, nm, nr); for mi in 1:nm, li in mi:nl, r in 1:nr; a[li,mi,r] = rand(rng) - 0.5; end; a)
    bt_r0 = mk(); bt_i0 = mk(); bp_r0 = mk(); bp_i0 = mk()
    pnt_r0 = mk(); pnt_i0 = mk(); pnp_r0 = mk(); pnp_i0 = mk()

    nlops = (; d1, d2, lfac, rinv, rinv2, rscale)

    function run_gpu(arch_dev, continuity)
        d = arch_dev === :cpu ? identity : (x -> GeoDynamo.on_architecture(GPU(), x))
        tor = (; spec_r = d(copy(bt_r0)), spec_i = d(copy(bt_i0)),
                 prev_nl_r = d(copy(pnt_r0)), prev_nl_i = d(copy(pnt_i0)), lin = d(lin_tor), lu = d(lu_tor))
        pol = (; spec_r = d(copy(bp_r0)), spec_i = d(copy(bp_i0)),
                 prev_nl_r = d(copy(pnp_r0)), prev_nl_i = d(copy(pnp_i0)), lin = d(lin_pol), lu = d(lu_pol))
        nlo = arch_dev === :cpu ? nlops :
            (; d1 = d(d1), d2 = d(d2), lfac = d(lfac), rinv = d(rinv), rinv2 = d(rinv2), rscale = d(rscale))
        GeoDynamo.gpu_magnetic_field_step!(tor, pol, d(copy(u_r)), d(copy(u_θ)), d(copy(u_φ)),
            cfg, nlo, inv_dt, linear_weight, cfg.lmax, bw; continuity_mag = continuity)
        return tor, pol
    end

    # manual chain on CPU for the given continuity flag
    function manual(continuity)
        bt_r = copy(bt_r0); bt_i = copy(bt_i0); bp_r = copy(bp_r0); bp_i = copy(bp_i0)
        pnt_r = copy(pnt_r0); pnt_i = copy(pnt_i0); pnp_r = copy(pnp_r0); pnp_i = copy(pnp_i0)
        nlt_r = similar(bt_r); nlt_i = similar(bt_i); nlp_r = similar(bp_r); nlp_i = similar(bp_i)
        GeoDynamo.gpu_magnetic_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i, bt_r, bt_i, bp_r, bp_i,
            copy(u_r), copy(u_θ), copy(u_φ), cfg, d1, d2, lfac, rinv, rinv2, rscale, cfg.lmax, bw)
        bcin_r = zeros(nl, nm); bcin_i = zeros(nl, nm); z = zeros(nl, nm)
        if continuity
            bcin_r .= .-nlp_r[:, :, 1] .+ pnp_r[:, :, 1]
            bcin_i .= .-nlp_i[:, :, 1] .+ pnp_i[:, :, 1]
        end
        rt_r = similar(bt_r); rt_i = similar(bt_i); rp_r = similar(bp_r); rp_i = similar(bp_i)
        GeoDynamo.gpu_build_rhs_cnab2!(rt_r, rt_i, bt_r, bt_i, nlt_r, nlt_i, pnt_r, pnt_i, lin_tor, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rt_r, rt_i, lu_tor, bcin_r, bcin_i, z, z, bw)
        GeoDynamo.gpu_build_rhs_cnab2!(rp_r, rp_i, bp_r, bp_i, nlp_r, nlp_i, pnp_r, pnp_i, lin_pol, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rp_r, rp_i, lu_pol, z, z, z, z, bw)
        return (rt_r, rt_i, rp_r, rp_i, nlt_r, nlt_i, nlp_r, nlp_i)
    end

    @testset "step == manual chain, no continuity (exact) [LOCAL]" begin
        tor, pol = run_gpu(:cpu, false)
        rt_r, rt_i, rp_r, rp_i, nlt_r, nlt_i, nlp_r, nlp_i = manual(false)
        @test tor.spec_r == rt_r && tor.spec_i == rt_i
        @test pol.spec_r == rp_r && pol.spec_i == rp_i
        @test tor.prev_nl_r == nlt_r && tor.prev_nl_i == nlt_i
        @test pol.prev_nl_r == nlp_r && pol.prev_nl_i == nlp_i
        @test all(isfinite, tor.spec_r) && all(isfinite, pol.spec_r)
    end

    @testset "step == manual chain, CONTINUITY_MAG (exact) [LOCAL]" begin
        tor, pol = run_gpu(:cpu, true)
        rt_r, rt_i, rp_r, rp_i, nlt_r, nlt_i, nlp_r, nlp_i = manual(true)
        @test tor.spec_r == rt_r && tor.spec_i == rt_i
        @test pol.spec_r == rp_r && pol.spec_i == rp_i
        @test tor.prev_nl_r == nlt_r && pol.prev_nl_r == nlp_r
    end

    @testset "continuity changes the toroidal result [LOCAL]" begin
        # sanity: the CONTINUITY_MAG BC actually does something (toroidal differs)
        t0, _ = run_gpu(:cpu, false)
        t1, _ = run_gpu(:cpu, true)
        @test t0.spec_r != t1.spec_r
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5m gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            ctor, cpol = run_gpu(:cpu, true)
            gtor, gpol = run_gpu(:gpu, true)
            @test gtor.spec_r isa CUDA.CuArray
            @test gpol.spec_r isa CUDA.CuArray
            @test isapprox(Array(gtor.spec_r), ctor.spec_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gtor.spec_i), ctor.spec_i; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.spec_r), cpol.spec_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.spec_i), cpol.spec_i; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gtor.prev_nl_r), ctor.prev_nl_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.prev_nl_i), cpol.prev_nl_i; atol = 1e-9, rtol = 1e-8)
        end
    end
end
