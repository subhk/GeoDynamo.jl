using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5k — Velocity Field CNAB2 Step (tor + pol)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 4)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 4
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    rng = MersenneTwister(11)

    # banded operators (2bw+1, nr) shared by the nonlinear curls
    function band(N, b; seed)
        rng2 = MersenneTwister(seed); d = zeros(2b+1, N)
        for j in 1:N, i in max(1,j-b):min(N,j+b); d[b+1+i-j,j] = rand(rng2) - 0.5; end
        d
    end
    d1 = band(nr, bw; seed = 1); d2 = band(nr, bw; seed = 2)
    lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
    rinv = [1.0/(0.5 + 0.1k) for k in 1:nr]; rinv2 = rinv .^ 2; rscale = copy(rinv)
    sinθ = sin.(range(0.1, π - 0.1; length = nlat)); cosθ = cos.(range(0.1, π - 0.1; length = nlat))
    E = 1.3e-3

    # per-l linear ops + LU for tor and pol — batched (2bw+1, nr, nl). (Wiring test:
    # same matrices feed GPU step and the manual chain, so exact == holds regardless
    # of conditioning. Make the diagonal dominant so the solve is well-posed.)
    function batched(seed)
        a = zeros(2bw+1, nr, nl); r = MersenneTwister(seed)
        for li in 1:nl, j in 1:nr, i in max(1,j-bw):min(nr,j+bw)
            a[bw+1+i-j, j, li] = rand(r) - 0.5
        end
        for li in 1:nl, j in 1:nr; a[bw+1, j, li] += 5.0; end   # diagonal dominance
        a
    end
    lin_tor = batched(10); lu_tor = batched(11)
    lin_pol = batched(20); lu_pol = batched(21)

    # BC vectors (nl, nm) — random toroidal (exercises BC propagation incl an l=1,m=0
    # rotation value), zero poloidal (homogeneous).
    bc_in_tor_r  = rand(rng, nl, nm) .- 0.5; bc_in_tor_i  = rand(rng, nl, nm) .- 0.5
    bc_out_tor_r = rand(rng, nl, nm) .- 0.5; bc_out_tor_i = rand(rng, nl, nm) .- 0.5
    bc_in_pol_r  = zeros(nl, nm); bc_in_pol_i  = zeros(nl, nm)
    bc_out_pol_r = zeros(nl, nm); bc_out_pol_i = zeros(nl, nm)

    # poloidal influence operators
    influence_dict = Dict{Int, GeoDynamo.ERK2InfluenceOp{Float64}}()
    for l in 1:cfg.lmax
        influence_dict[l] = GeoDynamo.ERK2InfluenceOp{Float64}(rand(rng, nr, 2) .- 0.5, rand(rng, 2, 2) .- 0.5, l)
    end
    Gre_b, invG_b = GeoDynamo.gpu_pack_influence(influence_dict, nl, nr, CPU())

    inv_dt = E / 5.0e-4          # mass_coeff(E) / dt
    linear_weight = 0.5          # 1 − θ

    # initial spectral state + history (random, upper-triangle modes only nonzero is
    # fine; the kernels handle empty modes as zeros)
    mk() = (a = zeros(nl, nm, nr); for mi in 1:nm, li in mi:nl, r in 1:nr; a[li,mi,r] = rand(rng) - 0.5; end; a)
    tor_r0 = mk(); tor_i0 = mk(); pol_r0 = mk(); pol_i0 = mk()
    pnt_r0 = mk(); pnt_i0 = mk(); pnp_r0 = mk(); pnp_i0 = mk()

    nlops = (; d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E)
    influence = (; Gre_b, invG_b)

    @testset "step == manual chain (exact) [LOCAL]" begin
        # ---- GPU step (on copies) ----
        tor = (; spec_r = copy(tor_r0), spec_i = copy(tor_i0),
                 prev_nl_r = copy(pnt_r0), prev_nl_i = copy(pnt_i0),
                 lin = lin_tor, lu = lu_tor,
                 bc_in_r = bc_in_tor_r, bc_in_i = bc_in_tor_i,
                 bc_out_r = bc_out_tor_r, bc_out_i = bc_out_tor_i)
        pol = (; spec_r = copy(pol_r0), spec_i = copy(pol_i0),
                 prev_nl_r = copy(pnp_r0), prev_nl_i = copy(pnp_i0),
                 lin = lin_pol, lu = lu_pol,
                 bc_in_r = bc_in_pol_r, bc_in_i = bc_in_pol_i,
                 bc_out_r = bc_out_pol_r, bc_out_i = bc_out_pol_i)
        GeoDynamo.gpu_velocity_field_step!(tor, pol, cfg, nlops, influence,
                                           inv_dt, linear_weight, cfg.lmax, bw)

        # ---- manual chain (same kernels, same order, on independent copies) ----
        mtr = copy(tor_r0); mti = copy(tor_i0); mpr = copy(pol_r0); mpi = copy(pol_i0)
        mpnt_r = copy(pnt_r0); mpnt_i = copy(pnt_i0); mpnp_r = copy(pnp_r0); mpnp_i = copy(pnp_i0)
        nlt_r = similar(mtr); nlt_i = similar(mti); nlp_r = similar(mpr); nlp_i = similar(mpi)
        GeoDynamo.gpu_velocity_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i, mtr, mti, mpr, mpi,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw)
        rt_r = similar(mtr); rt_i = similar(mti); rp_r = similar(mpr); rp_i = similar(mpi)
        GeoDynamo.gpu_build_rhs_cnab2!(rt_r, rt_i, mtr, mti, nlt_r, nlt_i, mpnt_r, mpnt_i,
            lin_tor, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rt_r, rt_i, lu_tor,
            bc_in_tor_r, bc_in_tor_i, bc_out_tor_r, bc_out_tor_i, bw)
        GeoDynamo.gpu_build_rhs_cnab2!(rp_r, rp_i, mpr, mpi, nlp_r, nlp_i, mpnp_r, mpnp_i,
            lin_pol, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rp_r, rp_i, lu_pol,
            bc_in_pol_r, bc_in_pol_i, bc_out_pol_r, bc_out_pol_i, bw)
        GeoDynamo.gpu_velocity_poloidal_influence_correction!(rp_r, rp_i, Gre_b, invG_b)

        @test tor.spec_r == rt_r
        @test tor.spec_i == rt_i
        @test pol.spec_r == rp_r
        @test pol.spec_i == rp_i
        @test tor.prev_nl_r == nlt_r
        @test tor.prev_nl_i == nlt_i
        @test pol.prev_nl_r == nlp_r
        @test pol.prev_nl_i == nlp_i
        @test all(isfinite, tor.spec_r) && all(isfinite, tor.spec_i) &&
              all(isfinite, pol.spec_r) && all(isfinite, pol.spec_i)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5k gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            # CPU reference
            ctor = (; spec_r = copy(tor_r0), spec_i = copy(tor_i0),
                      prev_nl_r = copy(pnt_r0), prev_nl_i = copy(pnt_i0),
                      lin = lin_tor, lu = lu_tor, bc_in_r = bc_in_tor_r, bc_in_i = bc_in_tor_i,
                      bc_out_r = bc_out_tor_r, bc_out_i = bc_out_tor_i)
            cpol = (; spec_r = copy(pol_r0), spec_i = copy(pol_i0),
                      prev_nl_r = copy(pnp_r0), prev_nl_i = copy(pnp_i0),
                      lin = lin_pol, lu = lu_pol, bc_in_r = bc_in_pol_r, bc_in_i = bc_in_pol_i,
                      bc_out_r = bc_out_pol_r, bc_out_i = bc_out_pol_i)
            GeoDynamo.gpu_velocity_field_step!(ctor, cpol, cfg, nlops, influence,
                                               inv_dt, linear_weight, cfg.lmax, bw)
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            gGre, ginvG = GeoDynamo.gpu_pack_influence(influence_dict, nl, nr, GPU())
            gnlops = (; d1 = d(d1), d2 = d(d2), lfac = d(lfac), rinv = d(rinv), rinv2 = d(rinv2),
                        rscale = d(rscale), sinθ = d(sinθ), cosθ = d(cosθ), E = E)
            gtor = (; spec_r = d(copy(tor_r0)), spec_i = d(copy(tor_i0)),
                      prev_nl_r = d(copy(pnt_r0)), prev_nl_i = d(copy(pnt_i0)),
                      lin = d(lin_tor), lu = d(lu_tor),
                      bc_in_r = d(bc_in_tor_r), bc_in_i = d(bc_in_tor_i),
                      bc_out_r = d(bc_out_tor_r), bc_out_i = d(bc_out_tor_i))
            gpol = (; spec_r = d(copy(pol_r0)), spec_i = d(copy(pol_i0)),
                      prev_nl_r = d(copy(pnp_r0)), prev_nl_i = d(copy(pnp_i0)),
                      lin = d(lin_pol), lu = d(lu_pol),
                      bc_in_r = d(bc_in_pol_r), bc_in_i = d(bc_in_pol_i),
                      bc_out_r = d(bc_out_pol_r), bc_out_i = d(bc_out_pol_i))
            GeoDynamo.gpu_velocity_field_step!(gtor, gpol, cfg,
                gnlops, (; Gre_b = gGre, invG_b = ginvG), inv_dt, linear_weight, cfg.lmax, bw)
            @test gtor.spec_r isa CUDA.CuArray
            @test gpol.spec_r isa CUDA.CuArray
            @test isapprox(Array(gtor.spec_r), ctor.spec_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.spec_r), cpol.spec_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gtor.prev_nl_r), ctor.prev_nl_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.prev_nl_i), cpol.prev_nl_i; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gtor.spec_i), ctor.spec_i; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.spec_i), cpol.spec_i; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gtor.prev_nl_i), ctor.prev_nl_i; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.prev_nl_r), cpol.prev_nl_r; atol = 1e-9, rtol = 1e-8)
        end
    end
end
