using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5f — Full Scalar Field Step" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 4)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 4
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    function band(::Type{T}, N, bw; seed, dd = false) where {T}
        rng = MersenneTwister(seed); d = zeros(T, 2bw+1, N)
        for j in 1:N, i in max(1,j-bw):min(N,j+bw)
            d[bw+1+i-j,j] = (dd && i==j) ? (T(2bw)+rand(rng,T)) : (rand(rng,T)-T(0.5))
        end
        GeoDynamo.BandedMatrix{T}(d, bw, N)
    end
    d1 = band(Float64, nr, bw; seed = 1).data
    mvals = Float64.(0:(nm-1)); rinv = [1.0/(0.5+0.1k) for k in 1:nr]
    # per-l linear operators (L) + system LU (factorized, non-singular)
    linmats = [band(Float64, nr, bw; seed = 10 + l) for l in 1:nl]
    lin = zeros(Float64, 2bw+1, nr, nl); for l in 1:nl; lin[:,:,l] .= linmats[l].data; end
    sysmats = [band(Float64, nr, bw; seed = 20 + l, dd = true) for l in 1:nl]
    lus = [GeoDynamo.factorize_banded(m) for m in sysmats]
    lub = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
    rng = MersenneTwister(2)
    spec_r = zeros(nl,nm,nr); spec_i = zeros(nl,nm,nr)
    for mi in 1:nm, li in mi:nl, r in 1:nr; spec_r[li,mi,r]=rand(rng); spec_i[li,mi,r]=rand(rng); end
    pnl_r = rand(rng,nl,nm,nr); pnl_i = rand(rng,nl,nm,nr)
    u_r = rand(rng,nlat,nlon,nr); u_θ = rand(rng,nlat,nlon,nr); u_φ = rand(rng,nlat,nlon,nr)
    bir = rand(rng,nl,nm); bii = rand(rng,nl,nm); bor = rand(rng,nl,nm); boi = rand(rng,nl,nm)
    inv_dt = 1.0/0.01; lw = 0.5

    @testset "step == manual chain [LOCAL]" begin
        # GPU step (copies of mutable inputs)
        sr = copy(spec_r); si = copy(spec_i); pr = copy(pnl_r); pi_ = copy(pnl_i)
        GeoDynamo.gpu_scalar_field_step!(sr, si, pr, pi_, u_r, u_θ, u_φ, cfg, d1, mvals, rinv,
                                         lin, lub, bir, bii, bor, boi, inv_dt, lw, cfg.lmax, bw)
        # manual chain
        msr = copy(spec_r); msi = copy(spec_i); mpr = copy(pnl_r); mpi = copy(pnl_i)
        mnl_r = zeros(nl,nm,nr); mnl_i = zeros(nl,nm,nr)
        GeoDynamo.gpu_scalar_nonlinear!(mnl_r, mnl_i, msr, msi, u_r, u_θ, u_φ, cfg, d1, mvals, rinv, cfg.lmax, bw)
        rhs_r = zeros(nl,nm,nr); rhs_i = zeros(nl,nm,nr)
        GeoDynamo.gpu_build_rhs_cnab2!(rhs_r, rhs_i, msr, msi, mnl_r, mnl_i, mpr, mpi, lin, inv_dt, lw, bw)
        GeoDynamo.gpu_implicit_solve_field!(rhs_r, rhs_i, lub, bir, bii, bor, boi, bw)
        msr .= rhs_r; msi .= rhs_i; mpr .= mnl_r; mpi .= mnl_i

        @test sr == msr && si == msi           # updated field
        @test pr == mpr && pi_ == mpi          # rolled-over nl_prev (= this step's nl)
        @test all(isfinite, sr) && all(isfinite, pr)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5f gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            # CPU reference
            csr = copy(spec_r); csi = copy(spec_i); cpr = copy(pnl_r); cpi = copy(pnl_i)
            GeoDynamo.gpu_scalar_field_step!(csr, csi, cpr, cpi, u_r, u_θ, u_φ, cfg, d1, mvals, rinv,
                                             lin, lub, bir, bii, bor, boi, inv_dt, lw, cfg.lmax, bw)
            # GPU
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            glub = GeoDynamo.gpu_pack_banded_lu(lus, GPU())
            gsr=d(copy(spec_r)); gsi=d(copy(spec_i)); gpr=d(copy(pnl_r)); gpi=d(copy(pnl_i))
            GeoDynamo.gpu_scalar_field_step!(gsr, gsi, gpr, gpi, d(u_r), d(u_θ), d(u_φ), cfg,
                                             d(d1), d(mvals), d(rinv), d(lin), glub,
                                             d(bir), d(bii), d(bor), d(boi), inv_dt, lw, cfg.lmax, bw)
            @test gsr isa CUDA.CuArray
            @test isapprox(Array(gsr), csr; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpr), cpr; atol = 1e-9, rtol = 1e-8)
        end
    end
end
