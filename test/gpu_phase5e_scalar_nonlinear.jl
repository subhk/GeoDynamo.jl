using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5e — Scalar Nonlinear (explicit half)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 3)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 3
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    # banded d1, mvals, rinv
    function band(::Type{T}, N, bw; seed) where {T}
        rng = MersenneTwister(seed); d = zeros(T, 2bw+1, N)
        for j in 1:N, i in max(1,j-bw):min(N,j+bw); d[bw+1+i-j,j]=rand(rng,T)-T(0.5); end
        GeoDynamo.BandedMatrix{T}(d, bw, N)
    end
    d1 = band(Float64, nr, bw; seed = 1).data
    mvals = Float64.(0:(nm-1)); rinv = [1.0/(0.5+0.1k) for k in 1:nr]
    rng = MersenneTwister(2)
    s_r = zeros(nl,nm,nr); s_i = zeros(nl,nm,nr)
    for mi in 1:nm, li in mi:nl, r in 1:nr; s_r[li,mi,r]=rand(rng); s_i[li,mi,r]=rand(rng); end
    u_r = rand(rng, nlat,nlon,nr); u_θ = rand(rng, nlat,nlon,nr); u_φ = rand(rng, nlat,nlon,nr)

    @testset "compose == manual chain [LOCAL]" begin
        nl_r = zeros(nl,nm,nr); nl_i = zeros(nl,nm,nr)
        GeoDynamo.gpu_scalar_nonlinear!(nl_r, nl_i, s_r, s_i, u_r, u_θ, u_φ, cfg, d1, mvals, rinv, cfg.lmax, bw)

        # manual reference: the same kernels, same order
        gr_r=zeros(nl,nm,nr); gr_i=zeros(nl,nm,nr); gt_r=zeros(nl,nm,nr); gt_i=zeros(nl,nm,nr); gp_r=zeros(nl,nm,nr); gp_i=zeros(nl,nm,nr)
        GeoDynamo.gpu_scalar_gradient!(gr_r,gr_i, gt_r,gt_i, gp_r,gp_i, s_r,s_i, d1, mvals, rinv, cfg.lmax, bw)
        # Note: actual signatures are:
        #   gpu_scalar_spectral_to_physical!(phys, spec, config) — phys first
        #   gpu_scalar_physical_to_spectral!(spec, phys, config) — spec first
        mkspec(a,b) = GeoDynamo.GPUSpectralField{Float64,typeof(a)}(cfg, nl, nm, nr, a, b)
        mkphys() = GeoDynamo.allocate_gpu_physical_field(Float64, CPU(), cfg, nr)
        grP=mkphys(); gtP=mkphys(); gpP=mkphys()
        GeoDynamo.gpu_scalar_spectral_to_physical!(grP, mkspec(gr_r,gr_i), cfg)
        GeoDynamo.gpu_scalar_spectral_to_physical!(gtP, mkspec(gt_r,gt_i), cfg)
        GeoDynamo.gpu_scalar_spectral_to_physical!(gpP, mkspec(gp_r,gp_i), cfg)
        adv = mkphys()
        GeoDynamo.gpu_scalar_advection!(adv.data, u_r, u_θ, u_φ, grP.data, gtP.data, gpP.data)
        advspec_r = zeros(nl,nm,nr); advspec_i = zeros(nl,nm,nr)
        GeoDynamo.gpu_scalar_physical_to_spectral!(mkspec(advspec_r, advspec_i), adv, cfg)

        @test nl_r == advspec_r
        @test nl_i == advspec_i
        @test all(isfinite, nl_r) && all(isfinite, nl_i)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5e gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            # CPU reference
            cnl_r = zeros(nl,nm,nr); cnl_i = zeros(nl,nm,nr)
            GeoDynamo.gpu_scalar_nonlinear!(cnl_r, cnl_i, s_r, s_i, u_r, u_θ, u_φ, cfg, d1, mvals, rinv, cfg.lmax, bw)
            # GPU
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            gnl_r = d(zeros(nl,nm,nr)); gnl_i = d(zeros(nl,nm,nr))
            GeoDynamo.gpu_scalar_nonlinear!(gnl_r, gnl_i, d(s_r), d(s_i), d(u_r), d(u_θ), d(u_φ),
                                            cfg, d(d1), d(mvals), d(rinv), cfg.lmax, bw)
            @test gnl_r isa CUDA.CuArray
            @test isapprox(Array(gnl_r), cnl_r; atol = 1e-10, rtol = 1e-9)
            @test isapprox(Array(gnl_i), cnl_i; atol = 1e-10, rtol = 1e-9)
        end
    end
end
