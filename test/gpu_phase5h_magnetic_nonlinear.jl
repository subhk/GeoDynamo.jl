using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5h — Magnetic Nonlinear (induction ∇×(u×B))" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 4)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 4
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    function band(::Type{T}, N, bw; seed) where {T}
        rng = MersenneTwister(seed); d = zeros(T, 2bw+1, N)
        for j in 1:N, i in max(1,j-bw):min(N,j+bw); d[bw+1+i-j,j]=rand(rng,T)-T(0.5); end
        d
    end
    d1 = band(Float64, nr, bw; seed = 1); d2 = band(Float64, nr, bw; seed = 2)
    lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
    rinv = [1.0/(0.5+0.1k) for k in 1:nr]; rinv2 = rinv .^ 2; rscale = copy(rinv)
    rng = MersenneTwister(3)
    btr=zeros(nl,nm,nr); bti=zeros(nl,nm,nr); bpr=zeros(nl,nm,nr); bpi=zeros(nl,nm,nr)
    for mi in 1:nm, li in mi:nl, r in 1:nr
        btr[li,mi,r]=rand(rng); bti[li,mi,r]=rand(rng); bpr[li,mi,r]=rand(rng); bpi[li,mi,r]=rand(rng)
    end
    u_r=rand(rng,nlat,nlon,nr); u_θ=rand(rng,nlat,nlon,nr); u_φ=rand(rng,nlat,nlon,nr)

    @testset "magnetic nonlinear == manual chain [LOCAL]" begin
        ntr=zeros(nl,nm,nr); nti=zeros(nl,nm,nr); npr=zeros(nl,nm,nr); npi=zeros(nl,nm,nr)
        GeoDynamo.gpu_magnetic_nonlinear!(ntr,nti, npr,npi, btr,bti, bpr,bpi, u_r,u_θ,u_φ,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, cfg.lmax, bw)

        # manual chain
        spec(a,b) = GeoDynamo.GPUSpectralField{Float64,typeof(a)}(cfg, nl, nm, nr, a, b)
        ph() = GeoDynamo.allocate_gpu_physical_field(Float64, CPU(), cfg, nr)
        # 1. B → physical
        Br=ph(); Bθ=ph(); Bφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(Br,Bθ,Bφ, spec(btr,bti), spec(bpr,bpi), cfg, lfac, rscale)
        # 2. uB = u×B
        ubr=ph(); ubθ=ph(); ubφ=ph()
        GeoDynamo.gpu_cross!(ubr.data,ubθ.data,ubφ.data, u_r,u_θ,u_φ, Br.data,Bθ.data,Bφ.data, 1.0)
        # 3. uB → spectral (work_tor, work_pol)
        wtr=zeros(nl,nm,nr); wti=zeros(nl,nm,nr); wpr=zeros(nl,nm,nr); wpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_vector_physical_to_spectral!(spec(wtr,wti), spec(wpr,wpi), ubθ, ubφ, cfg)
        # 4. curl(work) → nl
        mntr=zeros(nl,nm,nr); mnti=zeros(nl,nm,nr); mnpr=zeros(nl,nm,nr); mnpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_spectral_curl!(mntr,mnti, mnpr,mnpi, wtr,wti, wpr,wpi, d1,d2, lfac, rinv, rinv2, bw)

        @test ntr == mntr
        @test nti == mnti
        @test npr == mnpr
        @test npi == mnpi
        @test all(isfinite, ntr) && all(isfinite, npr)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5h gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            cntr=zeros(nl,nm,nr); cnti=zeros(nl,nm,nr); cnpr=zeros(nl,nm,nr); cnpi=zeros(nl,nm,nr)
            GeoDynamo.gpu_magnetic_nonlinear!(cntr,cnti, cnpr,cnpi, btr,bti, bpr,bpi, u_r,u_θ,u_φ,
                cfg, d1, d2, lfac, rinv, rinv2, rscale, cfg.lmax, bw)
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            gntr=d(zeros(nl,nm,nr)); gnti=d(zeros(nl,nm,nr)); gnpr=d(zeros(nl,nm,nr)); gnpi=d(zeros(nl,nm,nr))
            GeoDynamo.gpu_magnetic_nonlinear!(gntr,gnti, gnpr,gnpi, d(btr),d(bti), d(bpr),d(bpi),
                d(u_r),d(u_θ),d(u_φ), cfg, d(d1), d(d2), d(lfac), d(rinv), d(rinv2), d(rscale), cfg.lmax, bw)
            @test gntr isa CUDA.CuArray
            @test isapprox(Array(gntr), cntr; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gnti), cnti; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gnpr), cnpr; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gnpi), cnpi; atol = 1e-9, rtol = 1e-8)
        end
    end
end
