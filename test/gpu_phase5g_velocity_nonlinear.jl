using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5g — Velocity Nonlinear (u×ω + Coriolis)" begin
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
    sinθ = [sin(π*(i-0.5)/nlat) for i in 1:nlat]; cosθ = [cos(π*(i-0.5)/nlat) for i in 1:nlat]
    E = 1e-3
    rng = MersenneTwister(3)
    tor_r=zeros(nl,nm,nr); tor_i=zeros(nl,nm,nr); pol_r=zeros(nl,nm,nr); pol_i=zeros(nl,nm,nr)
    for mi in 1:nm, li in mi:nl, r in 1:nr
        tor_r[li,mi,r]=rand(rng); tor_i[li,mi,r]=rand(rng); pol_r[li,mi,r]=rand(rng); pol_i[li,mi,r]=rand(rng)
    end

    @testset "velocity nonlinear == manual chain [LOCAL]" begin
        ntr=zeros(nl,nm,nr); nti=zeros(nl,nm,nr); npr=zeros(nl,nm,nr); npi=zeros(nl,nm,nr)
        GeoDynamo.gpu_velocity_nonlinear!(ntr,nti, npr,npi, tor_r,tor_i, pol_r,pol_i,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw)

        # manual chain
        spec(a,b) = GeoDynamo.GPUSpectralField{Float64,typeof(a)}(cfg, nl, nm, nr, a, b)
        ph() = GeoDynamo.allocate_gpu_physical_field(Float64, CPU(), cfg, nr)
        # 1. velocity → physical
        ur=ph(); uθ=ph(); uφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(ur,uθ,uφ, spec(tor_r,tor_i), spec(pol_r,pol_i), cfg, lfac, rscale)
        # 2. vorticity (curl) spectral
        wtr=zeros(nl,nm,nr); wti=zeros(nl,nm,nr); wpr=zeros(nl,nm,nr); wpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_spectral_curl!(wtr,wti, wpr,wpi, tor_r,tor_i, pol_r,pol_i, d1,d2, lfac, rinv, rinv2, bw)
        # 3. vorticity → physical
        wr=ph(); wθ=ph(); wφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(wr,wθ,wφ, spec(wtr,wti), spec(wpr,wpi), cfg, lfac, rscale)
        # 4. adv = E·(u×ω) − ẑ×u
        ar=ph(); aθ=ph(); aφ=ph()
        GeoDynamo.gpu_cross!(ar.data,aθ.data,aφ.data, ur.data,uθ.data,uφ.data, wr.data,wθ.data,wφ.data, E)
        GeoDynamo.gpu_coriolis_sub!(ar.data,aθ.data,aφ.data, ur.data,uθ.data,uφ.data, sinθ, cosθ)
        # 5. analyze tangential → nl_tor/nl_pol
        mntr=zeros(nl,nm,nr); mnti=zeros(nl,nm,nr); mnpr=zeros(nl,nm,nr); mnpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_vector_physical_to_spectral!(spec(mntr,mnti), spec(mnpr,mnpi), aθ, aφ, cfg)

        @test ntr == mntr
        @test nti == mnti
        @test npr == mnpr
        @test npi == mnpi
        @test all(isfinite, ntr) && all(isfinite, npr)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5g gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            cntr=zeros(nl,nm,nr); cnti=zeros(nl,nm,nr); cnpr=zeros(nl,nm,nr); cnpi=zeros(nl,nm,nr)
            GeoDynamo.gpu_velocity_nonlinear!(cntr,cnti, cnpr,cnpi, tor_r,tor_i, pol_r,pol_i,
                cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw)
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            gntr=d(zeros(nl,nm,nr)); gnti=d(zeros(nl,nm,nr)); gnpr=d(zeros(nl,nm,nr)); gnpi=d(zeros(nl,nm,nr))
            GeoDynamo.gpu_velocity_nonlinear!(gntr,gnti, gnpr,gnpi, d(tor_r),d(tor_i), d(pol_r),d(pol_i),
                cfg, d(d1), d(d2), d(lfac), d(rinv), d(rinv2), d(rscale), d(sinθ), d(cosθ), E, cfg.lmax, bw)
            @test gntr isa CUDA.CuArray
            @test isapprox(Array(gntr), cntr; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gnpr), cnpr; atol = 1e-9, rtol = 1e-8)
        end
    end
end
