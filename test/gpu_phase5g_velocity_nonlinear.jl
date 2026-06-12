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
        # The Stage-2 vector transforms are un-gated (Task 1), so the chain runs
        # again — but its force projection is still the legacy tangential-only
        # raw-sphtor analysis (results WRONG until the Stage-4B N_W projection).
        @test_skip "un-gated in Task 4 (Stage-4B velocity N_W force projection)"
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
