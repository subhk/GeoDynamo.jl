using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5i — Coupled Velocity (buoyancy + Lorentz)" begin
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
    r_vec = [0.5+0.1k for k in 1:nr]
    sinθ = [sin(π*(i-0.5)/nlat) for i in 1:nlat]; cosθ = [cos(π*(i-0.5)/nlat) for i in 1:nlat]
    E = 1e-3
    rng = MersenneTwister(3)
    tor_r=zeros(nl,nm,nr); tor_i=zeros(nl,nm,nr); pol_r=zeros(nl,nm,nr); pol_i=zeros(nl,nm,nr)
    for mi in 1:nm, li in mi:nl, r in 1:nr
        tor_r[li,mi,r]=rand(rng); tor_i[li,mi,r]=rand(rng); pol_r[li,mi,r]=rand(rng); pol_i[li,mi,r]=rand(rng)
    end
    Tp = rand(rng,nlat,nlon,nr); Cp = rand(rng,nlat,nlon,nr)
    Jr=rand(rng,nlat,nlon,nr); Jθ=rand(rng,nlat,nlon,nr); Jφ=rand(rng,nlat,nlon,nr)
    Br=rand(rng,nlat,nlon,nr); Bθ=rand(rng,nlat,nlon,nr); Bφ=rand(rng,nlat,nlon,nr)
    tf = 0.7; cf = 0.4; lc = 1.0/0.3

    @testset "coupled == core + buoyancy + Lorentz manual chain [LOCAL]" begin
        ntr=zeros(nl,nm,nr); nti=zeros(nl,nm,nr); npr=zeros(nl,nm,nr); npi=zeros(nl,nm,nr)
        GeoDynamo.gpu_velocity_nonlinear!(
            ntr,nti, npr,npi, tor_r,tor_i, pol_r,pol_i,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw;
            T_phys = Tp, thermal_factor = tf, r_vec = r_vec, C_phys = Cp, comp_factor = cf,
            J_r = Jr, J_θ = Jθ, J_φ = Jφ, B_r = Br, B_θ = Bθ, B_φ = Bφ, lorentz_coeff = lc)

        btr=zeros(nl,nm,nr); bti=zeros(nl,nm,nr); bpr=zeros(nl,nm,nr); bpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_velocity_nonlinear!(
            btr,bti, bpr,bpi, tor_r,tor_i, pol_r,pol_i,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw)

        arch = CPU()
        spec(a, b) = GeoDynamo.GPUSpectralField{eltype(a), typeof(a)}(cfg, size(a,1), size(a,2), size(a,3), a, b)
        ph() = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfg, nr)
        ar=ph(); aθ=ph(); aφ=ph()
        GeoDynamo.gpu_buoyancy_add!(ar.data, Tp, r_vec, tf)
        GeoDynamo.gpu_buoyancy_add!(ar.data, Cp, r_vec, cf)
        GeoDynamo.gpu_cross_add!(ar.data, aθ.data, aφ.data, Jr,Jθ,Jφ, Br,Bθ,Bφ, lc)
        dtr=zeros(nl,nm,nr); dti=zeros(nl,nm,nr); dpr=zeros(nl,nm,nr); dpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_vector_physical_to_spectral!(spec(dtr,dti), spec(dpr,dpi), aθ, aφ, cfg)
        qr=zeros(nl,nm,nr); qi=zeros(nl,nm,nr)
        GeoDynamo.gpu_scalar_physical_to_spectral!(spec(qr,qi), ar, cfg)
        GeoDynamo.gpu_poloidal_force_projection!(dpr,dpi, qr,qi, d1,rinv,bw)

        @test isapprox(ntr, btr .+ dtr; atol = 1e-10, rtol = 1e-10)
        @test isapprox(nti, bti .+ dti; atol = 1e-10, rtol = 1e-10)
        @test isapprox(npr, bpr .+ dpr; atol = 1e-10, rtol = 1e-10)
        @test isapprox(npi, bpi .+ dpi; atol = 1e-10, rtol = 1e-10)
    end

    @testset "no couplings == velocity-only (5g unchanged) [LOCAL]" begin
        a1=zeros(nl,nm,nr); a2=zeros(nl,nm,nr); a3=zeros(nl,nm,nr); a4=zeros(nl,nm,nr)
        b1=zeros(nl,nm,nr); b2=zeros(nl,nm,nr); b3=zeros(nl,nm,nr); b4=zeros(nl,nm,nr)
        GeoDynamo.gpu_velocity_nonlinear!(
            a1,a2, a3,a4, tor_r,tor_i, pol_r,pol_i,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw)
        GeoDynamo.gpu_velocity_nonlinear!(
            b1,b2, b3,b4, tor_r,tor_i, pol_r,pol_i,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw;
            T_phys = nothing, C_phys = nothing, J_r = nothing)
        @test a1 == b1
        @test a2 == b2
        @test a3 == b3
        @test a4 == b4
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5i gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            cntr=zeros(nl,nm,nr); cnti=zeros(nl,nm,nr); cnpr=zeros(nl,nm,nr); cnpi=zeros(nl,nm,nr)
            GeoDynamo.gpu_velocity_nonlinear!(cntr,cnti, cnpr,cnpi, tor_r,tor_i, pol_r,pol_i,
                cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw;
                T_phys=Tp, thermal_factor=tf, r_vec=r_vec, C_phys=Cp, comp_factor=cf,
                J_r=Jr, J_θ=Jθ, J_φ=Jφ, B_r=Br, B_θ=Bθ, B_φ=Bφ, lorentz_coeff=lc)
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            gntr=d(zeros(nl,nm,nr)); gnti=d(zeros(nl,nm,nr)); gnpr=d(zeros(nl,nm,nr)); gnpi=d(zeros(nl,nm,nr))
            GeoDynamo.gpu_velocity_nonlinear!(gntr,gnti, gnpr,gnpi, d(tor_r),d(tor_i), d(pol_r),d(pol_i),
                cfg, d(d1), d(d2), d(lfac), d(rinv), d(rinv2), d(rscale), d(sinθ), d(cosθ), E, cfg.lmax, bw;
                T_phys=d(Tp), thermal_factor=tf, r_vec=d(r_vec), C_phys=d(Cp), comp_factor=cf,
                J_r=d(Jr), J_θ=d(Jθ), J_φ=d(Jφ), B_r=d(Br), B_θ=d(Bθ), B_φ=d(Bφ), lorentz_coeff=lc)
            @test gntr isa CUDA.CuArray
            @test isapprox(Array(gntr), cntr; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gnti), cnti; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gnpr), cnpr; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gnpi), cnpi; atol = 1e-9, rtol = 1e-8)
        end
    end
end
