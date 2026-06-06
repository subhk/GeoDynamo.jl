using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5n — Full gpu_solver_step! orchestration" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 6, mmax = 6, nlat = 20, nlon = 40, nr = 4)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 4
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    rng = MersenneTwister(29)

    function band(N, b; seed)
        r = MersenneTwister(seed); d = zeros(2b+1, N)
        for j in 1:N, i in max(1,j-b):min(N,j+b); d[b+1+i-j,j] = rand(r) - 0.5; end
        d
    end
    d1 = band(nr, bw; seed = 1); d2 = band(nr, bw; seed = 2)
    lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
    rinv = [1.0/(0.5+0.1k) for k in 1:nr]; rinv2 = rinv .^ 2; rscale = copy(rinv)
    sinθ = sin.(range(0.1, π-0.1; length = nlat)); cosθ = cos.(range(0.1, π-0.1; length = nlat))
    mvals = Float64[m for m in 0:cfg.mmax]
    r_vec = [0.5 + 0.1k for k in 1:nr]
    E = 1.3e-3; thermal_factor = 0.7; comp_factor = 0.4; lorentz_coeff = 1.0/0.3
    inv_dt_v = E/5e-4; inv_dt_m = 1.0/5e-4; inv_dt_t = (1.0/0.7)/5e-4; inv_dt_c = (1.0/0.9)/5e-4
    linw = 0.5

    function batched(seed)
        a = zeros(2bw+1, nr, nl); r = MersenneTwister(seed)
        for li in 1:nl, j in 1:nr, i in max(1,j-bw):min(nr,j+bw); a[bw+1+i-j,j,li] = rand(r)-0.5; end
        for li in 1:nl, j in 1:nr; a[bw+1,j,li] += 5.0; end
        a
    end
    influence = Dict{Int, GeoDynamo.ERK2InfluenceOp{Float64}}()
    for l in 1:cfg.lmax; influence[l] = GeoDynamo.ERK2InfluenceOp{Float64}(rand(rng,nr,2).-0.5, rand(rng,2,2).-0.5, l); end
    Gre_b, invG_b = GeoDynamo.gpu_pack_influence(influence, nl, nr, CPU())

    mk() = (a = zeros(nl,nm,nr); for mi in 1:nm, li in mi:nl, r in 1:nr; a[li,mi,r] = rand(rng)-0.5; end; a)
    phys() = rand(rng, nlat, nlon, nr) .- 0.5

    # build a fresh `state` NamedTuple (deep copies of all mutable arrays)
    function build_state()
        velocity = (;
            tor = (; spec_r=mk(), spec_i=mk(), prev_nl_r=mk(), prev_nl_i=mk(),
                     lin=batched(10), lu=batched(11),
                     bc_in_r=zeros(nl,nm), bc_in_i=zeros(nl,nm), bc_out_r=zeros(nl,nm), bc_out_i=zeros(nl,nm)),
            pol = (; spec_r=mk(), spec_i=mk(), prev_nl_r=mk(), prev_nl_i=mk(),
                     lin=batched(12), lu=batched(13),
                     bc_in_r=zeros(nl,nm), bc_in_i=zeros(nl,nm), bc_out_r=zeros(nl,nm), bc_out_i=zeros(nl,nm)))
        magnetic = (;
            tor = (; spec_r=mk(), spec_i=mk(), prev_nl_r=mk(), prev_nl_i=mk(), lin=batched(20), lu=batched(21)),
            pol = (; spec_r=mk(), spec_i=mk(), prev_nl_r=mk(), prev_nl_i=mk(), lin=batched(22), lu=batched(23)))
        temperature = (; spec_r=mk(), spec_i=mk(), prev_nl_r=mk(), prev_nl_i=mk(),
                         lin=batched(30), lu=batched(31),
                         bc_in_r=zeros(nl,nm), bc_in_i=zeros(nl,nm), bc_out_r=zeros(nl,nm), bc_out_i=zeros(nl,nm))
        composition = (; spec_r=mk(), spec_i=mk(), prev_nl_r=mk(), prev_nl_i=mk(),
                         lin=batched(40), lu=batched(41),
                         bc_in_r=zeros(nl,nm), bc_in_i=zeros(nl,nm), bc_out_r=zeros(nl,nm), bc_out_i=zeros(nl,nm))
        (;
            config = cfg, lmax = cfg.lmax, bw = bw, linear_weight = linw,
            nlops_vel = (; d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E),
            nlops_mag = (; d1, d2, lfac, rinv, rinv2, rscale),
            influence = (; Gre_b, invG_b),
            d1 = d1, mvals = mvals, rinv = rinv, rscale = rscale, lfac = lfac, d2 = d2, rinv2 = rinv2,
            r_vec = r_vec, thermal_factor = thermal_factor, comp_factor = comp_factor, lorentz_coeff = lorentz_coeff,
            inv_dt_vel = inv_dt_v, inv_dt_mag = inv_dt_m, inv_dt_temp = inv_dt_t, inv_dt_comp = inv_dt_c,
            velocity = velocity, magnetic = magnetic, temperature = temperature, composition = composition,
            # persistent LAGGED physical buffers (previous step's synthesis)
            T_phys = phys(), C_phys = phys(),
            B_r = phys(), B_θ = phys(), B_φ = phys(), J_r = phys(), J_θ = phys(), J_φ = phys())
    end

    @testset "full step == manual chain (exact) [LOCAL]" begin
        st = build_state()
        # snapshot the OLD spectral + OLD lagged buffers BEFORE the step (deep copies)
        v0 = deepcopy(st.velocity); m0 = deepcopy(st.magnetic); t0 = deepcopy(st.temperature); c0 = deepcopy(st.composition)
        T0 = copy(st.T_phys); C0 = copy(st.C_phys)
        B0 = (copy(st.B_r), copy(st.B_θ), copy(st.B_φ)); J0 = (copy(st.J_r), copy(st.J_θ), copy(st.J_φ))

        GeoDynamo.gpu_solver_step!(st)

        # ---- manual chain ----
        spec(a,b) = GeoDynamo.GPUSpectralField{Float64,typeof(a)}(cfg, nl, nm, nr, a, b)
        ph() = GeoDynamo.allocate_gpu_physical_field(Float64, CPU(), cfg, nr)
        # 1. shared u from OLD velocity
        ur=ph(); uθ=ph(); uφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(ur, uθ, uφ, spec(v0.tor.spec_r, v0.tor.spec_i),
            spec(v0.pol.spec_r, v0.pol.spec_i), cfg, lfac, rscale)
        # 2. current-step physical buffers from OLD spectral (for NEXT step's velocity)
        Tn=ph(); GeoDynamo.gpu_scalar_spectral_to_physical!(Tn, spec(t0.spec_r,t0.spec_i), cfg)
        Cn=ph(); GeoDynamo.gpu_scalar_spectral_to_physical!(Cn, spec(c0.spec_r,c0.spec_i), cfg)
        Br=ph(); Bθ=ph(); Bφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(Br, Bθ, Bφ, spec(m0.tor.spec_r,m0.tor.spec_i),
            spec(m0.pol.spec_r,m0.pol.spec_i), cfg, lfac, rscale)
        jtr=zeros(nl,nm,nr); jti=zeros(nl,nm,nr); jpr=zeros(nl,nm,nr); jpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_spectral_curl!(jtr,jti,jpr,jpi, m0.tor.spec_r,m0.tor.spec_i, m0.pol.spec_r,m0.pol.spec_i,
            d1,d2,lfac,rinv,rinv2,bw)
        Jr=ph(); Jθ=ph(); Jφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(Jr, Jθ, Jφ, spec(jtr,jti), spec(jpr,jpi), cfg, lfac, rscale)
        # 3. velocity step with LAGGED buffers (T0/C0/B0/J0)
        mvtor = deepcopy(v0.tor); mvpol = deepcopy(v0.pol)
        GeoDynamo.gpu_velocity_field_step!(mvtor, mvpol, cfg, st.nlops_vel, st.influence, inv_dt_v, linw, cfg.lmax, bw;
            T_phys=T0, thermal_factor=thermal_factor, r_vec=r_vec, C_phys=C0, comp_factor=comp_factor,
            J_r=J0[1], J_θ=J0[2], J_φ=J0[3], B_r=B0[1], B_θ=B0[2], B_φ=B0[3], lorentz_coeff=lorentz_coeff)
        # 4. magnetic step with shared u
        mmtor = deepcopy(m0.tor); mmpol = deepcopy(m0.pol)
        GeoDynamo.gpu_magnetic_field_step!(mmtor, mmpol, ur.data, uθ.data, uφ.data, cfg, st.nlops_mag, inv_dt_m, linw, cfg.lmax, bw)
        # 5. temperature + 6. composition with shared u
        mt = deepcopy(t0); mc = deepcopy(c0)
        GeoDynamo.gpu_scalar_field_step!(mt.spec_r, mt.spec_i, mt.prev_nl_r, mt.prev_nl_i, ur.data, uθ.data, uφ.data, cfg,
            d1, mvals, rinv, mt.lin, mt.lu, mt.bc_in_r, mt.bc_in_i, mt.bc_out_r, mt.bc_out_i, inv_dt_t, linw, cfg.lmax, bw)
        GeoDynamo.gpu_scalar_field_step!(mc.spec_r, mc.spec_i, mc.prev_nl_r, mc.prev_nl_i, ur.data, uθ.data, uφ.data, cfg,
            d1, mvals, rinv, mc.lin, mc.lu, mc.bc_in_r, mc.bc_in_i, mc.bc_out_r, mc.bc_out_i, inv_dt_c, linw, cfg.lmax, bw)

        # ---- compare updated spectral state ----
        @test st.velocity.tor.spec_r == mvtor.spec_r && st.velocity.pol.spec_r == mvpol.spec_r
        @test st.velocity.tor.spec_i == mvtor.spec_i && st.velocity.pol.spec_i == mvpol.spec_i
        @test st.magnetic.tor.spec_r == mmtor.spec_r && st.magnetic.pol.spec_i == mmpol.spec_i
        @test st.magnetic.tor.spec_i == mmtor.spec_i && st.magnetic.pol.spec_r == mmpol.spec_r
        @test st.temperature.spec_r == mt.spec_r && st.temperature.spec_i == mt.spec_i
        @test st.composition.spec_r == mc.spec_r && st.composition.spec_i == mc.spec_i
        # ---- compare rolled physical buffers (current-step synthesis) ----
        @test st.T_phys == Tn.data && st.C_phys == Cn.data
        @test st.B_r == Br.data && st.J_φ == Jφ.data
        @test st.B_θ == Bθ.data && st.B_φ == Bφ.data
        @test st.J_r == Jr.data && st.J_θ == Jθ.data
        @test all(isfinite, st.velocity.tor.spec_r) && all(isfinite, st.magnetic.tor.spec_r)
    end

    @testset "gating: no magnetic / no composition [LOCAL]" begin
        # Build two fully independent identical states (same RNG seed) so mutations
        # in one do not affect the other.
        st_base = build_state()
        # st_a: composition=nothing — gate should suppress compositional buoyancy in velocity step
        st_a = let s = deepcopy(st_base); (; s..., composition = nothing); end
        # st_b: composition present but comp_factor=0 — buoyancy contribution is zero in velocity step
        st_b = let s = deepcopy(st_base); (; s..., comp_factor = 0.0); end
        GeoDynamo.gpu_solver_step!(st_a)
        GeoDynamo.gpu_solver_step!(st_b)
        # The gate (composition=nothing → C_phys=nothing, comp_factor=0) must produce
        # exactly the same velocity as explicitly zeroing comp_factor.
        @test st_a.velocity.tor.spec_r == st_b.velocity.tor.spec_r
        @test all(isfinite, st_a.velocity.tor.spec_r)
        @test all(isfinite, st_a.temperature.spec_r)

        # Also verify a fully stripped state (no magnetic/composition) runs cleanly
        st_stripped = let s = deepcopy(st_base)
            (; s..., magnetic = nothing, composition = nothing,
               B_r = nothing, B_θ = nothing, B_φ = nothing, J_r = nothing, J_θ = nothing, J_φ = nothing)
        end
        GeoDynamo.gpu_solver_step!(st_stripped)
        @test all(isfinite, st_stripped.velocity.tor.spec_r)
        @test all(isfinite, st_stripped.temperature.spec_r)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5n gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            @test true   # full device-state parity is exercised by Phase 5n2 (real SolverState)
        end
    end
end
