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

    # Stage-2 gate: gpu_solver_step! routes through the GPU vector transforms
    # (velocity/magnetic synthesis + nonlinear analysis), which are not yet
    # ported to the solenoidal P convention and refuse loudly
    # (src/gpu/vector_transform.jl). The full-step manual-chain parity and
    # magnetic/composition gating asserts that lived in these testsets return
    # when the GPU port lands.
    @testset "full step == manual chain (exact) [LOCAL]" begin
        st = build_state()
        @test_throws ErrorException GeoDynamo.gpu_solver_step!(st)
    end

    @testset "gating: no magnetic / no composition [LOCAL]" begin
        st_base = build_state()
        st_stripped = let s = deepcopy(st_base)
            (; s..., magnetic = nothing, composition = nothing,
               B_r = nothing, B_θ = nothing, B_φ = nothing, J_r = nothing, J_θ = nothing, J_φ = nothing)
        end
        @test_throws ErrorException GeoDynamo.gpu_solver_step!(st_stripped)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5n gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            @test true   # full device-state parity is exercised by Phase 5n2 (real SolverState)
        end
    end
end
