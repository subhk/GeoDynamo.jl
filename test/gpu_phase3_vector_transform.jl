using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
import SHTnsKit
using Random

# ---------------------------------------------------------------------------
# Stage-2 solenoidal fixtures (shell domain; radial axis fully local).
# CPU reference = the public vector transform pair (numerics.jl):
#   synthesis: tangential sphtor of (S = (∂_r P)/r, T); v_r = scalar synth of
#              P·l(l+1)/r²
#   analysis:  sphtor → raw (S,T), toroidal ← T; poloidal ← Q-based recovery
#              P = r²·Q/λ with Q = scalar analysis of v_r (l=0 zeroed)
# ---------------------------------------------------------------------------
const VT_NR = 16

_vt_spec(cfg, dom) = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
_vt_vec(cfg, dom)  = GeoDynamo.create_shtns_vector_field(Float64, cfg, dom,
    (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))

# Smooth band-limited random fill with endpoint-vanishing radial profiles
# (the solenoidal_transform_pair.jl idiom).
function _vt_fill!(specs, cfg, dom)
    for spec in specs
        sr = parent(spec.data_real); si = parent(spec.data_imag)
        for lm in 1:cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]; m = cfg.m_values[lm]
            l >= 1 || continue
            for r_idx in 1:dom.N
                x = (dom.r[r_idx, 4] - dom.r[1, 4]) / (dom.r[dom.N, 4] - dom.r[1, 4])
                v = sinpi(x) * randn() * 1e-2
                GeoDynamo.set_local_spectral_value!(sr, slot, r_idx, v)
                m > 0 && GeoDynamo.set_local_spectral_value!(si, slot, r_idx, 0.7v)
            end
        end
    end
    return nothing
end

# Scatter a CPU slot-packed spectral field into a freshly allocated dense GPU
# spectral field on `arch`.
function _vt_to_gpu(spec_cpu, cfg, arch)
    g = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfg, VT_NR)
    dr, di = GeoDynamo.cpu_spectral_to_dense(spec_cpu, cfg, VT_NR, Float64)
    copyto!(g.data_real, dr); copyto!(g.data_imag, di)
    return g
end

@testset "GPU Phase 3 — Vector Transform" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 3)
    sht = cfg.sht_config
    nl, nm = cfg.lmax + 1, cfg.mmax + 1

    @testset "sphtor helper CPU method [LOCAL]" begin
        S = zeros(ComplexF64, nl, nm); T = zeros(ComplexF64, nl, nm)
        S[3, 1] = 1.0; T[4, 2] = 0.5 - 0.25im
        vt, vp = GeoDynamo._vector_synth_sphtor(sht, S, T)
        @test size(vt) == (cfg.nlat, cfg.nlon) && size(vp) == (cfg.nlat, cfg.nlon)
        rt, rp = SHTnsKit.synthesis_sphtor(sht, S, T; real_output = true)
        @test vt == rt && vp == rp
        S2, T2 = GeoDynamo._vector_anal_sphtor(sht, vt, vp)
        rS, rT = SHTnsKit.analysis_sphtor(sht, vt, vp)
        @test S2 == rS && T2 == rT
    end

    @testset "gpu_vr_scale! [LOCAL]" begin
        nr = 3
        pr = rand(Float64, nl, nm, nr); pi_ = rand(Float64, nl, nm, nr)
        lfac = Float64[l * (l + 1) for l in 0:cfg.lmax]      # length nl
        rscale = [1.0 / (0.5 + 0.1k)^2 for k in 1:nr]         # length nr (Stage-2: 1/r²)
        vr = zeros(Float64, nl, nm, nr); vi = zeros(Float64, nl, nm, nr)
        GeoDynamo.gpu_vr_scale!(vr, vi, pr, pi_, lfac, rscale)
        refr = similar(vr); refi = similar(vi)
        @inbounds for k in 1:nr, m in 1:nm, l in 1:nl
            f = lfac[l] * rscale[k]
            refr[l,m,k] = pr[l,m,k] * f
            refi[l,m,k] = pi_[l,m,k] * f
        end
        @test vr == refr && vi == refi
    end

    # --- Stage-2 fixtures shared by the equivalence testsets below ---
    dom  = GeoDynamo.create_radial_domain(VT_NR)
    cfgv = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48,
        nr = VT_NR)
    # The SAME banded ∂r operator the CPU _spheroidal_from_poloidal! builds.
    D1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    bw = (size(D1.data, 1) - 1) ÷ 2
    lfacv  = Float64[l * (l + 1) for l in 0:cfgv.lmax]
    rscale = dom.r[1:VT_NR, 2]   # 1/r² — Stage-2 v_r factor (u_r = l(l+1)·P/r²)
    rinv   = dom.r[1:VT_NR, 3]   # 1/r  — spheroidal coupling S = (∂_r P)/r
    r2     = dom.r[1:VT_NR, 6]   # r²   — Q-based poloidal recovery P = r²·Q/λ

    @testset "domain.r power columns (p ⇒ r^(p-4))" begin
        @test all(k -> isapprox(dom.r[k, 6], dom.r[k, 4]^2), 1:VT_NR)
        @test all(k -> isapprox(dom.r[k, 2], 1.0 / dom.r[k, 4]^2), 1:VT_NR)
        @test all(k -> isapprox(dom.r[k, 3], 1.0 / dom.r[k, 4]), 1:VT_NR)
    end

    @testset "GPU vector synthesis == CPU (Stage-2 solenoidal) [LOCAL]" begin
        Random.seed!(1101)
        tor = _vt_spec(cfgv, dom); pol = _vt_spec(cfgv, dom)
        _vt_fill!((tor, pol), cfgv, dom)
        V = _vt_vec(cfgv, dom)
        GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom)
        vr_c = copy(parent(V.r_component.data))
        vθ_c = copy(parent(V.θ_component.data))
        vφ_c = copy(parent(V.φ_component.data))

        arch = CPU()
        tor_g = _vt_to_gpu(tor, cfgv, arch); pol_g = _vt_to_gpu(pol, cfgv, arch)
        gvr = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfgv, VT_NR)
        gvθ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfgv, VT_NR)
        gvφ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfgv, VT_NR)
        GeoDynamo.gpu_vector_spectral_to_physical!(gvr, gvθ, gvφ, tor_g, pol_g, cfgv,
            lfacv, rscale, D1.data, rinv, bw)
        # 1e-13: CPU couples S=(∂_r P)/r via BandedMatrix mul! whose accumulation
        # order can differ from the GPU banded kernel.
        @test maximum(abs, gvr.data .- vr_c) <= 1e-13 * max(1.0, maximum(abs, vr_c))
        @test maximum(abs, gvθ.data .- vθ_c) <= 1e-13 * max(1.0, maximum(abs, vθ_c))
        @test maximum(abs, gvφ.data .- vφ_c) <= 1e-13 * max(1.0, maximum(abs, vφ_c))
    end

    @testset "GPU vector analysis == CPU (Q-based poloidal) [LOCAL]" begin
        Random.seed!(1102)
        tor = _vt_spec(cfgv, dom); pol = _vt_spec(cfgv, dom)
        _vt_fill!((tor, pol), cfgv, dom)
        V = _vt_vec(cfgv, dom)
        # Physical data from the CPU synthesis → solenoidal-consistent input.
        GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom)
        tor2 = _vt_spec(cfgv, dom); pol2 = _vt_spec(cfgv, dom)
        GeoDynamo.vector_physical_to_spectral!(V, tor2, pol2; domain = dom)
        ct_r, ct_i = GeoDynamo.cpu_spectral_to_dense(tor2, cfgv, VT_NR, Float64)
        cp_r, cp_i = GeoDynamo.cpu_spectral_to_dense(pol2, cfgv, VT_NR, Float64)

        arch = CPU()
        gvr = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfgv, VT_NR)
        gvθ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfgv, VT_NR)
        gvφ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfgv, VT_NR)
        copyto!(gvr.data, parent(V.r_component.data))
        copyto!(gvθ.data, parent(V.θ_component.data))
        copyto!(gvφ.data, parent(V.φ_component.data))
        tor_g = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfgv, VT_NR)
        pol_g = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfgv, VT_NR)
        GeoDynamo.gpu_vector_physical_to_spectral!(tor_g, pol_g, gvθ, gvφ, cfgv;
            vr = gvr, lfac = lfacv, r2 = r2)
        @test maximum(abs, tor_g.data_real .- ct_r) <= 1e-13 * max(1.0, maximum(abs, ct_r))
        @test maximum(abs, tor_g.data_imag .- ct_i) <= 1e-13 * max(1.0, maximum(abs, ct_i))
        @test maximum(abs, pol_g.data_real .- cp_r) <= 1e-13 * max(1.0, maximum(abs, cp_r))
        @test maximum(abs, pol_g.data_imag .- cp_i) <= 1e-13 * max(1.0, maximum(abs, cp_i))
        # Solenoidal (default) mode without vr/lfac/r2 must refuse loudly.
        @test_throws ArgumentError GeoDynamo.gpu_vector_physical_to_spectral!(
            tor_g, pol_g, gvθ, gvφ, cfgv)
    end

    @testset "GPU raw_spheroidal mode == CPU (tangential-basis primitive) [LOCAL]" begin
        Random.seed!(1103)
        tor = _vt_spec(cfgv, dom); pol = _vt_spec(cfgv, dom)
        _vt_fill!((tor, pol), cfgv, dom)

        # Raw synthesis: S = stored coefficients verbatim (no P′/r prep);
        # v_r still l(l+1)·P/r² (domain supplied on the CPU side).
        V = _vt_vec(cfgv, dom)
        GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom,
            raw_spheroidal = true)
        vr_c = copy(parent(V.r_component.data))
        vθ_c = copy(parent(V.θ_component.data))
        vφ_c = copy(parent(V.φ_component.data))

        arch = CPU()
        tor_g = _vt_to_gpu(tor, cfgv, arch); pol_g = _vt_to_gpu(pol, cfgv, arch)
        gvr = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfgv, VT_NR)
        gvθ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfgv, VT_NR)
        gvφ = GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfgv, VT_NR)
        GeoDynamo.gpu_vector_spectral_to_physical!(gvr, gvθ, gvφ, tor_g, pol_g, cfgv,
            lfacv, rscale, D1.data, rinv, bw; raw_spheroidal = true)
        @test maximum(abs, gvr.data .- vr_c) <= 1e-13 * max(1.0, maximum(abs, vr_c))
        @test maximum(abs, gvθ.data .- vθ_c) <= 1e-13 * max(1.0, maximum(abs, vθ_c))
        @test maximum(abs, gvφ.data .- vφ_c) <= 1e-13 * max(1.0, maximum(abs, vφ_c))

        # Raw analysis: poloidal slot ← raw sphtor S; v_r NOT consumed.
        torR = _vt_spec(cfgv, dom); polR = _vt_spec(cfgv, dom)
        GeoDynamo.vector_physical_to_spectral!(V, torR, polR; domain = dom,
            raw_spheroidal = true)
        ct_r, ct_i = GeoDynamo.cpu_spectral_to_dense(torR, cfgv, VT_NR, Float64)
        cp_r, cp_i = GeoDynamo.cpu_spectral_to_dense(polR, cfgv, VT_NR, Float64)
        torR_g = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfgv, VT_NR)
        polR_g = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, arch, cfgv, VT_NR)
        GeoDynamo.gpu_vector_physical_to_spectral!(torR_g, polR_g, gvθ, gvφ, cfgv;
            raw_spheroidal = true)
        @test maximum(abs, torR_g.data_real .- ct_r) <= 1e-13 * max(1.0, maximum(abs, ct_r))
        @test maximum(abs, torR_g.data_imag .- ct_i) <= 1e-13 * max(1.0, maximum(abs, ct_i))
        @test maximum(abs, polR_g.data_real .- cp_r) <= 1e-13 * max(1.0, maximum(abs, cp_r))
        @test maximum(abs, polR_g.data_imag .- cp_i) <= 1e-13 * max(1.0, maximum(abs, cp_i))
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-3 gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            Random.seed!(1104)
            tor = _vt_spec(cfgv, dom); pol = _vt_spec(cfgv, dom)
            _vt_fill!((tor, pol), cfgv, dom)
            mkphys(arch) = (GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfgv, VT_NR),
                            GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfgv, VT_NR),
                            GeoDynamo.allocate_gpu_physical_field(Float64, arch, cfgv, VT_NR))
            # CPU (Array-backend) reference through the SAME GPU code path
            ctor = _vt_to_gpu(tor, cfgv, CPU()); cpol = _vt_to_gpu(pol, cfgv, CPU())
            cvr, cvθ, cvφ = mkphys(CPU())
            GeoDynamo.gpu_vector_spectral_to_physical!(cvr, cvθ, cvφ, ctor, cpol, cfgv,
                lfacv, rscale, D1.data, rinv, bw)
            ctor2 = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, CPU(), cfgv, VT_NR)
            cpol2 = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, CPU(), cfgv, VT_NR)
            GeoDynamo.gpu_vector_physical_to_spectral!(ctor2, cpol2, cvθ, cvφ, cfgv;
                vr = cvr, lfac = lfacv, r2 = r2)

            dev(x) = GeoDynamo.on_architecture(GPU(), x)
            gtor = _vt_to_gpu(tor, cfgv, GPU()); gpol = _vt_to_gpu(pol, cfgv, GPU())
            gvr, gvθ, gvφ = mkphys(GPU())
            GeoDynamo.gpu_vector_spectral_to_physical!(gvr, gvθ, gvφ, gtor, gpol, cfgv,
                dev(lfacv), dev(rscale), dev(Matrix(D1.data)), dev(rinv), bw)
            @test gvr.data isa CUDA.CuArray
            @test gvθ.data isa CUDA.CuArray
            @test gvφ.data isa CUDA.CuArray
            @test isapprox(Array(gvr.data), cvr.data; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gvθ.data), cvθ.data; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gvφ.data), cvφ.data; atol = 1e-12, rtol = 1e-10)

            gtor2 = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, GPU(), cfgv, VT_NR)
            gpol2 = GeoDynamo.allocate_gpu_spectral_field(ComplexF64, GPU(), cfgv, VT_NR)
            GeoDynamo.gpu_vector_physical_to_spectral!(gtor2, gpol2, gvθ, gvφ, cfgv;
                vr = gvr, lfac = dev(lfacv), r2 = dev(r2))
            @test isapprox(Array(gtor2.data_real), ctor2.data_real; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gtor2.data_imag), ctor2.data_imag; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gpol2.data_real), cpol2.data_real; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gpol2.data_imag), cpol2.data_imag; atol = 1e-12, rtol = 1e-10)
        end
    end
end
