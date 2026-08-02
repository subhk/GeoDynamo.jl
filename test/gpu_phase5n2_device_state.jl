using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

# Build a small configured CPU SolverState with homogeneous scalar BCs (so the
# velocity/temperature/composition boundary perturbations are zero — the GPU
# device-state carries zero BC rows) and an insulating magnetic inner core.
# radial_bandwidth=3 is the minimum the 3rd-derivative stencil needs at nr=8.
function build_small_cpu_state()
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 6, mmax = 6, nlat = 14, nlon = 28, nr = 8, nr_inner = 4,
        radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e5, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
        include_magnetic = true, include_composition = true,
        temperature_bcs = GeoDynamo.BoundaryConditions(
            inner = GeoDynamo.FixedTemperature(0.0), outer = GeoDynamo.FixedTemperature(0.0)),
        composition_bcs = GeoDynamo.BoundaryConditions(
            inner = GeoDynamo.FixedTemperature(0.0), outer = GeoDynamo.FixedTemperature(0.0)))
    st = GeoDynamo.initialize_solver_state(Float64; params = params)
    # Seed a small reproducible perturbation so the step is non-trivial. Magnetic
    # gets a perturbation too (else B/J stay zero → no Lorentz / no induction).
    rng = MersenneTwister(7)
    seed_fields = (st.fields.temperature.spectral, st.fields.composition.spectral,
        st.fields.velocity.toroidal, st.fields.velocity.poloidal,
        st.fields.magnetic.toroidal, st.fields.magnetic.poloidal)
    for f in seed_fields
        dr = parent(f.data_real)
        di = parent(f.data_imag)
        dr .+= 1e-3 .* (rand(rng, size(dr)...) .- 0.5)
        di .+= 1e-3 .* (rand(rng, size(di)...) .- 0.5)
    end
    return st
end

@testset "GPU Phase 5n2 — Device-State Builder + GPU≈CPU Gate" begin
    st = build_small_cpu_state()
    cfg = st.backend.shtns_config
    nl = cfg.lmax + 1
    nm = cfg.mmax + 1
    nr = st.runtime.outer_core_domain.N

    @testset "cpu_spectral_to_dense roundtrip + shape [LOCAL]" begin
        dr, di = GeoDynamo.cpu_spectral_to_dense(st.fields.temperature.spectral, cfg, nr, Float64)
        @test size(dr) == (nl, nm, nr) && size(di) == (nl, nm, nr)
        ok = true
        pr = parent(st.fields.temperature.spectral.data_real)
        pim = parent(st.fields.temperature.spectral.data_imag)
        for lm_idx in 1:cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            l = cfg.l_values[lm_idx]
            m = cfg.m_values[lm_idx]
            for k in 1:nr
                ok &= (dr[l + 1, m + 1, k] == GeoDynamo.local_spectral_value(pr, slot, k))
                ok &= (di[l + 1, m + 1, k] == GeoDynamo.local_spectral_value(pim, slot, k))
            end
        end
        @test ok
    end

    @testset "build_gpu_solver_state extracts matrices/factors [LOCAL]" begin
        gst = GeoDynamo.build_gpu_solver_state(st)
        # lin/lu for temperature match the CPU implicit matrices per degree
        mset = st.implicit_matrices[:temperature]
        for (i, l) in enumerate(mset.l_values)
            @test gst.temperature.lin[:, :, l + 1] == mset.linear_matrices[i].data
            @test gst.temperature.lu[:, :, l + 1] == mset.factorizations[i].lu
        end
        # velocity lin/lu too (different mass coefficient baked in)
        vset = st.implicit_matrices[:velocity_tor]
        for (i, l) in enumerate(vset.l_values)
            @test gst.velocity.tor.lin[:, :, l + 1] == vset.linear_matrices[i].data
            @test gst.velocity.tor.lu[:, :, l + 1] == vset.factorizations[i].lu
        end
        # coupling factors
        p = st.parameters
        # Scalar mass coefficient is 1 (diffusivity is folded into the implicit
        # operator L, which the GPU reuses from the CPU matrices) ⇒ inv_dt = 1/dt,
        # NOT (Pm/Pr)/dt. (Pm=Pr=1 here, so this only changed the intent, not the value.)
        @test gst.inv_dt_temp ≈ 1.0 / p.timestep
        @test gst.inv_dt_comp ≈ 1.0 / p.timestep
        @test gst.inv_dt_vel ≈ p.Ek / p.timestep
        @test gst.inv_dt_mag ≈ 1.0 / p.timestep
        @test gst.thermal_factor ≈ (p.Pm / p.Pr) * p.Ra
        @test gst.comp_factor ≈ (p.Pm / p.Sc) * p.RaC
        @test gst.lorentz_coeff ≈ 1.0 / p.Pm
        @test gst.nlops_vel.E ≈ p.Ek
        @test gst.linear_weight ≈ 0.5
        # operators present + right length / orientation
        @test length(gst.nlops_vel.sinθ) == cfg.nlat
        @test length(gst.nlops_vel.cosθ) == cfg.nlat
        @test gst.nlops_vel.sinθ == Float64[st.fields.velocity.coriolis_factors[1, i] for i in 1:cfg.nlat]
        @test gst.nlops_vel.cosθ == Float64[st.fields.velocity.coriolis_factors[2, i] for i in 1:cfg.nlat]
        @test length(gst.r_vec) == nr
        @test gst.r_vec == Float64[st.runtime.outer_core_domain.r[k, 4] for k in 1:nr]
        @test gst.rinv == Float64[st.runtime.outer_core_domain.r[k, 3] for k in 1:nr]
        @test gst.nlops_vel.rinv2 == Float64[st.runtime.outer_core_domain.r[k, 2] for k in 1:nr]
        @test size(gst.nlops_vel.d1) == (2 * gst.bw + 1, nr)
        @test gst.nlops_vel.d1 == Array{Float64}(st.fields.velocity.∂r.data)
        # influence pack present + correct shapes
        @test size(gst.influence.Gre_b) == (nr, 2, nl)
        @test size(gst.influence.invG_b) == (2, 2, nl)
        # physical lag buffers extracted
        @test size(gst.T_phys) == (cfg.nlat, cfg.nlon, nr)
        @test gst.B_r !== nothing && gst.J_r !== nothing && gst.C_phys !== nothing
    end

    @testset "gpu_to_device round-trips on CPU backend [LOCAL]" begin
        gst = GeoDynamo.build_gpu_solver_state(st)
        g2 = GeoDynamo.gpu_to_device(gst, CPU())
        # arrays equal in value but independent copies (deep move), scalars/config preserved
        @test g2.temperature.spec_r == gst.temperature.spec_r
        @test g2.temperature.spec_r !== gst.temperature.spec_r          # not aliased
        @test g2.nlops_vel.d1 == gst.nlops_vel.d1 && g2.nlops_vel.E == gst.nlops_vel.E
        @test g2.influence.Gre_b == gst.influence.Gre_b
        @test g2.B_r == gst.B_r && g2.T_phys == gst.T_phys
        @test g2.config === gst.config && g2.bw == gst.bw
        # mutating the copy does not touch the original
        g2.temperature.spec_r[1, 1, 1] += 1.0
        @test g2.temperature.spec_r[1, 1, 1] != gst.temperature.spec_r[1, 1, 1]
    end

    @testset "GPU≈CPU full step (insulating) [LOCAL]" begin
        # Single-step parity from a warmed state, full MHD. Multi-step trajectories
        # are gated by gpu_phase6_run.jl; BC-combination coverage by
        # gpu_bc_combo_parity.jl; ERK2 by gpu_erk2_step.jl.
        sts = build_small_cpu_state()
        GeoDynamo.solver_step!(sts)                      # warm-up
        gst1 = GeoDynamo.build_gpu_solver_state(sts)
        GeoDynamo.gpu_solver_step!(gst1)
        GeoDynamo.solver_step!(sts)
        cfgs = sts.backend.shtns_config
        nrs = sts.runtime.outer_core_domain.N
        for (cpu_spec, gr, gi) in [
                (sts.fields.temperature.spectral, gst1.temperature.spec_r, gst1.temperature.spec_i),
                (sts.fields.velocity.toroidal,    gst1.velocity.tor.spec_r, gst1.velocity.tor.spec_i),
                (sts.fields.velocity.poloidal,    gst1.velocity.pol.spec_r, gst1.velocity.pol.spec_i),
                (sts.fields.magnetic.toroidal,    gst1.magnetic.tor.spec_r, gst1.magnetic.tor.spec_i),
                (sts.fields.magnetic.poloidal,    gst1.magnetic.pol.spec_r, gst1.magnetic.pol.spec_i),
                (sts.fields.composition.spectral, gst1.composition.spec_r, gst1.composition.spec_i)]
            cr, ci = GeoDynamo.cpu_spectral_to_dense(cpu_spec, cfgs, nrs, Float64)
            @test isapprox(gr, cr; atol = GPU_LOCAL_ATOL, rtol = GPU_LOCAL_RTOL)
            @test isapprox(gi, ci; atol = GPU_LOCAL_ATOL, rtol = GPU_LOCAL_RTOL)
        end
    end

    @testset "GPU≈CPU full step on GPU [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            # Compare the CUDA device execution against the same dense GPU path
            # on Array; CPU-solver parity is gated by the [LOCAL] testset above.
            stb = build_small_cpu_state()
            GeoDynamo.solver_step!(stb)                          # warm-up
            gst_box = GeoDynamo.build_gpu_solver_state(stb)      # Array bundle from warmed state
            gst_gpu = GeoDynamo.gpu_to_device(gst_box, GPU())    # deep-move to device
            GeoDynamo.gpu_solver_step!(gst_box)                  # dense Array path
            GeoDynamo.gpu_solver_step!(gst_gpu)                  # GPU step n+1 (CuArray)
            @test gst_gpu.temperature.spec_r isa CUDA.CuArray
            @test gst_gpu.magnetic.tor.spec_r isa CUDA.CuArray
            for (ga, ca) in zip(
                    (gst_gpu.temperature.spec_r, gst_gpu.temperature.spec_i,
                     gst_gpu.velocity.tor.spec_r, gst_gpu.velocity.tor.spec_i,
                     gst_gpu.velocity.pol.spec_r, gst_gpu.velocity.pol.spec_i,
                     gst_gpu.magnetic.tor.spec_r, gst_gpu.magnetic.tor.spec_i,
                     gst_gpu.magnetic.pol.spec_r, gst_gpu.magnetic.pol.spec_i,
                     gst_gpu.composition.spec_r, gst_gpu.composition.spec_i),
                    (gst_box.temperature.spec_r, gst_box.temperature.spec_i,
                     gst_box.velocity.tor.spec_r, gst_box.velocity.tor.spec_i,
                     gst_box.velocity.pol.spec_r, gst_box.velocity.pol.spec_i,
                     gst_box.magnetic.tor.spec_r, gst_box.magnetic.tor.spec_i,
                     gst_box.magnetic.pol.spec_r, gst_box.magnetic.pol.spec_i,
                     gst_box.composition.spec_r, gst_box.composition.spec_i))
                @test isapprox(Array(ga), ca; atol = 1e-8, rtol = 1e-6)
            end
        end
    end
end
