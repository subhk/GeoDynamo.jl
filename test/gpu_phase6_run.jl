using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

# Same small configured CPU state as the Phase 5n2 gate (insulating magnetic,
# homogeneous scalar BCs, radial_bandwidth=3 for the 3rd-derivative stencil at nr=8).
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
    rng = MersenneTwister(7)
    for f in (st.fields.temperature.spectral, st.fields.composition.spectral,
              st.fields.velocity.toroidal, st.fields.velocity.poloidal,
              st.fields.magnetic.toroidal, st.fields.magnetic.poloidal)
        dr = parent(f.data_real); di = parent(f.data_imag)
        dr .+= 1e-3 .* (rand(rng, size(dr)...) .- 0.5)
        di .+= 1e-3 .* (rand(rng, size(di)...) .- 0.5)
    end
    return st
end

@testset "GPU Phase 6 — gpu_run! loop + IO host-gather" begin
    NSTEPS = 4

    @testset "N-step GPU trajectory == N-step CPU (insulating) [LOCAL]" begin
        st = build_small_cpu_state()
        cfg = st.backend.shtns_config
        nr = st.runtime.outer_core_domain.N
        GeoDynamo.solver_step!(st)                        # warm-up
        gst = GeoDynamo.build_gpu_solver_state(st)
        GeoDynamo.gpu_run!(gst, NSTEPS)                   # GPU: NSTEPS steps
        for _ in 1:NSTEPS; GeoDynamo.solver_step!(st); end # CPU: NSTEPS steps

        function cmp(name, cpu_spec, gpu_r, gpu_i)
            cr, ci = GeoDynamo.cpu_spectral_to_dense(cpu_spec, cfg, nr, Float64)
            ar = isapprox(cr, gpu_r; atol = 1e-7, rtol = 1e-5)
            ai = isapprox(ci, gpu_i; atol = 1e-7, rtol = 1e-5)
            (ar && ai) || @info "TRAJ diff" field = name maxabs_r = maximum(abs, cr .- gpu_r) maxabs_i = maximum(abs, ci .- gpu_i)
            @test ar && ai
        end
        cmp("temperature", st.fields.temperature.spectral, gst.temperature.spec_r, gst.temperature.spec_i)
        cmp("velocity_tor", st.fields.velocity.toroidal, gst.velocity.tor.spec_r, gst.velocity.tor.spec_i)
        cmp("velocity_pol", st.fields.velocity.poloidal, gst.velocity.pol.spec_r, gst.velocity.pol.spec_i)
        cmp("magnetic_tor", st.fields.magnetic.toroidal, gst.magnetic.tor.spec_r, gst.magnetic.tor.spec_i)
        cmp("magnetic_pol", st.fields.magnetic.poloidal, gst.magnetic.pol.spec_r, gst.magnetic.pol.spec_i)
        cmp("composition", st.fields.composition.spectral, gst.composition.spec_r, gst.composition.spec_i)
    end

    @testset "step decomposition: gpu_run!(N) == N × gpu_solver_step! [LOCAL]" begin
        st = build_small_cpu_state(); GeoDynamo.solver_step!(st)
        a = GeoDynamo.build_gpu_solver_state(st)
        b = GeoDynamo.build_gpu_solver_state(st)
        GeoDynamo.gpu_run!(a, 3)
        for _ in 1:3; GeoDynamo.gpu_solver_step!(b); end
        @test a.velocity.tor.spec_r == b.velocity.tor.spec_r
        @test a.temperature.spec_r == b.temperature.spec_r
        @test a.magnetic.pol.spec_i == b.magnetic.pol.spec_i
    end

    @testset "output_fn host-gather hook [LOCAL]" begin
        st = build_small_cpu_state(); GeoDynamo.solver_step!(st)
        gst = GeoDynamo.build_gpu_solver_state(st)
        snaps = Tuple{Int, Any}[]
        GeoDynamo.gpu_run!(gst, 4; output_every = 2,
            output_fn = (hs, step) -> push!(snaps, (step, hs)))
        @test length(snaps) == 2                          # steps 2 and 4
        @test first.(snaps) == [2, 4]
        # snapshots are host (Array) deep copies, independent of the live state
        s2 = snaps[1][2]
        @test s2.temperature.spec_r isa Array
        @test s2.temperature.spec_r !== gst.temperature.spec_r
        before = copy(s2.temperature.spec_r)
        GeoDynamo.gpu_solver_step!(gst)                   # advance live state further
        @test s2.temperature.spec_r == before            # snapshot unchanged
    end

    @testset "nsteps=0 no-op + arg guard [LOCAL]" begin
        st = build_small_cpu_state(); GeoDynamo.solver_step!(st)
        gst = GeoDynamo.build_gpu_solver_state(st)
        before = copy(gst.velocity.tor.spec_r)
        GeoDynamo.gpu_run!(gst, 0)
        @test gst.velocity.tor.spec_r == before           # no step taken
        @test_throws ArgumentError GeoDynamo.gpu_run!(gst, -1)
    end

    @testset "gpu_run!(::SolverState) runs GPU + syncs back == CPU [LOCAL]" begin
        stA = build_small_cpu_state(); GeoDynamo.solver_step!(stA)   # warm-up
        stB = build_small_cpu_state(); GeoDynamo.solver_step!(stB)   # identical warm-up
        cfg = stA.backend.shtns_config; nr = stA.runtime.outer_core_domain.N
        step0 = stA.step; t0 = stA.time
        GeoDynamo.gpu_run!(stA, NSTEPS)                              # GPU path + sync-back into stA
        for _ in 1:NSTEPS; GeoDynamo.solver_step!(stB); end          # CPU path
        # stA's spectral fields (synced from the GPU run) match the CPU trajectory
        for (fa, fb) in [(stA.fields.temperature.spectral, stB.fields.temperature.spectral),
                         (stA.fields.velocity.toroidal,    stB.fields.velocity.toroidal),
                         (stA.fields.velocity.poloidal,    stB.fields.velocity.poloidal),
                         (stA.fields.magnetic.toroidal,    stB.fields.magnetic.toroidal),
                         (stA.fields.composition.spectral, stB.fields.composition.spectral)]
            ar, ai = GeoDynamo.cpu_spectral_to_dense(fa, cfg, nr, Float64)
            br, bi = GeoDynamo.cpu_spectral_to_dense(fb, cfg, nr, Float64)
            @test isapprox(ar, br; atol = 1e-7, rtol = 1e-5)
            @test isapprox(ai, bi; atol = 1e-7, rtol = 1e-5)
        end
        @test stA.step == step0 + NSTEPS                            # step/time advanced
        @test stA.time ≈ t0 + NSTEPS * stA.parameters.timestep
    end

    @testset "dense_to_cpu_spectral! roundtrip + sync_gpu_state_to_cpu! [LOCAL]" begin
        st = build_small_cpu_state(); GeoDynamo.solver_step!(st)
        cfg = st.backend.shtns_config; nr = st.runtime.outer_core_domain.N
        # write-back roundtrip: extract → perturb a stored mode (l=1,m=0 → slot [2,1]) → write back → re-extract
        f = st.fields.temperature.spectral
        dr, di = GeoDynamo.cpu_spectral_to_dense(f, cfg, nr, Float64)
        drn = copy(dr); drn[2, 1, :] .+= 5.0
        GeoDynamo.dense_to_cpu_spectral!(f, drn, di, cfg, nr)
        dr3, di3 = GeoDynamo.cpu_spectral_to_dense(f, cfg, nr, Float64)
        @test dr3 == drn                              # stored modes roundtrip exactly (empty slots 0 in both)
        @test di3 == di
        @test dr3[2, 1, :] != dr[2, 1, :]             # write-back actually changed the field
        # sync_gpu_state_to_cpu!: perturb the device bundle, sync, confirm the CPU field reflects it
        gst = GeoDynamo.build_gpu_solver_state(st)
        gst.velocity.tor.spec_r[2, 1, 1] += 7.0
        GeoDynamo.sync_gpu_state_to_cpu!(st, gst)
        vr, _ = GeoDynamo.cpu_spectral_to_dense(st.fields.velocity.toroidal, cfg, nr, Float64)
        @test vr[2, 1, 1] == gst.velocity.tor.spec_r[2, 1, 1]
    end

    @testset "GPU execution: device run + GPU≈CPU trajectory [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            st = build_small_cpu_state()
            cfg = st.backend.shtns_config
            nr = st.runtime.outer_core_domain.N
            GeoDynamo.solver_step!(st)
            gst_gpu = GeoDynamo.gpu_to_device(GeoDynamo.build_gpu_solver_state(st), GPU())
            GeoDynamo.gpu_run!(gst_gpu, NSTEPS)
            for _ in 1:NSTEPS; GeoDynamo.solver_step!(st); end
            @test gst_gpu.temperature.spec_r isa CUDA.CuArray
            for (cpu_spec, gr, gi) in [
                    (st.fields.temperature.spectral, gst_gpu.temperature.spec_r, gst_gpu.temperature.spec_i),
                    (st.fields.velocity.toroidal,    gst_gpu.velocity.tor.spec_r, gst_gpu.velocity.tor.spec_i),
                    (st.fields.magnetic.poloidal,    gst_gpu.magnetic.pol.spec_r, gst_gpu.magnetic.pol.spec_i),
                    (st.fields.composition.spectral, gst_gpu.composition.spec_r, gst_gpu.composition.spec_i)]
                cr, ci = GeoDynamo.cpu_spectral_to_dense(cpu_spec, cfg, nr, Float64)
                @test isapprox(Array(gr), cr; atol = 1e-7, rtol = 1e-5)
                @test isapprox(Array(gi), ci; atol = 1e-7, rtol = 1e-5)
            end
        end
    end
end
