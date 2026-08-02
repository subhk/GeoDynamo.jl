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

    # Full-MHD trajectory parity on the Array backend. This is the [LOCAL] twin of
    # the [GPU-BOX] gate at the bottom of this file: identical math, no device
    # transfer, so it runs everywhere and pins the loop + history rollover.
    @testset "N-step GPU trajectory == N-step CPU (insulating) [LOCAL]" begin
        st = build_small_cpu_state()
        cfg = st.backend.shtns_config
        nr = st.runtime.outer_core_domain.N
        GeoDynamo.solver_step!(st)                       # warm-up (history + lag buffers)
        gst = GeoDynamo.build_gpu_solver_state(st)
        GeoDynamo.gpu_run!(gst, NSTEPS)
        for _ in 1:NSTEPS; GeoDynamo.solver_step!(st); end
        for (cpu_spec, gr, gi) in [
                (st.fields.temperature.spectral, gst.temperature.spec_r, gst.temperature.spec_i),
                (st.fields.velocity.toroidal,    gst.velocity.tor.spec_r, gst.velocity.tor.spec_i),
                (st.fields.velocity.poloidal,    gst.velocity.pol.spec_r, gst.velocity.pol.spec_i),
                (st.fields.magnetic.toroidal,    gst.magnetic.tor.spec_r, gst.magnetic.tor.spec_i),
                (st.fields.magnetic.poloidal,    gst.magnetic.pol.spec_r, gst.magnetic.pol.spec_i),
                (st.fields.composition.spectral, gst.composition.spec_r, gst.composition.spec_i)]
            cr, ci = GeoDynamo.cpu_spectral_to_dense(cpu_spec, cfg, nr, Float64)
            @test isapprox(gr, cr; atol = GPU_LOCAL_ATOL, rtol = GPU_LOCAL_RTOL)
            @test isapprox(gi, ci; atol = GPU_LOCAL_ATOL, rtol = GPU_LOCAL_RTOL)
        end
    end

    @testset "step decomposition: gpu_run!(N) == N × gpu_solver_step! [LOCAL]" begin
        st = build_small_cpu_state(); GeoDynamo.solver_step!(st)
        looped = GeoDynamo.build_gpu_solver_state(st)
        manual = GeoDynamo.build_gpu_solver_state(st)     # independent bundle, same state
        GeoDynamo.gpu_run!(looped, NSTEPS)
        for _ in 1:NSTEPS; GeoDynamo.gpu_solver_step!(manual); end
        # gpu_run! is a plain repeat of gpu_solver_step! — bit-identical, not just close.
        @test looped.temperature.spec_r == manual.temperature.spec_r
        @test looped.velocity.pol.spec_r == manual.velocity.pol.spec_r
        @test looped.magnetic.tor.spec_i == manual.magnetic.tor.spec_i
        @test looped.composition.spec_r == manual.composition.spec_r
    end

    @testset "output_fn host-gather hook [LOCAL]" begin
        st = build_small_cpu_state(); GeoDynamo.solver_step!(st)
        gst = GeoDynamo.build_gpu_solver_state(st)
        seen = Tuple{Int, Array{Float64, 3}}[]
        GeoDynamo.gpu_run!(gst, 4; output_every = 2,
            output_fn = (host, step) -> push!(seen, (step, copy(host.temperature.spec_r))))
        @test [s for (s, _) in seen] == [2, 4]            # fires on multiples only
        @test seen[1][2] != seen[2][2]                    # snapshots are distinct steps
        # the gather is a host-side deep copy: mutating the live bundle must not
        # touch an already-handed-out snapshot
        snap = seen[end][2]
        before = copy(snap)
        gst.temperature.spec_r[2, 1, 1] += 11.0
        @test snap == before
        # ...and no output_fn without output_every
        empty!(seen)
        GeoDynamo.gpu_run!(gst, 2; output_fn = (host, step) -> push!(seen, (step, copy(host.temperature.spec_r))))
        @test isempty(seen)
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
        gpu_st = build_small_cpu_state(); GeoDynamo.solver_step!(gpu_st)
        ref_st = build_small_cpu_state(); GeoDynamo.solver_step!(ref_st)
        cfg = ref_st.backend.shtns_config
        nr = ref_st.runtime.outer_core_domain.N
        step0, time0 = gpu_st.step, gpu_st.time

        GeoDynamo.gpu_run!(gpu_st, NSTEPS)                # device loop + sync back
        for _ in 1:NSTEPS; GeoDynamo.solver_step!(ref_st); end

        for (a, b) in [
                (ref_st.fields.temperature.spectral, gpu_st.fields.temperature.spectral),
                (ref_st.fields.velocity.poloidal,    gpu_st.fields.velocity.poloidal),
                (ref_st.fields.magnetic.toroidal,    gpu_st.fields.magnetic.toroidal)]
            ar, ai = GeoDynamo.cpu_spectral_to_dense(a, cfg, nr, Float64)
            br, bi = GeoDynamo.cpu_spectral_to_dense(b, cfg, nr, Float64)
            @test isapprox(br, ar; atol = GPU_LOCAL_ATOL, rtol = GPU_LOCAL_RTOL)
            @test isapprox(bi, ai; atol = GPU_LOCAL_ATOL, rtol = GPU_LOCAL_RTOL)
        end
        # Both clocks advance: the public pair AND the runtime timestep_state that
        # get_current_simulation_time / the ERK2 diagnostics read.
        @test gpu_st.step == step0 + NSTEPS
        @test gpu_st.time ≈ time0 + NSTEPS * gpu_st.parameters.timestep
        @test gpu_st.runtime.timestep_state.step == gpu_st.step
        @test gpu_st.runtime.timestep_state.time ≈ gpu_st.time
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
