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

    function _state_arrays(gst)
        arrs = Any[
            gst.velocity.tor.spec_r, gst.velocity.tor.spec_i,
            gst.velocity.pol.spec_r, gst.velocity.pol.spec_i,
            gst.temperature.spec_r, gst.temperature.spec_i,
        ]
        if gst.magnetic !== nothing
            append!(arrs, Any[gst.magnetic.tor.spec_r, gst.magnetic.tor.spec_i,
                              gst.magnetic.pol.spec_r, gst.magnetic.pol.spec_i])
        end
        if gst.composition !== nothing
            append!(arrs, Any[gst.composition.spec_r, gst.composition.spec_i])
        end
        return arrs
    end

    function _assert_same_state(a, b; atol = 1e-10, rtol = 1e-10)
        aa = _state_arrays(a); bb = _state_arrays(b)
        @test length(aa) == length(bb)
        for i in eachindex(aa)
            @test isapprox(aa[i], bb[i]; atol, rtol)
        end
    end

    @testset "N-step GPU trajectory == N-step CPU (insulating) [LOCAL]" begin
        st = build_small_cpu_state()
        GeoDynamo.solver_step!(st)                        # warm-up
        gst = GeoDynamo.build_gpu_solver_state(st)
        GeoDynamo.gpu_run!(gst, NSTEPS)
        @test all(isfinite, gst.velocity.tor.spec_r)
        @test all(isfinite, gst.temperature.spec_r)
        for _ in 1:NSTEPS
            GeoDynamo.solver_step!(st)
        end
        cfg = st.backend.shtns_config
        nr = st.runtime.outer_core_domain.N
        cpu_match = true
        for (cpu_spec, gr, gi) in [
                (st.fields.temperature.spectral, gst.temperature.spec_r, gst.temperature.spec_i),
                (st.fields.velocity.toroidal,    gst.velocity.tor.spec_r, gst.velocity.tor.spec_i),
                (st.fields.velocity.poloidal,    gst.velocity.pol.spec_r, gst.velocity.pol.spec_i),
                (st.fields.magnetic.toroidal,    gst.magnetic.tor.spec_r, gst.magnetic.tor.spec_i),
                (st.fields.magnetic.poloidal,    gst.magnetic.pol.spec_r, gst.magnetic.pol.spec_i),
                (st.fields.composition.spectral, gst.composition.spec_r, gst.composition.spec_i)]
            cr, ci = GeoDynamo.cpu_spectral_to_dense(cpu_spec, cfg, nr, Float64)
            cpu_match &= isapprox(gr, cr; atol = 1e-8, rtol = 1e-6)
            cpu_match &= isapprox(gi, ci; atol = 1e-8, rtol = 1e-6)
        end
        @test cpu_match
    end

    @testset "step decomposition: gpu_run!(N) == N × gpu_solver_step! [LOCAL]" begin
        st = build_small_cpu_state(); GeoDynamo.solver_step!(st)
        a = GeoDynamo.build_gpu_solver_state(st)
        b = GeoDynamo.build_gpu_solver_state(st)
        GeoDynamo.gpu_run!(a, 3)
        for _ in 1:3
            GeoDynamo.gpu_solver_step!(b)
        end
        _assert_same_state(a, b)
    end

    @testset "output_fn host-gather hook [LOCAL]" begin
        st = build_small_cpu_state(); GeoDynamo.solver_step!(st)
        gst = GeoDynamo.build_gpu_solver_state(st)
        snaps = Tuple{Int, Any}[]
        GeoDynamo.gpu_run!(gst, 4; output_every = 2,
            output_fn = (hs, step) -> push!(snaps, (step, hs)))
        @test first.(snaps) == [2, 4]
        @test length(snaps) == 2
        @test snaps[1][2].temperature.spec_r isa Array
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
        stB = build_small_cpu_state(); GeoDynamo.solver_step!(stB)
        gst = GeoDynamo.build_gpu_solver_state(stB)
        GeoDynamo.gpu_run!(gst, NSTEPS)
        step0 = stA.step
        time0 = stA.time
        GeoDynamo.gpu_run!(stA, NSTEPS)
        @test stA.step == step0 + NSTEPS
        @test stA.time == time0 + NSTEPS * stA.parameters.timestep
        cfg = stA.backend.shtns_config
        nr = stA.runtime.outer_core_domain.N
        cr, ci = GeoDynamo.cpu_spectral_to_dense(stA.fields.velocity.toroidal, cfg, nr, Float64)
        @test isapprox(cr, gst.velocity.tor.spec_r; atol = 1e-10, rtol = 1e-10)
        @test isapprox(ci, gst.velocity.tor.spec_i; atol = 1e-10, rtol = 1e-10)
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
