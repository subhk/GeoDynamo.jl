using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

# Same small configured CPU state as the Phase 5n2 gate, but with the ERK2
# timestepper (staged exponential RK2; velocity poloidal in the Stage-4B
# W-split with φ1-column P-recovery).
function build_small_erk2_state()
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 6, mmax = 6, nlat = 14, nlon = 28, nr = 8, nr_inner = 4,
        radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e5, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
        include_magnetic = true, include_composition = true,
        timestepper = GeoDynamo.ERK2(),
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

@testset "GPU ERK2 — staged exponential step" begin
    NSTEPS = 4

    @testset "builder requires the ERK2 timestepper [LOCAL]" begin
        params = GeoDynamo.SolverParameters(
            geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8,
            nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35)
        st = GeoDynamo.initialize_solver_state(Float64; params = params)
        @test_throws ErrorException GeoDynamo.build_gpu_erk2_state(st)
    end

    @testset "N-step GPU ERK2 trajectory == CPU (full MHD) [LOCAL]" begin
        st = build_small_erk2_state()
        GeoDynamo.solver_step!(st)                      # warm-up (caches + buffers)
        gst = GeoDynamo.build_gpu_solver_state(st)
        erk = GeoDynamo.build_gpu_erk2_state(st)
        cfg = st.backend.shtns_config
        nr = st.runtime.outer_core_domain.N

        cpu_match = true
        for _ in 1:NSTEPS
            GeoDynamo.solver_step!(st)
            GeoDynamo.gpu_erk2_solver_step!(gst, erk)
        end
        for (spec, gr, gi) in [
                (st.fields.temperature.spectral, gst.temperature.spec_r, gst.temperature.spec_i),
                (st.fields.composition.spectral, gst.composition.spec_r, gst.composition.spec_i),
                (st.fields.velocity.toroidal, gst.velocity.tor.spec_r, gst.velocity.tor.spec_i),
                (st.fields.velocity.poloidal, gst.velocity.pol.spec_r, gst.velocity.pol.spec_i),
                (st.fields.magnetic.toroidal, gst.magnetic.tor.spec_r, gst.magnetic.tor.spec_i),
                (st.fields.magnetic.poloidal, gst.magnetic.pol.spec_r, gst.magnetic.pol.spec_i)]
            cr, ci = GeoDynamo.cpu_spectral_to_dense(spec, cfg, nr, Float64)
            cpu_match &= isapprox(cr, gr; atol = 1e-9, rtol = 1e-7)
            cpu_match &= isapprox(ci, gi; atol = 1e-9, rtol = 1e-7)
        end
        @test cpu_match
        @test all(isfinite, gst.velocity.pol.spec_r)
        @test all(isfinite, gst.temperature.spec_r)
    end

    @testset "velocity-only configuration [LOCAL]" begin
        params = GeoDynamo.SolverParameters(
            geometry = :shell, lmax = 6, mmax = 6, nlat = 14, nlon = 28, nr = 8, nr_inner = 4,
            radial_bandwidth = 3, radius_ratio = 0.35,
            Ek = 1e-3, Ra = 1e5, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
            include_magnetic = false, include_composition = false,
            timestepper = GeoDynamo.ERK2(),
            temperature_bcs = GeoDynamo.BoundaryConditions(
                inner = GeoDynamo.FixedTemperature(0.0), outer = GeoDynamo.FixedTemperature(0.0)))
        st = GeoDynamo.initialize_solver_state(Float64; params = params)
        rng = MersenneTwister(3)
        for f in (st.fields.temperature.spectral,
                  st.fields.velocity.toroidal, st.fields.velocity.poloidal)
            parent(f.data_real) .+= 1e-3 .* (rand(rng, size(parent(f.data_real))...) .- 0.5)
            parent(f.data_imag) .+= 1e-3 .* (rand(rng, size(parent(f.data_imag))...) .- 0.5)
        end
        GeoDynamo.solver_step!(st)
        gst = GeoDynamo.build_gpu_solver_state(st)
        erk = GeoDynamo.build_gpu_erk2_state(st)
        cfg = st.backend.shtns_config
        nr = st.runtime.outer_core_domain.N
        GeoDynamo.solver_step!(st)
        GeoDynamo.gpu_erk2_solver_step!(gst, erk)
        ok = true
        for (spec, gr, gi) in [
                (st.fields.temperature.spectral, gst.temperature.spec_r, gst.temperature.spec_i),
                (st.fields.velocity.toroidal, gst.velocity.tor.spec_r, gst.velocity.tor.spec_i),
                (st.fields.velocity.poloidal, gst.velocity.pol.spec_r, gst.velocity.pol.spec_i)]
            cr, ci = GeoDynamo.cpu_spectral_to_dense(spec, cfg, nr, Float64)
            ok &= isapprox(cr, gr; atol = 1e-10, rtol = 1e-8)
            ok &= isapprox(ci, gi; atol = 1e-10, rtol = 1e-8)
        end
        @test ok
    end

    @testset "GPU execution: ERK2 on device [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            st = build_small_erk2_state()
            GeoDynamo.solver_step!(st)
            gst = GeoDynamo.gpu_to_device(GeoDynamo.build_gpu_solver_state(st), GPU())
            erk = GeoDynamo.gpu_to_device(GeoDynamo.build_gpu_erk2_state(st), GPU())
            GeoDynamo.solver_step!(st)
            GeoDynamo.gpu_erk2_solver_step!(gst, erk)
            cfg = st.backend.shtns_config
            nr = st.runtime.outer_core_domain.N
            @test gst.temperature.spec_r isa CUDA.CuArray
            cr, _ = GeoDynamo.cpu_spectral_to_dense(st.fields.temperature.spectral, cfg, nr, Float64)
            @test isapprox(Array(gst.temperature.spec_r), cr; atol = 1e-8, rtol = 1e-6)
        end
    end
end
