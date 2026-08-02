using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

# =============================================================================
# GPU Phase 5n — gpu_solver_step! orchestration.
#
# Numeric GPU≈CPU parity for the full step is gated elsewhere and is NOT
# duplicated here:
#   * gpu_phase5n2_device_state.jl — single step, full MHD, real SolverState
#   * gpu_phase6_run.jl            — N-step trajectory + gpu_run! decomposition
#   * gpu_bc_combo_parity.jl       — velocity codes 1–4 × scalar DD/DN/ND/NN
#   * gpu_erk2_step.jl             — ERK2 staged step
#
# What is left to this file is the ORCHESTRATION contract: the step must run,
# and stay correct, with the optional subsystems switched off — the branches the
# always-on fixtures above never take.
# =============================================================================

function build_gated_cpu_state(; magnetic::Bool, composition::Bool)
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 6, mmax = 6, nlat = 14, nlon = 28, nr = 8, nr_inner = 4,
        radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e5, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
        include_magnetic = magnetic, include_composition = composition,
        temperature_bcs = GeoDynamo.BoundaryConditions(
            inner = GeoDynamo.FixedTemperature(0.0), outer = GeoDynamo.FixedTemperature(0.0)),
        composition_bcs = GeoDynamo.BoundaryConditions(
            inner = GeoDynamo.FixedTemperature(0.0), outer = GeoDynamo.FixedTemperature(0.0)))
    st = GeoDynamo.initialize_solver_state(Float64; params = params)
    rng = MersenneTwister(7)
    specs = Any[st.fields.temperature.spectral,
                st.fields.velocity.toroidal, st.fields.velocity.poloidal]
    composition && push!(specs, st.fields.composition.spectral)
    magnetic && append!(specs, (st.fields.magnetic.toroidal, st.fields.magnetic.poloidal))
    for f in specs
        dr = parent(f.data_real); di = parent(f.data_imag)
        dr .+= 1e-3 .* (rand(rng, size(dr)...) .- 0.5)
        di .+= 1e-3 .* (rand(rng, size(di)...) .- 0.5)
    end
    return st
end

@testset "GPU Phase 5n — gpu_solver_step! orchestration" begin
    NSTEPS = 3

    @testset "gating: no magnetic / no composition [LOCAL]" begin
        for (magnetic, composition) in ((false, true), (true, false), (false, false))
            label = "magnetic=$magnetic composition=$composition"
            @testset "$label" begin
                st = build_gated_cpu_state(; magnetic = magnetic, composition = composition)
                cfg = st.backend.shtns_config
                nr = st.runtime.outer_core_domain.N
                GeoDynamo.solver_step!(st)                      # warm-up
                gst = GeoDynamo.build_gpu_solver_state(st)

                # the disabled subsystem is absent from the bundle, not zero-filled
                @test (gst.magnetic === nothing) == !magnetic
                @test (gst.composition === nothing) == !composition

                for _ in 1:NSTEPS
                    GeoDynamo.gpu_solver_step!(gst)
                    GeoDynamo.solver_step!(st)
                end

                pairs = Any[(st.fields.temperature.spectral,
                             gst.temperature.spec_r, gst.temperature.spec_i),
                            (st.fields.velocity.toroidal,
                             gst.velocity.tor.spec_r, gst.velocity.tor.spec_i),
                            (st.fields.velocity.poloidal,
                             gst.velocity.pol.spec_r, gst.velocity.pol.spec_i)]
                magnetic && push!(pairs,
                    (st.fields.magnetic.toroidal, gst.magnetic.tor.spec_r, gst.magnetic.tor.spec_i))
                composition && push!(pairs,
                    (st.fields.composition.spectral, gst.composition.spec_r, gst.composition.spec_i))
                for (cpu_spec, gr, gi) in pairs
                    cr, ci = GeoDynamo.cpu_spectral_to_dense(cpu_spec, cfg, nr, Float64)
                    @test isapprox(gr, cr; atol = GPU_LOCAL_ATOL, rtol = GPU_LOCAL_RTOL)
                    @test isapprox(gi, ci; atol = GPU_LOCAL_ATOL, rtol = GPU_LOCAL_RTOL)
                end
            end
        end
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5n gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            # Device execution of the gated (velocity + temperature only) step:
            # the always-on device gates live in Phase 5n2 / Phase 6.
            st = build_gated_cpu_state(; magnetic = false, composition = false)
            cfg = st.backend.shtns_config
            nr = st.runtime.outer_core_domain.N
            GeoDynamo.solver_step!(st)
            gst = GeoDynamo.gpu_to_device(GeoDynamo.build_gpu_solver_state(st), GPU())
            GeoDynamo.gpu_solver_step!(gst)
            GeoDynamo.solver_step!(st)
            @test gst.temperature.spec_r isa CUDA.CuArray
            @test gst.magnetic === nothing
            cr, ci = GeoDynamo.cpu_spectral_to_dense(st.fields.temperature.spectral, cfg, nr, Float64)
            @test isapprox(Array(gst.temperature.spec_r), cr; atol = 1e-7, rtol = 1e-5)
            @test isapprox(Array(gst.temperature.spec_i), ci; atol = 1e-7, rtol = 1e-5)
        end
    end
end
