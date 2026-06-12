using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

# The GPUWorkspace contract: every pooled buffer is overwritten before it is
# read, so a POISONED pool (every cached array filled with NaN between steps)
# must produce bit-identical results to a clean run. A violated contract — a
# stale read — surfaces immediately as NaN in the fields.
@testset "GPUWorkspace pool-reuse contract" begin
    function build()
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
            parent(f.data_real) .+= 1e-3 .* (rand(rng, size(parent(f.data_real))...) .- 0.5)
            parent(f.data_imag) .+= 1e-3 .* (rand(rng, size(parent(f.data_imag))...) .- 0.5)
        end
        GeoDynamo.solver_step!(st)
        return st
    end

    poison!(ws) = (for v in values(ws.pool)
        v isa AbstractArray && eltype(v) <: AbstractFloat && fill!(v, NaN)
        v isa AbstractArray && eltype(v) <: Complex && fill!(v, complex(NaN, NaN))
    end)

    @testset "CNAB2 step survives a poisoned pool [LOCAL]" begin
        st = build()
        ref = GeoDynamo.build_gpu_solver_state(st)
        GeoDynamo.gpu_solver_step!(ref)   # populate the pool
        GeoDynamo.gpu_solver_step!(ref)

        st2 = build()
        gst = GeoDynamo.build_gpu_solver_state(st2)
        GeoDynamo.gpu_solver_step!(gst)
        poison!(gst.work)                 # every cached buffer → NaN
        GeoDynamo.gpu_solver_step!(gst)

        @test gst.velocity.pol.spec_r == ref.velocity.pol.spec_r
        @test gst.temperature.spec_r == ref.temperature.spec_r
        @test gst.magnetic.pol.spec_r == ref.magnetic.pol.spec_r
        @test all(isfinite, gst.velocity.pol.spec_r)
    end

    @testset "ERK2 step survives a poisoned pool [LOCAL]" begin
        function build_erk2()
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
                parent(f.data_real) .+= 1e-3 .* (rand(rng, size(parent(f.data_real))...) .- 0.5)
                parent(f.data_imag) .+= 1e-3 .* (rand(rng, size(parent(f.data_imag))...) .- 0.5)
            end
            GeoDynamo.solver_step!(st)
            return st
        end
        st = build_erk2()
        ref = GeoDynamo.build_gpu_solver_state(st)
        erk_ref = GeoDynamo.build_gpu_erk2_state(st)
        GeoDynamo.gpu_erk2_solver_step!(ref, erk_ref)
        GeoDynamo.gpu_erk2_solver_step!(ref, erk_ref)

        st2 = build_erk2()
        gst = GeoDynamo.build_gpu_solver_state(st2)
        erk = GeoDynamo.build_gpu_erk2_state(st2)
        GeoDynamo.gpu_erk2_solver_step!(gst, erk)
        poison!(gst.work)
        GeoDynamo.gpu_erk2_solver_step!(gst, erk)

        @test gst.velocity.pol.spec_r == ref.velocity.pol.spec_r
        @test gst.temperature.spec_r == ref.temperature.spec_r
        @test all(isfinite, gst.velocity.pol.spec_r)
    end
end
