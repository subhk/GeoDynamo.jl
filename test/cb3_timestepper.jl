using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

function build_small_cb3_state(; gpu = false)
    params = GeoDynamo.SolverParameters(
        geometry = :shell,
        lmax = 4,
        mmax = 4,
        nlat = 10,
        nlon = 20,
        nr = 8,
        nr_inner = 4,
        radial_bandwidth = 3,
        radius_ratio = 0.35,
        Ek = 1e-3,
        Ra = 1e5,
        Pm = 1.0,
        Pr = 1.0,
        timestep = 5e-5,
        include_magnetic = true,
        include_composition = true,
        timestepper = GeoDynamo.RungeKutta3(),
        temperature_bcs = GeoDynamo.BoundaryConditions(
            inner = GeoDynamo.FixedTemperature(0.0),
            outer = GeoDynamo.FixedTemperature(0.0),
        ),
        composition_bcs = GeoDynamo.BoundaryConditions(
            inner = GeoDynamo.FixedTemperature(0.0),
            outer = GeoDynamo.FixedTemperature(0.0),
        ),
    )
    st = GeoDynamo.initialize_solver_state(Float64; params)
    rng = MersenneTwister(11)
    for f in (st.fields.temperature.spectral, st.fields.composition.spectral,
              st.fields.velocity.toroidal, st.fields.velocity.poloidal,
              st.fields.magnetic.toroidal, st.fields.magnetic.poloidal)
        parent(f.data_real) .+= 1e-4 .* (rand(rng, size(parent(f.data_real))...) .- 0.5)
        parent(f.data_imag) .+= 1e-4 .* (rand(rng, size(parent(f.data_imag))...) .- 0.5)
    end
    return st
end

@testset "Cavaglieri-Bewley 2N-storage IMEX RK3 timestepper" begin
    @test GeoDynamo._timestepper_scheme(GeoDynamo.RungeKutta3()) === :cb3
    @test GeoDynamo._timestepper_from_scheme(:cb3, nothing, nothing, nothing) isa GeoDynamo.RungeKutta3

    st = build_small_cb3_state()
    GeoDynamo.solver_step!(st)
    @test st.step == 1
    @test st.time == st.parameters.timestep
    @test all(isfinite, parent(st.fields.temperature.spectral.data_real))
    @test all(isfinite, parent(st.fields.velocity.toroidal.data_real))

    # RK3 per-substage implicit operators are cached (built once per (stage, dt))
    # instead of rebuilt + refactorized every substage. After a step all three stage
    # slots are populated and keyed to the current dt.
    @test st.timestep_caches.cb3_built_dt == st.parameters.timestep
    @test all(!isnothing, st.timestep_caches.cb3_stage_matrices)
    @test all(!isnothing, st.timestep_caches.cb3_poloidal_split)
    # A second step reuses the cached operators (no rebuild) and stays finite.
    GeoDynamo.solver_step!(st)
    @test st.step == 2
    @test all(isfinite, parent(st.fields.temperature.spectral.data_real))
    @test all(isfinite, parent(st.fields.velocity.poloidal.data_real))

    st_cpu = build_small_cb3_state()
    GeoDynamo.solver_step!(st_cpu)
    gst = GeoDynamo.build_gpu_solver_state(st_cpu)
    GeoDynamo.solver_step!(st_cpu)
    GeoDynamo.gpu_cb3_solver_step!(gst)

    cfg = st_cpu.backend.shtns_config
    nr = st_cpu.runtime.outer_core_domain.N
    ok = true
    for (spec, gr, gi) in [
            (st_cpu.fields.temperature.spectral, gst.temperature.spec_r, gst.temperature.spec_i),
            (st_cpu.fields.composition.spectral, gst.composition.spec_r, gst.composition.spec_i),
            (st_cpu.fields.velocity.toroidal, gst.velocity.tor.spec_r, gst.velocity.tor.spec_i),
            (st_cpu.fields.velocity.poloidal, gst.velocity.pol.spec_r, gst.velocity.pol.spec_i),
            (st_cpu.fields.magnetic.toroidal, gst.magnetic.tor.spec_r, gst.magnetic.tor.spec_i),
            (st_cpu.fields.magnetic.poloidal, gst.magnetic.pol.spec_r, gst.magnetic.pol.spec_i)]
        cr, ci = GeoDynamo.cpu_spectral_to_dense(spec, cfg, nr, Float64)
        ok &= isapprox(cr, gr; atol = 1e-8, rtol = 1e-6)
        ok &= isapprox(ci, gi; atol = 1e-8, rtol = 1e-6)
    end
    @test ok
end
