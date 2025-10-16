#!/usr/bin/env julia

using GeoDynamo
using Printf

function step_once!(state)
    # Compute nonlinear terms
    compute_temperature_nonlinear!(state.temperature, state.velocity, state.oc_domain)
    compute_velocity_nonlinear!(state.velocity, state.temperature, state.composition, state.magnetic, state.oc_domain)
    if GeoDynamo.i_B == 1 && state.magnetic !== nothing
        compute_magnetic_nonlinear!(state.magnetic, state.velocity, state.oc_domain, state.ic_domain)
    end
    if state.composition !== nothing
        compute_composition_nonlinear!(state.composition, state.velocity, state.oc_domain)
    end
    # Implicit solve
    apply_master_implicit_step!(state, GeoDynamo.d_timestep)
end

function bench_scheme(ts::Symbol; steps::Int=10)
    # Set scheme and build a fresh state
    params = GeoDynamoParameters(ts_scheme = ts)
    GeoDynamo.set_parameters!(params)
    state = initialize_simulation(Float64; include_composition=false)
    # Warm-up
    step_once!(state)
    # Measure
    t0 = time()
    for _ in 1:steps
        step_once!(state)
    end
    dt = (time() - t0) / steps
    return dt
end

function main()
    steps = parse(Int, get(ENV, "BENCH_STEPS", "10"))
    dt_cnab2 = bench_scheme(:cnab2; steps)
    dt_theta = bench_scheme(:theta; steps)
    @printf("Average step time (CNAB2): %.3f ms\n", 1e3*dt_cnab2)
    @printf("Average step time (theta): %.3f ms\n", 1e3*dt_theta)
end

main()

