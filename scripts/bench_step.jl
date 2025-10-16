#!/usr/bin/env julia

using GeoDynamo
using Printf

function main()
    # Small ball setup for quick benchmarking
    params = GeoDynamoParameters(
        geometry=:ball,
        d_R_outer=1.0,
        i_N=64, i_L=32, i_M=32, i_Th=64, i_Ph=128,
        d_E=1e-4, d_Pr=1.0, d_Pm=1.0, d_Ra=1e6,
        i_B=1,
        d_timestep=1e-4,
        i_maxtstep=1,
    )
    set_parameters!(params)

    state = initialize_simulation(Float64; include_composition=false)
    initialize_fields!(state)

    # Warm-up
    GeoDynamo.compute_vorticity_spectral_full!(state.velocity, state.oc_domain)
    GeoDynamo.compute_all_nonlinear_terms!(state.velocity, state.temperature, nothing, state.magnetic, state.oc_domain)

    # Measure vorticity
    t0 = time()
    GeoDynamo.compute_vorticity_spectral_full!(state.velocity, state.oc_domain)
    t1 = time()

    # Measure nonlinear terms
    t2 = time()
    GeoDynamo.compute_all_nonlinear_terms!(state.velocity, state.temperature, nothing, state.magnetic, state.oc_domain)
    t3 = time()

    @printf("Vorticity step time: %.3f ms\n", 1e3*(t1-t0))
    @printf("Nonlinear step time: %.3f ms\n", 1e3*(t3-t2))
end

main()
