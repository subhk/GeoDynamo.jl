#!/usr/bin/env julia

# Minimalistic MHD dynamo simulation in a ball (solid sphere)
# Run: julia --project examples/ball_mhd_demo.jl

using GeoDynamo
using GeoDynamo.bcs: DIRICHLET
using Random

function build_model(;
        nr = 64,
        lmax = 32,
        mmax = 32,
        nlat = 64,
        nlon = 128,
        Ek = 1e-4,
        Pr = 1.0,
        Pm = 1.0,
        Sc = 1.0,
        Ra = 1e6,
        kwargs...
)
    grid = SphericalBallGrid(
        lmax = lmax,
        mmax = mmax,
        nlat = nlat,
        nlon = nlon,
        nr = nr
    )

    return GeodynamoModel(
        grid;
        Ek = Ek,
        Pr = Pr,
        Pm = Pm,
        Sc = Sc,
        Ra = Ra,
        velocity_bcs = BoundaryConditions(inner = NoSlip(), outer = NoSlip()),
        temperature_bcs = BoundaryConditions(inner = FixedTemperature(1.0), outer = FixedTemperature(0.0)),
        include_magnetic = true,
        include_composition = false,
        kwargs...
    )
end

function main(;
        run = true,
        nr = 64,
        lmax = 32,
        mmax = 32,
        nlat = 64,
        nlon = 128,
        Ek = 1e-4,
        Pr = 1.0,
        Pm = 1.0,
        Sc = 1.0,
        Ra = 1e6,
        dt = 1e-4,
        stop_iteration = 500,
        kwargs...
)
    println("Setting up ball MHD dynamo simulation...")

    model = build_model(
        nr = nr,
        lmax = lmax,
        mmax = mmax,
        nlat = nlat,
        nlon = nlon,
        Ek = Ek,
        Pr = Pr,
        Pm = Pm,
        Sc = Sc,
        Ra = Ra,
        ; kwargs...
    )

    simulation = Simulation(model; dt = dt, stop_iteration = stop_iteration)
    state = simulation.model.state
    GeoDynamo.initialize_fields!(state)
    domain = state.runtime.𝒟ᵒᶜ

    set_boundary_conditions!(state.fields.temperature;
        inner_bc_type = Int(DIRICHLET),
        inner_value = 1.0,
        outer_bc_type = Int(DIRICHLET),
        outer_value = 0.0
    )

    Random.seed!(1234)
    randomize_scalar_field!(state.fields.temperature; amplitude = 0.01, lmax = 8, domain = domain)
    randomize_vector_field!(state.fields.velocity; amplitude = 1e-5, lmax = 6, domain = domain)
    randomize_magnetic_field!(state.fields.magnetic; amplitude = 1e-4, lmax = 4, domain = domain)

    if run
        println("Starting simulation...")
        run!(simulation)
        println("Simulation complete!")
    end

    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
