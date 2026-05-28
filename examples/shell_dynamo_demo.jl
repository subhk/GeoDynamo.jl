#!/usr/bin/env julia

# Minimalistic MHD dynamo simulation in a spherical shell
# Run: julia --project examples/shell_dynamo_demo.jl
# Run with MPI: mpiexecjl -n 4 julia --project examples/shell_dynamo_demo.jl

using GeoDynamo
using GeoDynamo.bcs: DIRICHLET
using Random

function build_model(;
    nr=64,
    nr_inner=16,
    lmax=32,
    mmax=32,
    nlat=64,
    nlon=128,
    radius_ratio=0.35,
    Ek=1e-4,
    Pr=1.0,
    Pm=1.0,
    Sc=1.0,
    Ra=1e6,
    kwargs...,)

    grid = SphericalShellGrid(
        lmax=lmax,
        mmax=mmax,
        nlat=nlat,
        nlon=nlon,
        nr=nr,
        nr_inner=nr_inner,
        r_inner=radius_ratio,
    )

    return GeodynamoModel(
        grid;
        Ek=Ek,
        Pr=Pr,
        Pm=Pm,
        Sc=Sc,
        Ra=Ra,
        velocity_bcs=BoundaryConditions(inner=NoSlip(), outer=NoSlip()),
        temperature_bcs=BoundaryConditions(inner=FixedTemperature(1.0), outer=FixedTemperature(0.0)),
        include_magnetic=true,
        include_composition=false,
        kwargs...,
    )
end

function main(;
    run=true,
    nr=64,
    nr_inner=16,
    lmax=32,
    mmax=32,
    nlat=64,
    nlon=128,
    radius_ratio=0.35,
    Ek=1e-4,
    Pr=1.0,
    Pm=1.0,
    Sc=1.0,
    Ra=1e6,
    Δt=1e-4,
    stop_iteration=500,
    kwargs...,
)
    rank = get_rank()
    nprocs = get_nprocs()

    rank == 0 && println("Setting up spherical shell MHD dynamo (MPI ranks: $nprocs)...")

    model = build_model(
        nr=nr,
        nr_inner=nr_inner,
        lmax=lmax,
        mmax=mmax,
        nlat=nlat,
        nlon=nlon,
        radius_ratio=radius_ratio,
        Ek=Ek,
        Pr=Pr,
        Pm=Pm,
        Sc=Sc,
        Ra=Ra,
        ; kwargs...,
    )

    simulation = Simulation(model; Δt=Δt, stop_iteration=stop_iteration)
    state = simulation.model.state
    GeoDynamo.initialize_fields!(state)
    domain = state.runtime.𝒟ᵒᶜ

    set_boundary_conditions!(state.fields.temperature;
        inner_bc_type = Int(DIRICHLET),
        inner_value   = 1.0,
        outer_bc_type = Int(DIRICHLET),
        outer_value   = 0.0,
    )

    Random.seed!(1234 + rank)
    randomize_scalar_field!(state.fields.temperature; amplitude=0.01, lmax=8, domain=domain)
    randomize_vector_field!(state.fields.velocity; amplitude=1e-5, lmax=6, domain=domain)
    randomize_magnetic_field!(state.fields.magnetic; amplitude=1e-4, lmax=4, domain=domain)

    if run
        rank == 0 && println("Starting simulation...")
        run!(simulation)
        rank == 0 && println("Simulation complete!")
    end

    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
