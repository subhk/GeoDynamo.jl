#!/usr/bin/env julia

# Minimalistic MHD dynamo simulation in a ball (solid sphere)
# Run: julia --project examples/ball_mhd_demo.jl

using GeoDynamo
using GeoDynamo.bcs: DIRICHLET
using Random

println("Setting up ball MHD dynamo simulation...")

# 1) Set parameters
params = GeoDynamoParameters(
    geometry   = :ball,
    i_N        = 64,      # radial points
    i_L        = 32,      # lmax
    i_M        = 32,      # mmax
    i_Th       = 64,      # nlat
    i_Ph       = 128,     # nlon
    i_KL       = 4,       # FD bandwidth
    d_E        = 1e-4,    # Ekman number
    d_Pr       = 1.0,     # Prandtl number
    d_Pm       = 1.0,     # Magnetic Prandtl number
    d_Ra       = 1e6,     # Rayleigh number
    i_B        = 1,       # Enable magnetic field
    d_R_outer  = 1.0,     # Outer radius
    d_timestep = 1e-4,
    i_maxtstep = 500,
    i_save_rate2 = 50,
    ts_scheme  = :erk2,   # Exponential Runge-Kutta 2
)

set_parameters!(params)

# 2) Initialize simulation
state = initialize_simulation(Float64; include_composition=false)

# 3) Set temperature boundary conditions (Dirichlet: T_inner=1, T_outer=0)
set_boundary_conditions!(state.temperature;
    inner_bc_type = Int(DIRICHLET),
    inner_value   = 1.0,
    outer_bc_type = Int(DIRICHLET),
    outer_value   = 0.0,
)

# 4) Set initial conditions with random perturbations
Random.seed!(1234)
randomize_scalar_field!(state.temperature; amplitude=0.01, lmax=8, domain=state.oc_domain)
randomize_vector_field!(state.velocity; amplitude=1e-5, lmax=6, domain=state.oc_domain)
randomize_magnetic_field!(state.magnetic; amplitude=1e-4, lmax=4, domain=state.oc_domain)

apply_velocity_boundary_conditions!(state.velocity)
apply_magnetic_boundary_conditions!(state.magnetic)

println("Starting simulation...")

# 5) Run simulation
run_simulation!(state)

println("Simulation complete!")
