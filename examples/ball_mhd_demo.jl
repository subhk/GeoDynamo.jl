#!/usr/bin/env julia

# Rotating MHD convection in a ball (solid sphere)
# - Geometry: :ball with outer radius = 1.0
# - Ekman = 1e-4, Pr = 1, Pm = 1, Ra = 1e6
# - Magnetic field enabled (i_B = 1)
#
# Run:
#   julia --project examples/ball_mhd_demo.jl
# or with threads:
#   JULIA_NUM_THREADS=8 julia --project examples/ball_mhd_demo.jl

using Geodynamo
using Random

# 1) Set parameters (ball geometry and physics)
params = GeodynamoParameters(
    # Geometry + grid
    geometry   = :ball,   # solid sphere geometry
    i_N        = 64,      # radial points
    i_L        = 32,      # lmax
    i_M        = 32,      # mmax
    i_Th       = 64,      # nlat
    i_Ph       = 128,     # nlon
    i_KL       = 4,       # FD bandwidth

    # Physics
    d_E        = 1e-4,    # Ekman number
    d_Pr       = 1.0,     # Prandtl number
    d_Pm       = 1.0,     # Magnetic Prandtl number
    d_Ra       = 1e6,     # Rayleigh number

    # Magnetic field on
    i_B        = 1,

    # Ball size (outer radius)
    d_R_outer  = 1.0,

    # Timestepping / runtime controls (tune for your machine)
    d_timestep = 1e-4,
    i_maxtstep = 500,
    i_save_rate2 = 50,
    ts_scheme = begin
        s = lowercase(get(ENV, "GEODYNAMO_TS_SCHEME", "cnab2"))
        s == "theta" ? :theta : (s == "eab2" ? :eab2 : :cnab2)
    end,
    # Krylov controls (optional overrides via env)
    i_etd_m = parse(Int, get(ENV, "GEODYNAMO_ETD_M", "20")),
    d_krylov_tol = parse(Float64, get(ENV, "GEODYNAMO_KRYLOV_TOL", "1e-8")),
)

set_parameters!(params)
println("Time-stepping scheme: ", string(Geodynamo.ts_scheme))
println("Krylov m, tol: ", (Geodynamo.i_etd_m, Geodynamo.d_krylov_tol))

# 2) Initialize basic SHTns simulation (thermal + magnetic, no composition)
state = initialize_simulation(Float64; include_composition=false)

# Debug: print pencil layouts (axes_in) to verify decomposition
Geodynamo.print_pencil_axes(state.shtns_config.pencils)

# Optional: register a shared VelocityWorkspace to reduce allocations
if get(ENV, "GEODYNAMO_USE_WS", "1") == "1"
    ws = Geodynamo.create_velocity_workspace(Float64, state.oc_domain.N)
    Geodynamo.set_velocity_workspace!(ws)
end

# Optional: quick workspace equivalence check (set GEODYNAMO_TEST_WS=1)
if get(ENV, "GEODYNAMO_TEST_WS", "0") == "1"
    println("Running workspace equivalence check (GEODYNAMO_TEST_WS=1)...")
    # Save current workspace
    saved_ws = Geodynamo.VELOCITY_WS[]

    # Baseline without workspace
    Geodynamo.set_velocity_workspace!(nothing)
    Geodynamo.compute_vorticity_spectral_full!(state.velocity, state.oc_domain)
    base_tor = copy(parent(state.velocity.vort_toroidal.data_real))
    base_pol = copy(parent(state.velocity.vort_poloidal.data_real))

    # Zero and recompute with workspace
    fill!(parent(state.velocity.vort_toroidal.data_real), 0.0)
    fill!(parent(state.velocity.vort_toroidal.data_imag), 0.0)
    fill!(parent(state.velocity.vort_poloidal.data_real), 0.0)
    fill!(parent(state.velocity.vort_poloidal.data_imag), 0.0)

    # Ensure a workspace is present
    if saved_ws === nothing
        ws2 = Geodynamo.create_velocity_workspace(Float64, state.oc_domain.N)
        Geodynamo.set_velocity_workspace!(ws2)
    else
        Geodynamo.set_velocity_workspace!(saved_ws)
    end
    
    Geodynamo.compute_vorticity_spectral_full!(state.velocity, state.oc_domain)
    tor = parent(state.velocity.vort_toroidal.data_real)
    pol = parent(state.velocity.vort_poloidal.data_real)
    # Report max abs diff
    maxdiff_tor = maximum(abs.(tor .- base_tor))
    maxdiff_pol = maximum(abs.(pol .- base_pol))
    println("Max abs diff (tor, pol) = ", (maxdiff_tor, maxdiff_pol))
end

# 3) Temperature boundary conditions (Dirichlet inner/outer)
#    Options:
#    - Dirichlet (fixed T): inner_bc_type=1, outer_bc_type=1, values below
#    - Neumann (flux): use inner_bc_type=2/outer_bc_type=2; flux profile is built-in for l=0
set_boundary_conditions!(state.temperature;
    inner_bc_type=1, inner_value=1.0,
    outer_bc_type=1, outer_value=0.0,
)

# Alternative: programmatic uniform boundaries via hybrid API
# using GeodynamoBall
# temp_bc = GeodynamoBall.create_ball_hybrid_temperature_boundaries((:uniform, 1.0), (:uniform, 0.0), state.shtns_config)
# apply_netcdf_temperature_boundaries!(state.temperature, temp_bc)

# 4) Random initial conditions (small perturbations)
println("Setting up random initial conditions...")
Random.seed!(1234)

println("  - Temperature: random perturbations (amplitude=0.01, modes l ≤ 8)")
randomize_scalar_field!(state.temperature; amplitude=0.01, lmax=8, domain=state.oc_domain)

println("  - Velocity: small random perturbations (amplitude=1e-5, modes l ≤ 6)")
randomize_vector_field!(state.velocity; amplitude=1e-5, lmax=6, domain=state.oc_domain)

println("  - Magnetic field: tiny seed field (amplitude=1e-4, modes l ≤ 4)")
randomize_magnetic_field!(state.magnetic; amplitude=1e-4, lmax=4, domain=state.oc_domain)

apply_velocity_boundary_conditions!(state.velocity)
apply_magnetic_boundary_conditions!(state.magnetic)

# Add conductive temperature profile to l=0,m=0 mode
function _find_mode_index(config, l_target::Int, m_target::Int)
    for i in 1:config.nlm
        if config.l_values[i] == l_target && config.m_values[i] == m_target
            return i
        end
    end
    return 0
end

function set_conductive_ic!(temp_field, domain; T_in=1.0, T_out=0.0)
    spec_r = parent(temp_field.spectral.data_real)
    spec_i = parent(temp_field.spectral.data_imag)
    lm_rng = Geodynamo.get_local_range(temp_field.spectral.pencil, 1)
    r_rng  = Geodynamo.get_local_range(temp_field.spectral.pencil, 3)
    l0m0 = _find_mode_index(temp_field.config, 0, 0)
    if l0m0 != 0 && (first(lm_rng) <= l0m0 <= last(lm_rng))
        ll = l0m0 - first(lm_rng) + 1
        ri = domain.r[1, 4]
        ro = domain.r[end, 4]
        for r_idx in r_rng
            rr = r_idx - first(r_rng) + 1
            if rr <= size(spec_r, 3)
                r = domain.r[r_idx, 4]
                spec_r[ll, 1, rr] = T_in + (T_out - T_in) * (r - ri) / (ro - ri)
                spec_i[ll, 1, rr] = 0.0
            end
        end
    end
end

println("  - Adding conductive profile to temperature l=0,m=0 mode")
set_conductive_ic!(state.temperature, state.oc_domain; T_in=1.0, T_out=0.0)

# Report initial condition statistics
println("Initial condition statistics:")
temp_energy = sum(abs2.(parent(state.temperature.spectral.data_real))) +
              sum(abs2.(parent(state.temperature.spectral.data_imag)))
vel_tor_energy = sum(abs2.(parent(state.velocity.toroidal.data_real))) +
                 sum(abs2.(parent(state.velocity.toroidal.data_imag)))
vel_pol_energy = sum(abs2.(parent(state.velocity.poloidal.data_real))) +
                 sum(abs2.(parent(state.velocity.poloidal.data_imag)))
mag_tor_energy = sum(abs2.(parent(state.magnetic.toroidal.data_real))) +
                 sum(abs2.(parent(state.magnetic.toroidal.data_imag)))
mag_pol_energy = sum(abs2.(parent(state.magnetic.poloidal.data_real))) +
                 sum(abs2.(parent(state.magnetic.poloidal.data_imag)))

println("  Temperature energy: $(round(temp_energy, digits=6))")
println("  Velocity energy: $(round(vel_tor_energy + vel_pol_energy, digits=8))")
println("  Magnetic energy: $(round(mag_tor_energy + mag_pol_energy, digits=8))")

# 5) Enhanced output configuration using existing output writer
using Printf

# Configure enhanced output with more frequent saves
println("\n" * "="^70)
println("CONFIGURING ENHANCED OUTPUT & DIAGNOSTICS")
println("="^70)

# User configurable output precision - choose Float32 or Float64
# Set via environment variable: GEODYNAMO_OUTPUT_PRECISION=Float32 (or Float64)
output_precision_str = get(ENV, "GEODYNAMO_OUTPUT_PRECISION", "Float64")
output_precision = output_precision_str == "Float32" ? Float32 : Float64

# User configurable save rate
# Set via environment variable: GEODYNAMO_SAVE_RATE=10 (default 5)
user_save_rate = parse(Int, get(ENV, "GEODYNAMO_SAVE_RATE", "5"))

# Modify save rate for more frequent output using existing writer
original_save_rate = params.i_save_rate2
params.i_save_rate2 = user_save_rate
set_parameters!(params)

println("Enhanced output configuration:")
println("  Original save rate: $original_save_rate timesteps")
println("  New save rate: $(params.i_save_rate2) timesteps")
println("  Output precision: $output_precision ($(sizeof(output_precision(1.0)) * 8) bits)")
println("  Output format: NetCDF with full diagnostics")
println("  Location: Current directory")
println("\nTo customize:")
println("  GEODYNAMO_OUTPUT_PRECISION=Float32  # Use Float32 for smaller files")
println("  GEODYNAMO_OUTPUT_PRECISION=Float64  # Use Float64 for higher precision (default)")
println("  GEODYNAMO_SAVE_RATE=10              # Save every N timesteps (default: 5)")

# Custom diagnostics function for console monitoring
function compute_field_diagnostics(state)
    # Compute velocity statistics
    vel_tor_data_r = parent(state.velocity.toroidal.data_real)
    vel_tor_data_i = parent(state.velocity.toroidal.data_imag)
    vel_pol_data_r = parent(state.velocity.poloidal.data_real)
    vel_pol_data_i = parent(state.velocity.poloidal.data_imag)

    max_vel_tor = max(maximum(abs.(vel_tor_data_r)), maximum(abs.(vel_tor_data_i)))
    max_vel_pol = max(maximum(abs.(vel_pol_data_r)), maximum(abs.(vel_pol_data_i)))
    max_vel = max(max_vel_tor, max_vel_pol)

    # Compute temperature statistics
    temp_data_r = parent(state.temperature.spectral.data_real)
    temp_data_i = parent(state.temperature.spectral.data_imag)
    max_temp = max(maximum(abs.(temp_data_r)), maximum(abs.(temp_data_i)))

    # Compute magnetic field statistics
    max_mag = 0.0
    if state.magnetic !== nothing
        mag_tor_data_r = parent(state.magnetic.toroidal.data_real)
        mag_tor_data_i = parent(state.magnetic.toroidal.data_imag)
        mag_pol_data_r = parent(state.magnetic.poloidal.data_real)
        mag_pol_data_i = parent(state.magnetic.poloidal.data_imag)

        max_mag_tor = max(maximum(abs.(mag_tor_data_r)), maximum(abs.(mag_tor_data_i)))
        max_mag_pol = max(maximum(abs.(mag_pol_data_r)), maximum(abs.(mag_pol_data_i)))
        max_mag = max(max_mag_tor, max_mag_pol)
    end

    return max_vel, max_temp, max_mag
end

# Print initial diagnostics before simulation starts
max_vel_init, max_temp_init, max_mag_init = compute_field_diagnostics(state)
println("\nInitial field amplitudes:")
println(@sprintf("  Max Velocity: %12.6e", max_vel_init))
println(@sprintf("  Max Temperature: %12.6e", max_temp_init))
println(@sprintf("  Max Magnetic: %12.6e", max_mag_init))

println("\nStarting simulation with enhanced output...")
println("  Files will be written as: geodynamo_rank_XXXX_time_Y.nc")
println("  NetCDF files include: spectral coefficients + diagnostics + metadata")
println("  Console output will show simulation progress")

# Use the existing simulation which has built-in output writer
# The modified i_save_rate2 will make it save more frequently
run_simulation!(state)

println("\n" * "="^70)
println("SIMULATION COMPLETE WITH ENHANCED OUTPUT")
println("="^70)
println("The existing output writer has saved NetCDF files containing:")
println("  • Spectral coefficients for velocity, magnetic, and temperature fields")
println("  • Comprehensive field diagnostics (energies, extrema, etc.)")
println("  • Grid coordinates and SHT configuration")
println("  • Time series and metadata")
println("\nFiles saved every $(params.i_save_rate2) timesteps to current directory.")

# Usage examples:
#
# Basic run (Float64 precision, save every 5 steps):
#   julia --project examples/ball_mhd_demo.jl
#
# Float32 precision for smaller files:
#   GEODYNAMO_OUTPUT_PRECISION=Float32 julia --project examples/ball_mhd_demo.jl
#
# Custom save rate (every 10 timesteps):
#   GEODYNAMO_SAVE_RATE=10 julia --project examples/ball_mhd_demo.jl
#
# Full customization:
#   GEODYNAMO_OUTPUT_PRECISION=Float32 GEODYNAMO_SAVE_RATE=20 \
#   GEODYNAMO_TS_SCHEME=eab2 GEODYNAMO_ETD_M=30 GEODYNAMO_KRYLOV_TOL=1e-8 \
#   JULIA_NUM_THREADS=4 julia --project examples/ball_mhd_demo.jl
