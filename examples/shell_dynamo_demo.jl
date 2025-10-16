#!/usr/bin/env julia

# Rotating MHD convection in a spherical shell
# - Geometry: :shell with inner/outer radius ratio = 0.35
# - Ekman = 1e-4, Pr = 1, Pm = 1, Ra = 1e6
# - Magnetic field enabled (i_B = 1)
#
# Run (single process):
#   julia --project examples/shell_dynamo_demo.jl
#
# Run with MPI (multiple processes):
#   mpiexecjl -n 4 julia --project examples/shell_dynamo_demo.jl
#
# Run with MPI + threads:
#   JULIA_NUM_THREADS=8 mpiexecjl -n 4 julia --project examples/shell_dynamo_demo.jl

using GeoDynamo
using MPI
using Random

# Initialize MPI
if !MPI.Initialized()
    MPI.Init()
end

rank = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)

if rank == 0
    println("="^70)
    println("SPHERICAL SHELL DYNAMO SIMULATION")
    println("="^70)
    println("MPI processes: $nprocs")
    println("Threads per process: $(Threads.nthreads())")
    println("Total parallel workers: $(nprocs * Threads.nthreads())")
    println("="^70)
end

# 1) Set parameters (spherical shell geometry and physics)
params = GeoDynamoParameters(
    # Geometry + grid
    geometry   = :shell,  # spherical shell geometry (inner radius > 0)
    i_N        = 64,      # radial points in outer core
    i_Nic      = 16,      # radial points in inner core
    i_L        = 32,      # lmax
    i_M        = 32,      # mmax
    i_Th       = 64,      # nlat
    i_Ph       = 128,     # nlon
    i_KL       = 4,       # FD bandwidth

    # Shell radii
    d_rratio   = 0.35,    # inner radius / outer radius ratio
    d_R_outer  = 1.0,     # outer radius (Earth's CMB normalized to 1)

    # Physics
    d_E        = 1e-4,    # Ekman number
    d_Pr       = 1.0,     # Prandtl number
    d_Pm       = 1.0,     # Magnetic Prandtl number
    d_Ra       = 1e6,     # Rayleigh number

    # Magnetic field on
    i_B        = 1,

    # Timestepping / runtime controls
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

if rank == 0
    println("\nSimulation parameters:")
    println("  Time-stepping scheme: ", string(GeoDynamo.ts_scheme))
    println("  Krylov m, tol: ", (GeoDynamo.i_etd_m, GeoDynamo.d_krylov_tol))
    println("  Geometry: spherical shell with r_inner/r_outer = $(params.d_rratio)")
    println("  Grid: $(params.i_Th) × $(params.i_Ph) × $(params.i_N)")
    println("  Spectral truncation: L=$(params.i_L), M=$(params.i_M)")
end

# 2) Initialize simulation with MPI domain decomposition
if rank == 0
    println("\nInitializing simulation with MPI domain decomposition...")
end

state = initialize_simulation(Float64; include_composition=false)

# Debug: print pencil layouts (axes_in) to verify decomposition
if rank == 0
    println("\nPencil decomposition layout:")
    GeoDynamo.print_pencil_axes(state.shtns_config.pencils)
end

MPI.Barrier(MPI.COMM_WORLD)

# Optional: register a shared VelocityWorkspace to reduce allocations
if get(ENV, "GEODYNAMO_USE_WS", "1") == "1"
    ws = GeoDynamo.create_velocity_workspace(Float64, state.oc_domain.N)
    GeoDynamo.set_velocity_workspace!(ws)
end

# 3) Temperature boundary conditions for spherical shell
#    Fixed temperature at inner and outer boundaries (Dirichlet)
set_boundary_conditions!(state.temperature;
    inner_bc_type=1, inner_value=1.0,   # hot inner boundary
    outer_bc_type=1, outer_value=0.0,   # cold outer boundary
)

# 4) Random initial conditions (small perturbations)
if rank == 0
    println("\nSetting up random initial conditions...")
end
Random.seed!(1234 + rank)  # Different seed per MPI rank for variety

if rank == 0
    println("  - Temperature: random perturbations (amplitude=0.01, modes l ≤ 8)")
end
randomize_scalar_field!(state.temperature; amplitude=0.01, lmax=8, domain=state.oc_domain)

if rank == 0
    println("  - Velocity: small random perturbations (amplitude=1e-5, modes l ≤ 6)")
end
randomize_vector_field!(state.velocity; amplitude=1e-5, lmax=6, domain=state.oc_domain)

if rank == 0
    println("  - Magnetic field: tiny seed field (amplitude=1e-4, modes l ≤ 4)")
end
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

function set_shell_conductive_ic!(temp_field, domain; T_in=1.0, T_out=0.0)
    spec_r = parent(temp_field.spectral.data_real)
    spec_i = parent(temp_field.spectral.data_imag)
    lm_rng = GeoDynamo.get_local_range(temp_field.spectral.pencil, 1)
    r_rng  = GeoDynamo.get_local_range(temp_field.spectral.pencil, 3)
    l0m0 = _find_mode_index(temp_field.config, 0, 0)
    if l0m0 != 0 && (first(lm_rng) <= l0m0 <= last(lm_rng))
        ll = l0m0 - first(lm_rng) + 1
        ri = domain.r[1, 4]
        ro = domain.r[end, 4]
        for r_idx in r_rng
            rr = r_idx - first(r_rng) + 1
            if rr <= size(spec_r, 3)
                r = domain.r[r_idx, 4]
                # Conductive profile in spherical shell: T(r) = (T_in - T_out)/(1/ri - 1/ro) * (1/r - 1/ro) + T_out
                spec_r[ll, 1, rr] = T_in + (T_out - T_in) * (1/r - 1/ri) / (1/ro - 1/ri)
                spec_i[ll, 1, rr] = 0.0
            end
        end
    end
end

if rank == 0
    println("  - Adding conductive profile to temperature l=0,m=0 mode")
end
set_shell_conductive_ic!(state.temperature, state.oc_domain; T_in=1.0, T_out=0.0)

# Report initial condition statistics (local to each MPI rank)
temp_energy_local = sum(abs2.(parent(state.temperature.spectral.data_real))) +
                    sum(abs2.(parent(state.temperature.spectral.data_imag)))
vel_tor_energy_local = sum(abs2.(parent(state.velocity.toroidal.data_real))) +
                       sum(abs2.(parent(state.velocity.toroidal.data_imag)))
vel_pol_energy_local = sum(abs2.(parent(state.velocity.poloidal.data_real))) +
                       sum(abs2.(parent(state.velocity.poloidal.data_imag)))
mag_tor_energy_local = sum(abs2.(parent(state.magnetic.toroidal.data_real))) +
                       sum(abs2.(parent(state.magnetic.toroidal.data_imag)))
mag_pol_energy_local = sum(abs2.(parent(state.magnetic.poloidal.data_real))) +
                       sum(abs2.(parent(state.magnetic.poloidal.data_imag)))

# Global reduction of energies
temp_energy = MPI.Allreduce(temp_energy_local, MPI.SUM, MPI.COMM_WORLD)
vel_energy = MPI.Allreduce(vel_tor_energy_local + vel_pol_energy_local, MPI.SUM, MPI.COMM_WORLD)
mag_energy = MPI.Allreduce(mag_tor_energy_local + mag_pol_energy_local, MPI.SUM, MPI.COMM_WORLD)

if rank == 0
    println("\nInitial condition statistics (global):")
    println("  Temperature energy: $(round(temp_energy, digits=6))")
    println("  Velocity energy: $(round(vel_energy, digits=8))")
    println("  Magnetic energy: $(round(mag_energy, digits=8))")
end

# 5) Enhanced output configuration using existing output writer
using Printf

# Configure enhanced output with more frequent saves
if rank == 0
    println("\n" * "="^70)
    println("CONFIGURING ENHANCED OUTPUT & DIAGNOSTICS")
    println("="^70)
end

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

if rank == 0
    println("Enhanced output configuration:")
    println("  Original save rate: $original_save_rate timesteps")
    println("  New save rate: $(params.i_save_rate2) timesteps")
    println("  Output precision: $output_precision ($(sizeof(output_precision(1.0)) * 8) bits)")
    println("  Output format: NetCDF with full diagnostics")
    println("  Location: Current directory")
    println("  Each MPI rank writes its own file: geodynamo_rank_XXXX_time_Y.nc")
    println("\nTo customize:")
    println("  GEODYNAMO_OUTPUT_PRECISION=Float32  # Use Float32 for smaller files")
    println("  GEODYNAMO_OUTPUT_PRECISION=Float64  # Use Float64 for higher precision (default)")
    println("  GEODYNAMO_SAVE_RATE=10              # Save every N timesteps (default: 5)")
end

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

# Global max across all ranks
global_max_vel = MPI.Allreduce(max_vel_init, MPI.MAX, MPI.COMM_WORLD)
global_max_temp = MPI.Allreduce(max_temp_init, MPI.MAX, MPI.COMM_WORLD)
global_max_mag = MPI.Allreduce(max_mag_init, MPI.MAX, MPI.COMM_WORLD)

if rank == 0
    println("\nInitial field amplitudes (global max):")
    println(@sprintf("  Max Velocity: %12.6e", global_max_vel))
    println(@sprintf("  Max Temperature: %12.6e", global_max_temp))
    println(@sprintf("  Max Magnetic: %12.6e", global_max_mag))

    println("\nStarting spherical shell dynamo simulation...")
    println("  MPI processes: $nprocs (domain decomposed)")
    println("  Files will be written as: geodynamo_rank_XXXX_time_Y.nc")
    println("  NetCDF files include: spectral coefficients + diagnostics + metadata")
    println("  Console output will show simulation progress")
end

MPI.Barrier(MPI.COMM_WORLD)

# Use the existing simulation which has built-in output writer
# The modified i_save_rate2 will make it save more frequently
run_simulation!(state)

MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    println("\n" * "="^70)
    println("SPHERICAL SHELL DYNAMO SIMULATION COMPLETE")
    println("="^70)
    println("The existing output writer has saved NetCDF files containing:")
    println("  • Spectral coefficients for velocity, magnetic, and temperature fields")
    println("  • Comprehensive field diagnostics (energies, extrema, etc.)")
    println("  • Grid coordinates and SHT configuration")
    println("  • Time series and metadata")
    println("\nFiles saved every $(params.i_save_rate2) timesteps to current directory.")
    println("Total MPI ranks: $nprocs")
end

# Finalize MPI
MPI.Finalize()

# Usage examples:
#
# Single process run (Float64 precision, save every 5 steps):
#   julia --project examples/shell_dynamo_demo.jl
#
# MPI parallel run with 4 processes:
#   mpiexecjl -n 4 julia --project examples/shell_dynamo_demo.jl
#
# MPI + multithreading (4 MPI ranks, 8 threads each = 32 total workers):
#   JULIA_NUM_THREADS=8 mpiexecjl -n 4 julia --project examples/shell_dynamo_demo.jl
#
# Float32 precision for smaller files:
#   GEODYNAMO_OUTPUT_PRECISION=Float32 mpiexecjl -n 4 julia --project examples/shell_dynamo_demo.jl
#
# Custom save rate (every 10 timesteps):
#   GEODYNAMO_SAVE_RATE=10 mpiexecjl -n 4 julia --project examples/shell_dynamo_demo.jl
#
# Full customization (MPI + threads + custom parameters):
#   GEODYNAMO_OUTPUT_PRECISION=Float32 GEODYNAMO_SAVE_RATE=20 \
#   GEODYNAMO_TS_SCHEME=eab2 GEODYNAMO_ETD_M=30 GEODYNAMO_KRYLOV_TOL=1e-8 \
#   JULIA_NUM_THREADS=8 mpiexecjl -n 4 julia --project examples/shell_dynamo_demo.jl
#
# For HPC clusters, use your system's MPI launcher:
#   srun -n 64 julia --project examples/shell_dynamo_demo.jl  # SLURM
#   aprun -n 64 julia --project examples/shell_dynamo_demo.jl  # Cray
#   ibrun -n 64 julia --project examples/shell_dynamo_demo.jl  # TACC
