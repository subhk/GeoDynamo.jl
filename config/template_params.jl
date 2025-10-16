# ============================================================================
# GeoDynamo.jl Parameter Template
# ============================================================================
# 
# Copy this file to your desired location and modify the parameters below
# for your specific simulation needs.
#
# Usage:
#   using GeoDynamo
#   params = load_parameters("path/to/your_params.jl")
#   set_parameters!(params)
#
# ============================================================================

# Grid parameters
const i_N   = 64        # Number of radial points
const i_Nic = 16        # Number of inner core radial points
const i_L   = 32        # Maximum spherical harmonic degree
const i_M   = 32        # Maximum azimuthal wavenumber
const i_Th  = 64        # Number of theta points (must be compatible with SHTnsKit)
const i_Ph  = 128       # Number of phi points (must be compatible with SHTnsKit)
const i_KL  = 4         # Bandwidth for finite differences

# Derived parameters
const i_L1 = i_L
const i_M1 = i_M
const i_H1 = (i_L + 1) * (i_L + 2) ÷ 2 - 1
const i_pH1 = i_H1
const i_Ma = i_M ÷ 2

# Physical parameters
const d_PI = π
const d_rratio = 0.35         # Inner/outer core radius ratio
const d_Ra = 1e6              # Rayleigh number
const d_E = 1e-4              # Ekman number
const d_Pr = 1.0              # Prandtl number
const d_Pm = 1.0              # Magnetic Prandtl number
const d_Ro = 1e-4             # Rossby number
const d_q = 1.0               # Thermal diffusivity ratio

# Timestepping parameters
const d_timestep = 1e-4
const d_time = 0.0
const d_implicit = 0.5        # Crank-Nicolson parameter
const d_dterr = 1e-8          # Error tolerance
const d_courant = 0.5         # CFL factor
const i_maxtstep = 10000      # Maximum timesteps
const i_save_rate2 = 100      # Output frequency
const output_precision = :float64   # :float32 or :float64 for NetCDF output
const independent_output_files = true   # Each MPI rank writes its own files

# Boundary condition flags
const i_vel_bc = 1            # Velocity BC: 1=no-slip, 2=stress-free
const i_tmp_bc = 1            # Temperature BC
const i_cmp_bc = 1            # Composition BC

# Boolean flags
const b_mag_impose = false    # Imposed magnetic field

export i_N, i_Nic, i_L, i_M, i_Th, i_Ph, i_KL, i_L1, i_M1, i_H1
export d_PI, d_rratio, d_Ra, d_E, d_Pr, d_Pm, d_Ro, d_q
export d_timestep, d_time, d_implicit, d_dterr, d_courant, i_maxtstep
