# Data Output & Restart Files

GeoDynamo.jl provides a parallel NetCDF-based I/O system using MPI-IO. All ranks write concurrently to a single shared file per timestep via `NCDatasets.jl` with parallel HDF5. The system handles simulation snapshots, restart files, diagnostics, and boundary condition data.

!!! tip "Quick Start"
    ```julia
    config = default_config(precision=Float64)
    tracker = create_time_tracker(config, 0.0)
    write_fields!(fields, tracker, metadata, config, shtns_config, pencils)
    ```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GeoDynamo I/O System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ Simulation  │────▶│ TimeTracker │────▶│  write_     │        │
│  │    Loop     │     │             │     │  fields!    │        │
│  └─────────────┘     └─────────────┘     └───────┬─────┘        │
│                                                  │              │
│         ┌────────────────────────────────────────┼──────┐       │
│         │                                        ▼      │       │
│         │  ┌─────────────┐  ┌─────────────┐  ┌───────┐  │       │
│         │  │ Grid File   │  │ History     │  │Restart│  │       │
│         │  │ (rank 0)    │  │ (shared)    │  │ Files │  │       │
│         │  └─────────────┘  └─────────────┘  └───────┘  │       │
│         │      Parallel NetCDF (MPI-IO) Output Layer     │       │
│         └───────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | Description |
|:----------|:------------|
| **Parallel I/O** | All ranks write concurrently to a single shared NetCDF file via MPI-IO |
| **Mixed-space output** | Spectral data for velocity/magnetic, physical for T/C |
| **Memory safe** | Each rank holds only its local pencil slice; no gathering |
| **Time-based scheduling** | Automatic output at specified intervals |

---

## OutputConfig

The `OutputConfig` struct controls all aspects of data output.

### Creating Configurations

```julia
# Method 1: From defaults
config = default_config(precision=Float64)

# Method 2: From simulation parameters
config = output_config_from_parameters()

# Method 3: Modify existing config
config = with_output_precision(config, Float32)
```

### Configuration Options

| Field | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `output_space` | OutputSpace | `MIXED_FIELDS` | Data representation mode |
| `output_dir` | String | `"./output"` | Output directory |
| `filename_prefix` | String | `"geodynamo"` | File prefix |
| `include_metadata` | Bool | `true` | Include simulation metadata |
| `include_grid` | Bool | `true` | Include coordinate arrays |
| `include_diagnostics` | Bool | `true` | Include diagnostic scalars |
| `output_precision` | DataType | `Float64` | Precision for field data |
| `spectral_lmax_output` | Int | `-1` | Max ℓ to output (-1 = all) |
| `overwrite_files` | Bool | `true` | Overwrite existing files |
| `output_interval` | Float64 | `0.1` | Time between snapshots |
| `restart_interval` | Float64 | `1.0` | Time between restarts |
| `max_output_time` | Float64 | `Inf` | Stop output after this time |
| `time_tolerance` | Float64 | `1e-10` | Tolerance for time comparisons |

### Output Space Modes

```julia
@enum OutputSpace begin
    MIXED_FIELDS      # Spectral for velocity/magnetic, physical for T/C
    PHYSICAL_ONLY     # All fields in physical (θ, φ, r) space
    SPECTRAL_ONLY     # All fields in spectral (l, m, r) space
end
```

!!! tip "Recommended Mode"
    `MIXED_FIELDS` balances compact spectral storage for divergence-free fields with intuitive physical representation for scalars.

---

## File Layout

### Naming Convention

All ranks write to a single shared file per timestep:
```
{prefix}_{geometry}_{type}_{N}.nc
```

**Examples:**
```
geodynamo_shell_hist_1.nc       # History snapshot 1 (all ranks)
geodynamo_shell_hist_42.nc      # History snapshot 42 (all ranks)
geodynamo_shell_restart_3.nc    # Restart file 3 (all ranks)
geodynamo_shell_grid.nc         # Grid file (rank 0 only)
```

### NetCDF Structure

```
NetCDF File Structure
├── Dimensions
│   ├── theta         (nlat)
│   ├── phi           (nlon)
│   ├── r             (nr)
│   ├── spectral_mode (nlm)
│   ├── time          (1)
│   ├── scalar        (1)
│   └── meta          (1)
│
├── Coordinate Variables
│   ├── theta[theta]        : Colatitude (radians, 0 to π)
│   ├── phi[phi]            : Longitude (radians, 0 to 2π)
│   ├── r[r]                : Radial coordinate (dimensionless)
│   ├── l_values[spectral]  : Spherical harmonic degree
│   ├── m_values[spectral]  : Spherical harmonic order
│   ├── time[time]          : Simulation time
│   └── step[time]          : Timestep number
│
├── Physical Fields (MIXED_FIELDS or PHYSICAL_ONLY)
│   ├── temperature[theta,phi,r]
│   └── composition[theta,phi,r]
│
├── Spectral Fields (MIXED_FIELDS or SPECTRAL_ONLY)
│   ├── velocity_toroidal_real[spectral,r]
│   ├── velocity_toroidal_imag[spectral,r]
│   ├── velocity_poloidal_real[spectral,r]
│   ├── velocity_poloidal_imag[spectral,r]
│   ├── magnetic_toroidal_real[spectral,r]
│   ├── magnetic_poloidal_real[spectral,r]
│   └── (optional) temperature_spectral_*, composition_spectral_*
│
├── Diagnostics
│   ├── diag_temp_mean, diag_temp_std, diag_temp_min, diag_temp_max
│   ├── diag_velocity_toroidal_energy, diag_velocity_toroidal_rms
│   ├── diag_magnetic_toroidal_energy, diag_magnetic_poloidal_energy
│   └── diag_*_peak_l, diag_*_spectral_centroid
│
└── Global Attributes
    ├── title, source, history, Conventions
    ├── mpi_total_ranks
    ├── geometry (shell/ball)
    └── simulation parameters...
```

### Grid File

The grid file (`{prefix}_{geometry}_grid.nc`) is written once by rank 0 and contains:

- Full coordinate arrays at highest precision
- Gauss-Legendre quadrature weights
- SHTnsKit configuration metadata
- Grid type descriptors

!!! note
    This allows post-processing tools to reconstruct the full grid without duplicating coordinates in every history file.

---

## Time Tracking

The `TimeTracker` manages output scheduling:

```julia
# Create tracker starting at t=0
tracker = create_time_tracker(config, 0.0)

# In simulation loop
if should_output_now(tracker, current_time, config)
    write_fields!(fields, tracker, metadata, config)
end

# Query next output time for adaptive timestep
dt_to_output = time_to_next_output(tracker, current_time, config)
```

### TimeTracker Fields

| Field | Type | Description |
|:------|:-----|:------------|
| `last_output_time` | Float64 | Time of last snapshot |
| `last_restart_time` | Float64 | Time of last restart |
| `output_count` | Int | Total snapshots written |
| `restart_count` | Int | Total restarts written |
| `next_output_time` | Float64 | Scheduled next snapshot |
| `next_restart_time` | Float64 | Scheduled next restart |
| `grid_file_written` | Bool | Whether grid file exists |

### Adaptive Timestep Integration

```julia
# Adjust timestep to hit exact output times
time_to_next = time_to_next_output(tracker, t, config)
if 0 < time_to_next < dt
    dt = time_to_next  # Shorten step to hit output exactly
end
```

---

## Writing Simulation Data

### Main Output Function

```julia
function write_fields!(
    fields::Dict{String,Any},
    tracker::TimeTracker,
    metadata::Dict{String,Any},
    config::OutputConfig = output_config_from_parameters(),
    shtns_config::Union{SHTnsKitConfig,Nothing} = nothing,
    pencils::Union{NamedTuple,Nothing} = nothing
) -> Bool
```

Returns `true` if output was written.

### Field Data Format

```julia
fields = Dict(
    # Physical space fields (θ × φ × r arrays)
    "temperature" => Array{Float64,3}(nlat, nlon, nr),
    "composition" => Array{Float64,3}(nlat, nlon, nr),

    # Spectral fields (real/imag pairs)
    "velocity_toroidal" => Dict(
        "real" => Array{Float64,3}(lmax + 1, mmax + 1, nr),
        "imag" => Array{Float64,3}(lmax + 1, mmax + 1, nr)
    ),
    "velocity_poloidal" => Dict("real" => ..., "imag" => ...),
    "magnetic_toroidal" => Dict("real" => ..., "imag" => ...),
    "magnetic_poloidal" => Dict("real" => ..., "imag" => ...)
)
```

### Metadata Dictionary

```julia
metadata = Dict{String,Any}(
    "current_time" => t,
    "current_step" => step,
    "current_dt" => dt,
    "Rayleigh_number" => Ra,
    "Ekman_number" => E,
    "Prandtl_number" => Pr,
    "magnetic_Prandtl" => Pm,
    "geometry" => "shell"  # or "ball"
)
```

---

## Restart Files

### Writing Restarts

Restart files are automatically written at `restart_interval` using parallel I/O:

```julia
write_restart!(fields, tracker, metadata, config, pencils)
```

Restart files include:
- All field data (same as history files)
- Tracker state (`last_output_time`, `output_count`, etc.)
- `grid_file_written` flag for consistency

### Reading Restarts

```julia
# From directory (finds most recent restart)
restart_data, metadata = read_restart!(tracker, "output", target_time, config, pencils)

# From specific file
restart_data, metadata = _load_restart_file(filepath, tracker, config; pencils=pencils)

# Access fields (each rank gets its local slice)
temperature = restart_data["temperature"]       # local θ×φ×r slice
velocity_tor = restart_data["velocity_toroidal"] # local lm×r slice

# Access simulation state
t = metadata["current_time"]
step = metadata["current_step"]
```

!!! tip "Restart Tips"
    - Set `restart_interval` longer than `output_interval` unless frequent checkpointing is needed
    - All ranks open the restart file collectively; each reads only its local pencil slice
    - To change precision: load restart, apply `with_output_precision`, then continue

---

## Boundary Condition I/O

### Reading Boundary Data

```julia
using GeoDynamo.bcs

# Read from NetCDF
bc_data = read_netcdf_boundary_data("boundary_temperature.nc"; precision=Float64)

# Access fields
bc_data.values       # The boundary values
bc_data.theta        # Colatitude coordinates
bc_data.phi          # Longitude coordinates
bc_data.time         # Time values (if time-dependent)
```

### Writing Boundary Data

```julia
# Create boundary data structure
bc = create_boundary_data(
    values_array, "temperature";
    theta=theta_grid, phi=phi_grid, time=time_series,
    units="K", description="Surface temperature"
)

# Write to file
write_netcdf_boundary_data("boundary_out.nc", bc)
```

### Validating Boundary Files

```julia
# Check file structure
validate_netcdf_boundary_file("boundary.nc")

# Get file information
info = get_netcdf_file_info("boundary.nc")
println("Grid: $(info["nlat"]) × $(info["nlon"])")
println("Time-dependent: $(info["is_time_dependent"])")
```

### Boundary File Format

```
boundary_temperature.nc
├── Dimensions
│   ├── theta (nlat)
│   ├── phi (nlon)
│   └── time (ntime)  # optional
│
├── Coordinates
│   ├── theta[theta]   : 0 to π
│   ├── phi[phi]       : 0 to 2π
│   └── time[time]     : simulation times
│
└── Field
    └── temperature[theta,phi] or temperature[theta,phi,time]
```

---

## Diagnostics

### Automatic Diagnostics

Each output file includes computed diagnostics:

**Scalar Field Statistics:**

| Diagnostic | Description |
|:-----------|:------------|
| `diag_temp_mean` | Volume-averaged temperature |
| `diag_temp_std` | Temperature standard deviation |
| `diag_temp_min`, `diag_temp_max` | Temperature extrema |
| `diag_temp_radial_variation` | Radial temperature variation |

**Spectral Field Diagnostics:**

| Diagnostic | Description |
|:-----------|:------------|
| `diag_{field}_energy` | Total spectral energy |
| `diag_{field}_rms` | RMS amplitude |
| `diag_{field}_max` | Maximum coefficient magnitude |
| `diag_{field}_peak_l` | Degree with maximum energy |
| `diag_{field}_spectral_centroid` | Energy-weighted mean degree |
| `diag_{field}_low_mode_fraction` | Energy fraction in ℓ ≤ 10 |

### Custom Diagnostics

```julia
function compute_diagnostics(fields::Dict{String,Any}, field_info::FieldInfo)
    diagnostics = Dict{String, Float64}()

    if haskey(fields, "temperature")
        T = fields["temperature"]
        diagnostics["temp_nusselt"] = compute_nusselt(T, field_info)
    end

    return diagnostics
end
```

---

## MPI Parallel I/O

### How It Works

All MPI ranks collectively open a single NetCDF file using parallel HDF5 (MPI-IO). Each rank writes only its local pencil slice at the correct global offset. No data gathering is required.

```julia
# File is opened collectively by all ranks
ds = NCDataset(filename, "c"; comm=comm, info=MPI.Info())

# Each rank writes its portion using pencil ranges
θ_range = range_local(pencils.r, 1)   # local theta indices in global array
φ_range = range_local(pencils.r, 2)   # local phi indices in global array
ds["temperature"][θ_range, φ_range, :] = local_temperature

mode_indices = local_spectral_mode_indices(shtns_config) # local spectral mode indices
r_range = range_local(pencils.spec, 3)                    # local radial indices
write_local_spectral_coefficients!(ds["velocity_toroidal_real"], mode_indices, r_range, local_vel_tor_real)
```

### Pencil-to-Global Index Mapping

| Field Type | Pencil | Local Dims | Write Pattern |
|:-----------|:-------|:-----------|:--------------|
| Temperature/Composition | `pencils.r` | r local, (θ,φ) distributed | `ds["temperature"][θ_range, φ_range, :]` |
| Spectral fields | `pencils.spec` | (lm, r) distributed | `ds["vel_tor_real"][lm_range, r_range]` |

### Runtime Check

At initialization, the system verifies that parallel HDF5 is available:

```julia
check_parallel_netcdf_support(comm)
```

This will error immediately if parallel NetCDF is not supported, with instructions on how to install a parallel-enabled HDF5.

### Verification

```julia
# Verify the shared output file exists and has correct dimensions
success, missing, info = verify_all_ranks_wrote(
    "output", hist_number;
    geometry="shell",
    expected_dims=Dict("theta" => 64, "phi" => 128, "r" => 20)
)
```

---

## Post-Processing Tools

### Spectral to Physical Conversion

```bash
julia --project extras/spectral_to_physical.jl \
    --input output \
    --output physical \
    --precision Float64
```

### File Analysis Utilities

```julia
# Get available time series from output directory
times = get_time_series("output")

# Get file information
info = get_file_info("geodynamo_shell_hist_1.nc")
println("Time: $(info["time"]), Step: $(info["step"])")
println("Dimensions: $(info["dimensions"])")
println("Variables: $(info["variables"])")
```

### Cleanup Old Files

```julia
# Keep only last 10 time snapshots
cleanup_old_files("output", 10)
```

### Reading Output Files

Since all data is in a single file per timestep, reading is straightforward:

```julia
using NCDatasets

NCDataset("output/geodynamo_shell_hist_5.nc", "r") do ds
    temperature = Array(ds["temperature"][:, :, :])   # Full global array
    vel_tor_real = Array(ds["velocity_toroidal_real"][:, :])  # (nlm, nr)
    println("Dimensions: theta=$(ds.dim["theta"]), phi=$(ds.dim["phi"]), r=$(ds.dim["r"])")
end
```

---

## Performance

### Precision vs Size

| Precision | Bytes/Element | Relative Size |
|:----------|:--------------|:--------------|
| Float64 | 8 | 100% |
| Float32 | 4 | 50% |

!!! tip "Recommendation"
    Use `Float32` for history files, `Float64` for restart files:

    ```julia
    history_config = with_output_precision(default_config(), Float32)
    restart_config = with_output_precision(default_config(), Float64)
    ```

### Memory-Efficient Output

For large arrays, the system automatically uses optimized copying:

```julia
if length(data) > 10000
    data_out = similar(data, output_precision)
    copyto!(data_out, data)  # Efficient in-place conversion
else
    data_out = output_precision.(data)  # Direct broadcast
end
```

---

## Troubleshooting

### Files Not Created

```julia
# Check directory exists and is writable
mkpath(config.output_dir)
@assert isdir(config.output_dir) && iswritable(config.output_dir)

# Verify parallel HDF5 is available
check_parallel_netcdf_support(MPI.COMM_WORLD)
```

### Output File Verification

```julia
success, missing, info = verify_all_ranks_wrote("output", 1;
    expected_dims=Dict("theta" => nlat, "phi" => nlon, "r" => nr))
if !success
    @warn "Output verification failed: $missing"
end
```

### Dimension Mismatches

```julia
validate_output_compatibility(field_info, shtns_config)
```

### NaN in Output

```julia
if any(isnan, temperature)
    @warn "NaN detected in temperature field!"
end
```

### Debug Mode

```julia
# Enable verbose output
ENV["GEODYNAMO_IO_DEBUG"] = "true"

# Check each rank's status
rank = MPI.Comm_rank(MPI.COMM_WORLD)
println("Rank $rank: Writing to $(config.output_dir)")
```

---

## Complete Example

```julia
using GeoDynamo
using MPI

MPI.Init()

# Configuration
config = OutputConfig(
    MIXED_FIELDS,       # output space mode
    "./output",         # output directory
    "dynamo",           # filename prefix
    true, true, true,   # metadata, grid, diagnostics
    Float32,            # output precision
    -1,                 # all spectral modes
    true,               # overwrite files
    0.05, 0.5,          # output, restart intervals
    10.0, 1e-12         # max time, tolerance
)

# Verify parallel HDF5 support (call once at startup)
check_parallel_netcdf_support(MPI.COMM_WORLD)

# Initialize tracker
tracker = create_time_tracker(config, 0.0)
t, dt, step = 0.0, 0.001, 0

# Simulation loop
while t < 2.0
    t += dt
    step += 1

    # ... physics update ...

    # Prepare output data
    fields = Dict(
        "temperature" => T_physical,
        "velocity_toroidal" => Dict("real" => vT_real, "imag" => vT_imag),
        "velocity_poloidal" => Dict("real" => vP_real, "imag" => vP_imag),
        "magnetic_toroidal" => Dict("real" => bT_real, "imag" => bT_imag),
        "magnetic_poloidal" => Dict("real" => bP_real, "imag" => bP_imag)
    )

    metadata = Dict{String,Any}(
        "current_time" => t,
        "current_step" => step,
        "current_dt" => dt
    )

    # Time-based output
    if write_fields!(fields, tracker, metadata, config, shtns_config, pencils)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        rank == 0 && println("Output at t=$t")
    end

    # Adaptive timestep for exact output times
    dt_next = time_to_next_output(tracker, t, config)
    if 0 < dt_next < dt
        dt = dt_next
    end
end

MPI.Finalize()
```

---

## Next Steps

| Goal | Resource |
|:-----|:---------|
| Configure simulation parameters | [Configuration & Parameters](configuration.md) |
| Understand time integration | [Time Integration](timestepping.md) |
| Set up boundary conditions | [Boundary Topography](topography.md) |
| Contribute to development | [Developer Guide](developer.md) |
