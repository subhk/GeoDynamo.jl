# Data Output & Restart Files

GeoDynamo.jl provides a comprehensive NetCDF-based I/O system designed for scalable MPI parallelism. The system handles simulation snapshots, restart files, diagnostics, and boundary condition data.

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
│         │  │ (rank 0)    │  │ (per-rank)  │  │ Files │  │       │
│         │  └─────────────┘  └─────────────┘  └───────┘  │       │
│         │              NetCDF Output Layer              │       │
│         └───────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
- **Independent writes**: Each MPI rank writes its own file without synchronization
- **Mixed-space output**: Spectral data for velocity/magnetic, physical for temperature/composition
- **Compressed storage**: NetCDF4 with configurable deflate compression
- **Time-based scheduling**: Automatic output at specified intervals

---

## OutputConfig

The `OutputConfig` struct controls all aspects of data output:

```julia
config = default_config(precision=Float64, independent_writes=true)
```

### Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_space` | `OutputSpace` | `MIXED_FIELDS` | Data representation mode |
| `naming_scheme` | `FileNaming` | `RANK_TIME` | Filename pattern |
| `output_dir` | `String` | `"./output"` | Output directory |
| `filename_prefix` | `String` | `"geodynamo"` | File prefix |
| `compression_level` | `Int` | `6` | NetCDF deflate level (0-9) |
| `include_metadata` | `Bool` | `true` | Include simulation metadata |
| `include_grid` | `Bool` | `true` | Include coordinate arrays |
| `include_diagnostics` | `Bool` | `true` | Include diagnostic scalars |
| `output_precision` | `DataType` | `Float64` | Precision for field data |
| `spectral_lmax_output` | `Int` | `-1` | Max l to output (-1 = all) |
| `overwrite_files` | `Bool` | `true` | Overwrite existing files |
| `independent_writes` | `Bool` | `true` | Ranks write independently |
| `output_interval` | `Float64` | `0.1` | Time between snapshots |
| `restart_interval` | `Float64` | `1.0` | Time between restarts |
| `max_output_time` | `Float64` | `Inf` | Stop output after this time |
| `time_tolerance` | `Float64` | `1e-10` | Tolerance for time comparisons |

### Output Space Modes

```julia
@enum OutputSpace begin
    MIXED_FIELDS      # Spectral for velocity/magnetic, physical for T/C
    PHYSICAL_ONLY     # All fields in physical (θ, φ, r) space
    SPECTRAL_ONLY     # All fields in spectral (l, m, r) space
end
```

**Recommended:** `MIXED_FIELDS` balances compact spectral storage for divergence-free fields with intuitive physical representation for scalars.

### Creating Configurations

```julia
# Method 1: From defaults
config = default_config(precision=Float32)

# Method 2: From simulation parameters
config = output_config_from_parameters()

# Method 3: Modify existing config
config = with_output_precision(config, Float32)
config = with_independent_writes(config, true)
```

---

## File Layout

### Naming Convention

Files follow the pattern:
```
{prefix}_{geometry}_rank_{XXXX}_{type}_{N}.nc
```

Examples:
```
geodynamo_shell_rank_0000_hist_1.nc    # History snapshot 1, rank 0
geodynamo_shell_rank_0015_hist_42.nc   # History snapshot 42, rank 15
geodynamo_ball_rank_0000_restart_3.nc  # Restart file 3, rank 0
geodynamo_shell_grid.nc                # Grid file (rank 0 only)
```

### NetCDF Structure

Each output file contains:

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
├── Physical Fields (if MIXED_FIELDS or PHYSICAL_ONLY)
│   ├── temperature[theta,phi,r]
│   └── composition[theta,phi,r]
│
├── Spectral Fields (if MIXED_FIELDS or SPECTRAL_ONLY)
│   ├── velocity_toroidal_real[spectral,r]
│   ├── velocity_toroidal_imag[spectral,r]
│   ├── velocity_poloidal_real[spectral,r]
│   ├── velocity_poloidal_imag[spectral,r]
│   ├── magnetic_toroidal_real[spectral,r]
│   ├── magnetic_poloidal_real[spectral,r]
│   ├── temperature_spectral_real[spectral,r]  (optional)
│   └── composition_spectral_real[spectral,r]  (optional)
│
├── Diagnostics
│   ├── diag_temp_mean, diag_temp_std, diag_temp_min, diag_temp_max
│   ├── diag_velocity_toroidal_energy, diag_velocity_toroidal_rms
│   ├── diag_magnetic_toroidal_energy, diag_magnetic_poloidal_energy
│   └── diag_*_peak_l, diag_*_spectral_centroid
│
└── Global Attributes
    ├── title, source, history, Conventions
    ├── mpi_rank, mpi_total_ranks
    ├── geometry (shell/ball)
    └── simulation parameters...
```

### Grid File

The grid file (`{prefix}_{geometry}_grid.nc`) is written once by rank 0 and contains:

- Full coordinate arrays at highest precision
- Gauss-Legendre quadrature weights
- SHTnsKit configuration metadata
- Grid type descriptors

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
|-------|------|-------------|
| `last_output_time` | `Float64` | Time of last snapshot |
| `last_restart_time` | `Float64` | Time of last restart |
| `output_count` | `Int` | Total snapshots written |
| `restart_count` | `Int` | Total restarts written |
| `next_output_time` | `Float64` | Scheduled next snapshot |
| `next_restart_time` | `Float64` | Scheduled next restart |
| `grid_file_written` | `Bool` | Whether grid file exists |

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
        "real" => Array{Float64,3}(nlm, 1, nr),
        "imag" => Array{Float64,3}(nlm, 1, nr)
    ),
    "velocity_poloidal" => Dict(
        "real" => Array{Float64,3}(nlm, 1, nr),
        "imag" => Array{Float64,3}(nlm, 1, nr)
    ),
    "magnetic_toroidal" => Dict(...),
    "magnetic_poloidal" => Dict(...),

    # Optional: Scalar fields in spectral space
    "temperature_spectral" => Dict("real" => ..., "imag" => ...),
    "composition_spectral" => Dict("real" => ..., "imag" => ...)
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
    "geometry" => "shell",  # or "ball"
    # ... additional parameters
)
```

---

## Restart Files

### Writing Restarts

Restart files are automatically written at `restart_interval`:

```julia
write_restart!(fields, tracker, metadata, config)
```

Restart files include:
- All field data (same as history files)
- Tracker state (`last_output_time`, `output_count`, etc.)
- `grid_file_written` flag for consistency

### Reading Restarts

```julia
restart_data, metadata = read_restart!(tracker, "output", target_time, config)

# restart_data contains all fields
temperature = restart_data["temperature"]
velocity_tor = restart_data["velocity_toroidal"]

# metadata contains simulation state
t = metadata["current_time"]
step = metadata["current_step"]
```

### Restart Tips

1. **Cadence**: Set `restart_interval` longer than `output_interval` unless frequent checkpointing is needed
2. **Precision change**: Load restart, then apply `with_output_precision` before continuing
3. **File matching**: MPI ranks automatically find files matching their `rank_XXXX` suffix

---

## Boundary Condition I/O

### Reading Boundary Data

```julia
using GeoDynamo.bcs

# Read boundary data from NetCDF
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
- `diag_temp_mean`, `diag_temp_std`, `diag_temp_min`, `diag_temp_max`
- `diag_comp_mean`, `diag_comp_std`, `diag_comp_min`, `diag_comp_max`
- `diag_temp_radial_variation`, `diag_comp_radial_variation`

**Spectral Field Diagnostics:**
- `diag_{field}_energy` - Total spectral energy
- `diag_{field}_rms` - RMS amplitude
- `diag_{field}_max` - Maximum coefficient magnitude
- `diag_{field}_peak_l` - Degree with maximum energy
- `diag_{field}_spectral_centroid` - Energy-weighted mean degree
- `diag_{field}_low_mode_fraction` - Energy fraction in l ≤ 10

### Custom Diagnostics

```julia
function compute_diagnostics(fields::Dict{String,Any}, field_info::FieldInfo)
    diagnostics = Dict{String, Float64}()

    # Add custom diagnostics
    if haskey(fields, "temperature")
        T = fields["temperature"]
        diagnostics["temp_nusselt"] = compute_nusselt(T, field_info)
    end

    return diagnostics
end
```

---

## MPI Parallelization

### Independent vs Synchronized Writes

**Independent Mode** (`independent_writes=true`, default):
- Each rank writes immediately when triggered
- No barriers or synchronization
- Best for large-scale runs
- Files may have slightly different timestamps

**Synchronized Mode** (`independent_writes=false`):
- All ranks synchronize before/after writes
- Guarantees consistent output across ranks
- Required for parallel NetCDF collective I/O
- Overhead from `MPI.Barrier` calls

### Per-Rank Data Distribution

Each rank writes only the data it owns based on pencil decomposition:

```julia
# Spectral data: each rank owns a subset of (l,m) modes
lm_range = range_local(pencils.spec, 1)

# Physical data: each rank owns a portion of the grid
theta_range = range_local(pencils.θ, 1)
phi_range = range_local(pencils.φ, 2)
r_range = range_local(pencils.r, 3)
```

### Verification

```julia
# Verify all ranks wrote successfully
success, missing, info = verify_all_ranks_wrote(
    "output", hist_number, nprocs;
    geometry="shell"
)

# Print comprehensive report
print_output_verification_report("output", [1,2,3,4], nprocs)
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
# Get available time series
times = get_time_series("output", rank=0)

# Find files in time range
files = find_files_in_time_range("output", 1.0, 2.0, rank=0)

# Get file information
info = get_file_info("geodynamo_shell_rank_0000_hist_1.nc")
println("Time: $(info["time"]), Step: $(info["step"])")
println("Variables: $(info["variables"])")
```

### Cleanup Old Files

```julia
# Keep only last 10 time snapshots
cleanup_old_files("output", 10)
```

### Combining Multi-Rank Data

```julia
# Collect files from all ranks for a given time
nprocs = 16
files = String[]
for rank in 0:(nprocs-1)
    push!(files, "output/geodynamo_shell_rank_$(lpad(rank,4,'0'))_hist_5.nc")
end

# Read and combine (example for temperature)
using NCDatasets
global_temp = zeros(nlat_global, nlon_global, nr_global)
for (rank, file) in enumerate(files)
    NCDataset(file, "r") do ds
        # Map local data to global array based on pencil ranges
        # ...
    end
end
```

---

## Compression & Performance

### Compression Levels

| Level | Compression | Speed | Use Case |
|-------|-------------|-------|----------|
| 0 | None | Fastest | Development, SSDs |
| 1-3 | Low | Fast | Balance |
| 4-6 | Medium | Moderate | Production (default) |
| 7-9 | High | Slow | Archival |

### Precision vs Size

| Precision | Size/element | Relative Size |
|-----------|--------------|---------------|
| Float64 | 8 bytes | 100% |
| Float32 | 4 bytes | 50% |

**Recommendation:** Use `Float32` for history files, `Float64` for restart files.

```julia
# Different configs for different purposes
history_config = with_output_precision(default_config(), Float32)
restart_config = with_output_precision(default_config(), Float64)
```

### Memory-Efficient Output

For large arrays, the system automatically uses optimized copying:

```julia
# Automatic for arrays > 10000 elements
if length(data) > 10000
    data_out = similar(data, output_precision)
    copyto!(data_out, data)  # Efficient in-place conversion
else
    data_out = output_precision.(data)  # Direct broadcast
end
```

---

## Troubleshooting

### Common Issues

**Files not created:**
```julia
# Check directory exists
mkpath(config.output_dir)

# Verify permissions
isdir(config.output_dir) && iswritable(config.output_dir)
```

**Missing ranks in output:**
```julia
# Verify all ranks completed
success, missing, _ = verify_all_ranks_wrote("output", 1, nprocs)
if !success
    @warn "Missing output from ranks: $missing"
end
```

**Dimension mismatches:**
```julia
# Validate compatibility
validate_output_compatibility(field_info, shtns_config)
```

**NaN in output:**
```julia
# Check for NaN before writing
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
    MIXED_FIELDS, RANK_TIME,
    "./output", "dynamo",
    6, true, true, true,
    Float32, -1, true, true,
    0.05, 0.5, 10.0, 1e-12
)

# Initialize
tracker = create_time_tracker(config, 0.0)
t, dt, step = 0.0, 0.001, 0

# Simulation loop
while t < 2.0
    t += dt
    step += 1

    # ... physics update ...

    # Prepare output
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
