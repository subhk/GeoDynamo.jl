# Data Output & Restart Files

GeoDynamo.jl writes diagnostics, checkpoints, and restart files through the NetCDF-based `outputs_writer.jl`. The system is designed to scale with MPI: each rank writes only the data it owns, avoiding collective I/O bottlenecks.

## OutputConfig

The `OutputConfig` struct controls how data is written to NetCDF files. Create instances using:

- `default_config()` - Creates default configuration
- `output_config_from_parameters()` - Seeds from current GeoDynamoParameters
- `with_output_precision(config, T)` - Changes precision to type T
- `with_independent_writes(config, flag)` - Enables/disables independent writes

Key flags:

- `output_precision` – choose `Float32` to reduce file size; metadata and diagnostics stay in `Float64`.
- `independent_writes` – when `true` (default), every rank writes `geodynamo_output_rank_XXXX_time_T.nc`. Disable to enforce `MPI.Barrier` synchronisation at the cost of scalability.
- `output_dir` / `filename_prefix` – location and prefix for both output and restart files.
- `compression_level` – NetCDF deflate level (`0` disables compression).

Use `output_config_from_parameters()` to seed an OutputConfig from the current `GeoDynamoParameters`, then tweak it with `with_output_precision` or `with_independent_writes`.

## File Layout

Each NetCDF file stores:

- **Coordinates:** `theta`, `phi`, `r`, along with spherical-harmonic index arrays (`l_values`, `m_values`).
- **Spectral fields:** real/imaginary pairs for velocity (toroidal/poloidal), magnetic field, temperature, composition.
- **Physical fields:** optional temperature/composition grids when `output_space` includes physical data.
- **Diagnostics:** scalar variables summarising energy, RMS, spectral peaks, etc.
- **Metadata:** geometry, MPI rank, wall-clock time, simulation step.

Per-rank files make post-processing easy—filter by `rank_0003` to grab data from a specific MPI rank.

## NetCDF Boundaries

Boundary conditions can also be sourced from NetCDF. The helper `config/netcdf_boundaries.jl` and the Boundary Conditions module understand files containing:

- `boundary_type` (Dirichlet, Neumann, etc.)
- `inner_values` / `outer_values`
- Time-dependent series with interpolation metadata

Refer to `docs/src/NETCDF_BOUNDARIES.jl` (legacy) or the [developer guide](developer.md#boundary-conditions) for template creation.

## Restart Files

`write_restart!` mirrors the output structure and adds restart-specific scalars (`last_output_time`, `output_count`, `restart_count`). Use `read_restart!` to populate a fresh `SimulationState` and `TimeTracker`:

```julia
restart_data, metadata = read_restart!(tracker, "output", 1.0, config)
```

### Tips

- Keep restart cadence (`restart_interval`) longer than output cadence unless you need frequent checkpoints.
- To resume with new precision, load the restart and apply `with_output_precision` before continuing.
- MPI ranks search for files matching their `rank_XXXX` suffix; avoid manual renames.

## Diagnostics & Conversion

The `extras/spectral_to_physical.jl` script converts spectral NetCDF files to physical grids in batch mode. Run it with:

```bash
julia --project extras/spectral_to_physical.jl --input output --output physical
```

The script honours the same precision flags and uses the stored SHTnsKit configuration to perform inverse transforms.
