# Configuration & Parameters

GeoDynamo.jl follows a grid → model → simulation setup style. Build a `SphericalShellGrid` or `SphericalBallGrid`, pass it to `GeodynamoModel` with physical and boundary-condition keywords, then advance a `Simulation`.

## GPU Backend

The current `:gpu` backend is a hybrid solver path:

- SHTnsKit scalar and vector transforms use the CUDA GPU path
- radial operators, implicit solves, and most field storage remain CPU-backed
- GPU SHTns configs record their transform device and intentionally skip eager CPU transform plans/output buffers
- each solver runtime owns a `TransformWorkspace`; on GPU-marked runtimes its scratch allocations can be sourced from the backend-provided `scratch_zeros` hook
- scalar transform scratch gather/scatter can be supplied by the backend through `with_gpu_backend(...)` / `register_gpu_backend!(...)`
- vector transform scratch gather/store and vector component extract/store can also be supplied through the same backend hook surface
- the CUDA extension currently registers explicit host-backed implementations for those scratch hooks, so backend ownership is in place before full device-resident scratch storage lands
- `with_gpu_backend(...)` can temporarily install an alternate backend implementation for tests or experimental integrations, including `scratch_zeros`, and restores the previous backend automatically afterward

To use `:gpu`, load CUDA before creating the solver state:

```julia
using GeoDynamo
using CUDA

grid = SphericalShellGrid(GPU(); nr = 64, lmax = 31)
model = GeodynamoModel(grid; include_magnetic = true)
simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
```

If CUDA is not installed or no functional device is available, backend creation
fails explicitly instead of silently downgrading to CPU.

```@docs; canonical=false
GeoDynamo.SphericalShellGrid
GeoDynamo.SphericalBallGrid
GeoDynamo.GeodynamoModel
GeoDynamo.Simulation
```

---

## Quick Reference

!!! tip "Essential Parameters"
    For most simulations, you'll primarily configure:

    - **Geometry**: `SphericalShellGrid`, `SphericalBallGrid`, `nr`, `nr_inner`, `lmax`, `mmax`, `nlat`, `nlon`
    - **Physics**: `Ek`, `Ra`, `Pr`, `Pm`
    - **Time**: `Simulation(model; Δt, stop_time, stop_iteration)`
    - **Boundaries**: `BoundaryConditions(inner=..., outer=...)`

---

## Geometry & Resolution

### Grid Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `geometry` | Symbol | `:shell` or `:ball` — determines boundary conditions and initialization |
| `nr` | Int | Radial grid points (applies to outer-core and inner-core grids) |
| `lmax` | Int | Maximum spherical harmonic degree |
| `mmax` | Int | Maximum spherical harmonic order (defaults to `lmax`) |
| `nlat` | Int | Physical θ grid resolution |
| `nlon` | Int | Physical φ grid resolution |
| `radial_bandwidth` | Int | Radial finite-difference bandwidth (stencil width) |

!!! note "Resolution Guidelines"
    - Choose `lmax ≈ nr` for balanced spectral/radial workload
    - SHTnsKit requires `nlat ≥ lmax + 2` and `nlon ≥ 2*lmax + 1`
    - If `nlat`/`nlon` are incompatible, SHTnsKit will override them

### SHTnsKit Transform Options

These flags control SHTnsKit v1.1.15 optimizations (set in `transforms/spectral.jl`):

| Flag | Default | Effect |
|:-----|:--------|:-------|
| `SHTNSKIT_USE_DISTRIBUTED` | `true` | Use native MPI-distributed transforms |
| `SHTNSKIT_USE_QST` | `true` | Use full QST decomposition for 3D vectors |
| `SHTNSKIT_USE_SCRATCH_BUFFERS` | `true` | Pre-allocate transform buffers |

**Check feature availability at runtime:**

```julia
info = get_shtnskit_version_info()
println("Version: ", info.version)
println("QST transforms: ", info.has_qst_transforms)
println("Energy functions: ", info.has_energy_functions)
```

See [Spherical Harmonics](shtnskit.md) for the complete transform API.

---

## Physical Parameters

### Dimensionless Numbers

| Parameter | Symbol | Description |
|:----------|:-------|:------------|
| `radius_ratio` | — | Inner-to-outer radius ratio (shell geometry) |
| `Ra` | Ra | Thermal Rayleigh number |
| `RaC` | Ra_C | Compositional Rayleigh number |
| `Ek` | E | Ekman number |
| `Pr` | Pr | Prandtl number |
| `Pm` | Pm | Magnetic Prandtl number |
| `Sc` | Sc | Schmidt number |

### Magnetic Field Control

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `include_magnetic_field` | Bool | Enable magnetic field evolution |
| `impose_magnetic_field` | Bool | Enable imposed background magnetic field |

---

## Time Integration

Choose a timestepper object in the `Simulation` constructor:

```julia
simulation = Simulation(model; Δt = 1e-5, timestepper = CNAB2())
```

| Object | Description |
|:-------|:------------|
| `CNAB2()` | Crank-Nicolson Adams-Bashforth 2; production default |
| `CNAB2(theta = 0.6)` | CNAB2 with extra implicit damping |
| `EAB2(krylov_dimension = 20, tolerance = 1e-8)` | Exponential Adams-Bashforth 2 for stiff diffusion |
| `ERK2()` | Explicit second-order Runge-Kutta path |
| `ETD(krylov_dimension = 20, tolerance = 1e-8)` | Exponential time differencing path |
| `ThetaMethod(theta = 0.5)` | Direct theta-method configuration |

!!! tip "Scheme Selection"
    | Scheme | Best For |
    |:-------|:---------|
    | **CNAB2** | Production dynamo runs, moderate timesteps |
    | **EAB2** | Strongly diffusive regimes (low E, Pm) |
    | **ERK2** | Wave propagation, accuracy-critical applications |

See [Time Integration](timestepping.md) for detailed scheme documentation.

---

## Boundary Conditions

For complete documentation of all boundary condition types and their physical interpretation, see the dedicated **[Boundary Conditions](boundary-conditions.md)** page.

### Quick Reference

| Model keyword | Field | Options |
|:----------|:------|:--------|
| `velocity_bcs` | Velocity | `BoundaryConditions(inner = NoSlip(), outer = StressFree())` |
| `temperature_bcs` | Temperature | `FixedTemperature(value)` or `FixedFlux(value)` at each boundary |
| `composition_bcs` | Composition | `FixedTemperature(value)` or `FixedFlux(value)` at each boundary |

### Summary of BC Types

| Field | Available Options |
|:------|:------------------|
| **Velocity** | No-slip (T=0, ∂P/∂r=0), Stress-free (∂T/∂r=T/r, ∂²P/∂r²=0) |
| **Magnetic** | Insulating (default, automatic); Conducting inner core via `magnetic_inner_bc=:conducting_inner_core` (shell + CNAB2; equal σ, no inner-core rotation). Perfect conductor not yet implemented. |
| **Temperature** | Fixed temperature (Dirichlet), Fixed flux (Neumann) |
| **Composition** | Fixed composition (Dirichlet), Fixed flux (Neumann) |

!!! note "Neumann-Neumann Special Case"
    When both boundaries use flux (Neumann) conditions for temperature or composition, the l=0 mode automatically uses Dirichlet at the inner boundary to pin the mean value.

### Loading Custom Boundaries

```julia
using GeoDynamo

grid = SphericalShellGrid(nr = 64, lmax = 31)
model = GeodynamoModel(grid)
simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
state = simulation.model.state

GeoDynamo.bcs.load_boundary_conditions!(state.temperature, GeoDynamo.TEMPERATURE, Dict(
    :inner => (:uniform, 1.0),
    :outer => (:dirichlet, 0.0),
))

GeoDynamo.bcs.load_boundary_conditions!(state.composition, GeoDynamo.COMPOSITION, Dict(
    :inner => (:neumann, 0.0),
    :outer => (:dirichlet, 0.0),
))
```

Use this helper for programmatic scalar boundaries after the simulation state is
created. File-based scalar BC workflows should use the lower-level NetCDF and
spectral BC utilities directly.

---

## Boundary Topography

For non-spherical boundaries, these parameters control topography coupling. See [Boundary Topography](topography.md) for full theory.

### Master Controls

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `topography_enabled` | Bool | `false` | Master switch for topography coupling |
| `topography_epsilon` | Float64 | `0.01` | Topography amplitude parameter ε |
| `topography_degree` | Int | `-1` | Max spherical harmonic degree (-1 = auto) |

### Field-Specific Switches

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `include_topography_velocity` | Bool | `true` | Enable velocity BC corrections |
| `include_topography_magnetic` | Bool | `true` | Enable magnetic BC corrections |
| `include_topography_thermal` | Bool | `true` | Enable thermal BC corrections |
| `include_topography_slope_terms` | Bool | `true` | Include ∇h slope coupling terms |
| `include_topography_shift_terms` | Bool | `true` | Include h shift terms |

### Stefan Condition (Phase Change)

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `stefan_enabled` | Bool | `false` | Enable Stefan condition for ICB evolution |
| `stefan_number` | Float64 | `1.0` | Stefan number St = c_p ΔT / L |

### Topography Data Files

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `icb_topography_file` | String | Path to ICB topography NetCDF file |
| `ocb_topography_file` | String | Path to CMB topography NetCDF file |

### Example Configuration

**Via constructor:**

```julia
grid = SphericalShellGrid(nr = 64, lmax = 31)
model = GeodynamoModel(
    grid;
    topography_enabled = true,
    topography_epsilon = 0.01,
    include_topography_velocity = true,
    include_topography_magnetic = true,
    ocb_topography_file = "config/cmb_topography.nc",
)
simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
```

**At runtime:**

```julia
enable_topography!(epsilon = 0.02, velocity = true, magnetic = true)
```

---

## Initial Conditions & Restarts

The `InitialConditions` module provides high-level setup helpers:

### Available Functions

| Function | Purpose |
|:---------|:--------|
| `set_velocity_initial_conditions!` | Deterministic poloidal/toroidal seeds (solid-body, dipole, etc.) |
| `randomize_vector_field!` | Add random divergence-free perturbations |
| `set_temperature_ic!` | Conductive, mixed, or user-defined radial profiles |
| `set_composition_ic!` | Composition initialization |
| `randomize_scalar_field!` | Thermal/compositional noise with configurable amplitude |
| `load_initial_conditions!` | Load from saved snapshots (NetCDF/HDF5) |
| `save_initial_conditions` | Save current state to file |

### Typical Setup

```julia
grid = SphericalShellGrid(nr = 64, lmax = 31)
model = GeodynamoModel(grid; include_magnetic = true)
simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
state = simulation.model.state

# Temperature: conductive profile + perturbations
set_temperature_ic!(state.temperature; profile = :conductive)
randomize_scalar_field!(state.temperature; amplitude = 1e-3)

# Velocity: start at rest with small perturbations
set_velocity_initial_conditions!(state.velocity; kind = :rest)

# Magnetic: small random seed
randomize_magnetic_field!(state.magnetic; amplitude = 1e-5)
```

### Restarts

For reproducible continuation runs:

```julia
# Save state
write_restart!(state, tracker, metadata, config)

# Resume later
read_restart!("output/geodynamo_shell_rank_0000_restart_1.nc")
```

---

## Output Configuration

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `output_precision` | Symbol | `:float32` or `:float64` for NetCDF data |
| `output_interval` | Float64 | Output cadence in simulation time; prefer explicit `FieldWriter` schedules for new runs |

!!! note "Parallel I/O"
    All ranks write concurrently to a single shared NetCDF file via parallel HDF5 (MPI-IO). The `independent_output_files` parameter is deprecated and ignored.

!!! tip "Storage Optimization"
    Use `output_precision = :float32` to halve disk usage. Diagnostics remain in `Float64` where accuracy is required.

See [Data Output & Restart Files](io.md) for complete I/O configuration.

---

## Managing Parameters

### Creating and Setting

```julia
# Create with specific values
grid = SphericalShellGrid(nr = 96, lmax = 47)
model = GeodynamoModel(
    grid;
    Ek = 3e-5,
    Pr = 1.0,
    Pm = 1.0,
    Sc = 1.0,
    Ra = 1e6,
)

simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
```

### Saving and Loading

Put the grid, model, and simulation construction in a Julia script such as
`config/run_highres.jl`, then run that script directly or under MPI.

!!! note
    Configuration files under `config/` are plain Julia scripts. Prefer storing the high-level grid/model/simulation construction rather than separate parameter constants.

---

## Next Steps

| Goal | Resource |
|:-----|:---------|
| Understand boundary condition physics | [Boundary Conditions](boundary-conditions.md) |
| Understand time integration schemes | [Time Integration](timestepping.md) |
| Configure output and restarts | [Data Output & Restart Files](io.md) |
| Non-spherical boundary coupling | [Boundary Topography](topography.md) |
| Contribute to development | [Developer Guide](developer.md) |
