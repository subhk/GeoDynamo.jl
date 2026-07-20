# Configuration & Parameters

GeoDynamo.jl follows a grid → model → simulation setup style. Build a `SphericalShellGrid` or `SphericalBallGrid`, pass it to `GeodynamoModel` with physical and boundary-condition keywords, then advance a `Simulation`.

## GPU Backend

The current `:gpu` backend is an explicit single-device solver path:

- dense GPU field containers hold scalar, vector, nonlinear-history, and implicit-solve arrays on the selected architecture
- scalar transforms, Stage-2 solenoidal vector transforms, spectral curls, nonlinear projections, CNAB2 right-hand sides, batched radial solves, and the `gpu_run!` stepping loop have GPU-path implementations
- the CUDA extension routes SHTnsKit scalar and vector transforms to CUDA when the arrays and SHTnsKit configuration are device-backed
- each solver runtime owns a `TransformWorkspace`; on GPU-marked runtimes its scratch allocations can be sourced from the backend-provided `scratch_zeros` hook
- `with_gpu_backend(...)` can temporarily install alternate backend implementations for tests or experimental integrations and restores the previous backend automatically afterward
- full parity with the production CPU `SolverState` path is still guarded by broken tests while the remaining solver-integration differences are closed
- conducting inner-core magnetic coupling is not yet wired into `build_gpu_solver_state`; insulating magnetic cases are the supported solver-state path

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
| `include_magnetic` | Bool | Enable magnetic field evolution |
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
| `ExponentialAdamsBashforth2(krylov_dimension = 20, tolerance = 1e-8)` | Exponential Adams-Bashforth 2 for stiff diffusion |
| `ExponentialRungeKutta2()` | Explicit second-order Runge-Kutta path |
| `ETD(krylov_dimension = 20, tolerance = 1e-8)` | Exponential time differencing path |
| `ThetaMethod(theta = 0.5)` | Direct theta-method configuration |

!!! tip "Scheme Selection"
    | Scheme | Best For |
    |:-------|:---------|
    | **CNAB2** | Production dynamo runs, moderate timesteps |
    | **ExponentialAdamsBashforth2** | Strongly diffusive regimes (low E, Pm) |
    | **ExponentialRungeKutta2** | Wave propagation, accuracy-critical applications |

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
| **Velocity** | No-slip (T=0, P=0, ∂P/∂r=0), Stress-free (∂T/∂r=T/r, P=0, P″−2P′/r=0) |
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

| Public API | Purpose |
|:-----------|:--------|
| `set!(model; field = value)` | Apply initial conditions after constructing a model |
| `initial_conditions = (...)` | Apply initial conditions during `GeodynamoModel` construction |
| `RandomPerturbation(amplitude, lmax, seed)` | Spectral random perturbations for scalar and vector fields |
| `AnalyticIC(pattern; ...)` | Named deterministic patterns such as `:conductive`, `:dipole`, and `:convective` |
| `FileIC(path)` | Load a compatible initial-condition file |
| `ZeroIC()` | Explicit zero initial condition |

### Typical Setup

```julia
grid = SphericalShellGrid(nr = 64, lmax = 31)
model = GeodynamoModel(grid;
    include_magnetic = true,
    initial_conditions = (
        temperature = AnalyticIC(:conductive),
        velocity = RandomPerturbation(amplitude = 1e-4, lmax = 8),
        magnetic = AnalyticIC(:dipole; amplitude = 1.0),
    ),
)

set!(model; temperature = (r, θ, φ) -> (1 - r) + 1e-3 * sin(θ))
simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
```

### Restarts

For reproducible continuation runs:

```julia
# Recommended: schedule checkpoints on the simulation.
checkpoint = CheckpointWriter("output"; schedule = TimeInterval(0.5))
simulation = Simulation(model; Δt = 1e-5, stop_time = 1.0,
                        output_writers = (checkpoint,))
run!(simulation)

# Resume later from the restart directory. This is collective and should be
# called under MPI after MPI.Init().
simulation = Simulation(model; Δt = 1e-5, stop_time = 2.0,
                        restart_from = "output")
run!(simulation)
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
