# Developer Guide

This guide covers the repository structure, development workflow, testing, and contribution guidelines for GeoDynamo.jl.

!!! tip "Quick Links"
    - [Repository Layout](#repository-layout) — understand the codebase structure
    - [Setting Up](#setting-up-a-dev-environment) — get your environment ready
    - [Testing](#testing) — run and write tests
    - [Contributing](#contributing) — submit your changes

---

## Repository Layout

```
GeoDynamo.jl/
├── src/
│   ├── GeoDynamo.jl              # Module entry point & exports
│   ├── core/
│   │   ├── architecture.jl       # CPU/GPU architecture types
│   │   ├── parameters.jl         # Internal parameter I/O and validation
│   │   ├── initial_conditions.jl # InitialConditions module and loaders
│   │   ├── simulation_health.jl  # Runtime health / NaN checks
│   │   └── spectral_history.jl   # Shared spectral-history helpers
│   ├── parallel/
│   │   ├── mpi.jl                # MPI runtime helpers
│   │   ├── process_grid.jl       # GEODYNAMO_PROC_GRID parsing and subcomms
│   │   ├── pencils.jl            # Pencil topology, load-balance, validation
│   │   ├── transposes.jl         # Transpose planning and timing
│   │   ├── disttranspose_adapter.jl      # r×θ DistTransposePlan adapter
│   │   └── spectral_pencil_adapter.jl    # Legacy spectral adapter helpers
│   ├── transforms/
│   │   └── spectral.jl           # SHTnsKit config, FFT plans, transform kernels
│   ├── fields/
│   │   ├── containers.jl         # PencilArray-backed field/container types
│   │   ├── transforms.jl         # Transforms, spectra, rotations
│   │   └── scalar_operators.jl   # Shared scalar-field operators
│   ├── numerics/
│   │   └── banded_operators.jl   # Banded matrix operations
│   ├── timestep/
│   │   ├── state.jl              # Public timestep-state container
│   │   ├── implicit.jl           # Shared implicit/CNAB2 helpers
│   │   ├── imex.jl               # Solver-side IMEX/EAB2 helpers
│   │   ├── erk2.jl               # ERK2 include shell
│   │   ├── driver.jl             # Solver-side timestep driver
│   │   └── erk2/                 # ERK2 cache, influence, boundary, integration code
│   ├── physics/
│   │   ├── force_projection.jl   # Solenoidal force projection helpers
│   │   ├── nonlinear.jl          # Shared nonlinear-term assembly
│   │   ├── scalar_field_solver_common.jl # Shared scalar implicit-step code
│   │   ├── topography.jl         # Topography physics coupling
│   │   ├── velocity/
│   │   │   ├── field.jl          # Velocity field containers/operators
│   │   │   └── solver.jl         # Solver-owned velocity helpers
│   │   ├── magnetic/
│   │   │   ├── field.jl          # Magnetic field containers/operators
│   │   │   ├── inner_core.jl     # Conducting inner-core admittance & ICB coupling
│   │   │   └── solver.jl         # Solver-owned magnetic helpers
│   │   ├── temperature/
│   │   │   ├── field.jl          # Temperature field containers/operators
│   │   │   └── solver.jl         # Solver-owned temperature helpers
│   │   └── composition/
│   │       ├── field.jl          # Composition field containers/operators
│   │       └── solver.jl         # Solver-owned compositional helpers
│   ├── gpu/                      # Single-GPU solver port (Array/CuArray backends)
│   │   ├── device.jl             # Device selection and array movement
│   │   ├── fields.jl             # GPU spectral/physical field containers
│   │   ├── scalar_transform.jl   # GPU scalar synthesis/analysis
│   │   ├── vector_transform.jl   # GPU vector synthesis/analysis
│   │   ├── nonlinear.jl          # Shared GPU nonlinear kernels (cross, Coriolis, buoyancy)
│   │   ├── scalar_gradient.jl    # GPU scalar gradient
│   │   ├── spectral_curl.jl      # GPU spectral curl
│   │   ├── scalar_nonlinear.jl   # GPU scalar advection assembly
│   │   ├── velocity_nonlinear.jl # GPU velocity nonlinear (buoyancy + Lorentz)
│   │   ├── magnetic_nonlinear.jl # GPU induction term ∇×(u×B)
│   │   ├── banded_solve.jl       # Batched banded LU on device
│   │   ├── cnab2_rhs.jl          # GPU CNAB2 right-hand side
│   │   ├── implicit_solve.jl     # GPU implicit solve with BC rows
│   │   ├── influence_correction.jl # Poloidal influence-matrix correction
│   │   ├── inner_core.jl         # Conducting inner-core GPU path
│   │   ├── scalar_step.jl        # GPU scalar field step
│   │   ├── velocity_step.jl      # GPU velocity field step
│   │   ├── magnetic_step.jl      # GPU magnetic field step
│   │   ├── solver_step.jl        # Full gpu_solver_step! orchestration
│   │   ├── device_state.jl       # Device-state builder from a SolverState
│   │   └── run.jl                # gpu_run! loop + host-gather output hook
│   ├── diagnostics/
│   │   └── solver.jl             # Solver diagnostics
│   ├── io/
│   │   ├── writer.jl             # NetCDF writer with MPI support
│   │   ├── config.jl             # Output configuration helpers
│   │   ├── field_info.jl         # Field metadata for output
│   │   ├── netcdf.jl             # NetCDF-specific helpers
│   │   ├── diagnostics.jl        # Output-side diagnostics helpers
│   │   ├── history.jl            # Time-series history utilities
│   │   ├── restart.jl            # Restart file read/write
│   │   └── utilities.jl          # Shared I/O utilities
│   ├── api/                      # High-level user-facing API (Oceananigans-style)
│   │   ├── grids.jl              # SphericalShellGrid / SphericalBallGrid
│   │   ├── boundary_conditions.jl
│   │   ├── initial_conditions.jl
│   │   ├── timesteppers.jl
│   │   ├── output_writers.jl
│   │   ├── callbacks.jl
│   │   ├── schedules.jl
│   │   ├── fields.jl
│   │   ├── clock.jl
│   │   ├── set.jl
│   │   └── show.jl
│   ├── bcs/
│   │   ├── bcs.jl                # Main BC module
│   │   ├── common.jl             # Shared BC utilities
│   │   ├── scalar_bc.jl          # Shared scalar BC core
│   │   ├── thermal_bc.jl         # Temperature BCs
│   │   ├── compositional_bc.jl   # Composition BCs
│   │   ├── velocity_bc.jl        # Velocity BCs
│   │   ├── magnetic_bc.jl        # Magnetic BCs (incl. conducting inner core)
│   │   ├── interpolation.jl      # BC interpolation
│   │   ├── integration.jl        # BC time integration
│   │   ├── file_bc_loader.jl     # Load BCs from files
│   │   ├── netcdf_io.jl          # BC NetCDF I/O
│   │   ├── programmatic.jl       # Programmatic BC definitions
│   │   └── topography/           # Boundary topography subsystem
│   │       ├── topography.jl
│   │       ├── topography_data.jl
│   │       ├── derivatives.jl
│   │       ├── gaunt_tensors.jl
│   │       ├── thermal_coupling.jl
│   │       ├── velocity_coupling.jl
│   │       ├── magnetic_coupling.jl
│   │       └── stefan_condition.jl
│   ├── Shell/                    # Spherical shell geometry
│   │   └── Shell.jl
│   ├── Ball/                     # Solid ball geometry
│   │   └── Ball.jl
│   ├── solver.jl                # Rewritten solver include shell
│   └── solver/
│       ├── interop.jl           # Narrow bridge to shared GeoDynamo backends
│       ├── parameters.jl        # Internal solver parameter state
│       ├── backend.jl           # Backend/runtime assembly
│       ├── state.jl             # SolverState and cache containers
│       ├── numerics.jl          # Shared solver numerics
│       ├── mainloop.jl          # Solver initialization and run loop
│       └── README.md            # Solver-stack notes
├── ext/
│   └── GeoDynamoCUDAExt.jl       # CUDA backend registration for the solver path
│
├── docs/
│   ├── src/                      # Documenter source pages
│   ├── build/                    # Generated local HTML output
│   ├── plans/                    # Development plans
│   └── make.jl                   # Documentation build script
├── examples/                     # Runnable examples and sample NetCDF files
├── extras/                       # Optional helper utilities
├── scripts/                      # Analysis and maintenance scripts
├── test/                         # Test suite
└── config/                       # Sample parameter files
```

---

## Setting Up a Dev Environment

### Clone and Initialize

```bash
git clone https://github.com/subhk/GeoDynamo.jl
cd GeoDynamo.jl
```

### Link Local SHTnsKit (Optional)

If developing against a local SHTnsKit checkout:

```bash
julia --project -e '
    using Pkg
    Pkg.develop(PackageSpec(path="../SHTnsKit.jl"))
    Pkg.instantiate()
'
```

### Install Dependencies

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### MPI Development

When working on MPI-dependent features, launch Julia with `mpiexec`:

```bash
mpiexec -n 4 julia --project
```

!!! note
    Inside the REPL, activate the project with `using Pkg; Pkg.activate(".")` and load utilities as needed.

---

## Testing

### Running Tests

| Command | Description |
|:--------|:------------|
| `julia --project -e 'using Pkg; Pkg.test()'` | Full test suite |
| `julia --project test/shtnskit_roundtrip.jl` | Single test file |
| `julia --project test/ball_finiteness.jl` | Specific test |
| `GEODYNAMO_PROC_GRID=2x2 mpiexec -n 4 julia --project test/r_theta_equivalence.jl` | One r×θ MPI step-equivalence driver |
| `test/run_mpi_r_theta_equivalence.sh` | Compare `1x1`, `4x1`, `1x4`, and `2x2` scalar/velocity layouts |
| `test/run_mpi_r_theta_equivalence_mhd.sh` | Same comparison with magnetic and composition enabled |
| `test/run_mpi_threaded_smoke.sh` | Multi-rank, multi-thread deadlock regression smoke |

### CI Matrix

The CI runs on multiple platforms via `.github/workflows/ci.yml`:

| Platform | Julia Versions | MPI | Notes |
|:---------|:---------------|:----|:------|
| **Linux (Ubuntu)** | 1.10, 1.11, 1.12 | MPICH | `libnetcdf-dev` |
| **macOS** | 1.12 | Open MPI | Homebrew packages |
| **Windows** | 1.12 | Microsoft MPI | Chocolatey |

The workflow:
1. Caches Julia artifacts
2. Instantiates the project
3. Executes `Pkg.test()`

!!! warning "Keep CI Green"
    After adding new features, ensure existing tests pass or extend the suite to cover new functionality. GitHub Actions must remain green before merging.

---

## Building Documentation

Documentation is built with [Documenter.jl](https://juliadocs.org/Documenter.jl/stable/).

### Local Build

```bash
# Install doc dependencies
julia --project=docs -e 'using Pkg; Pkg.instantiate()'

# Build documentation
julia --project=docs docs/make.jl

# Preview (open in browser)
open docs/build/index.html
```

### CI Deployment

The CI workflow automatically publishes to `gh-pages` on each push to `main`.

---

## Coding Guidelines

### Performance

| Guideline | Reason |
|:----------|:-------|
| Prefer **mutating** functions | Update preallocated buffers; garbage hurts scaling |
| Use `@inbounds` sparingly | Only after profiling confirms safety |
| Cache LU factorizations | Reuse across timesteps |

### MPI Safety

| Guideline | Reason |
|:----------|:-------|
| Test single-rank behavior | Ensure code works without implicit reductions |
| Use global loop bounds | Prevent deadlocks with collectives |
| All I/O is collective | All ranks must call `NCDataset(...)` together for parallel I/O |

### Documentation

| Guideline | Details |
|:----------|:--------|
| Add high-level docstrings | Documenter will surface them in API reference |
| Export new functionality | Add to `GeoDynamo.jl` exports |
| Update the docs | Add entries to relevant `.md` files |

---

## SHTnsKit Integration

The spherical harmonic transform layer spans SHTnsKit setup, field-level helpers,
and r×θ MPI redistribution glue:

| File | Purpose |
|:-----|:--------|
| `transforms/spectral.jl` | Configuration, pencil decomposition, FFT plans |
| `fields/transforms.jl` | Transform operations, energy spectra, rotations |
| `parallel/process_grid.jl` | Explicit `GEODYNAMO_PROC_GRID` parsing and θ/r subcommunicators |
| `parallel/disttranspose_adapter.jl` | `DistTransposePlan` scratch, r↔mode transpose, and m-axis bridge |
| `parallel/pencils.jl` | Shared pencil topology and validation helpers |

### Adding New Transform Functions

1. **Implement** the function in `fields/transforms.jl`

2. **Use try/catch** for version compatibility:

```julia
function my_new_function(config, alm)
    try
        return SHTnsKit.new_feature(config.sht_config, alm)
    catch e
        # Fallback implementation
        @debug "new_feature not available: $e"
        return manual_implementation(config, alm)
    end
end
```

3. **Export** in `GeoDynamo.jl`

4. **Document** in `docs/src/shtnskit.md`

### Feature Detection

Use `get_shtnskit_version_info()` to check capabilities:

```julia
info = get_shtnskit_version_info()
if info.has_qst_transforms
    # Use native QST
else
    # Use fallback
end
```

### Performance Tips

| Tip | Function |
|:----|:---------|
| Use in-place transforms | `shtnskit_synthesis_inplace!` / `shtnskit_analysis_inplace!` |
| Cache BC configs | `_get_cached_bc_shtns_config()` |
| Enable scratch buffers | `SHTNSKIT_USE_SCRATCH_BUFFERS = true` |
| Profile performance | `get_shtnskit_performance_stats()` |

---

## Boundary Conditions

Boundary definitions live under `src/bcs/`.

### Adding a New Boundary Type

1. **Extend** the relevant `bcs.*` module to parse your data source

2. **Update** `io/writer.jl` if fields should be recorded in NetCDF

3. **Document** the format in [Data Output & Restart Files](io.md)

### Module Structure

| Module | Purpose |
|:-------|:--------|
| `bcs.jl` | Main module, config caching |
| `common.jl` | Shared utilities and types |
| `scalar_bc.jl` | Shared scalar boundary-condition core |
| `thermal_bc.jl` | Temperature boundary handling |
| `compositional_bc.jl` | Composition boundary handling |
| `velocity_bc.jl` | Velocity boundary handling |
| `magnetic_bc.jl` | Magnetic boundary handling, including conducting inner-core support |
| `interpolation.jl` | Spatial/temporal interpolation |
| `integration.jl` | Boundary-condition time integration helpers |
| `file_bc_loader.jl` | Boundary-condition loading from files |
| `netcdf_io.jl` | NetCDF read/write |
| `programmatic.jl` | Code-defined boundaries |
| `topography/` | CMB/ICB topography, Gaunt tensors, and thermal/velocity/magnetic coupling |

---

## Contributing

### Workflow

1. **Fork** the repository and create a feature branch

2. **Implement** your changes with tests

3. **Run** the test suite and build docs locally:
   ```bash
   julia --project -e 'using Pkg; Pkg.test()'
   julia --project=docs docs/make.jl
   ```

4. **Open a pull request** describing:
   - Motivation for the change
   - Implementation approach
   - Validation performed

### Bug Reports

When filing issues, include:

| Information | Purpose |
|:------------|:--------|
| MPI configuration | Number of ranks, MPI implementation |
| SHTnsKit version | `get_shtnskit_version_info()` output |
| Parameter files | Reproduce the problem |
| Error messages | Full stack traces |
| Minimal example | Isolate the issue |

### Feature Requests

Feature requests are welcome! Please describe:
- The use case
- Expected behavior
- Any proposed implementation approach

---

## Next Steps

| Goal | Resource |
|:-----|:---------|
| Understand the I/O system | [Data Output & Restart Files](io.md) |
| Learn about time integration | [Time Integration](timestepping.md) |
| Explore configuration options | [Configuration & Parameters](configuration.md) |
| Browse the API | [API Reference](api.md) |
