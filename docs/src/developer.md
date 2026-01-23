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
│   │
│   │   # ─────────────────────────────────────────────────────────
│   │   # CORE INFRASTRUCTURE
│   │   # ─────────────────────────────────────────────────────────
│   ├── fields.jl                 # PencilArray-backed field types
│   ├── parameters.jl             # GeoDynamoParameters definition
│   ├── pencil_decomps.jl         # PencilArrays decomposition setup
│   ├── linear_algebra.jl         # Banded matrix operations
│   │
│   │   # ─────────────────────────────────────────────────────────
│   │   # SHTnsKit INTEGRATION
│   │   # ─────────────────────────────────────────────────────────
│   ├── shtnskit_transforms.jl    # SHTnsKit config, FFT plans, transposes
│   ├── shtnskit_field_functions.jl # Transforms, spectra, rotations
│   │
│   │   # ─────────────────────────────────────────────────────────
│   │   # PHYSICS KERNELS
│   │   # ─────────────────────────────────────────────────────────
│   ├── velocity.jl               # Velocity field evolution
│   ├── magnetic.jl               # Magnetic field induction/diffusion
│   ├── thermal.jl                # Temperature advection-diffusion
│   ├── compositional.jl          # Composition advection-diffusion
│   ├── scalar_field_common.jl    # Shared scalar field operations
│   │
│   │   # ─────────────────────────────────────────────────────────
│   │   # TIME INTEGRATION & SIMULATION
│   │   # ─────────────────────────────────────────────────────────
│   ├── timestep.jl               # CNAB2/EAB2/ERK2 integrators
│   ├── simulation.jl             # High-level driver
│   ├── InitialConditions.jl      # Initial condition setup
│   │
│   │   # ─────────────────────────────────────────────────────────
│   │   # I/O & UTILITIES
│   │   # ─────────────────────────────────────────────────────────
│   ├── outputs_writer.jl         # NetCDF writer with MPI support
│   ├── combiner.jl               # Distributed output combiner
│   ├── optimizations.jl          # Performance utilities
│   ├── gpu_backend.jl            # GPU acceleration (experimental)
│   │
│   │   # ─────────────────────────────────────────────────────────
│   │   # BOUNDARY CONDITIONS
│   │   # ─────────────────────────────────────────────────────────
│   ├── bcs/
│   │   ├── bcs.jl               # Main BC module
│   │   ├── common.jl             # Shared BC utilities
│   │   ├── thermal.jl            # Thermal BCs
│   │   ├── velocity.jl           # Velocity BCs
│   │   ├── magnetic.jl           # Magnetic BCs
│   │   ├── composition.jl        # Composition BCs
│   │   ├── interpolation.jl      # BC interpolation
│   │   ├── integration.jl        # BC time integration
│   │   ├── timestepping.jl       # BC timestepping
│   │   ├── netcdf_io.jl          # BC NetCDF I/O
│   │   └── programmatic.jl       # Programmatic BC definitions
│   │
│   │   # ─────────────────────────────────────────────────────────
│   │   # GEOMETRY MODULES
│   │   # ─────────────────────────────────────────────────────────
│   ├── Shell/                    # Spherical shell geometry
│   │   └── Shell.jl
│   └── Ball/                     # Solid ball geometry
│       └── Ball.jl
│
├── docs/                         # Documenter.jl configuration
├── extras/                       # CLI utilities
├── scripts/                      # Analysis scripts
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

### CI Matrix

The CI runs on multiple platforms via `.github/workflows/ci.yml`:

| Platform | Julia Versions | MPI | Notes |
|:---------|:---------------|:----|:------|
| **Linux (Ubuntu)** | 1.10, 1.11 | MPICH | `libnetcdf-dev` |
| **macOS** | 1.11 | Open MPI | Homebrew packages |
| **Windows** | 1.11 | Microsoft MPI | Chocolatey |

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

The spherical harmonic transform layer spans two files:

| File | Purpose |
|:-----|:--------|
| `shtnskit_transforms.jl` | Configuration, pencil decomposition, FFT plans |
| `shtnskit_field_functions.jl` | Transform operations, energy spectra, rotations |

### Adding New Transform Functions

1. **Implement** the function in `shtnskit_field_functions.jl`

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

2. **Update** `outputs_writer.jl` if fields should be recorded in NetCDF

3. **Document** the format in [Data Output & Restart Files](io.md)

### Module Structure

| Module | Purpose |
|:-------|:--------|
| `bcs.jl` | Main module, config caching |
| `common.jl` | Shared utilities and types |
| `thermal.jl` | Thermal boundary handling |
| `velocity.jl` | Velocity boundary handling |
| `magnetic.jl` | Magnetic boundary handling |
| `composition.jl` | Composition boundary handling |
| `interpolation.jl` | Spatial/temporal interpolation |
| `netcdf_io.jl` | NetCDF read/write |
| `programmatic.jl` | Code-defined boundaries |

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
