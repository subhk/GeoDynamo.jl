# Developer Guide

## Repository Layout

```
GeoDynamo.jl/
├── src/
│   ├── GeoDynamo.jl              # module entry point & exports
│   ├── fields.jl                 # PencilArray-backed field types
│   ├── shtnskit_transforms.jl    # SHTnsKit configuration + FFT/transposes
│   ├── shtnskit_field_functions.jl # v1.1.15 transforms, energy, rotations
│   ├── scalar_field_common.jl    # Shared scalar field operations
│   ├── timestep.jl               # CNAB2/EAB2/ERK2 integrators and Krylov tools
│   ├── velocity.jl, magnetic.jl  # Physics kernels and nonlinear terms
│   ├── thermal.jl, compositional.jl
│   ├── outputs_writer.jl         # NetCDF writer
│   ├── simulation.jl             # high-level driver/state orchestration
│   ├── BoundaryConditions/       # Boundary condition modules
│   │   ├── BoundaryConditions.jl # Main BC module with config caching
│   │   ├── thermal.jl, velocity.jl, ...
│   └── Ball/                     # Solid ball geometry support
├── docs/                         # Documenter configuration and Markdown pages
├── extras/                       # CLI utilities (spectral ↔ physical conversion)
├── scripts/                      # Analysis and utility scripts
├── test/                         # regression and unit tests
└── config/                       # sample parameter files
```

## Setting Up a Dev Environment

```bash
$ git clone https://github.com/subhk/GeoDynamo.jl
$ cd GeoDynamo.jl
$ julia --project -e 'using Pkg; Pkg.develop(PackageSpec(path="../SHTnsKit.jl")); Pkg.instantiate()'
```

The command above ensures the local SHTnsKit checkout is used instead of the registry version. When working on MPI-dependent features, launch Julia with `mpiexec`:

```bash
$ mpiexec -n 4 julia --project
```

Inside the REPL activate the project and load utilities as needed (`using GeoDynamo`).

## Testing

- **Full suite:** `julia --project -e 'using Pkg; Pkg.test()'`
- **Single file:** run the script under `test/` directly (e.g. `test/shtnskit_roundtrip.jl`).
- **CI matrix:** `.github/workflows/ci.yml` runs on Ubuntu (Julia 1.10/1.11), macOS, and Windows (Julia 1.11). Linux installs `mpich`/`libnetcdf-dev`, macOS uses Homebrew (`open-mpi`, `netcdf`), and Windows relies on Microsoft MPI via Chocolatey. The workflow caches Julia artifacts, instantiates the project, and executes `Pkg.test()`.

After adding new features make sure either the existing tests cover them or you extend the suite—GitHub Actions must remain green before merging.

## Building Documentation

Documentation is built with [Documenter.jl](https://juliadocs.org/Documenter.jl/stable/).

```bash
$ julia --project=docs -e 'using Pkg; Pkg.instantiate()'
$ julia --project=docs docs/make.jl
```

The CI workflow publishes the generated site to `gh-pages`. To preview locally, open `docs/build/index.html` after running `make.jl`.

## Boundary Conditions

Boundary definitions live under `src/BoundaryConditions/`. To add a new boundary type:

1. Extend the relevant `BoundaryConditions.*` module to parse your data source.
2. Update `outputs_writer.jl` if you want the new fields recorded in NetCDF.
3. Document the format in [Data Output & Restart Files](io.md).

## Coding Guidelines

- Prefer **mutating** functions that update preallocated buffers; garbage hurt scaling.
- Keep new modules MPI-safe: ensure rank-local code runs without implicit reductions when `independent_output_files = true`.
- Use `@inbounds` only after profiling, and add high-level docstrings so Documenter can surface them.
- When exposing new functionality, add it to the exports in `GeoDynamo.jl` and the [API reference](api.md).

## SHTnsKit Integration

The spherical harmonic transform layer is split across two files:

- `shtnskit_transforms.jl` – Configuration, pencil decomposition, FFT plans
- `shtnskit_field_functions.jl` – Transform operations, energy spectra, rotations, operators

### Adding New Transform Functions

1. Implement the function in `shtnskit_field_functions.jl`
2. Use `try/catch` with fallback for version compatibility:
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
3. Add export to `GeoDynamo.jl`
4. Add documentation to `docs/src/shtnskit.md`

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

### Performance Considerations

- Use `shtnskit_synthesis_inplace!` / `shtnskit_analysis_inplace!` for hot paths
- Cache SHTnsKit configurations via `_get_cached_bc_shtns_config()` for boundary transforms
- Enable scratch buffers with `SHTNSKIT_USE_SCRATCH_BUFFERS = true`
- Profile with `get_shtnskit_performance_stats()` to verify optimizations are active

## Contributing

1. Fork the repository and create a feature branch.
2. Add tests (or docs) illustrating the behaviour.
3. Run the test-suite and `docs/make.jl`.
4. Open a pull request describing motivation, approach, and validation.

Bug reports and feature requests are welcome via GitHub issues. Include MPI size, SHTnsKit revision, and parameter files to help reproduce problems quickly.
