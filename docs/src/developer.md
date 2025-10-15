# Developer Guide

## Repository Layout

```
Geodynamo.jl/
├── src/
│   ├── Geodynamo.jl          # module entry point & exports
│   ├── fields.jl             # PencilArray-backed field types
│   ├── shtnskit_transforms.jl# SHTnsKit configuration + FFT/transposes
│   ├── timestep.jl           # CNAB2/EAB2/ERK2 integrators and Krylov tools
│   ├── velocity.jl, ...      # Physics kernels and nonlinear terms
│   ├── outputs_writer.jl     # NetCDF writer
│   └── simulation.jl         # high-level driver/state orchestration
├── docs/                     # Documenter configuration and Markdown pages
├── extras/                   # CLI utilities (spectral ↔ physical conversion)
├── test/                     # regression and unit tests
└── config/                   # sample parameter files
```

## Setting Up a Dev Environment

```bash
$ git clone https://github.com/subhk/Geodynamo.jl
$ cd Geodynamo.jl
$ julia --project -e 'using Pkg; Pkg.develop(PackageSpec(path="../SHTnsKit.jl")); Pkg.instantiate()'
```

The command above ensures the local SHTnsKit checkout is used instead of the registry version. When working on MPI-dependent features, launch Julia with `mpiexec`:

```bash
$ mpiexec -n 4 julia --project
```

Inside the REPL activate the project and load utilities as needed (`using Geodynamo`).

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
- When exposing new functionality, add it to the exports in `Geodynamo.jl` and the [API reference](api.md).

## Contributing

1. Fork the repository and create a feature branch.
2. Add tests (or docs) illustrating the behaviour.
3. Run the test-suite and `docs/make.jl`.
4. Open a pull request describing motivation, approach, and validation.

Bug reports and feature requests are welcome via GitHub issues. Include MPI size, SHTnsKit revision, and parameter files to help reproduce problems quickly.
