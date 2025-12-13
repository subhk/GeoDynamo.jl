# GeoDynamo.jl

[![CI Status](https://github.com/subhk/GeoDynamo.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/subhk/GeoDynamo.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/GeoDynamo.jl/stable/)
[![Dev Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/GeoDynamo.jl/dev/)

GeoDynamo.jl is a Julia package for simulating self-sustained planetary dynamos in rotating spherical shells or full balls. It combines toroidal–poloidal decompositions, fast SHTns-based spherical harmonic transforms, and MPI-enabled PencilArrays to reach large problem sizes on modern clusters.

## Highlights

- **Flexible geometries** – run both spherical shell and full ball configurations with consistent APIs (`GeoDynamoShell`, `GeoDynamoBall`).
- **Full MHD physics** – coupled velocity, magnetic, thermal, and compositional fields with buoyancy forcing and mixed boundary conditions.
- **Spectral accuracy** – SHTnsKit.jl backed transforms, high-order radial operators, and optimized banded linear algebra.
- **Parallel-first design** – MPI domain decomposition, PencilArrays communication helpers, and batch transform optimizations.
- **Time-stepping options** – CNAB2, ERK2, Theta, plus diagnostics tooling for monitoring residuals and cache reuse.
- **Structured outputs** – NetCDF/NCDatasets writers, restart capability, and utilities for spectral→physical conversions.

## Requirements

- Julia `v1.10`–`v1.12` (see `Project.toml`)
- A working MPI installation (`MPI.jl` autodetects your vendor library)
- [SHTnsKit.jl](https://github.com/subhk/SHTnsKit.jl) (installed automatically as a dependency)
- Optional: NetCDF/HDF5 libraries for parallel I/O support

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/subhk/GeoDynamo.jl")

# or for local development
Pkg.develop(path = "/path/to/GeoDynamo.jl")
Pkg.instantiate()
```

After installation load the package to trigger precompilation and parameter initialization:

```julia
julia> using GeoDynamo
```


# Try different output precision
GEODYNAMO_OUTPUT_PRECISION=Float32 julia --project examples/ball_mhd_demo.jl
```


