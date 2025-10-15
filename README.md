# Geodynamo.jl

[![CI](https://github.com/subhk/Geodynamo.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/Geodynamo.jl/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/Geodynamo.jl/stable/)
[![Dev Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/Geodynamo.jl/dev/)

A tool for modeling self-sustained planetary dynamos in rotating spherical shells. 
It leverages a toroidal–poloidal field decomposition and fast spherical-harmonic transforms via [SHTnsKit.jl](https://github.com/subhk/SHTnsKit.jl).

## Features

- **Spherical geometries**: Both spherical shell and solid ball configurations
- **MHD equations**: Full magnetohydrodynamic system with thermal and compositional convection
- **Spectral methods**: Efficient spherical harmonic decomposition using SHTnsKit.jl
- **Parallel computing**: MPI-based domain decomposition for high-performance computing
- **Multiple time-stepping schemes**: CNAB2, ERK2, and Theta method implementations
- **Flexible output**: NetCDF format with configurable precision (Float32/Float64)

## Quick Start

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/Geodynamo.jl")

using Geodynamo

# Run the ball MHD demo
julia --project examples/ball_mhd_demo.jl

# With custom output precision
GEODYNAMO_OUTPUT_PRECISION=Float32 julia --project examples/ball_mhd_demo.jl
```

The generated HTML is written to `docs/build/`.



