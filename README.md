# GeoDynamo.jl

[![CI Status](https://github.com/subhk/GeoDynamo.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/subhk/GeoDynamo.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/GeoDynamo.jl/stable/)
[![Dev Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/GeoDynamo.jl/dev/)
[![codecov](https://codecov.io/gh/subhk/GeoDynamo.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/subhk/GeoDynamo.jl)


GeoDynamo.jl is a Julia package for simulating self-sustained planetary dynamos in rotating spherical shells or full balls. It combines toroidal–poloidal decompositions, fast SHTns-based spherical harmonic transforms, and MPI-enabled PencilArrays to reach large problem sizes on modern clusters.

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

## Quick Start

Configure and run a basic spherical shell dynamo simulation:

```julia
using GeoDynamo

# Build a grid, model, and simulation
grid = SphericalShellGrid(
    nr = 64,
    nr_inner = 16,
    lmax = 32,
    mmax = 32,
    nlat = 64,
    nlon = 128,
)

model = GeodynamoModel(
    grid;
    Ek = 1e-4,
    Pr = 1.0,
    Pm = 1.0,
    Sc = 1.0,
    Ra = 1e6,
    include_magnetic = true,
)

simulation = Simulation(model; Δt = 1e-5, stop_time = 0.1)
run!(simulation)
```

For MPI-parallel runs:

```bash
mpiexecjl -n 4 julia my_simulation.jl
```

## Boundary Conditions

Magnetic boundary conditions default to **insulating** and are applied automatically
when the magnetic field is enabled. A **conducting inner core** is available as an
opt-in for shell geometry:

```julia
model = GeodynamoModel(
    grid;
    include_magnetic = true,
    magnetic_inner_bc = :conducting_inner_core,  # default: :insulating
)
```

The inner core then evolves by magnetic diffusion and couples to the outer core at
the inner-core boundary (continuity of the field and its radial derivative). Current
scope: shell geometry, `CNAB2` timestepper, equal inner/outer conductivity.

See the [boundary conditions documentation](https://subhk.github.io/GeoDynamo.jl/stable/boundary-conditions/)
for the full set of velocity, temperature, composition, and magnetic options.

## GPU Backend

The rewritten solver path supports a first real GPU backend through SHTnsKit's
CUDA transform path. This is currently a hybrid backend:

- spherical harmonic analysis and synthesis run through the GPU path
- radial operators, implicit solves, and most solver state remain CPU-backed

To use it, load CUDA before creating the solver backend:

```julia
using GeoDynamo
using CUDA

grid = SphericalShellGrid(GPU(); nr = 64, lmax = 32)
model = GeodynamoModel(grid; include_magnetic = true)
simulation = Simulation(model; Δt = 1e-5, stop_time = 0.1)
```

If CUDA is not installed or no functional device is available, `architecture = :gpu`
fails with an explicit GPU-availability error instead of silently falling back.
