# GeoDynamo.jl

[![CI Status](https://github.com/subhk/GeoDynamo.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/subhk/GeoDynamo.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/GeoDynamo.jl/stable/)
[![Dev Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/GeoDynamo.jl/dev/)

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


