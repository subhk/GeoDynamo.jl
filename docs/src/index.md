# GeoDynamo.jl

> High-performance spherical-MHD solver for geodynamo and planetary-core studies built on SHTnsKit and PencilArrays

GeoDynamo.jl couples spectral spherical-harmonic transforms with domain-decomposed finite-difference operators to evolve the governing magnetohydrodynamic (MHD) equations for rapidly rotating planetary interiors. The code targets multi-node simulations, supports mixed toroidal/poloidal representations, and provides a modern Julia interface for extending dynamo studies.

## Why GeoDynamo.jl?

- **End-to-end MHD pipeline** – temperature, composition, velocity, and magnetic sub-systems are advanced in a single, tightly coupled solver.
- **Hybrid spectral–radial discretisation** – SHTnsKit supplies fast spherical harmonics while pencil-decomposed finite differences handle radial terms.
- **Time integration options** – CNAB2 IMEX, exponential AB2 (EAB2), and exponential RK2 (ERK2) schemes with Krylov-based linear operators.
- **MPI-friendly I/O** – each rank owns its checkpoint/output files with selectable precision (Float32/Float64) and NetCDF metadata.
- **Composable Julia design** – the module exposes field constructors, transform utilities, and workflow helpers for custom studies.

## Architecture Overview

The main pieces of the package are:

### Core Infrastructure
- **Field abstractions** (`fields.jl`) – PencilArray-backed spectral and physical fields with boundary metadata.
- **Parameters** (`parameters.jl`) – `GeoDynamoParameters` configuration and runtime management.
- **Pencil decomposition** (`pencil_decomps.jl`) – MPI domain decomposition setup via PencilArrays.
- **Linear algebra** (`linear_algebra.jl`) – Banded matrix operations for radial finite differences.

### SHTnsKit Integration
- **Transform configuration** (`shtnskit_transforms.jl`) – SHTnsKit grid setup, FFT plans, and transpose operators.
- **Field operations** (`shtnskit_field_functions.jl`) – Transforms, energy spectra, rotations, spectral operators.

### Physics Kernels
- **Velocity** (`velocity.jl`, `velocity_bc.jl`) – Toroidal-poloidal velocity evolution and boundary conditions.
- **Magnetic** (`magnetic.jl`) – Magnetic field induction, diffusion, and inner core coupling.
- **Thermal** (`thermal.jl`) – Temperature advection-diffusion with scalar field operations.
- **Compositional** (`compositional.jl`) – Composition advection-diffusion matching thermal structure.
- **Shared operations** (`scalar_field_common.jl`) – Common scalar field gradients, transforms, and utilities.

### Time Integration & Simulation
- **Timestep** (`timestep.jl`) – CNAB2/EAB2/ERK2 integrators, Krylov utilities, implicit solvers.
- **Simulation driver** (`simulation.jl`) – Assembles `SimulationState`, manages timestepping, coordinates output.
- **Initial conditions** (`InitialConditions.jl`) – Setup routines for all field types.

### I/O & Utilities
- **Output writer** (`outputs_writer.jl`) – NetCDF writer with MPI support and precision controls.
- **Optimizations** (`optimizations.jl`) – Performance utilities and profiling support.

### Boundary Conditions (`BoundaryConditions/`)
- Modular BC system with thermal, velocity, magnetic, and composition handlers.
- Supports NetCDF-based, programmatic, and interpolated boundary data.

### Required Dependencies

| Layer | Library | Responsibility |
| --- | --- | --- |
| Message passing | [MPI.jl](https://github.com/JuliaParallel/MPI.jl) | Communicator management, collective reductions. |
| Pencil decomposition | [PencilArrays.jl](https://github.com/chriselrod/PencilArrays.jl) & [PencilFFTs.jl](https://github.com/chriselrod/PencilFFTs.jl) | Domain decomposition, transpose plans, distributed FFTs. |
| Spherical harmonics | [SHTnsKit.jl](https://github.com/subhk/SHTnsKit.jl) **v1.1.15+** | Gauss grids, spectral transforms, energy spectra, field rotations, QST decomposition. |
| I/O | [NetCDF.jl](https://github.com/JuliaGeo/NetCDF.jl) & [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) | Structured output, restart files, metadata. |

These dependencies are pulled in automatically via the package manifest. You only need a working MPI implementation and NetCDF C libraries at runtime.

### SHTnsKit v1.1.15 Features

GeoDynamo.jl leverages advanced features from SHTnsKit.jl v1.1.15:

| Feature | Description |
| --- | --- |
| **Energy Spectra** | Native `energy_scalar`, `energy_vector`, `enstrophy` functions for spectral analysis. |
| **QST Transforms** | Full 3D vector field decomposition with `SHqst_to_spat` / `spat_to_SHqst`. |
| **Field Rotations** | Wigner D-matrix rotations via `SH_Zrotate`, `SH_Yrotate`, Euler angle compositions. |
| **Spectral Operators** | Horizontal gradient, divergence, vorticity via `SH_to_grad_spat`. |
| **In-Place Transforms** | Memory-efficient `synthesis!` / `analysis!` operations. |
| **Scratch Buffers** | Pre-allocated `scratch_spatial` / `scratch_fft` for reduced allocations. |

Use `get_shtnskit_version_info()` to check available features at runtime.

## Governing Equations

The solver advances the nondimensional Boussinesq MHD system presented in Sreenivasan & Kar, *Phys. Rev. Fluids* **3**, 093801 (2018), Eqs. (1)–(4). In magnetic-diffusion units the equations read

```math
\begin{aligned}
\frac{E}{\mathrm{Pm}}\frac{\partial \boldsymbol{u}}{\partial t}
  + (\nabla \times \boldsymbol{u}) \times \boldsymbol{u}
  + \hat{\boldsymbol{z}} \times \boldsymbol{u}
  &= -\nabla p^\star
     + \frac{\mathrm{Pm}}{\mathrm{Pr}}\,\mathrm{Ra}\,T\,\boldsymbol{r}
     + (\nabla \times \boldsymbol{B}) \times \boldsymbol{B}
     + E \nabla^2 \boldsymbol{u}, \\
\frac{\partial T}{\partial t} + \boldsymbol{u} \cdot \nabla T
  &= \frac{\mathrm{Pm}}{\mathrm{Pr}} \nabla^2 T, \\
\frac{\partial \boldsymbol{B}}{\partial t}
  &= \nabla \times (\boldsymbol{u} \times \boldsymbol{B}) + \nabla^2 \boldsymbol{B}, \\
\nabla \cdot \boldsymbol{u} &= \nabla \cdot \boldsymbol{B} = 0.
\end{aligned}
```

Internally we divide the momentum equation by `E/Pm`, so the CNAB2/ERK2/EAB2 integrators work with a unit mass matrix and a viscous operator scaled by `Pm`. Thermal and compositional diffusion use `(Pm/Pr)` and `(Pm/Sc)` respectively, while the induction equation diffuses with unit coefficient.

The solver stores both velocity and magnetic fields in a **toroidal–poloidal decomposition**, which enforces `∇·u = ∇·B = 0` spectrally. Each timestep reconstructs physical-space vectors through SHTnsKit transforms, evaluates nonlinear terms, and projects back to toroidal/poloidal coefficients before applying the implicit diffusive operators above.

## Documentation Map

The remainder of the docs walk you through the typical workflow:

- [Getting Started](getting-started.md) – installation, verifying the build, running the quick example.
- [Configuration & Parameters](configuration.md) – how `GeoDynamoParameters` map to grids, physics, and timestepping.
- [Time Integration](timestepping.md) – CNAB2, EAB2, and ERK2 schemes, caches, and recommended settings.
- [Spherical Harmonics](shtnskit.md) – SHTnsKit v1.1.15 transforms, energy spectra, rotations, and spectral operators.
- [Data Output & Restart Files](io.md) – per-rank NetCDF layout, precision control, diagnostics, and boundary datasets.
- [API Reference](api.md) – automatically generated index of exported types and functions.
- [Developer Guide](developer.md) – project layout, testing, documentation build, and contribution guidelines.

If you are familiar with SHTnsKit itself, skim the configuration guide and jump straight to the timestepping section. Otherwise, start with [Getting Started](getting-started.md) to ensure the MPI/SHTnsKit toolchain is correctly configured.

## Cite & Support

If GeoDynamo.jl supports your research, please cite the repository and consider opening an issue or PR with improvements. Community feedback is the fastest way to expand the physics modules and add new diagnostic hooks.

## Release Cadence & Support Matrix

| Branch | Status | Julia | Notes |
| --- | --- | --- | --- |
| `main` | active development | 1.10–1.11 | Latest features, documentation builds on each push. |
| release tags | snapshots | 1.10 | Stable checkpoints for published results. |

We run CI on Linux (OpenMPI) and macOS (MPICH). Contributions targeting other MPI distributions are welcome—check [Developer Guide](developer.md) for instructions on adding regression tests.
