# GeoDynamo.jl

**High-performance spherical-MHD solver for geodynamo and planetary-core simulations**

[![Julia](https://img.shields.io/badge/Julia-1.10%2B-blue)](https://julialang.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

!!! tip "New to GeoDynamo.jl?"
    Get up and running in minutes with our [Getting Started](getting-started.md) guide.

---

## What is GeoDynamo.jl?

GeoDynamo.jl is a modern Julia package for simulating magnetohydrodynamic (MHD) flows in rapidly rotating planetary interiors. It couples:

- **Spectral spherical-harmonic transforms** via SHTnsKit
- **Domain-decomposed finite differences** via PencilArrays
- **Scalable MPI parallelism** for multi-node simulations

```julia
using GeoDynamo

# Configure and run a simulation
params = GeoDynamoParameters(geometry=:shell, i_N=64, i_L=31, d_E=1e-4, d_Ra=1e6)
set_parameters!(params)
state = initialize_simulation(Float64)
run_simulation!(state; t_end=0.1)
```

---

## Highlights

!!! success "End-to-End MHD Pipeline"
    Temperature, composition, velocity, and magnetic fields evolved in a single tightly coupled solver.

!!! success "Hybrid Spectral-Radial Discretization"
    SHTnsKit for fast spherical harmonics + pencil-decomposed finite differences for radial terms.

!!! success "Multiple Time Integrators"
    CNAB2 IMEX, exponential AB2 (EAB2), and exponential RK2 (ERK2) with Krylov-based operators.

!!! success "Scalable MPI I/O"
    Per-rank checkpoint/output with selectable precision (Float32/Float64) and NetCDF metadata.

!!! success "Boundary Topography"
    Linearized non-spherical CMB/ICB boundaries with Gaunt tensor mode coupling.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          GeoDynamo.jl                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐            │
│   │    Fields     │   │  Parameters   │   │    Pencil     │            │
│   │  & Transforms │   │  & Config     │   │ Decomposition │            │
│   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘            │
│           │                   │                   │                     │
│           └───────────────────┼───────────────────┘                     │
│                               ▼                                         │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │                   Physics Kernels                        │          │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │          │
│   │  │Velocity │ │Magnetic │ │ Thermal │ │Compositional│   │          │
│   │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘   │          │
│   └─────────────────────────────────────────────────────────┘          │
│                               │                                         │
│                               ▼                                         │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │              Time Integration (CNAB2/EAB2/ERK2)          │          │
│   └─────────────────────────────────────────────────────────┘          │
│                               │                                         │
│                               ▼                                         │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │                  NetCDF Output & Restart                 │          │
│   └─────────────────────────────────────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### Infrastructure

| Module | Description |
|:-------|:------------|
| `fields.jl` | PencilArray-backed spectral and physical fields |
| `parameters.jl` | `GeoDynamoParameters` configuration management |
| `pencil_decomps.jl` | MPI domain decomposition via PencilArrays |
| `linear_algebra.jl` | Banded matrix operations for radial derivatives |

### SHTnsKit Integration

| Module | Description |
|:-------|:------------|
| `shtnskit_transforms.jl` | Grid setup, FFT plans, transpose operators |
| `shtnskit_field_functions.jl` | Transforms, energy spectra, rotations |

### Physics

| Module | Description |
|:-------|:------------|
| `velocity.jl` | Toroidal-poloidal velocity evolution |
| `magnetic.jl` | Magnetic field induction and diffusion |
| `thermal.jl` | Temperature advection-diffusion |
| `compositional.jl` | Composition advection-diffusion |

### Boundary Conditions

| Module | Description |
|:-------|:------------|
| `bcs/` | Modular BC system (thermal, velocity, magnetic, composition) |
| `bcs/topography/` | Non-spherical CMB/ICB with Gaunt tensors |

---

## Dependencies

GeoDynamo.jl builds on a robust stack of Julia packages:

```
                    ┌─────────────────┐
                    │   GeoDynamo.jl  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   SHTnsKit    │   │ PencilArrays  │   │    MPI.jl     │
│  (Harmonics)  │   │(Decomposition)│   │  (Parallel)   │
└───────────────┘   └───────────────┘   └───────────────┘
                             │
                             ▼
                    ┌───────────────┐
                    │   NetCDF.jl   │
                    │     (I/O)     │
                    └───────────────┘
```

!!! note "Automatic Installation"
    All dependencies are pulled in automatically via the package manifest. You only need a working MPI implementation and NetCDF C libraries at runtime.

---

## Governing Equations

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ∂u/∂t  =  viscous diffusion  +  buoyancy  +  Lorentz force       │
│                     ↓                 ↓              ↓              │
│                   E∇²u            Ra·T·r̂        (∇×B)×B            │
│                                                                     │
│   ∂T/∂t  =  thermal diffusion  -  advection                        │
│                     ↓                  ↓                            │
│                (Pm/Pr)∇²T            u·∇T                           │
│                                                                     │
│   ∂B/∂t  =  magnetic diffusion  +  induction                       │
│                     ↓                  ↓                            │
│                   ∇²B              ∇×(u×B)                          │
│                                                                     │
│   Constraints:      ∇·u = 0           ∇·B = 0                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

The solver advances the nondimensional Boussinesq MHD system from *Sreenivasan & Kar, Phys. Rev. Fluids* **3**, 093801 (2018).

### Momentum

```math
\frac{E}{\mathrm{Pm}}\frac{\partial \boldsymbol{u}}{\partial t}
  + (\nabla \times \boldsymbol{u}) \times \boldsymbol{u}
  + \hat{\boldsymbol{z}} \times \boldsymbol{u}
  = -\nabla p^\star
     + \frac{\mathrm{Pm}}{\mathrm{Pr}}\,\mathrm{Ra}\,T\,\boldsymbol{r}
     + (\nabla \times \boldsymbol{B}) \times \boldsymbol{B}
     + E \nabla^2 \boldsymbol{u}
```

### Temperature & Magnetic Field

```math
\frac{\partial T}{\partial t} + \boldsymbol{u} \cdot \nabla T = \frac{\mathrm{Pm}}{\mathrm{Pr}} \nabla^2 T
```

```math
\frac{\partial \boldsymbol{B}}{\partial t} = \nabla \times (\boldsymbol{u} \times \boldsymbol{B}) + \nabla^2 \boldsymbol{B}
```

### Constraints

```math
\nabla \cdot \boldsymbol{u} = 0 \qquad \nabla \cdot \boldsymbol{B} = 0
```

!!! info "Toroidal-Poloidal Decomposition"
    Both velocity and magnetic fields use toroidal-poloidal decomposition, which spectrally enforces the divergence-free constraints.

---

## Documentation

| Page | Description |
|:-----|:------------|
| **[Getting Started](getting-started.md)** | Installation and first simulation |
| **[Configuration](configuration.md)** | All parameter options explained |
| **[Boundary Conditions](boundary-conditions.md)** | Velocity, magnetic, thermal, and compositional BCs |
| **[Time Integration](timestepping.md)** | CNAB2, EAB2, ERK2 schemes |
| **[Spherical Harmonics](shtnskit.md)** | SHTnsKit transforms and operators |
| **[Boundary Topography](topography.md)** | Non-spherical boundary coupling |
| **[Data Output](io.md)** | NetCDF files, restarts, diagnostics |
| **[Developer Guide](developer.md)** | Contributing and code structure |
| **[API Reference](api.md)** | Complete function documentation |

---

## Compatibility

| Environment | Versions | Status |
|:------------|:---------|:-------|
| **Julia** | 1.10, 1.11 | Supported |
| **Linux** | OpenMPI, MPICH | Tested in CI |
| **macOS** | MPICH (Homebrew) | Tested in CI |
| **Windows** | Microsoft MPI | Tested in CI |

---

## Citation

If GeoDynamo.jl supports your research, please cite:

```bibtex
@software{geodynamo_jl,
  author = {Kar, Subhajit},
  title  = {GeoDynamo.jl: Spherical-MHD Solver for Planetary Cores},
  url    = {https://github.com/subhk/GeoDynamo.jl},
  year   = {2024}
}
```

---

!!! tip "Get Involved"
    Community feedback via [GitHub Issues](https://github.com/subhk/GeoDynamo.jl/issues) and PRs is the fastest way to expand physics modules and add diagnostic hooks!
