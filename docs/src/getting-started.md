# Getting Started

Welcome to GeoDynamo.jl! This guide will get you from zero to running your first simulation.

---

## Prerequisites

!!! note "What You Need"
    | Requirement | Version |
    |:------------|:--------|
    | Julia | **1.10** or **1.11** |
    | MPI | OpenMPI, MPICH, or Intel MPI |
    | NetCDF | C libraries for output |

---

## Installation

### Step 1: Install Julia

Download from [julialang.org/downloads](https://julialang.org/downloads/) and ensure it's on your `PATH`.

### Step 2: Install MPI & NetCDF

=== "Ubuntu/Debian"
    ```bash
    sudo apt install mpich libnetcdf-dev
    ```

=== "macOS"
    ```bash
    brew install mpich netcdf
    ```

=== "Fedora/RHEL"
    ```bash
    sudo dnf install mpich netcdf-devel
    ```

Verify MPI is working:

```
╭─────────────────────────────────────────────────────────────────────────────╮
│  Verify MPI                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   $ mpiexec --version                                                       │
│                                                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
```

### Step 3: Clone GeoDynamo.jl

```
╭─────────────────────────────────────────────────────────────────────────────╮
│  Clone repository                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   $ git clone https://github.com/subhk/GeoDynamo.jl                         │
│   $ cd GeoDynamo.jl                                                         │
│                                                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
```

### Step 4: Install Dependencies

```
╭─────────────────────────────────────────────────────────────────────────────╮
│  Install Julia packages                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   $ julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'     │
│                                                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
```

!!! tip "Development Mode"
    For development with a local SHTnsKit checkout at `../SHTnsKit.jl`:
    ```bash
    julia --project -e 'using Pkg; Pkg.develop(PackageSpec(path="../SHTnsKit.jl"))'
    ```

!!! tip "Optional CUDA GPU Backend"
    The rewritten solver supports a hybrid GPU backend for SHTnsKit transforms.
    To use it, add CUDA to your environment and load it before building the solver:
    ```julia
    using GeoDynamo
    using CUDA

    grid = SphericalShellGrid(GPU(); nr = 64, lmax = 31)
    model = GeodynamoModel(grid; include_magnetic = true)
    simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
    ```

---

## Verification

### Test the Installation

```
╭─────────────────────────────────────────────────────────────────────────────╮
│  Quick test                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   $ julia --project test/shtnskit_roundtrip.jl                              │
│                                                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
```

```
╭─────────────────────────────────────────────────────────────────────────────╮
│  Full test suite                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   $ julia --project -e 'using Pkg; Pkg.test("GeoDynamo")'                   │
│                                                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
```

### Check SHTnsKit Features

```julia
julia> using GeoDynamo

julia> info = get_shtnskit_version_info()
julia> @show info.version
julia> @show info.has_qst_transforms
julia> @show info.has_energy_functions
```

!!! warning "Missing Features?"
    Update SHTnsKit with:
    ```julia
    using Pkg; Pkg.update("SHTnsKit")
    ```

---

## Your First Simulation

### Minimal Example

```julia
using GeoDynamo

grid  = SphericalShellGrid(CPU(); lmax=32, nr=64, nr_inner=16)
model = GeodynamoModel(grid; Ek=1e-4, Ra=1e6, include_magnetic=true)

set!(model; temperature = RandomPerturbation(amplitude=0.1, lmax=10),
            magnetic    = AnalyticIC(:dipole; amplitude=1.0))

sim = Simulation(model; Δt=1e-5, stop_time=0.1, stop_iteration=10_000)
add_callback!(sim, sim -> @info("step", n=sim.model.clock.iteration);
              schedule=IterationInterval(100))
run!(sim)
```

### Running with MPI

```
╭─────────────────────────────────────────────────────────────────────────────╮
│  MPI execution (4 processes)                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   $ mpiexec -n 4 julia --project -e '                                       │
│       using GeoDynamo                                                       │
│       grid = SphericalShellGrid(nr = 64, lmax = 31)                         │
│       model = GeodynamoModel(grid; Ek = 1e-4, Ra = 1e6)                     │
│       simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)           │
│       run!(simulation)                                                      │
│   '                                                                         │
│                                                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
```

!!! success "Output"
    NetCDF files are written to `./output/` by default. See [Data Output](io.md) for details.

---

## Understanding the Physics

GeoDynamo.jl solves the Boussinesq MHD equations in a rotating spherical shell:

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

!!! info "Toroidal-Poloidal Decomposition"
    Fields are represented as:
    ```math
    \boldsymbol{u} = \nabla \times (T \hat{r}) + \nabla \times \nabla \times (P \hat{r})
    ```
    This automatically satisfies the divergence-free constraint.

---

## Boundary Conditions

### Default Setup (Shell Geometry)

| Field | Inner (ICB) | Outer (CMB) | Model keyword |
|:------|:------------|:------------|:----------|
| **Velocity** | No-slip | No-slip | `velocity_bcs = BoundaryConditions(inner = NoSlip(), outer = NoSlip())` |
| **Velocity** | Stress-free | Stress-free | `velocity_bcs = BoundaryConditions(inner = StressFree(), outer = StressFree())` |
| **Temperature** | Fixed T | Fixed T | `temperature_bcs = BoundaryConditions(inner = FixedTemperature(1), outer = FixedTemperature(0))` |
| **Magnetic** | Insulating | Insulating | (automatic) |
| **Composition** | Fixed C | Fixed C | `composition_bcs = BoundaryConditions(inner = FixedTemperature(1), outer = FixedTemperature(0))` |

### Custom Boundaries

```julia
using GeoDynamo

grid = SphericalShellGrid(nr = 64, lmax = 31)
model = GeodynamoModel(
    grid;
    temperature_bcs = BoundaryConditions(
        inner = FixedTemperature(1.0),
        outer = FixedTemperature(0.0),
    ),
)
simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
state = simulation.model.state

GeoDynamo.bcs.load_boundary_conditions!(state.temperature, GeoDynamo.TEMPERATURE, Dict(
    :inner => (:uniform, 1.0),
    :outer => (:dirichlet, 0.0),
))
```

---

## Initial Conditions

### Setting Up Fields

Use the high-level `set!` interface to apply initial conditions by field name:

```julia
using GeoDynamo

grid  = SphericalShellGrid(CPU(); lmax=32, nr=64, nr_inner=16)
model = GeodynamoModel(grid; include_magnetic = true)

# Oceananigans-style: set! dispatches to set_initial_condition! per field
set!(model;
     temperature = RandomPerturbation(amplitude=0.1, lmax=10),
     magnetic    = AnalyticIC(:dipole; amplitude=1.0))

simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
```

For lower-level access the field-specific helpers remain available:

```julia
state = simulation.model.state

# Temperature
set_temperature_ic!(state.temperature; profile = :conductive)
randomize_scalar_field!(state.temperature; amplitude = 1e-3)

# Velocity
randomize_vector_field!(state.velocity.velocity; amplitude = 1e-4)

# Magnetic Field
randomize_magnetic_field!(state.magnetic; amplitude = 1e-5)
```

### Loading from Files

```julia
# From restart file
read_restart!("output/geodynamo_shell_rank_0000_restart_1.nc")

# From snapshot
load_initial_conditions!("path/to/snapshot.nc")
```

---

## Workflow Overview

```
    ┌─────────────────────────────────────────────────────────┐
    │  1. GRID                                                │
    │     SphericalShellGrid(...)                             │
    └───────────────────────────┬─────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │  2. BOUNDARIES (optional)                               │
    │     bcs.load_boundary_conditions!(state.temperature,    │
    │         TEMPERATURE, Dict(...))                         │
    └───────────────────────────┬─────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │  3. MODEL + SIMULATION                                  │
    │     GeodynamoModel(grid; ...)                           │
    │     Simulation(model; Δt, stop_time)                    │
    │     set_temperature_ic!(...) / randomize_*(...)         │
    └───────────────────────────┬─────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │  4. RUN                                                 │
    │     run!(simulation)                                    │
    └───────────────────────────┬─────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │  5. ANALYZE                                             │
    │     Inspect NetCDF output in ./output/                  │
    └───────────────────────────┬─────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │  6. RESTART (optional)                                  │
    │     read_restart!(...) → run!(simulation)               │
    └─────────────────────────────────────────────────────────┘
```

---

## Next Steps

!!! tip "Where to Go From Here"
    | I want to... | Read... |
    |:-------------|:--------|
    | Understand all parameters | [Configuration](configuration.md) |
    | Learn about time integration | [Time Integration](timestepping.md) |
    | Explore spherical harmonics | [Spherical Harmonics](shtnskit.md) |
    | Configure output files | [Data Output](io.md) |
    | Add non-spherical boundaries | [Boundary Topography](topography.md) |
    | Contribute code | [Developer Guide](developer.md) |
