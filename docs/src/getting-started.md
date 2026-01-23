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

# === Configure Parameters ===

params = GeoDynamoParameters(
    # Geometry
    geometry = :shell,         # Spherical shell
    i_N  = 64,                 # Radial points
    i_L  = 31,                 # Max spherical harmonic degree
    i_M  = 31,                 # Max spherical harmonic order

    # Physics
    d_E  = 1e-4,               # Ekman number
    d_Ra = 1e6,                # Rayleigh number
    d_Pr = 1.0,                # Prandtl number
    d_Pm = 1.0,                # Magnetic Prandtl number
    i_B  = 1,                  # Enable magnetic field

    # Output
    output_precision = :float32,
)

# === Initialize & Run ===

set_parameters!(params)
state = initialize_simulation(Float64)
run_simulation!(state; t_end = 0.02)

# Save for reproducibility
save_parameters(params, "config/my_run.jl")
```

### Running with MPI

```
╭─────────────────────────────────────────────────────────────────────────────╮
│  MPI execution (4 processes)                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   $ mpiexec -n 4 julia --project -e '                                       │
│       using GeoDynamo                                                       │
│       state = initialize_simulation(Float64)                                │
│       run_simulation!(state; t_end = 0.02)                                  │
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

| Field | Inner (ICB) | Outer (CMB) | Parameter |
|:------|:------------|:------------|:----------|
| **Velocity** | No-slip | No-slip | `i_vel_bc = 1` |
| **Velocity** | Stress-free | Stress-free | `i_vel_bc = 2` |
| **Temperature** | Fixed T | Fixed T | `i_tmp_bc = 1` |
| **Magnetic** | Insulating | Insulating | (automatic) |
| **Composition** | Fixed C | Fixed C | `i_cmp_bc = 1` |

### Custom Boundaries

```julia
using GeoDynamo

# Load custom boundary data before initialization
GeoDynamo.bcs.load_boundary_conditions!(
    velocity    = "config/boundaries/velocity.nc",
    temperature = "config/boundaries/thermal_flux.nc",
)

# Then initialize
state = initialize_simulation(Float64)
```

---

## Initial Conditions

### Setting Up Fields

```julia
using GeoDynamo
using Random

state = initialize_simulation(Float64)

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
    │  1. CONFIGURE                                           │
    │     GeoDynamoParameters(...) → set_parameters!(...)     │
    └───────────────────────────┬─────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │  2. BOUNDARIES (optional)                               │
    │     bcs.load_boundary_conditions!(...)                  │
    └───────────────────────────┬─────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │  3. INITIALIZE                                          │
    │     state = initialize_simulation(Float64)              │
    │     set_temperature_ic!(...) / randomize_*(...)         │
    └───────────────────────────┬─────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │  4. RUN                                                 │
    │     run_simulation!(state; t_end = ...)                 │
    └───────────────────────────┬─────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │  5. ANALYZE                                             │
    │     Inspect NetCDF output in ./output/                  │
    └───────────────────────────┬─────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │  6. RESTART (optional)                                  │
    │     read_restart!(...) → run_simulation!(...)           │
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
