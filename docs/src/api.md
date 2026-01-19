# API Reference

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GeoDynamo.jl API                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   GeoDynamo                                                             │
│   ├── Core Types & Functions                                            │
│   ├── bcs (Boundary Conditions)                                         │
│   │   └── topography                                                    │
│   ├── InitialConditions                                                 │
│   ├── GeoDynamoShell                                                    │
│   └── GeoDynamoBall                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

This page provides a comprehensive reference for all exported types and functions.

---

## Module Index

!!! tip "Quick Navigation"

    | Module | Description | Jump To |
    |:-------|:------------|:--------|
    | `GeoDynamo` | Core types and simulation | [Main Module](#main-module) |
    | `GeoDynamo.bcs` | Boundary conditions | [Boundary Conditions](#boundary-conditions) |
    | `GeoDynamo.bcs.topography` | Non-spherical boundaries | [Topography](#boundary-topography) |
    | `GeoDynamo.InitialConditions` | Field initialization | [Initial Conditions](#initial-conditions) |
    | `GeoDynamo.GeoDynamoShell` | Shell geometry | [Shell](#spherical-shell-geometry) |
    | `GeoDynamo.GeoDynamoBall` | Ball geometry | [Ball](#solid-ball-geometry) |

---

## Main Module

The main `GeoDynamo` module exports core types, simulation drivers, and utilities.

### At a Glance

```
GeoDynamo
├── Types
│   ├── GeoDynamoParameters    # Simulation configuration
│   ├── SimulationState        # Runtime state container
│   └── SHTnsKitConfig         # Transform configuration
│
├── Simulation
│   ├── initialize_simulation  # Create initial state
│   ├── run_simulation!        # Main time loop
│   └── set_parameters!        # Apply configuration
│
├── I/O
│   ├── write_fields!          # Output to NetCDF
│   ├── read_restart!          # Load checkpoint
│   └── save_parameters        # Save configuration
│
└── Transforms
    ├── shtnskit_synthesis!    # Spectral → Physical
    ├── shtnskit_analysis!     # Physical → Spectral
    └── get_shtnskit_version_info
```

```@autodocs
Modules = [GeoDynamo]
Order   = [:module, :constant, :type, :macro, :function]
Filter = t -> !(t === GeoDynamo.GeoDynamoParameters)
```

---

## Boundary Conditions

The `bcs` submodule handles boundary condition loading, interpolation, and application.

### At a Glance

```
GeoDynamo.bcs
├── load_boundary_conditions!     # Load from files
├── read_netcdf_boundary_data     # Read raw data
├── write_netcdf_boundary_data    # Write data
└── validate_netcdf_boundary_file # Validate structure
```

!!! example "Usage"
    ```julia
    using GeoDynamo

    GeoDynamo.bcs.load_boundary_conditions!(
        temperature = "thermal_bc.nc",
        velocity    = "velocity_bc.nc"
    )
    ```

```@autodocs
Modules = [GeoDynamo.bcs]
Order   = [:module, :constant, :type, :macro, :function]
```

---

## Boundary Topography

The `topography` submodule provides linearized boundary topography coupling.

### At a Glance

```
GeoDynamo.bcs.topography
├── enable_topography!           # Turn on coupling
├── disable_topography!          # Turn off coupling
├── is_topography_enabled        # Check status
├── TopographyCouplingConfig     # Configuration struct
├── TopographyField              # Topography data
├── GauntTensorCache             # Coupling coefficients
└── precompute_gaunt_tensors!    # Compute tensors
```

!!! example "Usage"
    ```julia
    using GeoDynamo

    enable_topography!(
        epsilon  = 0.01,
        velocity = true,
        magnetic = true
    )
    ```

```@autodocs
Modules = [GeoDynamo.bcs.topography]
Order   = [:module, :constant, :type, :macro, :function]
```

---

## Initial Conditions

The `InitialConditions` module provides helpers for setting up simulation fields.

### At a Glance

```
GeoDynamo.InitialConditions
├── Scalar Fields
│   ├── set_temperature_ic!
│   ├── set_composition_ic!
│   └── randomize_scalar_field!
│
├── Vector Fields
│   ├── set_velocity_initial_conditions!
│   └── randomize_vector_field!
│
├── Magnetic Field
│   └── randomize_magnetic_field!
│
└── File I/O
    ├── load_initial_conditions!
    └── save_initial_conditions
```

!!! example "Usage"
    ```julia
    state = initialize_simulation(Float64)

    set_temperature_ic!(state.temperature; profile = :conductive)
    randomize_scalar_field!(state.temperature; amplitude = 1e-3)
    randomize_magnetic_field!(state.magnetic; amplitude = 1e-5)
    ```

```@autodocs
Modules = [GeoDynamo.InitialConditions]
Order   = [:module, :constant, :type, :macro, :function]
```

---

## Spherical Shell Geometry

The `GeoDynamoShell` module provides shell-specific domain setup.

### Overview

```
          CMB (Core-Mantle Boundary)
         ╱                          ╲
        ╱    Outer Core (Fluid)      ╲
       ╱   ┌─────────────────────┐    ╲
      │    │                     │     │
      │    │  ICB (Inner Core    │     │
      │    │      Boundary)      │     │
      │    │                     │     │
       ╲   └─────────────────────┘    ╱
        ╲                            ╱
         ╲__________________________╱
```

Shell geometry simulates fluid dynamics between two concentric spherical boundaries—like Earth's outer core.

```@autodocs
Modules = [GeoDynamo.GeoDynamoShell]
Order   = [:module, :constant, :type, :macro, :function]
```

---

## Solid Ball Geometry

The `GeoDynamoBall` module provides ball-specific domain setup.

### Overview

```
              ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
             ╱                      ╲
            │     Full Sphere        │
            │                        │
            │    (No inner core)     │
            │                        │
             ╲                      ╱
              ╲____________________╱
```

Ball geometry is for full-sphere simulations without an inner boundary—like stellar cores or early planetary interiors.

```@autodocs
Modules = [GeoDynamo.GeoDynamoBall]
Order   = [:module, :constant, :type, :macro, :function]
```

---

## External Dependencies

GeoDynamo.jl builds on these Julia packages:

| Package | Purpose | Documentation |
|:--------|:--------|:--------------|
| **SHTnsKit.jl** | Spherical harmonic transforms | [GitHub](https://github.com/subhk/SHTnsKit.jl) |
| **PencilArrays.jl** | Domain decomposition | [Docs](https://jipolanco.github.io/PencilArrays.jl/) |
| **MPI.jl** | Message passing | [Docs](https://juliaparallel.org/MPI.jl/) |
| **NetCDF.jl** | File I/O | [Docs](https://alexander-barth.github.io/NCDatasets.jl/) |

---

## See Also

!!! info "Related Pages"

    | Topic | Page |
    |:------|:-----|
    | Parameter configuration | [Configuration](configuration.md) |
    | Time integration | [Time Integration](timestepping.md) |
    | Output formats | [Data Output](io.md) |
    | Transforms | [Spherical Harmonics](shtnskit.md) |
    | Development | [Developer Guide](developer.md) |
