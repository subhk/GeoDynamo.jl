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

This is the reference for all exported types and functions. The reference is
split across several pages to keep each one fast to load; use the table below to
jump to a section.

---

## Reference Map

!!! tip "Quick Navigation"

    | Page | Contents |
    |:-----|:---------|
    | [Solver & Timestep](api/solver.md) | Grids, model, simulation driver, time integration, output |
    | [Physics & Fields](api/fields.md) | Velocity, magnetic, temperature, composition fields and field infrastructure |
    | [Transforms & Spectral](api/transforms.md) | SHTnsKit configs, parallel layout, spectral operators |
    | [Internals](api/internals.md) | Remaining documented `GeoDynamo` symbols |
    | [Boundary Conditions](api/boundary-conditions.md) | `GeoDynamo.bcs` loading, interpolation, application |
    | [Boundary Topography](api/topography.md) | `GeoDynamo.bcs.topography` coupling |
    | [Initial Conditions](api/initial-conditions.md) | `GeoDynamo.InitialConditions` field setup |
    | [Shell Geometry](api/shell.md) | `GeoDynamo.GeoDynamoShell` domain setup |
    | [Ball Geometry](api/ball.md) | `GeoDynamo.GeoDynamoBall` domain setup |

---

## Module Layout

| Module | Description |
|:-------|:------------|
| `GeoDynamo` | Core types and simulation |
| `GeoDynamo.bcs` | Boundary conditions |
| `GeoDynamo.bcs.topography` | Non-spherical boundaries |
| `GeoDynamo.InitialConditions` | Field initialization |
| `GeoDynamo.GeoDynamoShell` | Shell geometry |
| `GeoDynamo.GeoDynamoBall` | Ball geometry |

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
