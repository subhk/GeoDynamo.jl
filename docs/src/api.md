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

## Simulation

```julia
using GeoDynamo

grid  = SphericalShellGrid(CPU(); lmax=32, nr=64, nr_inner=16)
model = GeodynamoModel(grid; Ek=1e-4, Ra=1e6, include_magnetic=true)

sim = Simulation(model; Δt=1e-5, stop_time=0.1, stop_iteration=10_000)
add_callback!(sim, sim -> @info("step", n=sim.model.clock.iteration);
              schedule=IterationInterval(100))
run!(sim)
```

!!! note "Δt and dt"
    `Δt` is the canonical timestep keyword (Oceananigans convention); `dt` is
    accepted as an alias. `sim.Δt` reads and writes the same value as `sim.dt`.

`prettytime(t)` formats wall-clock durations ("2.341 seconds", "1.500 days").
Model time is nondimensional and prints compactly (e.g. `time = 0.25`).

Four callbacks are registered automatically on every new `Simulation`:
`stop_time_exceeded`, `stop_iteration_exceeded`, `wall_time_limit_exceeded`,
and `nan_checker`. A user callback registered under one of those names
replaces the built-in stop guard — pick a different name unless that is what
you want. Use `SpecifiedTimes(t1, t2, ...)` as a schedule to trigger output or
callbacks at exact model times.

---

## Boundary Conditions

Boundary conditions use Oceananigans-style names: `ValueBoundaryCondition`
(Dirichlet), `FluxBoundaryCondition` (Neumann), wrapped per field in
`FieldBoundaryConditions(inner=…, outer=…)`. The original names
(`FixedTemperature`, `FixedFlux`, `BoundaryConditions`) remain as aliases.
They can be passed per field or as one NamedTuple:

```julia
model = GeodynamoModel(grid;
    boundary_conditions = (
        temperature = FieldBoundaryConditions(
            inner = ValueBoundaryCondition(1.0),
            outer = ValueBoundaryCondition(0.0)),
    ))
```

`set!` accepts numbers, functions of `(r, θ, φ)`, and physical-grid arrays for
scalar fields:

```julia
set!(model; temperature = (r, θ, φ) -> 1 - r)
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
