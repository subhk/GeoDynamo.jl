# Initial Conditions

!!! tip "Looking for the how-to guide?"
    This page is the auto-generated reference for the internal
    `InitialConditions` module. For setting up initial conditions with the
    public API (`set!`, `RandomPerturbation`, `AnalyticIC`, direct values),
    see the [Initial Conditions guide](../initial-conditions.md).

The `InitialConditions` module provides the low-level helpers behind the
public IC API.

## At a Glance

```
GeoDynamo.InitialConditions
├── Scalar Fields
│   ├── set_temperature_ic!
│   ├── set_composition_ic!
│   └── randomize_scalar_field!
│
├── Vector Fields
│   └── randomize_vector_field!
│
├── Magnetic Field
│   └── randomize_magnetic_field!
│
└── File I/O
    ├── load_initial_conditions!
    └── save_initial_conditions
```

!!! example "Usage (public API)"
    ```julia
    grid = SphericalShellGrid(nr = 64, lmax = 31)
    model = GeodynamoModel(grid; include_magnetic = true)

    set!(model;
         temperature = AnalyticIC(:conductive),
         magnetic    = RandomPerturbation(amplitude = 1e-5, lmax = 8))

    simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
    ```

## Initial Condition API

```@docs
GeoDynamo.InitialConditions.load_initial_conditions!
GeoDynamo.InitialConditions.generate_random_initial_conditions!
GeoDynamo.InitialConditions.set_analytical_initial_conditions!
GeoDynamo.InitialConditions.save_initial_conditions
GeoDynamo.InitialConditions.randomize_scalar_field!
GeoDynamo.InitialConditions.randomize_vector_field!
GeoDynamo.InitialConditions.randomize_magnetic_field!
```

```@autodocs
Modules = [GeoDynamo.InitialConditions]
Order   = [:module, :constant, :type, :macro, :function]
Filter = t -> !(t in (
    GeoDynamo.InitialConditions.load_initial_conditions!,
    GeoDynamo.InitialConditions.generate_random_initial_conditions!,
    GeoDynamo.InitialConditions.set_analytical_initial_conditions!,
    GeoDynamo.InitialConditions.save_initial_conditions,
    GeoDynamo.InitialConditions.randomize_scalar_field!,
    GeoDynamo.InitialConditions.randomize_vector_field!,
    GeoDynamo.InitialConditions.randomize_magnetic_field!,
))
```
