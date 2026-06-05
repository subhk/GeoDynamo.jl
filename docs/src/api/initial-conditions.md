# Initial Conditions

The `InitialConditions` module provides helpers for setting up simulation fields.

## At a Glance

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
    grid = SphericalShellGrid(nr = 64, lmax = 31)
    model = GeodynamoModel(grid; include_magnetic = true)
    simulation = Simulation(model; dt = 1e-5, stop_time = 0.02)
    state = simulation.model.state

    set_temperature_ic!(state.temperature; profile = :conductive)
    randomize_scalar_field!(state.temperature; amplitude = 1e-3)
    randomize_magnetic_field!(state.magnetic; amplitude = 1e-5)
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
