# Solver & Timestep

The main `GeoDynamo` module exports the high-level simulation API — grids, the
physical model, the simulation driver — together with the time-integration and
output utilities.

## At a Glance

```
GeoDynamo
├── Types
│   ├── SphericalShellGrid     # Shell grid
│   ├── SphericalBallGrid      # Ball grid
│   ├── GeodynamoModel        # Physical model
│   ├── Clock                 # Tracks time and iteration count
│   └── Simulation            # Time integration wrapper
│
├── Simulation
│   ├── run!                  # Main time loop
│   ├── time_step!            # Advance a single step
│   ├── set!                  # Set initial conditions by field name
│   ├── fields                # NamedTuple of all model fields
│   ├── prognostic_fields     # NamedTuple of prognostic fields only
│   ├── add_callback!         # Register a scheduled callback
│   ├── Callback              # Scheduled callbacks
│   └── FieldWriter           # Scheduled field output
│
├── GPU Backend
│   ├── register_gpu_backend!  # Install a backend implementation
│   ├── with_gpu_backend       # Scoped backend override
│   └── gpu_backend_loaded     # Inspect backend availability
│
├── I/O
│   ├── write_fields!          # Output to NetCDF
│   ├── read_restart!          # Load checkpoint
│   └── CheckpointWriter       # Scheduled checkpoint output
│
└── Transforms
    ├── shtnskit_synthesis!    # Spectral → Physical
    ├── shtnskit_analysis!     # Physical → Spectral
    └── get_shtnskit_version_info
```

`with_gpu_backend(...)` can override scalar/vector transform callbacks plus the
transform-workspace scratch gather/store hooks used by GPU-marked solver
runtimes during tests or experimental backend integrations.

## Public Solver API

```@docs
GeoDynamo.SphericalShellGrid
GeoDynamo.SphericalBallGrid
GeoDynamo.GeodynamoModel
GeoDynamo.Simulation
GeoDynamo.Clock
GeoDynamo.run!
GeoDynamo.time_step!
GeoDynamo.set!
GeoDynamo.fields
GeoDynamo.prognostic_fields
GeoDynamo.add_callback!
```

## Lower-Level Solver API

The procedural entry points underlying the high-level `Simulation`/`run!` API.

```@docs
GeoDynamo.initialize_simulation
GeoDynamo.run_simulation!
GeoDynamo.create_solver_backend
GeoDynamo.initialize_solver_state
GeoDynamo.solver_step!
GeoDynamo.run_solver!
```

## Timestep And Output API

```@docs
GeoDynamo.TimestepState
GeoDynamo.SHTnsImplicitMatrices
GeoDynamo.create_shtns_timestepping_matrices
GeoDynamo.apply_explicit_operator!
GeoDynamo.solve_implicit_step!
GeoDynamo.compute_timestep_error
GeoDynamo.OutputConfig
GeoDynamo.default_config
GeoDynamo.output_config_from_parameters
GeoDynamo.TimeTracker
GeoDynamo.create_time_tracker
GeoDynamo.should_output_now
GeoDynamo.should_restart_now
GeoDynamo.FieldInfo
```
