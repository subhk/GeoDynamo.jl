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

### Public Solver API

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

### Timestep And Output API

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

### Physics Field API

```@docs
GeoDynamo.SHTnsVelocityFields
GeoDynamo.create_shtns_velocity_fields
GeoDynamo.compute_kinetic_energy
GeoDynamo.compute_reynolds_stress
GeoDynamo.SHTnsMagneticFields
GeoDynamo.create_shtns_magnetic_fields
GeoDynamo.compute_magnetic_energy
GeoDynamo.compute_ohmic_dissipation
GeoDynamo.SHTnsTemperatureField
GeoDynamo.create_shtns_temperature_field
GeoDynamo.compute_nusselt_number
GeoDynamo.compute_thermal_energy
GeoDynamo.get_temperature_statistics
GeoDynamo.SHTnsCompositionField
GeoDynamo.create_shtns_composition_field
GeoDynamo.compute_composition_rms
GeoDynamo.compute_composition_energy
GeoDynamo.get_composition_statistics
```

### Parallel And Transform API

```@docs
GeoDynamo.SHTnsKitConfig
GeoDynamo.create_shtnskit_config
GeoDynamo.shtnskit_spectral_to_physical!
GeoDynamo.shtnskit_physical_to_spectral!
GeoDynamo.shtnskit_vector_synthesis!
GeoDynamo.shtnskit_vector_analysis!
GeoDynamo.batch_shtnskit_transforms!
GeoDynamo.get_shtnskit_performance_stats
GeoDynamo.set_shtnskit_threads
GeoDynamo.get_shtnskit_version_info
GeoDynamo.get_comm
GeoDynamo.get_rank
GeoDynamo.get_nprocs
GeoDynamo.create_pencil_topology
GeoDynamo.create_transpose_plans
GeoDynamo.transpose_with_timer!
GeoDynamo.print_transpose_statistics
GeoDynamo.create_pencil_array
GeoDynamo.print_pencil_info
GeoDynamo.print_pencil_axes
GeoDynamo.optimize_communication_order
GeoDynamo.validate_radial_distribution
GeoDynamo.check_transform_synchronization
```

### Advanced Spectral Tools API

```@docs
GeoDynamo.compute_scalar_energy_spectrum
GeoDynamo.compute_vector_energy_spectrum
GeoDynamo.compute_total_scalar_energy
GeoDynamo.compute_total_vector_energy
GeoDynamo.compute_enstrophy
GeoDynamo.spectral_gradient!
GeoDynamo.extract_divergence_coefficients
GeoDynamo.extract_vorticity_coefficients
GeoDynamo.shtnskit_qst_to_spatial!
GeoDynamo.shtnskit_spatial_to_qst!
GeoDynamo.shtnskit_synthesis_inplace!
GeoDynamo.shtnskit_analysis_inplace!
GeoDynamo.rotate_field_z!
GeoDynamo.rotate_field_y!
GeoDynamo.rotate_field_90y!
GeoDynamo.rotate_field_90x!
GeoDynamo.rotate_field_euler!
GeoDynamo.apply_horizontal_laplacian!
GeoDynamo.apply_inverse_horizontal_laplacian!
GeoDynamo.compute_horizontal_gradient_magnitude
GeoDynamo.apply_spectral_filter!
GeoDynamo.apply_exponential_filter!
GeoDynamo.truncate_spectral_modes!
GeoDynamo.index_to_lm_fast
GeoDynamo.build_lm_lookup_tables
GeoDynamo.get_cached_buffer!
GeoDynamo.clear_buffer_cache!
```

### Performance And Runtime API

The older performance-manager and conversion/combination helper APIs are no
longer part of the supported public surface. The current documented workflow is
the solver API, the field/transform utilities below, and the explicit example
programs under `examples/`.

### Field Infrastructure And Operator API

```@docs
GeoDynamo.RadialDomain
GeoDynamo.SHTnsTorPolField
GeoDynamo.create_radial_domain
GeoDynamo.create_shtns_physical_field
GeoDynamo.create_shtns_vector_field
GeoDynamo.GradientWorkspace
GeoDynamo.create_gradient_workspace
GeoDynamo.clear_mode_index_cache!
GeoDynamo.clear_scalar_field_caches!
GeoDynamo.VelocityWorkspace
GeoDynamo.create_velocity_workspace
GeoDynamo.set_velocity_workspace!
GeoDynamo.compute_surface_flux
GeoDynamo.set_temperature_ic!
GeoDynamo.set_boundary_conditions!
GeoDynamo.set_internal_heating!
GeoDynamo.set_composition_ic!
GeoDynamo.set_composition_boundary_conditions!
GeoDynamo.create_velocity_green_matrices
GeoDynamo.solve_velocity_implicit_step!
GeoDynamo.solve_temperature_implicit_step!
GeoDynamo.solve_composition_implicit_step!
GeoDynamo.solve_magnetic_implicit_step!
```

```@autodocs
Modules = [GeoDynamo]
Order   = [:module, :constant, :type, :macro, :function]
Filter = t -> !(t in (
    GeoDynamo.initialize_simulation,
    GeoDynamo.run_simulation!,
    GeoDynamo.load_solver_parameters,
    GeoDynamo.create_solver_parameter_template,
    GeoDynamo.create_solver_backend,
    GeoDynamo.initialize_solver_state,
    GeoDynamo.advance_solver_step!,
    GeoDynamo.run_solver!,
    GeoDynamo.TimestepState,
    GeoDynamo.SHTnsImplicitMatrices,
    GeoDynamo.create_shtns_timestepping_matrices,
    GeoDynamo.apply_explicit_operator!,
    GeoDynamo.solve_implicit_step!,
    GeoDynamo.compute_timestep_error,
    GeoDynamo.OutputConfig,
    GeoDynamo.default_config,
    GeoDynamo.output_config_from_parameters,
    GeoDynamo.TimeTracker,
    GeoDynamo.create_time_tracker,
    GeoDynamo.should_output_now,
    GeoDynamo.should_restart_now,
    GeoDynamo.FieldInfo,
    GeoDynamo.SHTnsVelocityFields,
    GeoDynamo.create_shtns_velocity_fields,
    GeoDynamo.compute_kinetic_energy,
    GeoDynamo.compute_reynolds_stress,
    GeoDynamo.SHTnsMagneticFields,
    GeoDynamo.create_shtns_magnetic_fields,
    GeoDynamo.compute_magnetic_energy,
    GeoDynamo.compute_ohmic_dissipation,
    GeoDynamo.SHTnsTemperatureField,
    GeoDynamo.create_shtns_temperature_field,
    GeoDynamo.compute_nusselt_number,
    GeoDynamo.compute_thermal_energy,
    GeoDynamo.get_temperature_statistics,
    GeoDynamo.SHTnsCompositionField,
    GeoDynamo.create_shtns_composition_field,
    GeoDynamo.compute_composition_rms,
    GeoDynamo.compute_composition_energy,
    GeoDynamo.get_composition_statistics,
    GeoDynamo.SHTnsKitConfig,
    GeoDynamo.create_shtnskit_config,
    GeoDynamo.shtnskit_spectral_to_physical!,
    GeoDynamo.shtnskit_physical_to_spectral!,
    GeoDynamo.shtnskit_vector_synthesis!,
    GeoDynamo.shtnskit_vector_analysis!,
    GeoDynamo.batch_shtnskit_transforms!,
    GeoDynamo.get_shtnskit_performance_stats,
    GeoDynamo.set_shtnskit_threads,
    GeoDynamo.get_shtnskit_version_info,
    GeoDynamo.get_comm,
    GeoDynamo.get_rank,
    GeoDynamo.get_nprocs,
    GeoDynamo.create_pencil_topology,
    GeoDynamo.create_transpose_plans,
    GeoDynamo.transpose_with_timer!,
    GeoDynamo.print_transpose_statistics,
    GeoDynamo.create_pencil_array,
    GeoDynamo.print_pencil_info,
    GeoDynamo.print_pencil_axes,
    GeoDynamo.optimize_communication_order,
    GeoDynamo.validate_radial_distribution,
    GeoDynamo.check_transform_synchronization,
    GeoDynamo.compute_scalar_energy_spectrum,
    GeoDynamo.compute_vector_energy_spectrum,
    GeoDynamo.compute_total_scalar_energy,
    GeoDynamo.compute_total_vector_energy,
    GeoDynamo.compute_enstrophy,
    GeoDynamo.spectral_gradient!,
    GeoDynamo.extract_divergence_coefficients,
    GeoDynamo.extract_vorticity_coefficients,
    GeoDynamo.shtnskit_qst_to_spatial!,
    GeoDynamo.shtnskit_spatial_to_qst!,
    GeoDynamo.shtnskit_synthesis_inplace!,
    GeoDynamo.shtnskit_analysis_inplace!,
    GeoDynamo.rotate_field_z!,
    GeoDynamo.rotate_field_y!,
    GeoDynamo.rotate_field_90y!,
    GeoDynamo.rotate_field_90x!,
    GeoDynamo.rotate_field_euler!,
    GeoDynamo.apply_horizontal_laplacian!,
    GeoDynamo.apply_inverse_horizontal_laplacian!,
    GeoDynamo.compute_horizontal_gradient_magnitude,
    GeoDynamo.apply_spectral_filter!,
    GeoDynamo.apply_exponential_filter!,
    GeoDynamo.truncate_spectral_modes!,
    GeoDynamo.index_to_lm_fast,
    GeoDynamo.build_lm_lookup_tables,
    GeoDynamo.get_cached_buffer!,
    GeoDynamo.clear_buffer_cache!,
    GeoDynamo.RadialDomain,
    GeoDynamo.SHTnsTorPolField,
    GeoDynamo.create_radial_domain,
    GeoDynamo.create_shtns_physical_field,
    GeoDynamo.create_shtns_vector_field,
    GeoDynamo.GradientWorkspace,
    GeoDynamo.create_gradient_workspace,
    GeoDynamo.clear_mode_index_cache!,
    GeoDynamo.clear_scalar_field_caches!,
    GeoDynamo.VelocityWorkspace,
    GeoDynamo.create_velocity_workspace,
    GeoDynamo.set_velocity_workspace!,
    GeoDynamo.compute_surface_flux,
    GeoDynamo.set_temperature_ic!,
    GeoDynamo.set_boundary_conditions!,
    GeoDynamo.set_internal_heating!,
    GeoDynamo.set_composition_ic!,
    GeoDynamo.set_composition_boundary_conditions!,
    GeoDynamo.create_velocity_green_matrices,
    GeoDynamo.solve_velocity_implicit_step!,
    GeoDynamo.solve_temperature_implicit_step!,
    GeoDynamo.solve_composition_implicit_step!,
    GeoDynamo.solve_magnetic_implicit_step!,
))
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

    grid = SphericalShellGrid(nr = 64, lmax = 31)
    model = GeodynamoModel(grid)
    simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
    state = simulation.model.state

    GeoDynamo.bcs.load_boundary_conditions!(state.temperature, GeoDynamo.TEMPERATURE, Dict(
        :inner => (:uniform, 1.0),
        :outer => (:dirichlet, 0.0),
    ))
    ```

### Boundary Condition API

```@docs
GeoDynamo.bcs.load_boundary_conditions!
GeoDynamo.bcs.update_time_dependent_boundaries!
GeoDynamo.bcs.validate_boundary_files
GeoDynamo.bcs.get_current_boundaries
GeoDynamo.bcs.print_boundary_summary
GeoDynamo.bcs.get_boundary_module_info
GeoDynamo.bcs.SpectralBoundaryCoefficients
GeoDynamo.bcs.load_spectral_bc_from_file
GeoDynamo.bcs.store_bc_in_field!
GeoDynamo.bcs.get_bc_vectors_from_field
```

### Boundary Data And File API

```@docs
GeoDynamo.bcs.BoundaryData
GeoDynamo.bcs.BoundaryConditionSet
GeoDynamo.bcs.BoundaryCache
GeoDynamo.bcs.create_boundary_data
GeoDynamo.bcs.validate_boundary_compatibility
GeoDynamo.bcs.get_boundary_statistics
GeoDynamo.bcs.print_boundary_data_info
GeoDynamo.bcs.print_boundary_info
GeoDynamo.bcs.cache_boundary_data!
GeoDynamo.bcs.get_cached_data
GeoDynamo.bcs.clear_boundary_cache!
GeoDynamo.bcs.find_boundary_time_index
GeoDynamo.bcs.read_netcdf_boundary_data
GeoDynamo.bcs.write_netcdf_boundary_data
GeoDynamo.bcs.validate_netcdf_boundary_file
GeoDynamo.bcs.get_netcdf_file_info
```

### Boundary Application API

```@docs
GeoDynamo.bcs.initialize_boundary_conditions!
GeoDynamo.bcs.apply_boundary_conditions!
GeoDynamo.bcs.validate_field_boundary_compatibility
GeoDynamo.bcs.copy_boundary_conditions!
GeoDynamo.bcs.reset_boundary_conditions!
GeoDynamo.bcs.get_boundary_condition_summary
GeoDynamo.bcs.apply_temperature_boundaries!
GeoDynamo.bcs.apply_composition_boundaries!
GeoDynamo.bcs.update_boundary_conditions_for_timestep!
GeoDynamo.bcs.apply_boundary_conditions_to_rhs!
GeoDynamo.bcs.enforce_boundary_conditions_in_solution!
GeoDynamo.bcs.compute_boundary_condition_residual
GeoDynamo.bcs.log_boundary_condition_status
GeoDynamo.bcs.get_boundary_data
GeoDynamo.bcs.get_time_index
GeoDynamo.bcs.get_field_from_state
```

### Boundary Interpolation And Programmatic API

```@docs
GeoDynamo.bcs.interpolate_boundary_to_grid
GeoDynamo.bcs.create_interpolation_cache
GeoDynamo.bcs.interpolate_with_cache
GeoDynamo.bcs.validate_interpolation_grids
GeoDynamo.bcs.check_interpolation_bounds
GeoDynamo.bcs.get_interpolation_statistics
GeoDynamo.bcs.estimate_interpolation_error
GeoDynamo.bcs.Ylm
GeoDynamo.bcs.ProgrammaticBoundarySet
GeoDynamo.bcs.create_programmatic_boundary
GeoDynamo.bcs.create_time_dependent_programmatic_boundary
GeoDynamo.bcs.add_noise_to_boundary
GeoDynamo.bcs.smooth_boundary_data
GeoDynamo.bcs.create_programmatic_temperature_boundaries
GeoDynamo.bcs.create_programmatic_composition_boundaries
GeoDynamo.bcs.create_hybrid_temperature_boundaries
GeoDynamo.bcs.create_hybrid_composition_boundaries
GeoDynamo.bcs.load_temperature_boundaries_from_files
GeoDynamo.bcs.load_composition_boundaries_from_files
```

```@autodocs
Modules = [GeoDynamo.bcs]
Order   = [:module, :constant, :type, :macro, :function]
Filter = t -> !(t in (
    GeoDynamo.load_boundary_conditions!,
    GeoDynamo.update_time_dependent_boundaries!,
    GeoDynamo.validate_boundary_files,
    GeoDynamo.get_current_boundaries,
    GeoDynamo.print_boundary_summary,
    GeoDynamo.get_boundary_module_info,
    GeoDynamo.SpectralBoundaryCoefficients,
    GeoDynamo.load_spectral_bc_from_file,
    GeoDynamo.store_bc_in_field!,
    GeoDynamo.get_bc_vectors_from_field,
    GeoDynamo.bcs.BoundaryData,
    GeoDynamo.bcs.BoundaryConditionSet,
    GeoDynamo.bcs.BoundaryCache,
    GeoDynamo.bcs.create_boundary_data,
    GeoDynamo.bcs.validate_boundary_compatibility,
    GeoDynamo.bcs.get_boundary_statistics,
    GeoDynamo.bcs.print_boundary_data_info,
    GeoDynamo.bcs.print_boundary_info,
    GeoDynamo.bcs.cache_boundary_data!,
    GeoDynamo.bcs.get_cached_data,
    GeoDynamo.bcs.clear_boundary_cache!,
    GeoDynamo.bcs.find_boundary_time_index,
    GeoDynamo.bcs.read_netcdf_boundary_data,
    GeoDynamo.bcs.write_netcdf_boundary_data,
    GeoDynamo.bcs.validate_netcdf_boundary_file,
    GeoDynamo.bcs.get_netcdf_file_info,
    GeoDynamo.bcs.initialize_boundary_conditions!,
    GeoDynamo.bcs.apply_boundary_conditions!,
    GeoDynamo.bcs.validate_field_boundary_compatibility,
    GeoDynamo.bcs.copy_boundary_conditions!,
    GeoDynamo.bcs.reset_boundary_conditions!,
    GeoDynamo.bcs.get_boundary_condition_summary,
    GeoDynamo.bcs.apply_temperature_boundaries!,
    GeoDynamo.bcs.apply_composition_boundaries!,
    GeoDynamo.bcs.update_boundary_conditions_for_timestep!,
    GeoDynamo.bcs.apply_boundary_conditions_to_rhs!,
    GeoDynamo.bcs.enforce_boundary_conditions_in_solution!,
    GeoDynamo.bcs.compute_boundary_condition_residual,
    GeoDynamo.bcs.log_boundary_condition_status,
    GeoDynamo.bcs.get_boundary_data,
    GeoDynamo.bcs.get_time_index,
    GeoDynamo.bcs.get_field_from_state,
    GeoDynamo.bcs.interpolate_boundary_to_grid,
    GeoDynamo.bcs.create_interpolation_cache,
    GeoDynamo.bcs.interpolate_with_cache,
    GeoDynamo.bcs.validate_interpolation_grids,
    GeoDynamo.bcs.check_interpolation_bounds,
    GeoDynamo.bcs.get_interpolation_statistics,
    GeoDynamo.bcs.estimate_interpolation_error,
    GeoDynamo.bcs.Ylm,
    GeoDynamo.bcs.ProgrammaticBoundarySet,
    GeoDynamo.bcs.create_programmatic_boundary,
    GeoDynamo.bcs.create_time_dependent_programmatic_boundary,
    GeoDynamo.bcs.add_noise_to_boundary,
    GeoDynamo.bcs.smooth_boundary_data,
    GeoDynamo.bcs.create_programmatic_temperature_boundaries,
    GeoDynamo.bcs.create_programmatic_composition_boundaries,
    GeoDynamo.bcs.create_hybrid_temperature_boundaries,
    GeoDynamo.bcs.create_hybrid_composition_boundaries,
    GeoDynamo.bcs.load_temperature_boundaries_from_files,
    GeoDynamo.bcs.load_composition_boundaries_from_files,
))
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

### Topography API

```@docs
GeoDynamo.bcs.topography.TopographyCouplingConfig
GeoDynamo.bcs.topography.get_topography_config
GeoDynamo.bcs.topography.set_topography_config!
GeoDynamo.bcs.topography.enable_topography!
GeoDynamo.bcs.topography.disable_topography!
GeoDynamo.bcs.topography.is_topography_enabled
GeoDynamo.bcs.topography.TopographyField
GeoDynamo.bcs.topography.TopographyData
GeoDynamo.bcs.topography.GauntTensorCache
GeoDynamo.bcs.topography.precompute_gaunt_tensors!
GeoDynamo.bcs.topography.StefanState
GeoDynamo.bcs.topography.initialize_stefan_state!
GeoDynamo.bcs.topography.update_icb_topography!
```

```@autodocs
Modules = [GeoDynamo.bcs.topography]
Order   = [:module, :constant, :type, :macro, :function]
Filter = t -> !(t in (
    GeoDynamo.TopographyCouplingConfig,
    GeoDynamo.get_topography_config,
    GeoDynamo.set_topography_config!,
    GeoDynamo.enable_topography!,
    GeoDynamo.disable_topography!,
    GeoDynamo.is_topography_enabled,
    GeoDynamo.TopographyField,
    GeoDynamo.TopographyData,
    GeoDynamo.GauntTensorCache,
    GeoDynamo.precompute_gaunt_tensors!,
    GeoDynamo.StefanState,
    GeoDynamo.initialize_stefan_state!,
    GeoDynamo.update_icb_topography!,
))
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
    grid = SphericalShellGrid(nr = 64, lmax = 31)
    model = GeodynamoModel(grid; include_magnetic = true)
    simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
    state = simulation.model.state

    set_temperature_ic!(state.temperature; profile = :conductive)
    randomize_scalar_field!(state.temperature; amplitude = 1e-3)
    randomize_magnetic_field!(state.magnetic; amplitude = 1e-5)
    ```

### Initial Condition API

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

### Shell Geometry API

```@docs
GeoDynamo.GeoDynamoShell.ShellConfig
GeoDynamo.GeoDynamoShell.create_shell_pencils
GeoDynamo.GeoDynamoShell.create_shell_radial_domain
GeoDynamo.GeoDynamoShell.create_shell_velocity_fields
GeoDynamo.GeoDynamoShell.create_shell_temperature_field
GeoDynamo.GeoDynamoShell.create_shell_composition_field
GeoDynamo.GeoDynamoShell.create_shell_magnetic_fields
```

```@autodocs
Modules = [GeoDynamo.GeoDynamoShell]
Order   = [:module, :constant, :type, :macro, :function]
Filter = t -> !(t in (
    GeoDynamo.GeoDynamoShell.ShellConfig,
    GeoDynamo.GeoDynamoShell.create_shell_pencils,
    GeoDynamo.GeoDynamoShell.create_shell_radial_domain,
    GeoDynamo.GeoDynamoShell.create_shell_velocity_fields,
    GeoDynamo.GeoDynamoShell.create_shell_temperature_field,
    GeoDynamo.GeoDynamoShell.create_shell_composition_field,
    GeoDynamo.GeoDynamoShell.create_shell_magnetic_fields,
))
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

### Ball Geometry API

```@docs
GeoDynamo.GeoDynamoBall.BallConfig
GeoDynamo.GeoDynamoBall.create_ball_pencils
GeoDynamo.GeoDynamoBall.create_ball_radial_domain
GeoDynamo.GeoDynamoBall.create_ball_velocity_fields
GeoDynamo.GeoDynamoBall.create_ball_temperature_field
GeoDynamo.GeoDynamoBall.create_ball_composition_field
GeoDynamo.GeoDynamoBall.create_ball_magnetic_fields
GeoDynamo.GeoDynamoBall.enforce_ball_scalar_regularity!
GeoDynamo.GeoDynamoBall.enforce_ball_vector_regularity!
```

```@autodocs
Modules = [GeoDynamo.GeoDynamoBall]
Order   = [:module, :constant, :type, :macro, :function]
Filter = t -> !(t in (
    GeoDynamo.GeoDynamoBall.BallConfig,
    GeoDynamo.GeoDynamoBall.create_ball_pencils,
    GeoDynamo.GeoDynamoBall.create_ball_radial_domain,
    GeoDynamo.GeoDynamoBall.create_ball_velocity_fields,
    GeoDynamo.GeoDynamoBall.create_ball_temperature_field,
    GeoDynamo.GeoDynamoBall.create_ball_composition_field,
    GeoDynamo.GeoDynamoBall.create_ball_magnetic_fields,
    GeoDynamo.GeoDynamoBall.enforce_ball_scalar_regularity!,
    GeoDynamo.GeoDynamoBall.enforce_ball_vector_regularity!,
))
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
