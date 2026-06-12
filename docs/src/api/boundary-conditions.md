# Boundary Conditions

The `bcs` submodule handles boundary condition loading, interpolation, and application.

## At a Glance

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

## Boundary Condition API

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

## Boundary Data And File API

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

## Boundary Application API

```@docs
GeoDynamo.bcs.initialize_boundary_conditions!
GeoDynamo.bcs.apply_boundary_conditions!
GeoDynamo.bcs.validate_field_boundary_compatibility
GeoDynamo.bcs.copy_boundary_conditions!
GeoDynamo.bcs.reset_boundary_conditions!
GeoDynamo.bcs.get_boundary_condition_summary
GeoDynamo.bcs.apply_temperature_boundaries!
GeoDynamo.bcs.apply_composition_boundaries!
```

## Boundary Interpolation And Programmatic API

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
