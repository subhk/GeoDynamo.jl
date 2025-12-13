# API Reference

The table below lists the main entry points exported by `GeoDynamo`. The listing is automatically generated during the documentation build.

## Main Module

```@autodocs
Modules = [GeoDynamo]
Order   = [:module, :constant, :type, :macro, :function]
Filter = t -> !(t === GeoDynamo.GeoDynamoParameters)
```

## SHTnsKit Transforms (v1.1.15+)

### Core Transforms

```@docs
SHTnsKitConfig
create_shtnskit_config
shtnskit_spectral_to_physical!
shtnskit_physical_to_spectral!
shtnskit_vector_synthesis!
shtnskit_vector_analysis!
shtnskit_synthesis_inplace!
shtnskit_analysis_inplace!
```

### QST Vector Transforms

```@docs
shtnskit_qst_to_spatial!
shtnskit_spatial_to_qst!
```

### Energy & Spectrum Analysis

```@docs
compute_scalar_energy_spectrum
compute_vector_energy_spectrum
compute_total_scalar_energy
compute_total_vector_energy
compute_enstrophy
```

### Spectral Operators

```@docs
spectral_gradient!
extract_divergence_coefficients
extract_vorticity_coefficients
apply_horizontal_laplacian!
apply_inverse_horizontal_laplacian!
compute_horizontal_gradient_magnitude
```

### Field Rotations

```@docs
rotate_field_z!
rotate_field_y!
rotate_field_90y!
rotate_field_90x!
rotate_field_euler!
```

### Spectral Filtering

```@docs
apply_spectral_filter!
apply_exponential_filter!
truncate_spectral_modes!
```

### Configuration & Diagnostics

```@docs
get_shtnskit_version_info
get_shtnskit_performance_stats
set_shtnskit_threads
validate_pencil_decomposition
```

## Boundary Conditions

```@autodocs
Modules = [GeoDynamo.BoundaryConditions]
Order   = [:module, :constant, :type, :macro, :function]
```

## Initial Conditions

```@autodocs
Modules = [GeoDynamo.InitialConditions]
Order   = [:module, :constant, :type, :macro, :function]
```

## Spherical Shell Geometry

```@autodocs
Modules = [GeoDynamo.GeoDynamoShell]
Order   = [:module, :constant, :type, :macro, :function]
```

## Solid Ball Geometry

```@autodocs
Modules = [GeoDynamo.GeoDynamoBall]
Order   = [:module, :constant, :type, :macro, :function]
```

For lower-level packages used internally (SHTnsKit, PencilArrays, MPI), refer to their respective documentation.
