# Boundary Topography

The `topography` submodule provides linearized boundary topography coupling.

## At a Glance

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

## Topography API

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
