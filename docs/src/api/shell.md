# Spherical Shell Geometry

The `GeoDynamoShell` module provides shell-specific domain setup.

## Overview

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

## Shell Geometry API

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
