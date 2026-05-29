# Solid Ball Geometry

The `GeoDynamoBall` module provides ball-specific domain setup.

## Overview

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

## Ball Geometry API

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
