# API Reference

The table below lists the main entry points exported by `GeoDynamo`. The listing is automatically generated during the documentation build.

## Main Module

```@autodocs
Modules = [GeoDynamo]
Order   = [:module, :constant, :type, :macro, :function]
Filter = t -> !(t === GeoDynamo.GeoDynamoParameters)
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
