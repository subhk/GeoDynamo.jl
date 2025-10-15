# API Reference

The table below lists the main entry points exported by `Geodynamo`. The listing is automatically generated during the documentation build.

## Main Module

```@autodocs
Modules = [Geodynamo]
Order   = [:module, :constant, :type, :macro, :function]
Filter = t -> !(t === Geodynamo.GeodynamoParameters)
```

## Boundary Conditions

```@autodocs
Modules = [Geodynamo.BoundaryConditions]
Order   = [:module, :constant, :type, :macro, :function]
```

## Initial Conditions

```@autodocs
Modules = [Geodynamo.InitialConditions]
Order   = [:module, :constant, :type, :macro, :function]
```

## Spherical Shell Geometry

```@autodocs
Modules = [Geodynamo.GeodynamoShell]
Order   = [:module, :constant, :type, :macro, :function]
```

## Solid Ball Geometry

```@autodocs
Modules = [Geodynamo.GeodynamoBall]
Order   = [:module, :constant, :type, :macro, :function]
```

For lower-level packages used internally (SHTnsKit, PencilArrays, MPI), refer to their respective documentation.
