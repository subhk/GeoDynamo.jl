# GeoDynamo.jl Configuration

This directory contains parameter files for GeoDynamo.jl simulations.

## Files

- `default_params.jl`: Default parameter values used by the package
- `template_params.jl`: Template file you can copy and modify for your simulations

## Usage

### Using Default Parameters
The package automatically loads `default_params.jl` when imported. No action needed.

### Using Custom Parameters
Use the public constructors directly in your run script:

```julia
using GeoDynamo

grid = SphericalShellGrid(lmax=64, mmax=64, nlat=128, nlon=256, nr=96, nr_inner=32)
model = GeodynamoModel(
    grid;
    Ek=1e-4, Pr=1.0, Pm=2.0, Sc=1.0, Ra=1e6,
)
simulation = Simulation(model; Δt=1e-5, max_steps=10_000)
```

### Creating New Parameter Files
```julia
using GeoDynamo

create_parameter_template("my_new_params.jl")
```

## Parameter Categories

### Grid Parameters
- `nr`, `nr_inner`: Radial points for the outer core and inner core
- `lmax`, `mmax`: Maximum spherical harmonic degree and order
- `nlat`, `nlon`: Physical theta and phi grid points

### Physical Parameters  
- `Ra`: Rayleigh number
- `Ek`: Ekman number
- `Pr`: Prandtl number
- `Pm`: Magnetic Prandtl number

### Timestepping Parameters
- `timestep`: Time step size
- `max_steps`: Maximum number of timesteps
- `timestep_error`: Error tolerance

See the parameter files for complete documentation of all available parameters.
