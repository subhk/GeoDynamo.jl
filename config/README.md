# Geodynamo.jl Configuration

This directory contains parameter files for Geodynamo.jl simulations.

## Files

- `default_params.jl`: Default parameter values used by the package
- `template_params.jl`: Template file you can copy and modify for your simulations

## Usage

### Using Default Parameters
The package automatically loads `default_params.jl` when imported. No action needed.

### Using Custom Parameters
1. Copy `template_params.jl` to your desired location (e.g., `my_simulation_params.jl`)
2. Edit the parameter values as needed
3. Load your custom parameters:

```julia
using Geodynamo

# Load custom parameters
params = load_parameters("path/to/my_simulation_params.jl")
set_parameters!(params)

# Or initialize with custom parameters at startup
initialize_parameters("path/to/my_simulation_params.jl")
```

### Creating New Parameter Files
```julia
using Geodynamo

# Create a template
create_parameter_template("my_new_params.jl")

# Or save current parameters
params = get_parameters()
# Modify params as needed...
save_parameters(params, "my_modified_params.jl")
```

### Accessing Parameters in Code
```julia
# Modern way (recommended)
params = get_parameters()
println("Rayleigh number: ", params.d_Ra)

# Backward compatible way
println("Rayleigh number: ", d_Ra())

# Using macro
println("Rayleigh number: ", @param(d_Ra))
```

## Parameter Categories

### Grid Parameters
- `i_N`: Number of radial points
- `i_L`, `i_M`: Maximum spherical harmonic degree and wavenumber
- `i_Th`, `i_Ph`: Number of theta and phi points

### Physical Parameters  
- `d_Ra`: Rayleigh number
- `d_E`: Ekman number
- `d_Pr`: Prandtl number
- `d_Pm`: Magnetic Prandtl number

### Timestepping Parameters
- `d_timestep`: Time step size
- `i_maxtstep`: Maximum number of timesteps
- `d_dterr`: Error tolerance

See the parameter files for complete documentation of all available parameters.