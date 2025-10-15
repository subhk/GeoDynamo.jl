# Getting Started

This quick-start guide walks through preparing the environment, running the sample simulation, and inspecting the first output files. It assumes familiarity with Julia's package manager and a working MPI installation.

## 1. Install Julia & MPI

1. Install Julia **1.10 or 1.11** from [julialang.org/downloads](https://julialang.org/downloads/).
2. Install an MPI distribution (OpenMPI, MPICH, Intel MPI). Ensure `mpiexec` is on your `PATH`.
3. On macOS/Linux you might also need NetCDF libraries (`libnetcdf`, `libnetcdff`) for output. Package managers usually provide them (`brew install netcdf`, `apt install libnetcdf-dev`).

## 2. Clone the Repository

```bash
$ git clone https://github.com/subhk/Geodynamo.jl
$ cd Geodynamo.jl
```

If you are working in a monorepo that already contains SHTnsKit, keep the sibling directory `../SHTnsKit.jl` checked out so Geodynamo can develop against the local copy.

## 3. Instantiate the Environment

Run the following once to download dependencies and build MPI/SHTnsKit artefacts:

```julia
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

For development, it is convenient to activate the environment inside the Julia REPL:

```julia
julia> using Pkg
julia> Pkg.activate(".")
[ Info: activating project at /path/to/Geodynamo.jl
julia> Pkg.instantiate()
```

## 4. Verify SHTnsKit & MPI

Before launching full simulations, confirm the angular transforms and MPI topology are functioning:

```julia
julia --project test/shtnskit_roundtrip.jl
```

Or launch the package test-suite (requires MPI to be initialised correctly):

```bash
$ julia --project -e 'using Pkg; Pkg.test("Geodynamo")'
```

## 5. Minimal Example

```julia
julia> using Geodynamo

julia> params = GeodynamoParameters(
           geometry = :shell,
           i_N = 64,
           i_L = 31,
           i_M = 31,
           d_E = 1e-4,
           d_Ra = 1e6,
           d_Pr = 1.0,
           d_Pm = 1.0,
           i_B = 1,
           output_precision = :float32,
           independent_output_files = true,
       );

julia> set_parameters!(params);   # push to global configuration

julia> state = initialize_simulation(Float64);

julia> run_simulation!(state; t_end = 0.02)

# Save parameters for reproducibility
julia> save_parameters(params, "config/run_local_shell.jl");
```

### Running with MPI

Launch the same setup on four processes:

```bash
mpiexec -n 4 julia --project -e 'using Geodynamo; state = initialize_simulation(Float64); run_simulation!(state; t_end = 0.02)'
```

The driver will write NetCDF files into `./output/` (see the [Output guide](io.md) for details). When running under MPI, start the executable with `mpiexec` and the desired number of ranks.

## 6. What the Solver Advances

Geodynamo.jl integrates the nondimensional Boussinesq MHD system from Sreenivasan & Kar (2024), Eqs. (1)–(4). In magnetic-diffusion units the fields satisfy

```math
\frac{E}{\mathrm{Pm}}\frac{\partial \boldsymbol{u}}{\partial t}
  + (\nabla \times \boldsymbol{u}) \times \boldsymbol{u}
  + \hat{\boldsymbol{z}} \times \boldsymbol{u}
  = -\nabla p^\star
    + \frac{\mathrm{Pm}}{\mathrm{Pr}} \mathrm{Ra}\,T\,\boldsymbol{r}
    + (\nabla \times \boldsymbol{B}) \times \boldsymbol{B}
    + E \nabla^2 \boldsymbol{u},
```

with thermal and magnetic evolution,

```math
\frac{\partial T}{\partial t} + \boldsymbol{u} \cdot \nabla T = \frac{\mathrm{Pm}}{\mathrm{Pr}} \nabla^2 T, \qquad
\frac{\partial \boldsymbol{B}}{\partial t} = \nabla \times (\boldsymbol{u} \times \boldsymbol{B}) + \nabla^2 \boldsymbol{B},
```

and solenoidality constraints `∇·u = ∇·B = 0`. The code automatically applies the published prefactors—no additional scaling is required from the user beyond specifying `(E, Pm, Pr, Ra, …)` in `GeodynamoParameters`.

Both `u` and `B` are represented spectrally via toroidal (`T`) and poloidal (`P`) potentials (`u = ∇×(T \hat{r}) + ∇×∇×(P \hat{r})`, and likewise for `B`). This guarantees divergence-free fields while keeping the spherical-harmonic bookkeeping minimal; the helper functions in `InitialConditions.jl` operate directly on these toroidal/poloidal coefficients.

### Boundary Conditions

The default shell setup enforces:

- **Velocity:** no-slip at both the inner-core boundary (ICB) and core-mantle boundary (CMB) when `i_vel_bc = 1`; use `2` for stress-free.
- **Temperature:** fixed values at each boundary (`i_tmp_bc = 1`). Mixed/flux conditions can be configured through the boundary files in `config/`.
- **Magnetic field:** electrically insulating boundaries, matching to potential fields outside the fluid shell.
- **Composition (optional):** Dirichlet when `i_cmp_bc = 1`.

To override boundary data, create files under `config/boundaries/` (see comments in `src/BoundaryConditions/` for formats) and load them with `BoundaryConditions.load_boundary_conditions!` before `run_simulation!`.

### Initial Conditions

Geodynamo provides helpers in `InitialConditions.jl`:

```julia
using Geodynamo

state = initialize_simulation(Float64)

# Simple conductive profile with random perturbations
set_temperature_ic!(state.temperature; profile = :conductive)
randomize_scalar_field!(state.temperature; amplitude = 1e-3, rng = Random.default_rng())

# Small random velocity and magnetic seeds
randomize_vector_field!(state.velocity.velocity; amplitude = 1e-4)
randomize_magnetic_field!(state.magnetic; amplitude = 1e-5)
```

You can also load spectral snapshots via `load_initial_conditions!` or restart files with `read_restart!`. Always reapply `set_parameters!` before initialising so the grid matches the data you load.

## 7. Typical Workflow

1. **Create or load parameters** (`load_parameters`, `set_parameters!`).
2. **Set boundary data** if you need non-default conditions (`BoundaryConditions.load_boundary_conditions!`), otherwise the built-ins are applied.
3. **Specify initial conditions** using the helpers above or your own spectral fields.
4. **Initialise transforms** (either `create_shtnskit_config` manually or let `initialize_simulation` handle it).
5. **Advance in time** with `run_simulation!` or step manually using the timestep utilities.
6. **Inspect diagnostics** from the NetCDF output, or use `extras/spectral_to_physical.jl` to convert files.
7. **Restart** from saved state via `read_restart!` and resume the run.

For an overview of all configuration options, continue to [Configuration & Parameters](configuration.md).
