# Configuration & Parameters

`GeoDynamoParameters` collects every tunable knob controlling geometry, resolution, physics, and time-stepping. This page explains how the fields fit together and provides guidance for common setups.

```@docs
GeoDynamo.GeoDynamoParameters
```

## Geometry & Resolution

| Field | Meaning | Notes |
| --- | --- | --- |
| `geometry` | `:shell` or `:ball` | Drives boundary conditions, initialisation, and diagnostic logic. |
| `i_N` | Radial grid points | Applies to both outer-core (`oc_domain`) and inner-core (`ic_domain`) grids. |
| `i_L`, `i_M` | Maximum spherical harmonic degree/order | Communicated to SHTnsKit. `i_M` defaults to `i_L`. |
| `i_Th`, `i_Ph` | Physical θ/φ grid resolution | Overridden by SHTnsKit heuristics if incompatible with gauss grids. |
| `i_KL` | Radial finite-difference bandwidth | Controls stencil width for derivative operators. |

In practice, choose `i_L ≈ i_N` for balanced spectral/radial workload, and scale `i_Th`, `i_Ph` so SHTnsKit can allocate Gauss–Legendre grid points (`nlat ≥ i_L + 2`, `nlon ≥ 2*i_L + 1`).

## Physical Parameters

| Field | Description |
| --- | --- |
| `d_rratio` | Inner-to-outer radius ratio (shell runs). |
| `d_Ra`, `d_Ra_C` | Thermal and compositional Rayleigh numbers. |
| `d_E`, `d_Pr`, `d_Pm`, `d_Sc` | Ekman, Prandtl, magnetic Prandtl, Schmidt. |
| `d_Ro`, `d_q` | Rossby number (informational; solver derives Pm/E internally) and thermal diffusivity ratio. |
| `b_mag_impose`, `i_B` | Flags for imposed background fields and enabling magnetic evolution. |

These directly scale the nondimensionalised MHD equations summarised on the [overview](index.md) page.

!!! note "Rossby prefactor"
    When advancing Eq. (1) the code computes the Rossby prefactor as `d_Pm / d_E` (the ratio \( \mathrm{Pm}/E \)) so that the time derivative matches the \(E/\mathrm{Pm}\) mass matrix in the published formulation. The `d_Ro` parameter is retained for backwards compatibility but is not used during timestepping.

## Time Integration

| Field | Meaning |
| --- | --- |
| `ts_scheme` | `:cnab2`, `:eab2`, or `:erk2`. |
| `d_timestep` | Base Δt. Adjusts CFL and ETD caches. |
| `d_time` | Initial simulation clock. |
| `d_implicit` | θ parameter for CNAB2 implicit solve (θ = 0.5 → Crank–Nicolson). |
| `d_dterr` | Error tolerance for adaptive stepping (future use). |
| `d_courant` | CFL safety factor used by `compute_cfl_timestep!`. |
| `i_etd_m`, `d_krylov_tol` | Arnoldi dimension and residual tolerance for ETD/EAB2/ERK2 Krylov actions. |

See [Time Integration](timestepping.md) for scheme-specific details and recommended values.

## Boundary Conditions

Boundary options are set through the integer selectors in `GeoDynamoParameters` and, optionally, via external files loaded through the `BoundaryConditions` module.

| Field | Meaning | Built-in options |
| --- | --- | --- |
| `i_vel_bc` | Velocity boundary type | `1` no-slip (default), `2` stress-free. |
| `i_tmp_bc` | Temperature BC | `1` fixed temperature. Flux/mixed profiles can be supplied via boundary files. |
| `i_cmp_bc` | Composition BC | `1` fixed composition. |
| `i_poloidal_stress_iters` | Extra iterations enforcing stress-free poloidal constraints | Increase when using `i_vel_bc = 2`. |

When a boundary file is present under `config/boundaries/<field>_boundary.nc` (or a custom path passed to `BoundaryConditions.load_boundary_conditions!`), those data override the analytic defaults. Each file provides spherical-harmonic coefficients for the inner and outer surfaces together with a `type` flag per mode (`DIRICHLET`, `NEUMANN`, `ROBIN`). See the docstrings in `src/BoundaryConditions/` for field-specific formats.

After loading parameters, call:

```julia
using GeoDynamo
GeoDynamo.BoundaryConditions.load_boundary_conditions!(
    velocity = "config/boundaries/velocity_default.nc",
    temperature = "config/boundaries/thermal_flux.nc",
)
```

before creating the simulation state so the coefficients are cached in spectral space.

## Initial Conditions & Restarts

The `InitialConditions` module offers high-level helpers:

| Function | Purpose |
| --- | --- |
| `set_velocity_initial_conditions!` | Deterministic poloidal/toroidal seeds (solid-body, dipole, etc.). |
| `randomize_vector_field!` | Add random divergence-free perturbations. |
| `set_temperature_ic!`, `set_composition_ic!` | Conductive, mixed, or user-defined radial profiles. |
| `randomize_scalar_field!` | Thermal/compositional noise with configurable amplitude. |
| `load_initial_conditions!`, `save_initial_conditions` | Work with saved snapshots in NetCDF/HDF5. |

A typical recipe is:

```julia
state = initialize_simulation(Float64)
set_temperature_ic!(state.temperature; profile = :conductive)
randomize_scalar_field!(state.temperature; amplitude = 1e-3)
set_velocity_initial_conditions!(state.velocity; kind = :rest)
randomize_magnetic_field!(state.magnetic; amplitude = 1e-5)
```

For reproducible continuation runs use `write_restart!` and `read_restart!`, which store the full spectral state together with time metadata.

## Output & Restart

| Field | Meaning |
| --- | --- |
| `output_precision` | `:float32` or `:float64` for NetCDF data. |
| `independent_output_files` | When `true` each MPI rank writes its own (rank-indexed) files without synchronisation. |
| `i_save_rate2` | Output cadence in steps (legacy). Prefer `outputs_writer` tracker for fine control. |

Set `output_precision = :float32` to halve disk usage; diagnostics remain in `Float64` where accuracy is required.

## Managing Parameters

```julia
julia> params = GeoDynamoParameters(i_N = 96, i_L = 47, d_E = 3e-5);

julia> set_parameters!(params);          # updates global state

julia> save_parameters(params, "config/run_highres.jl")

julia> params2 = load_parameters("config/run_highres.jl");
```

All configuration files under `config/` are plain Julia scripts assigning constants. They are parsed by `load_parameters` and can be version-controlled per experiment.

## Next Steps

- [Time Integration](timestepping.md) – understand how the selected scheme uses the parameters above.
- [Data Output & Restart Files](io.md) – tailor precision and I/O strategy to your cluster. 
- [Developer Guide](developer.md) – customise domain decompositions or add new physics modules.
