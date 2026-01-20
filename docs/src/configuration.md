# Configuration & Parameters

`GeoDynamoParameters` controls every tunable setting in GeoDynamo.jl—geometry, resolution, physics, time-stepping, and I/O. This page explains how the fields fit together and provides guidance for common setups.

```@docs
GeoDynamo.GeoDynamoParameters
```

---

## Quick Reference

!!! tip "Essential Parameters"
    For most simulations, you'll primarily configure:

    - **Geometry**: `geometry`, `i_N`, `i_L`, `i_M`
    - **Physics**: `d_E`, `d_Ra`, `d_Pr`, `d_Pm`
    - **Time**: `ts_scheme`, `d_timestep`
    - **Output**: `output_precision`, `independent_output_files`

---

## Geometry & Resolution

### Grid Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `geometry` | Symbol | `:shell` or `:ball` — determines boundary conditions and initialization |
| `i_N` | Int | Radial grid points (applies to outer-core and inner-core grids) |
| `i_L` | Int | Maximum spherical harmonic degree |
| `i_M` | Int | Maximum spherical harmonic order (defaults to `i_L`) |
| `i_Th` | Int | Physical θ grid resolution |
| `i_Ph` | Int | Physical φ grid resolution |
| `i_KL` | Int | Radial finite-difference bandwidth (stencil width) |

!!! note "Resolution Guidelines"
    - Choose `i_L ≈ i_N` for balanced spectral/radial workload
    - SHTnsKit requires `nlat ≥ i_L + 2` and `nlon ≥ 2*i_L + 1`
    - If `i_Th`/`i_Ph` are incompatible, SHTnsKit will override them

### SHTnsKit Transform Options

These flags control SHTnsKit v1.1.15 optimizations (set in `shtnskit_transforms.jl`):

| Flag | Default | Effect |
|:-----|:--------|:-------|
| `SHTNSKIT_USE_DISTRIBUTED` | `true` | Use native MPI-distributed transforms |
| `SHTNSKIT_USE_QST` | `true` | Use full QST decomposition for 3D vectors |
| `SHTNSKIT_USE_SCRATCH_BUFFERS` | `true` | Pre-allocate transform buffers |

**Check feature availability at runtime:**

```julia
info = get_shtnskit_version_info()
println("Version: ", info.version)
println("QST transforms: ", info.has_qst_transforms)
println("Energy functions: ", info.has_energy_functions)
```

See [Spherical Harmonics](shtnskit.md) for the complete transform API.

---

## Physical Parameters

### Dimensionless Numbers

| Parameter | Symbol | Description |
|:----------|:-------|:------------|
| `d_rratio` | — | Inner-to-outer radius ratio (shell geometry) |
| `d_Ra` | Ra | Thermal Rayleigh number |
| `d_Ra_C` | Ra_C | Compositional Rayleigh number |
| `d_E` | E | Ekman number |
| `d_Pr` | Pr | Prandtl number |
| `d_Pm` | Pm | Magnetic Prandtl number |
| `d_Sc` | Sc | Schmidt number |
| `d_Ro` | Ro | Rossby number (informational) |
| `d_q` | q | Thermal diffusivity ratio |

### Magnetic Field Control

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `b_mag_impose` | Bool | Enable imposed background magnetic field |
| `i_B` | Int | Enable magnetic evolution (1 = on, 0 = off) |

!!! info "Rossby Prefactor"
    When advancing the momentum equation, the code computes the Rossby prefactor as `Pm/E`. The `d_Ro` parameter is retained for backwards compatibility but is not used during timestepping.

---

## Time Integration

| Parameter | Description |
|:----------|:------------|
| `ts_scheme` | Time integration scheme: `:cnab2`, `:eab2`, or `:erk2` |
| `d_timestep` | Base Δt — affects CFL and ETD cache sizing |
| `d_time` | Initial simulation clock value |
| `d_implicit` | θ parameter for CNAB2 (θ = 0.5 → Crank–Nicolson) |
| `d_dterr` | Error tolerance for adaptive stepping (future use) |
| `d_courant` | CFL safety factor for `compute_cfl_timestep!` |
| `i_etd_m` | Arnoldi basis dimension for ETD/EAB2/ERK2 Krylov actions |
| `d_krylov_tol` | Residual tolerance for Krylov solvers |

!!! tip "Scheme Selection"
    | Scheme | Best For |
    |:-------|:---------|
    | **CNAB2** | Production dynamo runs, moderate timesteps |
    | **EAB2** | Strongly diffusive regimes (low E, Pm) |
    | **ERK2** | Wave propagation, accuracy-critical applications |

See [Time Integration](timestepping.md) for detailed scheme documentation.

---

## Boundary Conditions

For complete documentation of all boundary condition types and their physical interpretation, see the dedicated **[Boundary Conditions](boundary-conditions.md)** page.

### Quick Reference

| Parameter | Field | Options |
|:----------|:------|:--------|
| `i_vel_bc` | Velocity | `1` = no-slip, `2` = stress-free |
| `i_tmp_bc` | Temperature | `1` = Dirichlet/Dirichlet, `2` = Dirichlet/Neumann, `3` = Neumann/Dirichlet, `4` = Neumann/Neumann |
| `i_cmp_bc` | Composition | Same as `i_tmp_bc` |
| `i_poloidal_stress_iters` | Velocity | Extra iterations for stress-free poloidal constraints |

### Summary of BC Types

| Field | Available Options |
|:------|:------------------|
| **Velocity** | No-slip (T=0, ∂P/∂r=0), Stress-free (∂T/∂r=T/r, ∂²P/∂r²=0) |
| **Magnetic** | Insulating, Conducting inner core, Perfect conductor |
| **Temperature** | Fixed temperature (Dirichlet), Fixed flux (Neumann) |
| **Composition** | Fixed composition (Dirichlet), Fixed flux (Neumann) |

!!! note "Neumann-Neumann Special Case"
    When both boundaries use flux (Neumann) conditions for temperature or composition, the l=0 mode automatically uses Dirichlet at the inner boundary to pin the mean value.

### Loading Custom Boundaries

```julia
using GeoDynamo

GeoDynamo.bcs.load_boundary_conditions!(
    velocity = "config/boundaries/velocity_default.nc",
    temperature = "config/boundaries/thermal_flux.nc",
)
```

Call this **before** creating the simulation state so coefficients are cached in spectral space.

---

## Boundary Topography

For non-spherical boundaries, these parameters control topography coupling. See [Boundary Topography](topography.md) for full theory.

### Master Controls

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `b_topography_enabled` | Bool | `false` | Master switch for topography coupling |
| `d_topo_epsilon` | Float64 | `0.01` | Topography amplitude parameter ε |
| `i_topo_lmax` | Int | `-1` | Max spherical harmonic degree (-1 = auto) |

### Field-Specific Switches

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `b_topo_velocity` | Bool | `true` | Enable velocity BC corrections |
| `b_topo_magnetic` | Bool | `true` | Enable magnetic BC corrections |
| `b_topo_thermal` | Bool | `true` | Enable thermal BC corrections |
| `b_topo_slope_terms` | Bool | `true` | Include ∇h slope coupling terms |
| `b_topo_shift_terms` | Bool | `true` | Include h shift terms |

### Stefan Condition (Phase Change)

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `b_stefan_enabled` | Bool | `false` | Enable Stefan condition for ICB evolution |
| `d_stefan_number` | Float64 | `1.0` | Stefan number St = c_p ΔT / L |

### Topography Data Files

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `s_topo_icb_file` | String | Path to ICB topography NetCDF file |
| `s_topo_cmb_file` | String | Path to CMB topography NetCDF file |

### Example Configuration

**Via constructor:**

```julia
params = GeoDynamoParameters(
    b_topography_enabled = true,
    d_topo_epsilon = 0.01,
    b_topo_velocity = true,
    b_topo_magnetic = true,
    s_topo_cmb_file = "config/cmb_topography.nc"
)
```

**At runtime:**

```julia
enable_topography!(epsilon = 0.02, velocity = true, magnetic = true)
```

---

## Initial Conditions & Restarts

The `InitialConditions` module provides high-level setup helpers:

### Available Functions

| Function | Purpose |
|:---------|:--------|
| `set_velocity_initial_conditions!` | Deterministic poloidal/toroidal seeds (solid-body, dipole, etc.) |
| `randomize_vector_field!` | Add random divergence-free perturbations |
| `set_temperature_ic!` | Conductive, mixed, or user-defined radial profiles |
| `set_composition_ic!` | Composition initialization |
| `randomize_scalar_field!` | Thermal/compositional noise with configurable amplitude |
| `load_initial_conditions!` | Load from saved snapshots (NetCDF/HDF5) |
| `save_initial_conditions` | Save current state to file |

### Typical Setup

```julia
state = initialize_simulation(Float64)

# Temperature: conductive profile + perturbations
set_temperature_ic!(state.temperature; profile = :conductive)
randomize_scalar_field!(state.temperature; amplitude = 1e-3)

# Velocity: start at rest with small perturbations
set_velocity_initial_conditions!(state.velocity; kind = :rest)

# Magnetic: small random seed
randomize_magnetic_field!(state.magnetic; amplitude = 1e-5)
```

### Restarts

For reproducible continuation runs:

```julia
# Save state
write_restart!(state, tracker, metadata, config)

# Resume later
read_restart!("output/geodynamo_shell_rank_0000_restart_1.nc")
```

---

## Output Configuration

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `output_precision` | Symbol | `:float32` or `:float64` for NetCDF data |
| `independent_output_files` | Bool | Each MPI rank writes its own files (no synchronization) |
| `i_save_rate2` | Int | Output cadence in steps (legacy; prefer `outputs_writer` tracker) |

!!! tip "Storage Optimization"
    Use `output_precision = :float32` to halve disk usage. Diagnostics remain in `Float64` where accuracy is required.

See [Data Output & Restart Files](io.md) for complete I/O configuration.

---

## Managing Parameters

### Creating and Setting

```julia
# Create with specific values
params = GeoDynamoParameters(
    i_N = 96,
    i_L = 47,
    d_E = 3e-5
)

# Push to global state
set_parameters!(params)
```

### Saving and Loading

```julia
# Save to file
save_parameters(params, "config/run_highres.jl")

# Load from file
params = load_parameters("config/run_highres.jl")
```

!!! note
    Configuration files under `config/` are plain Julia scripts assigning constants. They can be version-controlled per experiment.

---

## Next Steps

| Goal | Resource |
|:-----|:---------|
| Understand boundary condition physics | [Boundary Conditions](boundary-conditions.md) |
| Understand time integration schemes | [Time Integration](timestepping.md) |
| Configure output and restarts | [Data Output & Restart Files](io.md) |
| Non-spherical boundary coupling | [Boundary Topography](topography.md) |
| Contribute to development | [Developer Guide](developer.md) |
