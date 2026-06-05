# Boundary Conditions

This page documents the boundary condition options for all field types in GeoDynamo.jl. Proper boundary conditions are essential for physically meaningful simulations of planetary core dynamics.

---

## Overview

GeoDynamo.jl supports boundary conditions at two radial surfaces:

- **Inner boundary (ICB)**: Inner core boundary or center (for ball geometry)
- **Outer boundary (CMB)**: Core-mantle boundary or outer surface

Each field type (velocity, magnetic, temperature, composition) has specific BC options appropriate for its physics.

---

## Velocity Boundary Conditions

Velocity fields use toroidal-poloidal decomposition: **u** = ∇×(T**r**) + ∇×∇×(P**r**)

### Available BC Types

| BC Type | Symbol | Toroidal (T) | Poloidal (P) | Physical Meaning |
|:--------|:-------|:-------------|:-------------|:-----------------|
| **No-slip** | `:no_slip` | T = 0 (Dirichlet) | ∂P/∂r = 0 | All velocity components vanish at boundary |
| **Stress-free** | `:stress_free` | ∂T/∂r = T/r (Neumann) | ∂²P/∂r² = 0 | Zero tangential stress, v_r = 0 |
| **Impermeable** | `:impermeable` | Unconstrained | P = 0 (Dirichlet) | Only radial velocity vanishes |

### Physical Interpretation

#### No-Slip (Rigid Boundary)
```
All velocity components = 0 at boundary
u_r = u_θ = u_φ = 0
```
- Appropriate for solid boundaries (e.g., rigid inner core, solid mantle)
- Most common choice for geodynamo simulations
- Toroidal: T = 0 directly sets tangential velocity to zero
- Poloidal: ∂P/∂r = 0 ensures radial velocity gradient vanishes

#### Stress-Free (Free-Slip)
```
Zero tangential stress at boundary
σ_rθ = σ_rφ = 0, with v_r = 0
```
- Appropriate for fluid-fluid interfaces or idealized boundaries
- Toroidal condition derived from: σ_rθ = η·r·∂(v_θ/r)/∂r = 0 → ∂T/∂r = T/r
- Poloidal condition: ∂²P/∂r² = 0 ensures stress-free radial component

### Usage

```julia
grid = SphericalShellGrid(nr = 64, lmax = 31)
model = GeodynamoModel(
    grid;
    velocity_bcs = BoundaryConditions(inner = NoSlip(), outer = NoSlip()),
)
simulation = Simulation(model; dt = 1e-5, stop_time = 0.02)
```

---

## Magnetic Field Boundary Conditions

Magnetic fields also use toroidal-poloidal decomposition: **B** = ∇×(T**r**) + ∇×∇×(P**r**)

### Available BC Types

| BC Type | Symbol | Toroidal | Poloidal Inner | Poloidal Outer | Status |
|:--------|:-------|:---------|:---------------|:---------------|:-------|
| **Insulating** | `:insulating` | T = 0 | (∂/∂r - l/r)P = 0 | (∂/∂r + (l+1)/r)P = 0 | ✅ Implemented (default, always applied) |
| **Conducting IC** | `:conducting_inner_core` | Continuous ∂T/∂r | Continuous ∂P/∂r | (∂/∂r + (l+1)/r)P = 0 | ✅ Implemented (shell + CNAB2; equal σ, no IC rotation) |
| **Perfect Conductor** | `:perfect_conductor` | T = 0 | P = 0 | P = 0 | 🚧 Not yet implemented |

!!! note "Implementation status"
    **Insulating** is the default and is embedded directly in the implicit time-stepping matrices (applied automatically whenever the magnetic field is enabled). **Conducting inner core** is available as an opt-in via `magnetic_inner_bc = :conducting_inner_core` (shell geometry, CNAB2 timestepper) — see scope notes below. **Perfect conductor** is not yet implemented. Design: `docs/superpowers/specs/2026-05-26-conducting-inner-core-design.md`.

### Physical Interpretation

#### Insulating Boundary
```
Exterior region has σ = 0 (no electrical conductivity)
Current cannot flow across boundary: J_n = 0
Field matches potential field solution outside
```

The l-dependent conditions ensure proper matching to potential field solutions:

- **Inner**: (∂/∂r - l/r)P = 0 → Field behaves as r^l (regular at origin)
- **Outer**: (∂/∂r + (l+1)/r)P = 0 → Field decays as r^{-(l+1)} (vanishes at infinity)

This is the standard choice for:
- CMB (core-mantle boundary): Mantle is nearly insulating
- ICB with no inner core: Insulating condition at inner boundary

#### Conducting Inner Core
```
Inner core has finite electrical conductivity
Magnetic field diffuses through inner core
B and ∂B/∂r continuous at ICB
```

For simulations with a solid, electrically conducting inner core:
- Field and its derivative must match across the ICB interface
- Outer boundary remains insulating (CMB)
- Inner core field satisfies regularity at r = 0

!!! note "Enabling a conducting inner core"
    Pass `magnetic_inner_bc = :conducting_inner_core` to `SolverParameters` /
    `GeodynamoModel` (requires `geometry = :shell`). The inner-core toroidal and
    poloidal scalars are then evolved by diffusion on the `[0, ri]` ball domain and
    coupled to the outer core at the ICB (continuity of the field and its radial
    derivative) via a Schur-complement / admittance method that reuses the implicit
    solver matrices.

    **Scope (current implementation):** equal conductivity `σ_ic = σ_oc`; inner core
    co-rotating with the frame (no differential rotation / advection); `CNAB2`
    timestepper only (conducting + `EAB2` raises an error). Variable
    `inner_core_conductivity_ratio` and inner-core rotation are not yet wired into
    the magnetic solve. Design + scope rationale:
    `docs/superpowers/specs/2026-05-26-conducting-inner-core-design.md`.

#### Perfect Conductor
```
Boundary material has σ → ∞
Tangential magnetic field = 0
B_θ = B_φ = 0 at boundary
```

Rarely used in geodynamo context, and **not yet implemented**.

### Usage

**Insulating** is the default — embedded in the implicit solver matrices and applied
automatically whenever the magnetic field is enabled. A **conducting inner core** is
opt-in via the `magnetic_inner_bc` parameter.

```julia
# Default: insulating magnetic BCs (applied automatically)
params = SolverParameters(; include_magnetic = true, geometry = :shell, ...)

# Opt in to a conducting inner core (shell geometry, CNAB2 timestepper):
params = SolverParameters(; include_magnetic = true, geometry = :shell,
                            magnetic_inner_bc = :conducting_inner_core, ...)
```

There is no `enforce_magnetic_boundary_constraints!` function. Perfect conductor is
not yet selectable.

---

## Temperature Boundary Conditions

Temperature is a scalar field with simpler BC structure.

### Available BC Types

| BC Type | Symbol | Mathematical Form | Physical Meaning |
|:--------|:-------|:------------------|:-----------------|
| **Fixed Temperature** | `:dirichlet` | T = T₀ | Prescribed temperature at boundary |
| **Fixed Heat Flux** | `:neumann` / `:flux` | ∂T/∂r = q | Prescribed heat flux at boundary |

### BC Combinations

| Inner | Outer | Use Case |
|:------|:------|:---------|
| Dirichlet | Dirichlet | Fixed temperatures at both boundaries |
| Dirichlet | Neumann | Fixed CMB temperature, prescribed ICB heat flux |
| Neumann | Dirichlet | Prescribed ICB heat flux, fixed CMB temperature |
| Neumann | Neumann | Prescribed heat flux at both boundaries |

### Special Case: Neumann-Neumann

!!! warning "Temperature Level Indeterminacy"
    When **both** boundaries have Neumann (flux) conditions, the temperature is only determined up to an additive constant. The governing equation only constrains temperature gradients, not the absolute level.

**Solution**: For Neumann-Neumann BCs, GeoDynamo.jl automatically uses **Dirichlet for the l=0 (mean) mode** at the inner boundary. This pins the average temperature while allowing flux conditions for all other modes.

```
l=0 mode (spherically symmetric mean): Dirichlet at inner boundary
l>0 modes (variations): Neumann at both boundaries
```

### Usage

```julia
# Fixed temperature boundaries
enforce_temperature_boundary_constraints!(temp_field, Dict(
    :inner => :dirichlet, :inner_value => 1.0,
    :outer => :dirichlet, :outer_value => 0.0
))

# Heat flux boundaries (l=0 Dirichlet applied automatically)
enforce_temperature_boundary_constraints!(temp_field, Dict(
    :inner => :flux, :inner_flux => 0.1,
    :outer => :flux, :outer_flux => 0.0
))
```

---

## Composition Boundary Conditions

Composition follows the same structure as temperature.

### Available BC Types

| BC Type | Symbol | Mathematical Form | Physical Meaning |
|:--------|:-------|:------------------|:-----------------|
| **Fixed Composition** | `:dirichlet` | C = C₀ | Prescribed composition at boundary |
| **Fixed Flux** | `:neumann` / `:flux` | ∂C/∂r = q | Prescribed compositional flux at boundary |

### Physical Interpretation

- **Dirichlet**: Fixed light element concentration (e.g., at ICB during inner core growth)
- **Neumann**: Prescribed mass flux (e.g., compositional release during freezing)
- **Zero flux** (∂C/∂r = 0): No compositional exchange across boundary

### Special Case: Neumann-Neumann

Same as temperature: when both boundaries have flux conditions, the l=0 mode uses Dirichlet at the inner boundary to pin the mean composition level.

### Usage

```julia
# Fixed composition at ICB, zero flux at CMB
enforce_composition_boundary_constraints!(comp_field, Dict(
    :inner => :dirichlet, :inner_value => 1.0,
    :outer => :flux, :outer_flux => 0.0
))
```

---

## Summary Table

| Field | BC Type | Inner Boundary | Outer Boundary |
|:------|:--------|:---------------|:---------------|
| **Velocity** | No-slip | T=0, ∂P/∂r=0 | T=0, ∂P/∂r=0 |
| | Stress-free | ∂T/∂r=T/r, ∂²P/∂r²=0 | ∂T/∂r=T/r, ∂²P/∂r²=0 |
| **Magnetic** | Insulating | T=0, (∂/∂r-l/r)P=0 | T=0, (∂/∂r+(l+1)/r)P=0 |
| | Conducting IC | ∂T/∂r continuous | T=0, (∂/∂r+(l+1)/r)P=0 |
| **Temperature** | Fixed T | T = T₀ | T = T₀ |
| | Fixed flux | ∂T/∂r = q | ∂T/∂r = q |
| | Both flux | **l=0: Dirichlet**, l>0: Neumann | Neumann |
| **Composition** | Fixed C | C = C₀ | C = C₀ |
| | Fixed flux | ∂C/∂r = q | ∂C/∂r = q |
| | Both flux | **l=0: Dirichlet**, l>0: Neumann | Neumann |

---

## Model Reference

```julia
grid = SphericalShellGrid(nr = 64, lmax = 31)
model = GeodynamoModel(
    grid;
    velocity_bcs = BoundaryConditions(inner = NoSlip(), outer = StressFree()),
    temperature_bcs = BoundaryConditions(inner = FixedTemperature(1.0), outer = FixedFlux(0.0)),
    composition_bcs = BoundaryConditions(inner = FixedTemperature(0.0), outer = FixedFlux(0.0)),
)
simulation = Simulation(model; dt = 1e-5, stop_time = 0.02)
```

### Programmatic API

```julia
# Velocity
enforce_velocity_boundary_constraints!(𝒰, :no_slip)
enforce_velocity_boundary_constraints!(𝒰, :stress_free)

# Magnetic
enforce_magnetic_boundary_constraints!(ℬ, :insulating)
enforce_magnetic_boundary_constraints!(ℬ, :conducting_inner_core)

# Temperature
enforce_temperature_boundary_constraints!(𝒯, Dict(
    :inner => :dirichlet, :inner_value => 1.0,
    :outer => :neumann, :outer_flux => 0.0
))

# Composition
enforce_composition_boundary_constraints!(𝔽, Dict(
    :inner => :flux, :inner_flux => 0.01,
    :outer => :dirichlet, :outer_value => 0.0
))
```

---

## Next Steps

| Topic | Resource |
|:------|:---------|
| Non-spherical boundaries | [Boundary Topography](topography.md) |
| Time integration with BCs | [Time Integration](timestepping.md) |
| Loading BC data from files | [Configuration](configuration.md) |
