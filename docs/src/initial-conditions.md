# Initial Conditions

This guide covers every way to set initial conditions on a
[`GeodynamoModel`](@ref): direct values (numbers, functions, arrays), the
four IC descriptor types, the `initial_conditions` model keyword, and file
load/save. For the auto-generated reference of the underlying module, see
[Initial Conditions API](api/initial-conditions.md).

## Overview

All paths go through one entry point, the Oceananigans-style `set!`:

```julia
set!(model; temperature = ..., velocity = ..., magnetic = ..., composition = ...)
```

Each keyword names a field; the value can be a **direct value** (scalar fields
only) or an **IC descriptor** (any field). `set!` can be called any time after
model construction and before `run!`.

| Field | Direct values (number / function / array) | Descriptors |
|:------|:------------------------------------------|:------------|
| `temperature` | ✓ | ✓ |
| `composition` | ✓ | ✓ |
| `velocity` | ✗ (toroidal–poloidal, see below) | ✓ |
| `magnetic` | ✗ (toroidal–poloidal, see below) | ✓ |

---

## Direct Values (scalar fields)

For `temperature` and `composition`, `set!` accepts the value itself —
no descriptor needed. The value is realized on the physical grid
`(nlat, nlon, nr)` and transformed to spectral coefficients with the standard
analysis path (MPI-safe; each rank fills its local portion).

### A number — uniform field

```julia
set!(model; temperature = 0.5)
```

### A function of `(r, θ, φ)`

The function receives **radius** `r` (dimensionless, `r_inner` to `r_outer`),
**colatitude** `θ ∈ [0, π]` (0 = north pole), and **longitude** `φ ∈ [0, 2π)`:

```julia
# conductive-like profile decreasing outward
set!(model; temperature = (r, θ, φ) -> 1 - r)

# equatorially enhanced composition with an m=4 longitudinal pattern
set!(model; composition = (r, θ, φ) -> 0.1 + 0.01 * sin(θ) * cos(4φ))
```

!!! warning "Resolution"
    The function is sampled on the model's physical grid and then projected
    onto spherical harmonics up to `lmax`/`mmax`. Structure sharper than the
    grid resolves is aliased — keep features representable at your truncation.

### An array on the physical grid

A 3-array matching the **local** physical grid size `(nlat, nlon, nr)`
(per-rank size under MPI):

```julia
arr = zeros(model.grid.nlat, model.grid.nlon, model.grid.nr)
arr[:, :, 1] .= 1.0          # hot inner boundary layer
set!(model; temperature = arr)
```

A wrong-size array throws an `ArgumentError` reporting both sizes.

### Why not velocity / magnetic?

Vector fields are stored as toroidal–poloidal potentials
(**u** = ∇×(T**r**) + ∇×∇×(P**r**)). A pointwise function `(r, θ, φ) -> value`
does not determine the two potentials, so direct values are rejected with an
`ArgumentError`. Use the descriptors below instead.

---

## IC Descriptors

Four descriptor types work for **every** field, including velocity and
magnetic:

### `RandomPerturbation`

Superimposes random spectral perturbations up to degree `lmax`:

```julia
RandomPerturbation(; amplitude, lmax, seed = nothing)
```

| Keyword | Required | Meaning |
|:--------|:---------|:--------|
| `amplitude` | yes | overall scale of the perturbation |
| `lmax` | yes | highest spherical-harmonic degree excited |
| `seed` | no | `Int` RNG seed for reproducible fields |

```julia
set!(model; temperature = RandomPerturbation(amplitude = 1e-3, lmax = 10, seed = 42))
```

!!! note
    Every IC path — descriptors and direct values alike — **replaces** the
    field's current content; applying two ICs to the same field keeps only the
    last one. To combine a deterministic background with noise, build the sum
    yourself in a function IC, e.g.
    `set!(model; temperature = (r, θ, φ) -> (1 - r) + 1e-3 * randn())`.

### `AnalyticIC`

Named analytic patterns, per field:

```julia
AnalyticIC(pattern::Symbol; amplitude = 1.0, parameters...)
```

| Field | Pattern | Extra keywords (defaults) | What it sets |
|:------|:--------|:--------------------------|:-------------|
| `temperature` | `:conductive` | — | linear profile `amplitude·(1 − r_frac)`, l=0 only |
| `temperature` | `:hot_blob` | `r_center = 0.5`, `blob_width = 0.2` | conductive background + Gaussian radial bump (l=0) |
| `magnetic` | `:dipole` | — | Earth-like l=1 poloidal `sin(πr)` + weak toroidal |
| `magnetic` | `:uniform_field` | `direction = :z` (or `:x`) | uniform axial (l=0) or transverse (l=1) poloidal field |
| `velocity` | `:convective` | — | small perturbations in degrees l=1–10, `sin(πr)` radial shape |
| `composition` | `:stratified` | `bottom_composition = 0.3`, `top_composition = 0.1` | linear stratification (l=0) |
| `composition` | `:blob` | `r_center = 0.3`, `blob_width = 0.2`, `blob_composition = 0.8` | background 0.1 + Gaussian compositional shell (l=0) |

```julia
set!(model;
     magnetic    = AnalyticIC(:dipole; amplitude = 1.0),
     composition = AnalyticIC(:blob; r_center = 0.35, blob_width = 0.15))
```

### `FileIC`

Load a field from a NetCDF file written by `save_initial_conditions` or a
compatible snapshot:

```julia
set!(model; temperature = FileIC("snapshots/temperature_ic.nc"))
```

### `ZeroIC`

Explicitly leave the field at zero (fields are zero after model construction,
so this is a documenting no-op):

```julia
set!(model; velocity = ZeroIC())
```

---

## At Model Construction

The same descriptors (and direct values for scalars) can be passed once via
the `initial_conditions` keyword:

```julia
model = GeodynamoModel(grid;
    Ra = 1e6,
    include_magnetic = true,
    initial_conditions = (
        temperature = RandomPerturbation(amplitude = 0.1, lmax = 10),
        magnetic    = AnalyticIC(:dipole; amplitude = 1.0),
    ))
```

This is equivalent to constructing the model and then calling `set!` with the
same keywords.

---

## Saving and Restarts

- `save_initial_conditions(path, fields...)` writes the current state for
  later `FileIC` use — see [Data Output & Restart Files](io.md).
- To continue a previous **run** (fields + clock + step), use
  `Simulation(model; restart_from = "checkpoint_dir")` instead of ICs — see
  [Data Output & Restart Files](io.md). ICs define a fresh start at `t = 0`;
  restarts resume time.

---

## Full Example

```julia
using GeoDynamo, MPI
MPI.Init()

grid = SphericalShellGrid(lmax = 31, nr = 64)
model = GeodynamoModel(grid; Ek = 1e-4, Ra = 1e6, Pr = 1.0)

# conductive background with small symmetry-breaking noise
set!(model; temperature = (r, θ, φ) -> (1 - r) + 1e-3 * (rand() - 0.5))

simulation = Simulation(model; Δt = 1e-5, stop_time = 0.02)
run!(simulation)
```

## Next Steps

| Goal | Resource |
|:-----|:---------|
| Set boundary conditions | [Boundary Conditions](boundary-conditions.md) |
| Configure output | [Data Output & Restart Files](io.md) |
| API reference | [Initial Conditions API](api/initial-conditions.md) |
