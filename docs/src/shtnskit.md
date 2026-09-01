# Spherical Harmonic Transforms

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Spectral ⟷ Physical Transforms                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    Physical Space              SHTnsKit              Spectral Space     │
│   ┌─────────────┐                                   ┌─────────────┐     │
│   │  f(θ,φ,r)   │  ──── analysis ────────────────▶  │   aₗₘ(r)    │     │
│   │             │                                   │             │     │
│   │  Grid Data  │  ◀──── synthesis ──────────────   │  Harmonics  │     │
│   └─────────────┘                                   └─────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

GeoDynamo.jl uses [SHTnsKit.jl](https://github.com/subhk/SHTnsKit.jl) v2.0.2 for all spherical harmonic operations.

---

## Quick Reference

!!! tip "Essential Functions"

    | Operation | Function |
    |:----------|:---------|
    | Spectral → Physical | `shtnskit_synthesis!(config, alm, f)` |
    | Physical → Spectral | `shtnskit_analysis!(config, f, alm)` |
    | Check features | `get_shtnskit_version_info()` |

---

## Configuration

The `SHTnsKitConfig` struct holds all transform state:

```julia
config = create_shtnskit_config(
    lmax = 63,              # Maximum spherical harmonic degree
    mmax = 63,              # Maximum order (defaults to lmax)
    nlat = 96,              # Latitude grid points (≥ lmax + 2)
    nlon = 192,             # Longitude grid points (≥ 2*lmax + 1)
    nr = 128,               # Radial grid points
    optimize_decomp = true  # Compatibility flag; topology uses GEODYNAMO_PROC_GRID
)
```

The configuration includes:
- SHTnsKit Gauss-Legendre grid setup
- PencilArrays decomposition for MPI parallelization
- Pre-computed FFT plans via PencilFFTs
- Transpose plans for pencil reorientations
- Scratch buffers for efficient memory reuse

### Feature Flags

GeoDynamo.jl exposes three feature flags that control v2 transform paths:

| Flag | Default | Description |
|:-----|:--------|:------------|
| `SHTNSKIT_USE_DISTRIBUTED` | `true` | Use native `dist_analysis`/`dist_synthesis` for MPI |
| `SHTNSKIT_USE_QST` | `true` | Use `synthesis_qst`/`analysis_qst` for 3D vectors |
| `SHTNSKIT_USE_SCRATCH_BUFFERS` | `true` | Pre-allocate scratch buffers |

They are compile-time `const`s declared in `src/transforms/spectral.jl`, not
runtime switches. Setting one to `false` requires a source edit and a
recompile; until then the alternative path (for QST, separate
`synthesis`/`synthesis_sphtor` calls) cannot be reached.

### Checking Available Features

```julia
info = get_shtnskit_version_info()
# Returns: (version, has_distributed_transforms, has_qst_transforms,
#           has_energy_functions, has_rotation_functions, has_inplace_transforms, ...)
```

---

## Core Transforms

### Scalar Fields

```julia
# Spectral → Physical (synthesis)
shtnskit_spectral_to_physical!(spectral_field, physical_field)

# Physical → Spectral (analysis)
shtnskit_physical_to_spectral!(physical_field, spectral_field)

# In-place variants - reduced allocations
shtnskit_synthesis_inplace!(config, alm, f_out)
shtnskit_analysis_inplace!(config, f, alm_out)
```

### Vector Fields (Toroidal-Poloidal)

```julia
# Spectral (T,P) → Physical (vr, vθ, vφ)
shtnskit_vector_synthesis!(toroidal, poloidal, vr, vtheta, vphi)

# Physical → Spectral (T,P)
shtnskit_vector_analysis!(vr, vtheta, vphi, toroidal, poloidal)
```

### QST Vector Fields

For full 3D vector handling with radial component:

```julia
# QST coefficients → Physical
shtnskit_qst_to_spatial!(config, Qlm, Slm, Tlm, vr, vtheta, vphi)

# Physical → QST coefficients
shtnskit_spatial_to_qst!(config, vr, vtheta, vphi, Qlm, Slm, Tlm)
```

!!! info "QST Decomposition"
    | Component | Symbol | Description |
    |:----------|:-------|:------------|
    | **Q** | Radial | Relates to radial velocity/field component |
    | **S** | Spheroidal/Poloidal | Divergent part of tangential flow |
    | **T** | Toroidal | Rotational part of tangential flow |

---

## Energy & Power Spectra

SHTnsKit v2 provides native energy spectrum computation:

```julia
# Scalar field energy spectrum by degree l
spectrum = compute_scalar_energy_spectrum(config, alm)  # Vector of length lmax+1

# Vector field kinetic energy spectrum
spectrum = compute_vector_energy_spectrum(config, Slm, Tlm)

# Total energies
E_scalar = compute_total_scalar_energy(config, alm)
E_vector = compute_total_vector_energy(config, Slm, Tlm)

# Enstrophy (mean square vorticity)
enstrophy = compute_enstrophy(config, Tlm)
```

!!! note
    All functions accept `real_field=true` (default) to account for the conjugate symmetry of real-valued fields.

---

## Spectral Differential Operators

### Horizontal Gradient

```julia
# Compute ∇_h f on the sphere
spectral_gradient!(config, Slm, grad_theta, grad_phi)
```

Uses `SHTnsKit.synthesis_grad`; transform failures are propagated instead of returning a silent zero gradient.

### Divergence and Vorticity

```julia
# From spheroidal potential → divergence coefficients
div_coeffs = extract_divergence_coefficients(config, Slm)

# From toroidal potential → vorticity coefficients
vort_coeffs = extract_vorticity_coefficients(config, Tlm)
```

### Horizontal Laplacian

The horizontal Laplacian on the unit sphere satisfies: ∇²_h Y_ℓ^m = -ℓ(ℓ+1) Y_ℓ^m

```julia
# Apply ∇²_h in spectral space
apply_horizontal_laplacian!(config, alm)  # In-place
apply_horizontal_laplacian!(config, alm; alm_out=result)

# Inverse Laplacian (for Poisson problems)
apply_inverse_horizontal_laplacian!(config, alm; regularize_l0=true)

# Gradient magnitude |∇_h f|²
grad_mag = compute_horizontal_gradient_magnitude(config, alm)
```

---

## Field Rotations

SHTnsKit v2 provides Wigner D-matrix rotations in spectral space:

### Basic Rotations

```julia
# Z-axis rotation (pure phase shift)
rotate_field_z!(config, alm, alpha)  # In-place
rotate_field_z!(config, alm, alpha; alm_out=result)

# Y-axis rotation (requires Wigner d-matrices)
rotate_field_y!(config, alm, beta; alm_out=result)

# Special 90° rotations (optimized)
rotate_field_90y!(config, alm; alm_out=result)
rotate_field_90x!(config, alm; alm_out=result)
```

### General Euler Rotation

```julia
# ZYZ convention: R = Rz(gamma) * Ry(beta) * Rz(alpha)
rotate_field_euler!(config, alm, alpha, beta, gamma; alm_out=result)
```

!!! tip
    Rotations in spectral space are exact and avoid interpolation artifacts.

---

## Spectral Filtering

### Custom Filters

```julia
# Apply any filter function (l, m) -> scale
my_filter(l, m) = l <= 32 ? 1.0 : 0.0  # Sharp truncation
apply_spectral_filter!(config, alm, my_filter)
```

### Exponential Filter (Dealiasing)

```julia
# Standard exponential filter for 2/3 dealiasing
apply_exponential_filter!(config, alm; order=16, cutoff=0.65)
```

The filter is: `exp(-α * (l/lmax)^order)` where α is chosen so `filter(cutoff*lmax) = 0.5`.

### Mode Truncation

```julia
# Truncate to lower resolution
truncate_spectral_modes!(config, alm, lmax_new=31, mmax_new=31)
```

---

## Threading Control

```julia
# SHTnsKit v2 uses Julia's launch-time thread count.
# Start Julia with: julia --threads=4 --project
set_shtnskit_threads(Threads.nthreads())  # Validate the active count
```

`set_shtnskit_threads` validates the active Julia thread count and reports how
to restart Julia if a different count is requested. This is useful for checking
hybrid MPI+threads configurations.

---

## Boundary Condition Transforms

For boundary data, use the cached transform utilities:

```julia
# Physical boundary data → spectral coefficients
coeffs = shtns_physical_to_spectral(physical_data, config)
coeffs = shtns_physical_to_spectral(physical_data, config; return_complex=true)

# Spectral → physical (inverse)
physical = shtns_spectral_to_physical(coeffs, config, nlat, nlon)

# Clear cached configurations (when grid changes)
clear_bc_shtns_config_cache!()
```

These functions cache SHTnsKit configurations to avoid repeated setup overhead.

---

## Performance Tips

| Tip | Details |
|:----|:--------|
| Use in-place transforms | Reduce allocations with `_inplace!` variants |
| Materialize distributed plan | `optimize_erk2_transforms!(config)` |
| Batch transforms | `batch_spectral_to_physical!` for multiple fields |
| Monitor usage | `get_shtnskit_performance_stats()` to verify features |
| Configure threading | Launch Julia with a thread count matched to the MPI configuration |

---

## Diagnostics

```julia
# Get performance statistics
stats = get_shtnskit_performance_stats()
# Returns: (library, version, parallelization, fft_backend, optimization,
#           distributed_transforms, qst_transforms, energy_functions)

# Validate decomposition efficiency
validate_pencil_decomposition(config)
```

---

## Verification

Verify that spherical harmonic transforms work correctly with a roundtrip test:

```
╭──────────────────────────────────────────────────────────────────────────────╮
│                                                                              │
│   $ julia --project test/shtnskit_roundtrip.jl                               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

This test validates:

| Test | Description |
|:-----|:------------|
| **Scalar roundtrip** | `spectral → physical → spectral` preserves coefficients |
| **Vector roundtrip** | Toroidal-poloidal `synthesis → analysis` cycle |
| **MPI consistency** | Distributed transforms match across ranks |

!!! tip "Expected Output"
    All tests should pass with relative errors below `1e-7`.

---

## API Summary

| Category | Functions |
|:---------|:----------|
| **Scalar Transforms** | `shtnskit_synthesis!`, `shtnskit_analysis!`, `*_inplace!` variants |
| **Vector Transforms** | `shtnskit_vector_synthesis!`, `shtnskit_vector_analysis!`, `shtnskit_qst_*` |
| **Energy Spectra** | `compute_*_energy_spectrum`, `compute_total_*_energy`, `compute_enstrophy` |
| **Operators** | `spectral_gradient!`, `apply_horizontal_laplacian!`, `extract_*_coefficients` |
| **Rotations** | `rotate_field_z!`, `rotate_field_y!`, `rotate_field_euler!`, `rotate_field_90*!` |
| **Filtering** | `apply_spectral_filter!`, `apply_exponential_filter!`, `truncate_spectral_modes!` |

See the [API Reference](api.md) for complete function documentation.
