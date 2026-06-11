# Boundary Topography Coupling

This module implements linearized boundary topography effects for geodynamo simulations. Topography at the core-mantle boundary (CMB) and inner-core boundary (ICB) introduces coupling between different spherical harmonic modes through Gaunt-type tensor integrals.

!!! tip "Quick Start"
    ```julia
    using GeoDynamo

    # Enable topography coupling
    enable_topography!(epsilon = 0.01, velocity = true, magnetic = true)

    # Check status
    is_topography_enabled()  # returns true

    # Disable when needed
    disable_topography!()
    ```

---

## Overview

In real planetary cores, the boundaries are not perfectly spherical:
- The **CMB** has topography due to mantle convection
- The **ICB** may have topography from preferential solidification patterns

These deviations introduce coupling between different spherical harmonic modes in the governing equations.

---

## Mathematical Formulation

### Linearized Boundary Topography

We consider small deviations from a spherical boundary at radius ``r_b``:

```math
r = r_b + \varepsilon h_b(\theta, \phi), \quad \varepsilon \ll 1
```

where:
- ``r_b`` is the reference spherical radius (CMB or ICB)
- ``\varepsilon`` is a small dimensionless amplitude parameter
- ``h_b(\theta, \phi)`` is the topography height function

The topography is expanded in spherical harmonics:

```math
h_b(\theta, \phi) = \sum_{L=0}^{L_{max}} \sum_{M=-L}^{L} h_{LM}^b Y_L^M(\theta, \phi)
```

### Surface Gradient Convention

We use the surface (tangential) gradient on the sphere:

```math
\nabla_H f = \hat{\theta}\,\frac{1}{r}\partial_{\theta} f
          + \hat{\phi}\,\frac{1}{r \sin\theta}\partial_{\phi} f,
\quad \nabla = \hat{r}\,\partial_r + \nabla_H
```

The surface Laplacian eigen-operator satisfies:

```math
L^2 \equiv -r^2 \nabla_H \cdot \nabla_H,
\quad L^2 Y_{\ell m} = \ell(\ell+1) Y_{\ell m}
```

### Linearized Geometry

The outward unit normal and normal derivative on the true surface, linearized at ``r=r_b``:

```math
\hat{n}_b = \hat{r} - \varepsilon \nabla_H h_b,
\quad \partial_n \approx \partial_r - \varepsilon (\nabla_H h_b)\cdot \nabla_H + \varepsilon h_b \partial_{rr}
```

The normal velocity evaluated on the reference sphere is:

```math
u_n \approx u_r - \varepsilon (\nabla_H h_b)\cdot u_t + \varepsilon h_b \partial_r u_r
```

---

## Gaunt Tensor Formulation

The coupling between spherical harmonic modes is described by Gaunt-type integrals.

### Basic Gaunt Integral

The fundamental coupling coefficient is:

```math
G_{\ell m, \ell' m', LM} = \int Y_\ell^{m*} Y_{\ell'}^{m'} Y_L^M \, d\Omega
```

!!! info "Selection Rules"
    | Rule | Condition |
    |:-----|:----------|
    | **Azimuthal** | ``m = m' + M`` |
    | **Triangle inequality** | ``\lvert\ell - \ell'\rvert \leq L \leq \ell + \ell'`` |
    | **Parity** | ``\ell + \ell' + L`` must be even |

The Gaunt integral can be computed analytically using Wigner 3j symbols:

```math
G = \sqrt{\frac{(2\ell+1)(2\ell'+1)(2L+1)}{4\pi}}
    \begin{pmatrix} \ell & \ell' & L \\ 0 & 0 & 0 \end{pmatrix}
    \begin{pmatrix} \ell & \ell' & L \\ -m & m' & M \end{pmatrix}
```

### Gradient Gaunt Integral

For slope coupling terms involving horizontal gradients:

```math
G^{(\nabla)}_{\ell m, \ell' m', LM} = \int Y_\ell^{m*} \nabla_H Y_{\ell'}^{m'} \cdot \nabla_H Y_L^M \, d\Omega
```

Computed efficiently using the identity:

```math
G^{(\nabla)} = \frac{1}{2}\left[\ell'(\ell'+1) + L(L+1) - \ell(\ell+1)\right] G
```

### Cross Gaunt Integral

For toroidal-poloidal coupling:

```math
G^{(\times)}_{\ell m, \ell' m', LM} = \int Y_\ell^{m*} \hat{r} \cdot (\nabla_H Y_{\ell'}^{m'} \times \nabla_H Y_L^M) \, d\Omega
```

!!! note
    Both ``G^{(\nabla)}`` and ``G^{(\times)}`` scale like ``1/r_b^2`` on the boundary. The implementation evaluates unit-sphere values and applies the scaling explicitly.

---

## Velocity Boundary Conditions

### Poloidal-Toroidal Decomposition

The velocity field is decomposed as:

```math
\mathbf{u} = \nabla \times (\mathcal{T} \hat{r}) + \nabla \times \nabla \times (\mathcal{P} \hat{r})
```

### Impermeability Condition

Zero normal velocity at the boundary:

```math
\boxed{u_r - \varepsilon (\nabla_H h_b)\cdot u_t + \varepsilon h_b \partial_r u_r = 0 \quad (r=r_b)}
```

The radial velocity in terms of poloidal scalar:

```math
u_r = \frac{\ell(\ell+1)}{r^2} P_{\ell m}
```

### No-Slip Condition

For viscous boundaries, zero tangential velocity:

```math
T_{\ell m}|_{r_b} + \varepsilon \sum G_{\ell m, \ell' m', LM} h_{LM} \partial_r T_{\ell' m'}|_{r_b} = 0
```

### Stress-Free Condition

Zero tangential stress:

```math
\boxed{\partial_r \left(\frac{u_t}{r}\right) = \frac{\varepsilon}{r_b}(\nabla_H h_b)\,\partial_r u_r \quad (r=r_b)}
```

---

## Magnetic Field Boundary Conditions

### CMB Insulating Condition

At the CMB, the mantle is an electrical insulator. The magnetic field must match a potential field.

**Toroidal vanishes:**
```math
\boxed{T_{\ell m}|_{r_o} + \varepsilon \sum G h_{LM} \partial_r T_{\ell' m'}|_{r_o} = 0}
```

**Poloidal matches potential:**
```math
\boxed{\left(\partial_r + \frac{\ell+1}{r_o}\right) P_{\ell m}
      + \varepsilon \sum h_{LM} \left[
        G\,\partial_r\!\left(\partial_r P_{\ell' m'} + \frac{\ell'+1}{r_o}P_{\ell' m'}\right)
        - G^{(\times)}\,\frac{T_{\ell' m'}}{r_o^2}
      \right] = 0}
```

### ICB Insulating Condition

For an insulating inner core, similar conditions apply with opposite radial matching:

```math
\left(\partial_r - \frac{\ell}{r_i}\right) P_{\ell m}
 + \varepsilon \sum h_{LM} \left[
   G\,\partial_r\!\left(\partial_r P_{\ell' m'} - \frac{\ell'}{r_i}P_{\ell' m'}\right)
   - G^{(\times)}\,\frac{T_{\ell' m'}}{r_i^2}
 \right] = 0
```

---

## Thermal Boundary Conditions

### Dirichlet Condition (Fixed Temperature)

For fixed temperature ``T_b`` at the boundary:

```math
\boxed{\Theta_{\ell m}|_{r_b} + \varepsilon \sum G_{\ell m, \ell' m', LM} h_{LM} \partial_r \Theta_{\ell' m'}|_{r_b}
      = [T_b - T_{cond}(r_b)]_{\ell m} - \varepsilon \sum h_{LM} G_{\ell m,00,LM} \partial_r T_{cond}}
```

### Neumann Condition (Fixed Heat Flux)

For fixed heat flux ``q_b``:

```math
\boxed{\partial_r \Theta_{\ell m}
 - \varepsilon \sum h_{LM} G^{(\nabla)}_{\ell m,\ell' m',LM} \Theta_{\ell' m'}
 + \varepsilon \sum h_{LM} G_{\ell m,\ell' m',LM} \partial_{rr} \Theta_{\ell' m'}
 = -q_{b,\ell m}/k}
```

---

## Stefan Condition for ICB Evolution

The inner-core boundary can evolve due to phase change (solidification/melting).

### Heat Balance at ICB

```math
k_{ic} \frac{\partial T_{ic}}{\partial n} - k \frac{\partial T}{\partial n} = \rho L (V_b - u_n)
```

| Symbol | Description |
|:-------|:------------|
| ``k_{ic}``, ``k`` | Thermal conductivities (inner core, outer core) |
| ``\rho`` | Density at ICB |
| ``L`` | Latent heat of fusion |
| ``V_b`` | Boundary velocity = ``\varepsilon \partial_t h_i`` |
| ``u_n`` | Normal fluid velocity |

### Topography Evolution

```math
\boxed{\varepsilon \frac{\partial h_i}{\partial t} = u_n + \frac{1}{\rho L}\left(k_{ic} \frac{\partial T_{ic}}{\partial n} - k \frac{\partial T}{\partial n}\right)}
```

---

## Implementation

### Data Structures

```julia
# Configuration
TopographyCouplingConfig(
    enabled = true,
    velocity_coupling = true,
    magnetic_coupling = true,
    thermal_coupling = true,
    stefan_enabled = false,
    epsilon = 0.01,
    include_slope_terms = true,
    include_shift_terms = true
)

# Topography field
TopographyField{T}(lmax, mmax, radius, location)

# Combined ICB + CMB data
TopographyData{T}(icb, cmb, gaunt_cache, epsilon)

# Gaunt tensor cache
GauntTensorCache{T}(lmax_field, lmax_topo)
```

### Precomputing Gaunt Tensors

!!! tip "Performance"
    Gaunt tensors should be precomputed once at initialization:

```julia
# Create cache
gaunt = GauntTensorCache{Float64}(lmax_field, lmax_topo)

# Precompute all non-zero tensors (uses Wigner 3j by default)
precompute_gaunt_tensors!(gaunt; verbose=true, use_wigner=true)
```

### Creating Topography

```julia
using GeoDynamo
const topo = GeoDynamo.bcs.topography

# From spherical harmonic (e.g., Y_2^0 ellipsoidal shape)
cmb_topo = topo.create_spherical_harmonic_topography(
    2, 0,              # l=2, m=0
    0.05,              # amplitude
    1.0,               # radius
    OUTER_BOUNDARY;
    lmax = 32
)

# Random topography with power spectrum. This helper is available from the
# topography submodule rather than the package root.
cmb_random = topo.create_random_topography(
    l -> 0.01 / max(l, 1)^2,  # Power ~ l^(-2)
    1.0,                       # radius
    OUTER_BOUNDARY;
    lmax = 32,
    seed = 42
)

# From NetCDF file
cmb_file = topo.load_topography_from_file("cmb_topo.nc", OUTER_BOUNDARY)
```

### Applying Corrections

```julia
# Apply all corrections during timestep
apply_all_topography_corrections!(
    (velocity = vel_field, magnetic = mag_field, temperature = temp_field),
    topo_data
)

# Or apply individually
apply_velocity_topography_correction!(vel_field, topo_data, config)
apply_magnetic_topography_correction!(mag_field, topo_data, config)
apply_thermal_topography_correction!(temp_field, topo_data, config)
```

---

## Configuration Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `topography_enabled` | Bool | `false` | Master switch for topography |
| `topography_epsilon` | Float64 | `0.01` | Amplitude parameter ε |
| `topography_degree` | Int | `-1` | Max degree for topography (-1 = auto) |
| `include_topography_velocity` | Bool | `true` | Enable velocity corrections |
| `include_topography_magnetic` | Bool | `true` | Enable magnetic corrections |
| `include_topography_thermal` | Bool | `true` | Enable thermal corrections |
| `include_topography_slope_terms` | Bool | `true` | Include ∇h terms |
| `include_topography_shift_terms` | Bool | `true` | Include h shift terms |
| `stefan_enabled` | Bool | `false` | Enable Stefan condition |
| `stefan_number` | Float64 | `1.0` | Stefan number |
| `icb_topography_file` | String | `""` | ICB topography file |
| `ocb_topography_file` | String | `""` | CMB topography file |

---

## Physical Considerations

### Choice of ε

The linearization is valid when ε ≪ 1.

| System | Typical ε | Notes |
|:-------|:----------|:------|
| **Earth's CMB** | 10⁻³ to 10⁻² | km-scale topography on ~3500 km radius |
| **Earth's ICB** | 10⁻² to 10⁻¹ | Larger relative variations possible |

### Dominant Coupling Modes

Due to selection rules, not all modes couple:

| Rule | Effect |
|:-----|:-------|
| **Triangle inequality** | Mode (ℓ, m) couples to (ℓ', m') through topography (L, M) only if \|ℓ - ℓ'\| ≤ L ≤ ℓ + ℓ' |
| **Azimuthal** | m = m' + M |

For low-degree topography (small L), coupling is predominantly local in ℓ.

### Computational Cost

The number of non-zero Gaunt coefficients scales as O(ℓ_max³ L_max). **Precomputation is essential for efficiency.**

---

## API Summary

| Function/Type | Description |
|:--------------|:------------|
| `enable_topography!` | Enable coupling with configuration options |
| `disable_topography!` | Disable all coupling |
| `is_topography_enabled` | Check if coupling is active |
| `TopographyCouplingConfig` | Configuration structure |
| `TopographyField` | Spectral representation of boundary topography |
| `TopographyData` | Container for ICB/CMB data |
| `GauntTensorCache` | Pre-computed Gaunt tensor storage |
| `precompute_gaunt_tensors!` | Compute and cache tensors |
| `compute_gaunt_tensor` | Compute single Gaunt integral numerically |
| `gaunt_on_the_fly` | Compute Gaunt integral (Wigner 3j or numerical) |
| `gradient_gaunt_from_basic` | Derive gradient Gaunt from basic Gaunt |
| `evaluate_spherical_harmonics_grid` | Evaluate Yₗᵐ on grid |
| `evaluate_spherical_harmonic_gradient_grid` | Evaluate ∇Yₗᵐ on grid |
| `StefanState` | State for Stefan condition ICB evolution |
| `update_icb_topography!` | Advance ICB topography in time |

For detailed API documentation, see the [API Reference](api.md).

---

## References

1. Glatzmaier, G.A. & Roberts, P.H. (1995). "A three-dimensional convective dynamo solution with rotating and finitely conducting inner core and mantle." *Physics of the Earth and Planetary Interiors*, 91, 63-75.

2. Kuang, W. & Chao, B.F. (2001). "Topographic core-mantle coupling in geodynamo modeling." *Geophysical Research Letters*, 28, 1871-1874.

3. Edmonds, A.R. (1957). *Angular Momentum in Quantum Mechanics*. Princeton University Press.

4. Christensen, U.R. & Wicht, J. (2015). "Numerical Dynamo Simulations." *Treatise on Geophysics*, 8, 245-277.
