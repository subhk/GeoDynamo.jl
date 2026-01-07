# Boundary Topography Coupling

This module implements linearized boundary topography effects for geodynamo simulations. Topography at the core-mantle boundary (CMB) and inner-core boundary (ICB) introduces coupling between different spherical harmonic modes through Gaunt-type tensor integrals.

## Overview

In real planetary cores, the boundaries are not perfectly spherical. The CMB has topography due to mantle convection, and the ICB may have topography from preferential solidification patterns. These deviations from sphericity introduce coupling between different spherical harmonic modes in the governing equations.

### Quick Start

```julia
using GeoDynamo

# Enable topography coupling
enable_topography!(
    epsilon = 0.01,      # Topography amplitude parameter
    velocity = true,     # Enable velocity BC corrections
    magnetic = true,     # Enable magnetic BC corrections
    thermal = true       # Enable thermal BC corrections
)

# Check if enabled
is_topography_enabled()  # returns true

# Disable when needed
disable_topography!()
```

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

### Linearized Geometry at the Boundary

The outward unit normal and normal derivative on the true surface, linearized at ``r=r_b``:

```math
\hat{n}_b = \hat{r} - \varepsilon \nabla_H h_b,
\quad \partial_n \approx \partial_r - \varepsilon (\nabla_H h_b)\cdot \nabla_H + \varepsilon h_b \partial_{rr}
```

The normal velocity evaluated on the reference sphere is:

```math
u_n \approx u_r - \varepsilon (\nabla_H h_b)\cdot u_t + \varepsilon h_b \partial_r u_r
```

### Taylor Expansion of Boundary Conditions

For any field ``f(r, \theta, \phi)`` with a boundary condition at ``r = r_b``, we expand about the reference sphere:

```math
f(r_b + \varepsilon h_b, \theta, \phi) = f(r_b, \theta, \phi) + \varepsilon h_b \frac{\partial f}{\partial r}\bigg|_{r_b} + O(\varepsilon^2)
```

This linearization introduces coupling through the product ``h_b \cdot \partial_r f``.

---

## Gaunt Tensor Formulation

The coupling between spherical harmonic modes is described by Gaunt-type integrals.

### Basic Gaunt Integral

The fundamental coupling coefficient is the Gaunt integral:

```math
G_{\ell m, \ell' m', LM} = \int Y_\ell^{m*} Y_{\ell'}^{m'} Y_L^M \, d\Omega
```

This integral satisfies selection rules:
1. **Azimuthal**: ``m = m' + M``
2. **Triangle inequality**: ``|\ell - \ell'| \leq L \leq \ell + \ell'``
3. **Parity**: ``\ell + \ell' + L`` must be even

The Gaunt integral can be computed analytically using Wigner 3j symbols:

```math
G = \sqrt{\frac{(2\ell+1)(2\ell'+1)(2L+1)}{4\pi}}
    \begin{pmatrix} \ell & \ell' & L \\ 0 & 0 & 0 \end{pmatrix}
    \begin{pmatrix} \ell & \ell' & L \\ -m & m' & M \end{pmatrix}
```

### Gradient Gaunt Integral

For slope coupling terms involving horizontal gradients ``\nabla_H``:

```math
G^{(\nabla)}_{\ell m, \ell' m', LM} = \int Y_\ell^{m*} \nabla_H Y_{\ell'}^{m'} \cdot \nabla_H Y_L^M \, d\Omega
```

This can be computed efficiently using the identity:

```math
G^{(\nabla)} = \frac{1}{2}\left[\ell'(\ell'+1) + L(L+1) - \ell(\ell+1)\right] G
```

### Cross Gaunt Integral

For toroidal-poloidal coupling:

```math
G^{(\times)}_{\ell m, \ell' m', LM} = \int Y_\ell^{m*} \hat{r} \cdot (\nabla_H Y_{\ell'}^{m'} \times \nabla_H Y_L^M) \, d\Omega
```

---

## Velocity Boundary Conditions

### Poloidal-Toroidal Decomposition

The velocity field is decomposed as:

```math
\mathbf{u} = \nabla \times (\mathcal{T} \hat{r}) + \nabla \times \nabla \times (\mathcal{P} \hat{r})
```

where ``\mathcal{P}`` and ``\mathcal{T}`` are the poloidal and toroidal scalars:

```math
\mathcal{P} = \sum_{\ell,m} P_{\ell m}(r) Y_\ell^m, \quad
\mathcal{T} = \sum_{\ell,m} T_{\ell m}(r) Y_\ell^m
```

### Impermeability Condition

The kinematic boundary condition requires zero normal velocity at the boundary:

```math
u_n = 0 \quad \text{at} \quad r = r_b + \varepsilon h_b
```

Expanding to first order in ``\varepsilon``:

```math
\boxed{u_r - \varepsilon (\nabla_H h_b)\cdot u_t + \varepsilon h_b \partial_r u_r = 0 \quad (r=r_b)}
```

The radial velocity in terms of poloidal scalar is:

```math
u_r = \frac{\ell(\ell+1)}{r^2} P_{\ell m}
```

The tangential velocity has both poloidal and toroidal parts:

```math
u_t = \partial_r (\nabla_H P) + \frac{1}{r}\nabla_H P + \hat{r}\times \nabla_H T
```

In spectral form (schematic):

```math
\frac{\ell(\ell+1)}{r_b^2} P_{\ell m}
+ \varepsilon \sum h_{LM} \Big[
G\,\partial_r\!\left(\frac{\ell'(\ell'+1)}{r_b^2} P_{\ell' m'}\right)
- G^{(\nabla)}\,\frac{\partial_r P_{\ell' m'}}{r_b^2}
- G^{(\times)}\,\frac{T_{\ell' m'}}{r_b^2}
\Big] = 0
```

### No-Slip Condition

For viscous boundaries, we require zero tangential velocity:

```math
u_\theta = u_\phi = 0 \quad \text{at} \quad r = r_b + \varepsilon h_b
```

The tangential velocity components involve both poloidal and toroidal parts. The linearized conditions become:

```math
u_t(r_b) + \varepsilon h_b \partial_r u_t(r_b) = U_{b,t}
```

For most cases ``U_{b,t}=0`` at the CMB; at the ICB it can include inner-core rotation.
The toroidal Dirichlet condition reduces to:

```math
T_{\ell m}|_{r_b} + \varepsilon \sum G_{\ell m, \ell' m', LM} h_{LM} \partial_r T_{\ell' m'}|_{r_b} = 0
```

### Stress-Free Condition

For stress-free boundaries (zero tangential stress):

```math
\sigma_{r\theta} = \sigma_{r\phi} = 0
```

In the linearized topographic form:

```math
\boxed{\partial_r \left(\frac{u_t}{r}\right) = \frac{\varepsilon}{r_b}(\nabla_H h_b)\,\partial_r u_r \quad (r=r_b)}
```

---

## Magnetic Field Boundary Conditions

### Poloidal-Toroidal Decomposition

The magnetic field is similarly decomposed:

```math
\mathbf{B} = \nabla \times (T \hat{r}) + \nabla \times \nabla \times (P \hat{r})
```

### CMB Insulating Condition

At the CMB, the mantle is effectively an electrical insulator. The magnetic field must match a potential field:

```math
\mathbf{B} = -\nabla \Phi_M \quad \text{for} \quad r > r_o
```

The matching conditions are:
1. **Toroidal vanishes**: ``T_{\ell m}|_{r_o} = 0``
2. **Poloidal matches potential**: ``\left(\partial_r + \frac{\ell+1}{r}\right) P_{\ell m}\bigg|_{r_o} = 0``

With topography corrections:

```math
\boxed{T_{\ell m}|_{r_o} + \varepsilon \sum G h_{LM} \partial_r T_{\ell' m'}|_{r_o} = 0}
```

```math
\boxed{\left(\partial_r + \frac{\ell+1}{r_o}\right) P_{\ell m}
      + \varepsilon \sum h_{LM} \left[
        G\,\partial_r\!\left(\partial_r P_{\ell' m'} + \frac{\ell'+1}{r_o}P_{\ell' m'}\right)
        - G^{(\times)}\,\frac{T_{\ell' m'}}{r_o^2}
      \right] = 0}
```

### ICB Insulating Condition

For an insulating inner core, the flat-sphere condition is
``(\partial_r - \ell/r_i) P_{\ell m} = 0`` and ``T_{\ell m}=0``, with topography couplings:

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
\Theta(r_b + \varepsilon h_b, \theta, \phi) = T_b(\theta, \phi)
```

Expanding:

```math
\boxed{\Theta_{\ell m}|_{r_b} + \varepsilon \sum G_{\ell m, \ell' m', LM} h_{LM} \partial_r \Theta_{\ell' m'}|_{r_b}
      = [T_b - T_{cond}(r_b)]_{\ell m} - \varepsilon \sum h_{LM} G_{\ell m,00,LM} \partial_r T_{cond}}
```

If the boundary temperature is uniform (``T_b`` = constant), the right-hand side is non-zero only for ``\ell = m = 0``.

### Neumann Condition (Fixed Heat Flux)

For fixed heat flux ``q_b`` at the boundary:

```math
-k\,\frac{\partial T}{\partial n}\bigg|_{r_b + \varepsilon h_b} = q_b(\theta, \phi)
```

The linearized condition:

```math
\boxed{\partial_r \Theta_{\ell m}
 - \varepsilon \sum h_{LM} G^{(\nabla)}_{\ell m,\ell' m',LM} \Theta_{\ell' m'}
 + \varepsilon \sum h_{LM} G_{\ell m,\ell' m',LM} \partial_{rr} \Theta_{\ell' m'}
 = -q_{b,\ell m}/k - \partial_r T_{cond}\,\delta_{\ell 0}\delta_{m 0}
   - \varepsilon \sum h_{LM} G_{\ell m,00,LM} \partial_{rr} T_{cond}}
```

The gradient Gaunt term ``G^{(\nabla)}`` appears because the normal direction varies with topography.

---

## Stefan Condition for ICB Evolution

The inner-core boundary can evolve due to phase change (solidification/melting). The Stefan condition relates boundary motion to heat flux imbalance.

### Heat Balance at ICB

```math
k_{ic} \frac{\partial T_{ic}}{\partial n} - k \frac{\partial T}{\partial n} = \rho L (V_b - u_n)
```

where:
- ``k_{ic}``, ``k`` = thermal conductivities (inner core, outer core)
- ``T_{ic}``, ``T`` = temperatures (inner core, outer core side)
- ``\rho`` = density at ICB
- ``L`` = latent heat of fusion
- ``V_b`` = boundary velocity = ``\varepsilon \partial_t h_i``
- ``u_n`` = normal fluid velocity

### Topography Evolution Equation

Rearranging for the topography evolution rate:

```math
\boxed{\varepsilon \frac{\partial h_i}{\partial t} = u_n + \frac{1}{\rho L}\left(k_{ic} \frac{\partial T_{ic}}{\partial n} - k \frac{\partial T}{\partial n}\right)}
```

Using the linearized operators at ``r=r_i``:

```math
\partial_n \approx \partial_r - \varepsilon \nabla_H h_i \cdot \nabla_H + \varepsilon h_i \partial_{rr},
\quad
u_n \approx u_r - \varepsilon \nabla_H h_i \cdot u_t + \varepsilon h_i \partial_r u_r
```

In spectral form:

```math
\varepsilon \frac{\partial h^i_{\ell m}}{\partial t} = u_{n,\ell m} + \frac{1}{\rho L} F_{\ell m}
```

where ``F_{\ell m}`` collects the heat flux contributions with topography corrections.

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

For efficiency, Gaunt tensors should be precomputed once at initialization:

```julia
# Create cache
gaunt = GauntTensorCache{Float64}(lmax_field, lmax_topo)

# Precompute all non-zero tensors (uses Wigner 3j by default)
precompute_gaunt_tensors!(gaunt; verbose=true, use_wigner=true)
```

The precomputation uses the analytic Wigner 3j formula, which is faster and more accurate than numerical quadrature.

### Creating Topography

```julia
# From spherical harmonic (e.g., Y_2^0 ellipsoidal shape)
cmb_topo = create_spherical_harmonic_topography(
    2, 0,              # l=2, m=0
    0.05,              # amplitude
    1.0,               # radius
    OUTER_BOUNDARY;
    lmax = 32
)

# Random topography with power spectrum
cmb_random = create_random_topography(
    l -> 0.01 / max(l, 1)^2,  # Power ~ l^(-2)
    1.0,                       # radius
    OUTER_BOUNDARY;
    lmax = 32,
    seed = 42
)

# From NetCDF file
cmb_file = load_topography_from_file("cmb_topo.nc", OUTER_BOUNDARY)

# Combine into TopographyData
topo_data = create_topography_data(
    cmb_coeffs = get_topography_coefficients(cmb_topo),
    cmb_radius = 1.0,
    lmax = 32,
    epsilon = 0.01
)
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

The following parameters control topography coupling in `GeoDynamoParameters`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `b_topography_enabled` | Bool | false | Master switch for topography |
| `d_topo_epsilon` | Float64 | 0.01 | Amplitude parameter ``\varepsilon`` |
| `i_topo_lmax` | Int | -1 | Max degree for topography (-1 = auto) |
| `b_topo_velocity` | Bool | true | Enable velocity corrections |
| `b_topo_magnetic` | Bool | true | Enable magnetic corrections |
| `b_topo_thermal` | Bool | true | Enable thermal corrections |
| `b_topo_slope_terms` | Bool | true | Include ``\nabla h`` terms |
| `b_topo_shift_terms` | Bool | true | Include ``h`` shift terms |
| `b_stefan_enabled` | Bool | false | Enable Stefan condition |
| `d_stefan_number` | Float64 | 1.0 | Stefan number |
| `s_topo_icb_file` | String | "" | ICB topography file |
| `s_topo_cmb_file` | String | "" | CMB topography file |

---

## Physical Considerations

### Choice of ``\varepsilon``

The linearization is valid when ``\varepsilon \ll 1``. Typical values:
- **Earth's CMB**: ``\varepsilon \sim 10^{-3}`` to ``10^{-2}`` (km-scale topography on ~3500 km radius)
- **Earth's ICB**: ``\varepsilon \sim 10^{-2}`` to ``10^{-1}`` (larger relative variations possible)

### Dominant Coupling Modes

Due to selection rules, not all modes couple:
- **Triangle inequality**: Mode ``(\ell, m)`` can only couple to modes ``(\ell', m')`` through topography mode ``(L, M)`` if ``|\ell - \ell'| \leq L \leq \ell + \ell'``
- **Azimuthal**: ``m = m' + M``

For low-degree topography (small ``L``), coupling is predominantly local in ``\ell``.

### Computational Cost

The number of non-zero Gaunt coefficients scales as ``O(\ell_{max}^3 L_{max})`` for topography truncated at ``L_{max}``. Precomputation is essential for efficiency.

---

## API Reference

```@docs
enable_topography!
disable_topography!
is_topography_enabled
get_topography_config
set_topography_config!
TopographyCouplingConfig
TopographyData
TopographyField
GauntTensorCache
precompute_gaunt_tensors!
apply_all_topography_corrections!
create_topography_data
load_topography_from_file
StefanState
update_icb_topography!
```

---

## References

1. Glatzmaier, G.A. & Roberts, P.H. (1995). "A three-dimensional convective dynamo solution with rotating and finitely conducting inner core and mantle." *Physics of the Earth and Planetary Interiors*, 91, 63-75.

2. Kuang, W. & Chao, B.F. (2001). "Topographic core-mantle coupling in geodynamo modeling." *Geophysical Research Letters*, 28, 1871-1874.

3. Edmonds, A.R. (1957). *Angular Momentum in Quantum Mechanics*. Princeton University Press. (For Gaunt integral and Wigner 3j symbols)

4. Christensen, U.R. & Wicht, J. (2015). "Numerical Dynamo Simulations." *Treatise on Geophysics*, 8, 245-277.
