# Time Integration

GeoDynamo.jl provides three production-grade implicit-explicit (IMEX) time-stepping schemes optimized for the stiff diffusion terms in magnetohydrodynamic simulations.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Time Integration Schemes                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│   │   CNAB2     │     │    ExponentialAdamsBashforth2     │     │    ExponentialRungeKutta2     │              │
│   │  ─────────  │     │  ─────────  │     │  ─────────  │              │
│   │  Workhorse  │     │  Stiff OK   │     │  Accurate   │              │
│   │  A-stable   │     │  L-stable   │     │  L-stable   │              │
│   └─────────────┘     └─────────────┘     └─────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Selection

!!! tip "Which Scheme Should I Use?"

    | Scenario | Scheme | Why |
    |:---------|:-------|:----|
    | **Production dynamo runs** | CNAB2 | Robust, well-tested, low cost |
    | **Strong diffusion** (low E, Pm) | ExponentialAdamsBashforth2 | Exact linear integration, larger Δt |
    | **Wave studies / benchmarks** | ExponentialRungeKutta2 | Best transient accuracy |
    | **Development / debugging** | CNAB2 | Simplest to understand |

---

## Governing Equations

The geodynamo equations contain both linear diffusion terms, advanced
implicitly, and explicit forcing terms such as advection, Coriolis, buoyancy,
and Lorentz force. GeoDynamo.jl uses magnetic-diffusion time units, and the
velocity update keeps the Ekman number `E == Ek` as the mass coefficient on
the time derivative.

**Velocity:**
```math
E \frac{\partial \mathbf{u}}{\partial t}
= \underbrace{E \nabla^2 \mathbf{u}}_{\text{viscous diffusion}}
+ \underbrace{\mathbf{N}_u(\mathbf{u}, \mathbf{B}, T, C)}_{\text{explicit forcing}}
```

The explicit momentum forcing assembled by the code is
```math
\mathbf{N}_u =
E(\mathbf{u}\times\boldsymbol{\omega})
-\hat{\mathbf{z}}\times\mathbf{u}
+\frac{Pm}{Pr}Ra\,rT\,\hat{\mathbf{r}}
+\frac{Pm}{Sc}Ra_C\,rC\,\hat{\mathbf{r}}
+\frac{1}{Pm}(\nabla\times\mathbf{B})\times\mathbf{B},
\qquad \boldsymbol{\omega}=\nabla\times\mathbf{u}.
```

**Magnetic field:**
```math
\frac{\partial \mathbf{B}}{\partial t}
= \underbrace{\nabla^2 \mathbf{B}}_{\text{magnetic diffusion}}
+ \underbrace{\nabla \times (\mathbf{u} \times \mathbf{B})}_{\text{induction}}
```

**Temperature:**
```math
\frac{\partial T}{\partial t}
= \underbrace{\frac{Pm}{Pr} \nabla^2 T}_{\text{thermal diffusion}}
+ \underbrace{N_T(\mathbf{u}, T)}_{\text{explicit scalar forcing}},
\qquad
N_T = -\mathbf{u}\cdot\nabla T + Q_T.
```

`Q_T` is the optional internal-heating/source profile. The compositional field,
when enabled, follows the same scalar form with diffusivity `Pm/Sc` and
`N_C = -\mathbf{u}\cdot\nabla C`.
The magnetic and compositional terms are omitted from the velocity forcing when
those fields are disabled.

!!! info "Stiffness"
    The **stiffness** comes from diffusion terms with eigenvalues scaling as `ℓ(ℓ+1)/r²` in spherical harmonics—potentially very large for high-degree modes.

---

## Toroidal-Poloidal Decomposition

For incompressible flow (∇·**u** = 0) and solenoidal magnetic fields (∇·**B** = 0), vector fields are decomposed into **toroidal** and **poloidal** scalar potentials:

```math
\mathbf{u} = \nabla \times (\mathcal{T} \hat{\mathbf{r}}) + \nabla \times \nabla \times (\mathcal{P} \hat{\mathbf{r}})
```

```math
\mathbf{B} = \nabla \times (T \hat{\mathbf{r}}) + \nabla \times \nabla \times (P \hat{\mathbf{r}})
```

where:
- **𝒯** (toroidal potential) generates flow/field with no radial component
- **𝒫** (poloidal potential) generates flow/field with radial component

This decomposition automatically satisfies the divergence-free constraints.

### Velocity Equations

Taking the toroidal and poloidal components of the momentum equation yields separate evolution equations for each potential.

**Toroidal velocity (𝒯):**
```math
E\frac{\partial \mathcal{T}}{\partial t}
= E \left( \nabla^2 - \frac{\ell(\ell+1)}{r^2} \right) \mathcal{T}
+ N_{\mathcal{T}}
```

**Poloidal velocity (𝒫):**
```math
E(\partial_t - D_{\ell})W = N_W,
\qquad W = D_{\ell}\mathcal{P}
```

where:
- `E` is the Ekman number and velocity mass coefficient in magnetic time units
- `N_𝒯` and `N_𝒫` contain the nonlinear terms (Coriolis, advection, Lorentz force, buoyancy)
- `W = D_ℓ𝒫` is the solver's poloidal split variable
- The radial Laplacian operator in spectral space is:
```math
\nabla^2_\ell = \frac{\partial^2}{\partial r^2} + \frac{2}{r}\frac{\partial}{\partial r} - \frac{\ell(\ell+1)}{r^2}
```

### Magnetic Field Equations

The induction equation similarly splits into toroidal and poloidal components.

**Toroidal magnetic field (T):**
```math
\frac{\partial T}{\partial t} = \left( \nabla^2 - \frac{\ell(\ell+1)}{r^2} \right) T + [\nabla \times (\mathbf{u} \times \mathbf{B})]_T
```

**Poloidal magnetic field (P):**
```math
\frac{\partial P}{\partial t} = \left( \nabla^2 - \frac{\ell(\ell+1)}{r^2} \right) P + [\nabla \times (\mathbf{u} \times \mathbf{B})]_P
```

where:
- The diffusivity is 1.0 (magnetic diffusion time scaling: τ = L²/η)
- The subscripts `T` and `P` denote the toroidal and poloidal projections of the induction term

### Scalar Fields

Temperature and composition are **scalar** fields—they do not have toroidal/poloidal decomposition:

**Temperature:**
```math
\frac{\partial \Theta}{\partial t} = \frac{Pm}{Pr} \nabla^2 \Theta - \mathbf{u} \cdot \nabla \Theta
```

**Composition (if enabled):**
```math
\frac{\partial C}{\partial t} = \frac{Pm}{Sc} \nabla^2 C - \mathbf{u} \cdot \nabla C
```

### Summary of Timestepped Fields

| Field | Type | Components | Diffusivity Coefficient |
|:------|:-----|:-----------|:------------------------|
| Velocity | Vector | 𝒯 (toroidal), 𝒫 (poloidal) | `E` |
| Magnetic | Vector | T (toroidal), P (poloidal) | `1.0` |
| Temperature | Scalar | Θ | `Pm/Pr` |
| Composition | Scalar | C | `Pm/Sc` |

!!! note "Physical Interpretation"
    - **Toroidal velocity** represents zonal flows and azimuthal circulation
    - **Poloidal velocity** represents meridional overturning and radial motion
    - **Toroidal magnetic field** generates the azimuthal (φ) component of **B**
    - **Poloidal magnetic field** generates the dipole and higher multipole structure

---

## IMEX Framework

All schemes split the equations:

| Part | Treatment | Physical Terms |
|:-----|:----------|:---------------|
| **Linear (L)** | Implicit | Diffusion: ∇²**u**, ∇²**B**, ∇²T |
| **Nonlinear (N)** | Explicit | Advection, Coriolis, Lorentz force, buoyancy |

This allows large timesteps limited only by the CFL condition on advection, not by stiff diffusion.

---

## CNAB2: Crank–Nicolson Adams–Bashforth 2

*The workhorse scheme for production simulations.*

### Mathematical Formulation

```math
\frac{u^{n+1} - u^n}{\Delta t} = \theta L u^{n+1} + (1-\theta) L u^n + \frac{3}{2} N^n - \frac{1}{2} N^{n-1}
```

Rearranging:

```math
\underbrace{(I - \theta \Delta t L)}_{A} u^{n+1} = \underbrace{(I + (1-\theta) \Delta t L) u^n + \Delta t \left(\frac{3}{2} N^n - \frac{1}{2} N^{n-1}\right)}_{\text{RHS}}
```

where θ = 0.5 gives classic Crank–Nicolson (second-order, A-stable).

### Implementation

1. **Build RHS** via `build_rhs_cnab2!`:
   - Apply `(1-θ) Δt L` to current solution
   - Add Adams–Bashforth extrapolation of nonlinear terms

2. **Solve** the implicit system using banded LU factorization:
   - One factorization per spherical harmonic degree ℓ
   - Cached in `SHTnsImplicitMatrices` for reuse

3. **Apply boundary conditions** after solve

### Usage

```julia
grid = SphericalShellGrid(lmax=64, mmax=64, nr=96, nr_inner=24)
model = GeodynamoModel(grid; Ek=1e-5, Pr=1, Pm=2, Sc=1, Ra=1e7)

simulation = Simulation(
    model;
    Δt=1e-5,
    timestepper=CNAB2(theta=0.5),
)
```

### Properties

| Property | Value |
|:---------|:------|
| Order | 2nd (both linear and nonlinear) |
| Stability | A-stable for θ ≥ 0.5 |
| Memory | 1 previous nonlinear term |
| Cost | 1 LU solve per (ℓ,m) mode per field |

---

## ExponentialAdamsBashforth2: Exponential Adams–Bashforth 2

*Uses matrix exponentials to exactly integrate the stiff linear part.*

### Mathematical Formulation

```math
u^{n+1} = e^{\Delta t L} u^n + \Delta t \, \varphi_1(\Delta t L) \left(\frac{3}{2} N^n - \frac{1}{2} N^{n-1}\right)
```

where:

```math
\varphi_1(z) = \frac{e^z - 1}{z}
```

!!! info "Key Insight"
    The matrix exponential `exp(Δt L)` **exactly** propagates linear dynamics, removing any timestep restriction from diffusion. Nonlinear terms are still treated explicitly with AB2 extrapolation.

### Implementation Strategy

GeoDynamo now routes ExponentialAdamsBashforth2 through the solver-managed Krylov path internally.
The old dense ETD cache builder and manual LU-cache entry points are retained
only as deprecated compatibility wrappers; they are no longer the documented
workflow.

The Krylov approach:
- Builds an Arnoldi basis of dimension `m`
- Computes `exp(Δt L) v` and `φ₁(Δt L) v` in reduced space
- Avoids forming or storing full `nr × nr` exponential matrices

### Usage

```julia
grid = SphericalShellGrid(lmax=64, mmax=64, nr=96, nr_inner=24)
model = GeodynamoModel(grid; Ek=1e-5, Pr=1, Pm=2, Sc=1, Ra=1e7)

simulation = Simulation(
    model;
    Δt=1e-5,
    timestepper=ExponentialAdamsBashforth2(krylov_dimension=20, tolerance=1e-8),
)
```

### Properties

| Property | Value |
|:---------|:------|
| Order | 2nd for nonlinear, exact for linear |
| Stability | L-stable (excellent for stiff problems) |
| Memory | 1 previous nonlinear term + Krylov workspace |
| Cost | ~m matrix-vector products per (ℓ,m) mode |

---

## ExponentialRungeKutta2: Exponential Runge–Kutta 2

*Two-stage exponential integrator for maximum accuracy.*

### Mathematical Formulation

**Stage 1:** Compute intermediate state at t + Δt/2

```math
u^* = e^{\frac{\Delta t}{2} L} u^n + \frac{\Delta t}{2} \varphi_1\left(\frac{\Delta t}{2} L\right) N^n
```

**Stage 2:** Full step with midpoint nonlinear evaluation

```math
u^{n+1} = e^{\Delta t L} u^n + \Delta t \, \varphi_1(\Delta t L) N^n + \Delta t \, \varphi_2(\Delta t L) \left[N(u^*) - N^n\right]
```

where:

```math
\varphi_2(z) = \frac{e^z - 1 - z}{z^2}
```

### Implementation

Uses dedicated cache structures:

```julia
struct ERK2Cache{T}
    exp_half::Matrix{T}    # exp(Δt/2 · L)
    exp_full::Matrix{T}    # exp(Δt · L)
    phi1_half::Matrix{T}   # φ₁(Δt/2 · L)
    phi1_full::Matrix{T}   # φ₁(Δt · L)
    phi2_full::Matrix{T}   # φ₂(Δt · L)
end
```

Helper functions:
- `erk2_prepare_field!` — Stage 1 computation
- `erk2_finalize_field!` — Stage 2 and final update

### Cache Management

**Precompute caches (recommended for production):**

```bash
julia --project scripts/precompute_erk2_caches.jl \
    --dt=1e-5 \
    --fields=temperature,vel_tor,vel_pol,mag_tor,mag_pol
```

**Load precomputed caches:**

```julia
load_erk2_cache_bundle!(state.erk2_caches, "erk2_caches.jld2")
```

### Diagnostics

```julia
# Enable stage residual monitoring
GeoDynamo.enable_erk2_diagnostics!(interval=10)

# Or via environment variable
ENV["GEODYNAMO_ERK2_DIAGNOSTICS"] = "true"

# Custom residual analysis
stats = erk2_stage_residual_stats(buffers)
println("Max residual: $(stats.max_residual)")
```

### Properties

| Property | Value |
|:---------|:------|
| Order | 2nd (but more accurate than ExponentialAdamsBashforth2) |
| Stability | L-stable |
| Memory | 2× ExponentialAdamsBashforth2 (half-step and full-step caches) |
| Cost | 2× nonlinear evaluations per step |

---

## Krylov Subspace Utilities

All exponential schemes share these core routines:

### `exp_action_krylov(Aop!, v, Δt; m=20, tol=1e-8)`

Computes `exp(Δt A) v` using Arnoldi iteration:

1. Build orthonormal Krylov basis `Vₘ = [v, Av, A²v, ...]`
2. Form upper Hessenberg matrix `Hₘ = Vₘᵀ A Vₘ`
3. Compute `exp(Δt Hₘ)` (small dense matrix)
4. Recover solution: `exp(Δt A) v ≈ ‖v‖ Vₘ exp(Δt Hₘ) e₁`

The remaining `φ₁` actions and per-degree LU-cache management are internal to
the timestep implementation in `src/timestep/imex.jl`. External users should
prefer the high-level timestep drivers instead of these low-level compatibility
hooks.

---

## MPI Parallelization

### Data Distribution

Spectral data is distributed across MPI ranks using PencilArrays:
- Each rank owns a subset of (ℓ, m) modes: `lm_range`
- Radial points may also be distributed: `r_range`

### Critical Pattern for MPI Safety

!!! danger "Avoiding Deadlocks"
    Time-stepping functions with MPI collectives **must** use global loop bounds:

```julia
nlm_total = field.nlm  # Same for all processes

for lm_idx in 1:nlm_total  # ALL processes iterate ALL modes
    owns_mode = lm_idx in lm_range  # Check ownership

    # Extract data (only if owned)
    if owns_mode
        profile[r_range] .= field_data[local_lm, :]
    end

    # ALL processes call collective (prevents deadlock)
    MPI.Allreduce!(profile, MPI.SUM, comm)

    # Perform computation...

    # Store result (only if owned)
    if owns_mode
        field_data[local_lm, :] .= result[r_range]
    end
end
```

**Why?** Different processes may own different numbers of modes. If we used `for lm_idx in lm_range`, processes would call `Allreduce` different numbers of times → **deadlock**.

---

## Adaptive Timestepping

### CFL-Based Control

```julia
# Compute CFL-limited timestep
dt_cfl = compute_cfl_timestep!(velocity_fields, domain)

# Apply Courant factor
dt = params.courant * dt_cfl  # courant typically 0.5-0.9
```

### Stability Monitoring

```julia
# Check for numerical instability
if check_simulation_state_for_nan(state, step)
    @warn "NaN detected! Reducing timestep."
    params.timestep *= 0.5
end
```

---

## Practical Recommendations

### Scheme Selection Guide

| Scenario | Scheme | Rationale |
|:---------|:-------|:----------|
| Production dynamo | **CNAB2** | Robust, well-tested, moderate cost |
| Strong diffusion (low E, Pm) | **ExponentialAdamsBashforth2** | Allows larger Δt, exact linear integration |
| Wave studies | **ExponentialRungeKutta2** | Best transient accuracy |
| Initial development/debugging | **CNAB2** | Simplest to understand |
| Benchmark comparisons | **ExponentialRungeKutta2** | Reference-quality accuracy |

### Timestepper Guidelines

| Option | CNAB2 | ExponentialAdamsBashforth2 | ExponentialRungeKutta2 |
|:-------|:------|:-----|:-----|
| Damping | `CNAB2(theta=0.5)` | N/A | N/A |
| Krylov dimension | N/A | `ExponentialAdamsBashforth2(krylov_dimension=20)` | N/A |
| Krylov tolerance | N/A | `ExponentialAdamsBashforth2(tolerance=1e-8)` | N/A |
| `courant` | 0.5-0.9 | 0.5-0.9 | 0.3-0.5 |

### Startup Protocol

!!! note "First Timestep"
    Always pre-warm the nonlinear history on the first step:

    ```julia
    if step == 1
        # N^{n-1} = N^n (no history available)
        copy!(prev_nonlinear, nonlinear)
    end
    ```

    This is handled automatically by `apply_implicit_step!`.

---

## Troubleshooting

### Simulation Blows Up

| Action | Details |
|:-------|:--------|
| Reduce timestep | Recreate `Simulation(model; Δt = old_Δt / 2)` |
| Check CFL | Enable `compute_cfl_timestep!` monitoring |
| Increase damping | For CNAB2, try `CNAB2(theta = 0.6)` |
| Check BCs | Ensure consistent boundary condition application |

### Simulation Hangs (MPI Deadlock)

| Check | Solution |
|:------|:---------|
| Loop bounds | Ensure loops with `Allreduce` use global bounds |
| Consistent `nlm_total` | Verify all ranks have same value |
| Debug MPI | `export MPI_DEBUG=1` |

### Poor Accuracy

| Action | Details |
|:-------|:--------|
| Increase Krylov dimension | Use `ExponentialAdamsBashforth2(krylov_dimension = 30)` or higher |
| Tighten tolerance | Use `ExponentialAdamsBashforth2(tolerance = 1e-10)` |
| Reduce timestep | For transient accuracy |
| Use ExponentialRungeKutta2 | For critical accuracy requirements |

### Memory Issues

| Action | Details |
|:-------|:--------|
| Use Krylov mode | Instead of dense matrices for ExponentialAdamsBashforth2/ExponentialRungeKutta2 |
| Reduce Krylov dimension | Use `ExponentialAdamsBashforth2(krylov_dimension = 15)` if memory-limited |
| Check for leaks | In nonlinear term caching |

---

## Summary Comparison

| Feature | CNAB2 | ExponentialAdamsBashforth2 | ExponentialRungeKutta2 |
|:--------|:------|:-----|:-----|
| **Order** | 2nd | 2nd (exact linear) | 2nd |
| **Stability** | A-stable | L-stable | L-stable |
| **Linear Treatment** | Implicit | Exponential | Exponential |
| **Nonlinear Treatment** | AB2 | AB2 | Midpoint RK |
| **Memory** | Low | Medium | High |
| **Cost per Step** | Low | Medium | High |
| **Best Use Case** | Production | Stiff problems | High accuracy |
