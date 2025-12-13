# Time Integration

GeoDynamo.jl provides three production-grade implicit-explicit (IMEX) time-stepping schemes optimized for the stiff diffusion terms that arise in magnetohydrodynamic simulations. All schemes leverage MPI-parallel PencilArray decompositions and respect the spectral structure of spherical harmonic transforms.

## Governing Equations

The geodynamo equations in non-dimensional form contain both linear (diffusion) and nonlinear (advection, Lorentz force) terms:

```math
\frac{\partial \mathbf{u}}{\partial t} = \underbrace{E \nabla^2 \mathbf{u}}_{\text{viscous diffusion}} + \underbrace{\mathbf{N}_u(\mathbf{u}, \mathbf{B}, T)}_{\text{nonlinear terms}}
```

```math
\frac{\partial \mathbf{B}}{\partial t} = \underbrace{\frac{E}{Pm} \nabla^2 \mathbf{B}}_{\text{magnetic diffusion}} + \underbrace{\nabla \times (\mathbf{u} \times \mathbf{B})}_{\text{induction}}
```

```math
\frac{\partial T}{\partial t} = \underbrace{\frac{E}{Pr} \nabla^2 T}_{\text{thermal diffusion}} + \underbrace{N_T(\mathbf{u}, T)}_{\text{advection}}
```

where:
- ``E`` = Ekman number (viscous diffusion)
- ``Pm`` = Magnetic Prandtl number
- ``Pr`` = Prandtl number

The **stiffness** comes from the diffusion terms, which have eigenvalues scaling as ``\ell(\ell+1)/r^2`` in spherical harmonics—potentially very large for high-degree modes.

## IMEX Framework

All schemes split the equations into:

| Part | Treatment | Physical Terms |
|------|-----------|----------------|
| **Linear (L)** | Implicit | Diffusion: ``\nabla^2 \mathbf{u}``, ``\nabla^2 \mathbf{B}``, ``\nabla^2 T`` |
| **Nonlinear (N)** | Explicit | Advection, Coriolis, Lorentz force, buoyancy |

This allows large timesteps limited only by the CFL condition on advection, not by the stiff diffusion.

---

## CNAB2 (Crank–Nicolson Adams–Bashforth 2)

The workhorse scheme for production simulations.

### Mathematical Formulation

```math
\frac{u^{n+1} - u^n}{\Delta t} = \theta L u^{n+1} + (1-\theta) L u^n + \frac{3}{2} N^n - \frac{1}{2} N^{n-1}
```

Rearranging:

```math
\underbrace{(I - \theta \Delta t L)}_{A} u^{n+1} = \underbrace{(I + (1-\theta) \Delta t L) u^n + \Delta t \left(\frac{3}{2} N^n - \frac{1}{2} N^{n-1}\right)}_{\text{RHS}}
```

where ``\theta = 0.5`` gives the classic Crank-Nicolson (second-order, A-stable).

### Implementation Details

1. **Build RHS** via `build_rhs_cnab2!`:
   - Apply ``(1-\theta) \Delta t L`` to current solution
   - Add Adams-Bashforth extrapolation of nonlinear terms

2. **Solve** the implicit system using banded LU factorization:
   - One factorization per spherical harmonic degree ``\ell``
   - Cached in `SHTnsImplicitMatrices` for reuse

3. **Apply boundary conditions** after solve

### Usage

```julia
# In simulation loop
params = get_parameters()
params.i_timestepper = :cnab2
params.d_implicit = 0.5  # θ parameter (0.5 = Crank-Nicolson)

# The scheme is applied automatically in the time loop
```

### Properties

| Property | Value |
|----------|-------|
| Order | 2nd (both linear and nonlinear) |
| Stability | A-stable for θ ≥ 0.5 |
| Memory | 1 previous nonlinear term |
| Cost | 1 LU solve per (ℓ,m) mode per field |

**Best for:** Production dynamo runs with moderate timesteps.

---

## EAB2 (Exponential Adams–Bashforth 2)

Uses matrix exponentials to exactly integrate the stiff linear part.

### Mathematical Formulation

```math
u^{n+1} = e^{\Delta t L} u^n + \Delta t \, \varphi_1(\Delta t L) \left(\frac{3}{2} N^n - \frac{1}{2} N^{n-1}\right)
```

where the ``\varphi_1`` function is:

```math
\varphi_1(z) = \frac{e^z - 1}{z}
```

### Key Insight

The matrix exponential ``e^{\Delta t L}`` **exactly** propagates the linear dynamics, removing any timestep restriction from diffusion. The nonlinear terms are still treated explicitly with AB2 extrapolation.

### Implementation Details

Two modes of operation:

#### 1. Dense Matrix Mode (small problems)
```julia
# Precompute exp(ΔtA) and φ₁(ΔtA) matrices
cache = create_etd_cache(domain, diffusivity, dt)
```

#### 2. Krylov Action Mode (large problems, recommended)
```julia
# Compute matrix-vector products without forming dense matrices
eab2_update_krylov_cached!(u, nl, nl_prev, alu_map, domain, ν, config, dt;
                           m=20,      # Arnoldi basis size
                           tol=1e-8)  # Convergence tolerance
```

The Krylov approach:
- Builds an Arnoldi basis of dimension `m`
- Computes ``e^{\Delta t L} v`` and ``\varphi_1(\Delta t L) v`` in this reduced space
- Avoids forming or storing the full ``nr \times nr`` exponential matrices

### Usage

```julia
params = get_parameters()
params.i_timestepper = :eab2
params.i_etd_m = 20        # Arnoldi basis size
params.d_krylov_tol = 1e-8 # Krylov convergence tolerance
```

### Properties

| Property | Value |
|----------|-------|
| Order | 2nd for nonlinear, exact for linear |
| Stability | L-stable (excellent for stiff problems) |
| Memory | 1 previous nonlinear term + Krylov workspace |
| Cost | ~m matrix-vector products per (ℓ,m) mode |

**Best for:** Strongly diffusive regimes where CNAB2 requires small timesteps.

---

## ERK2 (Exponential Runge–Kutta 2)

Two-stage exponential integrator for maximum accuracy.

### Mathematical Formulation

**Stage 1:** Compute intermediate state at ``t + \Delta t/2``

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

### Implementation Details

Uses dedicated cache structures:

```julia
# ERK2-specific buffers
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

```bash
# Precompute caches (recommended for production)
julia --project scripts/precompute_erk2_caches.jl \
    --dt=1e-5 \
    --fields=temperature,vel_tor,vel_pol,mag_tor,mag_pol
```

```julia
# Load precomputed caches
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
|----------|-------|
| Order | 2nd (but more accurate than EAB2) |
| Stability | L-stable |
| Memory | 2× EAB2 (half-step and full-step caches) |
| Cost | 2× nonlinear evaluations per step |

**Best for:** Wave propagation studies, transient dynamics, accuracy-critical applications.

---

## Krylov Subspace Utilities

All exponential schemes share these core routines:

### `exp_action_krylov(Aop!, v, Δt; m=20, tol=1e-8)`

Computes ``e^{\Delta t A} v`` using Arnoldi iteration:

1. Build orthonormal Krylov basis ``V_m = [v, Av, A^2v, \ldots]``
2. Form upper Hessenberg matrix ``H_m = V_m^T A V_m``
3. Compute ``e^{\Delta t H_m}`` (small dense matrix)
4. Recover solution: ``e^{\Delta t A} v \approx \|v\| V_m e^{\Delta t H_m} e_1``

### `phi1_action_krylov(Aop!, A_lu, v, Δt; m=20, tol=1e-8)`

Computes ``\varphi_1(\Delta t A) v`` using the identity:

```math
\varphi_1(z) v = \frac{e^z v - v}{z} = A^{-1}(e^z - I) v
```

Combined with Krylov exponential and banded LU solve.

### `get_eab2_alu_cache!(caches, key, ν, T, domain)`

Manages per-degree LU factorizations:

```julia
# Returns Dict{Int, Tuple{BandedMatrix, BandedLU}}
# Key: spherical harmonic degree ℓ
# Value: (A_banded, LU_factorization)
```

---

## MPI Parallelization

### Data Distribution

Spectral data is distributed across MPI ranks using PencilArrays:
- Each rank owns a subset of ``(\ell, m)`` modes: `lm_range`
- Radial points may also be distributed: `r_range`

### Critical Pattern for MPI Safety

Time-stepping functions with MPI collectives use **global loop bounds**:

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
dt = params.d_courant * dt_cfl  # d_courant typically 0.5-0.9
```

### Stability Monitoring

```julia
# Check for numerical instability
if check_simulation_state_for_nan(state, step)
    @warn "NaN detected! Reducing timestep."
    params.d_dt *= 0.5
end
```

---

## Practical Recommendations

### Scheme Selection Guide

| Scenario | Scheme | Rationale |
|----------|--------|-----------|
| Production dynamo | **CNAB2** | Robust, well-tested, moderate cost |
| Strong diffusion (low E, Pm) | **EAB2** | Allows larger Δt, exact linear integration |
| Wave studies | **ERK2** | Best transient accuracy |
| Initial development/debugging | **CNAB2** | Simplest to understand |
| Benchmark comparisons | **ERK2** | Reference-quality accuracy |

### Parameter Guidelines

| Parameter | CNAB2 | EAB2 | ERK2 |
|-----------|-------|------|------|
| `d_implicit` (θ) | 0.5 | N/A | N/A |
| `i_etd_m` (Krylov dim) | N/A | 20-30 | 20-30 |
| `d_krylov_tol` | N/A | 1e-8 | 1e-8 |
| `d_courant` | 0.5-0.9 | 0.5-0.9 | 0.3-0.5 |

### Startup Protocol

Always pre-warm the nonlinear history on the first step:

```julia
# First timestep: use forward Euler for nonlinear terms
if step == 1
    # N^{n-1} = N^n (no history available)
    copy!(prev_nonlinear, nonlinear)
end
```

This is handled automatically by `apply_master_implicit_step!`.

---

## Troubleshooting

### Simulation Blows Up

1. **Reduce timestep**: `params.d_dt *= 0.5`
2. **Check CFL**: Enable `compute_cfl_timestep!` monitoring
3. **Increase θ**: For CNAB2, try `d_implicit = 0.6` (more damping)
4. **Check boundary conditions**: Ensure consistent BC application

### Simulation Hangs (MPI Deadlock)

1. Check that loops with `Allreduce` use global bounds
2. Verify all ranks have consistent `nlm_total`
3. Enable MPI debugging: `export MPI_DEBUG=1`

### Poor Accuracy

1. **Increase Krylov dimension**: `i_etd_m = 30` or higher
2. **Tighten tolerance**: `d_krylov_tol = 1e-10`
3. **Reduce timestep** for transient accuracy
4. **Use ERK2** for critical accuracy requirements

### Memory Issues

1. Use Krylov mode instead of dense matrices for EAB2/ERK2
2. Reduce `i_etd_m` if memory-limited (minimum ~15)
3. Check for memory leaks in nonlinear term caching
