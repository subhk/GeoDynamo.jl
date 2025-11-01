# Time Integration

GeoDynamo.jl offers three production-grade time-integration strategies. All rely on the shared Krylov/ETD utilities in `timestep.jl` and respect PencilArray domain decompositions.

## CNAB2 (Crank–Nicolson Adams–Bashforth 2)

- **Type:** IMEX second-order scheme.
- **Linear part:** Treated implicitly with θ-weighting (`d_implicit`).
- **Nonlinear part:** Advanced using Adams–Bashforth with history buffers.
- **Implementation:**
  1. Build RHS via `build_rhs_cnab2!`, which incorporates `(1-θ)L uⁿ` and nonlinear extrapolation.
  2. Solve `(I - θΔt L) uⁿ⁺¹ = rhs` using banded LU factorisations stored in `SHTnsImplicitMatrices`.

CNAB2 is robust and favoured for production runs. Keep `d_implicit = 0.5` unless you need stronger damping of the linear part.

## EAB2 (Exponential Adams–Bashforth 2)

- **Type:** Exponential integrator using φ-functions.
- **Linear handling:** Precompute `exp(ΔtA)` and `φ₁(ΔtA)` per spherical harmonic degree (or evaluate via Krylov actions).
- **Nonlinear history:** Uses AB2 weights with the ETD basis.
- **Utilities:**
  - `create_etd_cache` – dense matrices for small nr (single-node runs).
  - `eab2_update_krylov_cached!` – action-only variant with cached banded LUs for each `ℓ`.

Use EAB2 when the linear term dominates and you need larger Δt without sacrificing stability. Tune `i_etd_m` and `d_krylov_tol` for the Arnoldi accuracy (20/1e-8 is a good starting point).

## ERK2 (Exponential Runge–Kutta 2)

- **Type:** Two-stage exponential RK with midpoint evaluation.
- **Stage 1:** `k₁ = φ₁(ΔtA) N(uⁿ)` using cached `φ₁` and exponentials.
- **Stage 2:** Build half-step state via `E(Δt/2)` and update nonlinear terms.
- **Final update:**
  ```
  uⁿ⁺¹ = E(Δt) uⁿ + Δt k₁ + Δt φ₂(ΔtA) [N(uⁿ+½) - N(uⁿ)]
  ```
- **Implementation helpers:** `ERK2Cache`, `ERK2FieldBuffers`, `erk2_prepare_field!`, `erk2_finalize_field!`.

ERK2 is the most accurate of the three but also the most expensive due to the additional φ₂ evaluations. Use when transient accuracy is critical (e.g., wave studies).

### ERK2 Cache Management and Diagnostics

- **Precompute caches:** Run `julia --project scripts/precompute_erk2_caches.jl --dt=Δt --fields=temperature,vel_tor` to build dense ERK2 caches ahead of time. The script saves a JLD2 bundle; load it via `load_erk2_cache_bundle!(state.erk2_caches, "erk2_caches.jld2")` before advancing timesteps.
- **Metadata:** Bundles store solver metadata (Δt, geometry, Arnoldi controls) so you can verify compatibility when restarting a long production run.
- **Runtime hooks:** Enable stage diagnostics with `GeoDynamo.enable_erk2_diagnostics!(interval=10)` or set `GEODYNAMO_ERK2_DIAGNOSTICS=true` in the environment. The integrator logs the max and L₂ residual norms of `N(uⁿ+½) - N(uⁿ)` at the desired cadence.
- **Custom analysis:** Use `erk2_stage_residual_stats(buffers)` if you need bespoke monitoring or to feed the residual signal into adaptive control logic.

## Krylov Action Utilities

All exponential schemes rely on the shared Krylov operators:

- `exp_action_krylov(Aop!, v, Δt)` – computes `exp(ΔtA) v` using an Arnoldi basis.
- `phi1_action_krylov(Aop!, A_lu, v, Δt)` – evaluates `φ₁` via banded solves and Krylov exponentials.
- `get_eab2_alu_cache!` – caches banded matrices and LU factorizations per `ℓ` for action-based updates.

Use a consistent `m` (`i_etd_m`) across your run to amortise Arnoldi setup costs and avoid dimension mismatches.

## Adaptive Timesteps

`compute_cfl_timestep!` estimates an explicit CFL-bound based on velocity magnitudes. Combine it with `d_courant` to limit Δt dynamically. Future releases will integrate error-based controllers for the ETD schemes.

## Practical Recommendations

| Scenario | Suggested Scheme | Notes |
| --- | --- | --- |
| Production shell dynamo | CNAB2 | Reliable; monitor nonlinear history to avoid startup transients. |
| Strong diffusion / linear damping | EAB2 | Larger Δt possible; ensure `i_etd_m` ≥ 20 for nr ≥ 128. |
| Wave / transient studies | ERK2 | Most accurate; cache reuse essential for performance. |

Always pre-warm the nonlinear history (`prev_nonlinear` buffers) on the first step as shown in `apply_master_implicit_step!`.
