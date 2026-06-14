# BC-aware conductive initial profile with radial source/sink

Date: 2026-06-13
Status: approved (design); implementation pending

## Problem

The conductive initial temperature profile is computed two inconsistent ways, and
neither honors the configured boundary conditions or any internal heating:

- `AnalyticIC(:conductive)` → `set_analytical_temperature!` (`core/initial_conditions.jl`)
  hardcodes a **linear** profile `T = amplitude·(1 − r_frac)`. Blind to geometry
  (`radius_ratio`), BC type, and BC values. Wrong interior shape for a shell.
- The solver default `initialize_temperature_field!` (`physics/temperature/solver.jl`)
  uses the **correct** closed-form `a + b/r` shell solution (`_shell_conductive_temperature`)
  / `1 − r²` ball solution (`_ball_conductive_temperature`) — but only for the
  default non-dimensional values (T=1 inner, T=0 outer); it reads only `radius_ratio`,
  not the actual BC values or type, and supports no internal source.

Composition has the same structure (`:stratified`, `initialize_composition_field!`).

## Goal

A single, BC-aware, source-aware conductive/diffusive background profile used by
**both** the solver default and the `AnalyticIC` path, for **temperature and
composition**, supporting **Dirichlet and Neumann** (incl. mixed) BCs and an
arbitrary **radial source/sink** `S(r)`.

## Decisions (locked)

| Axis | Choice |
|------|--------|
| BC types honored | Dirichlet + Neumann, including mixed (e.g. fixed-T inner / fixed-flux outer) |
| Source form | Radial profile `S(r)` via a discrete BVP solve |
| Scope | Unify: one profile function drives both solver-default and `AnalyticIC` |
| Fields | Temperature **and** composition |
| Source API | `Union{Nothing,Real,Function}` param; `Real`=uniform, `Function`=`r->S(r)` |
| Default | Geometry-aware: shell→0 (recovers `a+b/r`); ball→uniform S=6 (reproduces `1−r²`) |

## Physics

Steady l=0 diffusion with a volumetric source `S(r)`:

```
∇²₀ T(r) = −S(r),     ∇²₀ = d²/dr² + (2/r) d/dr
```

Analytic reference (uniform S): `T(r) = a + b/r − S·r²/6`, with `a,b` fixed by the
two BCs. Ball regularity kills `b` ⇒ `T(r) = T_o + S(r_o² − r²)/6`; with `T_o=0`,
`r_o=1`, `S=6` this is exactly today's `1 − r²` (and `∇²(1−r²) = −6 ⇒ S=6`).

## Architecture

### Core function (geometry-blind)
`conductive_profile_solve(T, domain, bc, source; geometry) -> Vector{T}` (length `nr`),
where `bc` carries `(inner_type, inner_value, outer_type, outer_value)` for the
l=0,m=0 mode and `source` is resolved to a per-grid `S(r)` vector.

Discrete assembly (dense `nr×nr`, solved once — negligible cost):
- `A = D2 + 2·diag(1/r)·D1`, reusing `create_derivative_matrix(T, 1, domain)` and
  `create_derivative_matrix(T, 2, domain)`.
- interior RHS `b[i] = −S(r_i)`.
- boundary rows replace row 1 and row N:
  - Dirichlet: `A[i,:] = eᵢ`, `b[i] = V`
  - Neumann:   `A[i,:] = D1[i,:]`, `b[i] = g`  (flux; sign matched to the implicit
    scalar BC-row builder — verified by test)
- solve `T = A \ b`; store as the `(0,0)` coefficient × √(4π) (orthonormal SH).

### Geometry
- **Shell**: rows 1 and N are the user inner/outer BCs.
- **Ball**: off-center grid, no `r=0` node. Row 1 = the l=0 **regularity row** reused
  from the ball implicit-matrix builder; only the **outer** BC is user-facing.
  Default source S=6.

### Source resolution
- New `SolverParameters` fields: `internal_heating::Union{Nothing,Real,Function}`
  (temperature) and `compositional_source::Union{Nothing,Real,Function}` (composition),
  default `nothing`.
- `nothing` → geometry-aware default (shell 0; ball: temperature 6, composition 0).
- `Real v` → `S(r) = v`. `Function f` → `S(r) = f(r)`.
- Public kwargs on `GeodynamoModel(grid; internal_heating=…, compositional_source=…)`.
- Per-IC override: `AnalyticIC(:conductive; source=…)` (read from `ic.parameters`).

### Unification
`conductive_profile_solve` is the single source of truth, called by:
- `initialize_temperature_field!` / `initialize_composition_field!` (replacing the
  closed-form `_shell_/_ball_conductive_temperature`),
- `set_analytical_temperature!(:conductive)` / `set_analytical_composition!(:conductive)`
  (with `amplitude` scaling + `source=` override).

A new `:conductive` pattern is added for composition (BC+source aware).

### BC extraction
Helper `_scalar_l0_bc(field) -> (in_type, in_val, out_type, out_val)` reads the
field's boundary set (`ProgrammaticBoundarySet.inner_bc_type/outer_bc_type` +
the uniform l=0 value). DIRICHLET/NEUMANN enums already exist in `bcs/common.jl`.

## Edge cases

- **Pure Neumann** (both boundaries Neumann on a shell, or outer-Neumann on a ball):
  the operator has a constant nullspace and a solvability condition. Gauge-fix the
  undetermined constant (pin the volume-mean), and **warn** if `∫ S dV` ≠ net
  boundary flux (inconsistent steady state).
- Neumann sign convention and the ball l=0 regularity row are matched to the
  existing implicit-matrix builders, asserted by a dedicated test.

## Backward compatibility

- Shell, default BCs (T=1 inner / T=0 outer), S=0 ⇒ discrete `a+b/r` ≈ current
  `_shell_conductive_temperature` (to discretization tolerance).
- Ball, default (S=6, outer T=0, r_o=1) ⇒ ≈ `1 − r²`.
- `AnalyticIC(:conductive)` interior shape changes (linear → correct `a+b/r`); this
  is the intended fix. Endpoints unchanged for default BCs.

## Test plan (TDD)

1. Shell Dirichlet 1/0, S=0 → matches analytic `a+b/r`.
2. Shell + uniform S → matches `a+b/r − S·r²/6`.
3. Shell mixed Dirichlet-inner / Neumann-outer → both BC rows satisfied; steady
   residual `‖∇²₀T + S‖ ≈ 0`.
4. Ball default → ≈ `1 − r²`; ball + S, outer Dirichlet `T_o` → `T_o + S(r_o²−r²)/6`.
5. Neumann-row sign matches the implicit scalar BC builder (direct comparison).
6. **Equilibrium test:** one `solver_step!` from this IC with zero velocity leaves
   T (and C) unchanged to discretization tolerance — proves IC↔operator consistency.
7. Composition analogs of 1–4.
8. Backward-compat: default shell/ball runs reproduce prior conductive ICs within tol.
9. Pure-Neumann gauge + compatibility warning fires as specified.

## Out of scope

- Non-l=0 (angular) background structure.
- Time-dependent sources.
- Robin/mixed-Robin BCs (only Dirichlet + Neumann).

## Open risks (resolved during implementation, asserted by tests)

- Exact Neumann-row sign convention (test 5).
- Ball l=0 regularity row reuse (tests 4, 6).
- Discretization tolerance for the backward-compat comparisons (choose atol/rtol
  empirically; document).
