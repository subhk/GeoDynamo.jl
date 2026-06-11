# Oceananigans API + Printing Parity

Date: 2026-06-11
Status: approved (brainstorm with user)

## Goal

Bring the `src/api/` user-facing layer (constructors, kwargs, runtime behavior)
and all printing (`Base.show` / `Base.summary`) to Oceananigans.jl conventions.
Internal solver code (`solver/`, `timestep/`, `physics/`, …) is out of scope and
stays ASCII-only per the SciML conformance policy.

Decisions made with the user:

| Question | Decision |
|---|---|
| Scope | Full parity: printing + behavior + naming + attribute access |
| `Δt` vs `dt` | `Δt` canonical in the api/ layer, `dt` stays as alias; internals stay `dt` |
| JLD2 writer | Skipped — NetCDF remains the single format (possible later extension) |
| Field access | `model.velocity` / `.temperature` / `.magnetic` / `.composition` property passthrough |
| Time display | Model time is nondimensional → compact number format (`prettysummary`), NOT `prettytime`; `prettytime` reserved for wall-clock durations |

## 1. Utilities — new `src/api/units.jl`

- `prettytime(t)`: Oceananigans port. Picks ns/μs/ms/seconds/minutes/hours/days,
  3-decimal mantissa, singular/plural ("2.341 seconds", "1.500 days", "100 μs",
  "0 seconds"). Exported. Used ONLY for wall-clock durations (elapsed wall time,
  wall time limit).
- `prettysummary(x)`: compact number formatting for dimensionless quantities
  (model time, Δt, stop_time, physics params): integers without trailing `.0`
  ("0", "1"), floats up to 4 significant digits ("1.0e-4", "0.35"),
  `Inf`/`typemax(Int)` → "Inf".

## 2. `Δt` canonical (api/ layer only)

- `Simulation(model; Δt, …)` documented canonical; `dt` accepted alias.
  Exactly one of the two must be given (existing both/neither `ArgumentError`
  stays, message updated to lead with `Δt`).
- Struct field remains `sim.dt` (ASCII internals). `Base.getproperty` /
  `Base.setproperty!` on `Simulation` map `:Δt` ↔ `:dt`;
  `propertynames` includes `:Δt`.
- `Clock`: field remains `last_dt`; `getproperty` exposes `:last_Δt`; printing
  uses `last_Δt`.
- Docs/examples switch to `Δt`.

## 3. Model property access

- `Base.getproperty(::GeodynamoModel)`: `:velocity`, `:temperature`,
  `:magnetic`, `:composition` → corresponding `state.fields.*`
  (`nothing` when disabled). Real struct fields (`state`, `grid`, `clock`)
  unchanged. `propertynames` lists both. `fields(model)` /
  `prognostic_fields(model)` unchanged.

## 4. Default callbacks + `sim.running` run model

- `Simulation` gains mutable `running::Bool` (false until `run!` starts).
- `run!(sim)` becomes: set `running = true`; `while sim.running` step + fire
  callbacks/writers; stop-condition checks move INTO default callbacks
  (Oceananigans semantics).
- Auto-registered defaults (before user callbacks, in this order):
  - `:stop_time_exceeded` — IterationInterval(1); `clock.time ≥ stop_time` →
    `@info` + `running = false`.
  - `:stop_iteration_exceeded` — IterationInterval(1); same for iteration.
  - `:wall_time_limit_exceeded` — IterationInterval(1); elapsed wall ≥ limit.
  - `:nan_checker` — IterationInterval(100); existing `HealthCheck` machinery,
    `abort=true` semantics → `running = false` instead of `error()`.
- User-supplied `callbacks` merge after the defaults; same names override
  (consistent with `add_callback!`).
- Guard: `time_step!(sim)` (single-step path) must not require `running`.

## 5. Schedules

- Add `SpecifiedTimes(times...)`: sorted unique times; fires once when
  `clock.time` reaches/passes each entry (index cursor, mutable). `summary`
  prints `SpecifiedTimes(0.1, 0.5, 1)` (prettysummary per entry).
- `AveragedTimeInterval` deliberately NOT added: it implies writer-side time
  averaging, which GeoDynamo writers don't implement. Follow-up note left in
  this spec; adding the schedule alone would misrepresent behavior.

## 6. Printing rewrite — `src/api/show.jl`

Every line below uses `prettysummary` for nondimensional numbers and
`prettytime` for wall-clock.

```
GeodynamoModel{CPU, Float64}(time = 0, iteration = 0)
├── grid: SphericalShellGrid(CPU, lmax=31, mmax=31, nr=64)
├── timestepper: CNAB2(theta=0.5)
├── physics: Ek=1.0e-4, Pr=1, Pm=1, Ra=1.0e6
└── active: magnetic=false, composition=false

Simulation of GeodynamoModel{CPU, Float64}(time = 0, iteration = 0)
├── Next time step: 1.0e-4
├── Elapsed wall time: 0 seconds
├── Stop time: 1
├── Stop iteration: Inf
├── Wall time limit: Inf
├── Callbacks: OrderedDict with 4 entries:
│   ├── stop_time_exceeded => Callback of stop_time_exceeded on IterationInterval(1)
│   ├── stop_iteration_exceeded => Callback of stop_iteration_exceeded on IterationInterval(1)
│   ├── wall_time_limit_exceeded => Callback of wall_time_limit_exceeded on IterationInterval(1)
│   └── nan_checker => Callback of HealthCheck on IterationInterval(100)
└── Output writers: OrderedDict with no entries

Clock(time = 0, iteration = 0, last_Δt = 0)
```

- `summary(model)` gains the architecture parameter:
  `GeodynamoModel{CPU, Float64}(time = 0, iteration = 0)` (spaces around `=`,
  Oceananigans style).
- `summary` methods added for every schedule (`IterationInterval(10)`,
  `TimeInterval(0.1)`, `WallTimeInterval(60)`, `SpecifiedTimes(…)`), every
  callback type (`Callback of <name> on <schedule summary>`), both writers
  (`FieldWriter writing (velocity, temperature) to snap.nc on TimeInterval(0.1)`-
  style one-liner), and timestepper structs (`CNAB2(theta=0.5)`).
- Nested OrderedDict tree for callbacks/output writers with `│ ├ └`
  connectors; empty → `OrderedDict with no entries`.
- `stop_iteration == typemax(Int)` and `stop_time/wall_time_limit == Inf`
  display as `Inf`.
- Grid `show`/`summary` unchanged except number formatting via `prettysummary`.

## 7. Tests + docs

- Extend `test/oceananigans_api.jl`:
  - `prettytime` table cases (0, sub-μs, ms, seconds, minutes, days; singular).
  - `prettysummary` cases (Int, float, Inf, typemax(Int)).
  - show-output assertions via `occursin` on key tree lines for model,
    simulation, clock (no full-string snapshots — cross-platform float
    formatting).
  - `Δt` canonical, `dt` alias, both → `ArgumentError`, `sim.Δt`
    getproperty/setproperty round-trip.
  - Property access: `model.velocity !== nothing`, `model.magnetic === nothing`
    when disabled, `propertynames` contains the four.
  - Default callbacks present + ordered first; `run!` stops at `stop_time` with
    `sim.running == false`; `stop_iteration` likewise.
  - `SpecifiedTimes` fires exactly once per entry across steps.
- `docs/src/api.md` examples switch to `Δt`; mention `dt` alias and
  `prettytime` export.

## Risks / constraints

- The `run!` loop refactor (§4) is the only behavioral change; existing
  `test/oceananigans_api.jl` + new tests gate it.
- A concurrent session is editing `src/timestep/` + `test/erk2_integration_step.jl`
  (ERK2 W-split port). This work touches only `src/api/`, `test/oceananigans_api.jl`,
  docs — no overlap. Implementation must NOT commit that session's files.
- `Δt` in api/ layer is an approved, documented exception to the repo ASCII
  policy (user decision 2026-06-11).

## Follow-ups (not in scope)

- `JLD2Writer` as a package extension.
- `AveragedTimeInterval` + writer-side time averaging.
