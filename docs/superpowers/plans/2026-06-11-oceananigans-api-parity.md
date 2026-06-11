# Oceananigans API + Printing Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `src/api/` (constructors, kwargs, runtime behavior) and all printing to Oceananigans.jl conventions per `docs/superpowers/specs/2026-06-11-oceananigans-api-parity-design.md`.

**Architecture:** All changes confined to `src/api/*.jl`, exports in `src/GeoDynamo.jl`, tests in `test/oceananigans_api.jl`, docs in `docs/src/api.md`. Internal solver code stays ASCII (`dt`); the api layer exposes `Δt` via property passthrough. New `units.jl` holds `prettytime`/`prettysummary`. BC types are renamed to Oceananigans names with `const` back-compat aliases. `run!` moves to a `sim.running` flag driven by auto-registered default callbacks.

**Tech Stack:** Julia 1.10+, OrderedCollections (already a dep), Test stdlib. Run tests with the direct binary: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` (the `julia` shim is broken on this machine).

**⚠️ Constraints:**
- A concurrent session edits `src/timestep/`, `src/physics/velocity/solver.jl`, `test/erk2_integration_step.jl`. NEVER `git add` those paths. Work in a fresh worktree off `main` (use superpowers:using-git-worktrees).
- `test/erk2_integration_step.jl` may error locally (their in-progress port) — not your failure signal.
- Full-suite verification command (never pipe through `tail`; exit code matters):
  `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/api_suite.log 2>&1; echo "exit=$?"`
- Quick single-file run during development:
  `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI; include("test/oceananigans_api.jl")'`
- Static-check tests pin source text in `src/`; after renames, run the FULL suite, not just the api file.

---

### Task 1: `prettytime` + `prettysummary` utilities

**Files:**
- Create: `src/api/units.jl`
- Modify: `src/GeoDynamo.jl` (include + export)
- Test: `test/oceananigans_api.jl` (append a testset)

- [ ] **Step 1: Write the failing tests** (append to `test/oceananigans_api.jl`, inside the top-level `@testset`)

```julia
@testset "prettytime / prettysummary" begin
    @test GeoDynamo.prettytime(0) == "0 seconds"
    @test GeoDynamo.prettytime(1) == "1 second"
    @test GeoDynamo.prettytime(1e-9) == "1 ns"
    @test GeoDynamo.prettytime(2.5e-6) == "2.500 μs"
    @test GeoDynamo.prettytime(0.012345) == "12.345 ms"
    @test GeoDynamo.prettytime(2.341) == "2.341 seconds"
    @test GeoDynamo.prettytime(90) == "1.500 minutes"
    @test GeoDynamo.prettytime(3600) == "1 hour"
    @test GeoDynamo.prettytime(129600) == "1.500 days"
    @test GeoDynamo.prettytime(Inf) == "Inf days"

    @test GeoDynamo.prettysummary(0) == "0"
    @test GeoDynamo.prettysummary(1) == "1"
    @test GeoDynamo.prettysummary(0.0) == "0"
    @test GeoDynamo.prettysummary(1.0) == "1"
    @test GeoDynamo.prettysummary(1e-4) == "0.0001"
    @test GeoDynamo.prettysummary(1.5e-7) == "1.5e-7"
    @test GeoDynamo.prettysummary(0.35) == "0.35"
    @test GeoDynamo.prettysummary(123456.0) == "1.23456e5" || GeoDynamo.prettysummary(123456.0) == "123456"
    @test GeoDynamo.prettysummary(Inf) == "Inf"
    @test GeoDynamo.prettysummary(typemax(Int)) == "Inf"
end
```

- [ ] **Step 2: Run to verify failure**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI; include("test/oceananigans_api.jl")'`
Expected: FAIL/ERROR `UndefVarError: prettytime not defined`

- [ ] **Step 3: Implement `src/api/units.jl`**

```julia
# Human-readable formatting helpers (Oceananigans conventions).
#
# `prettytime` is for WALL-CLOCK durations only. Model time in GeoDynamo is
# nondimensional (diffusion-time units), so model times/timesteps print via
# `prettysummary` instead (user decision, see the 2026-06-11 parity spec).

"""
    prettytime(t)

Format a duration `t` in seconds as a human-readable string, e.g.
`"2.341 seconds"`, `"1.500 days"`, `"100 ns"`. Follows Oceananigans
conventions: picks ns/μs/ms/seconds/minutes/hours/days; integer-valued
quantities drop the decimals and pluralize correctly.
"""
function prettytime(t::Real)
    t == 0 && return "0 seconds"
    isfinite(t) || return string(t > 0 ? "Inf" : "-Inf", " days")

    if t < 1e-6
        value, units = t * 1e9, "ns"
    elseif t < 1e-3
        value, units = t * 1e6, "μs"
    elseif t < 1
        value, units = t * 1e3, "ms"
    elseif t < 60
        value, units = t, "second"
    elseif t < 3600
        value, units = t / 60, "minute"
    elseif t < 86400
        value, units = t / 3600, "hour"
    else
        value, units = t / 86400, "day"
    end

    if units in ("ns", "μs", "ms")
        body = isinteger(value) ? string(Int(value)) :
               @sprintf("%.3f", value)
        return string(body, " ", units)
    else
        if isinteger(value)
            n = Int(value)
            return string(n, " ", units, n == 1 ? "" : "s")
        else
            return string(@sprintf("%.3f", value), " ", units, "s")
        end
    end
end

"""
    prettysummary(x)

Compact number formatting for nondimensional quantities (model time, Δt,
physics parameters): integers print without a trailing `.0`, `Inf` and
`typemax(Int)` print as `"Inf"`, everything else uses Julia's shortest
round-trip float printing.
"""
prettysummary(x::Integer) = x == typemax(typeof(x)) ? "Inf" : string(x)
function prettysummary(x::Real)
    isfinite(x) || return string(Float64(x) > 0 ? "Inf" : "-Inf")
    f = Float64(x)
    isinteger(f) && abs(f) < 1e15 && return string(Int(f))
    return string(f)
end
prettysummary(x) = string(x)
```

`@sprintf` needs `using Printf` — check `src/GeoDynamo.jl` for an existing `using Printf`; add it if absent.

- [ ] **Step 4: Wire into the module.** In `src/GeoDynamo.jl`, find the block of `include("api/…")` lines and add `include("api/units.jl")` BEFORE `include("api/show.jl")`. Add to the api exports block (search `export` near `Simulation, run!`): `export prettytime, prettysummary`.

- [ ] **Step 5: Run test → PASS.** Same command as Step 2. If `prettysummary(123456.0)` disagrees with both accepted alternatives, fix the test to the actual shortest-round-trip output (the intent is "no trailing .0, no information loss", not a specific notation).

- [ ] **Step 6: Commit**

```bash
git add src/api/units.jl src/GeoDynamo.jl test/oceananigans_api.jl
git commit -m "feat(api): prettytime + prettysummary formatting utilities"
```

---

### Task 2: `SpecifiedTimes` schedule

**Files:**
- Modify: `src/api/schedules.jl`
- Modify: `src/GeoDynamo.jl` (export)
- Test: `test/oceananigans_api.jl`

- [ ] **Step 1: Failing test**

```julia
@testset "SpecifiedTimes schedule" begin
    s = GeoDynamo.SpecifiedTimes(0.5, 0.1, 0.1, 1.0)   # unsorted + dup on purpose
    @test s.times == [0.1, 0.5, 1.0]
    ctx(t) = GeoDynamo._ScheduleContext(t, 0, 0.0)
    @test GeoDynamo.should_fire(s, ctx(0.05)) == false
    @test GeoDynamo.should_fire(s, ctx(0.1)) == true     # reaches 0.1
    @test GeoDynamo.should_fire(s, ctx(0.2)) == false    # 0.1 already fired
    @test GeoDynamo.should_fire(s, ctx(0.7)) == true     # passed 0.5
    @test GeoDynamo.should_fire(s, ctx(2.0)) == true     # passed 1.0
    @test GeoDynamo.should_fire(s, ctx(3.0)) == false    # exhausted
end
```

- [ ] **Step 2: Run → FAIL** (`SpecifiedTimes not defined`).

- [ ] **Step 3: Implement** (append to `src/api/schedules.jl`)

```julia
"""
    SpecifiedTimes(times...)

Schedule that fires once when simulation time reaches (or first passes) each of
the given times. Times are sorted and de-duplicated.
"""
mutable struct SpecifiedTimes <: AbstractSchedule
    times::Vector{Float64}
    _next::Int            # index of the next unfired entry
end
SpecifiedTimes(times::Real...) = SpecifiedTimes(unique(sort(Float64[times...])), 1)
SpecifiedTimes(times::AbstractVector{<:Real}) = SpecifiedTimes(unique(sort(Float64.(times))), 1)

function should_fire(s::SpecifiedTimes, ctx::_ScheduleContext)
    s._next > length(s.times) && return false
    if ctx.time >= s.times[s._next] - 1e-12
        # advance past every entry this step crossed; fire once
        while s._next <= length(s.times) && ctx.time >= s.times[s._next] - 1e-12
            s._next += 1
        end
        return true
    end
    return false
end
```

- [ ] **Step 4: Export.** Add `SpecifiedTimes` next to the existing `TimeInterval, IterationInterval, WallTimeInterval` export in `src/GeoDynamo.jl`.

- [ ] **Step 5: Run → PASS.**

- [ ] **Step 6: Commit** `git add src/api/schedules.jl src/GeoDynamo.jl test/oceananigans_api.jl && git commit -m "feat(api): SpecifiedTimes schedule"`

---

### Task 3: `Δt` property on Simulation, `last_Δt` on Clock

**Files:**
- Modify: `src/api/simulation.jl` (docstring + getproperty/setproperty!)
- Modify: `src/api/clock.jl`
- Test: `test/oceananigans_api.jl`

- [ ] **Step 1: Failing test**

```julia
@testset "Δt canonical property" begin
    grid = SphericalShellGrid(lmax = 4, nr = 8)
    model = GeodynamoModel(grid)
    sim = Simulation(model, Δt = 1e-4, stop_iteration = 1)
    @test sim.Δt == 1e-4
    @test sim.dt == 1e-4                       # alias field still readable
    sim.Δt = 2e-4
    @test sim.dt == 2e-4
    @test :Δt in propertynames(sim)
    @test model.clock.last_Δt == model.clock.last_dt
    @test :last_Δt in propertynames(model.clock)
end
```

NOTE: if `test/oceananigans_api.jl` already builds a small grid/model fixture near the top, reuse it instead of constructing a new one — model construction is the slow part of this file.

- [ ] **Step 2: Run → FAIL** (`type Simulation has no field Δt`).

- [ ] **Step 3: Implement.** Append to `src/api/simulation.jl`:

```julia
# Oceananigans-canonical `Δt` property (api-layer exception to the ASCII
# policy, approved 2026-06-11). The struct field stays ASCII `dt`.
function Base.getproperty(sim::Simulation, name::Symbol)
    name === :Δt && return getfield(sim, :dt)
    return getfield(sim, name)
end
function Base.setproperty!(sim::Simulation, name::Symbol, x)
    name === :Δt && return setfield!(sim, :dt, Float64(x))
    return setfield!(sim, name, x)
end
Base.propertynames(sim::Simulation) = (fieldnames(Simulation)..., :Δt)
```

Append to `src/api/clock.jl`:

```julia
function Base.getproperty(c::Clock, name::Symbol)
    name === :last_Δt && return getfield(c, :last_dt)
    return getfield(c, name)
end
function Base.setproperty!(c::Clock{T}, name::Symbol, x) where {T}
    name === :last_Δt && return setfield!(c, :last_dt, T(x))
    return setfield!(c, name, x)
end
Base.propertynames(c::Clock) = (fieldnames(Clock)..., :last_Δt)
```

Also update the `Simulation` docstring header from `Simulation(model; dt, …)` to `Simulation(model; Δt, …)` and reword the timestep paragraph: "pass it as `Δt` (canonical, Oceananigans convention) or `dt` (alias)". Swap the `ArgumentError` messages in the constructor to lead with `Δt`:

```julia
        throw(ArgumentError("Simulation: pass either `Δt` or `dt`, not both"))
...
        throw(ArgumentError("Simulation: a timestep is required (pass `Δt=` or `dt=`)"))
```

- [ ] **Step 4: Run → PASS.** Also re-run the WHOLE api test file — the existing both/neither ArgumentError tests must still pass (message wording may be asserted; update those assertions if they pin the old wording).

- [ ] **Step 5: Commit** `git add src/api/simulation.jl src/api/clock.jl test/oceananigans_api.jl && git commit -m "feat(api): Δt canonical property on Simulation, last_Δt on Clock"`

---

### Task 4: Model field property access

**Files:**
- Modify: `src/api/model.jl`
- Test: `test/oceananigans_api.jl`

- [ ] **Step 1: Failing test**

```julia
@testset "model field properties" begin
    grid = SphericalShellGrid(lmax = 4, nr = 8)
    model = GeodynamoModel(grid)                 # no magnetic, no composition
    @test model.velocity === model.state.fields.velocity
    @test model.temperature === model.state.fields.temperature
    @test model.magnetic === nothing
    @test model.composition === nothing
    @test :velocity in propertynames(model)
    @test :magnetic in propertynames(model)
end
```

- [ ] **Step 2: Run → FAIL** (`type GeodynamoModel has no field velocity`).

- [ ] **Step 3: Implement.** Append to `src/api/model.jl`:

```julia
# Oceananigans-style field access: model.velocity / model.temperature /
# model.magnetic / model.composition forward to the solver state's fields
# (`nothing` when the field is disabled).
const _MODEL_FIELD_PROPS = (:velocity, :temperature, :magnetic, :composition)

function Base.getproperty(m::GeodynamoModel, name::Symbol)
    name in _MODEL_FIELD_PROPS &&
        return getproperty(getfield(m, :state).fields, name)
    return getfield(m, name)
end
Base.propertynames(m::GeodynamoModel) = (fieldnames(GeodynamoModel)..., _MODEL_FIELD_PROPS...)
```

Check first that `state.fields` has exactly those four property names (`grep -n "fields = (" src/solver/state.jl` or wherever `SolverState.fields` is built — it is a NamedTuple with keys velocity/temperature/magnetic/composition; confirm with `propertynames(model.state.fields)` in a REPL probe if unsure).

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit** `git add src/api/model.jl test/oceananigans_api.jl && git commit -m "feat(api): Oceananigans-style model.velocity/.temperature property access"`

---

### Task 5: Oceananigans BC names (`ValueBoundaryCondition`, `FluxBoundaryCondition`, `FieldBoundaryConditions`)

**Files:**
- Modify: `src/api/boundary_conditions.jl`
- Modify: `src/GeoDynamo.jl` (exports)
- Test: `test/oceananigans_api.jl`

- [ ] **Step 1: Failing test**

```julia
@testset "Oceananigans BC names" begin
    @test ValueBoundaryCondition(0.5) == FixedTemperature(0.5)
    @test FluxBoundaryCondition(1.0) == FixedFlux(1.0)
    bcs = FieldBoundaryConditions(inner = ValueBoundaryCondition(1.0),
                                  outer = FluxBoundaryCondition(0.0))
    @test bcs isa BoundaryConditions          # alias identity
    @test bcs.inner == FixedTemperature(1.0)
    @test sprint(show, ValueBoundaryCondition(0.5)) == "ValueBoundaryCondition(0.5)"
    @test sprint(show, bcs) ==
        "FieldBoundaryConditions(inner = ValueBoundaryCondition(1.0), outer = FluxBoundaryCondition(0.0))"
end
```

- [ ] **Step 2: Run → FAIL** (`ValueBoundaryCondition not defined`).

- [ ] **Step 3: Implement.** In `src/api/boundary_conditions.jl` rename the structs and alias the old names (the structs are defined at lines 12–23 and 32–39):

```julia
# Thermal / composition BCs — Oceananigans-canonical names; the original
# GeoDynamo names remain as const aliases so existing code keeps working.
struct ValueBoundaryCondition{T} <: AbstractThermalBC   # Dirichlet
    value::T
end
struct FluxBoundaryCondition{T} <: AbstractThermalBC    # Neumann/flux
    value::T
end
ValueBoundaryCondition() = ValueBoundaryCondition(0.0)
FluxBoundaryCondition() = FluxBoundaryCondition(0.0)
const FixedTemperature = ValueBoundaryCondition
const FixedFlux = FluxBoundaryCondition
Base.show(io::IO, bc::ValueBoundaryCondition) = print(io, "ValueBoundaryCondition($(bc.value))")
Base.show(io::IO, bc::FluxBoundaryCondition) = print(io, "FluxBoundaryCondition($(bc.value))")
```

and

```julia
# Per-field wrapper holding an inner and outer BC (Oceananigans-canonical name;
# spherical inner/outer stands in for Oceananigans' bottom/top).
struct FieldBoundaryConditions{I, O}
    inner::I
    outer::O
end
FieldBoundaryConditions(; inner, outer) = FieldBoundaryConditions(inner, outer)
const BoundaryConditions = FieldBoundaryConditions
function Base.show(io::IO, bc::FieldBoundaryConditions)
    print(io, "FieldBoundaryConditions(inner = $(bc.inner), outer = $(bc.outer))")
end
```

The `==` methods and the `_velocity_bc_code` / `_thermal_bc_code` dispatch tables below reference `FixedTemperature`, `FixedFlux`, `BoundaryConditions` — they keep working unchanged through the const aliases. Do NOT edit them.

- [ ] **Step 4: Grep for breakage outside api/.**

Run: `grep -rn "FixedTemperature\|FixedFlux\|BoundaryConditions(" src/ --include="*.jl" | grep -v "src/api/" | head -30`
The aliases keep construction and dispatch working; you are looking for (a) `sprint(show, …)`-style string comparisons and (b) imports by name — both fine with const aliases. Note anything suspicious for Step 6.

- [ ] **Step 5: Export.** In `src/GeoDynamo.jl`, extend the BC export line with `ValueBoundaryCondition, FluxBoundaryCondition, FieldBoundaryConditions`.

- [ ] **Step 6: Run the FULL suite** (static checks + BC-string tests live outside the api file):

`~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/api_suite.log 2>&1; echo "exit=$?"`

Expected: same pass count as baseline, plus the new tests; the only acceptable error is the known in-progress `erk2_integration_step.jl` one (and only if the concurrent session's changes are still in the tree — in a fresh worktree there must be ZERO errors). If a test pinned `"FixedTemperature(1.0)"` show output, update that test to the new canonical string.

- [ ] **Step 7: Commit** `git add src/api/boundary_conditions.jl src/GeoDynamo.jl test/oceananigans_api.jl && git commit -m "feat(api): Oceananigans BC names with back-compat aliases"`

---

### Task 6: `boundary_conditions` NamedTuple kwarg on GeodynamoModel

**Files:**
- Modify: `src/api/model.jl`
- Test: `test/oceananigans_api.jl`

- [ ] **Step 1: Failing test**

```julia
@testset "boundary_conditions NamedTuple kwarg" begin
    grid = SphericalShellGrid(lmax = 4, nr = 8)
    t_bcs = FieldBoundaryConditions(inner = ValueBoundaryCondition(1.0),
                                    outer = ValueBoundaryCondition(0.0))
    model = GeodynamoModel(grid; boundary_conditions = (temperature = t_bcs,))
    @test model.state.parameters.temperature_bcs == t_bcs
    # specifying the same field both ways is an error
    @test_throws ArgumentError GeodynamoModel(grid;
        boundary_conditions = (temperature = t_bcs,),
        temperature_bcs = t_bcs)
    # unknown field name is an error
    @test_throws ArgumentError GeodynamoModel(grid;
        boundary_conditions = (pressure = t_bcs,))
end
```

- [ ] **Step 2: Run → FAIL** (`MethodError`/`UnexpectedKeyword` for `boundary_conditions`).

- [ ] **Step 3: Implement.** In `src/api/model.jl`, add a resolver above the public constructors:

```julia
# Oceananigans-style `boundary_conditions = (velocity=…, temperature=…,
# composition=…)` NamedTuple, merged with the per-field kwargs. A field given
# through BOTH paths is ambiguous → error.
function _resolve_bcs(boundary_conditions, velocity_bcs, temperature_bcs,
        composition_bcs, defaults)
    boundary_conditions === nothing &&
        return (something(velocity_bcs, defaults.velocity),
                something(temperature_bcs, defaults.temperature),
                something(composition_bcs, defaults.composition))
    valid = (:velocity, :temperature, :composition)
    for k in keys(boundary_conditions)
        k in valid || throw(ArgumentError(
            "GeodynamoModel: unknown boundary_conditions field :$k " *
            "(valid: :velocity, :temperature, :composition)"))
    end
    conflict(name, kwarg) = kwarg !== nothing && haskey(boundary_conditions, name) &&
        throw(ArgumentError(
            "GeodynamoModel: $(name) BCs given both via boundary_conditions " *
            "NamedTuple and the $(name)_bcs kwarg — pass one"))
    conflict(:velocity, velocity_bcs)
    conflict(:temperature, temperature_bcs)
    conflict(:composition, composition_bcs)
    vel = haskey(boundary_conditions, :velocity) ? boundary_conditions.velocity :
          something(velocity_bcs, defaults.velocity)
    temp = haskey(boundary_conditions, :temperature) ? boundary_conditions.temperature :
           something(temperature_bcs, defaults.temperature)
    comp = haskey(boundary_conditions, :composition) ? boundary_conditions.composition :
           something(composition_bcs, defaults.composition)
    return (vel, temp, comp)
end
```

In BOTH public constructors (`GeodynamoModel(grid::SphericalShellGrid; …)` at model.jl:109 and the `SphericalBallGrid` one at model.jl:153) change the three BC kwargs' defaults to `nothing` and add the new kwarg, then resolve:

```julia
        velocity_bcs = nothing,
        temperature_bcs = nothing,
        composition_bcs = nothing,
        boundary_conditions = nothing,
```

and immediately inside the body, before calling `_build_geodynamo_model`:

```julia
    velocity_bcs, temperature_bcs, composition_bcs = _resolve_bcs(
        boundary_conditions, velocity_bcs, temperature_bcs, composition_bcs,
        (velocity = BoundaryConditions(inner = NoSlip(), outer = NoSlip()),
         temperature = BoundaryConditions(inner = FixedFlux(1.0), outer = FixedTemperature(0.0)),
         composition = BoundaryConditions(inner = FixedFlux(0.0), outer = FixedTemperature(0.0))))
```

(The old literal defaults move into this `defaults` NamedTuple — delete them from the kwarg list.)

- [ ] **Step 4: Run → PASS.** Also run the full api test file: existing tests construct models with `temperature_bcs = …` explicitly and must still pass.

- [ ] **Step 5: Commit** `git add src/api/model.jl test/oceananigans_api.jl && git commit -m "feat(api): boundary_conditions NamedTuple kwarg on GeodynamoModel"`

---

### Task 7: `set!` with numbers, functions, arrays (scalar fields)

**Files:**
- Modify: `src/api/initial_conditions.jl` (new `set_initial_condition!` methods)
- Test: `test/oceananigans_api.jl`

- [ ] **Step 1: Failing test**

```julia
@testset "set! with number / function / array" begin
    grid = SphericalShellGrid(lmax = 8, nr = 8)
    model = GeodynamoModel(grid)

    # number → uniform value: l=0,m=0 spectral mode carries it, others ~0
    set!(model; temperature = 0.5)
    cfg = model.state.backend.shtns_config
    phys = GeoDynamo.get_main_physical_field(model.temperature)
    # round-trip through the field's physical storage: mean must be ≈ 0.5
    GeoDynamo.shtnskit_spectral_to_physical!(model.temperature.spectral, phys)
    @test isapprox(sum(parent(phys.data)) / length(parent(phys.data)), 0.5; atol = 1e-8)

    # function of (r, θ, φ)
    set!(model; temperature = (r, θ, φ) -> r)
    GeoDynamo.shtnskit_spectral_to_physical!(model.temperature.spectral, phys)
    rvals = [model.state.runtime.outer_core_domain.r[k, 4]
             for k in 1:model.state.runtime.outer_core_domain.N]
    pd = parent(phys.data)
    # physical value at every level ≈ r of that level (function only of r)
    @test all(isapprox(pd[1, 1, k], rvals[k]; atol = 1e-6) for k in axes(pd, 3))

    # array of physical size
    arr = fill(0.25, size(parent(phys.data)))
    set!(model; temperature = arr)
    GeoDynamo.shtnskit_spectral_to_physical!(model.temperature.spectral, phys)
    @test isapprox(parent(phys.data)[1, 1, 1], 0.25; atol = 1e-8)

    # vector fields reject the function path with a clear error
    @test_throws ArgumentError set!(model; velocity = (r, θ, φ) -> 1.0)
end
```

NOTE on the transform direction used in assertions: check the actual name/signature of the field-level synthesis helper before writing — `grep -n "function shtnskit_spectral_to_physical!" src/fields/transforms.jl`. If the signature is `(spec, phys)` vs `(phys, spec)` adjust. The analysis (physical→spectral) helper used by the implementation below is `shtnskit_physical_to_spectral!(phys::SHTnsPhysField, spec::SHTnsSpecField)` (src/fields/transforms.jl:229).

- [ ] **Step 2: Run → FAIL** ("Unrecognised initial condition type Float64").

- [ ] **Step 3: Implement.** Append to `src/api/initial_conditions.jl` (BEFORE the catch-all method — Julia dispatch handles ordering, but keep the file readable):

```julia
# ── Oceananigans-style direct values: number / function / array ──────────────
#
# Scalar fields only (temperature, composition). The value is realized on the
# field's physical grid, then transformed to spectral with the existing
# analysis path. Vector fields (velocity, magnetic) keep descriptor ICs — a
# pointwise function is ambiguous for a toroidal/poloidal decomposition.

const _SCALAR_IC_FIELDS = (:temperature, :composition)

function _check_scalar_ic_field(field::Symbol, ic)
    field in _SCALAR_IC_FIELDS || throw(ArgumentError(
        "set!: direct values (numbers, functions, arrays) are only supported " *
        "for scalar fields $( _SCALAR_IC_FIELDS ); :$field needs a descriptor " *
        "IC (RandomPerturbation, AnalyticIC, FileIC, ZeroIC) because of its " *
        "toroidal–poloidal decomposition"))
    return nothing
end

# Fill the field's main physical storage from fn(r, θ, φ) at the LOCAL grid
# points (works under MPI: axes_local gives this rank's global index ranges),
# then run the analysis transform into the spectral field.
function _set_scalar_from_function!(model::GeodynamoModel, field::Symbol, fn)
    f = _get_field(model, field)
    phys = get_main_physical_field(f)
    cfg = model.state.backend.shtns_config
    domain = model.state.runtime.outer_core_domain
    θs = cfg.theta_grid                     # colatitude, length nlat (global)
    φs = cfg.phi_grid                       # longitude, length nlon (global)
    data = parent(phys.data)
    ax = phys.pencil.axes_local             # (θ-range, φ-range, r-range) global
    for (kl, kg) in enumerate(ax[3]), (jl, jg) in enumerate(ax[2]), (il, ig) in enumerate(ax[1])
        data[il, jl, kl] = fn(domain.r[kg, 4], θs[ig], φs[jg])
    end
    shtnskit_physical_to_spectral!(phys, f.spectral)
    return model
end

function set_initial_condition!(model::GeodynamoModel, field::Symbol, value::Real)
    _check_scalar_ic_field(field, value)
    return _set_scalar_from_function!(model, field, (r, θ, φ) -> Float64(value))
end

function set_initial_condition!(model::GeodynamoModel, field::Symbol, fn::Function)
    _check_scalar_ic_field(field, fn)
    return _set_scalar_from_function!(model, field, fn)
end

function set_initial_condition!(model::GeodynamoModel, field::Symbol,
        arr::AbstractArray{<:Real, 3})
    _check_scalar_ic_field(field, arr)
    f = _get_field(model, field)
    phys = get_main_physical_field(f)
    data = parent(phys.data)
    size(arr) == size(data) || throw(ArgumentError(
        "set!: array size $(size(arr)) does not match the local physical grid " *
        "$(size(data)) (nlat, nlon, nr)"))
    copyto!(data, arr)
    shtnskit_physical_to_spectral!(phys, f.spectral)
    return model
end
```

Pre-implementation checks the engineer MUST do (the code above names internals that must be verified, fix the code to match reality):
1. `get_main_physical_field` — exists for temperature (src/physics/temperature/field.jl:132). Confirm the composition equivalent: `grep -n "get_main_physical_field" src/physics/composition/field.jl`. If composition uses a different accessor, branch on `field`.
2. `cfg.theta_grid` / `cfg.phi_grid` — confirm field names: `grep -n "theta_grid\|phi_grid" src/transforms/spectral.jl | head`.
3. `phys.pencil.axes_local` ordering — confirm with `grep -n "axes_local" src/fields/*.jl | head` and an existing usage; if the physical pencil layout is (nlat, nlon, nr) the code stands.
4. The spectral field accessor `f.spectral` — temperature stores it as `.spectral` (audit); confirm composition matches.

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit** `git add src/api/initial_conditions.jl test/oceananigans_api.jl && git commit -m "feat(api): set! accepts numbers, functions of (r,θ,φ), and arrays for scalar fields"`

---

### Task 8: Default callbacks + `sim.running` run model

**Files:**
- Modify: `src/api/simulation.jl`
- Test: `test/oceananigans_api.jl`

- [ ] **Step 1: Failing test**

```julia
@testset "default callbacks + sim.running" begin
    grid = SphericalShellGrid(lmax = 4, nr = 8)
    model = GeodynamoModel(grid)
    sim = Simulation(model, Δt = 1e-4, stop_iteration = 3)
    # defaults registered first, in order
    @test collect(keys(sim.callbacks))[1:4] ==
        [:stop_time_exceeded, :stop_iteration_exceeded,
         :wall_time_limit_exceeded, :nan_checker]
    @test sim.running == false
    run!(sim)
    @test sim.running == false
    @test model.clock.iteration == 3          # stopped BY the callback
    # stop_time path
    model2 = GeodynamoModel(grid)
    sim2 = Simulation(model2, Δt = 1e-4, stop_time = 2.5e-4)
    run!(sim2)
    @test model2.clock.time >= 2.5e-4
    @test sim2.running == false
    # user callbacks come after defaults and can share the dict
    model3 = GeodynamoModel(grid)
    fired = Ref(0)
    sim3 = Simulation(model3, Δt = 1e-4, stop_iteration = 2,
        callbacks = (progress = Callback(s -> fired[] += 1,
                                         schedule = IterationInterval(1)),))
    run!(sim3)
    @test fired[] >= 2
    @test collect(keys(sim3.callbacks))[end] == :progress
end
```

- [ ] **Step 2: Run → FAIL** (`type Simulation has no field running` / key order mismatch).

- [ ] **Step 3: Implement.** In `src/api/simulation.jl`:

(a) Add `running::Bool` to the struct after `wall_time_limit`:

```julia
mutable struct Simulation{M, C, O}
    model::M
    dt::Float64
    stop_time::Float64
    stop_iteration::Int
    wall_time_limit::Float64
    running::Bool
    callbacks::C
    output_writers::O
    _wall_start::Float64
end
```

(b) Define the default-callback functions and builder ABOVE the constructor:

```julia
# ── Default callbacks (Oceananigans semantics): stop conditions live in the
# callback table, not the run! loop. Each sets sim.running = false. ──────────

function stop_time_exceeded(sim)
    if sim.model.clock.time >= sim.stop_time - 1e-15
        @info "Simulation is stopping after reaching stop time " *
              "($(prettysummary(sim.stop_time)))."
        sim.running = false
    end
    return nothing
end

function stop_iteration_exceeded(sim)
    if sim.model.clock.iteration >= sim.stop_iteration
        @info "Simulation is stopping after reaching stop iteration " *
              "$(prettysummary(sim.stop_iteration))."
        sim.running = false
    end
    return nothing
end

function wall_time_limit_exceeded(sim)
    if sim._wall_start > 0 && (time() - sim._wall_start) >= sim.wall_time_limit
        @info "Simulation is stopping after exceeding the wall time limit " *
              "($(prettytime(sim.wall_time_limit)))."
        sim.running = false
    end
    return nothing
end

function nan_checker(sim)
    r = _health_check(sim.model)
    if r.has_issue
        @warn "NaN/Inf found in fields $(r.fields) at iteration " *
              "$(sim.model.clock.iteration); stopping simulation."
        sim.running = false
    end
    return nothing
end

function _default_callbacks()
    OrderedDict{Symbol, Any}(
        :stop_time_exceeded => Callback(stop_time_exceeded, IterationInterval(1)),
        :stop_iteration_exceeded => Callback(stop_iteration_exceeded, IterationInterval(1)),
        :wall_time_limit_exceeded => Callback(wall_time_limit_exceeded, IterationInterval(1)),
        :nan_checker => Callback(nan_checker, IterationInterval(100)),
    )
end
```

(c) In the constructor, merge defaults before user callbacks and add `false` for `running` in the `Simulation{…}(…)` call (position matters — after `Float64(wall_time_limit)`):

```julia
    callback_items = merge(_default_callbacks(), _to_ordered(callbacks, :callback))
```

(d) Rewrite `run!`:

```julia
"""
    run!(sim::Simulation)

Advance the simulation until a stop criterion fires. Stop conditions
(`stop_time`, `stop_iteration`, `wall_time_limit`, NaN detection) are enforced
by the default callbacks registered at construction; any callback may halt the
run by setting `sim.running = false`.
"""
function run!(sim::Simulation)
    sim._wall_start = time()
    sim.running = true
    _run_callbacks!(sim)            # fire once at the start: a sim already past
    while sim.running               # its stop time must not take a step
        time_step!(sim)
    end
    return sim
end
```

NOTE: `IterationInterval` fires when `step % interval == 0`, so at iteration 0 the pre-loop `_run_callbacks!` fires all four defaults — that is the desired "check before first step" behavior. The `nan_checker` interval of 100 also fires at 0; harmless.

- [ ] **Step 4: Run → PASS.** Then run the FULL api test file: pre-existing `run!` tests asserting exact final iteration counts must still hold (the callback fires after the step that reaches the limit — final iteration == stop_iteration exactly, same as the old `<` loop).

- [ ] **Step 5: Commit** `git add src/api/simulation.jl test/oceananigans_api.jl && git commit -m "feat(api): default stop/nan callbacks drive run! via sim.running"`

---

### Task 9: Printing rewrite (`show.jl`)

**Files:**
- Rewrite: `src/api/show.jl`
- Test: `test/oceananigans_api.jl`

- [ ] **Step 1: Failing test**

```julia
@testset "Oceananigans-style printing" begin
    grid = SphericalShellGrid(lmax = 4, nr = 8)
    model = GeodynamoModel(grid)
    sim = Simulation(model, Δt = 1e-4, stop_time = 1.0)

    msum = summary(model)
    @test occursin("GeodynamoModel{CPU, Float64}", msum)
    @test occursin("(time = 0, iteration = 0)", msum)

    mshow = sprint(show, MIME"text/plain"(), model)
    @test occursin("├── grid: SphericalShellGrid(CPU, lmax=4, mmax=4, nr=8)", mshow)
    @test occursin("├── timestepper:", mshow)
    @test occursin("└── active: magnetic=false, composition=false", mshow)

    sshow = sprint(show, MIME"text/plain"(), sim)
    @test occursin("Simulation of GeodynamoModel{CPU, Float64}", sshow)
    @test occursin("├── Next time step: 0.0001", sshow)
    @test occursin("├── Elapsed wall time: 0 seconds", sshow)
    @test occursin("├── Stop time: 1", sshow)
    @test occursin("├── Stop iteration: Inf", sshow)
    @test occursin("├── Wall time limit: Inf", sshow)
    @test occursin("├── Callbacks: OrderedDict with 4 entries:", sshow)
    @test occursin("│   ├── stop_time_exceeded => Callback of stop_time_exceeded on IterationInterval(1)", sshow)
    @test occursin("│   └── nan_checker => Callback of nan_checker on IterationInterval(100)", sshow)
    @test occursin("└── Output writers: OrderedDict with no entries", sshow)

    csum = summary(model.clock)
    @test occursin("Clock(time = 0, iteration = 0, last_Δt = 0)", csum)

    @test summary(IterationInterval(10)) == "IterationInterval(10)"
    @test summary(TimeInterval(0.1)) == "TimeInterval(0.1)"
    @test summary(GeoDynamo.SpecifiedTimes(0.1, 0.5)) == "SpecifiedTimes(0.1, 0.5)"
    cb = Callback(sim -> nothing, schedule = IterationInterval(2))
    @test startswith(summary(cb), "Callback of ")
    @test endswith(summary(cb), "on IterationInterval(2)")
end
```

- [ ] **Step 2: Run → FAIL** (header format mismatches).

- [ ] **Step 3: Rewrite `src/api/show.jl` in full:**

```julia
# Oceananigans-style summaries and tree-style show methods.
# Model time is nondimensional → prettysummary; wall-clock → prettytime.

_arch_name(arch) = arch isa CPU ? "CPU" : "GPU"

# ── Grids ─────────────────────────────────────────────────────────────────────
function Base.summary(g::SphericalShellGrid)
    "SphericalShellGrid($(_arch_name(g.arch)), lmax=$(g.lmax), mmax=$(g.mmax), nr=$(g.nr))"
end
function Base.summary(g::SphericalBallGrid)
    "SphericalBallGrid($(_arch_name(g.arch)), lmax=$(g.lmax), mmax=$(g.mmax), nr=$(g.nr))"
end

# ── Schedules ────────────────────────────────────────────────────────────────
Base.summary(s::IterationInterval) = "IterationInterval($(s.interval))"
Base.summary(s::TimeInterval) = "TimeInterval($(prettysummary(s.interval)))"
Base.summary(s::WallTimeInterval) = "WallTimeInterval($(prettysummary(s.interval)))"
Base.summary(s::SpecifiedTimes) =
    "SpecifiedTimes($(join(map(prettysummary, s.times), ", ")))"

# ── Callbacks / writers ──────────────────────────────────────────────────────
_callable_name(f::Function) = string(nameof(f))
_callable_name(f) = string(nameof(typeof(f)))
Base.summary(cb::Callback) =
    "Callback of $(_callable_name(cb.func)) on $(summary(cb.schedule))"
Base.summary(cb::EnergyDiagnostics) = "EnergyDiagnostics on $(summary(cb.schedule))"
Base.summary(cb::SolenoidalMonitor) =
    "SolenoidalMonitor (threshold=$(prettysummary(cb.threshold))) on $(summary(cb.schedule))"
Base.summary(cb::SimulationProgress) = "SimulationProgress on $(summary(cb.schedule))"
Base.summary(cb::HealthCheck) = "HealthCheck on $(summary(cb.schedule))"
Base.summary(ow::FieldWriter) =
    "FieldWriter writing ($(join(ow.fields, ", "))) to $(ow.path) on $(summary(ow.schedule))"
Base.summary(ow::CheckpointWriter) =
    "CheckpointWriter writing to $(ow.path) on $(summary(ow.schedule))"

# ── Timesteppers ─────────────────────────────────────────────────────────────
Base.summary(ts::CNAB2) = "CNAB2(theta=$(prettysummary(ts.theta)))"
Base.summary(ts::ERK2) = "ERK2()"
Base.summary(ts::EAB2) =
    "EAB2(krylov_dimension=$(ts.krylov_dimension), tolerance=$(prettysummary(ts.tolerance)))"
Base.summary(ts::ETD) =
    "ETD(krylov_dimension=$(ts.krylov_dimension), tolerance=$(prettysummary(ts.tolerance)))"
Base.summary(ts::ThetaMethod) = "ThetaMethod(theta=$(prettysummary(ts.theta)))"

# ── Clock ────────────────────────────────────────────────────────────────────
Base.summary(c::Clock) =
    "Clock(time = $(prettysummary(c.time)), iteration = $(c.iteration), " *
    "last_Δt = $(prettysummary(c.last_dt)))"
Base.show(io::IO, ::MIME"text/plain", c::Clock) = print(io, summary(c))

# ── Model ────────────────────────────────────────────────────────────────────
function Base.summary(m::GeodynamoModel{T}) where {T}
    arch = _arch_name(m.state.backend.architecture)
    "GeodynamoModel{$arch, $T}(time = $(prettysummary(m.clock.time)), " *
    "iteration = $(m.clock.iteration))"
end

function Base.show(io::IO, ::MIME"text/plain", m::GeodynamoModel)
    p = m.state.parameters
    println(io, summary(m))
    println(io, "├── grid: ", summary(m.grid))
    println(io, "├── timestepper: ", summary(p.timestepper))
    println(io, "├── physics: Ek=", prettysummary(p.Ek), ", Pr=", prettysummary(p.Pr),
        ", Pm=", prettysummary(p.Pm), ", Sc=", prettysummary(p.Sc),
        ", Ra=", prettysummary(p.Ra))
    print(io, "└── active: magnetic=", p.include_magnetic,
        ", composition=", p.include_composition)
end

# ── Simulation ───────────────────────────────────────────────────────────────
function Base.summary(s::Simulation)
    "Simulation(Δt=$(prettysummary(s.dt)), stop_time=$(prettysummary(s.stop_time)), " *
    "stop_iteration=$(prettysummary(s.stop_iteration)))"
end

function _show_ordered_tree(io, label, dict)
    if isempty(dict)
        println(io, "├── ", label, ": OrderedDict with no entries")
    else
        n = length(dict)
        println(io, "├── ", label, ": OrderedDict with ", n,
            n == 1 ? " entry:" : " entries:")
        for (i, (k, v)) in enumerate(dict)
            conn = i == n ? "└──" : "├──"
            println(io, "│   ", conn, " ", k, " => ", summary(v))
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation)
    wall = sim._wall_start > 0 ? time() - sim._wall_start : 0.0
    println(io, "Simulation of ", summary(sim.model))
    println(io, "├── Next time step: ", prettysummary(sim.dt))
    println(io, "├── Elapsed wall time: ", prettytime(wall))
    println(io, "├── Stop time: ", prettysummary(sim.stop_time))
    println(io, "├── Stop iteration: ", prettysummary(sim.stop_iteration))
    println(io, "├── Wall time limit: ",
        isfinite(sim.wall_time_limit) ? prettytime(sim.wall_time_limit) : "Inf")
    _show_ordered_tree(io, "Callbacks", sim.callbacks)
    # Output writers are the last block → rebuild with └ root connector
    if isempty(sim.output_writers)
        print(io, "└── Output writers: OrderedDict with no entries")
    else
        n = length(sim.output_writers)
        println(io, "└── Output writers: OrderedDict with ", n,
            n == 1 ? " entry:" : " entries:")
        for (i, (k, v)) in enumerate(sim.output_writers)
            conn = i == n ? "└──" : "├──"
            sep = i == n ? print : println
            sep(io, "    ", conn, " ", k, " => ", summary(v))
        end
    end
end
```

Pre-implementation checks (verify, adjust if reality differs):
1. Timestepper struct field names: `grep -n "struct CNAB2\|struct EAB2\|struct ERK2\|struct ETD\|struct ThetaMethod" src/api/timesteppers.jl` and read the fields (`theta`, `krylov_dimension`, `tolerance` assumed).
2. `FieldWriter`/`CheckpointWriter` field names (`path`, `fields`, `schedule` assumed): `grep -n "struct FieldWriter\|struct CheckpointWriter" src/api/output_writers.jl`.
3. `p.timestepper` exists on SolverParameters (used at simulation.jl:138) — yes.
4. The old `Clock` show at clock.jl:35-38 must be DELETED (replaced here) — remove it from clock.jl to avoid duplicate-method overwrite warnings.

- [ ] **Step 4: Run → PASS.** Fix any field-name mismatches found by the greps rather than the tests.

- [ ] **Step 5: Run FULL suite** (other tests may pin the old `summary`/show strings — update THOSE tests to the new canonical strings if they were format-pins, but investigate first; a non-printing test failing means you broke behavior, stop and fix).

- [ ] **Step 6: Commit** `git add src/api/show.jl src/api/clock.jl test/oceananigans_api.jl && git commit -m "feat(api): Oceananigans-style show/summary for model, simulation, clock, schedules, callbacks, writers"`

---

### Task 10: Docs + full-suite gate

**Files:**
- Modify: `docs/src/api.md`
- Test: full suite

- [ ] **Step 1: Update `docs/src/api.md`.** Switch every `Simulation(model, dt = …)` example to `Δt = …` and add a short paragraph after the Simulation section:

```markdown
!!! note "Δt and dt"
    `Δt` is the canonical timestep keyword (Oceananigans convention); `dt` is
    accepted as an alias. `sim.Δt` reads and writes the same value as `sim.dt`.

`prettytime(t)` formats wall-clock durations ("2.341 seconds", "1.500 days").
Model time is nondimensional and prints compactly (e.g. `time = 0.25`).
```

Document the new BC names next to the existing BC section:

```markdown
Boundary conditions use Oceananigans-style names: `ValueBoundaryCondition`
(Dirichlet), `FluxBoundaryCondition` (Neumann), wrapped per field in
`FieldBoundaryConditions(inner=…, outer=…)`. The original names
(`FixedTemperature`, `FixedFlux`, `BoundaryConditions`) remain as aliases.
They can be passed per field or as one NamedTuple:

​```julia
model = GeodynamoModel(grid;
    boundary_conditions = (
        temperature = FieldBoundaryConditions(
            inner = ValueBoundaryCondition(1.0),
            outer = ValueBoundaryCondition(0.0)),
    ))
​```

`set!` accepts numbers, functions of `(r, θ, φ)`, and physical-grid arrays for
scalar fields:

​```julia
set!(model; temperature = (r, θ, φ) -> 1 - r)
​```
```

- [ ] **Step 2: Run the FULL suite, capture output:**

`~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/api_suite_final.log 2>&1; echo "exit=$?"`

Expected: exit=0 in a clean worktree. Memory note: a single run with ~3 IC-normalization failures may be a flake — re-run before attributing to these changes.

- [ ] **Step 3: Commit** `git add docs/src/api.md && git commit -m "docs: Δt canonical, BC names, set! forms, prettytime"`

- [ ] **Step 4: Push branch + open PR** (branch name `feat/oceananigans-api-parity`, base `main`). PR body: link the spec, list the five user decisions, paste the new `show` output for model + simulation, note zero changes outside `src/api/` + tests + docs.

---

## Self-Review (done at plan time)

- **Spec coverage:** §1 units→Task 1; §2 Δt→Task 3; §3 properties→Task 4; §4 defaults/running→Task 8; §5 SpecifiedTimes→Task 2 (AveragedTimeInterval deliberately skipped per spec); §6 printing→Task 9; §7 BCs→Tasks 5–6; §8 set!→Task 7; §9 tests/docs→inline per task + Task 10. ✓
- **Placeholders:** none — every step carries code or an exact command. Pre-implementation grep checks are explicit verification steps, not deferred work. ✓
- **Type consistency:** `prettysummary`/`prettytime` defined in Task 1, used in Tasks 8–9; `SpecifiedTimes.times` field used by Task 9 summary; `running` field added in Task 8 before Task 9 prints it; `Callback(func, schedule)` positional form (existing) used by `_default_callbacks`. `FieldBoundaryConditions` const-aliases keep `_thermal_bc_code` dispatch intact. ✓
