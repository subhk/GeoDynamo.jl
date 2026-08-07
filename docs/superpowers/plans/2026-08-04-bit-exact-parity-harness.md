# Bit-Exact Parity Harness (SP-0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the differential correctness gate that every later v3.0.0 refactor sub-project is verified against — given two implementations of the same physics, prove bit-identical evolved solver state or report exactly which coefficient diverged.

**Architecture:** A shared comparator (`state_digest.jl`) walks a `SolverState` and emits every floating-point array as reinterpreted bits. Two front-ends sit on top: a cross-commit script for API-preserving refactors, and an in-test A/B harness for the two clean-break sub-projects. A shared fixture builder feeds both and replaces the near-duplicate state builders in the two existing probe files.

**Tech Stack:** Julia 1.10–1.12, Test stdlib, MPI.jl, PencilArrays.jl, GeoDynamo internals.

**Spec:** `docs/superpowers/specs/2026-08-04-bit-exact-parity-harness-design.md`

## Global Constraints

- **No `src/` changes.** SP-0 adds `test/parity/*` and `scripts/*` and rewrites two existing test files. If a task appears to need a `src/` edit, stop and revise the spec instead of widening scope.
- **Single-threaded, single-rank for the gate.** Bit-exactness is only well-defined at fixed thread and rank count.
- **Branch:** `test/bit-exact-parity-harness`, already cut from `origin/main` at `e316546e`.
- **Julia binary:** `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia` — the `julia` shim is broken (root-owned `juliaup.json`).
- **Never pipe `Pkg.test()` through `tail`** — it masks the Julia exit code. Redirect to a file and inspect.
- **Never commit without explicit user permission.**
- **Step count for fixtures: 4.** Fewer than 2 is blind to `prev_nonlinear` corruption.
- **Scratchpad for transient files:** `/private/tmp/claude-501/-Users-subha-Documents-GitHub-GeoDynamo-jl/c06b3e0f-44d4-4e52-ac67-e6cc6df6233b/scratchpad`

## Domain Notes (read before Task 1)

You will not know these from the code alone:

1. **`solver_step!` regenerates the IC on its first call.** `src/solver/mainloop.jl:92` is `state.is_initialized || initialize_solver_fields!(state)`, and `SolverState` is built with `is_initialized = false` (`mainloop.jl:56`). `initialize_solver_fields!` calls `Random.seed!(42 + rank)` and rebuilds every field. **Any perturbation applied before the first step is silently erased.** The fixture must call `GeoDynamo.initialize_solver_fields!(st)` explicitly, *then* perturb. This is a live bug in the two existing probe files — their `MersenneTwister` perturbation is dead code — and Task 5 fixes it.

2. **Timer fields are wall-clock.** `SHTnsTemperatureField` and friends carry `computation_time`, `transform_time`, `comm_time`, `spectral_time` as `Ref{Float64}`. Digesting them guarantees a spurious failure on every run. They must be skipped by name.

3. **`config` and `pencil` back-references create cycles.** Every `SHTnsSpecField` holds a `config::SHTnsKitConfig`, which holds pencils and C transform plans. Recursing into them is both cyclic and meaningless. Skip by field name.

4. **Field arrays are `PencilArray`s.** Use `parent(f.data_real)` to reach the backing `Array`, as the existing probe files already do.

5. **Field names differ across a clean break.** In SP-2, `state.fields.temperature.temperature` becomes something else. So the comparator must support comparing two digests *without* requiring field names to match — positional and shape comparison only. Task 3 depends on this.

## File Structure

| File | Responsibility |
|---|---|
| `test/parity/state_digest.jl` | Walk a `SolverState`, emit `StateDigest`, compare two digests on bits. The only unit that knows about bits. |
| `test/parity/fixtures.jl` | Build `SolverState`s from a `(timestepper, scalar BC, wall code, magnetic, composition)` tuple. Declares the default and full matrices. The only unit that knows about `SolverParameters`. |
| `test/parity/ab_harness.jl` | Mechanism B: `@testset` generator asserting two in-tree implementations agree. |
| `test/parity/ab_harness_test.jl` | Proves the A/B harness both passes on agreement and reports on divergence. |
| `test/parity/digest_negative_controls.jl` | Proves the comparator can fail. |
| `test/parity/fixtures_test.jl` | Proves the fixture perturbation actually survives to the evolved state. |
| `scripts/parity_crosscommit.jl` | Mechanism C: build two git refs in worktrees, dump digests, diff. |
| `test/scalar_bc_timestepper_parity.jl` | Rewritten onto the shared fixture. Assertions unchanged. |
| `test/velocity_magnetic_bc_timestepper_parity.jl` | Rewritten onto the shared fixture. Assertions unchanged. |
| `test/runtests.jl` | Registration. |

---

### Task 1: State digest core

**Files:**
- Create: `test/parity/state_digest.jl`
- Create: `test/parity/digest_negative_controls.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `ParityDigest.FieldBits` — fields `name::String`, `dims::Vector{Int}`, `values::Vector{Float64}`
  - `ParityDigest.StateDigest` — fields `env::Dict{String,Any}`, `info::Dict{String,Any}`, `fields::Vector{FieldBits}`, `hash::UInt64`
  - `ParityDigest.digest_state(state) -> StateDigest`
  - `ParityDigest.digests_equal(a, b; compare_names::Bool = true) -> Tuple{Bool, String}`

- [ ] **Step 1: Write the failing negative-control tests**

Create `test/parity/digest_negative_controls.jl`:

```julia
using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

include(joinpath(@__DIR__, "state_digest.jl"))
using .ParityDigest

# A digest built by hand, so these controls test the COMPARATOR in isolation
# from the walker. The walker is exercised by fixtures_test.jl in Task 2.
function _ctl_digest(values::Vector{Float64}; name = "a.b", dims = [length(values)])
    env = Dict{String, Any}("nthreads" => 1, "nranks" => 1,
        "word_size" => 64, "julia" => "1.11.1")
    fb = ParityDigest.FieldBits(name, dims, copy(values))
    return ParityDigest.StateDigest(env, Dict{String, Any}(), [fb],
        ParityDigest._hash_fields([fb]))
end

@testset "digest comparator negative controls" begin
    base = [1.0, 2.0, 3.0]

    @testset "identical digests compare equal" begin
        ok, msg = ParityDigest.digests_equal(_ctl_digest(base), _ctl_digest(base))
        @test ok
        @test isempty(msg)
    end

    @testset "1 ULP difference is detected" begin
        perturbed = copy(base)
        perturbed[2] = nextfloat(perturbed[2])
        ok, msg = ParityDigest.digests_equal(_ctl_digest(base), _ctl_digest(perturbed))
        @test !ok
        @test occursin("a.b", msg)
        @test occursin("index 2", msg)
    end

    @testset "signed zero is detected" begin
        z = [0.0, 0.0]
        nz = [0.0, -0.0]
        ok, msg = ParityDigest.digests_equal(_ctl_digest(z), _ctl_digest(nz))
        @test !ok
    end

    @testset "matching NaNs in the same slot compare equal" begin
        n = [1.0, NaN, 3.0]
        ok, _ = ParityDigest.digests_equal(_ctl_digest(n), _ctl_digest(n))
        @test ok
    end

    @testset "environment mismatch reports non-comparable, not physics" begin
        a = _ctl_digest(base)
        b = _ctl_digest(base)
        b.env["nthreads"] = 4
        ok, msg = ParityDigest.digests_equal(a, b)
        @test !ok
        @test occursin("not comparable", msg)
        @test !occursin("index", msg)
    end

    @testset "shape mismatch is detected" begin
        a = _ctl_digest(base; dims = [3])
        b = _ctl_digest(base; dims = [1, 3])
        ok, msg = ParityDigest.digests_equal(a, b)
        @test !ok
        @test occursin("dims", msg)
    end

    @testset "name mismatch honours compare_names" begin
        a = _ctl_digest(base; name = "old.temperature")
        b = _ctl_digest(base; name = "new.payload")
        ok_strict, _ = ParityDigest.digests_equal(a, b; compare_names = true)
        @test !ok_strict
        ok_loose, _ = ParityDigest.digests_equal(a, b; compare_names = false)
        @test ok_loose
    end

    @testset "hash agreeing does not alone produce a pass" begin
        # Raw values are always confirmed, so a forged matching hash must still fail.
        a = _ctl_digest([1.0, 2.0])
        b = _ctl_digest([1.0, 3.0])
        forged = ParityDigest.StateDigest(b.env, b.info, b.fields, a.hash)
        ok, _ = ParityDigest.digests_equal(a, forged)
        @test !ok
    end
end
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. \
  test/parity/digest_negative_controls.jl
```
Expected: FAIL — `SystemError` / `could not open file .../state_digest.jl`.

- [ ] **Step 3: Implement the digest module**

Create `test/parity/state_digest.jl`:

```julia
"""
    ParityDigest

Bit-exact comparison of two `SolverState`s.

This is the shared comparator for the v3.0.0 refactor program. It exists because
the repo's existing end-to-end probes are INVARIANT probes: they assert a physical
property (boundary residual near zero) on the evolved state, which any correct
implementation satisfies. They cannot prove that a refactored implementation
matches the original. This module does.

Comparison is on reinterpreted bits, not `==` and not `isapprox`. `==` gives the
wrong answer twice for this purpose: `-0.0 == 0.0` is `true`, and `NaN == NaN` is
`false`. For a differential gate, matching NaNs in the same slot are a PASS —
objecting to a NaN is the invariant probes' job, not this one's.
"""
module ParityDigest

using GeoDynamo
using MPI

export FieldBits, StateDigest, digest_state, digests_equal

# Field names that must never be walked.
#
# :config and :pencil are back-references into SHTnsKitConfig / Pencil, which are
# cyclic and hold C transform plans.
# :domain is the static radial grid; it is already pinned by the recorded params.
# The four *_time fields are Ref{Float64} wall-clock counters — digesting them
# guarantees a spurious failure on every single run.
const SKIP_FIELDS = Set{Symbol}((
    :config, :pencil, :domain,
    :computation_time, :transform_time, :comm_time, :spectral_time,
))

struct FieldBits
    name::String
    dims::Vector{Int}
    values::Vector{Float64}
end

struct StateDigest
    env::Dict{String, Any}     # compared for comparability
    info::Dict{String, Any}    # recorded only, never compared
    fields::Vector{FieldBits}
    hash::UInt64
end

# Widening Float32 -> Float64 is exact and injective, so bit comparison on the
# widened values is still an exact comparison of the originals. This keeps the
# digest element-type agnostic.
_push_array!(out, name, A) =
    push!(out, FieldBits(name, collect(size(A)), Float64.(vec(A))))

function _walk!(out::Vector{FieldBits}, seen::Base.IdSet{Any}, name::String, x)
    x === nothing && return nothing

    if x isa GeoDynamo.PencilArray
        _push_array!(out, name, parent(x))
        return nothing
    elseif x isa AbstractArray
        # Integer arrays are BC type codes and mode tables: static, not evolved.
        eltype(x) <: AbstractFloat && _push_array!(out, name, x)
        return nothing
    elseif x isa GeoDynamo.SHTnsKitConfig
        return nothing
    end

    T = typeof(x)
    isstructtype(T) || return nothing
    # Numbers, Symbols, Strings are structs by isstructtype but carry no arrays.
    (x isa Number || x isa Symbol || x isa AbstractString) && return nothing

    if ismutable(x)
        x in seen && return nothing
        push!(seen, x)
    end

    for f in fieldnames(T)
        f in SKIP_FIELDS && continue
        isdefined(x, f) || continue
        _walk!(out, seen, string(name, ".", f), getfield(x, f))
    end
    return nothing
end

function _hash_fields(fields::Vector{FieldBits})
    h = UInt64(0x9e3779b97f4a7c15)
    for fb in fields
        h = hash(fb.name, h)
        h = hash(fb.dims, h)
        h = hash(reinterpret(UInt64, fb.values), h)
    end
    return h
end

"""
    digest_state(state) -> StateDigest

Walk `state.fields` and capture every floating-point array reachable from it.

`prev_nonlinear` / `prev_nl_*` are captured deliberately. They are not derived
state: they carry the CNAB2 history, and a refactor that corrupts them produces a
state that looks correct for exactly one step and diverges afterward.
"""
function digest_state(state)
    out = FieldBits[]
    _walk!(out, Base.IdSet{Any}(), "fields", state.fields)

    env = Dict{String, Any}(
        "nthreads" => Threads.nthreads(),
        "nranks" => MPI.Initialized() ? MPI.Comm_size(MPI.COMM_WORLD) : 1,
        "word_size" => Sys.WORD_SIZE,
        "julia" => string(VERSION),
    )
    # Recorded but NOT compared: a clean-break sub-project legitimately changes
    # how parameters print, and that must not read as a physics difference.
    info = Dict{String, Any}(
        "params" => string(state.parameters),
        "time" => state.time,
        "step" => state.step,
    )
    return StateDigest(env, info, out, _hash_fields(out))
end

function _first_difference(a::FieldBits, b::FieldBits)
    ba = reinterpret(UInt64, a.values)
    bb = reinterpret(UInt64, b.values)
    @inbounds for i in eachindex(ba)
        if ba[i] != bb[i]
            va, vb = a.values[i], b.values[i]
            ulps = (isfinite(va) && isfinite(vb)) ?
                   string(abs(reinterpret(Int64, va) - reinterpret(Int64, vb))) :
                   "n/a"
            return "$(a.name): differs at index $i of $(a.dims) — " *
                   "$(repr(va)) vs $(repr(vb)) ($ulps ULP)"
        end
    end
    return ""
end

"""
    digests_equal(a, b; compare_names = true) -> (ok::Bool, message::String)

Bit-compare two digests.

`compare_names = false` is for mechanism B (in-tree A/B across a clean break),
where the two implementations legitimately expose different field names for the
same physical quantity. Order, shape, and bits must still match exactly.

The stored hash is a fast-rejection convenience only. Equality is always
confirmed against the raw values before this returns `true`, so a hash collision
cannot produce a false green.
"""
function digests_equal(a::StateDigest, b::StateDigest; compare_names::Bool = true)
    if a.env != b.env
        return (false, "digests not comparable: env differs — $(a.env) vs $(b.env)")
    end
    if length(a.fields) != length(b.fields)
        return (false, "field count differs: $(length(a.fields)) vs $(length(b.fields))")
    end
    for (fa, fb) in zip(a.fields, b.fields)
        if compare_names && fa.name != fb.name
            return (false, "field name differs: $(fa.name) vs $(fb.name)")
        end
        if fa.dims != fb.dims
            return (false, "$(fa.name): dims differ — $(fa.dims) vs $(fb.dims)")
        end
        msg = _first_difference(fa, fb)
        isempty(msg) || return (false, msg)
    end
    return (true, "")
end

end # module
```

- [ ] **Step 4: Run the negative controls to verify they pass**

Run:
```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. \
  test/parity/digest_negative_controls.jl
```
Expected: PASS, all 8 testsets green, zero failures.

- [ ] **Step 5: Register in the suite**

In `test/runtests.jl`, inside the top-level `@testset "GeoDynamo.jl"`, add:

```julia
    @testset "Parity digest negative controls" begin
        include(joinpath(TEST_DIR, "parity", "digest_negative_controls.jl"))
    end
```

- [ ] **Step 6: Commit**

```bash
git add test/parity/state_digest.jl test/parity/digest_negative_controls.jl test/runtests.jl
git commit -m "test(parity): bit-exact state digest comparator

Differential gate for the v3.0.0 refactor program. Compares on
reinterpreted bits rather than == or isapprox: == answers wrong twice
here, since -0.0 == 0.0 and NaN != NaN. Matching NaNs in the same slot
are a pass for a differential gate.

Ships negative controls first, because a green result is this module's
output and a comparator that cannot fail is worse than no comparator."
```

---

### Task 2: Shared fixtures

**Files:**
- Create: `test/parity/fixtures.jl`
- Create: `test/parity/fixtures_test.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `ParityDigest.digest_state` from Task 1.
- Produces:
  - `ParityFixtures.ParityCase` — fields `timestepper_name::String`, `timestepper`, `scalar_code::Int`, `wall_code::Int`, `magnetic::Bool`, `composition::Bool`
  - `ParityFixtures.build_state(case::ParityCase; seed::Int = 11)` → an initialized, perturbed `SolverState`
  - `ParityFixtures.evolve!(state; nsteps::Int = 4)` → the same state, stepped
  - `ParityFixtures.PARITY_MATRIX_DEFAULT::Vector{ParityCase}`
  - `ParityFixtures.PARITY_MATRIX_FULL::Vector{ParityCase}`
  - `ParityFixtures.select_matrix()` → full matrix if `ENV["GEODYNAMO_PARITY_FULL"] == "1"`, else default

- [ ] **Step 1: Write the failing fixture test**

Create `test/parity/fixtures_test.jl`:

```julia
using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

include(joinpath(@__DIR__, "state_digest.jl"))
include(joinpath(@__DIR__, "fixtures.jl"))
using .ParityDigest
using .ParityFixtures

@testset "parity fixtures" begin
    case = ParityFixtures.PARITY_MATRIX_DEFAULT[1]

    @testset "perturbation survives to the evolved state" begin
        # This is the whole point of the fixture. solver_step! regenerates the IC
        # on its first call unless is_initialized is already true
        # (src/solver/mainloop.jl:92), so a naive perturb-then-step silently
        # discards the seed. Two different seeds MUST produce different states.
        a = ParityFixtures.evolve!(ParityFixtures.build_state(case; seed = 11))
        b = ParityFixtures.evolve!(ParityFixtures.build_state(case; seed = 12))
        ok, _ = ParityDigest.digests_equal(
            ParityDigest.digest_state(a), ParityDigest.digest_state(b))
        @test !ok
    end

    @testset "same seed is reproducible bit-for-bit" begin
        a = ParityFixtures.evolve!(ParityFixtures.build_state(case; seed = 11))
        b = ParityFixtures.evolve!(ParityFixtures.build_state(case; seed = 11))
        ok, msg = ParityDigest.digests_equal(
            ParityDigest.digest_state(a), ParityDigest.digest_state(b))
        @test ok
        @test isempty(msg)
    end

    @testset "digest captures the fields that matter" begin
        st = ParityFixtures.evolve!(ParityFixtures.build_state(
            ParityFixtures.ParityCase("CNAB2", GeoDynamo.CNAB2(), 1, 1, true, true)))
        names = [f.name for f in ParityDigest.digest_state(st).fields]
        @test any(n -> occursin("velocity.toroidal.data_real", n), names)
        @test any(n -> occursin("velocity.poloidal.data_real", n), names)
        @test any(n -> occursin("temperature.spectral.data_real", n), names)
        @test any(n -> occursin("magnetic.toroidal.data_real", n), names)
        @test any(n -> occursin("composition.spectral.data_real", n), names)
        # CNAB2 history must be captured; corrupting it stays invisible for one step.
        @test any(n -> occursin("prev_nl_toroidal", n), names)
        # Wall-clock timers must NOT be captured or every run fails spuriously.
        @test !any(n -> occursin("computation_time", n), names)
        @test !any(n -> occursin("transform_time", n), names)
    end

    @testset "matrices are well formed" begin
        @test length(ParityFixtures.PARITY_MATRIX_FULL) == 192
        d = ParityFixtures.PARITY_MATRIX_DEFAULT
        @test 8 <= length(d) <= 24
        # every level of every factor appears at least once
        @test sort(unique(c.timestepper_name for c in d)) == ["CNAB2", "ERK2", "RK3"]
        @test sort(unique(c.scalar_code for c in d)) == [1, 2, 3, 4]
        @test sort(unique(c.wall_code for c in d)) == [1, 2, 3, 4]
        @test sort(unique(c.magnetic for c in d)) == [false, true]
        @test sort(unique(c.composition for c in d)) == [false, true]
    end
end
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. \
  test/parity/fixtures_test.jl
```
Expected: FAIL — `could not open file .../fixtures.jl`.

- [ ] **Step 3: Implement the fixtures module**

Create `test/parity/fixtures.jl`:

```julia
"""
    ParityFixtures

Shared `SolverState` builder for the parity harness.

Extracted from the near-identical `_sbp_state` (scalar_bc_timestepper_parity.jl)
and `_vm_state` (velocity_magnetic_bc_timestepper_parity.jl).

The grid is the one both of those already use: small enough that a wide matrix is
affordable, and already demonstrated to exercise the ERK2 magnetic Robin, ERK2
scalar Neumann, and ERK2 l=0 NN bugs.
"""
module ParityFixtures

using GeoDynamo
using MPI
using Random

export ParityCase, build_state, evolve!

const VAL = GeoDynamo.ValueBoundaryCondition
const FLX = GeoDynamo.FluxBoundaryCondition

# scalar_code => BoundaryConditions.  1 = DD, 2 = DN, 3 = ND, 4 = NN.
const SCALAR_BCS = Dict(
    1 => GeoDynamo.BoundaryConditions(inner = VAL(1.0), outer = VAL(0.0)),
    2 => GeoDynamo.BoundaryConditions(inner = VAL(1.0), outer = FLX(0.0)),
    3 => GeoDynamo.BoundaryConditions(inner = FLX(1.0), outer = VAL(0.0)),
    4 => GeoDynamo.BoundaryConditions(inner = FLX(1.0), outer = FLX(0.0)),
)

# wall_code => (inner, outer).  1 = NS/NS, 2 = NS/SF, 3 = SF/NS, 4 = SF/SF.
const WALL_BCS = Dict(
    1 => (GeoDynamo.NoSlip(), GeoDynamo.NoSlip()),
    2 => (GeoDynamo.NoSlip(), GeoDynamo.StressFree()),
    3 => (GeoDynamo.StressFree(), GeoDynamo.NoSlip()),
    4 => (GeoDynamo.StressFree(), GeoDynamo.StressFree()),
)

const TIMESTEPPERS = [
    ("CNAB2", GeoDynamo.CNAB2()),
    ("ERK2", GeoDynamo.ExponentialRungeKutta2()),
    ("RK3", GeoDynamo.RungeKutta3()),
]

struct ParityCase
    timestepper_name::String
    timestepper::Any
    scalar_code::Int
    wall_code::Int
    magnetic::Bool
    composition::Bool
end

function Base.show(io::IO, c::ParityCase)
    print(io, "$(c.timestepper_name)/scalar$(c.scalar_code)/wall$(c.wall_code)",
        c.magnetic ? "/mag" : "", c.composition ? "/comp" : "")
end

"""
    build_state(case; seed = 11)

Build a `SolverState`, initialize its fields, then perturb them deterministically.

The explicit `initialize_solver_fields!` call is load-bearing. `solver_step!`
does `state.is_initialized || initialize_solver_fields!(state)`
(src/solver/mainloop.jl:92), and `SolverState` is constructed with
`is_initialized = false` (mainloop.jl:56). Perturbing before that flag is set
means the first step silently erases the perturbation and every seed produces an
identical trajectory.
"""
function build_state(case::ParityCase; seed::Int = 11)
    kw = Dict{Symbol, Any}(
        :geometry => :shell, :lmax => 4, :mmax => 4, :nlat => 12, :nlon => 24,
        :nr => 16, :nr_inner => 4, :radial_bandwidth => 3, :radius_ratio => 0.35,
        :Ek => 1e-3, :Ra => 1e3, :Pm => 1.0, :Pr => 1.0, :timestep => 1e-5,
        :include_magnetic => case.magnetic,
        :include_composition => case.composition,
        :timestepper => case.timestepper,
        :temperature_bcs => SCALAR_BCS[case.scalar_code],
        :velocity_bcs => WALL_BCS[case.wall_code],
    )
    case.composition && (kw[:composition_bcs] = SCALAR_BCS[case.scalar_code])

    st = GeoDynamo.initialize_solver_state(
        Float64; params = GeoDynamo.SolverParameters(; kw...))

    GeoDynamo.initialize_solver_fields!(st)

    rng = MersenneTwister(seed)
    for f in _perturbable(st)
        dr = parent(f.data_real)
        di = parent(f.data_imag)
        dr .+= 1e-3 .* (rand(rng, size(dr)...) .- 0.5)
        di .+= 1e-3 .* (rand(rng, size(di)...) .- 0.5)
    end
    return st
end

function _perturbable(st)
    fs = Any[st.fields.temperature.spectral,
        st.fields.velocity.toroidal,
        st.fields.velocity.poloidal]
    st.fields.magnetic === nothing ||
        append!(fs, (st.fields.magnetic.toroidal, st.fields.magnetic.poloidal))
    st.fields.composition === nothing ||
        push!(fs, st.fields.composition.spectral)
    return fs
end

"""
    evolve!(state; nsteps = 4)

Step the real solver. Four steps, not one: CNAB2's `prev_nonlinear` history does
not participate until the second step, so a shorter trajectory is blind to
exactly the corruption the digest captures it for.
"""
function evolve!(state; nsteps::Int = 4)
    for _ in 1:nsteps
        GeoDynamo.solver_step!(state)
    end
    return state
end

const PARITY_MATRIX_FULL = [
    ParityCase(tsname, ts, sc, wc, mag, comp)
    for (tsname, ts) in TIMESTEPPERS
    for sc in 1:4
    for wc in 1:4
    for mag in (false, true)
    for comp in (false, true)
]

# Pairwise-covering subset: every level of every factor appears, and every pair of
# factors is exercised at least once. 12 cases against the full matrix's 192.
const PARITY_MATRIX_DEFAULT = [
    ParityCase("CNAB2", TIMESTEPPERS[1][2], 1, 1, false, false),
    ParityCase("CNAB2", TIMESTEPPERS[1][2], 2, 2, true, true),
    ParityCase("CNAB2", TIMESTEPPERS[1][2], 3, 3, true, false),
    ParityCase("CNAB2", TIMESTEPPERS[1][2], 4, 4, false, true),
    ParityCase("ERK2", TIMESTEPPERS[2][2], 1, 2, true, false),
    ParityCase("ERK2", TIMESTEPPERS[2][2], 2, 1, false, true),
    ParityCase("ERK2", TIMESTEPPERS[2][2], 3, 4, false, false),
    ParityCase("ERK2", TIMESTEPPERS[2][2], 4, 3, true, true),
    ParityCase("RK3", TIMESTEPPERS[3][2], 1, 3, false, true),
    ParityCase("RK3", TIMESTEPPERS[3][2], 2, 4, true, false),
    ParityCase("RK3", TIMESTEPPERS[3][2], 3, 1, true, true),
    ParityCase("RK3", TIMESTEPPERS[3][2], 4, 2, false, false),
]

"""
    select_matrix()

The default subset, or all 192 when `GEODYNAMO_PARITY_FULL=1`. The full matrix is
for a once-per-sub-project pre-PR run, not for routine use.
"""
select_matrix() = get(ENV, "GEODYNAMO_PARITY_FULL", "0") == "1" ?
                  PARITY_MATRIX_FULL : PARITY_MATRIX_DEFAULT

end # module
```

- [ ] **Step 4: Run the fixture test to verify it passes**

Run:
```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. \
  test/parity/fixtures_test.jl
```
Expected: PASS, all 4 testsets green.

If "perturbation survives" fails (digests compare *equal* for different seeds), the explicit `initialize_solver_fields!` is not taking effect — do not weaken the test; investigate `is_initialized` handling instead.

- [ ] **Step 5: Register in the suite**

In `test/runtests.jl`, after the Task 1 entry:

```julia
    @testset "Parity fixtures" begin
        include(joinpath(TEST_DIR, "parity", "fixtures_test.jl"))
    end
```

- [ ] **Step 6: Commit**

```bash
git add test/parity/fixtures.jl test/parity/fixtures_test.jl test/runtests.jl
git commit -m "test(parity): shared solver-state fixture and case matrices

Extracted from the near-duplicate _sbp_state and _vm_state builders.

Calls initialize_solver_fields! explicitly before perturbing. solver_step!
regenerates the IC on its first call unless is_initialized is already set,
so the existing probes' MersenneTwister perturbation was dead code — every
seed produced the same trajectory. The fixture test pins this by asserting
two seeds diverge."
```

---

### Task 3: A/B harness (mechanism B)

**Files:**
- Create: `test/parity/ab_harness.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `ParityFixtures.ParityCase`, `ParityFixtures.select_matrix`, `ParityDigest.digest_state`, `ParityDigest.digests_equal`.
- Produces: `ParityAB.assert_ab_parity(legacy_build, new_build; cases, compare_names, nsteps)` — takes two `case -> SolverState` callables, returns `nothing`, emits `@test`s.

- [ ] **Step 1: Write the failing self-test**

The harness and its test live in separate files: `ab_harness.jl` defines a module that sub-projects will `include`, so it cannot carry its own `@testset` at load time.

Create `test/parity/ab_harness_test.jl`:

```julia
using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

include(joinpath(@__DIR__, "state_digest.jl"))
include(joinpath(@__DIR__, "fixtures.jl"))
include(joinpath(@__DIR__, "ab_harness.jl"))
using .ParityDigest
using .ParityFixtures
using .ParityAB

@testset "A/B harness" begin
    cases = ParityFixtures.PARITY_MATRIX_DEFAULT[1:2]

    @testset "identical builders agree" begin
        # Both sides build the same way, so this must pass. This is the shape
        # every clean-break sub-project will use: two builders, one assertion.
        build(case) = ParityFixtures.build_state(case; seed = 11)
        ParityAB.assert_ab_parity(build, build; cases = cases)
    end

    @testset "a divergent builder is caught" begin
        build_a(case) = ParityFixtures.build_state(case; seed = 11)
        build_b(case) = ParityFixtures.build_state(case; seed = 12)
        results = ParityAB.compare_ab(build_a, build_b; cases = cases)
        @test all(r -> !r.ok, results)
        @test all(r -> !isempty(r.message), results)
    end
end
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. \
  test/parity/ab_harness_test.jl
```
Expected: FAIL — `could not open file .../ab_harness.jl`.

- [ ] **Step 3: Implement the A/B harness**

Create `test/parity/ab_harness.jl`:

```julia
"""
    ParityAB

Mechanism B of the parity harness: in-tree A/B.

For a sub-project that BREAKS the public API (SP-2 scalar collapse, SP-3 geometry
type), a single driver script cannot reference a type that exists on only one
side of the change. So both implementations stay live in `src/` for the duration
of the sub-project, this harness asserts they agree bit-for-bit, and the legacy
path is deleted in that sub-project's final commit.

Usage from a sub-project's temporary test file:

    ParityAB.assert_ab_parity(
        case -> build_with_legacy_scalar_field(case),
        case -> build_with_collapsed_scalar_field(case);
        compare_names = false)
"""
module ParityAB

using Test
using ..ParityDigest
using ..ParityFixtures

export assert_ab_parity, compare_ab

struct ABResult
    case::ParityFixtures.ParityCase
    ok::Bool
    message::String
end

"""
    compare_ab(legacy_build, new_build; cases, compare_names, nsteps) -> Vector{ABResult}

Build, evolve, and digest both sides of every case. Returns results without
asserting, so a caller can inspect them — used by the harness's own self-test to
prove it can report a difference.
"""
function compare_ab(legacy_build, new_build;
        cases = ParityFixtures.select_matrix(),
        compare_names::Bool = false,
        nsteps::Int = 4)
    results = ABResult[]
    for case in cases
        a = ParityFixtures.evolve!(legacy_build(case); nsteps = nsteps)
        b = ParityFixtures.evolve!(new_build(case); nsteps = nsteps)
        ok, msg = ParityDigest.digests_equal(
            ParityDigest.digest_state(a), ParityDigest.digest_state(b);
            compare_names = compare_names)
        push!(results, ABResult(case, ok, msg))
    end
    return results
end

"""
    assert_ab_parity(legacy_build, new_build; cases, compare_names, nsteps)

Assert every case agrees bit-for-bit, one `@test` per case so a failure names the
configuration that diverged.

`compare_names` defaults to `false` because this mechanism exists for clean
breaks, where the two implementations legitimately expose different field names
for the same quantity. Order, shape, and bits must still match exactly.
"""
function assert_ab_parity(legacy_build, new_build;
        cases = ParityFixtures.select_matrix(),
        compare_names::Bool = false,
        nsteps::Int = 4)
    for r in compare_ab(legacy_build, new_build;
        cases = cases, compare_names = compare_names, nsteps = nsteps)
        @testset "$(r.case)" begin
            @test r.ok
            r.ok || @info "A/B parity failure" case = r.case detail = r.message
        end
    end
    return nothing
end

end # module
```

Note the `using ..ParityDigest` / `using ..ParityFixtures`: `ab_harness.jl` assumes the two sibling modules are already included into the same enclosing scope, which the test file does.

- [ ] **Step 4: Run the self-test to verify it passes**

Run:
```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. \
  test/parity/ab_harness_test.jl
```
Expected: PASS. Both testsets green — identical builders agree, divergent builders are reported.

- [ ] **Step 5: Register in the suite**

In `test/runtests.jl`, after the Task 2 entry:

```julia
    @testset "Parity A/B harness" begin
        include(joinpath(TEST_DIR, "parity", "ab_harness_test.jl"))
    end
```

- [ ] **Step 6: Commit**

```bash
git add test/parity/ab_harness.jl test/parity/ab_harness_test.jl test/runtests.jl
git commit -m "test(parity): in-tree A/B harness for clean-break sub-projects

Mechanism B. Cross-commit diffing cannot gate SP-2 or SP-3, because a
single driver cannot reference a type that exists on only one side of a
clean break. Both implementations stay live, this asserts they agree
bit-for-bit, and the legacy path goes in the sub-project's last commit.

compare_names defaults to false: a clean break legitimately renames the
field holding the same physical quantity. Order, shape and bits still
must match exactly."
```

---

### Task 4: Cross-commit script (mechanism C)

**Files:**
- Create: `scripts/parity_crosscommit.jl`

**Interfaces:**
- Consumes: `ParityFixtures`, `ParityDigest` (loaded from within each worktree's own checkout).
- Produces: a CLI. `julia --project=. scripts/parity_crosscommit.jl <refA> <refB>`; exit code 0 on bit-identical, 1 otherwise.

- [ ] **Step 1: Write the script**

Create `scripts/parity_crosscommit.jl`:

```julia
#!/usr/bin/env julia
"""
Cross-commit bit-exact parity check — mechanism C of the SP-0 harness.

    julia --project=. scripts/parity_crosscommit.jl <refA> <refB>
    GEODYNAMO_PARITY_FULL=1 julia --project=. scripts/parity_crosscommit.jl <refA> <refB>

Builds each git ref in its own worktree, runs an identical driver under each that
evolves every fixture in the selected matrix and serializes the resulting digest,
then compares the two dumps.

This is NOT a Pkg.test() test. It needs two builds, and cross-platform
bit-exactness is not something CI can assert — a committed reference snapshot
would fail on macOS and on Julia 1.10/1.12 for reasons unrelated to any refactor.
It is a local pre-merge gate whose result belongs in the PR body.

Digest dumps are written under a scratch directory and never committed.
"""

using Serialization

const REPO = normpath(joinpath(@__DIR__, ".."))
const JULIA = joinpath(homedir(),
    ".julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia")

# Driver executed INSIDE each worktree. It must only use API that exists in both
# refs — keep it to the parity modules and Serialization.
const DRIVER = raw"""
using Serialization
include(joinpath(@__DIR__, "..", "test", "parity", "state_digest.jl"))
include(joinpath(@__DIR__, "..", "test", "parity", "fixtures.jl"))
using .ParityDigest
using .ParityFixtures
using MPI
MPI.Initialized() || MPI.Init()

out = ARGS[1]
dumps = Dict{String, Any}()
for case in ParityFixtures.select_matrix()
    st = ParityFixtures.evolve!(ParityFixtures.build_state(case))
    dumps[string(case)] = ParityDigest.digest_state(st)
end
serialize(out, dumps)
println("wrote $(length(dumps)) digests to $out")
"""

function dump_ref(ref::String, scratch::String)
    wt = joinpath(scratch, "wt-" * replace(ref, r"[^A-Za-z0-9]" => "_"))
    out = joinpath(scratch, "digest-" * basename(wt) * ".jls")
    isdir(wt) && run(`git -C $REPO worktree remove --force $wt`)
    run(`git -C $REPO worktree add --detach $wt $ref`)
    try
        drv = joinpath(wt, "scripts", "_parity_driver.jl")
        mkpath(dirname(drv))
        write(drv, DRIVER)
        run(`$JULIA --project=$wt --threads=1 $drv $out`)
    finally
        run(`git -C $REPO worktree remove --force $wt`)
    end
    return out
end

function main()
    length(ARGS) == 2 || error("usage: parity_crosscommit.jl <refA> <refB>")
    refa, refb = ARGS

    scratch = get(ENV, "GEODYNAMO_PARITY_SCRATCH", mktempdir())
    mkpath(scratch)
    println("scratch: $scratch")

    da = deserialize(dump_ref(refa, scratch))
    db = deserialize(dump_ref(refb, scratch))

    # Compare using THIS checkout's comparator, so both dumps are judged by one
    # implementation even if the comparator itself changed between the refs.
    include(joinpath(REPO, "test", "parity", "state_digest.jl"))

    keys_a, keys_b = sort(collect(keys(da))), sort(collect(keys(db)))
    if keys_a != keys_b
        println("FAIL: case sets differ")
        println("  only in $refa: ", setdiff(keys_a, keys_b))
        println("  only in $refb: ", setdiff(keys_b, keys_a))
        exit(1)
    end

    failures = 0
    for k in keys_a
        ok, msg = ParityDigest.digests_equal(da[k], db[k])
        if ok
            println("  ok    $k")
        else
            failures += 1
            println("  DIFF  $k")
            println("        $msg")
        end
    end

    println()
    if failures == 0
        println("PASS: $(length(keys_a)) cases bit-identical between $refa and $refb")
        exit(0)
    else
        println("FAIL: $failures of $(length(keys_a)) cases differ")
        exit(1)
    end
end

main()
```

- [ ] **Step 2: Run the self-check — a ref against itself must be bit-identical**

Run:
```bash
cd /Users/subha/Documents/GitHub/GeoDynamo.jl
GEODYNAMO_PARITY_FULL=1 \
  ~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. \
  scripts/parity_crosscommit.jl HEAD HEAD \
  > /private/tmp/claude-501/-Users-subha-Documents-GitHub-GeoDynamo-jl/c06b3e0f-44d4-4e52-ac67-e6cc6df6233b/scratchpad/crosscommit-self.log 2>&1
echo "exit: $?"
tail -5 /private/tmp/claude-501/-Users-subha-Documents-GitHub-GeoDynamo-jl/c06b3e0f-44d4-4e52-ac67-e6cc6df6233b/scratchpad/crosscommit-self.log
```
Expected: exit 0, final line `PASS: 192 cases bit-identical between HEAD and HEAD`.

This is the harness's own determinism check and uses the full matrix, not the default subset. A failure here means the harness is nondeterministic and nothing built on it can be trusted — investigate before continuing.

- [ ] **Step 3: Verify a real difference is caught**

Temporarily perturb a physics constant to prove the script reports a diff rather than always passing:

```bash
cd /Users/subha/Documents/GitHub/GeoDynamo.jl
git stash list > /dev/null  # ensure clean tree first
# create a throwaway commit with a deliberately altered timestep default
sed -i '' 's/:timestep => 1e-5/:timestep => 1.0000001e-5/' test/parity/fixtures.jl
git commit -qam "TEMP: perturb fixture timestep to prove the gate fails"
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. \
  scripts/parity_crosscommit.jl HEAD~1 HEAD; echo "exit: $?"
git reset --hard HEAD~1
```
Expected: exit 1, with `DIFF` lines naming specific cases and a `... differs at index N ... ULP` message.

- [ ] **Step 4: Commit**

```bash
git add scripts/parity_crosscommit.jl
git commit -m "test(parity): cross-commit bit-exact gate

Mechanism C. Builds two refs in worktrees, evolves every fixture under
each, compares digests. Covers the API-preserving sub-projects, and is the
only meaningful gate for pure file motion.

Not a Pkg.test() test: it needs two builds, and cross-platform
bit-exactness is not something CI can assert. A committed reference
snapshot would fail on macOS and Julia 1.10/1.12 for reasons unrelated to
any refactor, which is exactly the trap this design avoids."
```

---

### Task 5: Rewrite the two existing probes onto the shared fixture

**Files:**
- Modify: `test/scalar_bc_timestepper_parity.jl`
- Modify: `test/velocity_magnetic_bc_timestepper_parity.jl`

**Interfaces:**
- Consumes: `ParityFixtures.build_state`, `ParityFixtures.SCALAR_BCS`, `ParityFixtures.WALL_BCS`.
- Produces: nothing new. Both files keep their existing residual helpers and assertions verbatim.

**This task is where SP-0 can silently do harm.** A rewrite that weakens an assertion is invisible to a green suite. The assertions, tolerances, `NSTEPS`, and the `@test ni > 0` vacuity guards must survive byte-identical; only state construction changes.

- [ ] **Step 1: Record the current assertions**

```bash
cd /Users/subha/Documents/GitHub/GeoDynamo.jl
grep -n "@test\|rel_tol\|NSTEPS" test/scalar_bc_timestepper_parity.jl \
  test/velocity_magnetic_bc_timestepper_parity.jl \
  > /private/tmp/claude-501/-Users-subha-Documents-GitHub-GeoDynamo-jl/c06b3e0f-44d4-4e52-ac67-e6cc6df6233b/scratchpad/probe-assertions-before.txt
cat /private/tmp/claude-501/-Users-subha-Documents-GitHub-GeoDynamo-jl/c06b3e0f-44d4-4e52-ac67-e6cc6df6233b/scratchpad/probe-assertions-before.txt
```

- [ ] **Step 2: Replace `_sbp_state` with the shared fixture**

In `test/scalar_bc_timestepper_parity.jl`, add near the top after `using Random`:

```julia
include(joinpath(@__DIR__, "parity", "fixtures.jl"))
using .ParityFixtures
```

Delete the `_sbp_state` function body and replace it with a shim that preserves the existing call signature so no call site below changes:

```julia
# State construction now comes from the shared parity fixture. The previous local
# builder perturbed the fields BEFORE is_initialized was set, so solver_step!
# regenerated the IC on its first call and discarded the perturbation entirely —
# every seed produced the same trajectory. ParityFixtures.build_state calls
# initialize_solver_fields! first, so the perturbation now survives and these
# assertions run against a genuinely richer state.
function _sbp_state(timestepper, temp_bcs; composition_bcs = nothing, seed = 11)
    code = findfirst(c -> ParityFixtures.SCALAR_BCS[c] === temp_bcs, 1:4)
    case = ParityFixtures.ParityCase(
        string(typeof(timestepper).name.name), timestepper,
        code === nothing ? 1 : code, 1, false, composition_bcs !== nothing)
    return ParityFixtures.build_state(case; seed = seed)
end
```

- [ ] **Step 3: Run the scalar probe and confirm it still passes**

Run:
```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. \
  test/scalar_bc_timestepper_parity.jl
```
Expected: PASS, same testset names and counts as before.

If a residual assertion now FAILS, that is a genuine finding, not a regression to paper over: the assertion was previously being evaluated on the default IC rather than a perturbed state. Report it and stop rather than loosening `rel_tol`.

- [ ] **Step 4: Apply the same change to the velocity/magnetic probe**

In `test/velocity_magnetic_bc_timestepper_parity.jl`, same two edits:

```julia
include(joinpath(@__DIR__, "parity", "fixtures.jl"))
using .ParityFixtures
```

```julia
function _vm_state(timestepper; vel_bcs = nothing, magnetic = false, seed = 11)
    code = vel_bcs === nothing ? 1 :
           something(findfirst(c -> ParityFixtures.WALL_BCS[c] === vel_bcs, 1:4), 1)
    case = ParityFixtures.ParityCase(
        string(typeof(timestepper).name.name), timestepper, 1, code, magnetic, false)
    return ParityFixtures.build_state(case; seed = seed)
end
```

- [ ] **Step 5: Run the velocity/magnetic probe**

Run:
```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. \
  test/velocity_magnetic_bc_timestepper_parity.jl
```
Expected: PASS.

- [ ] **Step 6: Verify no assertion was weakened**

```bash
cd /Users/subha/Documents/GitHub/GeoDynamo.jl
grep -n "@test\|rel_tol\|NSTEPS" test/scalar_bc_timestepper_parity.jl \
  test/velocity_magnetic_bc_timestepper_parity.jl \
  > /private/tmp/claude-501/-Users-subha-Documents-GitHub-GeoDynamo-jl/c06b3e0f-44d4-4e52-ac67-e6cc6df6233b/scratchpad/probe-assertions-after.txt
diff /private/tmp/claude-501/-Users-subha-Documents-GitHub-GeoDynamo-jl/c06b3e0f-44d4-4e52-ac67-e6cc6df6233b/scratchpad/probe-assertions-before.txt \
     /private/tmp/claude-501/-Users-subha-Documents-GitHub-GeoDynamo-jl/c06b3e0f-44d4-4e52-ac67-e6cc6df6233b/scratchpad/probe-assertions-after.txt
```
Expected: no output beyond line-number shifts. Any changed tolerance, removed `@test`, or reduced `NSTEPS` is a defect in this task — revert it.

- [ ] **Step 7: Commit**

```bash
git add test/scalar_bc_timestepper_parity.jl test/velocity_magnetic_bc_timestepper_parity.jl
git commit -m "test: move BC probes onto the shared parity fixture

Both files built states through near-identical private helpers, and both
perturbed the fields before is_initialized was set — so solver_step!
regenerated the IC on its first call and discarded the perturbation. The
seed was inert and both probes ran against the default IC.

Assertions, tolerances and step counts are unchanged; only state
construction moved. Verified by diffing the extracted assertion lines
before and after."
```

---

### Task 6: Full-suite verification

**Files:**
- Modify: none (verification only, plus any registration fix the suite reveals).

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: evidence for the PR body.

- [ ] **Step 1: Run the full suite, redirected to a file**

```bash
cd /Users/subha/Documents/GitHub/GeoDynamo.jl
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Pkg; Pkg.test()' \
  > /private/tmp/claude-501/-Users-subha-Documents-GitHub-GeoDynamo-jl/c06b3e0f-44d4-4e52-ac67-e6cc6df6233b/scratchpad/suite.log 2>&1
echo "exit: $?"
```

Never pipe this through `tail` — that masks the Julia exit code and reports `tail`'s 0.

- [ ] **Step 2: Confirm the totals**

```bash
grep -E "Test Summary|GeoDynamo.jl \|" /private/tmp/claude-501/-Users-subha-Documents-GitHub-GeoDynamo-jl/c06b3e0f-44d4-4e52-ac67-e6cc6df6233b/scratchpad/suite.log | tail -20
grep -cE "^Test Failed|Error During Test" /private/tmp/claude-501/-Users-subha-Documents-GitHub-GeoDynamo-jl/c06b3e0f-44d4-4e52-ac67-e6cc6df6233b/scratchpad/suite.log
```
Expected: exit 0, zero fails, zero errors. The pre-SP-0 baseline is roughly 9519 passes / 39 broken / 0 fails; the new parity testsets add to the pass count.

Three scalar-IC files are known-flaky (`temperature_ic_normalization.jl`, `composition_analytical_ic.jl`, `nusselt_and_analytical_ic.jl`). If only those fail, re-run before attributing it to SP-0.

- [ ] **Step 3: Re-run the cross-commit self-check on the final tree**

```bash
cd /Users/subha/Documents/GitHub/GeoDynamo.jl
GEODYNAMO_PARITY_FULL=1 \
  ~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. \
  scripts/parity_crosscommit.jl HEAD HEAD; echo "exit: $?"
```
Expected: exit 0, `PASS: 192 cases bit-identical`.

- [ ] **Step 4: Confirm no `src/` file was touched**

```bash
git diff --stat origin/main...HEAD -- src/
```
Expected: empty. Any output means SP-0 violated its scope constraint — stop and revise the spec rather than accepting it.

- [ ] **Step 5: Report and request permission to push**

Do not push or open a PR without explicit user permission. Report: suite totals from Step 2, cross-commit result from Step 3, the empty `src/` diff from Step 4, and the assertion diff from Task 5 Step 6.

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `test/parity/state_digest.jl` | 1 |
| `test/parity/fixtures.jl` | 2 |
| `test/parity/ab_harness.jl` | 3 |
| `test/parity/digest_negative_controls.jl` (5 controls) | 1 — implemented as 8 testsets, superset of the 5 |
| `scripts/parity_crosscommit.jl` | 4 |
| Rewrite both probe files | 5 |
| Register in `runtests.jl` | 1, 2, 3 (each registers its own) |
| Full suite green | 6 |
| `HEAD` vs `HEAD` bit-identical on full matrix | 4 Step 2, re-confirmed 6 Step 3 |
| Assertion diff verified by reading | 5 Step 6 |
| No `src/` changes | 6 Step 4 |
| Single-threaded gate | 4 (`--threads=1` in driver invocation) |
| 4-step trajectory | 2 (`evolve!` default) |
| Default + full matrices | 2 |
| Hash cannot produce a false green | 1 (final negative-control testset) |

Spec control #5 ("two independent builds of the same ref agree") maps to Task 4 Step 2 rather than to the negative-control file, since it needs two real builds. Noted rather than duplicated.

**Placeholder scan:** No TBD/TODO. Every code step carries runnable code. Every run step names an exact command and an expected result.

**Type consistency:** `FieldBits(name, dims, values)` and `StateDigest(env, info, fields, hash)` are constructed identically in Task 1's tests and implementation. `ParityCase(timestepper_name, timestepper, scalar_code, wall_code, magnetic, composition)` has the same 6-positional form in Tasks 2, 3, and 5. `digests_equal` returns `(Bool, String)` at every call site. `ParityFixtures.evolve!` takes `nsteps` as a keyword everywhere.

**Known risk carried into execution:** Task 5's shim recovers the BC code by identity comparison (`===`) against `ParityFixtures.SCALAR_BCS`/`WALL_BCS`. If `BoundaryConditions` values are constructed fresh at each call site rather than shared, `findfirst` returns `nothing` and the shim falls back to code 1, silently narrowing coverage. Task 5 Step 3 will catch this as a changed testset name or an unexpected pass/fail pattern; if `findfirst` misses, switch the shim to take the code directly from the caller's loop variable instead of recovering it.
