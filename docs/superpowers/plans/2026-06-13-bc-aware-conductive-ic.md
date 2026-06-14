# BC-aware Conductive IC with Radial Source/Sink — Implementation Plan

> **STATUS: COMPLETE (2026-06-14).** Tasks 1–7 implemented + committed (`b04b252`→`0d6c025`); test registered (`runtests.jl:236`). Full suite green: **8748 pass / 45 broken / 8793 total, EXIT=0** (broken baseline unchanged). Follow-up `f1ad967` fixes a byproduct: `nothing`-valued `internal_heating`/`compositional_source` now round-trip through param files (6 spurious parse warnings removed). Local `main` only (NOT pushed).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two inconsistent conductive-profile implementations with one BC-aware, source-aware discrete solve, used by both the solver default and `AnalyticIC(:conductive)`, for temperature and composition.

**Architecture:** Build the steady l=0 radial operator by REUSING the solver's own pieces — `create_radial_laplacian(domain)` for `∇²₀`, and `_zero_scalar_boundary_rows!` + `_apply_scalar_boundary_rows!` (bcs/scalar_bc.jl) for the Dirichlet/Neumann/ball-regularity boundary rows — then solve `∇²₀ c = −S` for the (0,0) coefficient profile `c(r)` (already √(4π)-scaled, matching `boundary_values`). Because the operator and BC rows are identical to the implicit timestep matrices, the IC is a discrete equilibrium of the actual solver: one step from rest leaves the field unchanged.

**Tech Stack:** Julia, banded radial operators (`src/numerics/banded_operators.jl`), scalar BC rows (`src/bcs/scalar_bc.jl`), SHTnsKit spectral storage.

**Source convention:** `S` is the RHS of `∇²T = −S` (diffusion-normalized units). Uniform `S` ⇒ `T = a + b/r − S·r²/6`. Ball default `S=6` ⇒ `1 − r²` (with outer `T=0`, `r_o=1`).

---

## File Structure

- **Modify** `src/core/parameters.jl` — add `internal_heating`, `compositional_source` fields; register in kwarg/param-key lists.
- **Modify** `src/api/model.jl` — forward the two new kwargs into `SolverParameters`.
- **Modify** `src/physics/scalar_field_solver_common.jl` — add `conductive_profile_solve` + helpers `_scalar_bc_code_l0`, `_resolve_source` (shared by temperature & composition; already included after `bcs/` and `numerics/`).
- **Modify** `src/physics/temperature/solver.jl` — `initialize_temperature_field!` l=0 branch calls `conductive_profile_solve`; keep `_shell_/_ball_conductive_temperature` as analytic test references.
- **Modify** `src/physics/composition/solver.jl` — `initialize_composition_field!` l=0 branch calls `conductive_profile_solve` (replaces linear interp).
- **Modify** `src/core/initial_conditions.jl` — `set_analytical_temperature!(:conductive)` + new `set_analytical_composition!(:conductive)` call `conductive_profile_solve`; thread `geometry` + `source` from the `AnalyticIC` dispatch.
- **Modify** `src/api/initial_conditions.jl` — `_apply_initial_condition!(::AnalyticIC)` passes `geometry` and resolved `source` to `set_analytical_initial_conditions!`.
- **Create** `test/conductive_profile.jl` — unit + equilibrium + backward-compat tests; register in `test/runtests.jl`.

---

## Task 1: New SolverParameters fields + propagation

**Files:**
- Modify: `src/core/parameters.jl` (struct ~line 16, kwarg list ~105, param-key list ~133)
- Modify: `src/api/model.jl` (kwargs → `SolverParameters(...)` call ~line 60-100)
- Test: `test/conductive_profile.jl`

- [ ] **Step 1: Write the failing test**

Create `test/conductive_profile.jl` with:

```julia
using Test
using GeoDynamo
const G = GeoDynamo

@testset "conductive IC: source params" begin
    p0 = G.SolverParameters(nr = 16, lmax = 4)
    @test p0.internal_heating === nothing
    @test p0.compositional_source === nothing
    p1 = G.SolverParameters(nr = 16, lmax = 4, internal_heating = 3.0)
    @test p1.internal_heating == 3.0
    p2 = G.SolverParameters(nr = 16, lmax = 4, internal_heating = (r -> 2r))
    @test p2.internal_heating isa Function
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `JL=~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia; $JL --project=. test/conductive_profile.jl`
Expected: FAIL — `type SolverParameters has no field internal_heating`.

- [ ] **Step 3: Add the fields**

In `src/core/parameters.jl`, inside `Base.@kwdef struct SolverParameters` (near `radius_ratio::Float64 = 0.35`), add:

```julia
    internal_heating::Union{Nothing, Float64, Function} = nothing
    compositional_source::Union{Nothing, Float64, Function} = nothing
```

If `load_parameters`/`from_file` filters keys via an allow-list (the `(:nlon, :radial_bandwidth, :radius_ratio, :r_outer)`-style tuples around lines 105 and 133), add `:internal_heating, :compositional_source` to BOTH tuples so file/dict construction does not drop them. (Functions cannot come from a config file; only the `Float64`/`Nothing` forms will arrive via that path — that is fine.)

- [ ] **Step 4: Forward the kwargs in the model constructor**

In `src/api/model.jl`, find the `SolverParameters(...)` construction inside `GeodynamoModel(grid::SphericalShellGrid; ...)` and `GeodynamoModel(grid::SphericalBallGrid; ...)`. Add `internal_heating` and `compositional_source` to BOTH the function keyword signature (default `nothing`) and the forwarded `SolverParameters(...; ... )` call:

```julia
function GeodynamoModel(grid::SphericalShellGrid;
        # ... existing kwargs ...
        internal_heating = nothing,
        compositional_source = nothing,
        # ...
    )
    # ...
    params = SolverParameters(;
        # ... existing ...
        internal_heating = internal_heating,
        compositional_source = compositional_source,
    )
```

Repeat for the `SphericalBallGrid` method.

- [ ] **Step 5: Run test, verify it passes**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: PASS (the `source params` testset).

- [ ] **Step 6: Commit**

```bash
git add src/core/parameters.jl src/api/model.jl test/conductive_profile.jl
git commit -m "feat(ic): add internal_heating / compositional_source params"
```

---

## Task 2: BC-code + source helpers

**Files:**
- Modify: `src/physics/scalar_field_solver_common.jl` (append helpers)
- Test: `test/conductive_profile.jl`

- [ ] **Step 1: Write the failing test**

Append to `test/conductive_profile.jl`:

```julia
@testset "bc-code mapping + source resolution" begin
    # DIRICHLET/NEUMANN are exported BoundaryType enum values (bcs/common.jl)
    DI = Int(GeoDynamo.DIRICHLET); NE = Int(GeoDynamo.NEUMANN)
    @test G._scalar_bc_code_from_types(DI, DI) == 1   # DD
    @test G._scalar_bc_code_from_types(DI, NE) == 2   # DN
    @test G._scalar_bc_code_from_types(NE, DI) == 3   # ND
    @test G._scalar_bc_code_from_types(NE, NE) == 4   # NN

    dom = G.create_radial_domain(8)
    r = [dom.r[k, 4] for k in 1:dom.N]
    @test G._resolve_source(nothing, dom, 0.0) == zeros(dom.N)      # default
    @test G._resolve_source(2.0, dom, 0.0) == fill(2.0, dom.N)      # uniform
    @test G._resolve_source(x -> x, dom, 0.0) ≈ r                   # function
    @test G._resolve_source(nothing, dom, 6.0) == fill(6.0, dom.N)  # geometry default
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: FAIL — `UndefVarError: _scalar_bc_code_from_types`.

- [ ] **Step 3: Implement the helpers**

Append to `src/physics/scalar_field_solver_common.jl`:

```julia
# Map per-boundary DIRICHLET/NEUMANN ints to the scalar_bc_code used by
# _apply_scalar_boundary_rows! (1=DD, 2=DN, 3=ND, 4=NN).
@inline function _scalar_bc_code_from_types(inner_type::Int, outer_type::Int)
    di = Int(DIRICHLET)
    inner_d = inner_type == di
    outer_d = outer_type == di
    return inner_d ? (outer_d ? 1 : 2) : (outer_d ? 3 : 4)
end

# Resolve a source spec to a per-radial-node vector S(r). `default` is the
# geometry-aware fallback used when `source === nothing`.
function _resolve_source(source, domain, default::Real)
    nr = domain.N
    if source === nothing
        return fill(Float64(default), nr)
    elseif source isa Function
        return Float64[source(domain.r[k, 4]) for k in 1:nr]
    else
        return fill(Float64(source), nr)
    end
end
```

- [ ] **Step 4: Run test, verify it passes**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/physics/scalar_field_solver_common.jl test/conductive_profile.jl
git commit -m "feat(ic): bc-code mapping + source resolution helpers"
```

---

## Task 3: Core `conductive_profile_solve`

**Files:**
- Modify: `src/physics/scalar_field_solver_common.jl` (append)
- Test: `test/conductive_profile.jl`

Reuses: `create_radial_laplacian(domain)`, `create_derivative_matrix(T,1,domain)`,
`radial_bandwidth(domain)`, `_zero_scalar_boundary_rows!`, `_apply_scalar_boundary_rows!`,
`BandedMatrix`, `factorize_banded`, `solve_banded!` (all already in scope via numerics/bcs).

- [ ] **Step 1: Write the failing test (shell Dirichlet, S=0 → a+b/r)**

Append to `test/conductive_profile.jl`:

```julia
@testset "conductive_profile_solve" begin
    DI = Int(GeoDynamo.DIRICHLET); NE = Int(GeoDynamo.NEUMANN)
    # conductive_profile_solve is linear and unit-agnostic: this UNIT test uses
    # raw values (no √(4π)); callers pre-scale boundary values + source by √(4π).
    nr = 24
    dom = G.create_radial_domain(nr)              # whatever ri/ro the ctor yields
    r = [dom.r[k, 4] for k in 1:nr]; ri = r[1]; ro = r[end]

    # Shell, Dirichlet inner=1 outer=0, S=0 → a + b/r with T(ri)=1, T(ro)=0.
    c = G.conductive_profile_solve(; domain = dom,
        bc_code = G._scalar_bc_code_from_types(DI, DI),
        inner_value = 1.0, outer_value = 0.0,
        source = zeros(nr), inner_regularity = false)
    b = 1.0 / (1/ri - 1/ro); a = -b / ro
    @test c ≈ (a .+ b ./ r) atol = 1e-6 rtol = 1e-6

    # Shell + uniform S=4, Dirichlet 0/0 → T = a + b/r − S r²/6.
    S = 4.0
    c2 = G.conductive_profile_solve(; domain = dom,
        bc_code = G._scalar_bc_code_from_types(DI, DI),
        inner_value = 0.0, outer_value = 0.0, source = fill(S, nr),
        inner_regularity = false)
    part(rr) = -S*rr^2/6
    bb = (part(ro) - part(ri)) / (1/ri - 1/ro); aa = -part(ro) - bb/ro
    @test c2 ≈ (aa .+ bb ./ r .+ part.(r)) atol = 1e-5 rtol = 1e-5
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: FAIL — `UndefVarError: conductive_profile_solve`.

- [ ] **Step 3: Implement**

Append to `src/physics/scalar_field_solver_common.jl`:

```julia
"""
    conductive_profile_solve(; domain, bc_code, inner_value, outer_value,
                             source, inner_regularity) -> Vector{Float64}

Steady l=0 (0,0)-coefficient profile c(r) solving ∇²₀ c = −source with the
boundary rows that the implicit scalar matrices use (same operator ⇒ the IC is
a discrete equilibrium). `inner_value`/`outer_value` are the √(4π)-scaled
boundary coefficients (the values stored in `field.boundary_values`); for a
Neumann boundary they are the prescribed flux coefficient. `source` is a length-N
vector (RHS of ∇²T = −S, √(4π)-scaled is NOT needed: the operator is linear and
the boundary values already carry √(4π), so pass the physical source × √(4π) to
keep the coefficient profile consistent — see callers). `inner_regularity=true`
selects the ball centre row (Θ′(r₁)=0 for l=0).
"""
function conductive_profile_solve(; domain, bc_code::Int,
        inner_value::Real, outer_value::Real,
        source::AbstractVector, inner_regularity::Bool = false)
    T = Float64
    N = domain.N
    bw = radial_bandwidth(domain)
    lap = create_radial_laplacian(domain)            # ∇²₀ banded (l=0: no l(l+1) term)
    d1 = create_derivative_matrix(T, 1, domain)
    sys = T.(copy(lap.data))
    _zero_scalar_boundary_rows!(sys, bw, N)
    _apply_scalar_boundary_rows!(sys, d1.data, bc_code, 0, bw, N,
        inner_regularity, domain.r[1, 3])
    A = BandedMatrix{T}(sys, bw, N)
    lu = factorize_banded(A)
    rhs = Vector{T}(undef, N)
    @inbounds for i in 1:N
        rhs[i] = -T(source[i])      # interior: ∇²c = −S
    end
    rhs[1] = T(inner_value)         # boundary rows overwrite endpoints
    rhs[N] = T(outer_value)
    solve_banded!(rhs, lu)          # in-place solve; rhs ← c(r)
    return rhs
end
```

NOTE: verify `solve_banded!(x, lu)` signature/in-place semantics against
`src/numerics/banded_operators.jl` (or `src/solver/numerics.jl`). If the in-place
solver is `solve_banded!(x, b, lu)` or returns a new vector, adapt this call and
return value accordingly. The test in Step 1 is the guard.

- [ ] **Step 4: Run test, verify it passes**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: PASS (both `conductive_profile_solve` cases).

- [ ] **Step 5: Add mixed-BC + ball tests**

Append:

```julia
@testset "conductive_profile_solve mixed + ball" begin
    DI = Int(GeoDynamo.DIRICHLET); NE = Int(GeoDynamo.NEUMANN)
    nr = 24
    dom = G.create_radial_domain(nr)
    r = [dom.r[k, 4] for k in 1:nr]
    # Mixed: Dirichlet inner=1, Neumann outer flux=0; interior residual ∇²c+S ≈ 0.
    c = G.conductive_profile_solve(; domain = dom,
        bc_code = G._scalar_bc_code_from_types(DI, NE),
        inner_value = 1.0, outer_value = 0.0, source = fill(2.0, nr),
        inner_regularity = false)
    lap = G.create_radial_laplacian(dom)
    res = (Matrix(G.BandedMatrix{Float64}(copy(lap.data), G.radial_bandwidth(dom), nr)) * c) .+ 2.0
    @test maximum(abs, res[3:nr-2]) < 1e-6

    # Ball: regularity inner, outer Dirichlet 0, S=6 → ≈ 1 - r² when ro≈1.
    domb = G.create_radial_domain(nr)            # ball domain via the ctor/default
    rb = [domb.r[k, 4] for k in 1:nr]
    cb = G.conductive_profile_solve(; domain = domb,
        bc_code = G._scalar_bc_code_from_types(NE, DI),  # inner replaced by regularity row
        inner_value = 0.0, outer_value = 0.0, source = fill(6.0, nr),
        inner_regularity = true)
    # compare to the analytic ball solution for this domain's actual ro
    rob = rb[end]
    @test cb ≈ (6.0 .* (rob^2 .- rb .^ 2) ./ 6) atol = 1e-3 rtol = 1e-3
end
```

NOTE: `apply_banded` is the banded mat-vec; confirm its name (`apply_banded`/
`banded_matvec`) in `src/numerics/banded_operators.jl` and adjust. If absent,
compute the residual with a small dense `Matrix(lap) * c .+ S`.

- [ ] **Step 6: Run, verify pass; Commit**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: PASS.

```bash
git add src/physics/scalar_field_solver_common.jl test/conductive_profile.jl
git commit -m "feat(ic): discrete BC-aware conductive_profile_solve"
```

---

## Task 4: Wire into the solver default (temperature)

**Files:**
- Modify: `src/physics/temperature/solver.jl` (`initialize_temperature_field!`, l=0 branch ~lines 30-40)
- Test: `test/conductive_profile.jl`

- [ ] **Step 1: Write the failing equilibrium test**

Append:

```julia
@testset "temperature conductive IC == discrete equilibrium" begin
    params = G.SolverParameters(architecture = :cpu, geometry = :shell,
        nr = 16, nr_inner = 4, lmax = 4, mmax = 4, nlat = 12, nlon = 24,
        Ra = 0.0, Ek = 1e-2, Pr = 1.0, timestep = 1e-3,
        include_magnetic = false, include_composition = false,
        internal_heating = 3.0)
    st = G.initialize_simulation(Float64, params)
    G.initialize_solver_fields!(st)               # consume one-shot init
    tmp = st.fields.temperature
    before = copy(parent(tmp.spectral.data_real))
    G.solver_step!(st)                            # zero flow, Ra=0 ⇒ pure diffusion
    after = parent(tmp.spectral.data_real)
    rel = maximum(abs, after .- before) / max(maximum(abs, before), eps())
    @test rel < 1e-6                              # IC is a discrete steady state
end
```

- [ ] **Step 2: Run, verify it fails**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: FAIL — `rel` is not < 1e-6 (current closed-form IC ignores `internal_heating`, so it is NOT the steady state of the operator with that source).

- [ ] **Step 3: Rewrite the l=0 branch of `initialize_temperature_field!`**

In `src/physics/temperature/solver.jl`, replace the per-mode l=0 assignment with a precomputed `conductive_profile_solve` vector. Before the `@inbounds for lm_idx` loop, add:

```julia
    m00 = get_mode_index(temperature.config, 0, 0)
    in_t = m00 > 0 ? temperature.bc_type_inner[m00] : Int(DIRICHLET)
    out_t = m00 > 0 ? temperature.bc_type_outer[m00] : Int(DIRICHLET)
    in_v = m00 > 0 ? temperature.boundary_values[1, m00] : zero(T)
    out_v = m00 > 0 ? temperature.boundary_values[2, m00] : zero(T)
    default_S = state.parameters.geometry === :ball ? 6.0 : 0.0
    Svec = _resolve_source(state.parameters.internal_heating, domain, default_S) .*
           sqrt(4 * Float64(π))     # coefficient-space RHS
    cond_c = conductive_profile_solve(; domain = domain,
        bc_code = _scalar_bc_code_from_types(in_t, out_t),
        inner_value = in_v, outer_value = out_v, source = Svec,
        inner_regularity = state.parameters.geometry === :ball)
```

Then in the loop, the `l == 0 && m == 0` branch becomes:

```julia
            if l == 0 && m == 0
                set_local_spectral_value!(spec_real, slot, r_idx, T(cond_c[r_idx]))
```

(Keep the rest of the loop — non-zero modes — unchanged. `r_idx` here is the global radial index used to index `cond_c`; if the spec pencil is r-distributed, index `cond_c[r_idx]` by the global radial index — confirm `r_range`/`r_idx` are global. For Phase-1 main, r is local==global.)

- [ ] **Step 4: Run, verify pass**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: PASS (equilibrium `rel < 1e-6`).

- [ ] **Step 5: Backward-compat test (default shell/ball)**

Append:

```julia
@testset "temperature default conductive backward-compat" begin
    s4pi = sqrt(4π)
    for (geom, rr) in ((:shell, 0.35), (:ball, 0.0))
        params = G.SolverParameters(architecture = :cpu, geometry = geom,
            nr = 24, nr_inner = 4, lmax = 2, mmax = 2, nlat = 8, nlon = 16,
            radius_ratio = rr, Ra = 0.0, Ek = 1e-2, Pr = 1.0, timestep = 1e-3,
            include_magnetic = false, include_composition = false)
        st = G.initialize_simulation(Float64, params); G.initialize_solver_fields!(st)
        tmp = st.fields.temperature; dom = st.backend.outer_core_domain
        m00 = G.get_mode_index(tmp.config, 0, 0)
        slot = G.local_spectral_storage_slot(tmp.config, m00)
        ref = geom === :ball ? G._ball_conductive_temperature :
                               G._shell_conductive_temperature
        for rk in 1:dom.N
            got = G.local_spectral_value(parent(tmp.spectral.data_real), slot, rk) / s4pi
            @test isapprox(got, ref(params, dom.r[rk, 4]); atol = 1e-3, rtol = 1e-3)
        end
    end
end
```

- [ ] **Step 6: Run, verify pass; Commit**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: PASS.

```bash
git add src/physics/temperature/solver.jl test/conductive_profile.jl
git commit -m "feat(ic): BC+source-aware default temperature conductive IC"
```

---

## Task 5: Wire into the solver default (composition)

**Files:**
- Modify: `src/physics/composition/solver.jl` (`initialize_composition_field!`, l=0 branch)
- Test: `test/conductive_profile.jl`

- [ ] **Step 1: Write the failing test**

Append:

```julia
@testset "composition conductive IC == discrete equilibrium" begin
    params = G.SolverParameters(architecture = :cpu, geometry = :shell,
        nr = 16, nr_inner = 4, lmax = 4, mmax = 4, nlat = 12, nlon = 24,
        Ra = 0.0, Ek = 1e-2, Pr = 1.0, Sc = 1.0, timestep = 1e-3,
        include_magnetic = false, include_composition = true,
        compositional_source = 2.0)
    st = G.initialize_simulation(Float64, params); G.initialize_solver_fields!(st)
    comp = st.fields.composition
    before = copy(parent(comp.spectral.data_real))
    G.solver_step!(st)
    after = parent(comp.spectral.data_real)
    rel = maximum(abs, after .- before) / max(maximum(abs, before), eps())
    @test rel < 1e-6
end
```

- [ ] **Step 2: Run, verify it fails**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: FAIL (current linear-interp IC is not the steady state with a source).

- [ ] **Step 3: Rewrite the l=0 branch of `initialize_composition_field!`**

Mirror Task 4, using `composition`, `state.parameters.compositional_source`, and
`default_S = 0.0` for BOTH geometries (no default compositional volumetric source):

```julia
    m00 = get_mode_index(composition.config, 0, 0)
    in_t = m00 > 0 ? composition.bc_type_inner[m00] : Int(DIRICHLET)
    out_t = m00 > 0 ? composition.bc_type_outer[m00] : Int(DIRICHLET)
    in_v = m00 > 0 ? composition.boundary_values[1, m00] : zero(T)
    out_v = m00 > 0 ? composition.boundary_values[2, m00] : zero(T)
    Svec = _resolve_source(state.parameters.compositional_source, domain, 0.0) .*
           sqrt(4 * Float64(π))
    cond_c = conductive_profile_solve(; domain = domain,
        bc_code = _scalar_bc_code_from_types(in_t, out_t),
        inner_value = in_v, outer_value = out_v, source = Svec,
        inner_regularity = state.parameters.geometry === :ball)
```

Replace the `l == 0 && m == 0` linear-interp assignment with
`set_local_spectral_value!(spec_real, slot, r_idx, T(cond_c[r_idx]))`. Leave the
`1 <= l <= 3` seed-perturbation branch unchanged.

- [ ] **Step 4: Run, verify pass**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/physics/composition/solver.jl test/conductive_profile.jl
git commit -m "feat(ic): BC+source-aware default composition conductive IC"
```

---

## Task 6: Wire into `AnalyticIC(:conductive)`

**Files:**
- Modify: `src/core/initial_conditions.jl` (`set_analytical_temperature!`, add `set_analytical_composition!(:conductive)`)
- Modify: `src/api/initial_conditions.jl` (`_apply_initial_condition!(::AnalyticIC)` threads geometry + source)
- Test: `test/conductive_profile.jl`

- [ ] **Step 1: Write the failing test**

Append:

```julia
@testset "AnalyticIC(:conductive) is BC-aware" begin
    s4pi = sqrt(4π)
    grid = G.SphericalShellGrid(G.CPU(); lmax = 2, mmax = 2, nlat = 8, nlon = 16,
        nr = 24, nr_inner = 4)
    model = G.GeodynamoModel(grid; Ek = 1e-2, Ra = 0.0,
        include_magnetic = false, include_composition = false,
        internal_heating = 4.0)
    G.set!(model; temperature = G.AnalyticIC(:conductive))
    tmp = model.state.fields.temperature; dom = model.state.backend.outer_core_domain
    m00 = G.get_mode_index(tmp.config, 0, 0)
    slot = G.local_spectral_storage_slot(tmp.config, m00)
    r = [dom.r[k, 4] for k in 1:dom.N]; ri = r[1]; ro = r[end]; S = 4.0
    part(rr) = -S*rr^2/6
    bb = (part(ro)-part(ri))/(1/ri-1/ro); aa = -part(ro)-bb/ro    # T=1 inner,0 outer
    bb += 1.0/(1/ri-1/ro); aa += -(1.0/(1/ri-1/ro))/ro
    for k in 1:dom.N
        got = G.local_spectral_value(parent(tmp.spectral.data_real), slot, k)/s4pi
        @test isapprox(got, aa + bb/r[k] + part(r[k]); atol = 1e-4, rtol = 1e-4)
    end
end
```

- [ ] **Step 2: Run, verify it fails**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: FAIL (current `set_analytical_temperature!(:conductive)` is the linear hardcode, ignores BCs + source).

- [ ] **Step 3: Thread geometry + source through the AnalyticIC dispatch**

In `src/api/initial_conditions.jl`, change `_apply_initial_condition!(model, field, ic::AnalyticIC)` to pass geometry and the resolved source via the parameters splat:

```julia
function _apply_initial_condition!(model::GeodynamoModel, field::Symbol, ic::AnalyticIC)
    f = _get_field(model, field)
    p = model.state.parameters
    src = get(ic.parameters, :source,
        field === :composition ? p.compositional_source : p.internal_heating)
    InitialConditions.set_analytical_initial_conditions!(f, field, ic.pattern;
        amplitude = ic.amplitude, geometry = p.geometry, source = src,
        ic.parameters...)
    return model
end
```

In `src/core/initial_conditions.jl`, extend `set_analytical_initial_conditions!` and
`set_analytical_temperature!`/`set_analytical_composition!` to accept
`geometry::Symbol = :shell` and `source = nothing` kwargs (ignore for patterns that
don't use them — splat-safe). Add a `:conductive` branch to BOTH that calls the solve:

```julia
    elseif pattern == :conductive
        m00 = get_mode_index(spectral.config, 0, 0)
        in_t = m00 > 0 ? field_bc_type_inner(temp_field)[m00] : Int(DIRICHLET)
        out_t = m00 > 0 ? field_bc_type_outer(temp_field)[m00] : Int(DIRICHLET)
        in_v = m00 > 0 ? boundary_values_of(temp_field)[1, m00] : zero(T)
        out_v = m00 > 0 ? boundary_values_of(temp_field)[2, m00] : zero(T)
        default_S = geometry === :ball ? (is_temperature ? 6.0 : 0.0) : 0.0
        Svec = GEODYNAMO_PARENT._resolve_source(source, domain_of(spectral), default_S) .*
               sqrt(4 * T(π))
        cond_c = GEODYNAMO_PARENT.conductive_profile_solve(; domain = domain_of(spectral),
            bc_code = GEODYNAMO_PARENT._scalar_bc_code_from_types(in_t, out_t),
            inner_value = T(amplitude) * in_v, outer_value = T(amplitude) * out_v,
            source = Svec, inner_regularity = geometry === :ball)
        r_range = get_local_range(spectral.pencil, 3)
        slot0 = local_spectral_storage_slot(spectral.config, m00)
        if slot0 !== nothing
            for (lr, gr) in enumerate(r_range)
                lr <= size(real_data, 3) && set_local_spectral_value!(real_data, slot0, lr, T(cond_c[gr]))
            end
        end
```

NOTE: `set_analytical_*` operates on a field with `.spectral`/`.bc_type_inner`/
`.boundary_values`/domain access. Confirm how to reach the radial domain from inside
`set_analytical_temperature!` (the InitialConditions module). If the field does not
carry the domain, add a `domain` kwarg threaded from `_apply_initial_condition!`
(`domain = model.state.backend.outer_core_domain`) — preferred, avoids new accessors.
Replace `field_bc_type_inner`/`boundary_values_of`/`domain_of` above with the field's
actual property accesses (`temp_field.bc_type_inner`, `temp_field.boundary_values`,
and the threaded `domain`). `amplitude` scales the Dirichlet boundary targets so
`amplitude` keeps its "overall magnitude" meaning for the default 1/0 BC.

- [ ] **Step 4: Run, verify pass**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: PASS.

- [ ] **Step 5: Add a composition `:conductive` smoke test + commit**

Append a composition variant of the Step-1 test (build a model with
`include_composition=true, compositional_source=2.0`, `set!(model; composition=AnalyticIC(:conductive))`,
check the (0,0) profile matches `a+b/r−Sr²/6` with its BCs). Run; expect PASS.

```bash
git add src/core/initial_conditions.jl src/api/initial_conditions.jl test/conductive_profile.jl
git commit -m "feat(ic): BC+source-aware AnalyticIC(:conductive) for temp + composition"
```

---

## Task 7: Register test + full suite

**Files:**
- Modify: `test/runtests.jl`

- [ ] **Step 1: Register**

In `test/runtests.jl`, add to the `additional_tests` tuple right after
`"initial_condition_application.jl",`:

```julia
    "conductive_profile.jl",
```

- [ ] **Step 2: Run the focused file once more**

Run: `$JL --project=. test/conductive_profile.jl`
Expected: PASS (all testsets).

- [ ] **Step 3: Full suite**

Run: `$JL --project=. -e 'using Pkg; Pkg.test()' > /tmp/suite_cond.log 2>&1; echo EXIT=$?`
Expected: EXIT=0; `Testing GeoDynamo tests passed`. No new failures; broken count unchanged (45). Pass count = prior + the new conductive testsets.

Check no regression in the pre-existing IC tests (`temperature_ic_normalization.jl`,
`composition_analytical_ic.jl`, `nusselt_and_analytical_ic.jl`) — these touch the
conductive/analytic paths. If a flake appears, re-run once (see memory: IC
normalization tests are state/ordering-sensitive).

- [ ] **Step 4: Commit**

```bash
git add test/runtests.jl
git commit -m "test(ic): wire conductive_profile.jl into the suite"
```

---

## Notes / risks (carried from spec)

- **Neumann sign + ball regularity:** inherited exactly by reusing
  `_apply_scalar_boundary_rows!` — the equilibrium tests (Tasks 4/5) are the guard.
- **Pure-Neumann gauge:** `_apply_scalar_boundary_rows!` already pins the inner l=0
  row to Dirichlet for `bc_code==4` (NN), so the constant is fixed automatically and
  the IC matches the solver. No separate handling needed; an optional compatibility
  `@warn` (∫S dV vs net flux) can be added later if desired (out of scope for green).
- **r-distributed spec pencil:** index `cond_c` by GLOBAL radial index. On current
  `main` (Phase-1) r is local==global; if run r-distributed, gather/scatter
  `cond_c` by global `r_idx` (note in Tasks 4/5).
- **`solve_banded!` / `apply_banded` exact names:** verify against
  `src/numerics/banded_operators.jl` (Task 3 notes) — tests are the guard.
```
