# GPU Phase 5l — Magnetic Conducting-Inner-Core Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the conducting-inner-core history-flux and reconstruction operations (`inner_core_history_flux`, `reconstruct_inner_core`) to GPU — batched over all spectral modes — bit-exact against the CPU references.

**Architecture:** A new file `src/gpu/inner_core.jl`. The two CPU functions each do, per harmonic degree `l`: a banded matvec (`L·S_old`, `L=η∇²_l`), a CNAB2-history assembly (`b = inv_dt·S_old + weight·L·S_old`), a boundary-row override, a length-`Nic` banded solve, and (for the flux) a `d1_top·y` reduction. All but the reduction REUSE the already-verified batched kernels — `gpu_batched_banded_matvec_perl!` (5c, reproduces `apply_∂r!` exactly), `gpu_implicit_solve_field!` (5d, BC rows + `solve_banded!`) — operating on a length-`Nic` radial axis. The only NEW kernel is a per-mode weighted radial reduction. A host packer flattens the per-degree `InnerCoreAdmittance` operators into batched arrays (degree `l` → slot `l+1`; **identity** LU + **zero** `L` for non-stored degrees so the batched solve over empty modes is a safe no-op, never a divide-by-zero). Runs on Array (locally testable) and CuArray.

**Tech Stack:** Julia, KernelAbstractions (via the reused kernels + one new reduction), `src/gpu/*` kernels from Phases 4/5c/5d.

---

## Background — the CPU reference (read, do not modify)

`src/physics/magnetic/inner_core.jl`:

```julia
struct InnerCoreAdmittance{T}
    factor::Vector{BandedLU{T}}    # M_ic = (1/dt)I − θ·η∇²_l, LU per stored l
    alpha::Vector{T}               # ICB admittance per stored l (NOT used by the two ported fns)
    d1_top::Vector{T}              # one-sided ∂/∂r row at r=ri, length Nic
    lookup::Dict{Int, Int}         # degree l → index in factor/lin
    Nic::Int
    lin::Vector{BandedMatrix{T}}   # L = η∇²_l per stored l (full band)
    dt::T
    theta::T
end

# b = (1/dt)·S_old + (1−θ)·L·S_old   (L applied via apply_∂r!, full band)
_ic_build_bic(a, l, S_old)            # inner_core.jl:149-157

# φ0 = d1_top · y, where M_ic y = b with b[1]=b[Nic]=0
inner_core_history_flux(a, l, S_old)  # inner_core.jl:168-175

# S = solution of M_ic S = b with b[1]=0, b[Nic]=g
reconstruct_inner_core(a, l, g, S_old) # inner_core.jl:185-192
```

Key facts (verified):
- `apply_∂r!` (`src/numerics/banded_operators.jl:246`) accumulates `out[i] += data[bw+1+i-j, j]·v[j]` in **ascending-j** order — byte-identical to `apply_banded_full!`, which `gpu_batched_banded_matvec_perl!` (5c) already reproduces exactly. So the GPU matvec matches `_ic_build_bic`'s `L·S_old` exactly.
- `solve_banded!` (`banded_operators.jl:84`) is exactly what `gpu_implicit_solve_field!`/`gpu_batched_banded_solve!` (5d/4) reproduce.
- `BandedLU{T}` has fields `.lu::Matrix{T}` `(2bw+1, Nic)`, `.bandwidth`, `.size`. `BandedMatrix{T}` has `.data::Matrix{T}` `(2bw+1, Nic)`, `.bandwidth`, `.size`.
- `dot(d1_top, y)` (real) sums ascending-i → a simple `for i in 1:Nic` reduce matches exactly.
- `b = (one(T)/a.dt).*S_old .+ (one(T)-a.theta).*Lx` → `inv_dt = 1/dt`, `weight = 1−θ`.

**GPU layout:** dense `(nl, nm, Nic)` inner-core spectral field; dim-1 slot `li` ↔ degree `l = li-1`. Per-mode outputs (`φ0`, `g`) are `(nl, nm)`. Batched operators are `(2bw+1, Nic, nl)` (dim-3 = degree slot).

---

## Task 1: Packer + reduction kernel + the two orchestrators

**Files:**
- Create: `src/gpu/inner_core.jl`
- Modify: `src/GeoDynamo.jl` (add `include("gpu/inner_core.jl")` after the `gpu/velocity_step.jl` include — both must come after `solver.jl`, where `InnerCoreAdmittance`/`BandedLU` are defined; add exports)
- Test: `test/gpu_phase5l_inner_core.jl`

- [ ] **Step 1: Write the failing test**

Create `test/gpu_phase5l_inner_core.jl`:

```julia
using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random
using LinearAlgebra: dot

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5l — Magnetic Conducting Inner Core (flux + reconstruct)" begin
    nl, nm, Nic = 6, 4, 7          # degrees 0..5, orders 0..3, inner-core points
    bw = 2
    dt = 5.0e-4; theta = 0.5
    inv_dt = 1.0 / dt; weight = 1.0 - theta
    rng = MersenneTwister(51)

    # synthetic per-degree operators for stored degrees 1..lmax (magnetic: no l=0)
    function banddata(N, b; seed, diagdom = false)
        r = MersenneTwister(seed); d = zeros(2b+1, N)
        for j in 1:N, i in max(1,j-b):min(N,j+b); d[b+1+i-j, j] = rand(r) - 0.5; end
        diagdom && (for j in 1:N; d[b+1, j] += 5.0; end)
        d
    end
    stored = collect(1:(nl-1))     # degrees 1..5
    facs = GeoDynamo.BandedLU{Float64}[]; lins = GeoDynamo.BandedMatrix{Float64}[]
    lookup = Dict{Int,Int}()
    for (idx, l) in enumerate(stored)
        M = GeoDynamo.BandedMatrix{Float64}(banddata(Nic, bw; seed = 100 + l, diagdom = true), bw, Nic)
        push!(facs, GeoDynamo.factorize_banded(M))
        push!(lins, GeoDynamo.BandedMatrix{Float64}(banddata(Nic, bw; seed = 200 + l), bw, Nic))
        lookup[l] = idx
    end
    d1_top = rand(rng, Nic) .- 0.5
    alphas = rand(rng, length(stored))     # unused by the two ported fns
    adm = GeoDynamo.InnerCoreAdmittance{Float64}(facs, alphas, d1_top, lookup, Nic, lins, dt, theta)

    # dense inner-core spectral state (real + imag), all (l,m) slots filled
    mk() = (a = zeros(nl, nm, Nic); for li in 1:nl, mi in 1:nm, r in 1:Nic; a[li,mi,r] = rand(rng) - 0.5; end; a)
    S_old_r = mk(); S_old_i = mk()
    g_r = rand(rng, nl, nm) .- 0.5; g_i = rand(rng, nl, nm) .- 0.5

    ic = GeoDynamo.gpu_pack_inner_core(adm, nl, CPU())

    @testset "packer bundle shape + identity-fill for non-stored degrees [LOCAL]" begin
        @test size(ic.lin_ic) == (2bw+1, Nic, nl)
        @test size(ic.lu_ic)  == (2bw+1, Nic, nl)
        @test length(ic.d1_top) == Nic
        @test ic.Nic == Nic && ic.bw == bw
        @test ic.inv_dt == inv_dt && ic.weight == weight
        # degree 0 (slot 1) is non-stored: zero L, identity LU (diag row == 1)
        @test all(==(0.0), ic.lin_ic[:, :, 1])
        @test all(==(1.0), ic.lu_ic[bw+1, :, 1])
        @test all(==(0.0), ic.lu_ic[1:bw, :, 1]) && all(==(0.0), ic.lu_ic[bw+2:2bw+1, :, 1])
        # stored degree 2 (slot 3) carries the operator data
        @test ic.lin_ic[:, :, 3] == lins[lookup[2]].data
        @test ic.lu_ic[:, :, 3]  == facs[lookup[2]].lu
    end

    @testset "history flux == CPU inner_core_history_flux (exact) [LOCAL]" begin
        φ0_r = zeros(nl, nm); φ0_i = zeros(nl, nm)
        GeoDynamo.gpu_inner_core_history_flux!(φ0_r, φ0_i, copy(S_old_r), copy(S_old_i), ic)
        for l in stored, mi in 1:nm
            li = l + 1
            ref_r = GeoDynamo.inner_core_history_flux(adm, l, S_old_r[li, mi, :])
            ref_i = GeoDynamo.inner_core_history_flux(adm, l, S_old_i[li, mi, :])
            @test φ0_r[li, mi] == ref_r
            @test φ0_i[li, mi] == ref_i
        end
        @test all(isfinite, φ0_r) && all(isfinite, φ0_i)
    end

    @testset "reconstruct == CPU reconstruct_inner_core (exact) [LOCAL]" begin
        S_new_r = similar(S_old_r); S_new_i = similar(S_old_i)
        GeoDynamo.gpu_reconstruct_inner_core!(S_new_r, S_new_i, copy(S_old_r), copy(S_old_i), g_r, g_i, ic)
        for l in stored, mi in 1:nm
            li = l + 1
            ref_r = GeoDynamo.reconstruct_inner_core(adm, l, g_r[li, mi], S_old_r[li, mi, :])
            ref_i = GeoDynamo.reconstruct_inner_core(adm, l, g_i[li, mi], S_old_i[li, mi, :])
            @test S_new_r[li, mi, :] == ref_r
            @test S_new_i[li, mi, :] == ref_i
        end
        @test all(isfinite, S_new_r) && all(isfinite, S_new_i)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5l gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            # CPU reference outputs
            cφ_r = zeros(nl, nm); cφ_i = zeros(nl, nm)
            GeoDynamo.gpu_inner_core_history_flux!(cφ_r, cφ_i, copy(S_old_r), copy(S_old_i), ic)
            cS_r = similar(S_old_r); cS_i = similar(S_old_i)
            GeoDynamo.gpu_reconstruct_inner_core!(cS_r, cS_i, copy(S_old_r), copy(S_old_i), g_r, g_i, ic)

            d(x) = GeoDynamo.on_architecture(GPU(), x)
            gic = GeoDynamo.gpu_pack_inner_core(adm, nl, GPU())
            gφ_r = d(zeros(nl, nm)); gφ_i = d(zeros(nl, nm))
            GeoDynamo.gpu_inner_core_history_flux!(gφ_r, gφ_i, d(copy(S_old_r)), d(copy(S_old_i)), gic)
            gS_r = d(similar(S_old_r)); gS_i = d(similar(S_old_i))
            GeoDynamo.gpu_reconstruct_inner_core!(gS_r, gS_i, d(copy(S_old_r)), d(copy(S_old_i)), d(g_r), d(g_i), gic)
            @test gφ_r isa CUDA.CuArray
            @test gS_r isa CUDA.CuArray
            @test isapprox(Array(gφ_r), cφ_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gφ_i), cφ_i; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gS_r), cS_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gS_i), cS_i; atol = 1e-9, rtol = 1e-8)
        end
    end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5l_inner_core.jl")'
```
Expected: FAIL — `UndefVarError: gpu_pack_inner_core` not defined.

- [ ] **Step 3: Write the implementation**

Create `src/gpu/inner_core.jl`:

```julia
# =============================================================================
# GPU Phase 5l — magnetic conducting-inner-core kernels: the CNAB2 history flux
# φ0 = d1_top·(M_ic⁻¹ b) and the inner-core profile reconstruction S = M_ic⁻¹ b,
# batched over all spectral modes. Mirrors inner_core_history_flux /
# reconstruct_inner_core (src/physics/magnetic/inner_core.jl:168-192). Each is
# a per-degree banded matvec (L·S_old, reusing 5c) + CNAB2 assembly + boundary
# rows + length-Nic banded solve (reusing 5d) [+ a d1_top·y reduction for the
# flux]. The per-degree InnerCoreAdmittance operators are packed into batched
# arrays; non-stored degrees (incl l=0) get a ZERO L and an IDENTITY LU so the
# batched pass over empty modes is a safe no-op (no divide-by-zero in the solve).
# Runs on Array + CuArray. (Per-call scratch — Phase-6 may cache.)
# =============================================================================

"""
    gpu_pack_inner_core(adm::InnerCoreAdmittance, nl, arch)
        -> (; lin_ic, lu_ic, d1_top, inv_dt, weight, Nic, bw)

Flatten the per-degree conducting-inner-core operators into batched arrays on
`arch`'s backend: `lin_ic` / `lu_ic` are `(2bw+1, Nic, nl)` indexed by dim-3 =
degree slot `li` (degree `l = li-1`).  Stored degrees carry `adm.lin[…].data` and
`adm.factor[…].lu`; non-stored degrees (including `l=0`) get a **zero** `L` and an
**identity** LU (diagonal row = 1) so the batched solve treats those empty modes as
`x = b` rather than dividing by a zero pivot.  Returns the operators plus the CNAB2
scalars `inv_dt = 1/dt`, `weight = 1−θ`, and `Nic`/`bw`.
"""
function gpu_pack_inner_core(adm::InnerCoreAdmittance{T}, nl::Int, arch::AbstractArchitecture) where {T}
    Nic = adm.Nic
    bw = isempty(adm.factor) ? 0 : adm.factor[1].bandwidth
    lin = zeros(T, 2bw + 1, Nic, nl)
    lu = zeros(T, 2bw + 1, Nic, nl)
    @inbounds for j in 1:Nic, li in 1:nl
        lu[bw + 1, j, li] = one(T)            # default = identity LU (overwritten for stored l)
    end
    for (l, idx) in adm.lookup
        slot = l + 1                          # degree l (0-based) → dim-3 slot
        (1 <= slot <= nl) || continue
        (adm.factor[idx].bandwidth == bw && adm.factor[idx].size == Nic) ||
            throw(ArgumentError("gpu_pack_inner_core: factor for l=$l has bw/size mismatch"))
        lin[:, :, slot] .= adm.lin[idx].data
        lu[:, :, slot]  .= adm.factor[idx].lu
    end
    return (; lin_ic = on_architecture(arch, lin), lu_ic = on_architecture(arch, lu),
              d1_top = on_architecture(arch, Vector{T}(adm.d1_top)),
              inv_dt = one(T) / adm.dt, weight = one(T) - adm.theta, Nic = Nic, bw = bw)
end

# One workitem per (l,m). φ0[li,mi] = Σ_i d1_top[i]·y[li,mi,i], ascending-i to
# match dot(d1_top, y) (inner_core.jl:174). y is the per-mode inner-core profile.
@kernel function _ic_flux_reduce_kernel!(φ0, @Const(y), @Const(d1_top), Nic::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(φ0)
    s = zero(T)
    @inbounds for i in 1:Nic
        s += d1_top[i] * y[li, mi, i]
    end
    @inbounds φ0[li, mi] = s
end

# CNAB2 history assembly b = inv_dt·S_old + weight·(L·S_old), written into `b_*`.
# Mirrors _ic_build_bic (inner_core.jl:149-157): same op order, same scalars.
function _gpu_ic_build_bic!(b_r, b_i, S_old_r, S_old_i, ic)
    Lx_r = similar(S_old_r); Lx_i = similar(S_old_i)     # Phase-6: workspace
    gpu_batched_banded_matvec_perl!(Lx_r, S_old_r, ic.lin_ic, ic.bw)
    gpu_batched_banded_matvec_perl!(Lx_i, S_old_i, ic.lin_ic, ic.bw)
    b_r .= ic.inv_dt .* S_old_r .+ ic.weight .* Lx_r
    b_i .= ic.inv_dt .* S_old_i .+ ic.weight .* Lx_i
    return nothing
end

"""
    gpu_inner_core_history_flux!(φ0_r, φ0_i, S_old_r, S_old_i, ic) -> nothing

Per-mode conducting-inner-core CNAB2 history flux `φ0 = d1_top·y`, where
`M_ic y = b` with `b = inv_dt·S_old + weight·L·S_old` and homogeneous boundary
rows (`b[1]=b[Nic]=0`).  `S_old_*` are dense `(nl,nm,Nic)` inner-core spectra;
`φ0_*` are `(nl,nm)`.  `ic` is the bundle from [`gpu_pack_inner_core`](@ref).
Mirrors `inner_core_history_flux` (inner_core.jl:168-175).  All arrays on the
same backend.
"""
function gpu_inner_core_history_flux!(φ0_r, φ0_i, S_old_r, S_old_i, ic)
    nl, nm, _ = size(S_old_r)
    y_r = similar(S_old_r); y_i = similar(S_old_i)       # Phase-6: workspace
    _gpu_ic_build_bic!(y_r, y_i, S_old_r, S_old_i, ic)
    z = similar(φ0_r, nl, nm); fill!(z, zero(eltype(φ0_r)))   # zero BC rows (inner=outer=0)
    gpu_implicit_solve_field!(y_r, y_i, ic.lu_ic, z, z, z, z, ic.bw)
    backend = KernelAbstractions.get_backend(φ0_r)
    _ic_flux_reduce_kernel!(backend)(φ0_r, y_r, ic.d1_top, ic.Nic; ndrange = (nl, nm))
    _ic_flux_reduce_kernel!(backend)(φ0_i, y_i, ic.d1_top, ic.Nic; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    gpu_reconstruct_inner_core!(S_new_r, S_new_i, S_old_r, S_old_i, g_r, g_i, ic) -> nothing

Per-mode conducting-inner-core reconstruction: solve `M_ic S = b` with
`b = inv_dt·S_old + weight·L·S_old`, regularity `b[1]=0`, and ICB Dirichlet
`b[Nic]=g` (the outer-core value at the ICB).  `S_old_*` dense `(nl,nm,Nic)`,
`g_*` `(nl,nm)`; the solution is written to `S_new_*`.  Mirrors
`reconstruct_inner_core` (inner_core.jl:185-192).  `S_new_*` may not alias
`S_old_*`.  All arrays on the same backend.
"""
function gpu_reconstruct_inner_core!(S_new_r, S_new_i, S_old_r, S_old_i, g_r, g_i, ic)
    nl, nm, _ = size(S_old_r)
    _gpu_ic_build_bic!(S_new_r, S_new_i, S_old_r, S_old_i, ic)   # b into S_new
    z = similar(g_r); fill!(z, zero(eltype(g_r)))                # inner BC = 0
    gpu_implicit_solve_field!(S_new_r, S_new_i, ic.lu_ic, z, z, g_r, g_i, ic.bw)  # outer BC = g
    return nothing
end
```

Modify `src/GeoDynamo.jl` — add the include immediately after `include("gpu/velocity_step.jl")`:

```julia
include("gpu/inner_core.jl")
```

And add the exports near the other `gpu_*` exports (after `export gpu_velocity_field_step!`):

```julia
export gpu_pack_inner_core, gpu_inner_core_history_flux!, gpu_reconstruct_inner_core!
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5l_inner_core.jl")'
```
Expected: the three `[LOCAL]` testsets PASS (packer shape/identity-fill, history flux exact `==`, reconstruct exact `==`); the `[GPU-BOX]` testset shows 1 Broken (`@test_skip`).

- [ ] **Step 5: Verify the module still loads**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; println("LOAD OK")'
```
Expected: `LOAD OK`.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/inner_core.jl src/GeoDynamo.jl test/gpu_phase5l_inner_core.jl
git commit -m "feat(gpu): Phase 5l conducting-inner-core flux + reconstruction

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Register the test + regression check

**Files:**
- Modify: `test/runtests.jl` (add the Phase 5l entry after the Phase 5k entry)

- [ ] **Step 1: Add the test to the suite**

In `test/runtests.jl`, find the line that includes `"gpu_phase5k_velocity_step.jl"` and add immediately after it (same indentation):

```julia
    "gpu_phase5l_inner_core.jl",
```

- [ ] **Step 2: Confirm the new test still passes in isolation**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5l_inner_core.jl")' > /tmp/phase5l.log 2>&1; echo "exit=$?"; tail -20 /tmp/phase5l.log
```
Expected: `exit=0`, the three `[LOCAL]` testsets pass, 1 Broken for the GPU-box gate.

- [ ] **Step 3: Confirm the allocation guards still pass**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/allocation_runtime_checks.jl")' > /tmp/allocguards.log 2>&1; echo "exit=$?"; tail -8 /tmp/allocguards.log
```
Expected: `exit=0`, 39/39 unchanged.

- [ ] **Step 4: Commit**

```bash
git add test/runtests.jl
git commit -m "test(gpu): register Phase 5l inner-core in suite

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** packer flattens `InnerCoreAdmittance` → `(2bw+1,Nic,nl)` batched `lin`/`lu` with identity-LU + zero-L for non-stored degrees ✓; history flux = matvec(5c) → CNAB2 assembly → zero BC rows → solve(5d) → `d1_top·y` reduction ✓; reconstruct = matvec → assembly → BC(0,g) → solve ✓; `inv_dt=1/dt`, `weight=1−θ` from `adm.dt`/`adm.theta` ✓; reduction sums ascending-i to match `dot` ✓; runs on Array + CuArray ✓.

**Placeholder scan:** none — every step has full code/commands.

**Type consistency:** `gpu_pack_inner_core` returns the `ic` bundle with fields `lin_ic/lu_ic/d1_top/inv_dt/weight/Nic/bw`, consumed identically by both orchestrators and `_gpu_ic_build_bic!`; the reduction kernel arg order `(φ0, y, d1_top, Nic)` matches both launch sites; `_gpu_ic_build_bic!` writes the CNAB2 history into its first two args (reused as `y_*` for the flux, `S_new_*` for the reconstruct).

**Divide-by-zero safety:** non-stored / `l=0` slots carry an identity LU (diagonal 1) → the batched solve returns `x=b` for those empty modes, never divides by a zero pivot. Tested by the packer identity-fill assertions.

**Aliasing:** `gpu_reconstruct_inner_core!` documents that `S_new_*` may not alias `S_old_*` (the matvec reads `S_old` while writing `S_new`).
