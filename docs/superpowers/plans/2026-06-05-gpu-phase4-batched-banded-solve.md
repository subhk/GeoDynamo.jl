# GPU Phase 4 — Batched Radial Banded Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Solve the per-mode radial banded linear systems `A_l · x = b` on a single GPU — one banded LU factorization per spherical-harmonic degree `l` (reused across all `m`), thousands of `(l,m)` modes solved in parallel — matching the CPU `solve_banded!` exactly.

**Architecture:** From the CPU map: `A_l` is **banded** (half-bandwidth `bw`, typically 2), depends only on `l` (via `l(l+1)/r²`), is LU-factored once at setup and reused every timestep; the per-step solve is forward+back substitution on the banded LU. All `(l,m)` solves are independent. We implement the substitution as a **KernelAbstractions `@kernel`** — one workitem per `(l,m)` mode, each doing the sequential length-`nr` forward/back sweep, reading its degree's LU factor from a batched `(2bw+1, nr, nl)` array. KA runs on the CPU backend (Array → **locally testable** against the CPU `solve_banded!`) and on CUDA (CuArray). The per-`l` LU factors are built once on the host with the existing `factorize_banded` and packed into the batched array. **Chosen over hand-written `CUDA.@cuda`** for local testability + portability (KA emits the CUDA kernel).

**Tech Stack:** Julia, KernelAbstractions (already a GeoDynamo dep — `core/architecture.jl` does `using KernelAbstractions`), reuses `GeoDynamo.BandedMatrix`/`factorize_banded`/`BandedLU`/`solve_banded!` (`src/numerics/banded_operators.jl`). No CUDA extension methods needed — KA dispatches on the array backend.

---

## Background (CPU reference — `src/numerics/banded_operators.jl`)

- `BandedLU{T}` stores `lu::Matrix{T}` shape `(2bw+1, N)`; element `A[i,j]` (within band) lives at `lu[bw+1+i-j, j]` (helper `_band_row(i,j,bw)=bw+1+i-j`).
- `solve_banded!(x, lu, b)` (lines 84-125):
  - **Forward** (`L y = b`, unit diag): `for i in 1:N: x[i] = b[i] - Σ_{j=max(1,i-bw)}^{i-1} lu[bw+1+i-j, j]·x[j]`.
  - **Back** (`U x = y`): `for i in N:-1:1: x[i] = (x[i] - Σ_{j=i+1}^{min(N,i+bw)} lu[bw+1+i-j, j]·x[j]) / lu[bw+1, i]`.
  - In-place `x === b` is safe (forward ascending, back descending).
- One factorization per degree `l`; `bw` is uniform across modes.

## Testing without a local GPU

- **[LOCAL]** — the KA kernel runs on the CPU backend (Array). Tests build banded matrices, factor with `factorize_banded`, solve a batch via the kernel, and assert **each `(l,m)` column equals `solve_banded!`** (exact `==` — the kernel replicates the same sequential arithmetic). Real verification.
- **[GPU-BOX]** — same on `CuArray` (CUDA backend); guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu` (branch `feat/gpu-phase0`, on Phases 0–3). **Never pipe test runs through `tail`.**

## File Structure

- **Create** `src/gpu/banded_solve.jl` — the KA kernel, the `gpu_batched_banded_solve!` driver, and `gpu_pack_banded_lu` (host pack).
- **Modify** `src/GeoDynamo.jl` — `include("gpu/banded_solve.jl")` (after `gpu/vector_transform.jl`); export `gpu_batched_banded_solve!`, `gpu_pack_banded_lu`.
- **Create** `test/gpu_phase4_banded_solve.jl` — `[LOCAL]` + `[GPU-BOX]` tests.
- **Modify** `test/runtests.jl` — register.

Locked interfaces:

```julia
gpu_pack_banded_lu(lus::AbstractVector, arch) -> Array/CuArray  # (2bw+1, nr, nl) batched LU, from per-l BandedLU
gpu_batched_banded_solve!(X, B, lu_batched, bw)                 # X[l,m,:] = A_l \ B[l,m,:] for all (l,m); in-place X===B ok
```

`X`/`B` are `(nl, nm, nr)` real arrays (a spectral field's `data_real` or `data_imag`); `lu_batched` is `(2bw+1, nr, nl)`; degree `l = dim-1 index − 1`; all on the same backend.

---

## Task 1: `gpu_pack_banded_lu` (host pack per-l LU → batched array)

**Files:** Create `src/gpu/banded_solve.jl`; Modify `src/GeoDynamo.jl`; Test `test/gpu_phase4_banded_solve.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase4_banded_solve.jl`:

```julia
using Test
using GeoDynamo

# Build a non-singular banded matrix (diagonally dominant) in BandedLU storage.
function _rand_banded(::Type{T}, N, bw; seed) where {T}
    import_rng = MersenneTwister(seed)
    data = zeros(T, 2bw+1, N)
    for j in 1:N, i in max(1,j-bw):min(N,j+bw)
        data[bw+1+i-j, j] = (i == j) ? (T(2bw) + rand(import_rng, T)) : (rand(import_rng, T) - T(0.5))
    end
    return GeoDynamo.BandedMatrix{T}(data, bw, N)
end

using Random
@testset "GPU Phase 4 — Batched Banded Solve" begin
    @testset "pack banded LU [LOCAL]" begin
        N, bw, nl = 8, 2, 3
        lus = [GeoDynamo.factorize_banded(_rand_banded(Float64, N, bw; seed = 10 + l)) for l in 1:nl]
        packed = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
        @test size(packed) == (2bw + 1, N, nl)
        for l in 1:nl
            @test packed[:, :, l] == lus[l].lu
        end
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase4_banded_solve.jl")'`
Expected: FAIL — `gpu_pack_banded_lu` undefined.

- [ ] **Step 3: Implement**

Create `src/gpu/banded_solve.jl`:

```julia
# =============================================================================
# GPU Phase 4 — batched radial banded solve (one banded LU per degree l, reused
# across all m; all (l,m) modes solved in parallel).  A KernelAbstractions kernel
# does the per-mode forward/back substitution, replicating the CPU solve_banded!
# (src/numerics/banded_operators.jl) exactly.  Runs on the CPU backend (Array,
# locally testable) and CUDA (CuArray).  The per-l LU factors are built on the
# host with factorize_banded and packed into a (2bw+1, nr, nl) batched array.
# =============================================================================

"""
    gpu_pack_banded_lu(lus, arch) -> (2bw+1, nr, nl) array on arch's backend

Stack the per-degree `BandedLU` factors (`lus[l]` for l-slot `l`) into a single
batched array `lu_batched[:, :, l] = lus[l].lu`, on `arch`'s backend.  All factors
must share the same bandwidth `bw` and size `nr`.
"""
function gpu_pack_banded_lu(lus::AbstractVector, arch::AbstractArchitecture)
    nl = length(lus)
    bw = lus[1].bandwidth
    nr = lus[1].size
    host = Array{eltype(lus[1].lu)}(undef, 2bw + 1, nr, nl)
    for l in 1:nl
        @assert lus[l].bandwidth == bw && lus[l].size == nr "gpu_pack_banded_lu: factors must share bw/size"
        host[:, :, l] .= lus[l].lu
    end
    return on_architecture(arch, host)
end
```

- [ ] **Step 4: Include + export**

In `src/GeoDynamo.jl`, after `include("gpu/vector_transform.jl")` add `include("gpu/banded_solve.jl")`. Add export line:
```julia
export gpu_pack_banded_lu, gpu_batched_banded_solve!
```

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase4_banded_solve.jl")'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/banded_solve.jl src/GeoDynamo.jl test/gpu_phase4_banded_solve.jl
git commit -m "feat(gpu): gpu_pack_banded_lu (batch per-l banded LU factors) (Phase 4)"
```

---

## Task 2: KA kernel + `gpu_batched_banded_solve!` driver

**Files:** Modify `src/gpu/banded_solve.jl`; Test `test/gpu_phase4_banded_solve.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase4_banded_solve.jl`:

```julia
@testset "batched solve == solve_banded! (multiple l, bw=2) [LOCAL]" begin
    N, bw, nl, nm = 10, 2, 4, 3
    mats = [_rand_banded(Float64, N, bw; seed = 100 + l) for l in 1:nl]
    lus = [GeoDynamo.factorize_banded(m) for m in mats]
    packed = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
    rng = MersenneTwister(7)
    B = rand(rng, Float64, nl, nm, N)
    X = zeros(Float64, nl, nm, N)
    GeoDynamo.gpu_batched_banded_solve!(X, B, packed, bw)
    # reference: per (l,m), solve_banded! with that l's factor on that column
    for l in 1:nl, m in 1:nm
        xref = zeros(Float64, N)
        GeoDynamo.solve_banded!(xref, lus[l], collect(B[l, m, :]))
        @test X[l, m, :] == xref
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase4_banded_solve.jl")'`
Expected: FAIL — `gpu_batched_banded_solve!` undefined.

- [ ] **Step 3: Implement the kernel + driver**

Append to `src/gpu/banded_solve.jl` (`@kernel`/`@index`/`@Const` come from KernelAbstractions, already `using`d by the module):

```julia
# One workitem per (l,m) mode. Each does the sequential length-nr forward/back
# substitution along dim 3, reading its degree's LU factor lu_batched[:,:,li].
# Mirrors solve_banded! exactly (banded_operators.jl:84-125). The bounded j-ranges
# guarantee the band row index bw+1+i-j ∈ [1, 2bw+1], so no in-loop guard is needed.
@kernel function _banded_solve_kernel!(X, @Const(B), @Const(lu_batched), bw::Int, nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(X)
    # Forward: L y = b  (unit diagonal)
    @inbounds for i in 1:nr
        s = zero(T)
        for j in max(1, i - bw):(i - 1)
            s += lu_batched[bw + 1 + i - j, j, li] * X[li, mi, j]
        end
        X[li, mi, i] = B[li, mi, i] - s
    end
    # Back: U x = y
    @inbounds for i in nr:-1:1
        s = zero(T)
        for j in (i + 1):min(nr, i + bw)
            s += lu_batched[bw + 1 + i - j, j, li] * X[li, mi, j]
        end
        X[li, mi, i] = (X[li, mi, i] - s) / lu_batched[bw + 1, i, li]
    end
end

"""
    gpu_batched_banded_solve!(X, B, lu_batched, bw) -> X

Solve `A_l · X[l,m,:] = B[l,m,:]` for every `(l,m)`, where `A_l`'s banded LU is
`lu_batched[:,:,l]` (degree `l` = dim-1 index).  `X`/`B` are `(nl,nm,nr)`; in-place
`X === B` is supported.  Backend (CPU/CUDA) is inferred from `X`.
"""
function gpu_batched_banded_solve!(X, B, lu_batched, bw::Int)
    nl, nm, nr = size(X)
    backend = KernelAbstractions.get_backend(X)
    _banded_solve_kernel!(backend)(X, B, lu_batched, bw, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)
    return X
end
```

- [ ] **Step 4: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase4_banded_solve.jl")'`
Expected: PASS — every `(l,m)` column equals `solve_banded!`.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/banded_solve.jl test/gpu_phase4_banded_solve.jl
git commit -m "feat(gpu): KA batched banded-solve kernel + driver (Phase 4)"
```

---

## Task 3: in-place + edge-case correctness

**Files:** Test `test/gpu_phase4_banded_solve.jl`

- [ ] **Step 1: Write the tests** `[LOCAL]`

Add to `test/gpu_phase4_banded_solve.jl`:

```julia
@testset "in-place X===B + bandwidth 1 + single l [LOCAL]" begin
    # in-place aliasing (X === B) must match the out-of-place result
    N, bw, nl, nm = 9, 2, 2, 2
    lus = [GeoDynamo.factorize_banded(_rand_banded(Float64, N, bw; seed = 200 + l)) for l in 1:nl]
    packed = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
    B = rand(MersenneTwister(11), Float64, nl, nm, N)
    Xout = zeros(Float64, nl, nm, N)
    GeoDynamo.gpu_batched_banded_solve!(Xout, B, packed, bw)        # out-of-place
    Xin = copy(B)
    GeoDynamo.gpu_batched_banded_solve!(Xin, Xin, packed, bw)        # in-place
    @test Xin == Xout

    # bandwidth 1 (tridiagonal) still correct
    lus1 = [GeoDynamo.factorize_banded(_rand_banded(Float64, N, 1; seed = 300 + l)) for l in 1:nl]
    p1 = GeoDynamo.gpu_pack_banded_lu(lus1, CPU())
    B1 = rand(MersenneTwister(13), Float64, nl, nm, N); X1 = zeros(Float64, nl, nm, N)
    GeoDynamo.gpu_batched_banded_solve!(X1, B1, p1, 1)
    for l in 1:nl, m in 1:nm
        xref = zeros(Float64, N); GeoDynamo.solve_banded!(xref, lus1[l], collect(B1[l,m,:]))
        @test X1[l, m, :] == xref
    end
end
```

- [ ] **Step 2: Run it, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase4_banded_solve.jl")'`
Expected: PASS — in-place matches out-of-place; bandwidth-1 matches reference.

- [ ] **Step 3: Commit**

```bash
git add test/gpu_phase4_banded_solve.jl
git commit -m "test(gpu): batched banded solve in-place + bandwidth-1 cases (Phase 4)"
```

---

## Task 4: GPU-box gate + register + regression

**Files:** Test `test/gpu_phase4_banded_solve.jl`, `test/runtests.jl`

- [ ] **Step 1: Add the GPU-box gate** `[GPU-BOX]`

Add to `test/gpu_phase4_banded_solve.jl`:

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-4 gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        N, bw, nl, nm = 12, 2, 4, 3
        lus = [GeoDynamo.factorize_banded(_rand_banded(Float64, N, bw; seed = 400 + l)) for l in 1:nl]
        B = rand(MersenneTwister(21), Float64, nl, nm, N)
        # CPU reference
        cpacked = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
        cX = zeros(Float64, nl, nm, N)
        GeoDynamo.gpu_batched_banded_solve!(cX, B, cpacked, bw)
        # GPU
        gpacked = GeoDynamo.gpu_pack_banded_lu(lus, GPU())
        gX = GeoDynamo.on_architecture(GPU(), zeros(Float64, nl, nm, N))
        gB = GeoDynamo.on_architecture(GPU(), B)
        GeoDynamo.gpu_batched_banded_solve!(gX, gB, gpacked, bw)
        @test gX isa CUDA.CuArray
        @test isapprox(Array(gX), cX; atol = 1e-12, rtol = 1e-10)
    end
end
```

- [ ] **Step 2: Run locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase4_banded_solve.jl")'`
Expected: `[LOCAL]` testsets pass; the gate skips. Mark **"implemented; awaiting GPU-box parity."**

- [ ] **Step 3: Register**

In `test/runtests.jl`, add `"gpu_phase4_banded_solve.jl"` (next to the Phase 3 entry).

- [ ] **Step 4: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI; include("test/gpu_phase3_vector_transform.jl"); include("test/allocation_runtime_checks.jl")'`
Expected: Phase 3 green; allocation guards 39/39.

- [ ] **Step 5: Commit**

```bash
git add test/gpu_phase4_banded_solve.jl test/runtests.jl
git commit -m "test(gpu): Phase-4 GPU-box gate + register batched banded solve"
```

---

## GPU-box validation handoff

On the GPU box:
```julia
using CUDA, Test, GeoDynamo, Random
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase4_banded_solve.jl")    # the [GPU-BOX] gate must PASS
```
**Phase 4 passes when:** the batched solve on `CuArray` matches the CPU result to ~1e-12 across all modes. Report any failure (a KA-on-CUDA indexing surprise, a `synchronize` issue, or a per-mode divergence) before Phase 5. NOTE: a banded substitution has a sequential recurrence per mode — on GPU each mode is one thread doing `nr` sequential steps; this is correct but not bandwidth-optimal (a Phase-5+ optimization could parallelize within the solve). Correctness-first here.

---

## Self-Review

**Spec coverage (design-doc Phase 4: "batched radial banded solve — 1 system/mode, CUDA batched Thomas/LU, device matrices; gate solve GPU≈CPU across modes/geometries"):** `gpu_pack_banded_lu` (device matrices, Task 1), the KA batched substitution kernel + driver (Task 2), in-place + bandwidth cases (Task 3), GPU-box parity gate + regression (Task 4). The "across geometries (shell/ball)" aspect is covered by the per-l factor reuse — the kernel is geometry-agnostic (it consumes whatever `factorize_banded` produced); shell vs ball differ only in the matrices fed in, which is a Phase-5 wiring concern. The RHS assembly (CNAB2 formula) + integration into the timestep is Phase 5. Covered.

**Placeholder scan:** none — every code step has complete code; every run step has the exact command + expected result. (`_rand_banded` test helper is fully defined.)

**Type consistency:** `gpu_pack_banded_lu(lus, arch)→(2bw+1,nr,nl)`, `gpu_batched_banded_solve!(X,B,lu_batched,bw)` over `(nl,nm,nr)` — consistent across tasks and the interface block. Reuses `GeoDynamo.BandedMatrix`/`factorize_banded`/`BandedLU`(`.lu`/`.bandwidth`/`.size`)/`solve_banded!` exactly as defined in `src/numerics/banded_operators.jl`. The kernel's band-row index `bw+1+i-j` and forward/back sweeps mirror `solve_banded!` (lines 84-125); the bounded `j`-ranges keep the row index in `[1,2bw+1]`. `on_architecture(arch, host)` (Phase 0) places the batched array on the backend; `KernelAbstractions.get_backend(X)` selects the launch backend.
