# Single-GPU CUDA Port — Design

**Status:** Approved design (scoping doc). Each phase below gets its own spec + implementation plan.

**Date:** 2026-06-05

## Goal

Run the full GeoDynamo physics (temperature, composition, velocity, magnetic / MHD) on **one** NVIDIA GPU, with the timestep **resident on-device**, producing results equivalent to the CPU path (within floating-point tolerance). A dedicated CUDA solver path, separate from the CPU path, built in phases that each leave a working, GPU-validatable solver.

## Decisions (locked during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Ambition | **Full physics parity** | A real GPU solver, not a proof-of-concept. |
| Path structure | **Separate CUDA path** (B) | Hand-written CUDA kernels for solver glue; CPU path untouched (zero regression risk). Maximum control over GPU kernels. |
| SH transform | **Reuse SHTnsKit `gpu_*`** | SHTnsKit's GPU extension (1401 lines: `gpu_synthesis`/`gpu_analysis`/`gpu_*_sphtor`, CuFFT, KA Legendre kernels) already exists and is numerically hardened (handles the `Plm` overflow fixed in SHTnsKit 1.2.10, normalization, Gauss grid). Reimplementing risks re-introducing solved bugs. Hand-rolling the transform deferred unless profiling proves it the bottleneck. |
| Radial implicit solve | **Batched on GPU** | One banded system per `(l,m)` mode, thousands solved in parallel as a CUDA kernel. Keeps data on-device — no per-step host↔device copy. |
| Data residency | **Device-resident timestep** | Fields live on-device; host↔device copy only at IO / diagnostics boundaries. |
| Multi-GPU | **Out of scope** | Single GPU only. SHTnsKit has `multi_gpu_*` / streaming; deferred to a later effort. |
| Backend | **CUDA only** | CUDA.jl-specific. Metal/AMD out of scope. |
| Validation | **Bit-equivalence vs CPU, ~1e-12, on the user's GPU box** | Claude cannot run CUDA locally (Apple Silicon, no NVIDIA). Every phase is gated on a GPU run by the user. GPU reductions reorder → tolerance, not bitwise. |

## Current state (from the 2026-06-05 audit)

The repo already has **orphaned** GPU scaffolding that targets the pre-Phase-1 (gather-replicate) transform, which the current Phase-3 DistTransposePlan path no longer uses:

- `core/architecture.jl`: `AbstractArchitecture`, `CPU`, `GPU{B}` types (exported).
- `ext/GeoDynamoCUDAExt.jl`: overrides legacy `host_*` helpers (`host_fill_scalar_coeff_buffer`, `host_extract_physical_slice`, …) — **dead** (0 live callers; the helpers they hook have 0–1 dead call-sites).
- `uses_gpu(config)` / `transform_arch(config)` (`solver/numerics.jl`): can flip true (a `SolverBackend` with a GPU arch sets `_buffers.transform_device`), but **the live transform path never consults it**.
- Fields are **unconditionally host `Array`** — `create_shtns_spectral_field`/`..._physical_field` (`fields/containers.jl`) allocate `PencilArray{T}(undef, pencil)`; no `on_architecture`/`arch_zeros`/`adapt`/`CuArray` anywhere in the live `fields/`/`solver/`/`parallel/` path.

**Consequence today:** a GPU config silently runs on CPU (no GPU memory allocated, no kernel launched, no crash). This port replaces the orphaned scaffolding with a real, live CUDA path.

The enabler: **`SHTnsKitGPUExt.jl`** exports `gpu_synthesis`, `gpu_analysis`, `gpu_analysis_safe`, `gpu_synthesis_safe`, `gpu_analysis_sphtor`, `gpu_synthesis_sphtor`, plus `to_device`, `get_device`, `set_gpu_device`, CuFFT plans. On a single GPU there is no MPI, so the serial GPU transform is used directly (the distributed `dist_*` path is irrelevant here).

## Architecture

A new `src/gpu/` module holding the CUDA solver path, selected at model construction when the architecture is `GPU`:

- **GPU fields** — CuArray-backed spectral and physical field containers, mirroring the CPU containers' shapes (spectral `(modes…, nr)`, physical `(nlat, nlon, nr)`), with explicit host↔device transfer.
- **GPU transform** — thin adapters that feed CuArray field data into SHTnsKit `gpu_synthesis`/`gpu_analysis` (scalar) and `gpu_*_sphtor` (vector), plus the small assembly kernels (e.g. `v_r` radial scaling, curl) the CPU path does inline.
- **GPU nonlinear** — CUDA kernels for advection `(u·∇)x`, Coriolis, Lorentz `J×B`, buoyancy coupling, operating on device physical fields.
- **GPU radial solve** — batched banded solver (one system per `(l,m)` mode) as a CUDA kernel; operator matrices precomputed per-degree `l` and resident on-device.
- **GPU timestepping** — CNAB2 / ERK2 stage orchestration using the GPU implicit solve + GPU explicit nonlinear; BC application kernels.
- **GPU run loop** — device-resident `run!`/`Simulation` integration; IO/restart/diagnostics gather to host for NetCDF.

The CPU path is not modified. The two paths share the high-level API (`GeodynamoModel`/`Simulation`/`run!`) via architecture dispatch at the entry points only; below that, the GPU path is its own code.

### Data flow (one timestep, GPU)

```
[device] spectral fields
   │  SHTnsKit gpu_synthesis / gpu_*_sphtor          (transform, reused)
   ▼
[device] physical fields ── CUDA nonlinear kernels ──► [device] nonlinear physical
   │  SHTnsKit gpu_analysis / gpu_*_sphtor
   ▼
[device] nonlinear spectral ── CUDA timestep (explicit) ──► RHS
   │  CUDA batched banded solve (implicit, per-mode)
   ▼
[device] updated spectral fields        (no host copy this whole loop)
```
Host transfer happens only when IO / diagnostics / restart fire.

## Phases

Each phase is an independent sub-project (own spec + plan), leaving a working, GPU-validatable increment. Gate = equivalence to the CPU path within ~1e-12 (relative/absolute as appropriate), measured on the user's GPU box.

### Phase 0 — GPU foundation + fields
- CUDA dependency wiring (live CUDA path, not the orphaned ext); device selection/management (`set_gpu_device`, `get_device` via SHTnsKit + CUDA.jl).
- CuArray-backed spectral + physical field containers; host↔device transfer utilities (`to_device`, `to_host`).
- **Gate:** allocate fields on GPU; copy a field host→device→host bit-identical; CPU path unchanged (CPU regression suite green).

### Phase 1 — GPU scalar transform (T / C)
- Adapt scalar `spectral↔physical` for temperature/composition to SHTnsKit `gpu_synthesis`/`gpu_analysis` on CuArray fields (single-GPU serial).
- **Gate:** scalar transform roundtrip on GPU ≈ CPU (~1e-12), dealiased grid.

### Phase 2 — GPU nonlinear terms
- CUDA kernels: advection `(u·∇)x`, Coriolis `2Ω×u`, Lorentz `(∇×B)×B`, buoyancy/codensity coupling — on device physical fields.
- **Gate:** each nonlinear term computed on GPU ≈ CPU per field.

### Phase 3 — GPU vector transform (velocity / magnetic)
- Toroidal–poloidal vector `spectral↔physical` via SHTnsKit `gpu_synthesis_sphtor`/`gpu_analysis_sphtor`; the `v_r` (`l(l+1)/r·P`) radial-scaling and curl-assembly kernels in CUDA.
- **Gate:** vector roundtrip + one velocity nonlinear step on GPU ≈ CPU.

### Phase 4 — GPU batched radial banded solve
- Batched banded solver kernel: one system per `(l,m)` mode, solved in parallel (batched Thomas for tridiagonal / banded-LU for wider bandwidth); operator matrices precomputed per `l`, device-resident.
- **Gate:** solve on GPU ≈ CPU banded solve for a known RHS, across all modes and both shell/ball geometries.

### Phase 5 — GPU timestepping + full step
- CNAB2 / ERK2 stage logic on GPU (implicit radial solve from Phase 4 + explicit nonlinear from Phases 2–3); boundary-condition application kernels; full `solver_step!` on GPU.
- **Gate:** one full MHD step (temperature, composition, velocity, magnetic) on GPU ≈ CPU.

### Phase 6 — run! / Simulation + IO
- Device-resident `run!`/`Simulation` loop; IO/restart host-gather (device→host for NetCDF write, host→device on read); diagnostics host-gather.
- **Gate:** a multi-step run on GPU ≈ CPU; restart round-trips through host.

## Validation strategy

- **No local CUDA.** Claude develops on Apple Silicon and cannot compile/run CUDA. The user runs every phase's gate on their GPU box; Claude writes, the user reports, Claude fixes. Mitigation: small, independently testable kernels; tight per-phase gates; defensive index-explicit kernels (CUDA disallows scalar indexing on `CuArray`).
- **Tolerance, not bitwise.** GPU reductions and FMA reorder arithmetic, so the gate is `≈` within ~1e-12 (tightened per component where the operation is reduction-free), unlike the bit-exact CPU r×θ / DistTransposePlan work.
- **Reuse the step-equivalence harness.** The existing `r_theta_equivalence*` snapshot/compare pattern (write a signature, diff against the CPU reference) extends to GPU≈CPU per field, per step.
- **CPU regression.** Each phase keeps the full CPU suite green (the CPU path must not change).

## Risks

1. **Blind CUDA development** (no local GPU) — slowest factor; every cycle is a user GPU run. Phase gates are small to keep cycles cheap.
2. **Batched banded solve (Phase 4)** — the riskiest custom kernel (correctness + numerical conditioning across modes/geometries).
3. **Determinism** — GPU non-associativity means equivalence is tolerance-based; a too-tight gate would false-fail. Gates specify realistic tolerances.
4. **Scalar-indexing traps** — any stray scalar index on a `CuArray` errors (`allowscalar`); all kernels must be index-explicit.
5. **Host↔device transfer discipline** — accidental copies inside the step would kill performance; the design forbids host transfer except at IO/diagnostics.

## Out of scope

- Multi-GPU and GPU+MPI (single GPU only).
- Non-CUDA backends (Metal, ROCm).
- Performance tuning beyond "device-resident and correct" (parity first; optimization is a follow-up once the path is validated).
- Hand-written CUDA SH transform (reuse SHTnsKit; revisit only if profiling proves it the bottleneck).
- Removing the orphaned legacy GPU scaffolding — tracked separately as cleanup; this port supersedes it but its removal is not a blocker.
