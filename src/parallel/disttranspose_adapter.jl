# =============================================================================
# Phase 3 — DistTransposePlan plumbing (Task 1)
# =============================================================================
#
# This file provides three public entry points for the Phase-3 θ-distributed
# transform path:
#
#   get_disttranspose_plan(cfg)              → cached DistTransposePlan
#   to_spec_solve(cfg, Alm, plan)            → solver-oriented PencilArray
#   from_spec_solve!(cfg, Alm, solve, plan)  → write result back into Alm
#
# Architecture:
#   Alm layout (from SHTnsKit.allocate_spectral):
#       parent dim-1 = l (lmax+1, full)
#       parent dim-2 = m-bin (nbin = nlon÷2+1, local subset via θ_comm)
#       parent dim-3 = r-level (nlev = nr_local, local)
#
#   spec_solve layout (for the radial implicit solve):
#       parent dim-1 = l-local (l distributed over r_comm)
#       parent dim-2 = r (nr_global, full)
#       parent dim-3 = m-bin (nml, same local count as Alm dim-2)
#
# The transpose between the two is a PencilArrays.transpose! over r_comm,
# treating Alm as (lmax+1, nr_global) with nml as a trailing batch dimension.
# Alm dim-2 (m-bin count) is uniform across r_comm members because all ranks
# in r_comm share the same θ-slab → same m-bin ownership.
#
# Caching: plans and scratch PencilArrays are stored in a module-level IdDict
# keyed by the config object.  Build-once semantics: the closure creates them
# on first call and subsequent calls return the cached value.
# =============================================================================

using SHTnsKit
using PencilArrays
using MPI

# ---------------------------------------------------------------------------
# Module-level cache (keyed by SHTnsKitConfig identity)
# ---------------------------------------------------------------------------

# Stores the DistTransposePlan (one per config).
const _DISTTRANSPOSE_PLAN_CACHE = IdDict{Any, Any}()

# Stores the scratch PencilArrays used by to_spec_solve / from_spec_solve!
# so they are only allocated once per config.
# Value: NamedTuple (pen_alm_r, pen_solve_r, almr_scratch, solve_scratch)
const _DISTTRANSPOSE_SCRATCH_CACHE = IdDict{Any, Any}()

const _DISTTRANSPOSE_LOCK = ReentrantLock()

# ---------------------------------------------------------------------------
# Plan construction helpers
# ---------------------------------------------------------------------------

"""
    _build_disttranspose_plan(cfg) -> SHTnsKit.DistTransposePlan

Build (but do not cache) a `DistTransposePlan` for `cfg`.  The plan is built
over `cfg.pencils.θ_comm` with `nlev = nr_local` (the number of radial levels
owned by this rank).
"""
function _build_disttranspose_plan(cfg)
    nr_local = length(PencilArrays.range_local(cfg.pencils.r)[3])
    nr_local > 0 || error(
        "DistTransposePlan: this rank owns 0 radial levels — " *
        "reduce the number of r-ranks so that every rank has nr_local ≥ 1."
    )
    return SHTnsKit.DistTransposePlan(
        cfg.sht_config;
        comm      = cfg.pencils.θ_comm,
        nlev      = nr_local,
        use_rfft  = true,
        with_vector = true,
    )
end

"""
    _build_disttranspose_scratch(cfg, plan) -> NamedTuple

Build the pair of scratch `PencilArray`s used for the Alm ↔ spec_solve
transpose, keyed by `cfg`.  Both arrays are pre-allocated once and reused.
Also builds persistent `Transposition` plans so that `transpose!` never
allocates MPI communication buffers on repeated calls.

Returns `(; pen_alm_r, pen_solve_r, almr, solve, t_fwd, t_bwd)` where:
- `t_fwd` : `Transposition(solve, almr)` — forward (almr → solve, for `to_spec_solve`)
- `t_bwd` : `Transposition(almr, solve)` — backward (solve → almr, for `from_spec_solve!`)
"""
function _build_disttranspose_scratch(cfg, plan)
    lmax  = cfg.lmax
    nr    = PencilArrays.size_global(cfg.pencils.r)[3]
    nml   = size(parent(SHTnsKit.allocate_spectral(plan)), 2)  # m-bin count (local)

    r_comm  = cfg.pencils.r_comm
    r_ranks = MPI.Comm_size(r_comm)

    TopoCtor = getproperty(PencilArrays, Symbol("MPITopology"))
    topo_r      = TopoCtor(r_comm, (r_ranks,))
    # pen_alm_r : r distributed (dim-2), l full — carries the reordered Alm data
    pen_alm_r   = Pencil(topo_r, (lmax + 1, nr), (2,))
    # pen_solve_r: l distributed (dim-1), r full — the solver orientation
    pen_solve_r = Pencil(topo_r, (lmax + 1, nr), (1,))

    almr  = PencilArray{ComplexF64}(undef, pen_alm_r,   nml)
    solve = PencilArray{ComplexF64}(undef, pen_solve_r, nml)

    # Persistent Transposition plans — bind to the specific almr/solve arrays so
    # that repeated `transpose!(t)` calls reuse the Pencil-internal send/recv
    # buffers without allocating a new Transposition struct each time.
    t_fwd = PencilArrays.Transpositions.Transposition(solve, almr)  # almr → solve
    t_bwd = PencilArrays.Transpositions.Transposition(almr, solve)  # solve → almr

    return (; pen_alm_r, pen_solve_r, almr, solve, t_fwd, t_bwd)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    get_disttranspose_plan(cfg) -> SHTnsKit.DistTransposePlan

Return the `DistTransposePlan` associated with `cfg`, building and caching it
on the first call.  Subsequent calls are O(1) dict-lookups.

The plan is constructed on `cfg.pencils.θ_comm` with `nlev = nr_local`.
"""
function get_disttranspose_plan(cfg)
    # Fast path: plan is present after the first call — avoid lock + closure alloc.
    p = get(_DISTTRANSPOSE_PLAN_CACHE, cfg, nothing)
    p !== nothing && return p
    lock(_DISTTRANSPOSE_LOCK) do
        get!(_DISTTRANSPOSE_PLAN_CACHE, cfg) do
            _build_disttranspose_plan(cfg)
        end
    end
end

"""
    _get_disttranspose_scratch(cfg, plan)

Return the cached scratch NamedTuple for `cfg`, building it on the first call.
Uses a fast non-locking lookup on the hot path (after warm-up the key always
exists); falls back to the locked build path only on the first call.
"""
@inline function _get_disttranspose_scratch(cfg, plan)
    # Fast path: key is present after the first call — avoid lock + closure alloc.
    s = get(_DISTTRANSPOSE_SCRATCH_CACHE, cfg, nothing)
    s !== nothing && return s
    # Slow path: first call, build under lock.
    lock(_DISTTRANSPOSE_LOCK) do
        get!(_DISTTRANSPOSE_SCRATCH_CACHE, cfg) do
            _build_disttranspose_scratch(cfg, plan)
        end
    end
end

"""
    to_spec_solve(cfg, Alm, plan) -> PencilArray{ComplexF64,3}

Transpose the distributed spectral coefficients `Alm` (shape `(lmax+1, nbin,
nlev)`, l-full / m-bin local / r-local) into the solver orientation (shape
`(l_local, nr_global, nbin)`), where `l` is now distributed over `r_comm` and
`r` is fully local on each rank.

The returned `PencilArray` is a cached scratch buffer — do not retain a
reference past the next call to `to_spec_solve` or `from_spec_solve!` on the
same `cfg`.
"""
# Type-stable reorder kernels (function barriers). `Ap`/`pa` come from `parent()`
# of ::Any-typed cached scratch/Alm (IdDict{Any,Any}); indexing them inline boxes
# every element. Passing them as plain args here forces concrete specialization.
function _reorder_alm_to_almr!(pa, Ap, nr_local::Int, nml::Int, lmax::Int)
    fill!(pa, zero(eltype(pa)))
    @inbounds for k in 1:nr_local, jm in 1:nml, il in 1:(lmax + 1)
        pa[il, k, jm] = Ap[il, jm, k]
    end
    return nothing
end

function _reorder_almr_to_alm!(Ap, pa, nr_local::Int, nml::Int, lmax::Int)
    @inbounds for k in 1:nr_local, jm in 1:nml, il in 1:(lmax + 1)
        Ap[il, jm, k] = pa[il, k, jm]
    end
    return nothing
end

function to_spec_solve(cfg, Alm, plan)
    scratch = _get_disttranspose_scratch(cfg, plan)
    # `scratch` is ::Any (IdDict cache) and holds PencilArrays whose concrete type
    # is config-dependent. Hand it to a barrier so the NamedTuple field accesses
    # and the reorder loop specialize on the concrete runtime type instead of
    # boxing. (spec_storage_to_solve! uses the equivalent field-assert pattern.)
    _to_spec_solve_impl!(scratch, Alm, cfg.lmax::Int,
        length(PencilArrays.range_local(cfg.pencils.r)[3]))
    return scratch.solve
end

function _to_spec_solve_impl!(scratch, Alm, lmax::Int, nr_local::Int)
    Ap = parent(Alm)
    # Reorder Alm parent (l, m_bin, r_local) → almr parent (l, r_local, m_bin)
    _reorder_alm_to_almr!(parent(scratch.almr), Ap, nr_local, size(Ap, 2), lmax)
    PencilArrays.transpose!(scratch.t_fwd)  # almr → solve (persistent plan, no alloc)
    return nothing
end

"""
    from_spec_solve!(cfg, Alm, solve, plan)

Transpose the solver-orientation `PencilArray` `solve` (output of the radial
implicit step) back into `Alm` (the DistTransposePlan layout).  Overwrites
`Alm` in-place.

`solve` MUST be `scratch.solve` (the adapter's cached solve buffer, returned by
`to_spec_solve` or `spec_storage_to_solve!`).  The persistent `Transposition`
plan `scratch.t_bwd` is bound to that specific array, allowing zero-alloc
repeated use.
"""
function from_spec_solve!(cfg, Alm, solve, plan)
    scratch = _get_disttranspose_scratch(cfg, plan)
    # Barrier on the ::Any scratch (see to_spec_solve): specialize on its concrete
    # runtime type so the transpose dispatch + reorder loop don't box.
    _from_spec_solve_impl!(scratch, Alm, solve, cfg.lmax::Int,
        length(PencilArrays.range_local(cfg.pencils.r)[3]))
    return nothing
end

function _from_spec_solve_impl!(scratch, Alm, solve, lmax::Int, nr_local::Int)
    # `solve` must be scratch.solve (the persistent plan t_bwd is bound to it).
    @assert solve === scratch.solve "from_spec_solve!: solve must be scratch.solve (the adapter's cached buffer)"
    # Transpose back into the (l, r_local, m_bin) layout using the persistent plan.
    PencilArrays.transpose!(scratch.t_bwd)  # solve → almr (no alloc)
    # Reorder almr parent (l, r_local, m_bin) → Alm parent (l, m_bin, r_local)
    Ap = parent(Alm)
    _reorder_almr_to_alm!(Ap, parent(scratch.almr), nr_local, size(Ap, 2), lmax)
    return nothing
end

# =============================================================================
# Phase 3 — scalar-storage ↔ spec_solve m-axis bridge (Task 2)
# =============================================================================
#
# The field's spectral storage (config.pencils.spec, decomp (2,1)) shares the
# `solve` orientation's l-over-r_comm and full-r layout EXACTLY, but its m-axis
# is the even split of (mmax+1) over θ_comm, whereas `solve`'s m-axis is the
# DistTransposePlan's nbin-based m-bin split (truncated to m ≤ mmax).  On
# canonical grids (nbin == mmax+1) these coincide and the copy is local; on
# DEALIASED grids (nbin > mmax+1) the per-θ_comm-rank m-ownership differs, so a
# θ_comm m-axis redistribution is required.
#
# We implement that redistribution by Allgatherv-ing the full m∈0..mmax columns
# onto every θ_comm rank (one collective, batched over l_local × nr).  This
# replicates only the small spectral m-axis; the heavy Legendre/FFT work stays
# θ-distributed inside dist_synthesis!/dist_analysis!.

# Cache: per-config metadata for the θ_comm m-redistribution.
#   (; θ_comm, θ_size, spec_m_counts, spec_m_offsets, full_block, l_local, nr)
const _DISTTRANSPOSE_MBRIDGE_CACHE = IdDict{Any, Any}()

function _build_mbridge(cfg, plan)
    θ_comm  = cfg.pencils.θ_comm
    θ_size  = MPI.Comm_size(θ_comm)
    spec    = cfg.pencils.spec

    # This rank's spec-storage m-range (global, 0-based) and l-range (global, 0-based).
    spec_m_range = PencilArrays.range_local(spec)[2]      # 1-based m-slots (= m+1)
    spec_l_range = PencilArrays.range_local(spec)[1]      # 1-based l-slots (= l+1)
    nr           = PencilArrays.size_global(spec)[3]
    l_local      = length(spec_l_range)
    m_local_cnt  = length(spec_m_range)

    # Gather every θ_comm member's spec m-slot count and first slot, so we can place
    # each contributor's columns at the right global m offset in the full-m block.
    counts  = MPI.Allgather(Int32(m_local_cnt), θ_comm)
    firsts  = MPI.Allgather(Int32(first(spec_m_range)), θ_comm)  # 1-based first m-slot
    m_counts = Int.(counts)

    # Pre-allocated scratch reused across every bridge call (fixed sizes per config;
    # transforms are serial per config so reuse is safe). Avoids per-call heap traffic.
    mmax       = cfg.mmax
    recvcounts = [c * l_local * nr for c in m_counts]
    send       = Vector{ComplexF64}(undef, l_local * m_local_cnt * nr)
    recv       = Vector{ComplexF64}(undef, sum(recvcounts))
    vbuf       = MPI.VBuffer(recv, recvcounts)
    full3      = Array{ComplexF64, 3}(undef, l_local, mmax + 1, nr)
    local_full = Array{ComplexF64, 3}(undef, l_local, mmax + 1, nr)

    return (; θ_comm, θ_size,
              spec_m_range, spec_l_range, nr, l_local, mmax,
              m_counts, m_firsts = Int.(firsts),
              recvcounts, send, recv, vbuf, full3, local_full)
end

@inline function _get_mbridge(cfg, plan)
    # Fast path: key is present after the first call — avoid lock + closure alloc.
    mb = get(_DISTTRANSPOSE_MBRIDGE_CACHE, cfg, nothing)
    mb !== nothing && return mb
    lock(_DISTTRANSPOSE_LOCK) do
        get!(_DISTTRANSPOSE_MBRIDGE_CACHE, cfg) do
            _build_mbridge(cfg, plan)
        end
    end
end

"""
    spec_storage_to_solve!(cfg, solve, sr, si, plan)

Fill the plan-oriented `solve` PencilArray (parent `(l_local, nr, nml)`) from the
field's spectral storage parents `sr`/`si` (parent `(l_local, m_local, nr)` on the
`(2,1)` spec pencil).  Performs the θ_comm m-axis redistribution.  `solve` columns
`1:length(plan.m_local)` are filled (degree `plan.m_local[mi]`); any trailing bins
are zeroed.
"""
function spec_storage_to_solve!(cfg, solve, sr, si, plan)
    mb = _get_mbridge(cfg, plan)
    # Type-assert the Any-typed cached fields / plan / cfg to concrete types so the
    # hot loops below specialize and DON'T box (the caches are IdDict{Any,Any}).
    θ_comm  = mb.θ_comm
    mmax    = cfg.mmax::Int
    nr      = mb.nr::Int
    l_local = mb.l_local::Int
    send    = mb.send::Vector{ComplexF64}
    recv    = mb.recv::Vector{ComplexF64}
    full    = mb.full3::Array{ComplexF64, 3}
    θ_size   = mb.θ_size::Int
    m_counts = mb.m_counts::Vector{Int}
    m_firsts = mb.m_firsts::Vector{Int}
    mlocal   = plan.m_local::Vector{Int}
    sp       = parent(solve)::AbstractArray{ComplexF64, 3}
    _spec_to_solve_kernel!(sp, send, recv, full, sr, si, mb.vbuf, θ_comm,
                           m_counts, m_firsts, mlocal, θ_size, nr, l_local, mmax,
                           length(mb.spec_m_range)::Int)
    return solve
end

# Type-stable kernel (function barrier): all args concrete → no boxing in the loops.
function _spec_to_solve_kernel!(sp, send::Vector{ComplexF64}, recv::Vector{ComplexF64},
        full::Array{ComplexF64, 3}, sr, si, vbuf, θ_comm,
        m_counts::Vector{Int}, m_firsts::Vector{Int}, mlocal::Vector{Int},
        θ_size::Int, nr::Int, l_local::Int, mmax::Int, m_local_cnt::Int)
    # 1. Pack this rank's spec m-columns into the cached send buffer (l, m, nr → flat).
    idx = 1
    @inbounds for k in 1:nr, jm in 1:m_local_cnt, il in 1:l_local
        send[idx] = complex(sr[il, jm, k], si[il, jm, k]); idx += 1
    end
    # 2. Allgatherv over θ_comm (cached vbuf) → every member gets all members' columns.
    MPI.Allgatherv!(send, vbuf, θ_comm)
    # 3. Reassemble into the cached full-m block full[(il, m+1, k)] for m = 0..mmax.
    fill!(full, zero(ComplexF64))
    base = 0
    @inbounds for src in 1:θ_size
        cnt = m_counts[src]
        m0  = m_firsts[src]
        for jm in 1:cnt, k in 1:nr, il in 1:l_local
            off = base + (k - 1) * (cnt * l_local) + (jm - 1) * l_local + il
            mslot = m0 + jm - 1
            mslot <= mmax + 1 || continue
            full[il, mslot, k] = recv[off]
        end
        base += cnt * l_local * nr
    end
    # 4. Scatter the plan's owned m-bins into solve.
    fill!(sp, zero(ComplexF64))
    @inbounds for (mi, m) in enumerate(mlocal)
        (0 <= m <= mmax) || continue
        for k in 1:nr, il in 1:l_local
            sp[il, k, mi] = full[il, m + 1, k]
        end
    end
    return nothing
end

"""
    solve_to_spec_storage!(cfg, sr, si, solve, plan)

Inverse of `spec_storage_to_solve!`: scatter the plan-oriented `solve`'s owned
m-bins back into the field's spectral storage parents `sr`/`si` (which use the
even-split (mmax+1) m-partition), performing the θ_comm m-axis redistribution.
"""
function solve_to_spec_storage!(cfg, sr, si, solve, plan)
    mb = _get_mbridge(cfg, plan)
    # Concrete-typed locals (caches are IdDict{Any,Any}) → type-stable kernel below.
    _solve_to_spec_kernel!(sr, si, mb.local_full::Array{ComplexF64, 3},
                           parent(solve)::AbstractArray{ComplexF64, 3}, mb.θ_comm,
                           plan.m_local::Vector{Int},
                           Int(first(mb.spec_m_range)), length(mb.spec_m_range)::Int,
                           mb.nr::Int, mb.l_local::Int, cfg.mmax::Int)
    return nothing
end

# Type-stable kernel (function barrier). `spec_m_range` is contiguous (PencilArrays
# local range) so global m-slot of local column jm = spec_m_first + jm - 1.
function _solve_to_spec_kernel!(sr, si, local_full::Array{ComplexF64, 3}, sp, θ_comm,
        mlocal::Vector{Int}, spec_m_first::Int, m_cnt::Int,
        nr::Int, l_local::Int, mmax::Int)
    # 1. This rank's contribution to the full-m block from its plan m-bins.
    fill!(local_full, zero(ComplexF64))
    @inbounds for (mi, m) in enumerate(mlocal)
        (0 <= m <= mmax) || continue
        for k in 1:nr, il in 1:l_local
            local_full[il, m + 1, k] = sp[il, k, mi]
        end
    end
    # 2. Sum-reduce over θ_comm (each m produced by exactly one member → SUM == concat).
    MPI.Allreduce!(local_full, +, θ_comm)
    # 3. Write this rank's spec m-columns (its even-split subset) from the full block.
    fill!(sr, zero(eltype(sr)))
    fill!(si, zero(eltype(si)))
    @inbounds for jm in 1:m_cnt
        mslot = spec_m_first + jm - 1          # global m-slot (= m+1)
        m = mslot - 1
        (0 <= m <= mmax) || continue
        for k in 1:nr, il in 1:l_local
            c = local_full[il, mslot, k]
            sr[il, jm, k] = real(c)
            si[il, jm, k] = (m == 0) ? zero(eltype(si)) : imag(c)
        end
    end
    return nothing
end
