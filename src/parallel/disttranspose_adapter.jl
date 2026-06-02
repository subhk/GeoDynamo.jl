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

Returns `(; pen_alm_r, pen_solve_r, almr, solve)`.
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

    return (; pen_alm_r, pen_solve_r, almr, solve)
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
    lock(_DISTTRANSPOSE_LOCK) do
        get!(_DISTTRANSPOSE_PLAN_CACHE, cfg) do
            _build_disttranspose_plan(cfg)
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
function to_spec_solve(cfg, Alm, plan)
    scratch = lock(_DISTTRANSPOSE_LOCK) do
        get!(_DISTTRANSPOSE_SCRATCH_CACHE, cfg) do
            _build_disttranspose_scratch(cfg, plan)
        end
    end

    lmax     = cfg.lmax
    nr_local = length(PencilArrays.range_local(cfg.pencils.r)[3])
    nml      = size(parent(Alm), 2)

    # Reorder Alm parent (l, m_bin, r_local) → almr parent (l, r_local, m_bin)
    Ap  = parent(Alm)
    pa  = parent(scratch.almr)
    fill!(pa, zero(ComplexF64))
    @inbounds for k in 1:nr_local, jm in 1:nml, il in 1:(lmax + 1)
        pa[il, k, jm] = Ap[il, jm, k]
    end

    PencilArrays.transpose!(scratch.solve, scratch.almr)
    return scratch.solve
end

"""
    from_spec_solve!(cfg, Alm, solve, plan)

Transpose the solver-orientation `PencilArray` `solve` (output of the radial
implicit step) back into `Alm` (the DistTransposePlan layout).  Overwrites
`Alm` in-place.
"""
function from_spec_solve!(cfg, Alm, solve, plan)
    scratch = lock(_DISTTRANSPOSE_LOCK) do
        get!(_DISTTRANSPOSE_SCRATCH_CACHE, cfg) do
            _build_disttranspose_scratch(cfg, plan)
        end
    end

    lmax     = cfg.lmax
    nr_local = length(PencilArrays.range_local(cfg.pencils.r)[3])
    nml      = size(parent(Alm), 2)

    # Transpose back into the (l, r_local, m_bin) layout
    PencilArrays.transpose!(scratch.almr, solve)

    # Reorder almr parent (l, r_local, m_bin) → Alm parent (l, m_bin, r_local)
    Ap  = parent(Alm)
    pa  = parent(scratch.almr)
    @inbounds for k in 1:nr_local, jm in 1:nml, il in 1:(lmax + 1)
        Ap[il, jm, k] = pa[il, k, jm]
    end
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

    return (; θ_comm, θ_size,
              spec_m_range, spec_l_range, nr, l_local,
              m_counts = Int.(counts), m_firsts = Int.(firsts))
end

function _get_mbridge(cfg, plan)
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
    θ_comm = mb.θ_comm
    mmax   = cfg.mmax
    nr     = mb.nr
    l_local = mb.l_local

    # 1. Pack this rank's spec m-columns into a contiguous complex send buffer
    #    laid out (l_local, m_local, nr) → flat.
    m_local_cnt = length(mb.spec_m_range)
    send = Vector{ComplexF64}(undef, l_local * m_local_cnt * nr)
    idx = 1
    @inbounds for k in 1:nr, jm in 1:m_local_cnt, il in 1:l_local
        send[idx] = complex(sr[il, jm, k], si[il, jm, k]); idx += 1
    end

    # 2. Allgatherv over θ_comm → every member receives all members' columns.
    recvcounts = [c * l_local * nr for c in mb.m_counts]
    recv = Vector{ComplexF64}(undef, sum(recvcounts))
    vbuf = MPI.VBuffer(recv, recvcounts)
    MPI.Allgatherv!(send, vbuf, θ_comm)

    # 3. Reassemble into a full-m block full[(il, m+1, k)] for m = 0..mmax.
    full = zeros(ComplexF64, l_local, mmax + 1, nr)
    base = 0
    @inbounds for src in 1:mb.θ_size
        cnt = mb.m_counts[src]
        m0  = mb.m_firsts[src]              # 1-based first m-slot of source
        for jm in 1:cnt, k in 1:nr, il in 1:l_local
            # source send order was (il fastest, then jm, then k)
            off = base + (k - 1) * (cnt * l_local) + (jm - 1) * l_local + il
            mslot = m0 + jm - 1            # 1-based global m-slot (= m+1)
            mslot <= mmax + 1 || continue
            full[il, mslot, k] = recv[off]
        end
        base += cnt * l_local * nr
    end

    # 4. Scatter the plan's owned m-bins into solve.
    sp = parent(solve)
    fill!(sp, zero(ComplexF64))
    @inbounds for (mi, m) in enumerate(plan.m_local)
        (0 <= m <= mmax) || continue
        for k in 1:nr, il in 1:l_local
            sp[il, k, mi] = full[il, m + 1, k]
        end
    end
    return solve
end

"""
    solve_to_spec_storage!(cfg, sr, si, solve, plan)

Inverse of `spec_storage_to_solve!`: scatter the plan-oriented `solve`'s owned
m-bins back into the field's spectral storage parents `sr`/`si` (which use the
even-split (mmax+1) m-partition), performing the θ_comm m-axis redistribution.
"""
function solve_to_spec_storage!(cfg, sr, si, solve, plan)
    mb = _get_mbridge(cfg, plan)
    θ_comm = mb.θ_comm
    mmax   = cfg.mmax
    nr     = mb.nr
    l_local = mb.l_local

    # 1. Build this rank's contribution to the full-m block from its plan m-bins.
    local_full = zeros(ComplexF64, l_local, mmax + 1, nr)
    sp = parent(solve)
    @inbounds for (mi, m) in enumerate(plan.m_local)
        (0 <= m <= mmax) || continue
        for k in 1:nr, il in 1:l_local
            local_full[il, m + 1, k] = sp[il, k, mi]
        end
    end

    # 2. Sum-reduce over θ_comm so every member has the complete full-m block (each
    #    m is produced by exactly one member, so SUM == concatenation).
    MPI.Allreduce!(local_full, +, θ_comm)

    # 3. Write this rank's spec m-columns (its even-split subset) from the full block.
    fill!(sr, zero(eltype(sr)))
    fill!(si, zero(eltype(si)))
    @inbounds for (jm, mslot) in enumerate(mb.spec_m_range)  # mslot = m+1 (1-based)
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
