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
# Caching: both the DistTransposePlan and the scratch PencilArrays are stored
# directly on cfg._buffers (disttranspose_plan / disttranspose_scratch), a mutable
# SHTnsBuffers holder owned by the config.  Both are build-once.
# =============================================================================

using SHTnsKit
using PencilArrays
using MPI

# ---------------------------------------------------------------------------
# Module-level lock (shared by plan, scratch, and m-bridge builders)
# ---------------------------------------------------------------------------

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
on the first call.  Subsequent calls are O(1) field reads.

The plan is stored in `cfg._buffers.disttranspose_plan` (a mutable holder).
The plan is constructed on `cfg.pencils.θ_comm` with `nlev = nr_local`.
"""
function get_disttranspose_plan(cfg)
    b = cfg._buffers
    p = b.disttranspose_plan
    p !== nothing && return p
    lock(_DISTTRANSPOSE_LOCK) do
        if b.disttranspose_plan === nothing
            b.disttranspose_plan = _build_disttranspose_plan(cfg)
        end
        return b.disttranspose_plan
    end
end

"""
    _get_disttranspose_scratch(cfg, plan)

Return the cached scratch NamedTuple for `cfg`, building it on the first call.
The scratch is stored in `cfg._buffers.disttranspose_scratch` (a mutable field
on the config-owned `SHTnsBuffers`).  Subsequent calls are O(1) field reads.
"""
function _get_disttranspose_scratch(cfg, plan)
    b = cfg._buffers
    s = b.disttranspose_scratch
    s !== nothing && return s
    lock(_DISTTRANSPOSE_LOCK) do
        if b.disttranspose_scratch === nothing
            b.disttranspose_scratch = _build_disttranspose_scratch(cfg, plan)
        end
        return b.disttranspose_scratch
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
# of the ::Any-typed scratch on `cfg._buffers` / `Alm`; indexing them inline boxes
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
    # `scratch` is ::Any (cfg._buffers.disttranspose_scratch) and holds PencilArrays whose concrete type
    # is config-dependent. Hand it to a barrier so the NamedTuple field accesses
    # and the reorder loop specialize on the concrete runtime type instead of
    # boxing. (spec_storage_to_solve! uses the equivalent field-assert pattern.)
    return _to_spec_solve_impl!(scratch, Alm, cfg.lmax::Int,
        length(PencilArrays.range_local(cfg.pencils.r)[3]))
end

function _to_spec_solve_impl!(scratch, Alm, lmax::Int, nr_local::Int)
    Ap = parent(Alm)
    # Reorder Alm parent (l, m_bin, r_local) → almr parent (l, r_local, m_bin)
    _reorder_alm_to_almr!(parent(scratch.almr), Ap, nr_local, size(Ap, 2), lmax)
    PencilArrays.transpose!(scratch.t_fwd)  # almr → solve (persistent plan, no alloc)
    # Return the dynamically typed field while `scratch` is behind this
    # specialization barrier; accessing it in `to_spec_solve` boxes the result.
    return scratch.solve
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
# Two redistribution schemes are available, selected ONCE at build time by a
# volume gate in `_build_mbridge`:
#   * Allgatherv (default): gather the full m∈0..mmax columns onto every θ_comm
#     rank (one collective, batched over l_local × nr). Simple and latency-cheap,
#     but every rank receives the whole spectrum and keeps only its ~1/θ_size share.
#   * Alltoallv (θ_size ≥ 4 on dealiased grids): a personalized exchange that moves
#     only the columns each rank actually needs — the source (even-split) and
#     destination (plan-bin) m-partitions are both disjoint & covering, so each
#     m-column has exactly one owner on each side. This drops the replicated
#     receive volume and the full-m reassembly at higher rank counts.
# Either way only the small spectral m-axis crosses the wire; the heavy Legendre/
# FFT work stays θ-distributed inside dist_synthesis!/dist_analysis!.

# Concrete struct for per-config θ_comm m-redistribution metadata.
# Replacing the old NamedTuple + IdDict{Any,Any} with a typed struct gives
# type-stable field access in spec_storage_to_solve! without ::Type asserts.
struct _MBridge
    θ_comm       ::MPI.Comm
    θ_size       ::Int
    spec_m_range ::UnitRange{Int}
    nr           ::Int
    l_local      ::Int
    mmax         ::Int
    m_counts     ::Vector{Int}
    m_firsts     ::Vector{Int}
    send         ::Vector{ComplexF64}
    recv         ::Vector{ComplexF64}
    vbuf         ::MPI.VBuffer{Vector{ComplexF64}}
    full3        ::Array{ComplexF64, 3}
    local_full   ::Array{ComplexF64, 3}
    # Backward (solve→spec) plan-partition metadata + buffers. The inverse
    # redistribution Allgatherv's each member's plan-owned m-columns (mirroring
    # the forward) instead of Allreduce-ing a zero-padded full-m block.
    plan_valid_cols ::Vector{Int}      # this rank's solve-column indices (mi) with 0≤m≤mmax
    plan_send_cnt   ::Int              # == length(plan_valid_cols)
    plan_counts     ::Vector{Int}      # per-θ_comm-member valid plan-bin count
    plan_m_all      ::Vector{Int}      # concatenated per-member valid m values (0-based)
    plan_send       ::Vector{ComplexF64}
    plan_recv       ::Vector{ComplexF64}
    plan_vbuf       ::MPI.VBuffer{Vector{ComplexF64}}
    # --- Alltoallv (personalized-exchange) path, selected ONCE at build time ---
    # When `use_a2a`, spec_storage_to_solve!/solve_to_spec_storage! use
    # _fwd_a2a!/_bwd_a2a!, which move only the m-columns each rank actually needs
    # (grouped per peer) instead of Allgatherv replicating the whole valid
    # m-spectrum to every θ_comm member. When `use_a2a` is false the Allgatherv
    # fields above are used. The gate is a volume check (see `_build_mbridge`).
    use_a2a         ::Bool
    # Forward (spec→solve): this rank's spec columns grouped by their plan-owner.
    f_scnt          ::Vector{Int}       # send elements to each θ-rank (dest-major)
    f_rcnt          ::Vector{Int}       # recv elements from each θ-rank (src-major)
    f_cols          ::Vector{Int}       # dest-major list of local spec column jm (pack)
    f_mi            ::Vector{Int}       # src-major list of this rank's plan mi (scatter)
    f_sbuf          ::MPI.VBuffer{Vector{ComplexF64}}
    f_rbuf          ::MPI.VBuffer{Vector{ComplexF64}}
    # Backward (solve→spec): this rank's plan columns grouped by their spec-owner.
    b_scnt          ::Vector{Int}
    b_rcnt          ::Vector{Int}
    b_mi            ::Vector{Int}       # dest-major list of this rank's plan mi (pack)
    b_cols          ::Vector{Int}       # src-major list of local spec column jm (scatter)
    b_sbuf          ::MPI.VBuffer{Vector{ComplexF64}}
    b_rbuf          ::MPI.VBuffer{Vector{ComplexF64}}
end

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

    # --- Backward (solve→spec) plan-partition metadata --------------------------
    # Each θ_comm member's plan m-bins (DistTransposePlan split) truncated to
    # 0≤m≤mmax. The union over members is exactly {0..mmax} with no overlap (each
    # m is produced by exactly one member), so an Allgatherv of only the owned
    # columns reconstructs the full-m block — half the bytes of the prior
    # Allreduce over a zero-padded block, and no redundant summation.
    mlocal          = plan.m_local
    plan_valid_cols = Int[mi for (mi, m) in enumerate(mlocal) if 0 <= m <= mmax]
    plan_valid_m    = Int[mlocal[mi] for mi in plan_valid_cols]
    plan_send_cnt   = length(plan_valid_cols)
    plan_counts     = Int.(MPI.Allgather(Int32(plan_send_cnt), θ_comm))
    plan_m_recv     = Vector{Int32}(undef, sum(plan_counts))
    MPI.Allgatherv!(Int32.(plan_valid_m), MPI.VBuffer(plan_m_recv, plan_counts), θ_comm)
    plan_m_all      = Int.(plan_m_recv)
    plan_recvcounts = [c * l_local * nr for c in plan_counts]
    plan_send       = Vector{ComplexF64}(undef, l_local * plan_send_cnt * nr)
    plan_recv       = Vector{ComplexF64}(undef, sum(plan_recvcounts))
    plan_vbuf       = MPI.VBuffer(plan_recv, plan_recvcounts)

    # --- Alltoallv path metadata + volume gate ----------------------------------
    # Both partitions of the valid m-set {0..mmax} are known here: the spec side
    # (even split — m_counts/firsts) and the plan side (bin split — plan_m_all/
    # plan_counts). Each is disjoint & covering, so every m-column has exactly one
    # owner on each side. Group this rank's columns by the OTHER side's owner so an
    # Alltoallv moves only the columns actually exchanged (vs Allgatherv, which
    # replicates the full spectrum to every rank).
    m_firsts_i = Int.(firsts)
    plan_sets  = Vector{Set{Int}}(undef, θ_size)
    let c = 0
        for d in 1:θ_size
            plan_sets[d] = Set(plan_m_all[c + 1 : c + plan_counts[d]])
            c += plan_counts[d]
        end
    end
    # spec-owner d holds the contiguous 0-based m-range [firsts[d]-1, +m_counts[d]).
    spec_sets = [Set((m_firsts_i[d] - 1):(m_firsts_i[d] - 1 + m_counts[d] - 1)) for d in 1:θ_size]
    # this rank's local spec columns → 0-based m (spec local range is contiguous).
    my_spec_m = [first(spec_m_range) - 1 + jm - 1 for jm in 1:m_local_cnt]
    # this rank's valid plan columns (m, mi), ordered by increasing m to match the
    # spec-side pack order for every peer.
    my_plan_sorted = sort([(mlocal[mi], mi) for mi in plan_valid_cols])

    unit   = l_local * nr
    f_scnt = zeros(Int, θ_size); f_rcnt = zeros(Int, θ_size)
    f_cols = Int[]; f_mi = Int[]
    b_scnt = zeros(Int, θ_size); b_rcnt = zeros(Int, θ_size)
    b_mi   = Int[]; b_cols = Int[]
    for d in 1:θ_size
        # fwd send to plan-owner d: my spec columns whose m lies in d's plan set.
        cols = [jm for (jm, m) in enumerate(my_spec_m) if 0 <= m <= mmax && m in plan_sets[d]]
        append!(f_cols, cols); f_scnt[d] = length(cols) * unit
        # bwd send to spec-owner d: my plan columns whose m lies in d's spec set.
        mis = [mi for (m, mi) in my_plan_sorted if m in spec_sets[d]]
        append!(b_mi, mis);    b_scnt[d] = length(mis) * unit
    end
    for s in 1:θ_size
        # fwd recv from spec-owner s: my plan columns whose m lies in s's spec set.
        mis = [mi for (m, mi) in my_plan_sorted if m in spec_sets[s]]
        append!(f_mi, mis);    f_rcnt[s] = length(mis) * unit
        # bwd recv from plan-owner s: my spec columns whose m lies in s's plan set.
        cols = [jm for (jm, m) in enumerate(my_spec_m) if 0 <= m <= mmax && m in plan_sets[s]]
        append!(b_cols, cols); b_rcnt[s] = length(cols) * unit
    end
    f_s    = Vector{ComplexF64}(undef, sum(f_scnt)); f_r = Vector{ComplexF64}(undef, sum(f_rcnt))
    b_s    = Vector{ComplexF64}(undef, sum(b_scnt)); b_r = Vector{ComplexF64}(undef, sum(b_rcnt))
    f_sbuf = MPI.VBuffer(f_s, f_scnt); f_rbuf = MPI.VBuffer(f_r, f_rcnt)
    b_sbuf = MPI.VBuffer(b_s, b_scnt); b_rbuf = MPI.VBuffer(b_r, b_rcnt)

    # Volume gate — identical on every rank (plan_counts/m_counts come from Allgather).
    # Allgatherv delivers the FULL valid m-spectrum (mmax+1 columns) to every rank.
    # Alltoallv delivers to the busiest rank only its plan-owned valid columns
    # (max(plan_counts)), but pays the higher latency of a full personalized
    # exchange (θ_size-1 peers) in place of Allgather's log-structured schedule.
    # That saved bandwidth overcomes the latency penalty only when the busiest rank
    # moves less than ~two-thirds of the replicated spectrum. This threshold was
    # calibrated against measured ms/step: the busiest rank's share of the valid
    # spectrum is ≥0.74 at θ_size==2 (Allgatherv wins — the dealiased plan piles the
    # low-m bins onto rank 0) but ≤0.53 at θ_size==4 (Alltoallv wins 1.7–4.6×), so
    # `3·max < 2·(mmax+1)` (share < 2/3) sits cleanly in the gap and reduces to
    # θ_size ≥ 4 on the dealiased grids used here.
    # `GEODYNAMO_MBRIDGE_A2A` is a validation/benchmark-only override ("1" forces
    # Alltoallv, "0" forces Allgatherv); unset ⇒ the volume gate decides (production).
    gate    = θ_size > 1 && 3 * maximum(plan_counts) < 2 * (mmax + 1)
    use_a2a = haskey(ENV, "GEODYNAMO_MBRIDGE_A2A") ? (ENV["GEODYNAMO_MBRIDGE_A2A"] == "1") : gate
    # The forward/backward kernels below issue a collective (Alltoallv! vs
    # Allgatherv!) over θ_comm, so every rank in the communicator MUST pick the same
    # branch. The volume gate already agrees on all ranks (its inputs come from
    # Allgather), but the ENV override does not — set inconsistently it would deadlock.
    # Broadcast the decision from the root so the communicator can never disagree.
    use_a2a = MPI.bcast(use_a2a, θ_comm; root = 0)

    return _MBridge(θ_comm, θ_size,
                    spec_m_range, nr, l_local, mmax,
                    m_counts, Int.(firsts),
                    send, recv, vbuf, full3, local_full,
                    plan_valid_cols, plan_send_cnt, plan_counts, plan_m_all,
                    plan_send, plan_recv, plan_vbuf,
                    use_a2a,
                    f_scnt, f_rcnt, f_cols, f_mi, f_sbuf, f_rbuf,
                    b_scnt, b_rcnt, b_mi, b_cols, b_sbuf, b_rbuf)
end

function _get_mbridge(cfg, plan)
    b  = cfg._buffers
    mb = b.disttranspose_mbridge
    mb !== nothing && return mb::_MBridge
    lock(_DISTTRANSPOSE_LOCK) do
        if b.disttranspose_mbridge === nothing
            b.disttranspose_mbridge = _build_mbridge(cfg, plan)
        end
        return b.disttranspose_mbridge::_MBridge
    end
end

# Alltoallv forward kernel (spec→solve). Groups this rank's spec columns by their
# plan-owner, does one personalized exchange, and writes received columns straight
# into `sp[il, k, mi]` — no full-m block, no replicate-to-all. All metadata is
# prebuilt in `_build_mbridge`; the send/recv layout is (il fastest, column, k) per
# peer block so pack and scatter agree. Selected only when `mb.use_a2a` (volume gate).
function _fwd_a2a!(sp, mb::_MBridge, sr, si)
    l_local = mb.l_local; nr = mb.nr; unit = l_local * nr
    s = mb.f_sbuf.data::Vector{ComplexF64}
    r = mb.f_rbuf.data::Vector{ComplexF64}
    f_scnt = mb.f_scnt; f_rcnt = mb.f_rcnt; f_cols = mb.f_cols; f_mi = mb.f_mi
    # 1. Pack dest-major: for each plan-owner d, my spec columns bound for d.
    idx = 1; c0 = 0
    @inbounds for d in 1:mb.θ_size
        n = f_scnt[d] ÷ unit
        for k in 1:nr, j in 1:n, il in 1:l_local
            jm = f_cols[c0 + j]
            s[idx] = complex(sr[il, jm, k], si[il, jm, k]); idx += 1
        end
        c0 += n
    end
    # 2. Personalized exchange over θ_comm (cached VBuffers).
    MPI.Alltoallv!(mb.f_sbuf, mb.f_rbuf, mb.θ_comm)
    # 3. Scatter received columns straight into the plan-oriented solve.
    fill!(sp, zero(ComplexF64))
    base = 0; c0 = 0
    @inbounds for src in 1:mb.θ_size
        n = f_rcnt[src] ÷ unit
        for j in 1:n
            mi = f_mi[c0 + j]
            for k in 1:nr, il in 1:l_local
                sp[il, k, mi] = r[base + (k - 1) * (n * l_local) + (j - 1) * l_local + il]
            end
        end
        base += f_rcnt[src]; c0 += n
    end
    return nothing
end

# Alltoallv backward kernel (solve→spec) — mirror of _fwd_a2a!. Groups this rank's
# plan-owned columns by their even-split spec-owner, exchanges, and writes straight
# into `sr`/`si` (dropping the zero-padded full-m block). Imag part of m==0 is pinned
# to zero, matching the Allgatherv path.
function _bwd_a2a!(sr, si, mb::_MBridge, sp)
    l_local = mb.l_local; nr = mb.nr; unit = l_local * nr
    s = mb.b_sbuf.data::Vector{ComplexF64}
    r = mb.b_rbuf.data::Vector{ComplexF64}
    b_scnt = mb.b_scnt; b_rcnt = mb.b_rcnt; b_mi = mb.b_mi; b_cols = mb.b_cols
    # 1. Pack dest-major: for each spec-owner d, my plan columns bound for d.
    idx = 1; c0 = 0
    @inbounds for d in 1:mb.θ_size
        n = b_scnt[d] ÷ unit
        for k in 1:nr, j in 1:n, il in 1:l_local
            mi = b_mi[c0 + j]
            s[idx] = sp[il, k, mi]; idx += 1
        end
        c0 += n
    end
    # 2. Personalized exchange over θ_comm.
    MPI.Alltoallv!(mb.b_sbuf, mb.b_rbuf, mb.θ_comm)
    # 3. Write received columns into this rank's even-split spec storage.
    fill!(sr, zero(eltype(sr))); fill!(si, zero(eltype(si)))
    spec_m_first = Int(first(mb.spec_m_range))
    base = 0; c0 = 0
    @inbounds for src in 1:mb.θ_size
        n = b_rcnt[src] ÷ unit
        for j in 1:n
            jm = b_cols[c0 + j]
            m  = spec_m_first + jm - 2            # 0-based m of local spec column jm
            for k in 1:nr, il in 1:l_local
                c = r[base + (k - 1) * (n * l_local) + (j - 1) * l_local + il]
                sr[il, jm, k] = real(c)
                si[il, jm, k] = (m == 0) ? zero(eltype(si)) : imag(c)
            end
        end
        base += b_rcnt[src]; c0 += n
    end
    return nothing
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
    # Volume gate chose the Alltoallv path at setup (θ_size ≥ 4 on dealiased grids).
    if mb.use_a2a
        _fwd_a2a!(parent(solve)::AbstractArray{ComplexF64, 3}, mb, sr, si)
        return solve
    end
    # Type-assert the cached fields / plan / cfg to concrete types so the
    # hot loops below specialize and DON'T box (mb is a concrete _MBridge; the
    # asserts are now redundant but harmless and document the kernel's expected types).
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
    # Volume gate chose the Alltoallv path at setup (mirrors spec_storage_to_solve!).
    if mb.use_a2a
        _bwd_a2a!(sr, si, mb, parent(solve)::AbstractArray{ComplexF64, 3})
        return nothing
    end
    # Concrete-typed locals (_get_mbridge returns a concrete _MBridge) → type-stable kernel below.
    _solve_to_spec_kernel!(sr, si, mb.local_full::Array{ComplexF64, 3},
                           parent(solve)::AbstractArray{ComplexF64, 3}, mb.θ_comm,
                           mb.plan_send::Vector{ComplexF64}, mb.plan_recv::Vector{ComplexF64},
                           mb.plan_vbuf,
                           mb.plan_valid_cols::Vector{Int}, mb.plan_send_cnt::Int,
                           mb.plan_counts::Vector{Int}, mb.plan_m_all::Vector{Int}, mb.θ_size::Int,
                           Int(first(mb.spec_m_range)), length(mb.spec_m_range)::Int,
                           mb.nr::Int, mb.l_local::Int, cfg.mmax::Int)
    return nothing
end

# Type-stable kernel (function barrier). `spec_m_range` is contiguous (PencilArrays
# local range) so global m-slot of local column jm = spec_m_first + jm - 1.
#
# The redistribution mirrors the forward `_spec_to_solve_kernel!`: pack only this
# rank's plan-owned m-columns, Allgatherv them to every θ_comm member, reassemble
# the full-m block, then write this rank's even-split spec columns. The earlier
# implementation Allreduce'd a zero-padded full-m block (~2× the bytes, plus a
# redundant sum over the padding); each m is produced by exactly one member, so a
# gather of owned columns is the exact, cheaper equivalent.
function _solve_to_spec_kernel!(sr, si, full::Array{ComplexF64, 3}, sp, θ_comm,
        plan_send::Vector{ComplexF64}, plan_recv::Vector{ComplexF64}, plan_vbuf,
        plan_valid_cols::Vector{Int}, plan_send_cnt::Int,
        plan_counts::Vector{Int}, plan_m_all::Vector{Int}, θ_size::Int,
        spec_m_first::Int, m_cnt::Int, nr::Int, l_local::Int, mmax::Int)
    # 1. Pack this rank's plan-owned m-columns (valid 0≤m≤mmax) into the cached
    #    send buffer, in the same (il fastest, jm, k) layout the forward uses.
    idx = 1
    @inbounds for k in 1:nr, jm in 1:plan_send_cnt, il in 1:l_local
        mi = plan_valid_cols[jm]
        plan_send[idx] = sp[il, k, mi]; idx += 1
    end
    # 2. Allgatherv over θ_comm (cached vbuf): every member receives all members'
    #    owned plan m-columns — no zero-padded block, no redundant summation.
    MPI.Allgatherv!(plan_send, plan_vbuf, θ_comm)
    # 3. Reassemble recv into the full-m block full[(il, m+1, k)] using the
    #    per-source plan-m mapping (each m∈0..mmax produced by exactly one member).
    fill!(full, zero(ComplexF64))
    base = 0       # element offset into recv for the current source block
    col  = 0       # running index into plan_m_all
    @inbounds for src in 1:θ_size
        cnt = plan_counts[src]
        for jm in 1:cnt
            m = plan_m_all[col + jm]
            (0 <= m <= mmax) || continue
            mslot = m + 1
            for k in 1:nr, il in 1:l_local
                off = base + (k - 1) * (cnt * l_local) + (jm - 1) * l_local + il
                full[il, mslot, k] = plan_recv[off]
            end
        end
        base += cnt * l_local * nr
        col  += cnt
    end
    # 4. Write this rank's spec m-columns (its even-split subset) from the full block.
    fill!(sr, zero(eltype(sr)))
    fill!(si, zero(eltype(si)))
    @inbounds for jm in 1:m_cnt
        mslot = spec_m_first + jm - 1          # global m-slot (= m+1)
        m = mslot - 1
        (0 <= m <= mmax) || continue
        for k in 1:nr, il in 1:l_local
            c = full[il, mslot, k]
            sr[il, jm, k] = real(c)
            si[il, jm, k] = (m == 0) ? zero(eltype(si)) : imag(c)
        end
    end
    return nothing
end
