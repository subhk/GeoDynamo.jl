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
