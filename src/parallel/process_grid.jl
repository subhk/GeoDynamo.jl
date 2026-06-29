"""
    parse_proc_grid(spec::Union{AbstractString,Nothing}, nprocs::Int) -> (θ_ranks, r_ranks)

Parse an explicit process grid "θxr" (e.g. "4x2"). At nprocs==1 returns (1,1) without
requiring `spec`. At nprocs>1 `spec` is REQUIRED and must satisfy θ_ranks·r_ranks==nprocs.
"""
function parse_proc_grid(spec::Union{AbstractString,Nothing}, nprocs::Int)
    nprocs == 1 && return (1, 1)
    spec === nothing && error("GEODYNAMO_PROC_GRID must be set at nprocs>1 (e.g. \"4x2\" = θ_ranks×r_ranks)")
    parts = split(spec, 'x')
    length(parts) == 2 || error("GEODYNAMO_PROC_GRID must be \"θxr\" (e.g. \"4x2\"), got \"$spec\"")
    θr = parse(Int, parts[1]); rr = parse(Int, parts[2])
    θr * rr == nprocs || error("GEODYNAMO_PROC_GRID $spec = $(θr*rr) ranks != nprocs=$nprocs")
    return (θr, rr)
end

"""
    read_proc_grid(nprocs::Int) -> (θ_ranks, r_ranks)

Read `GEODYNAMO_PROC_GRID` from the environment and parse it for `nprocs` ranks.
"""
read_proc_grid(nprocs::Int) = parse_proc_grid(get(ENV, "GEODYNAMO_PROC_GRID", nothing), nprocs)

# Size of the i-th (0-based) block when D items are split into P balanced contiguous
# blocks (sizes differ by at most 1) — matches PencilArrays' default axis split.
@inline _balanced_block_size(D::Int, P::Int, i::Int) = (b = D ÷ P; r = D % P; b + (i < r ? 1 : 0))
@inline _balanced_block_offset(D::Int, P::Int, i::Int) =
    sum(j -> _balanced_block_size(D, P, j), 0:(i - 1); init = 0)

"""
    spectral_mode_counts(θ_ranks, r_ranks, lmax, mmax) -> Vector{Int}

Number of spherical-harmonic modes each rank owns under the spectral pencil's
contiguous-block split (m over `θ_ranks`, l over `r_ranks`). Because of triangular
truncation (valid modes need `m ≤ l`), equal SLOT blocks give unequal MODE counts,
and a square grid can leave a rank with ZERO modes (high-m / low-l corner). Used to
warn about idle ranks and load imbalance at grid setup. Pure function (no MPI).
"""
function spectral_mode_counts(θ_ranks::Int, r_ranks::Int, lmax::Int, mmax::Int)
    counts = Int[]
    for iθ in 0:(θ_ranks - 1)
        m_sz = _balanced_block_size(mmax + 1, θ_ranks, iθ)
        m_off = _balanced_block_offset(mmax + 1, θ_ranks, iθ)   # m-slot offset (m = slot)
        for ir in 0:(r_ranks - 1)
            l_sz = _balanced_block_size(lmax + 1, r_ranks, ir)
            l_off = _balanced_block_offset(lmax + 1, r_ranks, ir)
            c = 0
            for ms in 0:(m_sz - 1)
                m = m_off + ms
                for ls in 0:(l_sz - 1)
                    l = l_off + ls
                    (0 <= m <= l <= lmax && m <= mmax) && (c += 1)
                end
            end
            push!(counts, c)
        end
    end
    return counts
end

"""
    validate_proc_grid(θ_ranks, r_ranks; nlat, nr, lmax, mmax)

Warn (once, on rank 0) about a process grid that over-decomposes an axis (idle
ranks) or yields a large spherical-harmonic mode-load imbalance. Purely advisory —
does not change the decomposition.
"""
function validate_proc_grid(θ_ranks::Int, r_ranks::Int; nlat::Int, nr::Int,
        lmax::Int, mmax::Int)
    (θ_ranks <= 1 && r_ranks <= 1) && return nothing
    get_rank() == 0 || return nothing

    msgs = String[]
    θ_ranks > nlat     && push!(msgs, "θ_ranks=$θ_ranks > nlat=$nlat ⇒ ranks with no latitudes")
    θ_ranks > mmax + 1 && push!(msgs, "θ_ranks=$θ_ranks > mmax+1=$(mmax + 1) ⇒ ranks with no m-modes")
    r_ranks > nr       && push!(msgs, "r_ranks=$r_ranks > nr=$nr ⇒ ranks with no radial levels")
    r_ranks > lmax + 1 && push!(msgs, "r_ranks=$r_ranks > lmax+1=$(lmax + 1) ⇒ ranks with no l-slots")

    counts = spectral_mode_counts(θ_ranks, r_ranks, lmax, mmax)
    if !isempty(counts)
        nzero = count(==(0), counts)
        if nzero > 0
            push!(msgs, "$nzero of $(length(counts)) ranks own ZERO spectral modes " *
                        "(idle in radial solves and the m-distributed transform). Prefer a grid " *
                        "without empty m>l blocks — e.g. $(θ_ranks * r_ranks)x1 over a square grid.")
        else
            mx = maximum(counts); av = sum(counts) / length(counts)
            av > 0 && mx / av > 1.5 && push!(msgs,
                "spectral mode-load imbalance max/avg=$(round(mx / av, digits = 2)) " *
                "(triangular truncation + contiguous blocks); the heaviest rank gates each step")
        end
    end

    isempty(msgs) ||
        @warn "Process grid $(θ_ranks)×$(r_ranks) decomposition warnings:\n - " * join(msgs, "\n - ")
    return nothing
end

"""
    make_subcomms(comm, pencil_r) -> (θ_transform_comm, r_transpose_comm)

Split `comm` into the two sub-communicators the r×θ decomposition needs, deriving the
split colors from the ACTUAL distribution of `pencil_r` (θ-dist / φ-local / r-dist) so
the result is correct for ANY process grid and any PencilArrays rank ordering — NOT
from an assumed `rank = f(θ_ranks, r_ranks)` formula (that only holds when
θ_ranks==r_ranks).

- `θ_transform_comm`: ranks that share the SAME r-slab and SPLIT θ — the group over
  which the SH transform distributes θ (so `theta_phys`/`dist_*` run here, and the
  per-level θ-mode gather reduces here). Color = this rank's first owned r index.
- `r_transpose_comm`: ranks that share the SAME θ-slab and SPLIT r — the group aligned
  with the r↔lm transpose's radial redistribution. Color = first owned θ index.
"""
function make_subcomms(comm, pencil_r)
    rank = MPI.Comm_rank(comm)
    lr = PencilArrays.range_local(pencil_r)   # (θ_range, φ_range, r_range)
    θ_lo = Int(first(lr[1]))                  # identifies this rank's θ-slab
    r_lo = Int(first(lr[3]))                  # identifies this rank's r-slab
    θ_transform_comm = MPI.Comm_split(comm, r_lo, rank)   # share r-slab, split θ
    r_transpose_comm = MPI.Comm_split(comm, θ_lo, rank)   # share θ-slab, split r
    # NOTE on lifetime: these sub-communicators live for the duration of the grid and
    # are intentionally NOT freed here. They remain in use across the whole run (every
    # transform/transpose), so calling MPI.Comm_free on them would free a comm still in
    # use; rely on MPI finalization to reclaim them instead.
    return θ_transform_comm, r_transpose_comm
end
