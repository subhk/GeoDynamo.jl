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

Check a process grid at setup.

**Fatal**: a grid that over-decomposes an axis past its global size, which hands some
rank an EMPTY local range. That is not merely wasteful — it DEADLOCKS. Reproduced at
4 ranks with `GEODYNAMO_PROC_GRID=4x1` on `lmax=mmax=1, nlat=4, nlon=8, nr=8`: the
spectral pencil came back as `axes_local=(1:2, 1:0, 1:8)` on rank 0 and
`(1:2, 2:1, 1:8)` on rank 2, `time_step!` completed on every rank, and `run!` never
returned — no error, no output, just a launcher timeout. The same grid runs to
completion at 1 and 2 ranks. Refusing it converts a silent hang into a legible abort.

The four fatal conditions are decided from values that are identical on every rank
(the grid comes from `GEODYNAMO_PROC_GRID`, the sizes from the shared config), so
every rank throws together and no collective is needed to agree. Deliberately NOT
behind the `rank == 0` guard below: a rank-0-only throw would leave every other rank
blocking in the next collective — the failure mode this check exists to remove.

**Advisory** (warned once, on rank 0): mode-load imbalance, and ranks that own zero
spherical-harmonic MODES. A zero-mode rank is not a zero-slot rank — a 2×2 grid at
`lmax=mmax=4` leaves the high-m/low-l corner with no valid `m ≤ l` pair and still
reproduces the serial result bit-exactly in the MPI gates — so it stays a warning.
"""
function validate_proc_grid(θ_ranks::Int, r_ranks::Int; nlat::Int, nr::Int,
        lmax::Int, mmax::Int)
    (θ_ranks <= 1 && r_ranks <= 1) && return nothing

    fatal = String[]
    θ_ranks > nlat     && push!(fatal, "θ_ranks=$θ_ranks > nlat=$nlat ⇒ ranks with no latitudes")
    θ_ranks > mmax + 1 && push!(fatal, "θ_ranks=$θ_ranks > mmax+1=$(mmax + 1) ⇒ ranks with no m-modes")
    r_ranks > nr       && push!(fatal, "r_ranks=$r_ranks > nr=$nr ⇒ ranks with no radial levels")
    r_ranks > lmax + 1 && push!(fatal, "r_ranks=$r_ranks > lmax+1=$(lmax + 1) ⇒ ranks with no l-slots")
    isempty(fatal) || throw(ArgumentError(
        "Process grid $(θ_ranks)×$(r_ranks) (θ×r) leaves ranks with an empty local " *
        "range, which deadlocks the run:\n - " * join(fatal, "\n - ") *
        "\nChoose a grid whose factors fit every axis (set GEODYNAMO_PROC_GRID), or " *
        "run with fewer ranks."))

    get_rank() == 0 || return nothing

    msgs = String[]
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
