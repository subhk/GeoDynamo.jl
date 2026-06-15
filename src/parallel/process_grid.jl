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
