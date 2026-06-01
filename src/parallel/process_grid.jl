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
    make_subcomms(comm, θ_ranks::Int, r_ranks::Int) -> (θ_comm, r_comm)

Split `comm` (row-major rank = r_group·θ_ranks + θ_index) into the θ-subcomm (ranks
sharing an r-slab) and the r-subcomm (ranks sharing a θ-column).
"""
function make_subcomms(comm, θ_ranks::Int, r_ranks::Int)
    rank = MPI.Comm_rank(comm)
    r_group = rank ÷ θ_ranks
    θ_index = rank % θ_ranks
    θ_comm = MPI.Comm_split(comm, r_group, rank)
    r_comm = MPI.Comm_split(comm, θ_index, rank)
    return θ_comm, r_comm
end
