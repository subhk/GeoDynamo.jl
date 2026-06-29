# ================================================================================
# MPI Runtime for Parallel Backend
# ================================================================================

# Global MPI state management
mutable struct MPIState
    initialized::Bool
    comm::Union{Nothing, MPI.Comm}
    rank::Int
    nprocs::Int
end

# Global MPI state (initialized lazily)
const MPI_STATE = MPIState(false, nothing, -1, -1)
const _MPI_INIT_LOCK = ReentrantLock()

"""
    get_comm()

Get MPI communicator, initializing MPI if needed.
Thread-safe via double-checked locking with `_MPI_INIT_LOCK`.
"""
function get_comm()
    MPI_STATE.initialized && return MPI_STATE.comm

    lock(_MPI_INIT_LOCK) do
        if !MPI_STATE.initialized
            if !MPI.Initialized()
                MPI.Init()
            end
            MPI_STATE.comm = MPI.COMM_WORLD
            MPI_STATE.rank = MPI.Comm_rank(MPI_STATE.comm)
            MPI_STATE.nprocs = MPI.Comm_size(MPI_STATE.comm)
            MPI_STATE.initialized = true
        end
    end
    return MPI_STATE.comm
end

"""
    get_rank()

Get MPI rank of current process.
"""
function get_rank()
    get_comm()
    return MPI_STATE.rank
end

"""
    get_nprocs()

Get total number of MPI processes.
"""
function get_nprocs()
    get_comm()
    return MPI_STATE.nprocs
end

"""
    rank_seed(seed, rank = get_rank())

Per-rank RNG seed offset for random initial conditions. Each MPI rank fills a
DISJOINT subset of spectral modes by stepping a shared `rand()` stream, so a
common seed gave the k-th owned mode the SAME random draw on every rank
(spurious cross-rank correlation). Offsetting by `rank` decorrelates the ranks.

Rank 0 keeps the seed unchanged, so single-rank (nprocs==1) runs are bit-for-bit
identical to before. `nothing` passes through (caller skips seeding and uses the
per-process default RNG). NOTE: the resulting IC depends on the rank count
(disjoint fill from per-rank streams) — it is reproducible for a fixed
(seed, rank-count), not across different rank counts.
"""
rank_seed(::Nothing, rank::Integer = 0) = nothing
rank_seed(seed::Integer, rank::Integer = get_rank()) = seed + rank
