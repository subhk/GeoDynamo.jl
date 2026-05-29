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
const __MPI_INIT_LOCK = ReentrantLock()

"""
    get_comm()

Get MPI communicator, initializing MPI if needed.
Thread-safe via double-checked locking with `__MPI_INIT_LOCK`.
"""
function get_comm()
    MPI_STATE.initialized && return MPI_STATE.comm

    lock(__MPI_INIT_LOCK) do
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
