# ================================================================================
# Public Timestep State
# ================================================================================

"""
    TimestepState

Legacy-compatible timestep bookkeeping state.

This small mutable container tracks the current simulation clock, timestep,
step counters, and simple convergence/error flags. It remains part of the
public API because restart/output helpers and older workflows still use it even
though the rewritten solver also exposes `SolverTimestepState`.
"""
mutable struct TimestepState
    time::Float64
    dt::Float64
    step::Int
    iteration::Int
    error::Float64
    converged::Bool
    needs_ab2_bootstrap::Bool
end
