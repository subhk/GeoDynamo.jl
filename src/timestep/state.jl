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

"""AB2 extrapolation weights for a step following an interval `previous_dt`."""
@inline function cnab2_weights(dt::Real, previous_dt::Real)
    isfinite(previous_dt) && previous_dt > 0 ||
        throw(ArgumentError("CNAB2 history timestep must be finite and positive"))
    half_ratio = 0.5 * (dt / previous_dt)
    return (1 + half_ratio, half_ratio)
end
