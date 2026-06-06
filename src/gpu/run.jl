# =============================================================================
# GPU Phase 6 — device-resident time loop driving gpu_solver_step!, plus the
# IO host-gather primitive. gpu_run! advances the device state `nsteps` CNAB2
# steps in place (the per-field prev_nl rollover and the lagged physical buffers
# carry across steps inside gpu_solver_step!, so the loop is a plain repeat).
# For output/restart, gpu_to_device(state, CPU()) gathers the (possibly device)
# state back to host dense Arrays — handed to an optional `output_fn` snapshot
# callback every `output_every` steps. No new kernels. Runs on Array + CuArray.
# =============================================================================

"""
    gpu_run!(state, nsteps; output_every = 0, output_fn = nothing) -> state

Advance the device solver `state` (from [`build_gpu_solver_state`](@ref), optionally
moved to a device via [`gpu_to_device`](@ref)) by `nsteps` CNAB2 steps, in place.

Each iteration calls [`gpu_solver_step!`](@ref); the per-field `prev_nl` rollover and
the persistent lagged physical buffers are mutated in place inside the step, so the
across-step history evolves correctly without extra bookkeeping here.

If `output_fn !== nothing` and `output_every > 0`, then after every `output_every`-th
step `output_fn(host_state, step)` is called, where `host_state = gpu_to_device(state,
CPU())` is a host-gathered deep copy (device → host dense `Array`s) suitable for IO /
restart writing without disturbing the live (possibly device-resident) `state`.
"""
function gpu_run!(state, nsteps::Int; output_every::Int = 0, output_fn = nothing)
    nsteps >= 0 || throw(ArgumentError("gpu_run!: nsteps must be ≥ 0, got $nsteps"))
    for step in 1:nsteps
        gpu_solver_step!(state)
        if output_fn !== nothing && output_every > 0 && step % output_every == 0
            output_fn(gpu_to_device(state, CPU()), step)
        end
    end
    return state
end
