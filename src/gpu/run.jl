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

"""
    gpu_run!(cpu_state::SolverState, nsteps; arch = CPU(),
             output_every = 0, output_fn = nothing) -> cpu_state

Public-API convenience: advance a configured CPU `SolverState` by `nsteps` CNAB2 steps
ON THE GPU PATH.  Builds the device state via [`build_gpu_solver_state`](@ref) (optionally
moved to `arch` with [`gpu_to_device`](@ref) — pass `arch = GPU()` on a CUDA box), runs the
device loop, then ALWAYS syncs the evolved state back into `cpu_state` via
[`sync_gpu_state_to_cpu!`](@ref) (spectral fields, CNAB2 `prev_nl` histories, and the lagged
physical buffers) and advances both clocks — `cpu_state.step`/`.time` and the
`runtime.timestep_state` pair the boundary-condition and diagnostic layers read — so
CPU-side stepping / diagnostics / output / restart can continue coherently from the
GPU-evolved state.  (To run the device loop without
syncing back — keeping a handle to the device state — call `build_gpu_solver_state` +
`gpu_run!(gst, …)` directly instead.)

`cpu_state` should be initialized (≥1 prior `solver_step!`, or `initialize_solver_fields!`) so
the CNAB2 history (`prev_nl`) and the lagged physical buffers are populated, exactly as the CPU
path requires.  Builder scope is insulating magnetic + CNAB2 (see `build_gpu_solver_state`).
"""
function gpu_run!(cpu_state::SolverState, nsteps::Int; arch::AbstractArchitecture = CPU(),
        output_every::Int = 0, output_fn = nothing)
    nsteps >= 0 || throw(ArgumentError("gpu_run!: nsteps must be ≥ 0, got $nsteps"))
    gst = build_gpu_solver_state(cpu_state)
    arch isa CPU || (gst = gpu_to_device(gst, arch))
    gpu_run!(gst, nsteps; output_every = output_every, output_fn = output_fn)
    sync_gpu_state_to_cpu!(cpu_state, gst)
    # Advance the runtime clock alongside the public one; downstream readers
    # (get_current_simulation_time, ERK2 diagnostics) go through
    # `runtime.timestep_state`, not `cpu_state.step`/`.time`.
    reset_solver_clock!(cpu_state;
        time = cpu_state.time + nsteps * cpu_state.parameters.timestep,
        step = cpu_state.step + nsteps)
    return cpu_state
end
