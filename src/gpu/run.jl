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

Each iteration calls `step!` — [`gpu_solver_step!`](@ref) (CNAB2) by default; the
`SolverState` overload below passes the device step matching
`parameters.timestepper`. The per-field `prev_nl` rollover and
the persistent lagged physical buffers are mutated in place inside the step, so the
across-step history evolves correctly without extra bookkeeping here.

If `output_fn !== nothing` and `output_every > 0`, then after every `output_every`-th
step `output_fn(host_state, step)` is called, where `host_state = gpu_to_device(state,
CPU())` is a host-gathered deep copy (device → host dense `Array`s) suitable for IO /
restart writing without disturbing the live (possibly device-resident) `state`.
"""
function gpu_run!(state, nsteps::Int; output_every::Int = 0, output_fn = nothing,
        step! = gpu_solver_step!)
    nsteps >= 0 || throw(ArgumentError("gpu_run!: nsteps must be ≥ 0, got $nsteps"))
    for step in 1:nsteps
        step!(state)
        if output_fn !== nothing && output_every > 0 && step % output_every == 0
            output_fn(gpu_to_device(state, CPU()), step)
        end
    end
    return state
end

"""
    _gpu_device_step(timestepper, erk) -> step function

The device step matching `timestepper`, for the `gpu_run!` loop: the same three-way
choice `_gpu_time_step!` (api/simulation.jl) makes. `erk` is the
[`build_gpu_erk2_state`](@ref) pack (only consulted for ExponentialRungeKutta2).

Written as dispatch with an erroring fallback so a timestepper added later fails loudly
here instead of falling through to the CNAB2 step — which is exactly how `gpu_run!` came
to integrate ERK2 and RungeKutta3 configurations as CNAB2 without a word.
"""
_gpu_device_step(::CNAB2, erk) = gpu_solver_step!
_gpu_device_step(::RungeKutta3, erk) = gpu_cb3_solver_step!
_gpu_device_step(::ExponentialRungeKutta2, erk) = st -> gpu_erk2_solver_step!(st, erk)
_gpu_device_step(ts, erk) = error(
    "gpu_run!: no device step exists for $(typeof(ts)); the GPU path implements CNAB2, " *
    "ExponentialRungeKutta2, and RungeKutta3. Use the CPU path " *
    "(`run_solver!`/`solver_step!`) for this timestepper.")

"""
    gpu_run!(cpu_state::SolverState, nsteps; arch = CPU(),
             output_every = 0, output_fn = nothing) -> cpu_state

Public-API convenience: advance a configured CPU `SolverState` by `nsteps` steps ON THE
GPU PATH, using the device step that matches `cpu_state.parameters.timestepper` (CNAB2,
ExponentialRungeKutta2, or RungeKutta3; anything else errors rather than being silently
integrated as CNAB2).  Builds the device state via [`build_gpu_solver_state`](@ref) (optionally
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
path requires.  Builder scope is a :shell, single-rank, topography-free, static-BC config;
a conducting inner core additionally requires CNAB2 (see `build_gpu_solver_state`).
"""
function gpu_run!(cpu_state::SolverState, nsteps::Int; arch::AbstractArchitecture = CPU(),
        output_every::Int = 0, output_fn = nothing)
    nsteps >= 0 || throw(ArgumentError("gpu_run!: nsteps must be ≥ 0, got $nsteps"))
    gst = build_gpu_solver_state(cpu_state)
    # Dispatch on the CONFIGURED timestepper, exactly as `_gpu_time_step!`
    # (api/simulation.jl) does. Running `gpu_solver_step!` unconditionally silently
    # integrated an ERK2 or RungeKutta3 configuration with CNAB2 — the bundle carries no
    # timestepper tag, so the step could not notice, and `build_gpu_solver_state` only
    # rejects a non-CNAB2 timestepper when a conducting inner core is configured.
    ts = cpu_state.parameters.timestepper
    erk = ts isa ExponentialRungeKutta2 ? build_gpu_erk2_state(cpu_state) : nothing
    arch isa CPU || (gst = gpu_to_device(gst, arch))
    if erk !== nothing && !(arch isa CPU)
        erk = gpu_to_device(erk, arch)
    end
    gpu_run!(gst, nsteps; output_every = output_every, output_fn = output_fn,
        step! = _gpu_device_step(ts, erk))
    sync_gpu_state_to_cpu!(cpu_state, gst)
    # Advance the runtime clock alongside the public one; downstream readers
    # (get_current_simulation_time, ERK2 diagnostics) go through
    # `runtime.timestep_state`, not `cpu_state.step`/`.time`.
    reset_solver_clock!(cpu_state;
        time = cpu_state.time + nsteps * cpu_state.parameters.timestep,
        step = cpu_state.step + nsteps)
    return cpu_state
end
