# ================================================================================
# Output Writers — Oceananigans-style schedule-driven I/O
# ================================================================================
#
# NOTE on write_fields! / write_restart! signatures
# --------------------------------------------------
# The underlying io/writer.jl functions require full OutputConfig and
# TimeTracker objects.  The thin wrappers below create lightweight per-writer
# trackers and configs so that FieldWriter / CheckpointWriter remain simple
# path+schedule value types while still delegating to the existing parallel I/O
# layer when MPI is live.
#
# If MPI is not initialized (e.g. during unit testing) the write methods fall
# back to a no-op with an @info message so that tests that don't spin up MPI
# do not error.
# ================================================================================

# ================================================================================
# FieldWriter
# ================================================================================

"""
    FieldWriter(path; schedule, fields=[:velocity, :temperature, :magnetic])

Schedule-driven writer that snapshots selected fields to `path` whenever
`schedule` fires.
"""
struct FieldWriter{S <: AbstractSchedule}
    path::String
    schedule::S
    fields::Vector{Symbol}
end

function FieldWriter(path::String; schedule, fields = [:velocity, :temperature, :magnetic])
    return FieldWriter{typeof(schedule)}(path, schedule, collect(Symbol, fields))
end

# ================================================================================
# CheckpointWriter
# ================================================================================

"""
    CheckpointWriter(path; schedule)

Schedule-driven writer that writes a restart/checkpoint file to `path` whenever
`schedule` fires.
"""
struct CheckpointWriter{S <: AbstractSchedule}
    path::String
    schedule::S
end

function CheckpointWriter(path::String; schedule)
    CheckpointWriter{typeof(schedule)}(path, schedule)
end

# ================================================================================
# Internal dispatch
# ================================================================================

"""
    _run_output_writer!(ow::FieldWriter, sim, ctx)

Fires a field snapshot write if the writer's schedule fires for `ctx`.

NOTE: The underlying `write_fields!` in `io/writer.jl` requires a
`TimeTracker` and `OutputConfig` and must be called collectively by all MPI
ranks.  When MPI is not initialized (e.g. unit tests) this falls back to
logging an info message without performing I/O.
"""
function _run_output_writer!(ow::FieldWriter, sim, ctx::_ScheduleContext)
    should_fire(ow.schedule, ctx) || return nothing

    if !MPI.Initialized()
        @info "FieldWriter (no-op: MPI not initialized)" path=ow.path step=ctx.step
        return nothing
    end

    state = sim.model.state
    config = OutputConfig(
        MIXED_FIELDS,   # output_space
        ow.path,        # output_dir
        "geodynamo",    # filename_prefix
        true,           # include_metadata
        true,           # include_grid
        true,           # include_diagnostics
        Float64,        # output_precision
        -1,             # spectral_lmax_output (-1 = all)
        true,           # overwrite_files
        Inf,            # output_interval (always write when called)
        Inf,            # restart_interval (no restart from FieldWriter)
        Inf,            # max_output_time
        1e-10          # time_tolerance
    )
    tracker = create_time_tracker(config, state.time - 1.0)  # force output now

    metadata = Dict{String, Any}(
        "current_time" => state.time,
        "current_step" => state.step
    )

    try
        write_fields!(state, tracker, metadata, config,
            state.runtime.shtns_config,
            state.runtime.shtns_config.pencils)
    catch e
        @warn "FieldWriter: write_fields! failed" exception=e path=ow.path
    end
    return nothing
end

"""
    _run_output_writer!(ow::CheckpointWriter, sim, ctx)

Fires a checkpoint write if the writer's schedule fires for `ctx`.

NOTE: The underlying `write_restart!` in `io/writer.jl` requires a
`Dict{String,Any}` of fields (from `extract_all_fields`), a `TimeTracker`, and
an `OutputConfig`.  When MPI is not initialized this falls back to logging.
"""
function _run_output_writer!(ow::CheckpointWriter, sim, ctx::_ScheduleContext)
    should_fire(ow.schedule, ctx) || return nothing

    if !MPI.Initialized()
        @info "CheckpointWriter (no-op: MPI not initialized)" path=ow.path step=ctx.step
        return nothing
    end

    state = sim.model.state
    config = OutputConfig(
        MIXED_FIELDS,
        ow.path,
        "geodynamo",
        true,
        false,  # no grid file for restart
        false,  # no diagnostics for restart
        Float64,
        -1,
        true,
        Inf,
        0.0,    # restart_interval = 0 so should_restart_now fires immediately
        Inf,
        1e-10
    )
    tracker = create_time_tracker(config, state.time - 1.0)

    metadata = Dict{String, Any}(
        "current_time" => state.time,
        "current_step" => state.step
    )

    try
        fields = extract_all_fields(state)
        write_restart!(fields, tracker, metadata, config,
            state.runtime.shtns_config.pencils;
            shtns_config = state.runtime.shtns_config,
            radial_grid = Float64.(state.runtime.𝒟ᵒᶜ.r[1:state.runtime.𝒟ᵒᶜ.N, 4]))
    catch e
        @warn "CheckpointWriter: write_restart! failed" exception=e path=ow.path
    end
    return nothing
end

# ================================================================================
# _run_output_writers!
# ================================================================================

"""
    _run_output_writers!(sim)

Iterates over `sim.output_writers`, builds a `_ScheduleContext` from the
current simulation state, and fires each writer whose schedule returns `true`.
"""
function _run_output_writers!(sim)
    wtime = sim._wall_start > 0.0 ? time() - sim._wall_start : 0.0
    ctx = _ScheduleContext(sim.model.clock.time, sim.model.clock.iteration, wtime)
    for ow in values(sim.output_writers)
        _run_output_writer!(ow, sim, ctx)
    end
    return nothing
end
