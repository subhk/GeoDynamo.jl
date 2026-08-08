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
    # Output numbering lives in a TimeTracker, and `write_fields!` advances it via
    # update_tracker!. Rebuilding the tracker on every firing reset output_count to
    # 0, so every write reused output #1 and (overwrite_files = true) deleted the
    # previous file. The tracker therefore has to persist on the writer.
    _tracker::Base.RefValue{Union{Nothing, TimeTracker}}
end

function FieldWriter(path::String; schedule, fields = [:velocity, :temperature, :magnetic])
    selected = collect(Symbol, fields)
    # Reject unknown selectors at construction rather than silently writing an
    # empty field set: an unknown key maps to () in _select_output_fields, which
    # would otherwise drop everything without warning.
    for f in selected
        haskey(_FIELD_KEY_GROUPS, f) || throw(ArgumentError(
            "FieldWriter: unknown field selector :$(f); valid selectors are " *
            "$(sort(collect(keys(_FIELD_KEY_GROUPS))))"))
    end
    return FieldWriter{typeof(schedule)}(path, schedule, selected,
        Base.RefValue{Union{Nothing, TimeTracker}}(nothing))
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
    # See FieldWriter: restart numbering must survive across firings, or every
    # checkpoint overwrites restart #1.
    _tracker::Base.RefValue{Union{Nothing, TimeTracker}}
end

function CheckpointWriter(path::String; schedule)
    CheckpointWriter{typeof(schedule)}(path, schedule,
        Base.RefValue{Union{Nothing, TimeTracker}}(nothing))
end

"""
    _writer_tracker!(ow, config, current_time) -> TimeTracker

The writer's persistent `TimeTracker`, created on first use.

`create_time_tracker(config, current_time - 1.0)` seeds `last_output_time` one
unit in the past so the first firing is authorized (the writer configs use
interval 0.0 — the schedule, not the tracker, decides *when*). Reusing the same
tracker afterwards is what makes `output_count`/`restart_count` — and hence the
filename numbers — advance.
"""
function _writer_tracker!(ow, config::OutputConfig, current_time::Float64)
    tracker = ow._tracker[]
    if tracker === nothing
        tracker = create_time_tracker(config, current_time - 1.0)
        ow._tracker[] = tracker
    end
    return tracker
end

# ================================================================================
# Internal dispatch
# ================================================================================

# Map user-facing field selectors to the dict keys produced by
# `extract_all_fields`. A FieldWriter `fields=[:temperature]` should write only
# the temperature entries, not the full state.
const _FIELD_KEY_GROUPS = Dict{Symbol, Tuple{Vararg{String}}}(
    :velocity => ("velocity_toroidal", "velocity_poloidal"),
    :temperature => ("temperature", "temperature_spectral"),
    :magnetic => ("magnetic_toroidal", "magnetic_poloidal"),
    :composition => ("composition", "composition_spectral"),
)

"""
    _select_output_fields(all_fields, selectors)

Return the subset of `all_fields` (the dict from `extract_all_fields`) selected
by `selectors` (e.g. `[:velocity, :temperature]`). An empty selector list means
"write everything". Unknown selectors and keys absent from `all_fields` are
skipped silently.
"""
function _select_output_fields(all_fields::AbstractDict, selectors)
    isempty(selectors) && return all_fields
    keep = Dict{String, Any}()
    for s in selectors
        for k in get(_FIELD_KEY_GROUPS, s, ())
            haskey(all_fields, k) && (keep[k] = all_fields[k])
        end
    end
    return keep
end

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
        0.0,            # output_interval = 0 -> write on every call
        Inf,            # restart_interval = Inf -> never restart from a FieldWriter
        Inf,            # max_output_time
        1e-10          # time_tolerance
    )
    tracker = _writer_tracker!(ow, config, state.time)  # persistent: numbers advance

    metadata = Dict{String, Any}(
        "current_time" => state.time,
        "current_step" => state.step
    )

    try
        # Honor the writer's field selection: filter the extracted dict and call
        # the Dict-based write_fields! so only requested fields are written.
        # extract_field_info / compute_diagnostics / setup_variables! all guard
        # on haskey, so a partial dict is safe.
        selected = _select_output_fields(extract_all_fields(state), ow.fields)
        write_fields!(selected, tracker, metadata, config,
            state.runtime.shtns_config,
            state.runtime.shtns_config.pencils;
            geometry = state.parameters.geometry,
            radius_ratio = state.parameters.radius_ratio,
            radial_grid = Float64.(state.runtime.outer_core_domain.r[1:state.runtime.outer_core_domain.N, 4]))
    catch e
        # Fail loud: a scheduled snapshot that silently fails would let a run
        # complete believing its output was written when nothing was saved.
        throw(ErrorException(
            "FieldWriter: write_fields! failed for path \"$(ow.path)\": " *
            sprint(showerror, e)))
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
    tracker = _writer_tracker!(ow, config, state.time)

    metadata = Dict{String, Any}(
        "current_time" => state.time,
        "current_step" => state.step
    )

    try
        fields = extract_all_fields(state)
        write_restart!(fields, tracker, metadata, config,
            state.runtime.shtns_config.pencils;
            shtns_config = state.runtime.shtns_config,
            geometry = state.parameters.geometry,
            radius_ratio = state.parameters.radius_ratio,
            radial_grid = Float64.(state.runtime.outer_core_domain.r[1:state.runtime.outer_core_domain.N, 4]))
    catch e
        # Fail loud: a silently dropped checkpoint means a crash later has no
        # restart point, defeating the purpose of the checkpoint schedule.
        throw(ErrorException(
            "CheckpointWriter: write_restart! failed for path \"$(ow.path)\": " *
            sprint(showerror, e)))
    end
    # `write_restart!` numbers the file from `tracker.restart_count + 1` but, unlike
    # `write_fields!`, never advances the tracker itself — so the count has to be
    # advanced here or every checkpoint would reuse restart #1 and delete the last.
    update_tracker!(tracker, state.time, config, false, true)
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
    # `write_fields!`/`write_restart!` are COLLECTIVE, and `should_fire` gates them
    # (line ~110). A WallTimeInterval read from each rank's own clock can answer
    # differently on the same step, putting one rank inside the write's MPI.Bcast!
    # while another steps on — the deadlock `write_fields!`'s docstring warns about.
    # `_collective_wtime` broadcasts rank 0's elapsed time so the gate is unanimous.
    wtime = _collective_wtime(sim)
    ctx = _ScheduleContext(sim.model.clock.time, sim.model.clock.iteration, wtime)
    for ow in values(sim.output_writers)
        _run_output_writer!(ow, sim, ctx)
    end
    return nothing
end
