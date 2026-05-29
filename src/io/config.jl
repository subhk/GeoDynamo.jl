# Use get_comm() from parallel/mpi.jl for lazy MPI initialization
const __output_comm = Ref{MPI.Comm}()
const __output_comm_initialized = Ref(false)
const __OUTPUT_COMM_LOCK = ReentrantLock()

"""
    output_comm()

Return the communicator used by the output layer.

The communicator is resolved lazily through `get_comm()` so importing the I/O
code does not force MPI initialization. The result is cached behind a lock
because output calls may be reached from different setup paths.
"""
function output_comm()
    if __output_comm_initialized[]
        return __output_comm[]
    end
    lock(__OUTPUT_COMM_LOCK) do
        if !__output_comm_initialized[]
            __output_comm[] = get_comm()
            __output_comm_initialized[] = true
        end
    end
    return __output_comm[]
end

# ================================================================================
# Configuration
# ================================================================================

"""
    OutputSpace

Selects whether output files store physical fields, spectral fields, or the
mixed layout used by GeoDynamo’s default writer.
"""
@enum OutputSpace begin
    MIXED_FIELDS        # Spectral for velocity/magnetic, physical for temperature/composition
    PHYSICAL_ONLY       # Output only physical space data
    SPECTRAL_ONLY       # Output only spectral coefficients
end

"""
    OutputConfig

Configuration for NetCDF history/restart output.

This groups together output layout, file naming, metadata toggles, precision,
and time-based write cadence so a simulation can keep one explicit output
policy object.
"""
struct OutputConfig
    output_space::OutputSpace
    output_dir::String
    filename_prefix::String
    include_metadata::Bool
    include_grid::Bool
    include_diagnostics::Bool
    output_precision::DataType
    spectral_lmax_output::Int
    overwrite_files::Bool

    # Time-based control
    output_interval::Float64
    restart_interval::Float64
    max_output_time::Float64
    time_tolerance::Float64
end

"""
    default_config(; precision=Float64)
    default_config(T)

Return the default `OutputConfig` used by GeoDynamo’s NetCDF writer.

The positional `default_config(T)` form is kept as a convenience alias for
call sites that already pass a floating-point type.
"""
function default_config(; precision::Type{<:AbstractFloat}=Float64)
    return OutputConfig(
        MIXED_FIELDS,       # spectral for velocity/magnetic, physical for temperature
        "./output",         # output directory
        "geodynamo",        # filename prefix
        true,               # include metadata
        true,               # include grid
        true,               # include diagnostics
        precision,          # output precision
        -1,                 # spectral lmax (-1 = all)
        true,               # overwrite files
        0.1,                # output every 0.1 time units
        1.0,                # restart every 1.0 time units
        Inf,                # max output time
        1e-10               # time tolerance
    )
end

default_config(T::Type{<:AbstractFloat}) = default_config(; precision=T)

"""
    resolve_output_precision(sym)

Map user/API precision symbols to concrete floating-point types.

Only `:float32` and `:float64` are accepted; unknown symbols warn and fall back
to `Float64` so output remains usable instead of failing late in a write path.
"""
@inline function resolve_output_precision(sym::Symbol)
    sym === :float32 && return Float32
    sym === :float64 && return Float64
    @warn "Unknown output precision symbol $(sym); defaulting to Float64"
    return Float64
end

"""
    output_config_from_parameters(; base_config=default_config(), output_precision=:float32, output_interval=1.0)

Create an `OutputConfig` using simulation parameters as the source of truth for
runtime-controlled output settings.
"""
function output_config_from_parameters(; base_config::OutputConfig=default_config(),
                                         output_precision::Symbol=:float32,
                                         output_interval::Float64=1.0)
    precision = resolve_output_precision(output_precision)

    return OutputConfig(
        base_config.output_space,
        base_config.output_dir,
        base_config.filename_prefix,
        base_config.include_metadata,
        base_config.include_grid,
        base_config.include_diagnostics,
        precision,
        base_config.spectral_lmax_output,
        base_config.overwrite_files,
        output_interval,
        base_config.restart_interval,
        base_config.max_output_time,
        base_config.time_tolerance
    )
end

"""
    with_output_precision(config, T)

Return a copy of `config` that writes field and scalar output with precision `T`.

All scheduling, metadata, directory, and layout settings are preserved.
"""
function with_output_precision(config::OutputConfig, ::Type{T}) where {T<:AbstractFloat}
    return OutputConfig(
        config.output_space,
        config.output_dir,
        config.filename_prefix,
        config.include_metadata,
        config.include_grid,
        config.include_diagnostics,
        T,
        config.spectral_lmax_output,
        config.overwrite_files,
        config.output_interval,
        config.restart_interval,
        config.max_output_time,
        config.time_tolerance
    )
end

# ================================================================================
# Time Tracking
# ================================================================================

"""
    TimeTracker

Tracks when history and restart files should next be written.

The output layer keeps this separate from the main solver state so output
cadence can be tested and advanced independently from the physics timestep
logic.
"""
mutable struct TimeTracker
    last_output_time::Float64
    last_restart_time::Float64
    output_count::Int
    restart_count::Int
    next_output_time::Float64
    next_restart_time::Float64
    grid_file_written::Bool
end

"""
    create_time_tracker(config[, start_time=0.0])

Construct a `TimeTracker` initialized for a run that starts at `start_time`.
"""
function create_time_tracker(config::OutputConfig, start_time::Float64 = 0.0)
    return TimeTracker(
        start_time - config.output_interval,
        start_time - config.restart_interval,
        0, 0,
        start_time, start_time,
        false
    )
end

"""
    should_output_now(tracker, current_time, config)

Return `true` when a history/output write should occur at `current_time`.
"""
function should_output_now(tracker::TimeTracker, current_time::Float64, config::OutputConfig)
    time_since_output = current_time - tracker.last_output_time
    return (time_since_output >= config.output_interval - config.time_tolerance &&
            current_time <= config.max_output_time)
end

"""
    should_restart_now(tracker, current_time, config)

Return `true` when a restart/checkpoint write should occur at `current_time`.
"""
function should_restart_now(tracker::TimeTracker, current_time::Float64, config::OutputConfig)
    time_since_restart = current_time - tracker.last_restart_time
    return time_since_restart >= config.restart_interval - config.time_tolerance
end

"""
    update_tracker!(tracker, current_time, config, did_output, did_restart)

Advance output and restart counters after a write attempt.

Pass the actual write decisions so history and restart cadence can advance
independently when only one output type was produced.
"""
function update_tracker!(tracker::TimeTracker, current_time::Float64,
                        config::OutputConfig, did_output::Bool, did_restart::Bool)
    if did_output
        tracker.last_output_time = current_time
        tracker.output_count += 1
        tracker.next_output_time = current_time + config.output_interval
    end
    if did_restart
        tracker.last_restart_time = current_time
        tracker.restart_count += 1
        tracker.next_restart_time = current_time + config.restart_interval
    end
end

"""
    time_to_next_output(tracker, current_time, config)

Return the smallest wait time until the next history or restart event.

This helper is used by scheduling code that wants a conservative next wake-up
time without duplicating the tracker cadence logic.
"""
function time_to_next_output(tracker::TimeTracker, current_time::Float64, config::OutputConfig)
    time_to_output = tracker.next_output_time - current_time
    time_to_restart = tracker.next_restart_time - current_time
    return min(time_to_output, time_to_restart, config.output_interval)
end

# ================================================================================
# Energy Tracking Utilities
# ================================================================================

"""
    EnergyTracker

In-memory accumulator for optional diagnostic energy histories.

The NetCDF writer currently uses the global `ENERGY_TRACKER` for lightweight
energy bookkeeping in tests and diagnostics helpers.
"""
mutable struct EnergyTracker
    kinetic_energy::Vector{Float64}
    magnetic_energy::Vector{Float64}
    thermal_energy::Vector{Float64}
    compositional_energy::Vector{Float64}
    total_energy::Vector{Float64}
    timestamps::Vector{Int}
    enable_tracking::Bool
end

const ENERGY_TRACKER = EnergyTracker(
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Int[],
    true,
)

"""
    compute_field_energy(field_data::Array{T,3}) where T

Compute L2 energy of a 3D scalar field.
"""
function compute_field_energy(field_data::Array{T, 3}) where T
    local_energy = 0.5 * sum(abs2, field_data)
    if MPI.Initialized()
        return MPI.Allreduce(local_energy, +, get_comm())
    end
    return local_energy
end

"""
    compute_vector_energy(v_r, v_theta, v_phi)

Compute kinetic or magnetic energy from three physical-space vector components.
"""
function compute_vector_energy(
    v_r::Array{T, 3},
    v_theta::Array{T, 3},
    v_phi::Array{T, 3},
) where T
    local_energy = 0.5 * (sum(abs2, v_r) + sum(abs2, v_theta) + sum(abs2, v_phi))
    if MPI.Initialized()
        return MPI.Allreduce(local_energy, +, get_comm())
    end
    return local_energy
end

"""
    reset_energy_tracker!()

Clear all accumulated diagnostic energy histories in `ENERGY_TRACKER`.
"""
function reset_energy_tracker!()
    empty!(ENERGY_TRACKER.kinetic_energy)
    empty!(ENERGY_TRACKER.magnetic_energy)
    empty!(ENERGY_TRACKER.thermal_energy)
    empty!(ENERGY_TRACKER.compositional_energy)
    empty!(ENERGY_TRACKER.total_energy)
    empty!(ENERGY_TRACKER.timestamps)
    return nothing
end
