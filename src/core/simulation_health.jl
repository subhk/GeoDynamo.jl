# ================================================================================
# Public NaN/Inf Detection Surface
# ================================================================================

"""
    NaNDetectionConfig

Configuration for NaN/Inf detection during simulation.
"""
struct NaNDetectionConfig
    enabled::Bool
    check_every_n_steps::Int
    abort_on_nan::Bool
    verbose::Bool
end

const DEFAULT_NAN_CONFIG = NaNDetectionConfig(true, 1, true, true)

"""
    check_field_for_nan(field_data, field_name, config, step)

Check a field for NaN or Inf values. Returns `(has_nan, has_inf, nan_count, inf_count)`.
"""
function check_field_for_nan(
    field_data::AbstractArray,
    field_name::String,
    config::NaNDetectionConfig,
    step::Int,
)
    if !config.enabled || step % config.check_every_n_steps != 0
        return (false, false, 0, 0)
    end

    nan_count = count(isnan, field_data)
    inf_count = count(isinf, field_data)
    has_nan = nan_count > 0
    has_inf = inf_count > 0

    if (has_nan || has_inf) && config.verbose && get_rank() == 0
        @warn "Numerical issue detected in $field_name at step $step" nan_count inf_count
    end

    return (has_nan, has_inf, nan_count, inf_count)
end

"""
    check_spectral_field_for_nan(field, field_name, config, step)

Check both real and imaginary parts of a spectral field.
"""
function check_spectral_field_for_nan(
    field,
    field_name::String,
    config::NaNDetectionConfig,
    step::Int,
)
    has_nan_r, has_inf_r, nan_r, inf_r = check_field_for_nan(
        parent(field.data_real),
        "$(field_name)__real",
        config,
        step,
    )
    has_nan_i, has_inf_i, nan_i, inf_i = check_field_for_nan(
        parent(field.data_imag),
        "$(field_name)__imag",
        config,
        step,
    )

    return (has_nan_r || has_nan_i, has_inf_r || has_inf_i, nan_r + nan_i, inf_r + inf_i)
end

"""
    check_simulation_state_for_nan(state, step; config=DEFAULT_NAN_CONFIG)

Comprehensive NaN/Inf check across all simulation fields.
Returns true if any NaN/Inf is detected.
"""
function check_simulation_state_for_nan(
    state,
    step::Int;
    config::NaNDetectionConfig=DEFAULT_NAN_CONFIG,
)
    if !config.enabled || step % config.check_every_n_steps != 0
        return false
    end

    any_issue = false

    has_nan, has_inf, _, _ = check_spectral_field_for_nan(
        state.velocity.𝒯,
        "velocity_toroidal",
        config,
        step,
    )
    any_issue |= (has_nan || has_inf)

    has_nan, has_inf, _, _ = check_spectral_field_for_nan(
        state.velocity.𝒫,
        "velocity_poloidal",
        config,
        step,
    )
    any_issue |= (has_nan || has_inf)

    has_nan, has_inf, _, _ = check_spectral_field_for_nan(
        state.magnetic.𝒯,
        "magnetic_toroidal",
        config,
        step,
    )
    any_issue |= (has_nan || has_inf)

    has_nan, has_inf, _, _ = check_spectral_field_for_nan(
        state.magnetic.𝒫,
        "magnetic_poloidal",
        config,
        step,
    )
    any_issue |= (has_nan || has_inf)

    has_nan, has_inf, _, _ = check_spectral_field_for_nan(
        state.temperature.spectral,
        "temperature",
        config,
        step,
    )
    any_issue |= (has_nan || has_inf)

    if state.composition !== nothing
        has_nan, has_inf, _, _ = check_spectral_field_for_nan(
            state.composition.spectral,
            "composition",
            config,
            step,
        )
        any_issue |= (has_nan || has_inf)
    end

    comm = get_comm()
    if comm !== nothing && MPI.Comm_size(comm) > 1
        local_flag = any_issue ? 1 : 0
        global_flag = MPI.Allreduce(local_flag, MPI.MAX, comm)
        any_issue = global_flag > 0
    end

    if any_issue && config.abort_on_nan
        error(
            "NaN or Inf detected in simulation fields at step $step. " *
            "Simulation aborted to prevent invalid results. " *
            "Check:\n" *
            "  1. Timestep size (may be too large for stability)\n" *
            "  2. Boundary conditions (check for invalid values)\n" *
            "  3. Initial conditions (check for NaN/Inf in IC)\n" *
            "  4. Physical parameters (negative diffusivities, etc.)\n" *
            "Set config.abort_on_nan=false to continue despite NaN.",
        )
    end

    return any_issue
end
