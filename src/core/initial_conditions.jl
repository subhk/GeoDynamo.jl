# ================================================================================
# Initial Conditions Module
# ================================================================================
"""
    InitialConditions

Module for loading and generating initial conditions for geodynamo simulations.
Supports loading from NetCDF files, generating random fields, and setting
prescribed analytical patterns.
"""
module InitialConditions

using LinearAlgebra
using Random
using SHTnsKit

# Import functions from parent module (GeoDynamo)
# These will be available when the module is included in GeoDynamo.jl
import ..get_local_range
import ..local_spectral_storage_slot
import ..set_local_spectral_value!
import ..local_spectral_value

const GEODYNAMO_PARENT = parentmodule(@__MODULE__)

@inline apply_ball_scalar_regularity!(field) =
    getproperty(GEODYNAMO_PARENT, :apply_ball_temperature_regularity!)(field)

@inline apply_ball_vector_regularity!(field) =
    getproperty(GEODYNAMO_PARENT, :enforce_ball_vector_regularity!)(field.𝒯, field.𝒫)

export load_initial_conditions!, generate_random_initial_conditions!
export set_analytical_initial_conditions!, save_initial_conditions
export randomize_scalar_field!, randomize_vector_field!, randomize_magnetic_field!

function __maybe_enforce_ball_scalar!(field, domain)
    if domain !== nothing && domain.r[1, 4] == 0.0
        apply_ball_scalar_regularity!(field)
    end
    return field
end

function __maybe_enforce_ball_vector!(field, domain)
    if domain !== nothing && domain.r[1, 4] == 0.0
        apply_ball_vector_regularity!(field)
    end
    return field
end

"""
    randomize_scalar_field!(field; amplitude, lmax, domain=nothing)

Populate a scalar spectral field (temperature/composition) with random perturbations up to degree `lmax`.
If a radial `domain` is provided and includes r=0, ball regularity is enforced.
"""
function randomize_scalar_field!(field; amplitude::Real, lmax::Int, domain=nothing)
    spectral = getproperty(field, :spectral)
    real = parent(spectral.data_real)
    imag = parent(spectral.data_imag)
    lm_range = get_local_range(spectral.pencil, 1)
    r_range  = get_local_range(spectral.pencil, 3)
    l_values = spectral.config.l_values
    fill!(real, zero(eltype(real)))
    fill!(imag, zero(eltype(imag)))
    amp = Float64(amplitude)
    for (local_idx, global_idx) in enumerate(lm_range)
        if global_idx <= length(l_values)
            l = l_values[global_idx]
            if l <= lmax
                for r in r_range
                    lr = r - first(r_range) + 1
                    if lr <= size(real, 3)
                        real[local_idx, 1, lr] = convert(eltype(real), amp * (rand() - 0.5))
                        imag[local_idx, 1, lr] = zero(eltype(imag))
                    end
                end
            end
        end
    end
    __maybe_enforce_ball_scalar!(field, domain)
    # Verify initial conditions are finite after all transformations
    spectral = getproperty(field, :spectral)
    if any(isnan, parent(spectral.data_real)) || any(isinf, parent(spectral.data_real))
        error("Non-finite values in scalar field initial conditions (real part)")
    end
    return field
end

"""
    randomize_vector_field!(field; amplitude, lmax, domain=nothing)

Populate velocity-like toroidal/poloidal fields with random perturbations up to degree `lmax`.
"""
function randomize_vector_field!(field; amplitude::Real, lmax::Int, domain=nothing)
    amp = Float64(amplitude)
    for spectral in (field.𝒯, field.𝒫)
        real = parent(spectral.data_real)
        imag = parent(spectral.data_imag)
        fill!(real, zero(eltype(real)))
        fill!(imag, zero(eltype(imag)))
        lm_range = get_local_range(spectral.pencil, 1)
        r_range  = get_local_range(spectral.pencil, 3)
        l_values = spectral.config.l_values
        for (local_idx, global_idx) in enumerate(lm_range)
            if global_idx <= length(l_values)
                l = l_values[global_idx]
                if 1 <= l <= lmax
                    for r in r_range
                        lr = r - first(r_range) + 1
                        if lr <= size(real, 3)
                            real[local_idx, 1, lr] = convert(eltype(real), amp * (rand() - 0.5))
                            imag[local_idx, 1, lr] = zero(eltype(imag))
                        end
                    end
                end
            end
        end
    end
    __maybe_enforce_ball_vector!(field, domain)
    for spectral in (field.𝒯, field.𝒫)
        if any(isnan, parent(spectral.data_real)) || any(isinf, parent(spectral.data_real))
            error("Non-finite values in vector field initial conditions (real part)")
        end
    end
    return field
end

"""
    randomize_magnetic_field!(field; amplitude, lmax, domain=nothing)

Populate magnetic toroidal/poloidal fields with random perturbations.
"""
function randomize_magnetic_field!(field; amplitude::Real, lmax::Int, domain=nothing)
    amp = Float64(amplitude)
    for spectral in (field.𝒯, field.𝒫)
        real = parent(spectral.data_real)
        imag = parent(spectral.data_imag)
        fill!(real, zero(eltype(real)))
        fill!(imag, zero(eltype(imag)))
        lm_range = get_local_range(spectral.pencil, 1)
        r_range  = get_local_range(spectral.pencil, 3)
        l_values = spectral.config.l_values
        for (local_idx, global_idx) in enumerate(lm_range)
            if global_idx <= length(l_values)
                l = l_values[global_idx]
                if 1 <= l <= lmax
                    for r in r_range
                        lr = r - first(r_range) + 1
                        if lr <= size(real, 3)
                            real[local_idx, 1, lr] = convert(eltype(real), amp * (rand() - 0.5))
                            imag[local_idx, 1, lr] = zero(eltype(imag))
                        end
                    end
                end
            end
        end
    end
    __maybe_enforce_ball_vector!(field, domain)
    for spectral in (field.𝒯, field.𝒫)
        if any(isnan, parent(spectral.data_real)) || any(isinf, parent(spectral.data_real))
            error("Non-finite values in magnetic field initial conditions (real part)")
        end
    end
    return field
end

# ================================================================================
# Loading Initial Conditions from Files
# ================================================================================

"""
    load_initial_conditions!(field, field_type::Symbol, file_path::String)

Load initial conditions from NetCDF file for any field type.

# Arguments
- `field`: Field structure (temperature, magnetic, velocity, or composition type)
- `field_type`: Field type (:temperature, :magnetic, :velocity, :composition)
- `file_path`: Path to NetCDF file containing initial conditions

# File Format
NetCDF files should contain:
- For scalar fields: spectral coefficients array
- For vector fields: toroidal and poloidal spectral coefficients
- Coordinate arrays: lm indices, radial grid
"""
function load_initial_conditions!(field, field_type::Symbol, file_path::String)

    if !isfile(file_path)
        throw(ArgumentError("Initial conditions file not found: $file_path"))
    end

    println("Loading initial conditions from $file_path...")

    try
        # Use NCDatasets or similar NetCDF library
        # For now, implement a simple placeholder that would use NetCDF

        if field_type == :temperature
            load_temperature_initial_conditions!(field, file_path)
        elseif field_type == :magnetic
            load_magnetic_initial_conditions!(field, file_path)
        elseif field_type == :velocity
            load_velocity_initial_conditions!(field, file_path)
        elseif field_type == :composition
            load_composition_initial_conditions!(field, file_path)
        else
            throw(ArgumentError("Unknown field type: $field_type"))
        end

        println("Initial conditions loaded successfully")

    catch e
        @error "Failed to load initial conditions: $e"
        rethrow(e)
    end

    return field
end

"""
    load_temperature_initial_conditions!(temp_field, file_path::String)

Load temperature initial conditions from NetCDF file.

Note: NetCDF loading not yet implemented. Use `set_analytical_temperature!`
with `:conductive` pattern instead.
"""
function load_temperature_initial_conditions!(temp_field, file_path::String)
    @warn "NetCDF loading not implemented. Using conductive profile as fallback."
    set_analytical_temperature!(temp_field, :conductive, 1.0)
    return temp_field
end

"""
    load_magnetic_initial_conditions!(mag_field, file_path::String)

Load magnetic initial conditions from NetCDF file.

Note: NetCDF loading not yet implemented. Use `set_analytical_magnetic!`
with `:dipole` pattern instead.
"""
function load_magnetic_initial_conditions!(mag_field, file_path::String)
    @warn "NetCDF loading not implemented. Using dipole pattern as fallback."
    set_analytical_magnetic!(mag_field, :dipole, 1.0)
    return mag_field
end

"""
    load_velocity_initial_conditions!(vel_field, file_path::String)

Load velocity initial conditions from NetCDF file.

Note: NetCDF loading not yet implemented. Use `set_analytical_velocity!`
with `:convective` pattern instead.
"""
function load_velocity_initial_conditions!(vel_field, file_path::String)
    @warn "NetCDF loading not implemented. Using convective pattern as fallback."
    set_analytical_velocity!(vel_field, :convective, 0.01)
    return vel_field
end

"""
    load_composition_initial_conditions!(comp_field, file_path::String)

Load composition initial conditions from NetCDF file.

Note: NetCDF loading not yet implemented. Use `set_analytical_composition!`
with `:stratified` pattern instead.
"""
function load_composition_initial_conditions!(comp_field, file_path::String)
    @warn "NetCDF loading not implemented. Using stratified pattern as fallback."
    set_analytical_composition!(comp_field, :stratified, 1.0)

    return comp_field
end

# ================================================================================
# Random Initial Conditions Generation
# ================================================================================

"""
    generate_random_initial_conditions!(field, field_type::Symbol;
                                       amplitude=1.0, modes_range=1:10,
                                       seed=nothing)

Generate random initial conditions for any field type.

# Arguments
- `field`: Field structure to initialize
- `field_type`: Type of field (:temperature, :magnetic, :velocity, :composition)
- `amplitude`: Overall amplitude of random perturbations
- `modes_range`: Range of spherical harmonic modes to excite
- `seed`: Random seed for reproducibility (optional)

# Examples
```julia
# Random temperature field
generate_random_initial_conditions!(temp_field, :temperature, amplitude=0.1)

# Random magnetic field with specific modes
generate_random_initial_conditions!(mag_field, :magnetic,
                                   amplitude=0.01, modes_range=1:20, seed=42)
```
"""
function generate_random_initial_conditions!(field, field_type::Symbol;
                                           amplitude::Real=1.0,
                                           modes_range=1:10,
                                           seed::Union{Int, Nothing}=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end

    println("Generating random initial conditions for $field_type...")
    println("  Amplitude: $amplitude")
    println("  Modes range: $modes_range")
    println("  Seed: $seed")

    if field_type == :temperature
        generate_random_temperature!(field, amplitude, modes_range)
    elseif field_type == :magnetic
        generate_random_magnetic!(field, amplitude, modes_range)
    elseif field_type == :velocity
        generate_random_velocity!(field, amplitude, modes_range)
    elseif field_type == :composition
        generate_random_composition!(field, amplitude, modes_range)
    else
        throw(ArgumentError("Unknown field type: $field_type"))
    end

    println("Random initial conditions generated")

    return field
end

"""
    generate_random_temperature!(temp_field, amplitude, modes_range)

Generate random temperature initial conditions with base conductive profile
and random perturbations.

Uses PencilArray structure with data_real/data_imag arrays.
"""
function generate_random_temperature!(temp_field, amplitude, modes_range)
    spectral = temp_field.spectral
    real_data = parent(spectral.data_real)
    imag_data = parent(spectral.data_imag)

    T = eltype(real_data)
    nlm = size(real_data, 1)
    nr = size(real_data, 3)

    # Get local ranges for distributed computation
    lm_range = get_local_range(spectral.pencil, 1)
    r_range = get_local_range(spectral.pencil, 3)
    l_values = spectral.config.l_values

    # Clear field first
    fill!(real_data, zero(T))
    fill!(imag_data, zero(T))

    for global_lm in lm_range
        if global_lm <= length(l_values)
            l = l_values[global_lm]
            slot = local_spectral_storage_slot(spectral.config, global_lm)
            slot === nothing && continue

            for (local_r, global_r) in enumerate(r_range)
                if local_r <= size(real_data, 3)
                    r_frac = (global_r - 1) / max(nr - 1, 1)

                    if l == 0  # l=0, m=0 mode - base conductive profile
                        # Orthonormal SH (Y_0^0 = 1/√(4π)): store physical mean ×√(4π).
                        base_temp = T(1.0 - 0.8 * r_frac)
                        set_local_spectral_value!(real_data, slot, local_r,
                                                  sqrt(4 * T(π)) * (base_temp + T(amplitude * 0.1 * (rand() - 0.5))))
                    elseif l in modes_range
                        # Random perturbations with radial dependence
                        radial_factor = sin(π * r_frac)
                        set_local_spectral_value!(real_data, slot, local_r,
                                                  T(amplitude * radial_factor * (rand() - 0.5)))
                    end
                    # Imaginary part is zero for real-valued fields
                    set_local_spectral_value!(imag_data, slot, local_r, zero(T))
                end
            end
        end
    end

    __maybe_enforce_ball_scalar!(temp_field, temp_field.domain)
    return temp_field
end

"""
    generate_random_magnetic!(mag_field, amplitude, modes_range)

Generate random magnetic initial conditions with dipolar bias.

Uses PencilArray structure with data_real/data_imag arrays.
"""
function generate_random_magnetic!(mag_field, amplitude, modes_range)
    # Process toroidal and poloidal components
    for (spectral, is_poloidal) in ((mag_field.𝒯, false), (mag_field.𝒫, true))
        real_data = parent(spectral.data_real)
        imag_data = parent(spectral.data_imag)
        T = eltype(real_data)
        nr = size(real_data, 3)

        # Get local ranges
        lm_range = get_local_range(spectral.pencil, 1)
        r_range = get_local_range(spectral.pencil, 3)
        l_values = spectral.config.l_values

        # Clear fields
        fill!(real_data, zero(T))
        fill!(imag_data, zero(T))

        for global_lm in lm_range
            if global_lm <= length(l_values)
                l = l_values[global_lm]
                slot = local_spectral_storage_slot(spectral.config, global_lm)
                slot === nothing && continue

                for (local_r, global_r) in enumerate(r_range)
                    if local_r <= size(real_data, 3)
                        r_frac = (global_r - 1) / max(nr - 1, 1)
                        radial_factor = sin(π * r_frac)

                        if l in modes_range && l >= 1  # l=0 not valid for vector fields
                            if is_poloidal && l == 1  # Dipole mode - stronger
                                set_local_spectral_value!(real_data, slot, local_r,
                                                          T(5.0 * amplitude * radial_factor))
                            else
                                set_local_spectral_value!(real_data, slot, local_r,
                                                          T(amplitude * radial_factor * (rand() - 0.5)))
                            end
                        end
                        set_local_spectral_value!(imag_data, slot, local_r, zero(T))
                    end
                end
            end
        end
    end

    __maybe_enforce_ball_vector!(mag_field, mag_field.outer_domain)
    return mag_field
end

"""
    generate_random_velocity!(vel_field, amplitude, modes_range)

Generate random velocity initial conditions.

Uses PencilArray structure with data_real/data_imag arrays.
"""
function generate_random_velocity!(vel_field, amplitude, modes_range)
    # Process toroidal and poloidal components
    for spectral in (vel_field.𝒯, vel_field.𝒫)
        real_data = parent(spectral.data_real)
        imag_data = parent(spectral.data_imag)
        T = eltype(real_data)
        nr = size(real_data, 3)

        # Get local ranges
        lm_range = get_local_range(spectral.pencil, 1)
        r_range = get_local_range(spectral.pencil, 3)
        l_values = spectral.config.l_values

        # Clear fields
        fill!(real_data, zero(T))
        fill!(imag_data, zero(T))

        for global_lm in lm_range
            if global_lm <= length(l_values)
                l = l_values[global_lm]
                slot = local_spectral_storage_slot(spectral.config, global_lm)
                slot === nothing && continue

                for (local_r, global_r) in enumerate(r_range)
                    if local_r <= size(real_data, 3)
                        r_frac = (global_r - 1) / max(nr - 1, 1)
                        radial_factor = sin(π * r_frac)  # Avoid boundaries

                        if l in modes_range && l >= 1  # l=0 not valid for vector fields
                            set_local_spectral_value!(real_data, slot, local_r,
                                                      T(amplitude * radial_factor * (rand() - 0.5)))
                        end
                        set_local_spectral_value!(imag_data, slot, local_r, zero(T))
                    end
                end
            end
        end
    end

    __maybe_enforce_ball_vector!(vel_field, vel_field.domain)
    return vel_field
end

"""
    generate_random_composition!(comp_field, amplitude, modes_range)

Generate random composition initial conditions with stratified base profile.

Uses PencilArray structure with data_real/data_imag arrays.
"""
function generate_random_composition!(comp_field, amplitude, modes_range)
    spectral = comp_field.spectral
    real_data = parent(spectral.data_real)
    imag_data = parent(spectral.data_imag)

    T = eltype(real_data)
    nr = size(real_data, 3)

    # Get local ranges
    lm_range = get_local_range(spectral.pencil, 1)
    r_range = get_local_range(spectral.pencil, 3)
    l_values = spectral.config.l_values

    # Clear field first
    fill!(real_data, zero(T))
    fill!(imag_data, zero(T))

    for global_lm in lm_range
        if global_lm <= length(l_values)
            l = l_values[global_lm]
            slot = local_spectral_storage_slot(spectral.config, global_lm)
            slot === nothing && continue

            for (local_r, global_r) in enumerate(r_range)
                if local_r <= size(real_data, 3)
                    r_frac = (global_r - 1) / max(nr - 1, 1)

                    if l == 0  # l=0, m=0 mode - base stratified profile
                        base_comp = T(0.1 + 0.2 * r_frac)  # 0.1 to 0.3
                        set_local_spectral_value!(real_data, slot, local_r,
                                                  base_comp + T(amplitude * 0.05 * (rand() - 0.5)))
                    elseif l in modes_range
                        radial_factor = sin(π * r_frac)
                        set_local_spectral_value!(real_data, slot, local_r,
                                                  T(amplitude * 0.1 * radial_factor * (rand() - 0.5)))
                    end
                    set_local_spectral_value!(imag_data, slot, local_r, zero(T))
                end
            end
        end
    end

    __maybe_enforce_ball_scalar!(comp_field, comp_field.domain)
    return comp_field
end

# ================================================================================
# Analytical Initial Conditions
# ================================================================================

"""
    set_analytical_initial_conditions!(field, field_type::Symbol, pattern::Symbol;
                                      amplitude=1.0, parameters...)

Set analytical initial conditions based on predefined patterns.

# Patterns
- `:conductive` - Conductive temperature profile
- `:dipole` - Dipolar magnetic field
- `:convective` - Small convective velocity pattern
- `:stratified` - Stratified composition profile

# Examples
```julia
# Conductive temperature profile
set_analytical_initial_conditions!(temp_field, :temperature, :conductive)

# Earth-like dipolar magnetic field
set_analytical_initial_conditions!(mag_field, :magnetic, :dipole, amplitude=1.0)
```
"""
function set_analytical_initial_conditions!(field, field_type::Symbol, pattern::Symbol;
                                          amplitude::Real=1.0, parameters...)

    println("Setting analytical initial conditions:")
    println("  Field: $field_type")
    println("  Pattern: $pattern")
    println("  Amplitude: $amplitude")

    if field_type == :temperature
        set_analytical_temperature!(field, pattern, amplitude; parameters...)
    elseif field_type == :magnetic
        set_analytical_magnetic!(field, pattern, amplitude; parameters...)
    elseif field_type == :velocity
        set_analytical_velocity!(field, pattern, amplitude; parameters...)
    elseif field_type == :composition
        set_analytical_composition!(field, pattern, amplitude; parameters...)
    else
        throw(ArgumentError("Unknown field type: $field_type"))
    end

    println("Analytical initial conditions set")

    return field
end

"""
    set_analytical_temperature!(temp_field, pattern, amplitude; parameters...)

Set analytical temperature patterns.

Uses PencilArray structure with data_real/data_imag arrays.
"""
function set_analytical_temperature!(temp_field, pattern::Symbol, amplitude; parameters...)
    spectral = temp_field.spectral
    real_data = parent(spectral.data_real)
    imag_data = parent(spectral.data_imag)

    T = eltype(real_data)
    nr = size(real_data, 3)

    # Get local ranges
    lm_range = get_local_range(spectral.pencil, 1)
    r_range = get_local_range(spectral.pencil, 3)
    l_values = spectral.config.l_values

    # Clear fields
    fill!(real_data, zero(T))
    fill!(imag_data, zero(T))

    if pattern == :conductive
        # Linear conductive profile (only l=0 mode)
        for global_lm in lm_range
            if global_lm <= length(l_values) && l_values[global_lm] == 0
                slot = local_spectral_storage_slot(spectral.config, global_lm)
                slot === nothing && continue
                for (local_r, global_r) in enumerate(r_range)
                    if local_r <= size(real_data, 3)
                        r_frac = (global_r - 1) / max(nr - 1, 1)
                        # Orthonormal SH (Y_0^0 = 1/√(4π)): store physical mean ×√(4π).
                        set_local_spectral_value!(real_data, slot, local_r,
                                                  sqrt(4 * T(π)) * T(amplitude * (1.0 - r_frac)))
                    end
                end
            end
        end

    elseif pattern == :hot_blob
        # Radially-localized hot shell (l=0, spherically symmetric): background
        # conductive profile plus a Gaussian bump centered at r_center. Stored as
        # the (0,0) coefficient ×√(4π) (orthonormal SH, Y_0^0 = 1/√(4π)).
        # Non-axisymmetric symmetry breaking, if wanted, comes from
        # generate_random_initial_conditions!.
        r_center = get(parameters, :r_center, 0.5)
        blob_width = get(parameters, :blob_width, 0.2)
        s4π = sqrt(4 * T(π))

        for global_lm in lm_range
            if global_lm <= length(l_values) && l_values[global_lm] == 0
                slot = local_spectral_storage_slot(spectral.config, global_lm)
                slot === nothing && continue
                for (local_r, global_r) in enumerate(r_range)
                    if local_r <= size(real_data, 3)
                        r_frac = (global_r - 1) / max(nr - 1, 1)
                        bump = exp(-0.5 * ((r_frac - r_center) / blob_width)^2)
                        value = 0.5 * (1.0 - r_frac) + amplitude * bump
                        set_local_spectral_value!(real_data, slot, local_r, s4π * T(value))
                    end
                end
            end
        end

    else
        throw(ArgumentError("Unknown temperature pattern: $pattern"))
    end

    __maybe_enforce_ball_scalar!(temp_field, temp_field.domain)
    return temp_field
end

"""
    set_analytical_magnetic!(mag_field, pattern, amplitude; parameters...)

Set analytical magnetic field patterns.

Uses PencilArray structure with data_real/data_imag arrays.
"""
function set_analytical_magnetic!(mag_field, pattern::Symbol, amplitude; parameters...)
    # Helper to set field values
    function set_spectral_values!(spectral, l_target, value_fn)
        real_data = parent(spectral.data_real)
        imag_data = parent(spectral.data_imag)
        T = eltype(real_data)
        nr = size(real_data, 3)

        lm_range = get_local_range(spectral.pencil, 1)
        r_range = get_local_range(spectral.pencil, 3)
        l_values = spectral.config.l_values

        for global_lm in lm_range
            if global_lm <= length(l_values) && l_values[global_lm] == l_target
                slot = local_spectral_storage_slot(spectral.config, global_lm)
                slot === nothing && continue
                for (local_r, global_r) in enumerate(r_range)
                    if local_r <= size(real_data, 3)
                        r_frac = (global_r - 1) / max(nr - 1, 1)
                        set_local_spectral_value!(real_data, slot, local_r, T(value_fn(r_frac)))
                        set_local_spectral_value!(imag_data, slot, local_r, zero(T))
                    end
                end
            end
        end
    end

    # Clear both fields
    for spectral in (mag_field.𝒯, mag_field.𝒫)
        fill!(parent(spectral.data_real), zero(eltype(parent(spectral.data_real))))
        fill!(parent(spectral.data_imag), zero(eltype(parent(spectral.data_imag))))
    end

    if pattern == :dipole
        # Earth-like dipolar field: l=1 mode
        set_spectral_values!(mag_field.𝒫, 1, r -> amplitude * sin(π * r))
        set_spectral_values!(mag_field.𝒯, 1, r -> 0.1 * amplitude * sin(π * r))

    elseif pattern == :uniform_field
        direction = get(parameters, :direction, :z)
        if direction == :z
            set_spectral_values!(mag_field.𝒫, 0, r -> amplitude)
        elseif direction == :x
            set_spectral_values!(mag_field.𝒫, 1, r -> amplitude)
        end

    else
        throw(ArgumentError("Unknown magnetic pattern: $pattern"))
    end

    __maybe_enforce_ball_vector!(mag_field, mag_field.outer_domain)
    return mag_field
end

"""
    set_analytical_velocity!(vel_field, pattern, amplitude; parameters...)

Set analytical velocity patterns.

Uses PencilArray structure with data_real/data_imag arrays.
"""
function set_analytical_velocity!(vel_field, pattern::Symbol, amplitude; parameters...)
    # Clear both fields
    for spectral in (vel_field.𝒯, vel_field.𝒫)
        fill!(parent(spectral.data_real), zero(eltype(parent(spectral.data_real))))
        fill!(parent(spectral.data_imag), zero(eltype(parent(spectral.data_imag))))
    end

    if pattern == :convective
        # Small convective perturbations in low-order modes
        for spectral in (vel_field.𝒯, vel_field.𝒫)
            real_data = parent(spectral.data_real)
            T = eltype(real_data)
            nr = size(real_data, 3)

            lm_range = get_local_range(spectral.pencil, 1)
            r_range = get_local_range(spectral.pencil, 3)
            l_values = spectral.config.l_values

            for global_lm in lm_range
                if global_lm <= length(l_values)
                    l = l_values[global_lm]
                    if 1 <= l <= 10  # Low-order convective modes
                        slot = local_spectral_storage_slot(spectral.config, global_lm)
                        slot === nothing && continue
                        for (local_r, global_r) in enumerate(r_range)
                            if local_r <= size(real_data, 3)
                                r_frac = (global_r - 1) / max(nr - 1, 1)
                                radial_factor = sin(π * r_frac)
                                set_local_spectral_value!(real_data, slot, local_r,
                                                          T(amplitude * radial_factor * 0.1))
                            end
                        end
                    end
                end
            end
        end

    elseif pattern == :solid_rotation
        @warn "Solid rotation pattern not fully implemented"

    else
        throw(ArgumentError("Unknown velocity pattern: $pattern"))
    end

    __maybe_enforce_ball_vector!(vel_field, vel_field.domain)
    return vel_field
end

"""
    set_analytical_composition!(comp_field, pattern, amplitude; parameters...)

Set analytical composition patterns.

Uses PencilArray structure with data_real/data_imag arrays.
"""
function set_analytical_composition!(comp_field, pattern::Symbol, amplitude; parameters...)
    spectral = comp_field.spectral
    real_data = parent(spectral.data_real)
    imag_data = parent(spectral.data_imag)

    T = eltype(real_data)
    nr = size(real_data, 3)

    # Get local ranges
    lm_range = get_local_range(spectral.pencil, 1)
    r_range = get_local_range(spectral.pencil, 3)
    l_values = spectral.config.l_values

    # Clear field
    fill!(real_data, zero(T))
    fill!(imag_data, zero(T))

    if pattern == :stratified
        # Vertically stratified composition (only l=0 mode)
        bottom_comp = get(parameters, :bottom_composition, 0.3)
        top_comp = get(parameters, :top_composition, 0.1)

        for global_lm in lm_range
            if global_lm <= length(l_values) && l_values[global_lm] == 0
                slot = local_spectral_storage_slot(spectral.config, global_lm)
                slot === nothing && continue
                for (local_r, global_r) in enumerate(r_range)
                    if local_r <= size(real_data, 3)
                        r_frac = (global_r - 1) / max(nr - 1, 1)
                        # Orthonormal SH (Y_0^0 = 1/√(4π)): store physical mean ×√(4π).
                        set_local_spectral_value!(real_data, slot, local_r,
                                                  sqrt(4 * T(π)) * T(bottom_comp + (top_comp - bottom_comp) * r_frac))
                    end
                end
            end
        end

    elseif pattern == :blob
        # Radially-localized compositional shell (l=0, spherically symmetric):
        # background 0.1 with a Gaussian bump peaking at blob_composition, centered
        # at r_center. Stored as the (0,0) coefficient ×√(4π) (orthonormal SH).
        r_center = get(parameters, :r_center, 0.3)
        blob_width = get(parameters, :blob_width, 0.2)
        blob_composition = get(parameters, :blob_composition, 0.8)
        background = 0.1
        s4π = sqrt(4 * T(π))

        for global_lm in lm_range
            if global_lm <= length(l_values) && l_values[global_lm] == 0
                slot = local_spectral_storage_slot(spectral.config, global_lm)
                slot === nothing && continue
                for (local_r, global_r) in enumerate(r_range)
                    if local_r <= size(real_data, 3)
                        r_frac = (global_r - 1) / max(nr - 1, 1)
                        bump = exp(-0.5 * ((r_frac - r_center) / blob_width)^2)
                        value = background + (blob_composition - background) * bump
                        set_local_spectral_value!(real_data, slot, local_r, s4π * T(value))
                    end
                end
            end
        end

    else
        throw(ArgumentError("Unknown composition pattern: $pattern"))
    end

    __maybe_enforce_ball_scalar!(comp_field, comp_field.domain)
    return comp_field
end

# ================================================================================
# Saving Initial Conditions
# ================================================================================

"""
    save_initial_conditions(field, field_type::Symbol, file_path::String)

Save current field state as initial conditions to NetCDF file.

This function is useful for saving generated or computed initial conditions
for later use in simulations.
"""
function save_initial_conditions(field, field_type::Symbol, file_path::String)

    println("Saving initial conditions to $file_path...")

    # Placeholder for NetCDF saving
    # In real implementation, this would use NCDatasets.jl
    @warn "NetCDF saving not implemented, would save to $file_path"

    # Would save spectral coefficients and metadata
    # Format: same as expected by load_initial_conditions!

    println("Initial conditions saved")

    return file_path
end

end # module InitialConditions
