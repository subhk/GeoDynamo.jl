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

function _maybe_enforce_ball_scalar!(field, domain)
    if domain !== nothing && domain.r[1, 4] == 0.0 && isdefined(@__MODULE__, :GeodynamoBall)
        getfield(@__MODULE__, :GeodynamoBall).apply_ball_temperature_regularity!(field)
    end
    return field
end

function _maybe_enforce_ball_vector!(field, domain)
    if domain !== nothing && domain.r[1, 4] == 0.0 && isdefined(@__MODULE__, :GeodynamoBall)
        getfield(@__MODULE__, :GeodynamoBall).enforce_ball_vector_regularity!(field.toroidal, field.poloidal)
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
    return _maybe_enforce_ball_scalar!(field, domain)
end

"""
    randomize_vector_field!(field; amplitude, lmax, domain=nothing)

Populate velocity-like toroidal/poloidal fields with random perturbations up to degree `lmax`.
"""
function randomize_vector_field!(field; amplitude::Real, lmax::Int, domain=nothing)
    amp = Float64(amplitude)
    for spectral in (field.toroidal, field.poloidal)
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
    return _maybe_enforce_ball_vector!(field, domain)
end

"""
    randomize_magnetic_field!(field; amplitude, lmax, domain=nothing)

Populate magnetic toroidal/poloidal fields with random perturbations.
"""
function randomize_magnetic_field!(field; amplitude::Real, lmax::Int, domain=nothing)
    amp = Float64(amplitude)
    for spectral in (field.toroidal, field.poloidal)
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
    return _maybe_enforce_ball_vector!(field, domain)
end
using SHTnsKit

export load_initial_conditions!, generate_random_initial_conditions!
export set_analytical_initial_conditions!, save_initial_conditions

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
"""
function load_temperature_initial_conditions!(temp_field, file_path::String)

    # Placeholder for NetCDF loading
    # In real implementation, this would use NCDatasets.jl

    # For now, generate a simple test pattern
    @warn "NetCDF loading not implemented, using test pattern"

    T = eltype(temp_field.spectral.data)
    nr, nlm = size(temp_field.spectral.data)

    # Simple hot bottom, cold top with small perturbation
    for r_idx in 1:nr
        r_frac = (r_idx - 1) / (nr - 1)  # 0 at bottom, 1 at top

        for lm in 1:nlm
            if lm == 1  # l=0, m=0 mode
                temp_field.spectral.data[r_idx, lm] = T(1.0 - r_frac)  # Linear profile
            elseif lm <= 4  # Small perturbations in low-order modes
                temp_field.spectral.data[r_idx, lm] = T(0.01 * sin(π * r_frac) * (rand() - 0.5))
            else
                temp_field.spectral.data[r_idx, lm] = T(0.0)
            end
        end
    end

    return temp_field
end

"""
    load_magnetic_initial_conditions!(mag_field, file_path::String)

Load magnetic initial conditions from NetCDF file.
"""
function load_magnetic_initial_conditions!(mag_field, file_path::String)

    @warn "NetCDF loading not implemented, using test pattern"

    T = eltype(mag_field.toroidal.data)
    nr_tor, nlm_tor = size(mag_field.toroidal.data)
    nr_pol, nlm_pol = size(mag_field.poloidal.data)

    # Simple dipolar field with small perturbations
    for r_idx in 1:nr_tor
        r_frac = (r_idx - 1) / (nr_tor - 1)

        for lm in 1:nlm_tor
            if lm == 3  # l=1, m=0 dipole mode for toroidal
                mag_field.toroidal.data[r_idx, lm] = T(0.1 * sin(π * r_frac))
            elseif lm <= 6  # Small perturbations
                mag_field.toroidal.data[r_idx, lm] = T(0.001 * (rand() - 0.5))
            else
                mag_field.toroidal.data[r_idx, lm] = T(0.0)
            end
        end
    end

    for r_idx in 1:nr_pol
        r_frac = (r_idx - 1) / (nr_pol - 1)

        for lm in 1:nlm_pol
            if lm == 3  # l=1, m=0 dipole mode for poloidal
                mag_field.poloidal.data[r_idx, lm] = T(1.0 * sin(π * r_frac))
            elseif lm <= 6  # Small perturbations
                mag_field.poloidal.data[r_idx, lm] = T(0.01 * (rand() - 0.5))
            else
                mag_field.poloidal.data[r_idx, lm] = T(0.0)
            end
        end
    end

    return mag_field
end

"""
    load_velocity_initial_conditions!(vel_field, file_path::String)

Load velocity initial conditions from NetCDF file.
"""
function load_velocity_initial_conditions!(vel_field, file_path::String)

    @warn "NetCDF loading not implemented, using test pattern"

    T = eltype(vel_field.toroidal.data)
    nr_tor, nlm_tor = size(vel_field.toroidal.data)
    nr_pol, nlm_pol = size(vel_field.poloidal.data)

    # Simple convective pattern with small velocities
    for r_idx in 1:nr_tor
        r_frac = (r_idx - 1) / (nr_tor - 1)

        for lm in 1:nlm_tor
            if lm <= 10  # Convective modes
                vel_field.toroidal.data[r_idx, lm] = T(0.01 * sin(2π * r_frac) * (rand() - 0.5))
            else
                vel_field.toroidal.data[r_idx, lm] = T(0.0)
            end
        end
    end

    for r_idx in 1:nr_pol
        r_frac = (r_idx - 1) / (nr_pol - 1)

        for lm in 1:nlm_pol
            if lm <= 10  # Convective modes
                vel_field.poloidal.data[r_idx, lm] = T(0.01 * sin(π * r_frac) * (rand() - 0.5))
            else
                vel_field.poloidal.data[r_idx, lm] = T(0.0)
            end
        end
    end

    return vel_field
end

"""
    load_composition_initial_conditions!(comp_field, file_path::String)

Load composition initial conditions from NetCDF file.
"""
function load_composition_initial_conditions!(comp_field, file_path::String)

    @warn "NetCDF loading not implemented, using test pattern"

    # Similar to temperature but with composition range [0,1]
    nr, nlm = size(comp_field.spectral.data)

    for r_idx in 1:nr
        r_frac = (r_idx - 1) / (nr - 1)

        for lm in 1:nlm
            if lm == 1  # l=0, m=0 mode
                comp_field.spectral.data[r_idx, lm] = 0.1 + 0.2 * r_frac  # 0.1 to 0.3 range
            elseif lm <= 4  # Small perturbations
                comp_field.spectral.data[r_idx, lm] = 0.01 * (rand() - 0.5)
            else
                comp_field.spectral.data[r_idx, lm] = 0.0
            end
        end
    end

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

Generate random temperature initial conditions.
"""
function generate_random_temperature!(temp_field, amplitude, modes_range)

    T = eltype(temp_field.spectral.data)
    nr, nlm = size(temp_field.spectral.data)

    # Clear field first
    fill!(temp_field.spectral.data, T(0))

    for r_idx in 1:nr
        r_frac = (r_idx - 1) / (nr - 1)

        # Base temperature profile (hot bottom, cold top)
        base_temp = T(1.0 - 0.8 * r_frac)

        for lm in modes_range
            if lm <= nlm
                if lm == 1  # l=0, m=0 mode
                    temp_field.spectral.data[r_idx, lm] = base_temp + T(amplitude * 0.1 * (rand() - 0.5))
                else
                    # Random perturbations with radial dependence
                    radial_factor = sin(π * r_frac)  # Peak in middle
                    temp_field.spectral.data[r_idx, lm] = T(amplitude * radial_factor * (rand() - 0.5))
                end
            end
        end
    end

    return temp_field
end

"""
    generate_random_magnetic!(mag_field, amplitude, modes_range)

Generate random magnetic initial conditions.
"""
function generate_random_magnetic!(mag_field, amplitude, modes_range)

    T = eltype(mag_field.toroidal.data)
    nr_tor, nlm_tor = size(mag_field.toroidal.data)
    nr_pol, nlm_pol = size(mag_field.poloidal.data)

    # Clear fields first
    fill!(mag_field.toroidal.data, T(0))
    fill!(mag_field.poloidal.data, T(0))

    # Toroidal field
    for r_idx in 1:nr_tor
        r_frac = (r_idx - 1) / (nr_tor - 1)
        radial_factor = sin(π * r_frac)  # Peak in middle

        for lm in modes_range
            if lm <= nlm_tor
                mag_field.toroidal.data[r_idx, lm] = T(amplitude * radial_factor * (rand() - 0.5))
            end
        end
    end

    # Poloidal field (typically stronger for dipolar field)
    for r_idx in 1:nr_pol
        r_frac = (r_idx - 1) / (nr_pol - 1)
        radial_factor = sin(π * r_frac)

        for lm in modes_range
            if lm <= nlm_pol
                if lm == 3  # l=1, m=0 dipole mode
                    mag_field.poloidal.data[r_idx, lm] = T(5.0 * amplitude * radial_factor)
                else
                    mag_field.poloidal.data[r_idx, lm] = T(amplitude * radial_factor * (rand() - 0.5))
                end
            end
        end
    end

    return mag_field
end

"""
    generate_random_velocity!(vel_field, amplitude, modes_range)

Generate random velocity initial conditions.
"""
function generate_random_velocity!(vel_field, amplitude, modes_range)

    T = eltype(vel_field.toroidal.data)
    nr_tor, nlm_tor = size(vel_field.toroidal.data)
    nr_pol, nlm_pol = size(vel_field.poloidal.data)

    # Clear fields first
    fill!(vel_field.toroidal.data, T(0))
    fill!(vel_field.poloidal.data, T(0))

    # Generate small random velocities
    for r_idx in 1:nr_tor
        r_frac = (r_idx - 1) / (nr_tor - 1)
        radial_factor = sin(π * r_frac)  # Avoid boundaries

        for lm in modes_range
            if lm <= nlm_tor
                vel_field.toroidal.data[r_idx, lm] = T(amplitude * radial_factor * (rand() - 0.5))
            end
        end
    end

    for r_idx in 1:nr_pol
        r_frac = (r_idx - 1) / (nr_pol - 1)
        radial_factor = sin(π * r_frac)

        for lm in modes_range
            if lm <= nlm_pol
                vel_field.poloidal.data[r_idx, lm] = T(amplitude * radial_factor * (rand() - 0.5))
            end
        end
    end

    return vel_field
end

"""
    generate_random_composition!(comp_field, amplitude, modes_range)

Generate random composition initial conditions.
"""
function generate_random_composition!(comp_field, amplitude, modes_range)

    nr, nlm = size(comp_field.spectral.data)

    # Clear field first
    fill!(comp_field.spectral.data, 0.0)

    for r_idx in 1:nr
        r_frac = (r_idx - 1) / (nr - 1)

        # Base composition profile
        base_comp = 0.1 + 0.2 * r_frac  # 0.1 to 0.3

        for lm in modes_range
            if lm <= nlm
                if lm == 1  # l=0, m=0 mode
                    comp_field.spectral.data[r_idx, lm] = base_comp + amplitude * 0.05 * (rand() - 0.5)
                else
                    radial_factor = sin(π * r_frac)
                    comp_field.spectral.data[r_idx, lm] = amplitude * 0.1 * radial_factor * (rand() - 0.5)
                end
            end
        end
    end

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
"""
function set_analytical_temperature!(temp_field, pattern::Symbol, amplitude; parameters...)

    T = eltype(temp_field.spectral.data)
    nr, nlm = size(temp_field.spectral.data)
    fill!(temp_field.spectral.data, T(0))

    if pattern == :conductive
        # Linear conductive profile
        for r_idx in 1:nr
            r_frac = (r_idx - 1) / (nr - 1)
            temp_field.spectral.data[r_idx, 1] = T(amplitude * (1.0 - r_frac))
        end

    elseif pattern == :hot_blob
        # Hot thermal blob in center
        r_center = get(parameters, :r_center, 0.5)
        blob_width = get(parameters, :blob_width, 0.2)

        for r_idx in 1:nr
            r_frac = (r_idx - 1) / (nr - 1)

            # Background conductive profile
            temp_field.spectral.data[r_idx, 1] = T(0.5 * (1.0 - r_frac))

            # Add hot blob
            if abs(r_frac - r_center) < blob_width
                temp_field.spectral.data[r_idx, 1] += T(amplitude)
                # Add some higher-order modes for blob shape
                temp_field.spectral.data[r_idx, 2] += T(0.3 * amplitude)
                temp_field.spectral.data[r_idx, 4] += T(0.2 * amplitude)
            end
        end

    else
        throw(ArgumentError("Unknown temperature pattern: $pattern"))
    end

    return temp_field
end

"""
    set_analytical_magnetic!(mag_field, pattern, amplitude; parameters...)

Set analytical magnetic field patterns.
"""
function set_analytical_magnetic!(mag_field, pattern::Symbol, amplitude; parameters...)

    T = eltype(mag_field.toroidal.data)
    nr_tor, nlm_tor = size(mag_field.toroidal.data)
    nr_pol, nlm_pol = size(mag_field.poloidal.data)

    fill!(mag_field.toroidal.data, T(0))
    fill!(mag_field.poloidal.data, T(0))

    if pattern == :dipole
        # Earth-like dipolar field
        for r_idx in 1:nr_pol
            r_frac = (r_idx - 1) / (nr_pol - 1)

            # Dipole field: l=1, m=0 mode
            if nlm_pol >= 3
                mag_field.poloidal.data[r_idx, 3] = T(amplitude * sin(π * r_frac))
            end
        end

        # Small toroidal component
        for r_idx in 1:nr_tor
            r_frac = (r_idx - 1) / (nr_tor - 1)
            if nlm_tor >= 3
                mag_field.toroidal.data[r_idx, 3] = T(0.1 * amplitude * sin(π * r_frac))
            end
        end

    elseif pattern == :uniform_field
        # Uniform axial field
        direction = get(parameters, :direction, :z)  # :x, :y, or :z

        for r_idx in 1:nr_pol
            if direction == :z && nlm_pol >= 1
                mag_field.poloidal.data[r_idx, 1] = T(amplitude)  # l=0, m=0 mode
            elseif direction == :x && nlm_pol >= 3
                mag_field.poloidal.data[r_idx, 3] = T(amplitude)  # l=1, m=0 mode
            end
        end

    else
        throw(ArgumentError("Unknown magnetic pattern: $pattern"))
    end

    return mag_field
end

"""
    set_analytical_velocity!(vel_field, pattern, amplitude; parameters...)

Set analytical velocity patterns.
"""
function set_analytical_velocity!(vel_field, pattern::Symbol, amplitude; parameters...)

    T = eltype(vel_field.toroidal.data)
    nr_tor, nlm_tor = size(vel_field.toroidal.data)
    nr_pol, nlm_pol = size(vel_field.poloidal.data)

    fill!(vel_field.toroidal.data, T(0))
    fill!(vel_field.poloidal.data, T(0))

    if pattern == :convective
        # Small convective perturbations
        for r_idx in 1:min(nr_tor, nr_pol)
            r_frac = (r_idx - 1) / (nr_tor - 1)
            radial_factor = sin(π * r_frac)  # Avoid boundaries

            # Add small perturbations in low-order modes
            for lm in 2:min(10, nlm_tor, nlm_pol)
                vel_field.toroidal.data[r_idx, lm] = T(amplitude * radial_factor * 0.1)
                vel_field.poloidal.data[r_idx, lm] = T(amplitude * radial_factor * 0.1)
            end
        end

    elseif pattern == :solid_rotation
        # Solid body rotation
        omega = get(parameters, :omega, 1.0)

        # This would require specific toroidal modes for solid rotation
        # Implementation depends on the exact spherical harmonic conventions
        @warn "Solid rotation pattern not fully implemented"

    else
        throw(ArgumentError("Unknown velocity pattern: $pattern"))
    end

    return vel_field
end

"""
    set_analytical_composition!(comp_field, pattern, amplitude; parameters...)

Set analytical composition patterns.
"""
function set_analytical_composition!(comp_field, pattern::Symbol, amplitude; parameters...)

    nr, nlm = size(comp_field.spectral.data)
    fill!(comp_field.spectral.data, 0.0)

    if pattern == :stratified
        # Vertically stratified composition
        bottom_comp = get(parameters, :bottom_composition, 0.3)
        top_comp = get(parameters, :top_composition, 0.1)

        for r_idx in 1:nr
            r_frac = (r_idx - 1) / (nr - 1)
            comp_field.spectral.data[r_idx, 1] = bottom_comp + (top_comp - bottom_comp) * r_frac
        end

    elseif pattern == :blob
        # Compositional blob
        r_center = get(parameters, :r_center, 0.3)
        blob_width = get(parameters, :blob_width, 0.2)
        blob_composition = get(parameters, :blob_composition, 0.8)

        for r_idx in 1:nr
            r_frac = (r_idx - 1) / (nr - 1)

            # Background
            comp_field.spectral.data[r_idx, 1] = 0.1

            # Add blob
            if abs(r_frac - r_center) < blob_width
                comp_field.spectral.data[r_idx, 1] = blob_composition
                comp_field.spectral.data[r_idx, 2] = 0.1 * blob_composition
            end
        end

    else
        throw(ArgumentError("Unknown composition pattern: $pattern"))
    end

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
