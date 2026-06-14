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
using NCDatasets
using MPI

# Import functions from parent module (GeoDynamo)
# These will be available when the module is included in GeoDynamo.jl
import ..get_local_range
import ..local_spectral_storage_slot
import ..set_local_spectral_value!
import ..local_spectral_value
import ..get_comm
import ..size_global

const GEODYNAMO_PARENT = parentmodule(@__MODULE__)

export load_initial_conditions!, generate_random_initial_conditions!
export set_analytical_initial_conditions!, save_initial_conditions
export randomize_scalar_field!, randomize_vector_field!, randomize_magnetic_field!

# NOTE: the old `_maybe_enforce_ball_*` r=0-plane zeroing is gone. The ball
# uses an off-center radial grid with no r=0 node; regularity at r=0 is
# imposed by the implicit-matrix Robin rows, never by zeroing field planes.

"""
    randomize_scalar_field!(field; amplitude, lmax, domain=nothing)

Populate a scalar spectral field (temperature/composition) with random perturbations up to degree `lmax`.
"""
function randomize_scalar_field!(field; amplitude::Real, lmax::Int, domain = nothing)
    spectral = getproperty(field, :spectral)
    real3 = parent(spectral.data_real)
    imag3 = parent(spectral.data_imag)
    cfg = spectral.config
    r_range = get_local_range(spectral.pencil, 3)
    T = eltype(real3)
    fill!(real3, zero(T))
    fill!(imag3, zero(T))
    amp = Float64(amplitude)
    # Index storage by (l_slot, m_slot) via the mode → slot map so EVERY owned
    # (l, m) mode is perturbed — not just the m = 0 column (the legacy
    # `[idx, 1, r]` idiom pinned dim-2 to slot 1, silently zeroing all m > 0).
    for lm in 1:cfg.nlm
        cfg.l_values[lm] <= lmax || continue
        slot = local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue          # mode not owned by this rank
        for (local_r, _global_r) in enumerate(r_range)
            local_r <= size(real3, 3) || continue
            set_local_spectral_value!(real3, slot, local_r,
                                      convert(T, amp * (rand() - 0.5)))
            set_local_spectral_value!(imag3, slot, local_r, zero(T))
        end
    end
    # Verify initial conditions are finite after all transformations
    if any(isnan, real3) || any(isinf, real3)
        error("Non-finite values in scalar field initial conditions (real part)")
    end
    return field
end

"""
    randomize_vector_field!(field; amplitude, lmax, domain=nothing)

Populate velocity-like toroidal/poloidal fields with random perturbations up to degree `lmax`.
"""
function randomize_vector_field!(field; amplitude::Real, lmax::Int, domain = nothing)
    amp = Float64(amplitude)
    for spectral in (field.toroidal, field.poloidal)
        real3 = parent(spectral.data_real)
        imag3 = parent(spectral.data_imag)
        cfg = spectral.config
        T = eltype(real3)
        fill!(real3, zero(T))
        fill!(imag3, zero(T))
        r_range = get_local_range(spectral.pencil, 3)
        # Perturb every owned (l, m) mode with 1 ≤ l ≤ lmax (l = 0 carries no
        # toroidal/poloidal vector content). Slot-indexed so m > 0 is populated.
        for lm in 1:cfg.nlm
            l = cfg.l_values[lm]
            (1 <= l <= lmax) || continue
            slot = local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            for (local_r, _global_r) in enumerate(r_range)
                local_r <= size(real3, 3) || continue
                set_local_spectral_value!(real3, slot, local_r,
                                          convert(T, amp * (rand() - 0.5)))
                set_local_spectral_value!(imag3, slot, local_r, zero(T))
            end
        end
    end
    for spectral in (field.toroidal, field.poloidal)
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
function randomize_magnetic_field!(field; amplitude::Real, lmax::Int, domain = nothing)
    amp = Float64(amplitude)
    for spectral in (field.toroidal, field.poloidal)
        real3 = parent(spectral.data_real)
        imag3 = parent(spectral.data_imag)
        cfg = spectral.config
        T = eltype(real3)
        fill!(real3, zero(T))
        fill!(imag3, zero(T))
        r_range = get_local_range(spectral.pencil, 3)
        # Slot-indexed perturbation over every owned (l, m) mode, 1 ≤ l ≤ lmax.
        for lm in 1:cfg.nlm
            l = cfg.l_values[lm]
            (1 <= l <= lmax) || continue
            slot = local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            for (local_r, _global_r) in enumerate(r_range)
                local_r <= size(real3, 3) || continue
                set_local_spectral_value!(real3, slot, local_r,
                                          convert(T, amp * (rand() - 0.5)))
                set_local_spectral_value!(imag3, slot, local_r, zero(T))
            end
        end
    end
    for spectral in (field.toroidal, field.poloidal)
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

# ================================================================================
# NetCDF spectral I/O helpers (shared by load_* and save_initial_conditions)
# ================================================================================

# Map a field type to the `(netcdf_variable_base, spectral_field)` components it
# stores. Scalars carry one spectral field; vectors carry toroidal + poloidal.
function _ic_components(field_type::Symbol, field)
    if field_type === :temperature || field_type === :composition
        return ((string(field_type), getproperty(field, :spectral)),)
    elseif field_type === :magnetic || field_type === :velocity
        base = string(field_type)
        return ((base * "_toroidal", getproperty(field, :toroidal)),
            (base * "_poloidal", getproperty(field, :poloidal)))
    else
        throw(ArgumentError("Unknown field type for NetCDF initial conditions: $field_type"))
    end
end

# Gather a distributed spectral field into dense `(nlm, nr)` real/imag matrices
# replicated on every rank. Each (l,m,r) coefficient is owned by exactly one
# rank, so a sum-reduction over the comm reassembles the global arrays.
function _gather_full_spectral(spec)
    config = spec.config
    nlm = config.nlm
    nr = size_global(spec.pencil)[3]
    real3 = parent(spec.data_real)
    imag3 = parent(spec.data_imag)
    r_range = get_local_range(spec.pencil, 3)
    full_real = zeros(Float64, nlm, nr)
    full_imag = zeros(Float64, nlm, nr)
    for global_lm in 1:nlm
        slot = local_spectral_storage_slot(config, global_lm)
        slot === nothing && continue
        for (local_r, global_r) in enumerate(r_range)
            local_r <= size(real3, 3) || continue
            full_real[global_lm, global_r] = Float64(local_spectral_value(real3, slot, local_r))
            full_imag[global_lm, global_r] = Float64(local_spectral_value(imag3, slot, local_r))
        end
    end
    comm = get_comm()
    if MPI.Initialized() && MPI.Comm_size(comm) > 1
        MPI.Allreduce!(full_real, +, comm)
        MPI.Allreduce!(full_imag, +, comm)
    end
    return full_real, full_imag
end

# Scatter dense `(nlm, nr)` real/imag matrices back into a distributed spectral
# field, writing only the coefficients this rank owns.
function _scatter_full_spectral!(spec, full_real, full_imag)
    config = spec.config
    nlm = config.nlm
    real3 = parent(spec.data_real)
    imag3 = parent(spec.data_imag)
    T = eltype(real3)
    r_range = get_local_range(spec.pencil, 3)
    fill!(real3, zero(T))
    fill!(imag3, zero(T))
    for global_lm in 1:nlm
        slot = local_spectral_storage_slot(config, global_lm)
        slot === nothing && continue
        for (local_r, global_r) in enumerate(r_range)
            local_r <= size(real3, 3) || continue
            set_local_spectral_value!(real3, slot, local_r, T(full_real[global_lm, global_r]))
            set_local_spectral_value!(imag3, slot, local_r, T(full_imag[global_lm, global_r]))
        end
    end
    return spec
end

# Read an IC NetCDF file into `field`, validating that it actually matches the
# requested field type and truncation. A missing variable or a mismatched
# `nlm`/`field_type` is an error — never a silent fallback.
function _read_ic_into_field!(field, field_type::Symbol, file_path::String)
    components = _ic_components(field_type, field)
    config = components[1][2].config
    NCDataset(file_path, "r") do ds
        stored_type = get(ds.attrib, "field_type", nothing)
        if stored_type !== nothing && string(stored_type) != string(field_type)
            throw(ArgumentError(
                "IC file '$file_path' holds field_type=$(stored_type), expected $(field_type)"))
        end
        if haskey(ds.dim, "spectral_mode") && ds.dim["spectral_mode"] != config.nlm
            file_nlm = ds.dim["spectral_mode"]
            throw(ArgumentError(
                "IC file '$file_path' has nlm=$(file_nlm), field expects nlm=$(config.nlm)"))
        end
        for (name, spec) in components
            rname, iname = "$(name)_real", "$(name)_imag"
            (haskey(ds, rname) && haskey(ds, iname)) || throw(ArgumentError(
                "IC file '$file_path' is missing spectral variables $(rname)/$(iname)"))
            full_real = Array(ds[rname][:, :])
            full_imag = Array(ds[iname][:, :])
            _scatter_full_spectral!(spec, full_real, full_imag)
        end
    end
    return field
end

"""
    load_temperature_initial_conditions!(temp_field, file_path::String)

Load temperature initial conditions from an IC NetCDF file written by
[`save_initial_conditions`](@ref).
"""
load_temperature_initial_conditions!(temp_field, file_path::String) =
    _read_ic_into_field!(temp_field, :temperature, file_path)

"""
    load_magnetic_initial_conditions!(mag_field, file_path::String)

Load magnetic (toroidal + poloidal) initial conditions from an IC NetCDF file.
"""
load_magnetic_initial_conditions!(mag_field, file_path::String) =
    _read_ic_into_field!(mag_field, :magnetic, file_path)

"""
    load_velocity_initial_conditions!(vel_field, file_path::String)

Load velocity (toroidal + poloidal) initial conditions from an IC NetCDF file.
"""
load_velocity_initial_conditions!(vel_field, file_path::String) =
    _read_ic_into_field!(vel_field, :velocity, file_path)

"""
    load_composition_initial_conditions!(comp_field, file_path::String)

Load composition initial conditions from an IC NetCDF file.
"""
load_composition_initial_conditions!(comp_field, file_path::String) =
    _read_ic_into_field!(comp_field, :composition, file_path)

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
        amplitude::Real = 1.0,
        modes_range = 1:10,
        seed::Union{Int, Nothing} = nothing)
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
    nr = size(real_data, 3)

    # Get local ranges for distributed computation
    cfg = spectral.config
    r_range = get_local_range(spectral.pencil, 3)

    # Clear field first
    fill!(real_data, zero(T))
    fill!(imag_data, zero(T))

    # Iterate ALL modes via the slot map (m-major lm index) so m>0 modes are
    # populated. The old get_local_range(pencil,1) idiom walked the l_slot axis
    # (1:lmax+1) and fed it as a mode index → only the m=0 column was filled.
    for lm in 1:cfg.nlm
        l = cfg.l_values[lm]
        slot = local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue

        for (local_r, global_r) in enumerate(r_range)
            if local_r <= size(real_data, 3)
                r_frac = (global_r - 1) / max(nr - 1, 1)

                if l == 0  # l=0, m=0 mode - base conductive profile
                    # Orthonormal SH (Y_0^0 = 1/√(4π)): store physical mean ×√(4π).
                    base_temp = T(1.0 - 0.8 * r_frac)
                    set_local_spectral_value!(real_data, slot, local_r,
                        sqrt(4 * T(π)) *
                        (base_temp + T(amplitude * 0.1 * (rand() - 0.5))))
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

    return temp_field
end

"""
    generate_random_magnetic!(mag_field, amplitude, modes_range)

Generate random magnetic initial conditions with dipolar bias.

Uses PencilArray structure with data_real/data_imag arrays.
"""
function generate_random_magnetic!(mag_field, amplitude, modes_range)
    # Process toroidal and poloidal components
    for (spectral, is_poloidal) in ((mag_field.toroidal, false), (mag_field.poloidal, true))
        real_data = parent(spectral.data_real)
        imag_data = parent(spectral.data_imag)
        T = eltype(real_data)
        nr = size(real_data, 3)

        # Get local ranges
        cfg = spectral.config
        r_range = get_local_range(spectral.pencil, 3)

        # Clear fields
        fill!(real_data, zero(T))
        fill!(imag_data, zero(T))

        # Iterate ALL modes via the slot map so m>0 modes are populated.
        for lm in 1:cfg.nlm
            l = cfg.l_values[lm]
            slot = local_spectral_storage_slot(cfg, lm)
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

    return mag_field
end

"""
    generate_random_velocity!(vel_field, amplitude, modes_range)

Generate random velocity initial conditions.

Uses PencilArray structure with data_real/data_imag arrays.
"""
function generate_random_velocity!(vel_field, amplitude, modes_range)
    # Process toroidal and poloidal components
    for spectral in (vel_field.toroidal, vel_field.poloidal)
        real_data = parent(spectral.data_real)
        imag_data = parent(spectral.data_imag)
        T = eltype(real_data)
        nr = size(real_data, 3)

        # Get local ranges
        cfg = spectral.config
        r_range = get_local_range(spectral.pencil, 3)

        # Clear fields
        fill!(real_data, zero(T))
        fill!(imag_data, zero(T))

        # Iterate ALL modes via the slot map so m>0 modes are populated.
        for lm in 1:cfg.nlm
            l = cfg.l_values[lm]
            slot = local_spectral_storage_slot(cfg, lm)
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
    cfg = spectral.config
    r_range = get_local_range(spectral.pencil, 3)

    # Clear field first
    fill!(real_data, zero(T))
    fill!(imag_data, zero(T))

    # Iterate ALL modes via the slot map so m>0 modes are populated.
    for lm in 1:cfg.nlm
        l = cfg.l_values[lm]
        slot = local_spectral_storage_slot(cfg, lm)
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
        amplitude::Real = 1.0, geometry::Symbol = :shell, source = nothing,
        diffusivity::Real = 1.0, domain = nothing, parameters...)
    println("Setting analytical initial conditions:")
    println("  Field: $field_type")
    println("  Pattern: $pattern")
    println("  Amplitude: $amplitude")

    if field_type == :temperature
        set_analytical_temperature!(field, pattern, amplitude;
            geometry = geometry, source = source, diffusivity = diffusivity,
            domain = domain, parameters...)
    elseif field_type == :magnetic
        set_analytical_magnetic!(field, pattern, amplitude; parameters...)
    elseif field_type == :velocity
        set_analytical_velocity!(field, pattern, amplitude; parameters...)
    elseif field_type == :composition
        set_analytical_composition!(field, pattern, amplitude;
            geometry = geometry, source = source, diffusivity = diffusivity,
            domain = domain, parameters...)
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
function set_analytical_temperature!(temp_field, pattern::Symbol, amplitude;
        geometry::Symbol = :shell, source = nothing, diffusivity::Real = 1.0,
        domain = nothing, parameters...)
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
        if domain === nothing
            # Legacy path (no model context, e.g. a bare field built directly):
            # linear conductive profile amplitude·(1 − r_frac) on the l=0 mode.
            # The model/AnalyticIC path always threads `domain`, which selects the
            # BC + source-aware discrete-equilibrium profile below instead.
            for global_lm in lm_range
                if global_lm <= length(l_values) && l_values[global_lm] == 0
                    slot = local_spectral_storage_slot(spectral.config, global_lm)
                    slot === nothing && continue
                    for (local_r, global_r) in enumerate(r_range)
                        if local_r <= size(real_data, 3)
                            r_frac = (global_r - 1) / max(nr - 1, 1)
                            # Orthonormal SH (Y_0^0 = 1/√(4π)): store mean ×√(4π).
                            set_local_spectral_value!(real_data, slot, local_r,
                                sqrt(4 * T(π)) * T(amplitude * (1.0 - r_frac)))
                        end
                    end
                end
            end
        elseif geometry === :ball && source === nothing
            # Backward-compat: ball with no explicit heating uses the closed-form
            # 1 − r² conductive profile (mirroring initialize_temperature_field!'s
            # _ball_conductive_temperature), stored as the (0,0) coeff ×√(4π).
            # `amplitude` is IGNORED — the profile is fixed by geometry/BCs.
            s4pi = sqrt(4 * T(π))
            for global_lm in lm_range
                if global_lm <= length(l_values) && l_values[global_lm] == 0
                    slot = local_spectral_storage_slot(spectral.config, global_lm)
                    slot === nothing && continue
                    for (local_r, global_r) in enumerate(r_range)
                        local_r <= size(real_data, 3) || continue
                        r = domain.r[global_r, 4]
                        set_local_spectral_value!(real_data, slot, local_r,
                            s4pi * T(1 - r^2))
                    end
                end
            end
        else
            # BC + source-aware conductive (0,0) profile, shared with the
            # default-IC path. `amplitude` is IGNORED: the profile is fully
            # determined by the boundary values + source (discrete l=0 BVP).
            GEODYNAMO_PARENT.apply_scalar_conductive_l0!(temp_field, domain,
                geometry, source, diffusivity, 0.0)
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
    for spectral in (mag_field.toroidal, mag_field.poloidal)
        fill!(parent(spectral.data_real), zero(eltype(parent(spectral.data_real))))
        fill!(parent(spectral.data_imag), zero(eltype(parent(spectral.data_imag))))
    end

    if pattern == :dipole
        # Earth-like dipolar field: l=1 mode
        set_spectral_values!(mag_field.poloidal, 1, r -> amplitude * sin(π * r))
        set_spectral_values!(mag_field.toroidal, 1, r -> 0.1 * amplitude * sin(π * r))

    elseif pattern == :uniform_field
        direction = get(parameters, :direction, :z)
        if direction == :z
            set_spectral_values!(mag_field.poloidal, 0, r -> amplitude)
        elseif direction == :x
            set_spectral_values!(mag_field.poloidal, 1, r -> amplitude)
        end

    else
        throw(ArgumentError("Unknown magnetic pattern: $pattern"))
    end

    return mag_field
end

"""
    set_analytical_velocity!(vel_field, pattern, amplitude; parameters...)

Set analytical velocity patterns.

Uses PencilArray structure with data_real/data_imag arrays.
"""
function set_analytical_velocity!(vel_field, pattern::Symbol, amplitude; parameters...)
    # Clear both fields
    for spectral in (vel_field.toroidal, vel_field.poloidal)
        fill!(parent(spectral.data_real), zero(eltype(parent(spectral.data_real))))
        fill!(parent(spectral.data_imag), zero(eltype(parent(spectral.data_imag))))
    end

    if pattern == :convective
        # Small convective perturbations in low-order modes
        for spectral in (vel_field.toroidal, vel_field.poloidal)
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

    return vel_field
end

"""
    set_analytical_composition!(comp_field, pattern, amplitude; parameters...)

Set analytical composition patterns.

Uses PencilArray structure with data_real/data_imag arrays.
"""
function set_analytical_composition!(comp_field, pattern::Symbol, amplitude;
        geometry::Symbol = :shell, source = nothing, diffusivity::Real = 1.0,
        domain = nothing, parameters...)
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

    if pattern == :conductive
        # BC + source-aware conductive (0,0) profile, shared with the default-IC
        # path. `amplitude` is IGNORED: the profile is fully determined by the
        # boundary values + source. No geometry default source (default_H = 0).
        domain === nothing && throw(ArgumentError(
            "set_analytical_composition!(:conductive): a radial `domain` is " *
            "required (thread it through from the model's outer_core_domain)."))
        GEODYNAMO_PARENT.apply_scalar_conductive_l0!(comp_field, domain,
            geometry, source, diffusivity, 0.0)

    elseif pattern == :stratified
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
                            sqrt(4 * T(π)) *
                            T(bottom_comp + (top_comp - bottom_comp) * r_frac))
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

    return comp_field
end

# ================================================================================
# Saving Initial Conditions
# ================================================================================

"""
    save_initial_conditions(field, field_type::Symbol, file_path::String)

Save a field's spectral coefficients to an IC NetCDF file that
[`load_initial_conditions!`](@ref) can read back.

The file stores dense `(spectral_mode, r)` real/imag matrices per component
(one for scalars, toroidal + poloidal for vectors) plus a `field_type`
attribute used to validate the field on load. All ranks participate in the
gather; rank 0 writes the file.
"""
function save_initial_conditions(field, field_type::Symbol, file_path::String)
    components = _ic_components(field_type, field)
    first_spec = components[1][2]
    nlm = first_spec.config.nlm
    nr = size_global(first_spec.pencil)[3]

    # Gather every component to dense replicated matrices first so all ranks
    # take part in the reductions before only rank 0 touches the filesystem.
    gathered = [(name, _gather_full_spectral(spec)...) for (name, spec) in components]

    comm = get_comm()
    rank = MPI.Initialized() ? MPI.Comm_rank(comm) : 0
    if rank == 0
        dir = dirname(file_path)
        isempty(dir) || isdir(dir) || mkpath(dir)
        NCDataset(file_path, "c") do ds
            defDim(ds, "spectral_mode", nlm)
            defDim(ds, "r", nr)
            ds.attrib["field_type"] = string(field_type)
            ds.attrib["nlm"] = nlm
            ds.attrib["nr"] = nr
            for (name, full_real, full_imag) in gathered
                rvar = defVar(ds, "$(name)_real", Float64, ("spectral_mode", "r"))
                ivar = defVar(ds, "$(name)_imag", Float64, ("spectral_mode", "r"))
                rvar[:, :] = full_real
                ivar[:, :] = full_imag
            end
        end
    end
    if MPI.Initialized() && MPI.Comm_size(comm) > 1
        MPI.Barrier(comm)
    end
    return file_path
end

end # module InitialConditions
