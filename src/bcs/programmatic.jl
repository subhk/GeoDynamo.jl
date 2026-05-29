# ================================================================================
# Programmatic Boundary Pattern Generation
# ================================================================================
#
# Utility for generating spherical harmonic patterns (Ylm)
# as BoundaryData structs. These can be used for:
#   - Setting non-homogeneous RHS boundary values
#   - Initial condition generation
#   - Testing and visualization
#
# Note: BC enforcement is done through the matrix-embedded system
# (see bcs/thermal_bc.jl, bcs/compositional_bc.jl, bcs/velocity_bc.jl, bcs/magnetic_bc.jl).
# The BoundaryData structs from this file are not directly used for BC enforcement.
#
# Note: This file is included within the bcs module
# All necessary packages are imported at the module level

using SHTnsKit

# ================================================================================
# Spherical Harmonic Pattern Type
# ================================================================================

"""
    Ylm(l::Int, m::Int)

Spherical harmonic pattern specifier for programmatic boundary conditions.

# Examples
```julia
# Create Y₂¹ pattern with amplitude 0.5
boundary = create_programmatic_boundary(Ylm(2, 1), config, 0.5)

# Create Y₄⁻² pattern
boundary = create_programmatic_boundary(Ylm(4, -2), config, 1.0)

# Time-dependent oscillating Y₃² pattern
boundary = create_time_dependent_programmatic_boundary(Ylm(3, 2), config, (0.0, 10.0), 100;
    amplitude=0.3)
```
"""
struct Ylm
    l::Int
    m::Int

    function Ylm(l::Int, m::Int)
        l >= 0 || throw(ArgumentError("Spherical harmonic degree l must be non-negative, got l=$l"))
        abs(m) <= l || throw(ArgumentError("Spherical harmonic order |m| must be ≤ l, got l=$l, m=$m"))
        new(l, m)
    end
end

Base.show(io::IO, ylm::Ylm) = print(io, "Y_$(ylm.l)^$(ylm.m)")

"""
    create_programmatic_boundary(ylm::Ylm, config, amplitude::Real=1.0; kwargs...)

Create a boundary pattern from a spherical harmonic Yₗᵐ using SHTnsKit synthesis.
Values are produced on the Gauss-Legendre grid of the given config.

# Examples
```julia
boundary = create_programmatic_boundary(Ylm(2, 1), config, 0.5)
boundary = create_programmatic_boundary(Ylm(4, -2), config, 1.0)
```
"""
function create_programmatic_boundary(ylm::Ylm, config, amplitude::Real=1.0;
                                    description::String="", field_type::String="temperature")
    l, m = ylm.l, ylm.m

    if l > config.lmax
        throw(ArgumentError("Spherical harmonic degree l=$l exceeds config.lmax=$(config.lmax)"))
    end

    # Use actual SHTnsKit grid coordinates (Gauss-Legendre theta, equispaced phi)
    # since SHTnsKit.synthesis produces values on this grid
    nlat, nlon = config.nlat, config.nlon
    theta = copy(config.theta_grid)
    phi = copy(config.phi_grid)

    # Create SHTnsKit coefficient matrix (lmax+1 × mmax+1, complex)
    lmax = config.lmax
    mmax = hasfield(typeof(config), :mmax) ? config.mmax : lmax
    coeffs = zeros(Complex{Float64}, lmax + 1, mmax + 1)

    # Set the (l, m) coefficient
    # SHTnsKit uses 1-based indexing: coeffs[l+1, |m|+1]
    if m >= 0
        coeffs[l + 1, m + 1] = Complex{Float64}(amplitude, 0.0)
    else
        # For negative m, use conjugate symmetry: Y_l^{-m} = (-1)^m conj(Y_l^m)
        phase = iseven(-m) ? 1.0 : -1.0
        coeffs[l + 1, abs(m) + 1] = Complex{Float64}(phase * amplitude, 0.0)
    end

    # Use SHTnsKit synthesis to convert to physical space
    values_complex = SHTnsKit.synthesis(config.sht_config, coeffs; real_output=false)

    # Extract appropriate component based on m
    values = zeros(eltype(amplitude), nlat, nlon)
    if m >= 0
        values .= real.(values_complex)
    else
        # For m < 0: sin(|m|*φ) component
        values .= imag.(values_complex)
    end

    # Create description if not provided
    if isempty(description)
        description = "Spherical harmonic Y_$(l)^$(m) (amplitude=$amplitude)"
    end

    return create_boundary_data(
        values, field_type;
        theta=theta, phi=phi, time=nothing,
        units=get_default_units(determine_field_type_from_name(field_type)),
        description=description,
        file_path="programmatic"
    )
end

"""
    create_time_dependent_programmatic_boundary(ylm::Ylm, config, time_span, ntime; kwargs...)

Create a time-dependent boundary pattern from a spherical harmonic Yₗᵐ with
cosine time modulation. Values are produced on the Gauss-Legendre grid.

# Example
```julia
boundary = create_time_dependent_programmatic_boundary(Ylm(3, 2), config, (0.0, 10.0), 100;
    amplitude=0.5, time_factor=2π)
```
"""
function create_time_dependent_programmatic_boundary(ylm::Ylm, config,
                                                   time_span::Tuple{Real, Real}, ntime::Int;
                                                   amplitude::Real=1.0, time_factor::Real=1.0,
                                                   phase_offset::Real=0.0, description::String="",
                                                   field_type::String="temperature")
    l, m = ylm.l, ylm.m

    if l > config.lmax
        throw(ArgumentError("Spherical harmonic degree l=$l exceeds config.lmax=$(config.lmax)"))
    end

    # Use actual SHTnsKit grid coordinates (Gauss-Legendre theta, equispaced phi)
    nlat, nlon = config.nlat, config.nlon
    theta = copy(config.theta_grid)
    phi = copy(config.phi_grid)
    time_coords = collect(range(time_span[1], time_span[2], length=ntime))

    # Initialize data array
    values = zeros(eltype(amplitude), nlat, nlon, ntime)

    # Create SHTnsKit coefficient matrix
    lmax = config.lmax
    mmax = hasfield(typeof(config), :mmax) ? config.mmax : lmax

    for (t, time_val) in enumerate(time_coords)
        time_phase = time_factor * time_val + phase_offset
        time_modulated_amplitude = amplitude * cos(time_phase)

        coeffs = zeros(Complex{Float64}, lmax + 1, mmax + 1)

        if m >= 0
            coeffs[l + 1, m + 1] = Complex{Float64}(time_modulated_amplitude, 0.0)
        else
            phase = iseven(-m) ? 1.0 : -1.0
            coeffs[l + 1, abs(m) + 1] = Complex{Float64}(phase * time_modulated_amplitude, 0.0)
        end

        values_complex = SHTnsKit.synthesis(config.sht_config, coeffs; real_output=false)

        if m >= 0
            values[:, :, t] .= real.(values_complex)
        else
            values[:, :, t] .= imag.(values_complex)
        end
    end

    if isempty(description)
        description = "Time-dependent spherical harmonic Y_$(l)^$(m) (amplitude=$amplitude)"
    end

    return create_boundary_data(
        values, field_type;
        theta=theta, phi=phi, time=time_coords,
        units=get_default_units(determine_field_type_from_name(field_type)),
        description=description,
        file_path="programmatic"
    )
end

"""
    add_noise_to_boundary(boundary_data::BoundaryData, noise_amplitude::Real, 
                         noise_type::Symbol=:gaussian)

Add noise to existing boundary data.
"""
function add_noise_to_boundary(boundary_data::BoundaryData, noise_amplitude::Real, 
                              noise_type::Symbol=:gaussian)
    
    noisy_values = copy(boundary_data.values)
    
    if noise_type == :gaussian
        noise = noise_amplitude * randn(size(noisy_values))
    elseif noise_type == :uniform
        noise = noise_amplitude * (2 * rand(size(noisy_values)) .- 1)
    else
        throw(ArgumentError("Unknown noise type: $noise_type"))
    end
    
    noisy_values .+= noise
    
    # Create new boundary data with noise added
    return BoundaryData(
        boundary_data.theta, boundary_data.phi, boundary_data.time,
        noisy_values, boundary_data.units, 
        boundary_data.description * " + $(noise_type) noise",
        boundary_data.file_path, boundary_data.field_type,
        boundary_data.is_time_dependent, boundary_data.nlat, 
        boundary_data.nlon, boundary_data.ntime, boundary_data.ncomponents
    )
end

"""
    smooth_boundary_data(boundary_data::BoundaryData, smoothing_radius::Real)

Apply spatial smoothing to boundary data.
"""
function smooth_boundary_data(boundary_data::BoundaryData, smoothing_radius::Real)
    
    smoothed_values = copy(boundary_data.values)
    
    if boundary_data.theta === nothing || boundary_data.phi === nothing
        @warn "Cannot smooth boundary data without coordinate information"
        return boundary_data
    end
    
    theta = boundary_data.theta
    phi = boundary_data.phi
    
    # Apply Gaussian smoothing kernel
    for time_idx in 1:boundary_data.ntime
        if boundary_data.is_time_dependent
            if boundary_data.ncomponents == 1
                data_slice = smoothed_values[:, :, time_idx]
            else
                # Smooth each component separately
                for comp in 1:boundary_data.ncomponents
                    data_slice = @view smoothed_values[:, :, time_idx, comp]
                    apply_gaussian_smoothing!(data_slice, theta, phi, smoothing_radius)
                end
                continue
            end
        else
            if boundary_data.ncomponents == 1
                data_slice = smoothed_values
            else
                for comp in 1:boundary_data.ncomponents
                    data_slice = @view smoothed_values[:, :, comp]
                    apply_gaussian_smoothing!(data_slice, theta, phi, smoothing_radius)
                end
                continue
            end
        end
        
        apply_gaussian_smoothing!(data_slice, theta, phi, smoothing_radius)
    end
    
    # Create new boundary data with smoothed values
    return BoundaryData(
        boundary_data.theta, boundary_data.phi, boundary_data.time,
        smoothed_values, boundary_data.units,
        boundary_data.description * " (smoothed)",
        boundary_data.file_path, boundary_data.field_type,
        boundary_data.is_time_dependent, boundary_data.nlat,
        boundary_data.nlon, boundary_data.ntime, boundary_data.ncomponents
    )
end

"""
    apply_gaussian_smoothing!(data::AbstractMatrix, theta::Vector, phi::Vector, radius::Real)

Apply Gaussian smoothing kernel to 2D data array.
"""
function apply_gaussian_smoothing!(data::AbstractMatrix, theta::Vector, phi::Vector, radius::Real)

    nlat, nlon = size(data)
    original_data = copy(data)

    # Cutoff at 3σ — beyond this the Gaussian weight is < exp(-9) ≈ 1.2e-4
    cutoff = 3.0 * radius

    # Precompute cos/sin of theta for efficiency
    cos_theta = cos.(theta)
    sin_theta = sin.(theta)

    for i in 1:nlat
        for j in 1:nlon
            weighted_sum = 0.0
            weight_total = 0.0

            for ii in 1:nlat
                # Quick latitude-only distance check to skip far-away rows
                lat_dist = abs(theta[i] - theta[ii])
                if lat_dist > cutoff
                    continue
                end

                for jj in 1:nlon
                    # Calculate angular distance
                    cos_angle = cos_theta[i] * cos_theta[ii] +
                                sin_theta[i] * sin_theta[ii] *
                                cos(phi[j] - phi[jj])
                    cos_angle = clamp(cos_angle, -1.0, 1.0)
                    angular_dist = acos(cos_angle)

                    if angular_dist > cutoff
                        continue
                    end

                    # Gaussian weight
                    weight = exp(-(angular_dist / radius)^2)

                    weighted_sum += weight * original_data[ii, jj]
                    weight_total += weight
                end
            end

            data[i, j] = weighted_sum / weight_total
        end
    end
end

# ================================================================================
# Programmatic Boundary Set Creation (for Shell API)
# ================================================================================

"""
    ProgrammaticBoundarySet{T}

Wrapper around BoundaryConditionSet that also stores BC types for each boundary.
Returned by `create_programmatic_temperature_boundaries` and
`create_programmatic_composition_boundaries`.
"""
struct ProgrammaticBoundarySet{T<:AbstractFloat}
    boundary_set::BoundaryConditionSet{T}
    inner_bc_type::BoundaryType
    outer_bc_type::BoundaryType
end

"""
    __parse_boundary_spec(spec::Tuple, cfg) -> (values::Matrix, bc_type::BoundaryType)

Parse a boundary specification tuple into physical-space values and BC type.

Supported spec formats:
- `(:uniform, value)` or `(:dirichlet, value)`: Dirichlet BC with uniform value
- `(:neumann, value)`: Neumann BC with uniform flux value
"""
function __parse_boundary_spec(spec::Tuple, cfg)
    pattern_type = spec[1]::Symbol
    value = Float64(spec[2])

    if pattern_type == :uniform || pattern_type == :dirichlet
        values = fill(value, cfg.nlat, cfg.nlon)
        bc_type = DIRICHLET
    elseif pattern_type == :neumann
        values = fill(value, cfg.nlat, cfg.nlon)
        bc_type = NEUMANN
    else
        throw(ArgumentError("Unknown boundary pattern type: $pattern_type. " *
            "Supported types: :uniform, :dirichlet, :neumann"))
    end

    return values, bc_type
end

"""
    create_programmatic_temperature_boundaries(inner_spec::Tuple, outer_spec::Tuple, cfg)

Create a `ProgrammaticBoundarySet` for temperature from programmatic specifications.

# Arguments
- `inner_spec`: Tuple specifying inner boundary, e.g. `(:uniform, 100.0)`
- `outer_spec`: Tuple specifying outer boundary, e.g. `(:uniform, 250.0)`
- `cfg`: SHTnsKitConfig with grid parameters

# Returns
A `ProgrammaticBoundarySet{Float64}` containing boundary data and BC types.

# Examples
```julia
bset = create_programmatic_temperature_boundaries((:uniform, 1.0), (:uniform, 0.0), cfg)
bset = create_programmatic_temperature_boundaries((:dirichlet, 1.0), (:neumann, 0.0), cfg)
```
"""
function create_programmatic_temperature_boundaries(inner_spec::Tuple, outer_spec::Tuple, cfg)
    inner_values, inner_bc_type = __parse_boundary_spec(inner_spec, cfg)
    outer_values, outer_bc_type = __parse_boundary_spec(outer_spec, cfg)

    inner_data = create_boundary_data(
        inner_values, "temperature";
        theta=copy(cfg.theta_grid), phi=copy(cfg.phi_grid), time=nothing,
        units=get_default_units(TEMPERATURE),
        description="Programmatic $(inner_spec[1]) boundary (value=$(inner_spec[2]))",
        file_path="programmatic"
    )

    outer_data = create_boundary_data(
        outer_values, "temperature";
        theta=copy(cfg.theta_grid), phi=copy(cfg.phi_grid), time=nothing,
        units=get_default_units(TEMPERATURE),
        description="Programmatic $(outer_spec[1]) boundary (value=$(outer_spec[2]))",
        file_path="programmatic"
    )

    bcs_set = BoundaryConditionSet{Float64}(inner_data, outer_data, "temperature", TEMPERATURE, time())
    return ProgrammaticBoundarySet{Float64}(bcs_set, inner_bc_type, outer_bc_type)
end

"""
    create_programmatic_composition_boundaries(inner_spec::Tuple, outer_spec::Tuple, cfg)

Create a `ProgrammaticBoundarySet` for composition from programmatic specifications.
Same interface as `create_programmatic_temperature_boundaries`.
"""
function create_programmatic_composition_boundaries(inner_spec::Tuple, outer_spec::Tuple, cfg)
    inner_values, inner_bc_type = __parse_boundary_spec(inner_spec, cfg)
    outer_values, outer_bc_type = __parse_boundary_spec(outer_spec, cfg)

    inner_data = create_boundary_data(
        inner_values, "composition";
        theta=copy(cfg.theta_grid), phi=copy(cfg.phi_grid), time=nothing,
        units=get_default_units(COMPOSITION),
        description="Programmatic $(inner_spec[1]) boundary (value=$(inner_spec[2]))",
        file_path="programmatic"
    )

    outer_data = create_boundary_data(
        outer_values, "composition";
        theta=copy(cfg.theta_grid), phi=copy(cfg.phi_grid), time=nothing,
        units=get_default_units(COMPOSITION),
        description="Programmatic $(outer_spec[1]) boundary (value=$(outer_spec[2]))",
        file_path="programmatic"
    )

    bcs_set = BoundaryConditionSet{Float64}(inner_data, outer_data, "composition", COMPOSITION, time())
    return ProgrammaticBoundarySet{Float64}(bcs_set, inner_bc_type, outer_bc_type)
end

"""
    load_temperature_boundaries_from_files(inner_file::String, outer_file::String, cfg)

Load temperature boundary conditions from two NetCDF files.
"""
function load_temperature_boundaries_from_files(inner_file::String, outer_file::String, cfg)
    inner_data = read_netcdf_boundary_data(inner_file)
    outer_data = read_netcdf_boundary_data(outer_file)
    bcs_set = BoundaryConditionSet{Float64}(inner_data, outer_data, "temperature", TEMPERATURE, time())
    return ProgrammaticBoundarySet{Float64}(bcs_set, DIRICHLET, DIRICHLET)
end

"""
    load_composition_boundaries_from_files(inner_file::String, outer_file::String, cfg)

Load composition boundary conditions from two NetCDF files.
"""
function load_composition_boundaries_from_files(inner_file::String, outer_file::String, cfg)
    inner_data = read_netcdf_boundary_data(inner_file)
    outer_data = read_netcdf_boundary_data(outer_file)
    bcs_set = BoundaryConditionSet{Float64}(inner_data, outer_data, "composition", COMPOSITION, time())
    return ProgrammaticBoundarySet{Float64}(bcs_set, DIRICHLET, DIRICHLET)
end

"""
    create_hybrid_temperature_boundaries(file_spec::String, prog_spec::Tuple, cfg; swap_boundaries=false)

Create temperature boundaries with one from file and one programmatic.
When `swap_boundaries=false`, file is inner and programmatic is outer.
When `swap_boundaries=true`, file is outer and programmatic is inner.
"""
function create_hybrid_temperature_boundaries(file_spec::String, prog_spec::Tuple, cfg; swap_boundaries::Bool=false)
    file_data = read_netcdf_boundary_data(file_spec)
    prog_values, prog_bc_type = __parse_boundary_spec(prog_spec, cfg)
    prog_data = create_boundary_data(
        prog_values, "temperature";
        theta=copy(cfg.theta_grid), phi=copy(cfg.phi_grid), time=nothing,
        units=get_default_units(TEMPERATURE),
        description="Programmatic $(prog_spec[1]) boundary (value=$(prog_spec[2]))",
        file_path="programmatic"
    )
    if swap_boundaries
        bcs_set = BoundaryConditionSet{Float64}(prog_data, file_data, "temperature", TEMPERATURE, time())
        return ProgrammaticBoundarySet{Float64}(bcs_set, prog_bc_type, DIRICHLET)
    else
        bcs_set = BoundaryConditionSet{Float64}(file_data, prog_data, "temperature", TEMPERATURE, time())
        return ProgrammaticBoundarySet{Float64}(bcs_set, DIRICHLET, prog_bc_type)
    end
end

"""
    create_hybrid_composition_boundaries(file_spec::String, prog_spec::Tuple, cfg; swap_boundaries=false)

Create composition boundaries with one from file and one programmatic.
"""
function create_hybrid_composition_boundaries(file_spec::String, prog_spec::Tuple, cfg; swap_boundaries::Bool=false)
    file_data = read_netcdf_boundary_data(file_spec)
    prog_values, prog_bc_type = __parse_boundary_spec(prog_spec, cfg)
    prog_data = create_boundary_data(
        prog_values, "composition";
        theta=copy(cfg.theta_grid), phi=copy(cfg.phi_grid), time=nothing,
        units=get_default_units(COMPOSITION),
        description="Programmatic $(prog_spec[1]) boundary (value=$(prog_spec[2]))",
        file_path="programmatic"
    )
    if swap_boundaries
        bcs_set = BoundaryConditionSet{Float64}(prog_data, file_data, "composition", COMPOSITION, time())
        return ProgrammaticBoundarySet{Float64}(bcs_set, prog_bc_type, DIRICHLET)
    else
        bcs_set = BoundaryConditionSet{Float64}(file_data, prog_data, "composition", COMPOSITION, time())
        return ProgrammaticBoundarySet{Float64}(bcs_set, DIRICHLET, prog_bc_type)
    end
end

export Ylm, ProgrammaticBoundarySet
export create_programmatic_boundary, create_time_dependent_programmatic_boundary
export add_noise_to_boundary, smooth_boundary_data
export create_programmatic_temperature_boundaries, create_programmatic_composition_boundaries
export create_hybrid_temperature_boundaries, create_hybrid_composition_boundaries
export load_temperature_boundaries_from_files, load_composition_boundaries_from_files