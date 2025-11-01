# ================================================================================
# Grid Interpolation Utilities for Boundary Conditions
# ================================================================================

using LinearAlgebra
using Statistics

"""
    interpolate_boundary_to_grid(boundary_data::BoundaryData, target_theta::Vector{T}, 
                                target_phi::Vector{T}, time_index::Int=1) where T

Interpolate boundary data to a target grid using bilinear interpolation.
"""
function interpolate_boundary_to_grid(boundary_data::BoundaryData, target_theta::Vector{T}, 
                                    target_phi::Vector{T}, time_index::Int=1) where T
    
    if boundary_data.theta === nothing || boundary_data.phi === nothing
        throw(ArgumentError("Source boundary data must have coordinate information for interpolation"))
    end
    
    # Validate inputs
    if isempty(target_theta) || isempty(target_phi)
        throw(ArgumentError("Target coordinate arrays cannot be empty"))
    end
    
    if time_index < 1 || (boundary_data.is_time_dependent && time_index > boundary_data.ntime)
        throw(BoundsError("Time index $time_index is out of bounds for boundary data with $(boundary_data.ntime) time steps"))
    end
    
    # Check interpolation bounds and warn if needed
    check_interpolation_bounds(boundary_data, target_theta, target_phi)
    
    # Get source coordinates and data
    src_theta = boundary_data.theta
    src_phi = boundary_data.phi
    
    # Extract data for the specified time index
    if boundary_data.is_time_dependent
        if time_index > boundary_data.ntime
            time_index = boundary_data.ntime  # Clamp to available range
        end
        if boundary_data.ncomponents == 1
            src_data = boundary_data.values[:, :, time_index]
        else
            src_data = boundary_data.values[:, :, time_index, :]
        end
    else
        if boundary_data.ncomponents == 1
            src_data = boundary_data.values
        else
            src_data = boundary_data.values
        end
    end
    
    # Initialize output array
    if boundary_data.ncomponents == 1
        interpolated = zeros(T, length(target_theta), length(target_phi))
    else
        interpolated = zeros(T, length(target_theta), length(target_phi), boundary_data.ncomponents)
    end
    
    # Perform bilinear interpolation
    for (i, theta_t) in enumerate(target_theta)
        for (j, phi_t) in enumerate(target_phi)
            
            # Find surrounding points in source grid
            theta_idx = find_grid_indices(src_theta, theta_t, is_periodic=false)  # theta is not periodic
            phi_idx = find_grid_indices(src_phi, phi_t, is_periodic=true)         # phi is periodic
            
            # Get interpolation weights
            theta_weights = get_interpolation_weights(src_theta, theta_t, theta_idx)
            phi_weights = get_interpolation_weights(src_phi, phi_t, phi_idx)
            
            # Perform bilinear interpolation
            if boundary_data.ncomponents == 1
                interpolated[i, j] = bilinear_interpolate(
                    src_data, theta_idx, phi_idx, theta_weights, phi_weights
                )
            else
                for comp in 1:boundary_data.ncomponents
                    interpolated[i, j, comp] = bilinear_interpolate(
                        src_data[:, :, comp], theta_idx, phi_idx, theta_weights, phi_weights
                    )
                end
            end
        end
    end
    
    return interpolated
end

"""
    find_grid_indices(coords::Vector{T}, target::T; is_periodic::Bool=false) where T

Find the two surrounding indices in a coordinate array for interpolation.
Handles periodic coordinates (e.g., longitude) if is_periodic=true.
"""
function find_grid_indices(coords::Vector{T}, target::T; is_periodic::Bool=false) where T
    n = length(coords)
    
    if n < 2
        throw(ArgumentError("Coordinate array must have at least 2 points"))
    end
    
    # Handle periodic coordinates (e.g., longitude)
    if is_periodic
        period = coords[end] - coords[1] + (coords[2] - coords[1])  # Assume uniform spacing
        
        # Wrap target to coordinate range
        while target < coords[1]
            target += period
        end
        while target > coords[end] + (coords[2] - coords[1])
            target -= period
        end
        
        # Check if target is beyond the last point but within one grid spacing
        if target > coords[end]
            return (n, 1)  # Wrap to beginning
        end
    end
    
    # Handle edge cases
    if target <= coords[1]
        return (1, min(2, n))
    elseif target >= coords[end]
        return (max(1, n-1), n)
    end
    
    # Binary search for surrounding indices
    low, high = 1, n
    while high - low > 1
        mid = (low + high) ÷ 2
        if coords[mid] <= target
            low = mid
        else
            high = mid
        end
    end
    
    return (low, high)
end

"""
    get_interpolation_weights(coords::Vector{T}, target::T, indices::Tuple{Int, Int}) where T

Calculate interpolation weights for linear interpolation.
"""
function get_interpolation_weights(coords::Vector{T}, target::T, indices::Tuple{Int, Int}) where T
    i1, i2 = indices
    
    if i1 == i2
        return (1.0, 0.0)
    end
    
    dx = coords[i2] - coords[i1]
    w2 = (target - coords[i1]) / dx
    w1 = 1.0 - w2
    
    return (w1, w2)
end

"""
    bilinear_interpolate(data::Matrix{T}, theta_idx::Tuple{Int, Int}, phi_idx::Tuple{Int, Int},
                        theta_weights::Tuple{T, T}, phi_weights::Tuple{T, T}) where T

Perform bilinear interpolation on a 2D data array.
Handles periodic boundary conditions in phi direction.
"""
function bilinear_interpolate(data::Matrix{T}, theta_idx::Tuple{Int, Int}, phi_idx::Tuple{Int, Int},
                            theta_weights::Tuple{T, T}, phi_weights::Tuple{T, T}) where T
    
    i1, i2 = theta_idx
    j1, j2 = phi_idx
    wt1, wt2 = theta_weights
    wp1, wp2 = phi_weights
    
    nlat, nlon = size(data)
    
    # Handle periodic boundary in phi (longitude) direction
    # If j2 would wrap around, use j2 = 1 (periodic boundary)
    if j2 > nlon
        j2 = 1
    elseif j2 < 1
        j2 = nlon
    end
    
    # Ensure theta indices are within bounds
    i1 = clamp(i1, 1, nlat)
    i2 = clamp(i2, 1, nlat)
    j1 = clamp(j1, 1, nlon)
    
    # Get the four surrounding points
    f11 = data[i1, j1]
    f12 = data[i1, j2]
    f21 = data[i2, j1]
    f22 = data[i2, j2]
    
    # Bilinear interpolation
    result = wt1 * wp1 * f11 + wt1 * wp2 * f12 + wt2 * wp1 * f21 + wt2 * wp2 * f22
    
    return result
end

"""
    create_interpolation_cache(boundary_data::BoundaryData, target_theta::Vector{T}, 
                              target_phi::Vector{T}) where T

Create interpolation cache for efficient repeated interpolations.
"""
function create_interpolation_cache(boundary_data::BoundaryData, target_theta::Vector{T}, 
                                  target_phi::Vector{T}) where T
    
    cache = Dict{String, Any}()
    
    if boundary_data.theta === nothing || boundary_data.phi === nothing
        return cache  # No interpolation needed
    end
    
    src_theta = boundary_data.theta
    src_phi = boundary_data.phi
    
    # Pre-compute interpolation indices and weights
    theta_indices = Vector{Tuple{Int, Int}}(undef, length(target_theta))
    theta_weights = Vector{Tuple{T, T}}(undef, length(target_theta))
    phi_indices = Vector{Tuple{Int, Int}}(undef, length(target_phi))
    phi_weights = Vector{Tuple{T, T}}(undef, length(target_phi))
    
    for (i, theta_t) in enumerate(target_theta)
        theta_indices[i] = find_grid_indices(src_theta, theta_t, is_periodic=false)
        theta_weights[i] = get_interpolation_weights(src_theta, theta_t, theta_indices[i])
    end
    
    for (j, phi_t) in enumerate(target_phi)
        phi_indices[j] = find_grid_indices(src_phi, phi_t, is_periodic=true)
        phi_weights[j] = get_interpolation_weights(src_phi, phi_t, phi_indices[j])
    end
    
    cache["theta_indices"] = theta_indices
    cache["theta_weights"] = theta_weights
    cache["phi_indices"] = phi_indices
    cache["phi_weights"] = phi_weights
    cache["target_shape"] = (length(target_theta), length(target_phi))
    
    return cache
end

"""
    interpolate_with_cache(boundary_data::BoundaryData, cache::Dict, time_index::Int=1)

Perform interpolation using pre-computed cache for efficiency.
"""
function interpolate_with_cache(boundary_data::BoundaryData, cache::Dict, time_index::Int=1)
    
    if isempty(cache)
        # No interpolation needed, return data as-is
        if boundary_data.is_time_dependent
            if boundary_data.ncomponents == 1
                return boundary_data.values[:, :, time_index]
            else
                return boundary_data.values[:, :, time_index, :]
            end
        else
            return boundary_data.values
        end
    end
    
    # Extract cached interpolation data
    theta_indices = cache["theta_indices"]
    theta_weights = cache["theta_weights"]
    phi_indices = cache["phi_indices"]
    phi_weights = cache["phi_weights"]
    nlat_tgt, nlon_tgt = cache["target_shape"]
    
    # Extract source data for the specified time index
    if boundary_data.is_time_dependent
        if time_index > boundary_data.ntime
            time_index = boundary_data.ntime
        end
        if boundary_data.ncomponents == 1
            src_data = boundary_data.values[:, :, time_index]
        else
            src_data = boundary_data.values[:, :, time_index, :]
        end
    else
        if boundary_data.ncomponents == 1
            src_data = boundary_data.values
        else
            src_data = boundary_data.values
        end
    end
    
    # Initialize output array
    if boundary_data.ncomponents == 1
        interpolated = zeros(eltype(boundary_data.values), nlat_tgt, nlon_tgt)
    else
        interpolated = zeros(eltype(boundary_data.values), nlat_tgt, nlon_tgt, boundary_data.ncomponents)
    end
    
    # Perform cached interpolation
    for i in 1:nlat_tgt
        for j in 1:nlon_tgt
            if boundary_data.ncomponents == 1
                interpolated[i, j] = bilinear_interpolate(
                    src_data, theta_indices[i], phi_indices[j], 
                    theta_weights[i], phi_weights[j]
                )
            else
                for comp in 1:boundary_data.ncomponents
                    interpolated[i, j, comp] = bilinear_interpolate(
                        src_data[:, :, comp], theta_indices[i], phi_indices[j],
                        theta_weights[i], phi_weights[j]
                    )
                end
            end
        end
    end
    
    return interpolated
end

"""
    validate_interpolation_grids(src_theta::Vector, src_phi::Vector, 
                                tgt_theta::Vector, tgt_phi::Vector)

Validate that source and target grids are compatible for interpolation.
"""
function validate_interpolation_grids(src_theta::Vector, src_phi::Vector, 
                                     tgt_theta::Vector, tgt_phi::Vector)
    
    errors = String[]
    
    # Check coordinate ranges
    if minimum(tgt_theta) < minimum(src_theta) || maximum(tgt_theta) > maximum(src_theta)
        push!(errors, "Target theta range exceeds source range")
    end
    
    if minimum(tgt_phi) < minimum(src_phi) || maximum(tgt_phi) > maximum(src_phi)
        push!(errors, "Target phi range exceeds source range")
    end
    
    # Check for monotonicity
    if !issorted(src_theta)
        push!(errors, "Source theta coordinates are not monotonic")
    end
    
    if !issorted(src_phi)
        push!(errors, "Source phi coordinates are not monotonic")
    end
    
    if !issorted(tgt_theta)
        push!(errors, "Target theta coordinates are not monotonic")
    end
    
    if !issorted(tgt_phi)
        push!(errors, "Target phi coordinates are not monotonic")
    end
    
    if !isempty(errors)
        error_msg = "Grid interpolation validation failed:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
    
    return true
end

"""
    check_interpolation_bounds(boundary_data::BoundaryData, target_theta::Vector{T}, 
                              target_phi::Vector{T}) where T

Check if target grid is within the bounds of the source grid and warn if extrapolation is needed.
"""
function check_interpolation_bounds(boundary_data::BoundaryData, target_theta::Vector{T}, 
                                   target_phi::Vector{T}) where T
    
    if boundary_data.theta === nothing || boundary_data.phi === nothing
        return  # No coordinate info available
    end
    
    src_theta_min, src_theta_max = extrema(boundary_data.theta)
    src_phi_min, src_phi_max = extrema(boundary_data.phi)
    tgt_theta_min, tgt_theta_max = extrema(target_theta)
    tgt_phi_min, tgt_phi_max = extrema(target_phi)
    
    # Check theta bounds
    if tgt_theta_min < src_theta_min || tgt_theta_max > src_theta_max
        @warn "Target theta range [$tgt_theta_min, $tgt_theta_max] extends beyond source range [$src_theta_min, $src_theta_max]. Extrapolation will be used."
    end
    
    # Check phi bounds (accounting for periodicity)
    phi_range = src_phi_max - src_phi_min
    if phi_range < 2π - 0.1  # Not a full periodic range
        if tgt_phi_min < src_phi_min || tgt_phi_max > src_phi_max
            @warn "Target phi range [$tgt_phi_min, $tgt_phi_max] extends beyond source range [$src_phi_min, $src_phi_max]. Extrapolation will be used."
        end
    end
end

"""
    get_interpolation_statistics(boundary_data::BoundaryData, interpolated_data::Array{T}) where T

Compute statistics comparing original and interpolated data.
"""
function get_interpolation_statistics(boundary_data::BoundaryData, interpolated_data::Array{T}) where T
    
    if boundary_data.is_time_dependent
        # For time-dependent data, use the first time step for comparison
        src_data = boundary_data.ncomponents == 1 ? boundary_data.values[:, :, 1] : boundary_data.values[:, :, 1, :]
    else
        src_data = boundary_data.values
    end
    
    # For interpolated data, use appropriate slice
    if boundary_data.ncomponents == 1
        interp_slice = interpolated_data
    else
        interp_slice = interpolated_data[:, :, :]
    end
    
    # Compute basic statistics
    src_min, src_max = extrema(src_data)
    src_mean = mean(src_data)
    src_std = std(src_data)
    
    interp_min, interp_max = extrema(interp_slice)
    interp_mean = mean(interp_slice)
    interp_std = std(interp_slice)
    
    return Dict(
        "source_range" => (src_min, src_max),
        "source_mean" => src_mean,
        "source_std" => src_std,
        "interpolated_range" => (interp_min, interp_max),
        "interpolated_mean" => interp_mean,
        "interpolated_std" => interp_std,
        "range_preservation" => (interp_min >= src_min - 1e-10 && interp_max <= src_max + 1e-10)
    )
end

"""
    estimate_interpolation_error(boundary_data::BoundaryData, target_theta::Vector{T}, 
                                target_phi::Vector{T}) where T

Estimate interpolation error based on grid resolution differences.
"""
function estimate_interpolation_error(boundary_data::BoundaryData, target_theta::Vector{T}, 
                                     target_phi::Vector{T}) where T
    
    if boundary_data.theta === nothing || boundary_data.phi === nothing
        return Dict("error" => "No coordinate information available")
    end
    
    # Compute grid spacings
    src_dtheta = length(boundary_data.theta) > 1 ? (boundary_data.theta[end] - boundary_data.theta[1]) / (length(boundary_data.theta) - 1) : 0.0
    src_dphi = length(boundary_data.phi) > 1 ? (boundary_data.phi[end] - boundary_data.phi[1]) / (length(boundary_data.phi) - 1) : 0.0
    
    tgt_dtheta = length(target_theta) > 1 ? (target_theta[end] - target_theta[1]) / (length(target_theta) - 1) : 0.0
    tgt_dphi = length(target_phi) > 1 ? (target_phi[end] - target_phi[1]) / (length(target_phi) - 1) : 0.0
    
    # Estimate relative error based on grid spacing ratios
    theta_error_est = src_dtheta > 0 ? abs(tgt_dtheta - src_dtheta) / src_dtheta : 0.0
    phi_error_est = src_dphi > 0 ? abs(tgt_dphi - src_dphi) / src_dphi : 0.0
    
    return Dict(
        "source_resolution" => (src_dtheta, src_dphi),
        "target_resolution" => (tgt_dtheta, tgt_dphi),
        "relative_error_estimate" => (theta_error_est, phi_error_est),
        "interpolation_quality" => theta_error_est < 0.1 && phi_error_est < 0.1 ? "good" : "fair"
    )
end

export interpolate_boundary_to_grid, create_interpolation_cache, interpolate_with_cache
export validate_interpolation_grids, check_interpolation_bounds
export get_interpolation_statistics, estimate_interpolation_error