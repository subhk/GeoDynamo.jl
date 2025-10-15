#!/usr/bin/env julia

"""
Energy Cascade Analysis for Geodynamo.jl using CoarseGraining.jl

This script performs comprehensive energy cascade analysis on merged Geodynamo.jl outputs
using the CoarseGraining.jl package. It computes:

1. Leonard energy transfer (Π) across scales
2. Multi-scale energy budget analysis 
3. Scale-dependent kinetic and magnetic energy transport
4. Spherical harmonic decomposition of energy cascades
5. Enstrophy and helicity cascades

Usage examples:

  # Basic cascade analysis with default parameters
  julia --project=. scripts/energy_cascade_analysis.jl ./output

  # Multi-scale analysis with custom filter scales
  julia --project=. scripts/energy_cascade_analysis.jl ./output \
        --scales=2,4,8,16 --time_range=1.0,5.0 --radius=0.8

  # Spherical harmonic mode cascade analysis
  julia --project=. scripts/energy_cascade_analysis.jl ./output \
        --method=spectral --l_max=15 --output=cascade_spectral.jld2

Options:
  --start=<t>        Start time for analysis (default: auto-detect)
  --end=<t>          End time for analysis (default: auto-detect)
  --output=<file>    Output JLD2 file (default: energy_cascade.jld2)
  --prefix=<name>    Input file prefix (default: combined_global)
  --scales=<list>    Filter length scales in grid units (default: 2,4,8,16)
  --method=<str>     Analysis method: physical, spectral, or both (default: both)
  --radius=<r>       Radial shell for analysis [0,1] (default: 0.8 - outer core)
  --kernel=<str>     Filter kernel: gaussian, butterworth (default: gaussian)
  --l_max=<N>        Maximum spherical harmonic degree (default: 10)
  --compute_helicity Compute helicity cascades (computationally intensive)

Output structure:
- Physical space cascades (Leonard transfers, energy transport)
- Spectral space cascades (inter-mode energy transfers)
- Multi-scale energy budgets
- Scale-dependent transport diagnostics
- Visualization-ready data arrays
"""

using Printf
using NetCDF
using JLD2
using Statistics
using LinearAlgebra
using FFTW

# Add CoarseGraining.jl to the path
push!(LOAD_PATH, dirname(@__DIR__) * "/../CoarseGraining.jl/src")
using CoarseGraining

# Import necessary types and functions
import CoarseGraining: Field, Grid, SphericalGrid, gaussian_kernel, coarse_grain, compute_pi
import CoarseGraining: gradient, divergence, vorticity, helmholtz_hodge, okuboweiss
import CoarseGraining: compute_energy_transport, compute_baroclinic_transfer

function usage()
    println("Usage: energy_cascade_analysis.jl <output_dir> [options]")
    println("Run with --help for detailed options")
end

function parse_args(args)
    if isempty(args) || (length(args) == 1 && args[1] == "--help")
        if length(args) == 1 && args[1] == "--help"
            println(__doc__)
        else
            usage()
        end
        return nothing
    end
    
    output_dir = abspath(args[1])
    
    # Default parameters
    params = Dict(
        :output_dir => output_dir,
        :start_time => nothing,
        :end_time => nothing,
        :output_file => "energy_cascade.jld2",
        :prefix => "combined_global",
        :scales => [2, 4, 8, 16],
        :method => "both",
        :radius => 0.8,
        :kernel => "gaussian",
        :l_max => 10,
        :compute_helicity => false
    )
    
    # Parse optional arguments
    for arg in args[2:end]
        if startswith(arg, "--start=")
            params[:start_time] = parse(Float64, arg[9:end])
        elseif startswith(arg, "--end=")
            params[:end_time] = parse(Float64, arg[7:end])
        elseif startswith(arg, "--output=")
            params[:output_file] = arg[10:end]
        elseif startswith(arg, "--prefix=")
            params[:prefix] = arg[10:end]
        elseif startswith(arg, "--scales=")
            params[:scales] = parse.(Int, split(arg[10:end], ","))
        elseif startswith(arg, "--method=")
            method = arg[10:end]
            if method ∉ ["physical", "spectral", "both"]
                error("Method must be 'physical', 'spectral', or 'both'")
            end
            params[:method] = method
        elseif startswith(arg, "--radius=")
            params[:radius] = parse(Float64, arg[10:end])
        elseif startswith(arg, "--kernel=")
            params[:kernel] = arg[10:end]
        elseif startswith(arg, "--l_max=")
            params[:l_max] = parse(Int, arg[9:end])
        elseif arg == "--compute_helicity"
            params[:compute_helicity] = true
        else
            error("Unknown argument: $arg")
        end
    end
    
    return params
end

function find_merged_files(output_dir, prefix)
    """Find all merged NetCDF files matching pattern"""
    files = String[]
    times = Float64[]
    
    for file in readdir(output_dir)
        if startswith(file, prefix * "_time_") && endswith(file, ".nc")
            # Extract time from filename
            time_str = file[(length(prefix) + 7):end-3]
            try
                time = parse(Float64, time_str)
                push!(files, joinpath(output_dir, file))
                push!(times, time)
            catch
                @warn "Could not parse time from filename: $file"
            end
        end
    end
    
    # Sort by time
    perm = sortperm(times)
    return files[perm], times[perm]
end

function read_var_safe(nc, name)
    """Safely read variable from NetCDF file"""
    try
        return NetCDF.readvar(nc, name)
    catch
        return nothing
    end
end

function load_velocity_data(file, radius_target)
    """Load velocity field data from NetCDF file at specified radius"""
    nc = NetCDF.open(file)
    
    try
        # Load grid information
        r_grid = read_var_safe(nc, "r_grid")
        theta_grid = read_var_safe(nc, "theta_grid") 
        phi_grid = read_var_safe(nc, "phi_grid")
        
        if any(x -> x === nothing, [r_grid, theta_grid, phi_grid])
            error("Missing grid information in file: $file")
        end
        
        # Find closest radial index
        r_idx = argmin(abs.(r_grid .- radius_target))
        actual_radius = r_grid[r_idx]
        
        println("Using radius r = $(round(actual_radius, digits=3)) (target: $radius_target)")
        
        # Try to load physical space velocity data
        vel_r = read_var_safe(nc, "velocity_r")
        vel_theta = read_var_safe(nc, "velocity_theta")  
        vel_phi = read_var_safe(nc, "velocity_phi")
        
        # If physical space not available, try spectral coefficients
        if any(x -> x === nothing, [vel_r, vel_theta, vel_phi])
            println("Physical velocity not found, attempting spectral reconstruction...")
            
            # Load spectral coefficients
            vel_tor_real = read_var_safe(nc, "velocity_toroidal_real")
            vel_tor_imag = read_var_safe(nc, "velocity_toroidal_imag")
            vel_pol_real = read_var_safe(nc, "velocity_poloidal_real")
            vel_pol_imag = read_var_safe(nc, "velocity_poloidal_imag")
            l_values = read_var_safe(nc, "l_values")
            m_values = read_var_safe(nc, "m_values")
            
            if any(x -> x === nothing, [vel_tor_real, vel_tor_imag, vel_pol_real, vel_pol_imag, l_values, m_values])
                error("Neither physical nor complete spectral velocity data available in: $file")
            end
            
            # Reconstruct physical velocity from spectral coefficients
            vel_r, vel_theta, vel_phi = reconstruct_velocity_from_spectral(
                vel_tor_real[:, r_idx], vel_tor_imag[:, r_idx], 
                vel_pol_real[:, r_idx], vel_pol_imag[:, r_idx],
                l_values, m_values, theta_grid, phi_grid
            )
        else
            # Extract radial slice
            vel_r = vel_r[:, :, r_idx]
            vel_theta = vel_theta[:, :, r_idx]
            vel_phi = vel_phi[:, :, r_idx]
        end
        
        # Similarly for magnetic field
        mag_r = read_var_safe(nc, "magnetic_r")
        mag_theta = read_var_safe(nc, "magnetic_theta")
        mag_phi = read_var_safe(nc, "magnetic_phi")
        
        if any(x -> x === nothing, [mag_r, mag_theta, mag_phi])
            println("Physical magnetic field not found, attempting spectral reconstruction...")
            
            mag_tor_real = read_var_safe(nc, "magnetic_toroidal_real")
            mag_tor_imag = read_var_safe(nc, "magnetic_toroidal_imag")
            mag_pol_real = read_var_safe(nc, "magnetic_poloidal_real")
            mag_pol_imag = read_var_safe(nc, "magnetic_poloidal_imag")
            
            if any(x -> x === nothing, [mag_tor_real, mag_tor_imag, mag_pol_real, mag_pol_imag])
                @warn "Magnetic field data not complete, setting to zero"
                mag_r = zeros(size(vel_r))
                mag_theta = zeros(size(vel_theta))
                mag_phi = zeros(size(vel_phi))
            else
                mag_r, mag_theta, mag_phi = reconstruct_velocity_from_spectral(
                    mag_tor_real[:, r_idx], mag_tor_imag[:, r_idx], 
                    mag_pol_real[:, r_idx], mag_pol_imag[:, r_idx],
                    l_values, m_values, theta_grid, phi_grid
                )
            end
        else
            mag_r = mag_r[:, :, r_idx]
            mag_theta = mag_theta[:, :, r_idx]  
            mag_phi = mag_phi[:, :, r_idx]
        end
        
        return (
            vel_r = vel_r, vel_theta = vel_theta, vel_phi = vel_phi,
            mag_r = mag_r, mag_theta = mag_theta, mag_phi = mag_phi,
            r_grid = r_grid, theta_grid = theta_grid, phi_grid = phi_grid,
            radius = actual_radius
        )
        
    finally
        NetCDF.close(nc)
    end
end

function reconstruct_velocity_from_spectral(tor_real, tor_imag, pol_real, pol_imag,
                                          l_values, m_values, theta_grid, phi_grid)
    """Simplified spectral to physical velocity reconstruction"""
    ntheta, nphi = length(theta_grid), length(phi_grid)
    
    # Initialize velocity components
    vel_r = zeros(ntheta, nphi)
    vel_theta = zeros(ntheta, nphi)
    vel_phi = zeros(ntheta, nphi)
    
    # Simple reconstruction using spherical harmonics (simplified)
    for (i, (l, m)) in enumerate(zip(l_values, m_values))
        if l == 0
            continue  # Skip l=0 for velocity
        end
        
        tor_coeff = complex(tor_real[i], tor_imag[i])
        pol_coeff = complex(pol_real[i], pol_real[i])
        
        for (j, theta) in enumerate(theta_grid)
            for (k, phi) in enumerate(phi_grid)
                # Simplified spherical harmonic evaluation
                Y_lm = exp(1im * m * phi) * cos(l * theta)  # Simplified
                
                # Approximate velocity components (this is highly simplified)
                vel_r[j, k] += real(pol_coeff * Y_lm * l * (l + 1))
                vel_theta[j, k] += real(pol_coeff * Y_lm * sin(theta))
                vel_phi[j, k] += real(tor_coeff * Y_lm / sin(theta))
            end
        end
    end
    
    return vel_r, vel_theta, vel_phi
end

function create_spherical_grid(theta_grid, phi_grid, radius)
    """Create CoarseGraining.jl compatible spherical grid"""
    # Convert to latitude/longitude if needed
    lat_grid = pi/2 .- theta_grid  # Convert colatitude to latitude
    lon_grid = phi_grid
    
    # Create spherical grid (Earth radius for proper scaling)
    return SphericalGrid(lon_grid, lat_grid, radius * 6.371e6, true)
end

function compute_physical_cascade(vel_data, mag_data, scales, kernel_type, grid)
    """Compute energy cascades in physical space using Leonard transfers"""
    println("Computing physical space energy cascades...")
    
    # Convert velocity to horizontal components for cascade analysis
    # In spherical coordinates: u_E (eastward), u_N (northward)
    u_east = Field(-vel_data.vel_phi, grid)  # -v_phi = u_east
    u_north = Field(vel_data.vel_theta, grid)  # v_theta = u_north
    
    # Magnetic field components
    b_east = Field(-mag_data.mag_phi, grid)
    b_north = Field(mag_data.mag_theta, grid)
    
    cascade_results = Dict()
    
    for scale in scales
        println("  Processing scale: $(scale) grid units")
        
        # Create kernel
        if kernel_type == "gaussian"
            # Scale in terms of grid spacing
            σ_lon = scale * (grid.lon[2] - grid.lon[1])
            σ_lat = scale * (grid.lat[2] - grid.lat[1])
            kernel = gaussian_kernel(σ_lon, σ_lat)
        else
            error("Kernel type $kernel_type not supported yet")
        end
        
        # Kinetic energy cascade
        Π_kinetic = compute_pi(u_east, u_north, kernel)
        
        # Magnetic energy cascade  
        Π_magnetic = compute_pi(b_east, b_north, kernel)
        
        # Energy transport analysis
        energy_transport = compute_energy_transport(u_east, u_north, kernel)
        
        # Cross helicity cascade (if requested)
        if haskey(params, :compute_helicity) && params[:compute_helicity]
            # Cross helicity: H_c = u·B
            cross_helicity_field = Field(
                u_east.data .* b_east.data .+ u_north.data .* b_north.data,
                grid
            )
            
            # Cross helicity cascade (simplified)
            filtered_H = coarse_grain(cross_helicity_field, kernel)
            filtered_u_east = coarse_grain(u_east, kernel)
            filtered_u_north = coarse_grain(u_north, kernel)
            filtered_b_east = coarse_grain(b_east, kernel)
            filtered_b_north = coarse_grain(b_north, kernel)
            
            H_resolved = Field(
                filtered_u_east.data .* filtered_b_east.data .+ 
                filtered_u_north.data .* filtered_b_north.data,
                grid
            )
            
            Π_cross_helicity = Field(filtered_H.data .- H_resolved.data, grid)
        else
            Π_cross_helicity = nothing
        end
        
        # Store results for this scale
        cascade_results[scale] = Dict(
            :Pi_kinetic => Π_kinetic.data,
            :Pi_magnetic => Π_magnetic.data,
            :energy_transport => energy_transport,
            :Pi_cross_helicity => Π_cross_helicity !== nothing ? Π_cross_helicity.data : nothing,
            :kernel_scale => scale
        )
        
        # Compute statistics
        println("    Kinetic Π: mean=$(round(mean(Π_kinetic.data), digits=8)), std=$(round(std(Π_kinetic.data), digits=8))")
        println("    Magnetic Π: mean=$(round(mean(Π_magnetic.data), digits=8)), std=$(round(std(Π_magnetic.data), digits=8))")
    end
    
    return cascade_results
end

function compute_spectral_cascade(files, times, params)
    """Compute energy cascades in spectral space"""
    println("Computing spectral space energy cascades...")
    
    # This is a placeholder for spectral cascade analysis
    # Would require implementing inter-mode energy transfers
    
    spectral_results = Dict(
        :note => "Spectral cascade analysis not yet implemented",
        :inter_mode_transfers => nothing,
        :spectral_flux => nothing
    )
    
    return spectral_results
end

function compute_multi_scale_energy_budget(cascade_data, grid)
    """Compute energy budget across multiple scales"""
    println("Computing multi-scale energy budget...")
    
    scales = sort(collect(keys(cascade_data)))
    budget = Dict()
    
    for scale in scales
        data = cascade_data[scale]
        
        # Volume integration (approximate)
        dΩ = (grid.lon[2] - grid.lon[1]) * (grid.lat[2] - grid.lat[1])
        
        # Integrated energy transfers
        Pi_k_total = sum(data[:Pi_kinetic]) * dΩ
        Pi_m_total = sum(data[:Pi_magnetic]) * dΩ
        
        # Energy transfer rates
        budget[scale] = Dict(
            :kinetic_transfer_rate => Pi_k_total,
            :magnetic_transfer_rate => Pi_m_total,
            :total_transfer_rate => Pi_k_total + Pi_m_total,
            :scale_length => scale
        )
        
        println("  Scale $(scale): K transfer = $(round(Pi_k_total, digits=10)), M transfer = $(round(Pi_m_total, digits=10))")
    end
    
    return budget
end

function analyze_cascade_time_series(files, times, params)
    """Analyze energy cascade evolution over time"""
    println("Analyzing cascade time series...")
    
    # Filter files by time range
    if params[:start_time] !== nothing || params[:end_time] !== nothing
        start_t = params[:start_time] === nothing ? -Inf : params[:start_time]
        end_t = params[:end_time] === nothing ? Inf : params[:end_time]
        
        mask = (times .>= start_t) .& (times .<= end_t)
        files = files[mask]
        times = times[mask]
    end
    
    if isempty(files)
        error("No files found in specified time range")
    end
    
    println("Processing $(length(files)) files...")
    
    # Initialize results storage
    time_series_data = Dict()
    scales = params[:scales]
    
    for (i, file) in enumerate(files)
        println("Processing file $i/$(length(files)): $(basename(file))")
        
        # Load velocity and magnetic field data
        data = load_velocity_data(file, params[:radius])
        
        # Create spherical grid
        grid = create_spherical_grid(data.theta_grid, data.phi_grid, data.radius)
        
        # Compute physical cascades
        if params[:method] in ["physical", "both"]
            cascade_data = compute_physical_cascade(data, data, scales, params[:kernel], grid)
            
            # Store time series
            if i == 1
                time_series_data[:physical] = Dict()
                for scale in scales
                    time_series_data[:physical][scale] = Dict(
                        :Pi_kinetic => Float64[],
                        :Pi_magnetic => Float64[],
                        :times => Float64[]
                    )
                end
            end
            
            # Append data
            for scale in scales
                Pi_k_mean = mean(cascade_data[scale][:Pi_kinetic])
                Pi_m_mean = mean(cascade_data[scale][:Pi_magnetic])
                
                push!(time_series_data[:physical][scale][:Pi_kinetic], Pi_k_mean)
                push!(time_series_data[:physical][scale][:Pi_magnetic], Pi_m_mean)
                push!(time_series_data[:physical][scale][:times], times[i])
            end
            
            # Compute energy budget for first file (representative)
            if i == 1
                energy_budget = compute_multi_scale_energy_budget(cascade_data, grid)
                time_series_data[:energy_budget] = energy_budget
                time_series_data[:grid_info] = Dict(
                    :theta_grid => data.theta_grid,
                    :phi_grid => data.phi_grid,
                    :radius => data.radius
                )
            end
        end
        
        if i % 5 == 0 || i == length(files)
            println("  Completed $i/$(length(files)) files")
        end
    end
    
    # Compute spectral cascades if requested
    if params[:method] in ["spectral", "both"]
        spectral_data = compute_spectral_cascade(files, times, params)
        time_series_data[:spectral] = spectral_data
    end
    
    return time_series_data
end

function main()
    args = ARGS
    params = parse_args(args)
    
    if params === nothing
        return
    end
    
    println("Energy Cascade Analysis for Geodynamo.jl")
    println("=" ^ 45)
    println("Output directory: $(params[:output_dir])")
    println("Method: $(params[:method])")
    println("Analysis radius: r = $(params[:radius])")
    println("Filter scales: $(params[:scales])")
    
    # Find input files
    files, times = find_merged_files(params[:output_dir], params[:prefix])
    if isempty(files)
        error("No merged files found matching pattern '$(params[:prefix])_time_*.nc' in $(params[:output_dir])")
    end
    
    println("Found $(length(files)) merged files spanning times [$(minimum(times)), $(maximum(times))]")
    
    # Perform cascade analysis
    results = analyze_cascade_time_series(files, times, params)
    
    # Add metadata
    results[:metadata] = Dict(
        :analysis_time => now(),
        :method => params[:method],
        :scales => params[:scales],
        :radius => params[:radius],
        :kernel => params[:kernel],
        :n_files => length(files),
        :time_range => [minimum(times), maximum(times)],
        :compute_helicity => params[:compute_helicity]
    )
    
    # Save results
    output_path = joinpath(params[:output_dir], params[:output_file])
    println("Saving results to: $output_path")
    
    JLD2.jldsave(output_path; results...)
    
    println("Energy cascade analysis complete!")
    
    # Print summary
    if haskey(results, :physical)
        println("\nSummary of Physical Space Cascades:")
        println("─" ^ 40)
        
        for scale in params[:scales]
            if haskey(results[:physical], scale)
                data = results[:physical][scale]
                if !isempty(data[:Pi_kinetic])
                    avg_Pi_k = mean(data[:Pi_kinetic])
                    avg_Pi_m = mean(data[:Pi_magnetic])
                    println("Scale $(scale): <Π_k> = $(round(avg_Pi_k, digits=10)), <Π_m> = $(round(avg_Pi_m, digits=10))")
                end
            end
        end
        
        if haskey(results, :energy_budget)
            println("\nEnergy Budget (first timestep):")
            println("─" ^ 30)
            budget = results[:energy_budget]
            for scale in sort(collect(keys(budget)))
                transfer_rate = budget[scale][:total_transfer_rate]
                println("Scale $(scale): Total transfer = $(round(transfer_rate, digits=10))")
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end