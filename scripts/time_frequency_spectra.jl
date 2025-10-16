#!/usr/bin/env julia

"""
Compute time-frequency spectra of kinetic and magnetic energy from merged GeoDynamo.jl outputs.

This script performs:
1. Time series analysis of energy components with optional tangent cylinder filtering
2. Short-Time Fourier Transform (STFT) for time-frequency analysis
3. Wavelet transform analysis for multi-scale temporal features
4. Power spectral density computation
5. Spectral energy in different l,m modes over time

Tangent Cylinder Options:
- full: Analyze entire domain (default)
- remove_tc: Exclude tangent cylinder region from analysis
- tc_only: Analyze only the tangent cylinder region

Usage examples:

  # Basic time-frequency analysis with default parameters
  julia --project=. scripts/time_frequency_spectra.jl ./output

  # Custom time range and output file
  julia --project=. scripts/time_frequency_spectra.jl ./output \
        --start=0.5 --end=5.0 --output=tf_spectra.jld2

  # Advanced analysis with custom windowing
  julia --project=. scripts/time_frequency_spectra.jl ./output \
        --start=1.0 --end=10.0 --window=512 --overlap=0.5 \
        --method=both --l_max=20

  # Analyze with tangent cylinder filtering
  julia --project=. scripts/time_frequency_spectra.jl ./output \
        --tc_mode=remove_tc --tc_radius=0.35 --method=both

  # Compare tangent cylinder region only
  julia --project=. scripts/time_frequency_spectra.jl ./output \
        --tc_mode=tc_only --tc_radius=0.4 --output=tf_tc_only.jld2

Options:
  --start=<t>        Start time for analysis (default: auto-detect)
  --end=<t>          End time for analysis (default: auto-detect) 
  --output=<file>    Output JLD2 file (default: time_frequency_spectra.jld2)
  --prefix=<name>    Input file prefix (default: combined_global)
  --window=<N>       STFT window size in time points (default: 256)
  --overlap=<f>      Window overlap fraction 0-1 (default: 0.75)
  --method=<str>     Analysis method: stft, wavelet, or both (default: both)
  --l_max=<N>        Maximum l degree to analyze (default: all available)
  --dt_target=<f>    Target time spacing for resampling (default: auto)
  --tc_mode=<mode>   Tangent cylinder mode: full, remove_tc, tc_only (default: full)
  --tc_radius=<r>    Tangent cylinder radius (default: 0.35)

Output structure:
- time_series: Raw energy time series data
- frequencies: Frequency arrays for STFT/PSD
- stft_data: Short-Time Fourier Transform results
- wavelet_data: Continuous wavelet transform results  
- psd_data: Power spectral density
- mode_evolution: Energy evolution in different l,m modes
- metadata: Analysis parameters and data info
"""

using Printf
using NetCDF
using JLD2
using Statistics
using LinearAlgebra
using FFTW
using DSP
using Dates

# Optional packages for advanced analysis
try
    using ContinuousWavelets
    global HAS_WAVELETS = true
catch
    global HAS_WAVELETS = false
    @warn "ContinuousWavelets.jl not available. Wavelet analysis disabled."
end

function usage()
    println("Usage: time_frequency_spectra.jl <output_dir> [options]")
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
        :output_file => "time_frequency_spectra.jld2",
        :prefix => "combined_global",
        :window_size => 256,
        :overlap => 0.75,
        :method => "both",
        :l_max => nothing,
        :dt_target => nothing,
        :tc_mode => "full",
        :tc_radius => 0.35
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
        elseif startswith(arg, "--window=")
            params[:window_size] = parse(Int, arg[10:end])
        elseif startswith(arg, "--overlap=")
            params[:overlap] = parse(Float64, arg[11:end])
        elseif startswith(arg, "--method=")
            method = arg[10:end]
            if method ∉ ["stft", "wavelet", "both"]
                error("Method must be 'stft', 'wavelet', or 'both'")
            end
            params[:method] = method
        elseif startswith(arg, "--l_max=")
            params[:l_max] = parse(Int, arg[9:end])
        elseif startswith(arg, "--dt_target=")
            params[:dt_target] = parse(Float64, arg[13:end])
        elseif startswith(arg, "--tc_mode=")
            tc_mode = arg[11:end]
            if tc_mode ∉ ["full", "remove_tc", "tc_only"]
                error("TC mode must be 'full', 'remove_tc', or 'tc_only'")
            end
            params[:tc_mode] = tc_mode
        elseif startswith(arg, "--tc_radius=")
            params[:tc_radius] = parse(Float64, arg[13:end])
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
            time_str = file[(length(prefix) + 7):end-3]  # Remove prefix_time_ and .nc
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

function create_tangent_cylinder_mask_spectral(l_values, m_values, r_grid, tc_radius)
    """Create tangent cylinder mask for spectral data based on radial location"""
    # For spectral data, we approximate the tangent cylinder effect
    # by considering the radial coordinate where the cylindrical radius s = r*sin(θ_eq)
    # reaches the tangent cylinder radius at the equator (θ = π/2)
    
    nlm = length(l_values)
    nr = length(r_grid)
    
    # Create mask: true = include, false = exclude
    tc_mask = ones(Bool, nlm, nr)
    
    # For spectral coefficients, the tangent cylinder primarily affects
    # low-l modes more strongly in the inner radial regions
    for lm in 1:nlm
        l = l_values[lm]
        for ir in 1:nr
            r = r_grid[ir]
            
            # Simple approximation: exclude inner regions for low-l modes
            # when r < tc_radius / sin(π/2) = tc_radius
            # For higher l modes, the effect is less pronounced
            
            if r < tc_radius && l <= 4
                # Strong tangent cylinder effect for low-l modes
                tc_mask[lm, ir] = false
            elseif r < tc_radius * 1.2 && l <= 2
                # Extended effect for l=1,2 modes
                tc_mask[lm, ir] = false
            end
        end
    end
    
    return tc_mask
end

function apply_tangent_cylinder_filter_spectral(spectral_data, tc_mask, tc_mode)
    """Apply tangent cylinder filtering to spectral coefficient data"""
    filtered_data = copy(spectral_data)
    
    if tc_mode == "full"
        # No filtering - return full domain
        return filtered_data
        
    elseif tc_mode == "remove_tc"
        # Set tangent cylinder affected modes to zero
        filtered_data[.!tc_mask] .= 0.0
        
    elseif tc_mode == "tc_only"
        # Keep only tangent cylinder affected modes
        filtered_data[tc_mask] .= 0.0
        
    else
        error("Unknown tangent cylinder mode: $tc_mode")
    end
    
    return filtered_data
end

function compute_energy_from_spectral(vel_tor_real, vel_tor_imag, vel_pol_real, vel_pol_imag, 
                                     mag_tor_real, mag_tor_imag, mag_pol_real, mag_pol_imag,
                                     l_values, m_values, r_grid, tc_mask=nothing, tc_mode="full")
    """Compute kinetic and magnetic energy from spectral coefficients with optional TC filtering"""
    nlm = length(l_values)
    nr = length(r_grid)
    
    # Apply tangent cylinder filtering if requested
    if tc_mask !== nothing && tc_mode != "full"
        vel_tor_real = apply_tangent_cylinder_filter_spectral(vel_tor_real, tc_mask, tc_mode)
        vel_tor_imag = apply_tangent_cylinder_filter_spectral(vel_tor_imag, tc_mask, tc_mode)
        vel_pol_real = apply_tangent_cylinder_filter_spectral(vel_pol_real, tc_mask, tc_mode)
        vel_pol_imag = apply_tangent_cylinder_filter_spectral(vel_pol_imag, tc_mask, tc_mode)
        mag_tor_real = apply_tangent_cylinder_filter_spectral(mag_tor_real, tc_mask, tc_mode)
        mag_tor_imag = apply_tangent_cylinder_filter_spectral(mag_tor_imag, tc_mask, tc_mode)
        mag_pol_real = apply_tangent_cylinder_filter_spectral(mag_pol_real, tc_mask, tc_mode)
        mag_pol_imag = apply_tangent_cylinder_filter_spectral(mag_pol_imag, tc_mask, tc_mode)
    end
    
    # Initialize energy arrays
    Ek_total = 0.0
    Eb_total = 0.0
    Ek_lm = zeros(nlm)
    Eb_lm = zeros(nlm)
    
    # Energy computation - integrated over radius
    for lm in 1:nlm
        l = l_values[lm]
        
        # Skip l=0 modes for velocity (no flow)
        if l > 0
            for ir in 1:nr
                r = r_grid[ir]
                dr = ir < nr ? (r_grid[ir+1] - r) : (r - r_grid[ir-1])
                
                # Kinetic energy: ∫ ½(|u_tor|² + |u_pol|²) r² dr
                u_tor_sq = vel_tor_real[lm, ir]^2 + vel_tor_imag[lm, ir]^2
                u_pol_sq = vel_pol_real[lm, ir]^2 + vel_pol_imag[lm, ir]^2
                
                dEk = 0.5 * (u_tor_sq + u_pol_sq) * r^2 * dr
                Ek_lm[lm] += dEk
                Ek_total += dEk
                
                # Magnetic energy: ∫ ½(|B_tor|² + |B_pol|²) r² dr  
                B_tor_sq = mag_tor_real[lm, ir]^2 + mag_tor_imag[lm, ir]^2
                B_pol_sq = mag_pol_real[lm, ir]^2 + mag_pol_imag[lm, ir]^2
                
                dEb = 0.5 * (B_tor_sq + B_pol_sq) * r^2 * dr
                Eb_lm[lm] += dEb  
                Eb_total += dEb
            end
        end
    end
    
    return Ek_total, Eb_total, Ek_lm, Eb_lm
end

function load_time_series(files, times, params)
    """Load energy time series from merged files"""
    println("Loading time series data from $(length(files)) files...")
    
    # Filter files by time range if specified
    if params[:start_time] !== nothing || params[:end_time] !== nothing
        start_t = params[:start_time] === nothing ? -Inf : params[:start_time]
        end_t = params[:end_time] === nothing ? Inf : params[:end_time]
        
        mask = (times .>= start_t) .& (times .<= end_t)
        files = files[mask]
        times = times[mask]
        
        println("Filtered to $(length(files)) files in time range [$start_t, $end_t]")
    end
    
    if isempty(files)
        error("No files found in specified time range")
    end
    
    # Load first file to get dimensions and structure
    nc = NetCDF.open(files[1])
    try
        l_values = Int.(read_var_safe(nc, "l_values"))
        m_values = Int.(read_var_safe(nc, "m_values"))
        r_grid = read_var_safe(nc, "r_grid")
        
        if any(x -> x === nothing, [l_values, m_values, r_grid])
            error("Missing required variables in file: $(files[1])")
        end
        
        # Apply l_max filter if specified
        if params[:l_max] !== nothing
            mask = l_values .<= params[:l_max]
            l_values = l_values[mask]
            m_values = m_values[mask]
        end
        
        nlm = length(l_values)
        nt = length(times)
        
        println("Data dimensions: nlm=$nlm, nt=$nt, nr=$(length(r_grid))")
        
        # Create tangent cylinder mask for spectral filtering
        tc_mask = nothing
        if params[:tc_mode] != "full"
            tc_mask = create_tangent_cylinder_mask_spectral(l_values, m_values, r_grid, params[:tc_radius])
            affected_points = sum(.!tc_mask)
            total_points = length(tc_mask)
            println("Tangent cylinder filtering ($(params[:tc_mode])):")
            println("  TC radius: $(params[:tc_radius])")
            println("  Affected spectral points: $affected_points/$(total_points) ($(round(100*affected_points/total_points, digits=1))%)")
        end
        
        # Initialize arrays
        Ek_time = zeros(nt)
        Eb_time = zeros(nt)
        Ek_lm_time = zeros(nlm, nt)
        Eb_lm_time = zeros(nlm, nt)
        
    finally
        NetCDF.close(nc)
    end
    
    # Load all time points
    for (i, file) in enumerate(files)
        nc = NetCDF.open(file)
        try
            # Load spectral coefficients
            vel_tor_real = read_var_safe(nc, "velocity_toroidal_real")
            vel_tor_imag = read_var_safe(nc, "velocity_toroidal_imag")
            vel_pol_real = read_var_safe(nc, "velocity_poloidal_real")
            vel_pol_imag = read_var_safe(nc, "velocity_poloidal_imag")
            
            mag_tor_real = read_var_safe(nc, "magnetic_toroidal_real")
            mag_tor_imag = read_var_safe(nc, "magnetic_toroidal_imag")
            mag_pol_real = read_var_safe(nc, "magnetic_poloidal_real")
            mag_pol_imag = read_var_safe(nc, "magnetic_poloidal_imag")
            
            if any(x -> x === nothing, [vel_tor_real, vel_tor_imag, vel_pol_real, vel_pol_imag,
                                       mag_tor_real, mag_tor_imag, mag_pol_real, mag_pol_imag])
                @warn "Missing spectral data in file: $file. Skipping..."
                continue
            end
            
            # Apply l_max filter to data if specified
            if params[:l_max] !== nothing
                mask = l_values .<= params[:l_max]
                vel_tor_real = vel_tor_real[mask, :]
                vel_tor_imag = vel_tor_imag[mask, :]
                vel_pol_real = vel_pol_real[mask, :]
                vel_pol_imag = vel_pol_imag[mask, :]
                mag_tor_real = mag_tor_real[mask, :]
                mag_tor_imag = mag_tor_imag[mask, :]
                mag_pol_real = mag_pol_real[mask, :]
                mag_pol_imag = mag_pol_imag[mask, :]
            end
            
            # Compute energies with tangent cylinder filtering
            Ek, Eb, Ek_lm, Eb_lm = compute_energy_from_spectral(
                vel_tor_real, vel_tor_imag, vel_pol_real, vel_pol_imag,
                mag_tor_real, mag_tor_imag, mag_pol_real, mag_pol_imag,
                l_values, m_values, r_grid, tc_mask, params[:tc_mode]
            )
            
            Ek_time[i] = Ek
            Eb_time[i] = Eb
            Ek_lm_time[:, i] = Ek_lm
            Eb_lm_time[:, i] = Eb_lm
            
            if i % 10 == 0 || i == length(files)
                println("Processed $i/$(length(files)) files...")
            end
            
        finally
            NetCDF.close(nc)
        end
    end
    
    return (
        times = times,
        Ek = Ek_time,
        Eb = Eb_time,
        Ek_lm = Ek_lm_time,
        Eb_lm = Eb_lm_time,
        l_values = l_values,
        m_values = m_values,
        r_grid = r_grid
    )
end

function resample_time_series(data, target_dt)
    """Resample time series to uniform spacing if needed"""
    times = data.times
    dt_current = median(diff(times))
    
    if target_dt === nothing
        target_dt = dt_current
        println("Using detected time spacing: Δt = $(round(dt_current, digits=6))")
    end
    
    if abs(target_dt - dt_current) / dt_current > 0.01  # Resample if >1% difference
        println("Resampling from Δt=$(round(dt_current, digits=6)) to Δt=$(round(target_dt, digits=6))")
        
        # Create uniform time grid
        t_start, t_end = extrema(times)
        new_times = t_start:target_dt:t_end
        
        # Interpolate all energy data
        Ek_new = linear_interp(times, data.Ek, new_times)
        Eb_new = linear_interp(times, data.Eb, new_times)
        
        Ek_lm_new = zeros(size(data.Ek_lm, 1), length(new_times))
        Eb_lm_new = zeros(size(data.Eb_lm, 1), length(new_times))
        
        for lm in 1:size(data.Ek_lm, 1)
            Ek_lm_new[lm, :] = linear_interp(times, data.Ek_lm[lm, :], new_times)
            Eb_lm_new[lm, :] = linear_interp(times, data.Eb_lm[lm, :], new_times)
        end
        
        return (
            times = collect(new_times),
            Ek = Ek_new,
            Eb = Eb_new, 
            Ek_lm = Ek_lm_new,
            Eb_lm = Eb_lm_new,
            l_values = data.l_values,
            m_values = data.m_values,
            r_grid = data.r_grid
        )
    end
    
    return data
end

function linear_interp(x, y, xi)
    """Simple linear interpolation"""
    yi = similar(xi, eltype(y))
    for (i, xi_val) in enumerate(xi)
        if xi_val <= x[1]
            yi[i] = y[1]
        elseif xi_val >= x[end]
            yi[i] = y[end]
        else
            # Find surrounding points
            j = searchsortedfirst(x, xi_val)
            if j == 1
                yi[i] = y[1]
            else
                # Linear interpolation
                x1, x2 = x[j-1], x[j]
                y1, y2 = y[j-1], y[j]
                yi[i] = y1 + (y2 - y1) * (xi_val - x1) / (x2 - x1)
            end
        end
    end
    return yi
end

function compute_stft(data, params)
    """Compute Short-Time Fourier Transform for time-frequency analysis"""
    println("Computing Short-Time Fourier Transform...")
    
    window_size = params[:window_size]
    overlap = params[:overlap]
    dt = median(diff(data.times))
    
    # Compute STFT for total energies
    stft_Ek = DSP.stft(data.Ek, window_size, Int(floor(window_size * (1 - overlap))))
    stft_Eb = DSP.stft(data.Eb, window_size, Int(floor(window_size * (1 - overlap))))
    
    # Time and frequency axes
    n_overlap = Int(floor(window_size * overlap))
    hop_size = window_size - n_overlap
    times_stft = data.times[1] .+ (0:size(stft_Ek, 2)-1) * hop_size * dt
    freqs_stft = fftfreq(window_size, 1/dt)[1:window_size÷2+1]
    
    # Compute power spectral densities
    psd_Ek = abs2.(stft_Ek)
    psd_Eb = abs2.(stft_Eb)
    
    # Mode-specific analysis for selected l values
    l_selected = unique(data.l_values)[1:min(5, length(unique(data.l_values)))]  # First 5 l values
    
    stft_modes = Dict()
    psd_modes = Dict()
    
    for l in l_selected
        l_mask = data.l_values .== l
        
        if any(l_mask)
            # Sum energy over all m for this l
            Ek_l = vec(sum(data.Ek_lm[l_mask, :], dims=1))
            Eb_l = vec(sum(data.Eb_lm[l_mask, :], dims=1))
            
            stft_Ek_l = DSP.stft(Ek_l, window_size, hop_size)
            stft_Eb_l = DSP.stft(Eb_l, window_size, hop_size)
            
            stft_modes["Ek_l$l"] = stft_Ek_l
            stft_modes["Eb_l$l"] = stft_Eb_l
            psd_modes["Ek_l$l"] = abs2.(stft_Ek_l)
            psd_modes["Eb_l$l"] = abs2.(stft_Eb_l)
        end
    end
    
    return (
        times = times_stft,
        frequencies = freqs_stft,
        stft_Ek = stft_Ek,
        stft_Eb = stft_Eb,
        psd_Ek = psd_Ek,
        psd_Eb = psd_Eb,
        stft_modes = stft_modes,
        psd_modes = psd_modes,
        l_selected = l_selected,
        window_size = window_size,
        overlap = overlap
    )
end

function compute_wavelet_transform(data, params)
    """Compute continuous wavelet transform if package available"""
    if !HAS_WAVELETS
        @warn "Wavelet analysis skipped - ContinuousWavelets.jl not available"
        return nothing
    end
    
    println("Computing continuous wavelet transform...")
    
    dt = median(diff(data.times))
    
    # Wavelet parameters
    wavelet = ContinuousWavelets.Morlet(π)  # Morlet wavelet
    n_voices = 16  # Voices per octave
    
    # Frequency range for wavelets
    f_min = 1.0 / (data.times[end] - data.times[1])  # Fundamental frequency
    f_max = 1.0 / (4 * dt)  # Nyquist/4 for safety
    
    scales = ContinuousWavelets.computeWavelets(length(data.Ek), wavelet)[2]
    freqs = 1 ./ (scales * dt)
    
    # Compute CWT for total energies
    cwt_Ek = ContinuousWavelets.cwt(data.Ek, wavelet)
    cwt_Eb = ContinuousWavelets.cwt(data.Eb, wavelet)
    
    # Power spectrograms
    power_Ek = abs2.(cwt_Ek)
    power_Eb = abs2.(cwt_Eb)
    
    return (
        times = data.times,
        frequencies = freqs,
        scales = scales,
        cwt_Ek = cwt_Ek,
        cwt_Eb = cwt_Eb,
        power_Ek = power_Ek,
        power_Eb = power_Eb,
        wavelet_type = "Morlet"
    )
end

function compute_power_spectral_density(data)
    """Compute overall power spectral density using periodogram"""
    println("Computing power spectral density...")
    
    dt = median(diff(data.times))
    
    # Periodogram for total energies
    psd_Ek = DSP.periodogram(data.Ek, fs=1/dt)
    psd_Eb = DSP.periodogram(data.Eb, fs=1/dt)
    
    return (
        frequencies = psd_Ek.freq,
        psd_Ek = psd_Ek.power,
        psd_Eb = psd_Eb.power
    )
end

function analyze_mode_evolution(data)
    """Analyze energy evolution in different spherical harmonic modes"""
    println("Analyzing modal energy evolution...")
    
    # Group by l degree
    l_unique = unique(data.l_values)
    
    Ek_by_l = Dict()
    Eb_by_l = Dict()
    
    for l in l_unique
        l_mask = data.l_values .== l
        Ek_by_l[l] = vec(sum(data.Ek_lm[l_mask, :], dims=1))
        Eb_by_l[l] = vec(sum(data.Eb_lm[l_mask, :], dims=1))
    end
    
    # Compute relative contributions
    total_Ek = vec(sum(data.Ek_lm, dims=1))
    total_Eb = vec(sum(data.Eb_lm, dims=1))
    
    Ek_frac_by_l = Dict()
    Eb_frac_by_l = Dict()
    
    for l in l_unique
        Ek_frac_by_l[l] = Ek_by_l[l] ./ (total_Ek .+ eps(eltype(total_Ek)))
        Eb_frac_by_l[l] = Eb_by_l[l] ./ (total_Eb .+ eps(eltype(total_Eb)))
    end
    
    return (
        l_values = l_unique,
        Ek_by_l = Ek_by_l,
        Eb_by_l = Eb_by_l,
        Ek_fraction_by_l = Ek_frac_by_l,
        Eb_fraction_by_l = Eb_frac_by_l,
        total_Ek = total_Ek,
        total_Eb = total_Eb
    )
end

function main()
    args = ARGS
    params = parse_args(args)
    
    if params === nothing
        return
    end
    
    println("Time-Frequency Spectral Analysis of GeoDynamo Energies")
    println("=" ^ 55)
    println("Output directory: $(params[:output_dir])")
    println("Method: $(params[:method])")
    println("Tangent cylinder mode: $(params[:tc_mode]) (radius: $(params[:tc_radius]))")
    
    # Find input files
    files, times = find_merged_files(params[:output_dir], params[:prefix])
    if isempty(files)
        error("No merged files found matching pattern '$(params[:prefix])_time_*.nc' in $(params[:output_dir])")
    end
    
    println("Found $(length(files)) merged files spanning times [$(minimum(times)), $(maximum(times))]")
    
    # Load time series data
    data = load_time_series(files, times, params)
    
    # Resample if needed
    data = resample_time_series(data, params[:dt_target])
    
    println("Time series loaded: $(length(data.times)) points, Δt = $(round(median(diff(data.times)), digits=6))")
    
    # Initialize results
    results = Dict(
        :metadata => Dict(
            :analysis_time => now(),
            :method => params[:method],
            :n_files => length(files),
            :time_range => [minimum(data.times), maximum(data.times)],
            :n_time_points => length(data.times),
            :dt => median(diff(data.times)),
            :l_max_requested => params[:l_max],
            :l_max_actual => maximum(data.l_values),
            :window_size => params[:window_size],
            :overlap => params[:overlap],
            :tangent_cylinder => Dict(
                :mode => params[:tc_mode],
                :radius => params[:tc_radius]
            )
        ),
        :time_series => Dict(
            :times => data.times,
            :Ek_total => data.Ek,
            :Eb_total => data.Eb,
            :Ek_lm => data.Ek_lm,
            :Eb_lm => data.Eb_lm,
            :l_values => data.l_values,
            :m_values => data.m_values
        )
    )
    
    # Power spectral density (always computed)
    psd_data = compute_power_spectral_density(data)
    results[:psd] = psd_data
    
    # Modal evolution analysis
    mode_data = analyze_mode_evolution(data)
    results[:mode_evolution] = mode_data
    
    # Time-frequency analysis based on method
    if params[:method] in ["stft", "both"]
        stft_data = compute_stft(data, params)
        results[:stft] = stft_data
    end
    
    if params[:method] in ["wavelet", "both"] && HAS_WAVELETS
        wavelet_data = compute_wavelet_transform(data, params)
        if wavelet_data !== nothing
            results[:wavelet] = wavelet_data
        end
    end
    
    # Save results
    output_path = joinpath(params[:output_dir], params[:output_file])
    println("Saving results to: $output_path")
    
    JLD2.jldsave(output_path; results...)
    
    println("Analysis complete!")
    println("Results saved with keys: $(join(keys(results), ", "))")
    
    # Print summary statistics
    println("\nSummary Statistics:")
    println("─" ^ 40)
    println("Kinetic Energy:  mean = $(round(mean(data.Ek), digits=6)), std = $(round(std(data.Ek), digits=6))")
    println("Magnetic Energy: mean = $(round(mean(data.Eb), digits=6)), std = $(round(std(data.Eb), digits=6))")
    println("Dominant l modes (by time-averaged energy):")
    
    for (i, l) in enumerate(mode_data.l_values[1:min(5, length(mode_data.l_values))])
        avg_frac_k = mean(mode_data.Ek_fraction_by_l[l])
        avg_frac_b = mean(mode_data.Eb_fraction_by_l[l])
        println("  l=$l: Ek = $(round(avg_frac_k*100, digits=2))%, Eb = $(round(avg_frac_b*100, digits=2))%")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end