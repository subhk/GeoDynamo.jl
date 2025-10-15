#!/usr/bin/env julia

"""
Time-Frequency Spectra Analysis at Constant Radius with Tangent Cylinder Options

This script analyzes the time-frequency evolution of kinetic and magnetic energy
from merged Geodynamo simulation data at specific radial locations. It provides 
options to analyze:
- Full domain data at constant radius
- Data with tangent cylinder removed at constant radius
- Tangent cylinder region only at constant radius

The script supports multiple time-frequency analysis methods:
- Short-Time Fourier Transform (STFT)
- Continuous Wavelet Transform (CWT)
- Power Spectral Density (PSD) evolution

Key Features:
- Analysis at user-specified radial locations
- Spatial integration over spherical surface at each radius
- Tangent cylinder filtering based on θ coordinates
- Modal decomposition by spherical harmonic degree l

Usage:
    julia time_frequency_spectra_constant_radius.jl [options]

Options:
    --data_file <path>          Path to merged data file
    --output_dir <path>         Output directory for results
    --radius <r>                Radial location for analysis (default: 0.8)
    --tc_mode <mode>           Tangent cylinder mode: "full", "remove_tc", "tc_only"
    --analysis_method <method>  Analysis method: "stft", "cwt", "psd"
    --window_length <int>       Window length for STFT (default: 128)
    --overlap_ratio <float>     Overlap ratio for STFT (default: 0.75)
    --frequency_range <f1,f2>   Frequency range for analysis
    --save_format <format>      Save format: "hdf5", "netcdf", "jld2"
    --tc_radius <r>            Tangent cylinder radius (default: 0.35)
    --l_max <int>              Maximum l degree to analyze (default: all)

Examples:
    # Basic analysis at outer core boundary
    julia time_frequency_spectra_constant_radius.jl --data_file=output.h5 --radius=0.9

    # Remove tangent cylinder at mid-depth
    julia time_frequency_spectra_constant_radius.jl --data_file=output.h5 \
          --radius=0.7 --tc_mode=remove_tc --tc_radius=0.35

    # Analyze only TC region at inner core boundary  
    julia time_frequency_spectra_constant_radius.jl --data_file=output.h5 \
          --radius=0.4 --tc_mode=tc_only --tc_radius=0.4
"""

using HDF5
using JLD2
using NetCDF
using FFTW
using DSP
using Statistics
using LinearAlgebra
using Wavelets
using ArgParse
using Dates
using Printf

# Custom types for analysis configuration
struct TCConfig
    mode::Symbol  # :full, :remove_tc, :tc_only
    r_tc::Float64 # Tangent cylinder radius (normalized)
end

struct RadialConfig
    target_radius::Float64
    analysis_radius::Float64  # Actual radius used (closest available)
    radius_index::Int        # Index in radial grid
end

struct TimeFrequencyConfig
    method::Symbol
    window_length::Int
    overlap_ratio::Float64
    frequency_range::Tuple{Float64, Float64}
    n_frequencies::Int
end

struct SpectralResultConstantR
    time::Vector{Float64}
    frequency::Vector{Float64}
    ke_spectra::Matrix{Float64}
    me_spectra::Matrix{Float64}
    total_ke::Vector{Float64}
    total_me::Vector{Float64}
    ke_by_l::Dict{Int, Matrix{Float64}}  # Spectral data by l degree
    me_by_l::Dict{Int, Matrix{Float64}}
    tc_config::TCConfig
    radial_config::RadialConfig
    analysis_config::TimeFrequencyConfig
    l_values::Vector{Int}
end

function parse_commandline()
    s = ArgParseSettings(
        description = "Time-frequency spectra analysis at constant radius with tangent cylinder options",
        version = "1.0.0"
    )

    @add_arg_table! s begin
        "--data_file"
            help = "Path to merged data file"
            required = true
        "--output_dir"
            help = "Output directory for results"
            default = "./time_frequency_radius_results"
        "--radius"
            help = "Radial location for analysis"
            arg_type = Float64
            default = 0.8
        "--tc_mode"
            help = "Tangent cylinder mode: full, remove_tc, tc_only"
            default = "full"
        "--analysis_method"
            help = "Analysis method: stft, cwt, psd"
            default = "stft"
        "--window_length"
            help = "Window length for STFT"
            arg_type = Int
            default = 128
        "--overlap_ratio"
            help = "Overlap ratio for STFT"
            arg_type = Float64
            default = 0.75
        "--frequency_range"
            help = "Frequency range as f1,f2"
            default = "0.01,10.0"
        "--tc_radius"
            help = "Tangent cylinder radius (normalized)"
            arg_type = Float64
            default = 0.35
        "--save_format"
            help = "Save format: hdf5, netcdf, jld2"
            default = "hdf5"
        "--n_frequencies"
            help = "Number of frequency bins for CWT"
            arg_type = Int
            default = 100
        "--l_max"
            help = "Maximum l degree to analyze"
            arg_type = Int
            default = 20
    end

    return parse_args(s)
end

function load_geodynamo_spectral_data(filename::String)
    """Load merged geodynamo spectral data from HDF5/NetCDF file"""
    
    println("Loading spectral data from: $filename")
    
    if endswith(filename, ".h5") || endswith(filename, ".hdf5")
        data = h5open(filename, "r") do file
            times = read(file, "time")
            
            # Read spectral coefficients
            vel_tor_real = read(file, "velocity_toroidal_real")
            vel_tor_imag = read(file, "velocity_toroidal_imag")
            vel_pol_real = read(file, "velocity_poloidal_real")
            vel_pol_imag = read(file, "velocity_poloidal_imag")
            
            mag_tor_real = read(file, "magnetic_toroidal_real")
            mag_tor_imag = read(file, "magnetic_toroidal_imag")
            mag_pol_real = read(file, "magnetic_poloidal_real")
            mag_pol_imag = read(file, "magnetic_poloidal_imag")
            
            # Read coordinate information
            l_values = Int.(read(file, "l_values"))
            m_values = Int.(read(file, "m_values"))
            r_coords = read(file, "r_grid")
            
            Dict("time" => times, 
                 "vel_tor_real" => vel_tor_real, "vel_tor_imag" => vel_tor_imag,
                 "vel_pol_real" => vel_pol_real, "vel_pol_imag" => vel_pol_imag,
                 "mag_tor_real" => mag_tor_real, "mag_tor_imag" => mag_tor_imag,
                 "mag_pol_real" => mag_pol_real, "mag_pol_imag" => mag_pol_imag,
                 "l_values" => l_values, "m_values" => m_values, "r" => r_coords)
        end
    elseif endswith(filename, ".nc")
        times = ncread(filename, "time")
        
        vel_tor_real = ncread(filename, "velocity_toroidal_real")
        vel_tor_imag = ncread(filename, "velocity_toroidal_imag")
        vel_pol_real = ncread(filename, "velocity_poloidal_real")
        vel_pol_imag = ncread(filename, "velocity_poloidal_imag")
        
        mag_tor_real = ncread(filename, "magnetic_toroidal_real")
        mag_tor_imag = ncread(filename, "magnetic_toroidal_imag")
        mag_pol_real = ncread(filename, "magnetic_poloidal_real")
        mag_pol_imag = ncread(filename, "magnetic_poloidal_imag")
        
        l_values = Int.(ncread(filename, "l_values"))
        m_values = Int.(ncread(filename, "m_values"))
        r_coords = ncread(filename, "r_grid")
        
        data = Dict("time" => times,
                   "vel_tor_real" => vel_tor_real, "vel_tor_imag" => vel_tor_imag,
                   "vel_pol_real" => vel_pol_real, "vel_pol_imag" => vel_pol_imag,
                   "mag_tor_real" => mag_tor_real, "mag_tor_imag" => mag_tor_imag,
                   "mag_pol_real" => mag_pol_real, "mag_pol_imag" => mag_pol_imag,
                   "l_values" => l_values, "m_values" => m_values, "r" => r_coords)
    else
        error("Unsupported file format. Use .h5, .hdf5, or .nc files")
    end
    
    println("Spectral data loaded successfully:")
    println("  Time points: $(length(data["time"]))")
    println("  Spectral modes: $(length(data["l_values"]))")
    println("  Radial points: $(length(data["r"]))")
    
    return data
end

function find_radius_index(r_coords, target_radius)
    """Find the closest radial index to target radius"""
    distances = abs.(r_coords .- target_radius)
    idx = argmin(distances)
    actual_radius = r_coords[idx]
    
    println("Target radius: $target_radius")
    println("Actual radius: $actual_radius (index: $idx)")
    
    return RadialConfig(target_radius, actual_radius, idx)
end

function create_tangent_cylinder_mask_angular(l_values, m_values, r_tc::Float64, analysis_radius::Float64)
    """Create tangent cylinder mask based on angular coordinates at constant radius
    
    For analysis at constant radius r, the tangent cylinder condition is:
    r * sin(θ) < r_tc
    => sin(θ) < r_tc / r
    => θ_critical = arcsin(r_tc / r) (if r > r_tc)
    
    For spectral data, this affects different l,m modes differently.
    """
    
    nlm = length(l_values)
    mask = ones(Bool, nlm)  # true = include, false = exclude
    
    if analysis_radius <= r_tc
        # If analysis radius is inside TC, all modes are affected
        println("Analysis radius ($analysis_radius) is inside TC radius ($r_tc)")
        println("All modes will be considered within the tangent cylinder")
        return mask  # Return all true for full domain analysis
    end
    
    # Critical colatitude for tangent cylinder boundary
    sin_theta_crit = r_tc / analysis_radius
    theta_crit = asin(sin_theta_crit)
    
    println("TC critical angle: $(round(rad2deg(theta_crit), digits=2))° from pole")
    
    # For each spherical harmonic mode, determine if it's significantly
    # affected by the tangent cylinder
    for lm in 1:nlm
        l = l_values[lm]
        m = abs(m_values[lm])
        
        # Heuristic: higher l modes are more localized in latitude
        # Lower l modes have broader latitudinal extent
        
        if l == 0
            # l=0 mode is uniform - partially affected
            mask[lm] = true  # Keep for now, will be handled by mode-specific logic
        elseif l == 1
            # Dipole modes are global - significantly affected by TC
            mask[lm] = true
        elseif l <= 4
            # Low-l modes are affected based on their typical latitudinal extent
            # These modes have significant amplitude at low latitudes (near equator)
            mask[lm] = true
        else
            # Higher l modes are more localized - less affected by TC at poles
            # Their main energy is at intermediate to high latitudes
            mask[lm] = true
        end
    end
    
    return mask
end

function apply_tangent_cylinder_filter_angular(spectral_data_r, l_values, m_values, 
                                             tc_mask, tc_mode, analysis_radius, r_tc)
    """Apply tangent cylinder filtering to spectral data at constant radius
    
    This is more sophisticated than the radial version as we need to consider
    the angular structure of spherical harmonics at the analysis radius.
    """
    
    filtered_data = copy(spectral_data_r)
    nlm = length(l_values)
    
    if tc_mode == "full"
        return filtered_data
    end
    
    if analysis_radius <= r_tc
        # Special case: analysis radius is inside TC
        if tc_mode == "remove_tc"
            # Remove everything (zero out all modes)
            return zeros(eltype(filtered_data), size(filtered_data))
        elseif tc_mode == "tc_only"
            # Keep everything (already inside TC)
            return filtered_data
        end
    end
    
    # Calculate the fraction of each mode's energy that's inside/outside TC
    sin_theta_crit = r_tc / analysis_radius
    theta_crit = asin(sin_theta_crit)
    
    for lm in 1:nlm
        l = l_values[lm]
        m = abs(m_values[lm])
        
        # Calculate approximate fraction of mode energy inside TC
        # This is a simplified model - real calculation would require
        # integration of |Y_l^m(θ,φ)|² over the TC region
        
        if l == 0
            # l=0 mode: uniform distribution
            # Fraction inside TC ≈ solid angle fraction
            solid_angle_frac = 2 * (1 - cos(theta_crit)) / 2  # Hemisphere fraction
            inside_frac = solid_angle_frac
        elseif l == 1
            # Dipole modes: significant energy at poles and equator
            if m == 0
                # Y₁⁰ ∝ cos(θ) - more energy at poles
                inside_frac = 0.7  # Rough estimate
            else
                # Y₁^±¹ ∝ sin(θ) - more energy at equator  
                inside_frac = 0.3  # Rough estimate
            end
        else
            # Higher l modes: estimate based on typical latitudinal distribution
            # Higher l modes tend to have more energy at intermediate latitudes
            if l <= 4
                inside_frac = 0.4 - 0.05 * (l - 2)  # Decreasing with l
            else
                inside_frac = 0.2 - 0.02 * (l - 4)  # Further decreasing
            end
            inside_frac = max(inside_frac, 0.05)  # Minimum 5%
        end
        
        # Apply filtering based on mode
        if tc_mode == "remove_tc"
            # Remove the inside-TC fraction
            filtered_data[lm] *= (1.0 - inside_frac)
        elseif tc_mode == "tc_only"
            # Keep only the inside-TC fraction
            filtered_data[lm] *= inside_frac
        end
    end
    
    return filtered_data
end

function compute_energy_at_radius(data::Dict, radial_config::RadialConfig, 
                                tc_config::TCConfig, l_max::Int)
    """Compute time series of kinetic and magnetic energy at specified radius"""
    
    println("\n" * "="^60)
    println("Energy Analysis at Constant Radius")
    println("  Analysis radius: $(radial_config.analysis_radius)")
    println("  TC Mode: $(tc_config.mode)")
    println("  TC Radius: $(tc_config.r_tc)")
    println("  Max l degree: $l_max")
    println("="^60)
    
    # Extract data at the specified radius
    r_idx = radial_config.radius_index
    times = data["time"]
    nt = length(times)
    
    # Filter by l_max
    l_mask = data["l_values"] .<= l_max
    l_values = data["l_values"][l_mask]
    m_values = data["m_values"][l_mask]
    nlm = length(l_values)
    
    println("Using $(nlm) spherical harmonic modes (l ≤ $l_max)")
    
    # Create tangent cylinder mask for angular filtering
    tc_mask = create_tangent_cylinder_mask_angular(l_values, m_values, 
                                                  tc_config.r_tc, radial_config.analysis_radius)
    
    # Extract spectral data at radius and apply l_max filter
    vel_tor_real_r = data["vel_tor_real"][l_mask, r_idx, :]
    vel_tor_imag_r = data["vel_tor_imag"][l_mask, r_idx, :]
    vel_pol_real_r = data["vel_pol_real"][l_mask, r_idx, :]
    vel_pol_imag_r = data["vel_pol_imag"][l_mask, r_idx, :]
    
    mag_tor_real_r = data["mag_tor_real"][l_mask, r_idx, :]
    mag_tor_imag_r = data["mag_tor_imag"][l_mask, r_idx, :]
    mag_pol_real_r = data["mag_pol_real"][l_mask, r_idx, :]
    mag_pol_imag_r = data["mag_pol_imag"][l_mask, r_idx, :]
    
    # Initialize energy time series
    ke_total = zeros(nt)
    me_total = zeros(nt)
    ke_by_l = Dict{Int, Vector{Float64}}()
    me_by_l = Dict{Int, Vector{Float64}}()
    
    # Initialize by l degree
    l_unique = unique(l_values)
    for l in l_unique
        ke_by_l[l] = zeros(nt)
        me_by_l[l] = zeros(nt)
    end
    
    # Compute energy at each time step
    for t in 1:nt
        # Extract time slice
        vtr = vel_tor_real_r[:, t]
        vti = vel_tor_imag_r[:, t]
        vpr = vel_pol_real_r[:, t]
        vpi = vel_pol_imag_r[:, t]
        
        mtr = mag_tor_real_r[:, t]
        mti = mag_tor_imag_r[:, t]
        mpr = mag_pol_real_r[:, t]
        mpi = mag_pol_imag_r[:, t]
        
        # Apply tangent cylinder filtering
        if tc_config.mode != :full
            vtr = apply_tangent_cylinder_filter_angular(vtr, l_values, m_values, tc_mask, 
                                                       String(tc_config.mode), radial_config.analysis_radius, tc_config.r_tc)
            vti = apply_tangent_cylinder_filter_angular(vti, l_values, m_values, tc_mask,
                                                       String(tc_config.mode), radial_config.analysis_radius, tc_config.r_tc)
            vpr = apply_tangent_cylinder_filter_angular(vpr, l_values, m_values, tc_mask,
                                                       String(tc_config.mode), radial_config.analysis_radius, tc_config.r_tc)
            vpi = apply_tangent_cylinder_filter_angular(vpi, l_values, m_values, tc_mask,
                                                       String(tc_config.mode), radial_config.analysis_radius, tc_config.r_tc)
            
            mtr = apply_tangent_cylinder_filter_angular(mtr, l_values, m_values, tc_mask,
                                                       String(tc_config.mode), radial_config.analysis_radius, tc_config.r_tc)
            mti = apply_tangent_cylinder_filter_angular(mti, l_values, m_values, tc_mask,
                                                       String(tc_config.mode), radial_config.analysis_radius, tc_config.r_tc)
            mpr = apply_tangent_cylinder_filter_angular(mpr, l_values, m_values, tc_mask,
                                                       String(tc_config.mode), radial_config.analysis_radius, tc_config.r_tc)
            mpi = apply_tangent_cylinder_filter_angular(mpi, l_values, m_values, tc_mask,
                                                       String(tc_config.mode), radial_config.analysis_radius, tc_config.r_tc)
        end
        
        # Compute energy for each mode
        ke_t = 0.0
        me_t = 0.0
        
        for lm in 1:nlm
            l = l_values[lm]
            
            # Skip l=0 for velocity
            if l > 0
                # Kinetic energy density at this radius
                ke_mode = 0.5 * (vtr[lm]^2 + vti[lm]^2 + vpr[lm]^2 + vpi[lm]^2)
                ke_t += ke_mode
                ke_by_l[l][t] += ke_mode
            end
            
            # Magnetic energy density at this radius
            me_mode = 0.5 * (mtr[lm]^2 + mti[lm]^2 + mpr[lm]^2 + mpi[lm]^2)
            me_t += me_mode
            me_by_l[l][t] += me_mode
        end
        
        ke_total[t] = ke_t
        me_total[t] = me_t
    end
    
    println("Energy computation complete:")
    println("  Mean KE: $(round(mean(ke_total), digits=6))")
    println("  Mean ME: $(round(mean(me_total), digits=6))")
    println("  KE RMS: $(round(sqrt(mean(ke_total.^2)), digits=6))")
    println("  ME RMS: $(round(sqrt(mean(me_total.^2)), digits=6))")
    
    return times, ke_total, me_total, ke_by_l, me_by_l, l_unique
end

function compute_stft_spectra(signal::Vector{Float64}, times::Vector{Float64}, 
                            config::TimeFrequencyConfig)
    """Compute Short-Time Fourier Transform spectra"""
    
    println("Computing STFT spectra...")
    
    # Parameters
    fs = 1.0 / (times[2] - times[1])  # Sampling frequency
    window_length = config.window_length
    overlap = Int(round(window_length * config.overlap_ratio))
    
    # Create analysis window
    window = DSP.hanning(window_length)
    
    # Compute STFT
    stft_result = DSP.stft(signal, window_length, overlap; 
                          window=window, fs=fs)
    
    # Extract frequency and time grids
    freqs = DSP.fftfreq(window_length, fs)[1:window_length÷2+1]
    n_windows = size(stft_result, 2)
    time_centers = [(i-1) * (window_length - overlap) / fs + times[1] + window_length/(2*fs) 
                   for i in 1:n_windows]
    
    # Filter frequency range
    f_min, f_max = config.frequency_range
    freq_mask = (freqs .>= f_min) .& (freqs .<= f_max)
    
    # Power spectral density
    psd = abs2.(stft_result[freq_mask, :])
    
    return time_centers, freqs[freq_mask], psd
end

function compute_cwt_spectra(signal::Vector{Float64}, times::Vector{Float64},
                           config::TimeFrequencyConfig)
    """Compute Continuous Wavelet Transform spectra"""
    
    println("Computing CWT spectra...")
    
    # Parameters
    fs = 1.0 / (times[2] - times[1])
    f_min, f_max = config.frequency_range
    n_freq = config.n_frequencies
    
    # Create frequency scale
    frequencies = exp.(range(log(f_min), log(f_max), length=n_freq))
    scales = fs ./ frequencies
    
    # Compute CWT using Morlet wavelet
    wavelet = Wavelets.Morlet()
    cwt_result = Wavelets.cwt(signal, wavelet, scales)
    
    # Power spectrum
    power = abs2.(cwt_result)
    
    return times, frequencies, power
end

function compute_psd_evolution(signal::Vector{Float64}, times::Vector{Float64},
                             config::TimeFrequencyConfig)
    """Compute evolution of Power Spectral Density using sliding windows"""
    
    println("Computing PSD evolution...")
    
    fs = 1.0 / (times[2] - times[1])
    window_length = config.window_length
    overlap = Int(round(window_length * config.overlap_ratio))
    step = window_length - overlap
    
    n_windows = (length(signal) - window_length) ÷ step + 1
    freqs = DSP.fftfreq(window_length, fs)[1:window_length÷2+1]
    
    # Filter frequency range
    f_min, f_max = config.frequency_range  
    freq_mask = (freqs .>= f_min) .& (freqs .<= f_max)
    filtered_freqs = freqs[freq_mask]
    
    psd_matrix = zeros(length(filtered_freqs), n_windows)
    time_centers = zeros(n_windows)
    
    for i in 1:n_windows
        start_idx = (i-1) * step + 1
        end_idx = start_idx + window_length - 1
        
        if end_idx > length(signal)
            break
        end
        
        window_signal = signal[start_idx:end_idx]
        time_centers[i] = times[start_idx + window_length÷2]
        
        # Compute PSD for this window
        psd_full = DSP.periodogram(window_signal, fs=fs).power
        psd_matrix[:, i] = psd_full[freq_mask]
    end
    
    return time_centers[1:size(psd_matrix,2)], filtered_freqs, psd_matrix
end

function analyze_time_frequency_constant_radius(data::Dict, tc_config::TCConfig, 
                                              radial_config::RadialConfig,
                                              tf_config::TimeFrequencyConfig, l_max::Int)
    """Main time-frequency analysis function at constant radius"""
    
    # Compute energy time series at specified radius
    times, ke_total, me_total, ke_by_l, me_by_l, l_unique = compute_energy_at_radius(
        data, radial_config, tc_config, l_max)
    
    # Compute time-frequency analysis for total energies
    if tf_config.method == :stft
        time_grid, freq_grid, ke_spectra = compute_stft_spectra(ke_total, times, tf_config)
        _, _, me_spectra = compute_stft_spectra(me_total, times, tf_config)
        
    elseif tf_config.method == :cwt
        time_grid, freq_grid, ke_spectra = compute_cwt_spectra(ke_total, times, tf_config)
        _, _, me_spectra = compute_cwt_spectra(me_total, times, tf_config)
        
    elseif tf_config.method == :psd
        time_grid, freq_grid, ke_spectra = compute_psd_evolution(ke_total, times, tf_config)
        _, _, me_spectra = compute_psd_evolution(me_total, times, tf_config)
        
    else
        error("Unknown analysis method: $(tf_config.method)")
    end
    
    # Compute time-frequency analysis by l degree
    ke_spectra_by_l = Dict{Int, Matrix{Float64}}()
    me_spectra_by_l = Dict{Int, Matrix{Float64}}()
    
    for l in l_unique[1:min(5, length(l_unique))]  # Analyze first 5 l modes
        if tf_config.method == :stft
            _, _, ke_l_spec = compute_stft_spectra(ke_by_l[l], times, tf_config)
            _, _, me_l_spec = compute_stft_spectra(me_by_l[l], times, tf_config)
        elseif tf_config.method == :cwt
            _, _, ke_l_spec = compute_cwt_spectra(ke_by_l[l], times, tf_config)
            _, _, me_l_spec = compute_cwt_spectra(me_by_l[l], times, tf_config)
        elseif tf_config.method == :psd
            _, _, ke_l_spec = compute_psd_evolution(ke_by_l[l], times, tf_config)
            _, _, me_l_spec = compute_psd_evolution(me_by_l[l], times, tf_config)
        end
        
        ke_spectra_by_l[l] = ke_l_spec
        me_spectra_by_l[l] = me_l_spec
    end
    
    # Create result structure
    result = SpectralResultConstantR(
        time_grid, freq_grid, ke_spectra, me_spectra,
        ke_total, me_total, ke_spectra_by_l, me_spectra_by_l,
        tc_config, radial_config, tf_config, collect(l_unique)
    )
    
    return result
end

function save_results_constant_radius(result::SpectralResultConstantR, output_dir::String, 
                                    save_format::String, data_file::String)
    """Save analysis results in specified format"""
    
    mkpath(output_dir)
    
    # Create filename
    tc_suffix = String(result.tc_config.mode)
    method_suffix = String(result.analysis_config.method)
    radius_str = replace(string(round(result.radial_config.analysis_radius, digits=3)), "." => "p")
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    
    base_filename = "time_freq_r$(radius_str)_$(method_suffix)_$(tc_suffix)_$(timestamp)"
    
    println("\nSaving results to: $output_dir")
    
    if save_format == "hdf5"
        filename = joinpath(output_dir, "$base_filename.h5")
        
        h5open(filename, "w") do file
            # Time-frequency data
            write(file, "time", result.time)
            write(file, "frequency", result.frequency)
            write(file, "ke_spectra", result.ke_spectra)
            write(file, "me_spectra", result.me_spectra)
            
            # Integrated time series
            write(file, "ke_total", result.total_ke)
            write(file, "me_total", result.total_me)
            
            # Modal analysis
            write(file, "l_values", result.l_values)
            
            # Spectra by l degree
            for l in keys(result.ke_by_l)
                write(file, "ke_spectra_l$l", result.ke_by_l[l])
                write(file, "me_spectra_l$l", result.me_by_l[l])
            end
            
            # Configuration
            attrs(file)["tc_mode"] = String(result.tc_config.mode)
            attrs(file)["tc_radius"] = result.tc_config.r_tc
            attrs(file)["target_radius"] = result.radial_config.target_radius
            attrs(file)["analysis_radius"] = result.radial_config.analysis_radius
            attrs(file)["radius_index"] = result.radial_config.radius_index
            attrs(file)["analysis_method"] = String(result.analysis_config.method)
            attrs(file)["window_length"] = result.analysis_config.window_length
            attrs(file)["overlap_ratio"] = result.analysis_config.overlap_ratio
            attrs(file)["frequency_range"] = [result.analysis_config.frequency_range...]
            attrs(file)["source_data_file"] = data_file
            attrs(file)["creation_time"] = string(now())
        end
        
    elseif save_format == "jld2"
        filename = joinpath(output_dir, "$base_filename.jld2")
        
        jldsave(filename;
               result=result,
               source_file=data_file,
               creation_time=now())
    else
        error("NetCDF format not implemented for this analysis. Use hdf5 or jld2.")
    end
    
    println("Results saved to: $filename")
    
    # Save summary statistics
    summary_file = joinpath(output_dir, "$(base_filename)_summary.txt")
    open(summary_file, "w") do f
        println(f, "Time-Frequency Spectra Analysis at Constant Radius")
        println(f, "="^60)
        println(f, "Source data file: $data_file")
        println(f, "Analysis timestamp: $(now())")
        println(f, "")
        println(f, "Configuration:")
        println(f, "  Target radius: $(result.radial_config.target_radius)")
        println(f, "  Analysis radius: $(result.radial_config.analysis_radius)")
        println(f, "  Radius index: $(result.radial_config.radius_index)")
        println(f, "  Tangent cylinder mode: $(result.tc_config.mode)")
        println(f, "  TC radius: $(result.tc_config.r_tc)")
        println(f, "  Analysis method: $(result.analysis_config.method)")
        println(f, "  Window length: $(result.analysis_config.window_length)")
        println(f, "  Overlap ratio: $(result.analysis_config.overlap_ratio)")
        println(f, "  Frequency range: $(result.analysis_config.frequency_range)")
        println(f, "")
        println(f, "Results:")
        println(f, "  Time points: $(length(result.time))")
        println(f, "  Frequency bins: $(length(result.frequency))")
        println(f, "  L degrees analyzed: $(result.l_values)")
        println(f, "  KE mean: $(round(mean(result.total_ke), digits=6))")
        println(f, "  ME mean: $(round(mean(result.total_me), digits=6))")
        println(f, "  KE max spectral power: $(round(maximum(result.ke_spectra), digits=6))")
        println(f, "  ME max spectral power: $(round(maximum(result.me_spectra), digits=6))")
        
        println(f, "")
        println(f, "Modal Energy Distribution:")
        for l in result.l_values[1:min(5, length(result.l_values))]
            if haskey(result.ke_by_l, l)
                ke_frac = mean(result.ke_by_l[l]) / mean(result.total_ke) * 100
                me_frac = mean(result.me_by_l[l]) / mean(result.total_me) * 100
                println(f, "  l=$l: KE $(round(ke_frac, digits=1))%, ME $(round(me_frac, digits=1))%")
            end
        end
    end
    
    return filename
end

function main()
    println("Time-Frequency Spectra Analysis at Constant Radius")
    println("="^60)
    
    # Parse command line arguments
    args = parse_commandline()
    
    # Parse frequency range
    f_range_str = split(args["frequency_range"], ",")
    f_range = (parse(Float64, f_range_str[1]), parse(Float64, f_range_str[2]))
    
    # Create configuration objects
    tc_config = TCConfig(Symbol(args["tc_mode"]), args["tc_radius"])
    
    tf_config = TimeFrequencyConfig(
        Symbol(args["analysis_method"]),
        args["window_length"],
        args["overlap_ratio"],
        f_range,
        args["n_frequencies"]
    )
    
    # Load spectral data
    data = load_geodynamo_spectral_data(args["data_file"])
    
    # Find radius index
    radial_config = find_radius_index(data["r"], args["radius"])
    
    # Perform analysis
    result = analyze_time_frequency_constant_radius(data, tc_config, radial_config, 
                                                   tf_config, args["l_max"])
    
    # Save results
    output_file = save_results_constant_radius(result, args["output_dir"], args["save_format"], 
                                              args["data_file"])
    
    println("\n" * "="^60)
    println("Analysis completed successfully!")
    println("Results saved to: $(dirname(output_file))")
    println("="^60)
    
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end