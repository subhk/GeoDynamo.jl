#!/usr/bin/env julia

"""
Plot time-frequency spectra results from time_frequency_spectra.jl output.

This script creates publication-ready plots including:
1. Time series plots of total and modal energies
2. Power spectral density plots  
3. STFT spectrograms (time-frequency plots)
4. Wavelet power spectrograms
5. Modal energy evolution plots
6. Comparative energy analysis

Usage examples:

  # Basic plotting with default parameters
  julia --project=. scripts/plot_time_frequency_spectra.jl time_frequency_spectra.jld2

  # Custom output directory and format
  julia --project=. scripts/plot_time_frequency_spectra.jl time_frequency_spectra.jld2 \
        --outdir=./plots --format=pdf --dpi=300

  # Select specific plot types
  julia --project=. scripts/plot_time_frequency_spectra.jl results.jld2 \
        --plots=timeseries,stft,psd --l_modes=1,2,3

Options:
  --outdir=<dir>     Output directory for plots (default: same as input file)
  --format=<fmt>     Output format: png, pdf, svg (default: png)
  --dpi=<N>          Resolution for raster formats (default: 200)
  --plots=<list>     Comma-separated plot types: timeseries, psd, stft, wavelet, modes, all (default: all)
  --l_modes=<list>   Comma-separated l values to highlight in mode plots (default: 1,2,3,4,5)
  --figsize=<WxH>    Figure size in inches (default: 12x8)
  --colormap=<name>  Colormap for spectrograms (default: viridis)
  --log_scale        Use log scale for frequency axes
  --time_range=<t1,t2>  Time range to plot (default: full range)
"""

using JLD2
using Printf
using Statistics
using LinearAlgebra

# Plotting packages
using Plots
using ColorSchemes
using LaTeXStrings
using StatsPlots

# Set default plotting backend
gr()

function usage()
    println("Usage: plot_time_frequency_spectra.jl <input.jld2> [options]")
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
    
    input_file = args[1]
    if !isfile(input_file)
        error("Input file not found: $input_file")
    end
    
    # Default parameters
    params = Dict(
        :input_file => input_file,
        :outdir => dirname(abspath(input_file)),
        :format => "png",
        :dpi => 200,
        :plots => "all",
        :l_modes => [1, 2, 3, 4, 5],
        :figsize => (12, 8),
        :colormap => :viridis,
        :log_scale => false,
        :time_range => nothing
    )
    
    # Parse optional arguments
    for arg in args[2:end]
        if startswith(arg, "--outdir=")
            params[:outdir] = arg[9:end]
        elseif startswith(arg, "--format=")
            fmt = arg[10:end]
            if fmt ∉ ["png", "pdf", "svg"]
                error("Format must be png, pdf, or svg")
            end
            params[:format] = fmt
        elseif startswith(arg, "--dpi=")
            params[:dpi] = parse(Int, arg[7:end])
        elseif startswith(arg, "--plots=")
            plots_str = arg[9:end]
            if plots_str == "all"
                params[:plots] = ["timeseries", "psd", "stft", "wavelet", "modes"]
            else
                params[:plots] = split(plots_str, ",")
            end
        elseif startswith(arg, "--l_modes=")
            params[:l_modes] = parse.(Int, split(arg[11:end], ","))
        elseif startswith(arg, "--figsize=")
            size_str = arg[11:end]
            w, h = parse.(Float64, split(size_str, "x"))
            params[:figsize] = (w, h)
        elseif startswith(arg, "--colormap=")
            cmap_str = arg[12:end]
            params[:colormap] = Symbol(cmap_str)
        elseif arg == "--log_scale"
            params[:log_scale] = true
        elseif startswith(arg, "--time_range=")
            t_str = arg[14:end]
            t1, t2 = parse.(Float64, split(t_str, ","))
            params[:time_range] = (t1, t2)
        else
            error("Unknown argument: $arg")
        end
    end
    
    return params
end

function load_results(filename)
    """Load time-frequency analysis results"""
    println("Loading results from: $filename")
    return JLD2.load(filename)
end

function apply_time_filter(data, time_range)
    """Filter data by time range if specified"""
    if time_range === nothing
        return data
    end
    
    times = data[:time_series][:times]
    t1, t2 = time_range
    mask = (times .>= t1) .& (times .<= t2)
    
    # Filter time series data
    filtered = deepcopy(data)
    filtered[:time_series][:times] = times[mask]
    filtered[:time_series][:Ek_total] = data[:time_series][:Ek_total][mask]
    filtered[:time_series][:Eb_total] = data[:time_series][:Eb_total][mask]
    filtered[:time_series][:Ek_lm] = data[:time_series][:Ek_lm][:, mask]
    filtered[:time_series][:Eb_lm] = data[:time_series][:Eb_lm][:, mask]
    
    # Note: STFT and wavelet data would need separate filtering
    # For now, we'll work with the full frequency domain data
    
    return filtered
end

function plot_time_series(data, params)
    """Plot energy time series"""
    println("Creating time series plots...")
    
    times = data[:time_series][:times]
    Ek = data[:time_series][:Ek_total]
    Eb = data[:time_series][:Eb_total]
    
    # Total energy plot
    p1 = plot(times, Ek, label="Kinetic Energy", linewidth=2, color=:blue,
             xlabel="Time", ylabel="Energy", title="Total Energy Evolution")
    plot!(p1, times, Eb, label="Magnetic Energy", linewidth=2, color=:red)
    plot!(p1, times, Ek .+ Eb, label="Total Energy", linewidth=2, color=:black, linestyle=:dash)
    
    # Modal energy evolution for selected l modes
    if haskey(data, :mode_evolution)
        mode_data = data[:mode_evolution]
        l_values = intersect(params[:l_modes], mode_data[:l_values])
        
        if !isempty(l_values)
            p2 = plot(xlabel="Time", ylabel="Energy Fraction", 
                     title="Modal Energy Fractions (Kinetic)")
            p3 = plot(xlabel="Time", ylabel="Energy Fraction",
                     title="Modal Energy Fractions (Magnetic)")
            
            colors = palette(:Set1_5)
            for (i, l) in enumerate(l_values)
                if haskey(mode_data[:Ek_fraction_by_l], l)
                    plot!(p2, times, mode_data[:Ek_fraction_by_l][l], 
                         label="l=$l", linewidth=2, color=colors[i])
                    plot!(p3, times, mode_data[:Eb_fraction_by_l][l], 
                         label="l=$l", linewidth=2, color=colors[i])
                end
            end
            
            layout = @layout [a; b c]
            combined = plot(p1, p2, p3, layout=layout, size=params[:figsize].*100, dpi=params[:dpi])
        else
            combined = plot(p1, size=params[:figsize].*100, dpi=params[:dpi])
        end
    else
        combined = plot(p1, size=params[:figsize].*100, dpi=params[:dpi])
    end
    
    return combined
end

function plot_power_spectral_density(data, params)
    """Plot power spectral density"""
    println("Creating PSD plots...")
    
    psd_data = data[:psd]
    freqs = psd_data[:frequencies]
    
    yscale = params[:log_scale] ? :log10 : :identity
    xscale = params[:log_scale] ? :log10 : :identity
    
    p1 = plot(freqs, psd_data[:psd_Ek], 
             xlabel="Frequency", ylabel="Power Spectral Density",
             title="Kinetic Energy PSD", linewidth=2, color=:blue,
             xscale=xscale, yscale=yscale)
             
    p2 = plot(freqs, psd_data[:psd_Eb],
             xlabel="Frequency", ylabel="Power Spectral Density", 
             title="Magnetic Energy PSD", linewidth=2, color=:red,
             xscale=xscale, yscale=yscale)
    
    combined = plot(p1, p2, layout=(2,1), size=params[:figsize].*100, dpi=params[:dpi])
    return combined
end

function plot_stft_spectrograms(data, params)
    """Plot STFT spectrograms"""
    if !haskey(data, :stft)
        @warn "No STFT data available"
        return nothing
    end
    
    println("Creating STFT spectrograms...")
    
    stft_data = data[:stft]
    times = stft_data[:times]
    freqs = stft_data[:frequencies]
    
    yscale = params[:log_scale] ? :log10 : :identity
    
    # Kinetic energy spectrogram
    p1 = heatmap(times, freqs, log10.(stft_data[:psd_Ek] .+ eps()),
                xlabel="Time", ylabel="Frequency", 
                title="Kinetic Energy STFT", 
                color=params[:colormap], yscale=yscale)
                
    # Magnetic energy spectrogram  
    p2 = heatmap(times, freqs, log10.(stft_data[:psd_Eb] .+ eps()),
                xlabel="Time", ylabel="Frequency",
                title="Magnetic Energy STFT",
                color=params[:colormap], yscale=yscale)
    
    combined = plot(p1, p2, layout=(2,1), size=params[:figsize].*100, dpi=params[:dpi])
    return combined
end

function plot_wavelet_spectrograms(data, params)
    """Plot wavelet spectrograms"""
    if !haskey(data, :wavelet)
        @warn "No wavelet data available"
        return nothing
    end
    
    println("Creating wavelet spectrograms...")
    
    wavelet_data = data[:wavelet]
    times = wavelet_data[:times]
    freqs = wavelet_data[:frequencies]
    
    yscale = params[:log_scale] ? :log10 : :identity
    
    # Kinetic energy wavelet
    p1 = heatmap(times, freqs, log10.(wavelet_data[:power_Ek] .+ eps()),
                xlabel="Time", ylabel="Frequency",
                title="Kinetic Energy Wavelet Transform",
                color=params[:colormap], yscale=yscale)
                
    # Magnetic energy wavelet
    p2 = heatmap(times, freqs, log10.(wavelet_data[:power_Eb] .+ eps()),
                xlabel="Time", ylabel="Frequency", 
                title="Magnetic Energy Wavelet Transform",
                color=params[:colormap], yscale=yscale)
    
    combined = plot(p1, p2, layout=(2,1), size=params[:figsize].*100, dpi=params[:dpi])
    return combined
end

function plot_mode_analysis(data, params)
    """Plot detailed modal analysis"""
    if !haskey(data, :mode_evolution)
        @warn "No mode evolution data available"
        return nothing
    end
    
    println("Creating modal analysis plots...")
    
    mode_data = data[:mode_evolution]
    times = data[:time_series][:times]
    l_values = intersect(params[:l_modes], mode_data[:l_values])
    
    if isempty(l_values)
        @warn "No requested l modes found in data"
        return nothing
    end
    
    # Time-averaged energy by l mode
    avg_Ek_by_l = [mean(mode_data[:Ek_by_l][l]) for l in mode_data[:l_values]]
    avg_Eb_by_l = [mean(mode_data[:Eb_by_l][l]) for l in mode_data[:l_values]]
    
    p1 = bar(mode_data[:l_values], avg_Ek_by_l, 
            xlabel="l mode", ylabel="Time-averaged Energy",
            title="Kinetic Energy by Spherical Harmonic Degree",
            color=:blue, alpha=0.7)
            
    p2 = bar(mode_data[:l_values], avg_Eb_by_l,
            xlabel="l mode", ylabel="Time-averaged Energy", 
            title="Magnetic Energy by Spherical Harmonic Degree",
            color=:red, alpha=0.7)
    
    # Energy fraction evolution for selected modes
    colors = palette(:Set1_9)
    p3 = plot(xlabel="Time", ylabel="Energy Fraction", 
             title="Kinetic Energy Fractions")
    p4 = plot(xlabel="Time", ylabel="Energy Fraction",
             title="Magnetic Energy Fractions")
    
    for (i, l) in enumerate(l_values)
        if haskey(mode_data[:Ek_fraction_by_l], l)
            plot!(p3, times, mode_data[:Ek_fraction_by_l][l] * 100, 
                 label="l=$l", linewidth=2, color=colors[i])
            plot!(p4, times, mode_data[:Eb_fraction_by_l][l] * 100,
                 label="l=$l", linewidth=2, color=colors[i])
        end
    end
    
    ylabel!(p3, "Energy Fraction (%)")
    ylabel!(p4, "Energy Fraction (%)")
    
    layout = @layout [a b; c d]
    combined = plot(p1, p2, p3, p4, layout=layout, size=params[:figsize].*100, dpi=params[:dpi])
    return combined
end

function save_plot(p, filename, format, dpi)
    """Save plot with specified format and DPI"""
    if p !== nothing
        full_path = "$filename.$format"
        savefig(p, full_path)
        println("Saved: $full_path")
    end
end

function main()
    args = ARGS
    params = parse_args(args)
    
    if params === nothing
        return
    end
    
    println("Time-Frequency Spectra Plotting")
    println("=" ^ 35)
    println("Input file: $(params[:input_file])")
    println("Output directory: $(params[:outdir])")
    println("Format: $(params[:format])")
    
    # Create output directory if needed
    if !isdir(params[:outdir])
        mkpath(params[:outdir])
    end
    
    # Load results
    data = load_results(params[:input_file])
    
    # Apply time filtering if specified
    data = apply_time_filter(data, params[:time_range])
    
    # Get base filename for outputs
    base_name = splitext(basename(params[:input_file]))[1]
    
    # Generate plots based on selection
    plot_types = isa(params[:plots], String) ? [params[:plots]] : params[:plots]
    
    for plot_type in plot_types
        output_file = joinpath(params[:outdir], "$(base_name)_$(plot_type)")
        
        if plot_type == "timeseries"
            p = plot_time_series(data, params)
            save_plot(p, output_file, params[:format], params[:dpi])
            
        elseif plot_type == "psd" 
            p = plot_power_spectral_density(data, params)
            save_plot(p, output_file, params[:format], params[:dpi])
            
        elseif plot_type == "stft"
            p = plot_stft_spectrograms(data, params)
            save_plot(p, output_file, params[:format], params[:dpi])
            
        elseif plot_type == "wavelet"
            p = plot_wavelet_spectrograms(data, params) 
            save_plot(p, output_file, params[:format], params[:dpi])
            
        elseif plot_type == "modes"
            p = plot_mode_analysis(data, params)
            save_plot(p, output_file, params[:format], params[:dpi])
            
        else
            @warn "Unknown plot type: $plot_type"
        end
    end
    
    println("\nPlotting complete!")
    
    # Print summary
    println("\nDataset Summary:")
    println("─" ^ 30)
    metadata = data[:metadata]
    println("Time range: [$(metadata[:time_range][1]), $(metadata[:time_range][2])]")
    println("Time points: $(metadata[:n_time_points])")
    println("Time spacing: Δt = $(round(metadata[:dt], digits=6))")
    println("Max l degree: $(metadata[:l_max_actual])")
    
    if haskey(data, :mode_evolution)
        mode_data = data[:mode_evolution]
        total_Ek = mean(mode_data[:total_Ek])
        total_Eb = mean(mode_data[:total_Eb])
        println("Mean total kinetic energy: $(round(total_Ek, digits=6))")
        println("Mean total magnetic energy: $(round(total_Eb, digits=6))")
        println("Kinetic/Magnetic ratio: $(round(total_Ek/total_Eb, digits=3))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end