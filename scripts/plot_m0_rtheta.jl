#!/usr/bin/env julia

# Plot the m=0 (axisymmetric) mode of a 3D field as an r-theta heatmap.
#
# Assumes a NetCDF variable with dimensions compatible with (theta, phi, r).
# Computes the zonal mean (average over phi), which corresponds to the m=0 mode
# when treating phi as the azimuthal coordinate.
#
# Usage:
#   # Temperature m=0
#   julia --project=. scripts/plot_m0_rtheta.jl out.nc temperature
#   # Composition m=0 with custom colormap and limits
#   julia --project=. scripts/plot_m0_rtheta.jl out.nc composition --cmap magma --clim -0.1 0.3
#   # Plot theta in degrees
#   julia --project=. scripts/plot_m0_rtheta.jl out.nc temperature --degrees

using NCDatasets

mutable struct Opts
    cmap::Union{Nothing,Symbol}
    clim::Union{Nothing,Tuple{Float64,Float64}}
    degrees::Bool
    transpose::Bool
end

Opts() = Opts(nothing, nothing, true, false)

function parse_args()
    path = get(ARGS, 1, "")
    varname = get(ARGS, 2, "temperature")
    extras = ARGS[3:end]
    opts = Opts()
    i = 1
    while i <= length(extras)
        a = extras[i]
        if a == "--cmap" && i+1 <= length(extras)
            opts.cmap = Symbol(extras[i+1]); i += 2
        elseif a == "--clim" && i+2 <= length(extras)
            vmin = parse(Float64, extras[i+1]); vmax = parse(Float64, extras[i+2])
            opts.clim = (vmin, vmax); i += 3
        elseif a == "--degrees"
            opts.degrees = true; i += 1
        elseif a == "--radians"
            opts.degrees = false; i += 1
        elseif a == "--transpose"
            # Plot with r on x-axis and theta on y-axis
            opts.transpose = true; i += 1
        else
            error("Unknown or malformed option '$a'. Supported: --cmap NAME --clim MIN MAX --degrees|--radians [--transpose]")
        end
    end
    return path, varname, opts
end

function ensure_backend()
    local backend
    try
        @eval using GLMakie
        backend = :gl
    catch
        try
            @eval using CairoMakie
            backend = :cairo
        catch
            error("No Makie backend found. Install GLMakie or CairoMakie: import Pkg; Pkg.add(\"GLMakie\")")
        end
    end
    return backend
end

function read_coords(ds::NCDataset)
    haskey(ds, "theta") || error("Coordinate 'theta' not found in file")
    haskey(ds, "phi")   || error("Coordinate 'phi' not found in file")
    haskey(ds, "r")     || error("Coordinate 'r' not found in file")
    theta = vec(ds["theta"][:])
    phi   = vec(ds["phi"][:])
    rvals = vec(ds["r"][:])
    return theta, phi, rvals
end

function to_theta_phi_r(data::AbstractArray, theta, phi, rvals)
    sz = size(data)
    # Drop singleton dims
    while any(sz .== 1) && ndims(data) > 3
        data = dropdims(data; dims=findfirst(==(1), size(data)))
        sz = size(data)
    end
    if ndims(data) == 3 && sz == (length(theta), length(phi), length(rvals))
        return Array(data)
    end
    # Try to infer permutation by matching sizes
    dimsizes = collect(sz)
    idxθ = findfirst(==(length(theta)), dimsizes)
    idxφ = findfirst(==(length(phi)),   dimsizes)
    idxr = findfirst(==(length(rvals)), dimsizes)
    if !(idxθ === nothing || idxφ === nothing || idxr === nothing)
        perm = (idxθ, idxφ, idxr)
        return Array(PermutedDimsArray(data, perm))
    else
        error("Variable dims do not match (theta,phi,r). Got size $(size(data)); expected some permutation of ($(length(theta)),$(length(phi)),$(length(rvals))).")
    end
end

function compute_m0(path::AbstractString, varname::AbstractString)
    isfile(path) || error("File not found: $path")
    ds = NCDataset(path)
    try
        haskey(ds, varname) || error("Variable '$varname' not found in $path")
        theta, phi, rvals = read_coords(ds)
        raw = ds[varname][:]
        arr = to_theta_phi_r(raw, theta, phi, rvals)
        # Average over phi dimension (dim=2)
        m0 = dropdims(mean(arr; dims=2), dims=2)  # (theta, r)
        return theta, rvals, m0
    finally
        close(ds)
    end
end

function main()
    path, varname, opts = parse_args()
    theta, rvals, m0 = compute_m0(path, varname)

    backend = ensure_backend()
    if backend == :gl
        using GLMakie
    else
        using CairoMakie
    end

    # Axes values
    θplot = opts.degrees ? theta .* (180 / pi) : theta
    xlab = opts.degrees ? "θ (deg)" : "θ (rad)"
    ylab = "r"

    # Build heatmap grid
    # m0 is (θ, r). Makie heatmap expects matrices indexed as (x,y) by default.
    H = opts.transpose ? permutedims(m0, (2,1)) : m0
    x = opts.transpose ? rvals : θplot
    y = opts.transpose ? θplot : rvals
    xlabel = opts.transpose ? "r" : xlab
    ylabel = opts.transpose ? xlab : "r"

    fig = Figure(resolution=(900, 650))
    ax = Axis(fig[1,1], xlabel=xlabel, ylabel=ylabel, title="$(varname) m=0 (zonal mean)")
    hm_kwargs = (;)
    if opts.cmap !== nothing
        hm_kwargs = merge(hm_kwargs, (; colormap = opts.cmap))
    end
    if opts.clim !== nothing
        hm_kwargs = merge(hm_kwargs, (; colorrange = opts.clim))
    end
    h = heatmap!(ax, x, y, H; interpolate=false, hm_kwargs...)
    cb = Colorbar(fig[1,2], h, label=varname)
    fig

    outpng = replace(basename(path), r"\.nc$" => "") * "_$(varname)_m0_rtheta.png"
    save(outpng, fig)
    @info "Saved" outpng
    display(fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
