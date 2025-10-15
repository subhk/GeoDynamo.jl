#!/usr/bin/env julia

"""
Plot radial magnetic field in Hammer projection from a merged NetCDF file.

Usage examples:
  # Radial magnetic field at r=0.8 in Hammer projection
  julia --project=. scripts/plot_hammer_magnetic.jl ./output/combined_time_1p000000.nc \
        --r=0.8 --out=./hammer_br_r0p8.png

  # Radial magnetic field at surface (r=1.0) with custom colormap
  julia --project=. scripts/plot_hammer_magnetic.jl ./output/combined_time_1p000000.nc \
        --r=1.0 --out=./hammer_br_surface.png --cmap=RdBu_r

Options:
  --r=<float>          Radius at which to evaluate (required)
  --out=<file>         Output image filename (PNG recommended)
  --cmap=<name>        Colormap name (default: RdBu_r for magnetic field)
  --title=<string>     Plot title
  --levels=<int>       Number of contour levels (default: 20)
"""

using NetCDF
using Printf

try
    using Plots
    using PlotlyJS
catch
    @error "Plots.jl and PlotlyJS not found. Install with: julia --project=. -e 'using Pkg; Pkg.add([\"Plots\", \"PlotlyJS\"])'"
    exit(1)
end

using SHTnsKit

function usage()
    println("Usage: plot_hammer_magnetic.jl <merged.nc> --r=<val> [--out=file.png] [--cmap=name] [--title=str] [--levels=int]")
end

function parse_args(args)
    isempty(args) && (usage(); error("missing arguments"))
    infile = abspath(args[1])
    r0 = nothing
    outfile = ""
    cmap = :RdBu_r
    title = ""
    levels = 20
    for a in args[2:end]
        if startswith(a, "--r="); r0 = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--out="); outfile = split(a, "=", limit=2)[2]
        elseif startswith(a, "--cmap="); cmap = Symbol(split(a, "=", limit=2)[2])
        elseif startswith(a, "--title="); title = split(a, "=", limit=2)[2]
        elseif startswith(a, "--levels="); levels = parse(Int, split(a, "=", limit=2)[2])
        else
            @warn "Unknown arg $a ignored"
        end
    end
    r0 === nothing && (usage(); error("--r is required"))
    return infile, r0, outfile, cmap, title, levels
end

readvar(nc, name) = (NetCDF.varid(nc, name) == -1 ? nothing : NetCDF.readvar(nc, name))

function build_cfg_from_nc(nc)
    lvals = Int.(readvar(nc, "l_values"))
    mvals = Int.(readvar(nc, "m_values"))
    lmax = maximum(lvals); mmax = maximum(mvals)
    nlat = (NetCDF.varid(nc, "theta") != -1) ? length(readvar(nc, "theta")) : (lmax+2)
    nlon = (NetCDF.varid(nc, "phi") != -1) ? length(readvar(nc, "phi")) : max(2*lmax+1, 4)
    cfg = SHTnsKit.create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon, norm=:orthonormal)
    θ = readvar(nc, "theta"); φ = readvar(nc, "phi"); r = readvar(nc, "r")
    return cfg, lvals, mvals, θ, φ, r
end

function synthesize_magnetic(cfg, lvals, mvals, mtor_r, mtor_i, mpol_r, mpol_i, rvec)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlat, nlon = cfg.nlat, cfg.nlon
    nlm, nr = size(mtor_r)
    bt = zeros(Float64, nlat, nlon, nr)
    bp = similar(bt); br = similar(bt)
    T = zeros(ComplexF64, lmax+1, mmax+1)
    S = zeros(ComplexF64, lmax+1, mmax+1)
    
    for k in 1:nr
        fill!(T,0); fill!(S,0)
        for i in 1:nlm
            l=lvals[i]; m=mvals[i]
            if l<=lmax && m<=mmax
                T[l+1,m+1] = complex(mtor_r[i,k], mtor_i[i,k])
                S[l+1,m+1] = complex(mpol_r[i,k], mpol_i[i,k])
            end
        end
        bt[:,:,k], bp[:,:,k] = SHsphtor_to_spat(cfg, S, T; real_output=true)
        rr = (rvec === nothing) ? 1.0 : rvec[k]
        if rr > eps()
            Q = zeros(ComplexF64, lmax+1, mmax+1)
            for i in 1:nlm
                l=lvals[i]; m=mvals[i]
                if l<=lmax && m<=mmax
                    Q[l+1,m+1] = S[l+1,m+1] * (l*(l+1)/rr)
                end
            end
            br[:,:,k] = synthesis(cfg, Q; real_output=true)
        end
    end
    return br, bt, bp
end

function extract_shell_r(field::Array{<:Real,3}, r, r0)
    nlat, nlon, nr = size(field)
    if r === nothing
        r = collect(range(0, stop=1, length=nr))
    end
    # Find nearest radii and linearly interpolate
    k2 = searchsortedfirst(r, r0)
    k1 = max(1, k2-1); k2 = min(nr, k2)
    if k1 == k2
        return field[:,:,k1]
    else
        w2 = (r0 - r[k1]) / (r[k2] - r[k1])
        w1 = 1.0 - w2
        return w1 .* field[:,:,k1] .+ w2 .* field[:,:,k2]
    end
end

function hammer_projection(θ, φ)
    """
    Convert spherical coordinates to Hammer projection
    θ: colatitude (0 to π)
    φ: longitude (0 to 2π)
    Returns (x, y) in Hammer projection
    """
    # Convert to latitude for Hammer projection
    lat = π/2 .- θ
    lon = φ .- π  # Center at longitude 0
    
    # Handle longitude wrapping
    lon = mod.(lon .+ π, 2π) .- π
    
    # Hammer projection formulas
    sqrt2 = sqrt(2.0)
    denom = sqrt.(1.0 .+ cos.(lat) .* cos.(lon ./ 2.0))
    x = 2 * sqrt2 * cos.(lat) .* sin.(lon ./ 2.0) ./ denom
    y = sqrt2 * sin.(lat) ./ denom
    
    return x, y
end

function create_hammer_grid(nlat, nlon)
    """
    Create coordinate grids for Hammer projection
    """
    θ_vec = range(0, stop=π, length=nlat)
    φ_vec = range(0, stop=2π, length=nlon)
    
    θ_grid = repeat(θ_vec, 1, nlon)
    φ_grid = repeat(φ_vec', nlat, 1)
    
    x_grid, y_grid = hammer_projection(θ_grid, φ_grid)
    
    return x_grid, y_grid, θ_grid, φ_grid
end

function plot_hammer_contour(field2d, θ, φ; cmap=:RdBu_r, title="", levels=20)
    """
    Create Hammer projection contour plot
    """
    nlat, nlon = size(field2d)
    
    # Create coordinate grids
    if θ === nothing
        θ = range(0, stop=π, length=nlat) |> collect
    end
    if φ === nothing
        φ = range(0, stop=2π, length=nlon) |> collect
    end
    
    # Create meshgrids
    θ_grid = repeat(θ, 1, nlon)
    φ_grid = repeat(φ', nlat, 1)
    
    # Convert to Hammer projection
    x_grid, y_grid = hammer_projection(θ_grid, φ_grid)
    
    # Create contour plot
    contour(x_grid, y_grid, field2d, 
           levels=levels,
           fill=true,
           c=cmap,
           aspect_ratio=:equal,
           title=title,
           xlabel="",
           ylabel="",
           colorbar=true,
           size=(800, 400),
           xlims=(-2*sqrt(2), 2*sqrt(2)),
           ylims=(-sqrt(2), sqrt(2)))
    
    # Add grid lines for reference
    # Meridians (longitude lines)
    for lon in -150:30:180
        lon_rad = deg2rad(lon)
        lat_line = range(-90, 90, length=100)
        lat_rad = deg2rad.(lat_line)
        θ_line = π/2 .- lat_rad
        φ_line = fill(lon_rad + π, length(lat_line))  # Adjust for projection center
        φ_line = mod.(φ_line, 2π)
        
        x_line, y_line = hammer_projection(θ_line, φ_line)
        plot!(x_line, y_line, color=:gray, alpha=0.3, linewidth=0.5, label="")
    end
    
    # Parallels (latitude lines)
    for lat in -60:30:60
        lat_rad = deg2rad(lat)
        lon_line = range(-180, 180, length=200)
        lon_rad = deg2rad.(lon_line)
        θ_line = fill(π/2 - lat_rad, length(lon_line))
        φ_line = lon_rad .+ π  # Adjust for projection center
        φ_line = mod.(φ_line, 2π)
        
        x_line, y_line = hammer_projection(θ_line, φ_line)
        plot!(x_line, y_line, color=:gray, alpha=0.3, linewidth=0.5, label="")
    end
end

function main()
    infile, r0, outfile, cmap, ttl, levels = parse_args(copy(ARGS))
    nc = NetCDF.open(infile, NC_NOWRITE)
    
    try
        cfg, lvals, mvals, θ, φ, r = build_cfg_from_nc(nc)
        nlat, nlon = cfg.nlat, cfg.nlon
        
        # Get magnetic field data
        mtor_r = readvar(nc, "magnetic_toroidal_real")
        mtor_i = readvar(nc, "magnetic_toroidal_imag")
        mpol_r = readvar(nc, "magnetic_poloidal_real") 
        mpol_i = readvar(nc, "magnetic_poloidal_imag")
        
        (mtor_r===nothing || mpol_r===nothing) && error("Magnetic spectral variables not found")
        
        # Synthesize magnetic field
        br, bt, bp = synthesize_magnetic(cfg, lvals, mvals, 
                                        Float64.(mtor_r), Float64.(mtor_i), 
                                        Float64.(mpol_r), Float64.(mpol_i), r)
        
        # Extract radial component at specified radius
        field2d = extract_shell_r(br, r, r0)
        
        # Prepare axes
        if θ === nothing
            θ = acos.(cfg.x)
        end
        if φ === nothing
            φ = range(0, stop=2π, length=nlon+1)[1:end-1] |> collect
        end
        
        title_str = (ttl=="" ? "Radial Magnetic Field B_r at r=$(r0)" : ttl)
        
        # Create Hammer projection plot
        plot_hammer_contour(field2d, θ, φ; cmap=cmap, title=title_str, levels=levels)
        
        if outfile != ""
            savefig(outfile)
            println("Saved Hammer projection plot to $(outfile)")
        else
            gui()
        end
        
    finally
        NetCDF.close(nc)
    end
end

isinteractive() || main()
