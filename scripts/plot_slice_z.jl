#!/usr/bin/env julia

"""
Plot a 2D slice at constant z of a velocity, magnetic component, or scalar field
from a merged NetCDF file (Geodynamo output).

Usage examples:
  # Temperature (scalar) at z=0.2, lat-lon heatmap
  julia --project=. scripts/plot_slice_z.jl ./output/combined_time_1p000000.nc \
        --quantity=temperature --z=0.2 --out=./slice_temp.png

  # Velocity z-component at z=0.1 (reconstructs from spectral), xy scatter map
  julia --project=. scripts/plot_slice_z.jl ./output/combined_time_1p000000.nc \
        --quantity=velocity_z --z=0.1 --plane=xy --out=./slice_uz.png

Quantities:
  Scalars: temperature, composition
  Velocity: velocity_r, velocity_theta, velocity_phi, velocity_z
  Magnetic: magnetic_r, magnetic_theta, magnetic_phi, magnetic_z

Options:
  --z=<float>        Constant z plane to extract (in same units as r)
  --quantity=<name>  Field to plot (see above)
  --out=<file>       Output image filename (PNG recommended)
  --plane=lonlat|xy  Plot in (θ,φ) heatmap (lonlat) or Cartesian (x,y) scatter (xy). Default: lonlat
  --cmap=<name>      Colormap name (default: viridis)
  --title=<string>   Plot title
"""

using NetCDF
using Printf

try
    using Plots
catch
    @error "Plots.jl not found. Please add with: julia --project=. -e 'using Pkg; Pkg.add("Plots")'"
    exit(1)
end

using SHTnsKit

function usage()
    println("Usage: plot_slice_z.jl <merged.nc> --quantity=<name> --z=<val> [--out=file.png] [--plane=lonlat|xy] [--cmap=name] [--title=str]")
end

function parse_args(args)
    isempty(args) && (usage(); error("missing arguments"))
    infile = abspath(args[1])
    quantity = ""
    z0 = nothing
    outfile = ""
    plane = "lonlat"
    cmap = :viridis
    title = ""
    for a in args[2:end]
        if startswith(a, "--quantity="); quantity = split(a, "=", limit=2)[2]
        elseif startswith(a, "--z="); z0 = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--out="); outfile = split(a, "=", limit=2)[2]
        elseif startswith(a, "--plane="); plane = lowercase(split(a, "=", limit=2)[2])
        elseif startswith(a, "--cmap="); cmap = Symbol(split(a, "=", limit=2)[2])
        elseif startswith(a, "--title="); title = split(a, "=", limit=2)[2]
        else
            @warn "Unknown arg $a ignored"
        end
    end
    (quantity == "" || z0 === nothing) && (usage(); error("--quantity and --z are required"))
    return infile, quantity, z0, outfile, plane, cmap, title
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

function synthesize_velocity(cfg, lvals, mvals, vtor_r, vtor_i, vpol_r, vpol_i, rvec)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlat, nlon = cfg.nlat, cfg.nlon

    nlm, nr = size(vtor_r)
    vt = zeros(Float64, nlat, nlon, nr)
    vp = similar(vt); vr = similar(vt)
    T = zeros(ComplexF64, lmax+1, mmax+1)
    S = zeros(ComplexF64, lmax+1, mmax+1)

    for k in 1:nr
        fill!(T,0); fill!(S,0)
        for i in 1:nlm
            l=lvals[i]; m=mvals[i]
            if l<=lmax && m<=mmax
                T[l+1,m+1] = complex(vtor_r[i,k], vtor_i[i,k])
                S[l+1,m+1] = complex(vpol_r[i,k], vpol_i[i,k])
            end
        end

        vt[:,:,k], vp[:,:,k] = SHsphtor_to_spat(cfg, S, T; real_output=true)
        rr = (rvec === nothing) ? 1.0 : rvec[k]

        if rr > eps()
            Q = zeros(ComplexF64, lmax+1, mmax+1)
            for i in 1:nlm
                l=lvals[i]; m=mvals[i]
                if l<=lmax && m<=mmax
                    Q[l+1,m+1] = S[l+1,m+1] * (l*(l+1)/rr)
                end
            end
            vr[:,:,k] = synthesis(cfg, Q; real_output=true)
        end
    end
    return vr, vt, vp
end

function synthesize_magnetic(cfg, lvals, mvals, mtor_r, mtor_i, mpol_r, mpol_i, rvec)
    return synthesize_velocity(cfg, lvals, mvals, mtor_r, mtor_i, mpol_r, mpol_i, rvec)
end

function extract_slice_z(field::Array{<:Real,3}, θ, φ, r, z0)
    nlat, nlon, nr = size(field)
    if θ === nothing
        θ = range(0, stop=π, length=nlat) |> collect
    end
    if φ === nothing
        φ = range(0, stop=2π, length=nlon+1)[1:end-1] |> collect
    end
    if r === nothing
        r = collect(range(0, stop=1, length=nr))
    end
    slice = fill(NaN, nlat, nlon)
    for i in 1:nlat
        cθ = cos(θ[i])
        if abs(cθ) < 1e-12
            continue
        end
        r_target = z0 / cθ
        
        # find nearest two radii for interpolation
        if r_target < minimum(r) || r_target > maximum(r)
            continue
        end

        # locate bracketing indices
        k2 = searchsortedfirst(r, r_target)
        k1 = max(1, k2-1); k2 = min(nr, k2)

        if k1 == k2
            w2 = 0.0; w1 = 1.0
        else
            w2 = (r_target - r[k1]) / (r[k2] - r[k1])
            w1 = 1.0 - w2
        end
        for j in 1:nlon
            slice[i,j] = w1*field[i,j,k1] + w2*field[i,j,k2]
        end

    end
    return slice, θ, φ
end

function main()
    infile, quantity, z0, outfile, plane, cmap, ttl = parse_args(copy(ARGS))
    nc = NetCDF.open(infile, NC_NOWRITE)
    try
        cfg, lvals, mvals, θ, φ, r = build_cfg_from_nc(nc)
        nlat, nlon = cfg.nlat, cfg.nlon
        # Decide source
        field3d = nothing
        if quantity in ("temperature","composition")
            var = readvar(nc, quantity)
            var === nothing && error("Variable $quantity not found")
            field3d = Float64.(var)
            size(field3d,1)==nlat || (field3d = permutedims(field3d,(1,2,3)))
        elseif startswith(quantity, "velocity")
            vtor_r = readvar(nc, "velocity_toroidal_real"); vtor_i = readvar(nc, "velocity_toroidal_imag")
            vpol_r = readvar(nc, "velocity_poloidal_real"); vpol_i = readvar(nc, "velocity_poloidal_imag")
            (vtor_r===nothing || vpol_r===nothing) && error("Velocity spectral variables not found")

            vr, vt, vp = synthesize_velocity(cfg, lvals, mvals, Float64.(vtor_r), 
                                        Float64.(vtor_i), Float64.(vpol_r), Float64.(vpol_i), r)

            if quantity == "velocity_r"; field3d = vr
            elseif quantity == "velocity_theta"; field3d = vt
            elseif quantity == "velocity_phi"; field3d = vp
            elseif quantity == "velocity_z"
                θv = θ === nothing ? acos.(cfg.x) : θ
                cθ = reshape(cos.(θv), nlat,1,1)
                sθ = reshape(sin.(θv), nlat,1,1)
                field3d = vr .* cθ .- vt .* sθ
            else
                error("Unknown velocity component $quantity")
            end
        elseif startswith(quantity, "magnetic")
            mtor_r = readvar(nc, "magnetic_toroidal_real"); mtor_i = readvar(nc, "magnetic_toroidal_imag")
            mpol_r = readvar(nc, "magnetic_poloidal_real"); mpol_i = readvar(nc, "magnetic_poloidal_imag")
            (mtor_r===nothing || mpol_r===nothing) && error("Magnetic spectral variables not found")

            br, bt, bp = synthesize_magnetic(cfg, lvals, mvals, Float64.(mtor_r), 
                                        Float64.(mtor_i), Float64.(mpol_r), Float64.(mpol_i), r)

            if quantity == "magnetic_r"; field3d = br
            elseif quantity == "magnetic_theta"; field3d = bt
            elseif quantity == "magnetic_phi"; field3d = bp
            elseif quantity == "magnetic_z"
                θv = θ === nothing ? acos.(cfg.x) : θ
                cθ = reshape(cos.(θv), nlat,1,1)
                sθ = reshape(sin.(θv), nlat,1,1)
                field3d = br .* cθ .- bt .* sθ
            else
                error("Unknown magnetic component $quantity")
            end
        else
            error("Unknown quantity $quantity")
        end

        slice, θv, φv = extract_slice_z(field3d, θ, φ, r, z0)
        # Plot
        if plane == "lonlat"
            heatmap(φv, θv, slice'; aspect_ratio=1, c=Symbol(cmap), xlabel="φ", 
                ylabel="θ", title=ttl=="" ? "$(quantity) at z=$(z0)" : ttl, colorbar=true)
        else
            # xy scatter (approximate): x=r sinθ cosφ, y=r sinθ sinφ at interpolated r
            xs = Float64[]; ys = Float64[]; vs = Float64[]
            for i in 1:length(θv), j in 1:length(φv)
                val = slice[i,j]
                if !isnan(val)
                    r_target = z0 / cos(θv[i])
                    x = r_target * sin(θv[i]) * cos(φv[j])
                    y = r_target * sin(θv[i]) * sin(φv[j])
                    push!(xs,x); push!(ys,y); push!(vs,val)
                end
            end
            scatter(xs, ys, marker_z=vs, markersize=3, c=Symbol(cmap), xlabel="x", 
                    ylabel="y", title=ttl=="" ? "$(quantity) at z=$(z0)" : ttl, colorbar=true, markerstrokewidth=0)
        end
        if outfile != ""
            savefig(outfile)
            println("Saved plot to $(outfile)")
        else
            gui()
        end
    finally
        NetCDF.close(nc)
    end
end

isinteractive() || main()
