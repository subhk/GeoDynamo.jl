#!/usr/bin/env julia

"""
Plot a quantity on a spherical surface at constant radius r from a merged NetCDF file.

Usage examples:
  # Temperature at r=0.8 (lon-lat heatmap)
  julia --project=. scripts/plot_sphere_r.jl ./output/combined_time_1p000000.nc \
        --quantity=temperature --r=0.8 --out=./sphere_temp_r0p8.png

  # Velocity z-component at r=0.6
  julia --project=. scripts/plot_sphere_r.jl ./output/combined_time_1p000000.nc \
        --quantity=velocity_z --r=0.6 --out=./sphere_uz_r0p6.png

Quantities:
  Scalars: temperature, composition
  Velocity: velocity_r, velocity_theta, velocity_phi, velocity_z
  Magnetic: magnetic_r, magnetic_theta, magnetic_phi, magnetic_z

Options:
  --r=<float>          Radius at which to evaluate
  --quantity=<name>    Field to plot (see above)
  --out=<file>         Output image filename (PNG recommended)
  --cmap=<name>        Colormap name (default: viridis)
  --title=<string>     Plot title
"""

using NetCDF
using Printf

try
    using Plots
catch
    @error "Plots.jl not found. Install with: julia --project=. -e 'using Pkg; Pkg.add("Plots")'"
    exit(1)
end

using SHTnsKit

function usage()
    println("Usage: plot_sphere_r.jl <merged.nc> --quantity=<name> --r=<val> [--out=file.png] [--cmap=name] [--title=str]")
end

function parse_args(args)
    isempty(args) && (usage(); error("missing arguments"))
    infile = abspath(args[1])
    quantity = ""
    r0 = nothing
    outfile = ""
    cmap = :viridis
    title = ""
    for a in args[2:end]
        if startswith(a, "--quantity="); quantity = split(a, "=", limit=2)[2]
        elseif startswith(a, "--r="); r0 = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--out="); outfile = split(a, "=", limit=2)[2]
        elseif startswith(a, "--cmap="); cmap = Symbol(split(a, "=", limit=2)[2])
        elseif startswith(a, "--title="); title = split(a, "=", limit=2)[2]
        else
            @warn "Unknown arg $a ignored"
        end
    end
    (quantity == "" || r0 === nothing) && (usage(); error("--quantity and --r are required"))
    return infile, quantity, r0, outfile, cmap, title
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

function main()
    infile, quantity, r0, outfile, cmap, ttl = parse_args(copy(ARGS))
    nc = NetCDF.open(infile, NC_NOWRITE)
    try
        cfg, lvals, mvals, θ, φ, r = build_cfg_from_nc(nc)
        nlat, nlon = cfg.nlat, cfg.nlon
        # Reconstruct or read field
        field2d = nothing
        if quantity in ("temperature","composition")
            var = readvar(nc, quantity)
            var === nothing && error("Variable $quantity not found")
            field2d = extract_shell_r(Float64.(var), r, r0)
        elseif startswith(quantity, "velocity")
            vtor_r = readvar(nc, "velocity_toroidal_real"); vtor_i = readvar(nc, "velocity_toroidal_imag")
            vpol_r = readvar(nc, "velocity_poloidal_real"); vpol_i = readvar(nc, "velocity_poloidal_imag")
            (vtor_r===nothing || vpol_r===nothing) && error("Velocity spectral variables not found")
            vr, vt, vp = synthesize_velocity(cfg, lvals, mvals, Float64.(vtor_r), Float64.(vtor_i), Float64.(vpol_r), Float64.(vpol_i), r)
            if quantity == "velocity_r"; field2d = extract_shell_r(vr, r, r0)
            elseif quantity == "velocity_theta"; field2d = extract_shell_r(vt, r, r0)
            elseif quantity == "velocity_phi"; field2d = extract_shell_r(vp, r, r0)
            elseif quantity == "velocity_z"
                θv = θ === nothing ? acos.(cfg.x) : θ
                cθ = reshape(cos.(θv), nlat,1,1)
                sθ = reshape(sin.(θv), nlat,1,1)
                uz = vr .* cθ .- vt .* sθ
                field2d = extract_shell_r(uz, r, r0)
            else
                error("Unknown velocity component $quantity")
            end
        elseif startswith(quantity, "magnetic")
            mtor_r = readvar(nc, "magnetic_toroidal_real"); mtor_i = readvar(nc, "magnetic_toroidal_imag")
            mpol_r = readvar(nc, "magnetic_poloidal_real"); mpol_i = readvar(nc, "magnetic_poloidal_imag")
            (mtor_r===nothing || mpol_r===nothing) && error("Magnetic spectral variables not found")
            br, bt, bp = synthesize_magnetic(cfg, lvals, mvals, Float64.(mtor_r), Float64.(mtor_i), Float64.(mpol_r), Float64.(mpol_i), r)
            if quantity == "magnetic_r"; field2d = extract_shell_r(br, r, r0)
            elseif quantity == "magnetic_theta"; field2d = extract_shell_r(bt, r, r0)
            elseif quantity == "magnetic_phi"; field2d = extract_shell_r(bp, r, r0)
            elseif quantity == "magnetic_z"
                θv = θ === nothing ? acos.(cfg.x) : θ
                cθ = reshape(cos.(θv), nlat,1,1)
                sθ = reshape(sin.(θv), nlat,1,1)
                bz = br .* cθ .- bt .* sθ
                field2d = extract_shell_r(bz, r, r0)
            else
                error("Unknown magnetic component $quantity")
            end
        else
            error("Unknown quantity $quantity")
        end

        # Prepare axes
        if θ === nothing
            θ = acos.(cfg.x)
        end
        if φ === nothing
            φ = range(0, stop=2π, length=nlon+1)[1:end-1] |> collect
        end
        title = (ttl=="" ? "$(quantity) at r=$(r0)" : ttl)
        heatmap(φ, θ, field2d'; aspect_ratio=1, c=Symbol(cmap), xlabel="φ", ylabel="θ", title=title, colorbar=true)
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
