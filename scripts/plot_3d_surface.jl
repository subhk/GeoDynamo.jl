#!/usr/bin/env julia

# 3D surface rendering and export utilities for spherical fields using Makie.
#
# - Reads a variable [theta,phi,r] from NetCDF outputs produced by this repo
# - Renders spherical surfaces (inner/outer/any radius) colored by the field
# - Exports surfaces to legacy VTK (.vtk) and XML VTK PolyData (.vtp)
# - Exports volumes to XML VTK StructuredGrid (.vts) for ParaView iso-surfaces
# - Animates across directories of NetCDF files with time filtering
#
# Usage examples:
#   # Temperature outer surface
#   julia --project=. scripts/plot_3d_surface.jl out.nc temperature outer
#   # Composition at r-index 10
#   julia --project=. scripts/plot_3d_surface.jl out.nc composition 10
#   # Any variable at nearest r-value 0.95
#   julia --project=. scripts/plot_3d_surface.jl out.nc myvar 0.95
#   # Inner and outer surfaces side-by-side
#   julia --project=. scripts/plot_3d_surface.jl out.nc temperature both
#   # Overlay mode with transparency
#   julia --project=. scripts/plot_3d_surface.jl out.nc temperature both --overlay --alpha 0.6
#   # Custom colormap and fixed limits
#   julia --project=. scripts/plot_3d_surface.jl out.nc temperature outer --cmap magma --clim 200 350
#   # Animate across a directory (sorted by time)
#   julia --project=. scripts/plot_3d_surface.jl outdir temperature outer --animate --out movie.mp4 --fps 20 --tmin 1.0 --tmax 5.0
#   # Export legacy VTK PolyData (.vtk) surfaces
#   julia --project=. scripts/plot_3d_surface.jl out.nc temperature both --vtk vtk_out
#   # Export XML VTK StructuredGrid (.vts) volume for ParaView iso-surfaces
#   julia --project=. scripts/plot_3d_surface.jl out.nc temperature outer --vts volume.vts
#   # Export XML VTK PolyData (.vtp) surfaces (requires WriteVTK)
#   julia --project=. scripts/plot_3d_surface.jl out.nc temperature both --vtp vtp_out
#   # Export per-frame VTK during animation
#   julia --project=. scripts/plot_3d_surface.jl outdir temperature outer --animate --vtkdir vtk_series --vtpdir vtp_series --vtsdir vts_series
#
# Notes:
# - Plotting requires GLMakie (interactive) or CairoMakie (PNG). Install:
#     import Pkg; Pkg.add("GLMakie")   # or Pkg.add("CairoMakie")
# - XML VTK export (.vts/.vtp) requires WriteVTK. Install:
#     import Pkg; Pkg.add("WriteVTK")

using NCDatasets
using Printf
const HAVE_WRITEVTK = try
    @eval using WriteVTK
    true
catch
    false
end

function parse_args()
    path = get(ARGS, 1, "")
    varname = get(ARGS, 2, "temperature")
    rsel = get(ARGS, 3, "outer")  # "outer" | "inner" | "both" | Int index | Float radius
    return path, varname, rsel
end

mutable struct PlotOpts
    overlay::Bool
    cmap::Union{Nothing,Symbol}
    clim::Union{Nothing,Tuple{Float64,Float64}}
    alpha::Float64
    animate::Bool
    dir::Union{Nothing,String}
    outpath::Union{Nothing,String}
    fps::Int
    tmin::Union{Nothing,Float64}
    tmax::Union{Nothing,Float64}
    vtkpath::Union{Nothing,String}
    vtkdir::Union{Nothing,String}
    vts::Union{Nothing,String}
    vtsdir::Union{Nothing,String}
    vtp::Union{Nothing,String}
    vtpdir::Union{Nothing,String}
end

function default_opts()
    return PlotOpts(false, nothing, nothing, 0.6, false, nothing, nothing, 20,
                    nothing, nothing, nothing, nothing,
                    nothing, nothing, nothing, nothing)
end

function parse_flags!(opts::PlotOpts, extras::Vector{String})
    i = 1
    while i <= length(extras)
        arg = extras[i]
        if arg == "--overlay"
            opts.overlay = true; i += 1
        elseif arg == "--cmap" && i+1 <= length(extras)
            opts.cmap = Symbol(extras[i+1]); i += 2
        elseif arg == "--clim" && i+2 <= length(extras)
            vmin = parse(Float64, extras[i+1]); vmax = parse(Float64, extras[i+2]);
            opts.clim = (vmin, vmax); i += 3
        elseif arg == "--alpha" && i+1 <= length(extras)
            opts.alpha = parse(Float64, extras[i+1]); i += 2
        elseif arg == "--animate"
            opts.animate = true; i += 1
        elseif arg == "--dir" && i+1 <= length(extras)
            opts.dir = extras[i+1]; i += 2
        elseif arg == "--out" && i+1 <= length(extras)
            opts.outpath = extras[i+1]; i += 2
        elseif arg == "--fps" && i+1 <= length(extras)
            opts.fps = parse(Int, extras[i+1]); i += 2
        elseif arg == "--tmin" && i+1 <= length(extras)
            opts.tmin = parse(Float64, extras[i+1]); i += 2
        elseif arg == "--tmax" && i+1 <= length(extras)
            opts.tmax = parse(Float64, extras[i+1]); i += 2
        elseif arg == "--vtk" && i+1 <= length(extras)
            opts.vtkpath = extras[i+1]; i += 2
        elseif arg == "--vtkdir" && i+1 <= length(extras)
            opts.vtkdir = extras[i+1]; i += 2
        elseif arg == "--vts" && i+1 <= length(extras)
            opts.vts = extras[i+1]; i += 2
        elseif arg == "--vtsdir" && i+1 <= length(extras)
            opts.vtsdir = extras[i+1]; i += 2
        elseif arg == "--vtp" && i+1 <= length(extras)
            opts.vtp = extras[i+1]; i += 2
        elseif arg == "--vtpdir" && i+1 <= length(extras)
            opts.vtpdir = extras[i+1]; i += 2
        else
            error("Unknown or malformed option '$arg'. Supported: --overlay --cmap NAME --clim MIN MAX --alpha A --animate --dir DIR --out PATH --fps N --tmin T --tmax T --vtk FILE --vtkdir DIR --vts FILE --vtsdir DIR --vtp FILE --vtpdir DIR")
        end
    end
    return opts
end

function pick_radius_index(rvals::AbstractVector{<:Real}, rsel)
    if rsel isa AbstractString
        s = lowercase(rsel)
        if s == "outer"
            return length(rvals)
        elseif s == "inner"
            return 1
        elseif s == "both"
            return (1, length(rvals))
        else
            try
                rv = parse(Float64, s)
                return findmin(abs.(rvals .- rv))[2]
            catch
                error("Unrecognized radius selector: $rsel. Use 'outer', 'inner', index, or radius value.")
            end
        end
    elseif rsel isa Integer
        return Int(rsel)
    elseif rsel isa Real
        rv = float(rsel); return findmin(abs.(rvals .- rv))[2]
    else
        error("Unsupported radius selector type: $(typeof(rsel))")
    end
end

function read_field(path::AbstractString, varname::AbstractString)
    isempty(path) && error("No NetCDF path provided. Pass the file path as the first argument.")
    isfile(path) || error("File not found: $path")
    ds = NCDataset(path)
    try
        haskey(ds, varname) || error("Variable '$varname' not found in $path")
        var = ds[varname]
        haskey(ds, "theta") || error("Coordinate 'theta' not found in $path")
        haskey(ds, "phi")   || error("Coordinate 'phi' not found in $path")
        haskey(ds, "r")     || error("Coordinate 'r' not found in $path")
        theta = vec(ds["theta"][:]); phi = vec(ds["phi"][:]); rvals = vec(ds["r"][:])
        return (ds=ds, var=var, theta=theta, phi=phi, rvals=rvals)
    catch
        close(ds); rethrow()
    end
end

function sphere_surface_coords(theta::AbstractVector, phi::AbstractVector, r::Real)
    sθ = sin.(theta); cθ = cos.(theta)
    sφ = sin.(phi);   cφ = cos.(phi)
    X = [r * sθ[i] * cφ[j] for i in eachindex(theta), j in eachindex(phi)]
    Y = [r * sθ[i] * sφ[j] for i in eachindex(theta), j in eachindex(phi)]
    Z = [r * cθ[i]        for i in eachindex(theta), j in eachindex(phi)]
    return X, Y, Z
end

function sphere_volume_coords(theta::AbstractVector, phi::AbstractVector, rvals::AbstractVector)
    nθ, nφ, nr = length(theta), length(phi), length(rvals)
    sθ = sin.(theta); cθ = cos.(theta)
    sφ = sin.(phi);   cφ = cos.(phi)
    X = Array{Float32}(undef, nθ, nφ, nr)
    Y = similar(X); Z = similar(X)
    for i in 1:nθ, j in 1:nφ, k in 1:nr
        r = rvals[k]
        X[i,j,k] = Float32(r * sθ[i] * cφ[j])
        Y[i,j,k] = Float32(r * sθ[i] * sφ[j])
        Z[i,j,k] = Float32(r * cθ[i])
    end
    return X, Y, Z
end

function field_at_radius(var, r_index::Int)
    T = Array(var[:, :, r_index])
    return T
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

function time_of_file(path::AbstractString)
    ds = NCDataset(path)
    try
        if haskey(ds, "time")
            t = ds["time"][:]; return length(t) == 1 ? float(t[1]) : float(first(t))
        elseif haskey(ds.attrib, "current_time")
            return float(ds.attrib["current_time"])
        else
            m = match(r"_time_([0-9p]+)", basename(path))
            if m !== nothing
                s = replace(m.captures[1], 'p' => '.')
                try return parse(Float64, s) catch end
            end
            return NaN
        end
    finally
        close(ds)
    end
end

function list_nc_files_for_animation(dir::AbstractString)
    isdir(dir) || error("Not a directory: $dir")
    files = filter(f -> endswith(f, ".nc"), joinpath.(dir, readdir(dir)))
    rank0 = filter(f -> occursin("rank_0000", f), files)
    files = isempty(rank0) ? files : rank0
    times = [(f, time_of_file(f)) for f in files]
    sorted = sort(times; by = x -> isfinite(x[2]) ? x[2] : Inf)
    return first.(sorted), last.(sorted)
end

function build_sphere_mesh(theta::AbstractVector, phi::AbstractVector, r::Real)
    nθ = length(theta); nφ = length(phi)
    sθ = sin.(theta); cθ = cos.(theta)
    sφ = sin.(phi);   cφ = cos.(phi)
    N = nθ * nφ
    pts = Array{Float32}(undef, N, 3)
    idx = (i,j) -> (i-1)*nφ + j
    for i in 1:nθ, j in 1:nφ
        k = idx(i,j)
        pts[k,1] = Float32(r * sθ[i] * cφ[j])
        pts[k,2] = Float32(r * sθ[i] * sφ[j])
        pts[k,3] = Float32(r * cθ[i])
    end
    faces = Vector{NTuple{3,Int}}()
    sizehint!(faces, 2*(nθ-1)*nφ)
    for i in 1:(nθ-1), j in 1:nφ
        jn = (j % nφ) + 1
        v1 = idx(i, j); v2 = idx(i, jn); v3 = idx(i+1, j); v4 = idx(i+1, jn)
        push!(faces, (v1, v2, v3)); push!(faces, (v2, v4, v3))
    end
    return pts, faces
end

function export_vtk_surface(path::AbstractString, theta, phi, r::Real, values::AbstractMatrix, name::AbstractString)
    pts, faces = build_sphere_mesh(theta, phi, r)
    N = size(pts,1); M = length(faces)
    length(vec(values)) == N || error("VTK export: values size mismatch with points")
    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, "GeoDynamo surface $name")
        println(io, "ASCII")
        println(io, "DATASET POLYDATA")
        println(io, "POINTS $N float")
        for k in 1:N
            println(io, "$(pts[k,1]) $(pts[k,2]) $(pts[k,3])")
        end
        println(io, "POLYGONS $M $(M*4)")
        for (a,b,c) in faces
            println(io, "3 $(a-1) $(b-1) $(c-1)")
        end
        println(io, "POINT_DATA $N")
        println(io, "SCALARS $name float 1")
        println(io, "LOOKUP_TABLE default")
        for v in vec(values)
            println(io, Float32(v))
        end
    end
end

function export_vts_volume_xml(path::AbstractString, theta, phi, rvals, values::AbstractArray, name::AbstractString)
    HAVE_WRITEVTK || error("WriteVTK.jl not installed. Install with: import Pkg; Pkg.add(\"WriteVTK\")")
    X, Y, Z = sphere_volume_coords(theta, phi, rvals)
    size(values) == size(X) || error("Volume export: values dims must be (theta,phi,r)")
    vtk = WriteVTK.vtk_grid("StructuredGrid", X, Y, Z)
    WriteVTK.vtk_point_data(vtk, name, values)
    WriteVTK.vtk_save(vtk, path)
end

function export_vtp_surface_xml(path::AbstractString, theta, phi, r::Real, values::AbstractMatrix, name::AbstractString)
    HAVE_WRITEVTK || error("WriteVTK.jl not installed. Install with: import Pkg; Pkg.add(\"WriteVTK\")")
    pts, faces = build_sphere_mesh(theta, phi, r)
    # WriteVTK expects 1-based connectivity as Vector{Vector{Int}}
    polys = [collect(f) for f in faces]
    P = Array{Float64}(undef, size(pts,1), 3)
    P[:,1] = Float64.(pts[:,1]); P[:,2] = Float64.(pts[:,2]); P[:,3] = Float64.(pts[:,3])
    vtk = WriteVTK.vtk_grid("PolyData", P; polys=polys)
    WriteVTK.vtk_point_data(vtk, name, vec(values))
    WriteVTK.vtk_save(vtk, path)
end

function main()
    path, varname, rsel_raw = parse_args()
    opts = parse_flags!(default_opts(), ARGS[4:end])

    backend = ensure_backend()
    if backend == :gl
        using GLMakie
    else
        using CairoMakie
    end

    function cmargs(arrs...)
        cm = (;)
        if opts.cmap !== nothing
            cm = merge(cm, (; colormap = opts.cmap))
        end
        if opts.clim !== nothing
            cm = merge(cm, (; colorrange = opts.clim))
        elseif length(arrs) > 1
            vmin = minimum(minimum.(arrs)); vmax = maximum(maximum.(arrs))
            cm = merge(cm, (; colorrange = (vmin, vmax)))
        end
        return cm
    end

    function render_file_once(ncpath::AbstractString)
        data = read_field(ncpath, varname)
        rsel = pick_radius_index(data.rvals, rsel_raw)
        if rsel isa Tuple{Int,Int}
            r_in, r_out = rsel
            Tin  = field_at_radius(data.var, r_in)
            Tout = field_at_radius(data.var, r_out)
            Xin, Yin, Zin = sphere_surface_coords(data.theta, data.phi, data.rvals[r_in])
            Xo,  Yo,  Zo  = sphere_surface_coords(data.theta, data.phi, data.rvals[r_out])
            args = cmargs(Tin, Tout)
            if opts.overlay
                fig = Figure(resolution=(900, 700))
                ax = Axis3(fig[1,1], aspect=:data, title="$(varname) inner+outer")
                m1 = surface!(ax, Xin, Yin, Zin; color=Tin, shading=true, transparency=true; args...)
                m2 = surface!(ax, Xo, Yo, Zo;   color=Tout, shading=true, transparency=true; args...)
                try
                    m1[:alpha][] = opts.alpha; m2[:alpha][] = opts.alpha
                catch; @warn "Alpha not supported by backend; overlay may be opaque" end
                Colorbar(fig[1,2], m2, label=varname)
                ax.azimuth[] = 30; ax.elevation[] = 20
                outpng = replace(basename(ncpath), r"\.nc$" => "") * "_$(varname)_overlay.png"
                save(outpng, fig); display(fig); @info "Saved" outpng
                if opts.vtkpath !== nothing
                    base = opts.vtkpath
                    if endswith(lowercase(base), ".vtk")
                        innerpath = replace(base, r"\.vtk$" => "_inner.vtk")
                        outerpath = replace(base, r"\.vtk$" => "_outer.vtk")
                    else
                        mkpath(base; exist_ok=true)
                        root = replace(basename(ncpath), r"\.nc$" => "")
                        innerpath = joinpath(base, "$(root)_$(varname)_inner.vtk")
                        outerpath = joinpath(base, "$(root)_$(varname)_outer.vtk")
                    end
                    export_vtk_surface(innerpath, data.theta, data.phi, data.rvals[r_in], Tin, varname)
                    export_vtk_surface(outerpath, data.theta, data.phi, data.rvals[r_out], Tout, varname)
                    @info "Exported VTK" inner=innerpath outer=outerpath
                end
                if opts.vtp !== nothing
                    base = opts.vtp
                    if endswith(lowercase(base), ".vtp")
                        innerxml = replace(base, r"\.vtp$" => "_inner.vtp")
                        outerxml = replace(base, r"\.vtp$" => "_outer.vtp")
                    else
                        mkpath(base; exist_ok=true)
                        root = replace(basename(ncpath), r"\.nc$" => "")
                        innerxml = joinpath(base, "$(root)_$(varname)_inner.vtp")
                        outerxml = joinpath(base, "$(root)_$(varname)_outer.vtp")
                    end
                    export_vtp_surface_xml(innerxml, data.theta, data.phi, data.rvals[r_in], Tin, varname)
                    export_vtp_surface_xml(outerxml, data.theta, data.phi, data.rvals[r_out], Tout, varname)
                    @info "Exported VTP (XML)" inner=innerxml outer=outerxml
                end
                close(data.ds)
                return fig, ax, (m1, m2), outpng, (:overlay, r_in, r_out, Xin, Yin, Zin, Xo, Yo, Zo)
            else
                fig = Figure(resolution=(1200, 600))
                ax1 = Axis3(fig[1,1], aspect=:data, title="$(varname) inner r=$(round(data.rvals[r_in], sigdigits=4))")
                m1 = surface!(ax1, Xin, Yin, Zin; color=Tin, shading=true; args...)
                Colorbar(fig[1,2], m1, label=varname)
                ax1.azimuth[] = 30; ax1.elevation[] = 20
                ax2 = Axis3(fig[1,3], aspect=:data, title="$(varname) outer r=$(round(data.rvals[r_out], sigdigits=4))")
                m2 = surface!(ax2, Xo, Yo, Zo; color=Tout, shading=true; args...)
                Colorbar(fig[1,4], m2, label=varname)
                ax2.azimuth[] = 30; ax2.elevation[] = 20
                outpng = replace(basename(ncpath), r"\.nc$" => "") * "_$(varname)_both.png"
                save(outpng, fig); display(fig); @info "Saved" outpng
                if opts.vtkpath !== nothing
                    base = opts.vtkpath
                    if endswith(lowercase(base), ".vtk")
                        innerpath = replace(base, r"\.vtk$" => "_inner.vtk")
                        outerpath = replace(base, r"\.vtk$" => "_outer.vtk")
                    else
                        mkpath(base; exist_ok=true)
                        root = replace(basename(ncpath), r"\.nc$" => "")
                        innerpath = joinpath(base, "$(root)_$(varname)_inner.vtk")
                        outerpath = joinpath(base, "$(root)_$(varname)_outer.vtk")
                    end
                    export_vtk_surface(innerpath, data.theta, data.phi, data.rvals[r_in], Tin, varname)
                    export_vtk_surface(outerpath, data.theta, data.phi, data.rvals[r_out], Tout, varname)
                    @info "Exported VTK" inner=innerpath outer=outerpath
                end
                if opts.vtp !== nothing
                    base = opts.vtp
                    if endswith(lowercase(base), ".vtp")
                        innerxml = replace(base, r"\.vtp$" => "_inner.vtp")
                        outerxml = replace(base, r"\.vtp$" => "_outer.vtp")
                    else
                        mkpath(base; exist_ok=true)
                        root = replace(basename(ncpath), r"\.nc$" => "")
                        innerxml = joinpath(base, "$(root)_$(varname)_inner.vtp")
                        outerxml = joinpath(base, "$(root)_$(varname)_outer.vtp")
                    end
                    export_vtp_surface_xml(innerxml, data.theta, data.phi, data.rvals[r_in], Tin, varname)
                    export_vtp_surface_xml(outerxml, data.theta, data.phi, data.rvals[r_out], Tout, varname)
                    @info "Exported VTP (XML)" inner=innerxml outer=outerxml
                end
                close(data.ds)
                return fig, (ax1, ax2), (m1, m2), outpng, (:split, r_in, r_out, Xin, Yin, Zin, Xo, Yo, Zo)
            end
        else
            r_index = rsel
            T = field_at_radius(data.var, r_index)
            X, Y, Z = sphere_surface_coords(data.theta, data.phi, data.rvals[r_index])
            args = cmargs(T)
            fig = Figure(resolution=(900, 700))
            ax = Axis3(fig[1,1], aspect=:data, title="$(varname) at r=$(round(data.rvals[r_index], sigdigits=4))")
            m = surface!(ax, X, Y, Z; color=T, shading=true; args...)
            Colorbar(fig[1,2], m, label=varname)
            ax.azimuth[] = 30; ax.elevation[] = 20
            outpng = replace(basename(ncpath), r"\.nc$" => "") * "_$(varname)_r$(r_index).png"
            save(outpng, fig); display(fig); @info "Saved" outpng
            if opts.vtkpath !== nothing
                base = opts.vtkpath
                vtkfile = endswith(lowercase(base), ".vtk") ? base : joinpath(base, replace(basename(ncpath), r"\.nc$" => "_$(varname)_r$(r_index).vtk"))
                if !endswith(lowercase(base), ".vtk")
                    mkpath(base; exist_ok=true)
                end
                export_vtk_surface(vtkfile, data.theta, data.phi, data.rvals[r_index], T, varname)
                @info "Exported VTK" file=vtkfile
            end
            if opts.vtp !== nothing
                vtpbase = opts.vtp
                vtpfile = endswith(lowercase(vtpbase), ".vtp") ? vtpbase : joinpath(vtpbase, replace(basename(ncpath), r"\.nc$" => "_$(varname)_r$(r_index).vtp"))
                if !endswith(lowercase(vtpbase), ".vtp")
                    mkpath(vtpbase; exist_ok=true)
                end
                export_vtp_surface_xml(vtpfile, data.theta, data.phi, data.rvals[r_index], T, varname)
                @info "Exported VTP (XML)" file=vtpfile
            end
            close(data.ds)
            return fig, ax, m, outpng, (:single, r_index, X, Y, Z)
        end
    end

    if opts.animate
        dir = opts.dir === nothing ? (isdir(path) ? path : dirname(path)) : opts.dir
        files, times = list_nc_files_for_animation(dir)
        if opts.tmin !== nothing || opts.tmax !== nothing
            files = [f for (f,t) in zip(files, times) if (opts.tmin === nothing || t >= opts.tmin) && (opts.tmax === nothing || t <= opts.tmax)]
        end
        isempty(files) && error("No .nc files found in directory (after time filtering): $dir")

        firstfig, axes, plots, _, meta = render_file_once(files[1])
        outmovie = opts.outpath === nothing ? (backend == :gl ? "movie.mp4" : "movie.gif") : opts.outpath
        fps = opts.fps

        function update_for(file)
            data = read_field(file, varname)
            if meta[1] == :single
                r_index = meta[2]
                T = field_at_radius(data.var, r_index)
                if opts.clim === nothing
                    plots[:colorrange][] = (minimum(T), maximum(T))
                end
                plots[:color][] = T
            elseif meta[1] == :overlay
                r_in, r_out = meta[2], meta[3]
                Tin  = field_at_radius(data.var, r_in)
                Tout = field_at_radius(data.var, r_out)
                if opts.clim === nothing
                    vmin = min(minimum(Tin), minimum(Tout))
                    vmax = max(maximum(Tin), maximum(Tout))
                    plots[1][:colorrange][] = (vmin, vmax)
                    plots[2][:colorrange][] = (vmin, vmax)
                end
                plots[1][:color][] = Tin
                plots[2][:color][] = Tout
            elseif meta[1] == :split
                r_in, r_out = meta[2], meta[3]
                Tin  = field_at_radius(data.var, r_in)
                Tout = field_at_radius(data.var, r_out)
                if opts.clim === nothing
                    vmin = min(minimum(Tin), minimum(Tout))
                    vmax = max(maximum(Tin), maximum(Tout))
                    plots[1][:colorrange][] = (vmin, vmax)
                    plots[2][:colorrange][] = (vmin, vmax)
                end
                plots[1][:color][] = Tin
                plots[2][:color][] = Tout
            end
            close(data.ds)
        end

        record(firstfig, outmovie, files; framerate=fps) do f
            for file in files
                update_for(file)
                recordframe!(f)
                if opts.vtkdir !== nothing
                    mkpath(opts.vtkdir; exist_ok=true)
                    t = time_of_file(file)
                    ts = replace(@sprintf("%.6f", t), '.' => 'p')
                    if meta[1] == :single
                        r_index = meta[2]
                        data = read_field(file, varname)
                        T = field_at_radius(data.var, r_index)
                        vtkfile = joinpath(opts.vtkdir, "$(varname)_r$(r_index)_t$(ts).vtk")
                        export_vtk_surface(vtkfile, data.theta, data.phi, data.rvals[r_index], T, varname)
                        close(data.ds)
                    else
                        r_in, r_out = meta[2], meta[3]
                        data = read_field(file, varname)
                        Tin  = field_at_radius(data.var, r_in)
                        Tout = field_at_radius(data.var, r_out)
                        vtkfile_in  = joinpath(opts.vtkdir, "$(varname)_inner_r$(r_in)_t$(ts).vtk")
                        vtkfile_out = joinpath(opts.vtkdir, "$(varname)_outer_r$(r_out)_t$(ts).vtk")
                        export_vtk_surface(vtkfile_in,  data.theta, data.phi, data.rvals[r_in],  Tin,  varname)
                        export_vtk_surface(vtkfile_out, data.theta, data.phi, data.rvals[r_out], Tout, varname)
                        close(data.ds)
                    end
                end
                if opts.vtpdir !== nothing
                    HAVE_WRITEVTK || error("WriteVTK.jl required for --vtpdir. Install with: import Pkg; Pkg.add(\"WriteVTK\")")
                    mkpath(opts.vtpdir; exist_ok=true)
                    t = time_of_file(file); ts = replace(@sprintf("%.6f", t), '.' => 'p')
                    if meta[1] == :single
                        r_index = meta[2]
                        data = read_field(file, varname)
                        T = field_at_radius(data.var, r_index)
                        vtpfile = joinpath(opts.vtpdir, "$(varname)_r$(r_index)_t$(ts).vtp")
                        export_vtp_surface_xml(vtpfile, data.theta, data.phi, data.rvals[r_index], T, varname)
                        close(data.ds)
                    else
                        r_in, r_out = meta[2], meta[3]
                        data = read_field(file, varname)
                        Tin  = field_at_radius(data.var, r_in)
                        Tout = field_at_radius(data.var, r_out)
                        vtp_in  = joinpath(opts.vtpdir, "$(varname)_inner_r$(r_in)_t$(ts).vtp")
                        vtp_out = joinpath(opts.vtpdir, "$(varname)_outer_r$(r_out)_t$(ts).vtp")
                        export_vtp_surface_xml(vtp_in,  data.theta, data.phi, data.rvals[r_in],  Tin,  varname)
                        export_vtp_surface_xml(vtp_out, data.theta, data.phi, data.rvals[r_out], Tout, varname)
                        close(data.ds)
                    end
                end
                if opts.vtsdir !== nothing
                    HAVE_WRITEVTK || error("WriteVTK.jl required for --vtsdir. Install with: import Pkg; Pkg.add(\"WriteVTK\")")
                    mkpath(opts.vtsdir; exist_ok=true)
                    data = read_field(file, varname)
                    vtsfile = joinpath(opts.vtsdir, replace(basename(file), r"\.nc$" => "_$(varname).vts"))
                    export_vts_volume_xml(vtsfile, data.theta, data.phi, data.rvals, Array(data.var[:,:,:]), varname)
                    close(data.ds)
                end
            end
        end
        @info "Saved animation" outmovie frames=length(files) fps=fps
    else
        fig, axes, plots, out, meta = render_file_once(path)
        if opts.vts !== nothing
            HAVE_WRITEVTK || error("WriteVTK.jl required for --vts. Install with: import Pkg; Pkg.add(\"WriteVTK\")")
            data = read_field(path, varname)
            export_vts_volume_xml(opts.vts, data.theta, data.phi, data.rvals, Array(data.var[:,:,:]), varname)
            close(data.ds)
            @info "Exported VTS (XML StructuredGrid)" file=opts.vts
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
