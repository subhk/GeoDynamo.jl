#!/usr/bin/env julia

"""
Compute kinetic helicity h = u · (∇×u) from each merged NetCDF file in a time range
and save the time-averaged helicity field to a compressed JLD2 file.

Assumptions:
- Inputs are merged global NetCDF outputs (contain spectral velocity coefficients and coordinates).
- Velocity is provided in spectral toroidal/poloidal form: velocity_toroidal_{real,imag}, velocity_poloidal_{real,imag}

Usage:
  julia --project=. scripts/time_average_helicity.jl <output_dir> --start=<t0> --end=<t1> [--prefix=<name>] [--out=<file.jld2>]

Saves:
  - helicity: time-averaged helicity field (nlat, nlon, nr)
  - counts: number of snapshots accumulated
  - times: times used for averaging
  - metadata: Dict with basic info (geometry if available)
  - coords: Dict with theta, phi, r (if available)
"""

using Printf
using NetCDF
using JLD2
using SHTnsKit
using Geodynamo
using MPI

function usage()
    println("Usage: time_average_helicity.jl <output_dir> --start=<t0> --end=<t1> [--prefix=name] [--out=path.jld2]")
end

function parse_args(args)
    isempty(args) && (usage(); error("missing arguments"))
    outdir = abspath(args[1])
    t0 = nothing
    t1 = nothing
    prefix = "combined_global"
    outpath = ""
    mode = "avg"  # avg | abs | rms
    for a in args[2:end]
        if startswith(a, "--start="); t0 = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--end="); t1 = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--prefix="); prefix = split(a, "=", limit=2)[2]
        elseif startswith(a, "--out="); outpath = split(a, "=", limit=2)[2]
        elseif startswith(a, "--mode=")
            mode = lowercase(split(a, "=", limit=2)[2])
            mode in ("avg","abs","rms") || error("--mode must be one of avg|abs|rms")
        else
            @warn "Unknown argument $a (ignored)"
        end
    end
    (t0 === nothing || t1 === nothing) && (usage(); error("--start and --end required"))
    return outdir, t0, t1, prefix, outpath, mode
end

format_time_str(t::Float64) = replace(@sprintf("%.6f", t), "." => "p")

function scan_times(dir::String, prefix::String)
    files = filter(f -> endswith(f, ".nc") && occursin(prefix, f), readdir(dir))
    times = Float64[]
    pat = r"time_(\d+p\d+)"
    for f in files
        if (m = match(pat, f)) !== nothing
            push!(times, parse(Float64, replace(m.captures[1], "p"=>".")))
        end
    end
    return sort(unique(times))
end

function build_filename(dir::String, prefix::String, t::Float64)
    ts = format_time_str(t)
    cands = [
        joinpath(dir, "$(prefix)_time_$(ts).nc"),
        joinpath(dir, "$(prefix)_output_time_$(ts)_rank_0000.nc"),
        joinpath(dir, "$(prefix)_output_time_$(ts).nc"),
    ]
    for c in cands
        if isfile(c); return c; end
    end
    for f in readdir(dir)
        full = joinpath(dir, f)
        if endswith(f, ".nc") && occursin("time_$(ts)", f) && occursin(prefix, f)
            return full
        end
    end
    return ""
end

read_var(nc, name) = (NetCDF.varid(nc, name) == -1 ? nothing : NetCDF.readvar(nc, name))

function build_sht_from_nc(nc)
    lvals = Int.(read_var(nc, "l_values"))
    mvals = Int.(read_var(nc, "m_values"))
    lmax = maximum(lvals); mmax = maximum(mvals)
    nlat = (NetCDF.varid(nc, "theta") != -1) ? length(read_var(nc, "theta")) : (lmax+2)
    nlon = (NetCDF.varid(nc, "phi") != -1) ? length(read_var(nc, "phi")) : max(2*lmax+1, 4)
    gcfg = Geodynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon)
    θ = read_var(nc, "theta"); φ = read_var(nc, "phi"); r = read_var(nc, "r")
    return gcfg, lvals, mvals, (θ, φ, r)
end

function vector_components(cfg::SHTnsKit.SHTConfig, lvals, mvals,
                           tor_r, tor_i, pol_r, pol_i, rvec)
    lmax=cfg.lmax; mmax=cfg.mmax; nlat=cfg.nlat; nlon=cfg.nlon
    nlm, nr = size(tor_r)
    vt = Array{Float64}(undef, nlat, nlon, nr)
    vp = Array{Float64}(undef, nlat, nlon, nr)
    vr = Array{Float64}(undef, nlat, nlon, nr)
    tor = zeros(ComplexF64, lmax+1, mmax+1)
    pol = zeros(ComplexF64, lmax+1, mmax+1)
    for k in 1:nr
        fill!(tor, 0); fill!(pol, 0)
        for i in 1:nlm
            l=lvals[i]; m=mvals[i]
            if l<=lmax && m<=mmax
                tor[l+1,m+1]=complex(tor_r[i,k], tor_i[i,k])
                pol[l+1,m+1]=complex(pol_r[i,k], pol_i[i,k])
            end
        end
        vt_slice, vp_slice = SHTnsKit.SHsphtor_to_spat(cfg, pol, tor; real_output=true)
        vt[:,:,k] = vt_slice; vp[:,:,k] = vp_slice
        if rvec !== nothing
            rr = rvec[min(k, length(rvec))]
            if rr > eps()
                pol_rad = zeros(ComplexF64, lmax+1, mmax+1)
                for i in 1:nlm
                    l=lvals[i]; m=mvals[i]
                    if l<=lmax && m<=mmax
                        pol_rad[l+1,m+1] = pol[l+1,m+1] * (l*(l+1)/rr)
                    end
                end
                vr[:,:,k] = SHTnsKit.synthesis(cfg, pol_rad; real_output=true)
            else
                vr[:,:,k] .= 0
            end
        else
            vr[:,:,k] .= 0
        end
    end
    return vr, vt, vp
end

function central_diff_φ(A, φ)
    # periodic
    nlat, nlon, nr = size(A)
    dA = similar(A)
    Δφ = φ[2] - φ[1]  # assume uniform
    @inbounds for k in 1:nr, i in 1:nlat
        @views row = A[i, :, k]
        for j in 1:nlon
            jp = (j % nlon) + 1
            jm = (j-2) % nlon + 1
            dA[i,j,k] = (row[jp] - row[jm]) / (2Δφ)
        end
    end
    return dA
end

function central_diff_θ(A, θ)
    nlat, nlon, nr = size(A)
    dA = similar(A)
    @inbounds for k in 1:nr, j in 1:nlon
        for i in 1:nlat
            if i == 1
                dA[i,j,k] = (A[i+1,j,k] - A[i,j,k]) / (θ[i+1] - θ[i])
            elseif i == nlat
                dA[i,j,k] = (A[i,j,k] - A[i-1,j,k]) / (θ[i] - θ[i-1])
            else
                dA[i,j,k] = (A[i+1,j,k] - A[i-1,j,k]) / (θ[i+1] - θ[i-1])
            end
        end
    end
    return dA
end

function central_diff_r(A, r)
    nlat, nlon, nr = size(A)
    dA = similar(A)
    @inbounds for i in 1:nlat, j in 1:nlon
        for k in 1:nr
            if r === nothing || nr == 1
                dA[i,j,k] = 0.0
            elseif k == 1
                dA[i,j,k] = (A[i,j,k+1] - A[i,j,k]) / (r[k+1] - r[k])
            elseif k == nr
                dA[i,j,k] = (A[i,j,k] - A[i,j,k-1]) / (r[k] - r[k-1])
            else
                dA[i,j,k] = (A[i,j,k+1] - A[i,j,k-1]) / (r[k+1] - r[k-1])
            end
        end
    end
    return dA
end

function helicity_field(vr, vt, vp, θ, φ, r)
    nlat, nlon, nr = size(vr)
    # Derivatives
    sinθ = θ === nothing ? ones(nlat) : sin.(θ)
    sinθ_clamped = clamp.(sinθ, 1e-8, Inf)
    dφ_vθ = central_diff_φ(vt, φ)
    dφ_vr = central_diff_φ(vr, φ)
    dθ_vφ = central_diff_θ(vp, θ)
    dθ_vr = central_diff_θ(vr, θ)
    dr_vθ = central_diff_r(vt, r)
    dr_vφ = central_diff_r(vp, r)

    ωr = similar(vr); ωθ = similar(vt); ωφ = similar(vp)
    @inbounds for k in 1:nr
        rr = (r === nothing || k > length(r)) ? 1.0 : r[k]
        for i in 1:nlat, j in 1:nlon
            sθ = sinθ_clamped[i]
            ωr[i,j,k] = (1/(rr*sθ)) * ( (dθ_vφ[i,j,k]*sθ) + vp[i,j,k]*cos(θ[i]) - dφ_vθ[i,j,k] )
            ωθ[i,j,k] = (1/rr) * ( (1/sθ)*dφ_vr[i,j,k] - (dr_vφ[i,j,k] + vp[i,j,k]/rr) )
            ωφ[i,j,k] = (1/rr) * ( dr_vθ[i,j,k] + vt[i,j,k]/rr - dθ_vr[i,j,k] )
        end
    end
    h = vr .* ωr .+ vt .* ωθ .+ vp .* ωφ
    return h
end

function compute_omega_r_from_Tlm(cfg::SHTnsKit.SHTConfig, lvals, mvals, tor_r, tor_i, rvec)
    lmax=cfg.lmax; mmax=cfg.mmax; nlat=cfg.nlat; nlon=cfg.nlon
    nlm, nr = size(tor_r)
    ωr = Array{Float64}(undef, nlat, nlon, nr)
    Tlm = zeros(ComplexF64, lmax+1, mmax+1)
    for k in 1:nr
        fill!(Tlm, 0)
        for i in 1:nlm
            l=lvals[i]; m=mvals[i]
            if l<=lmax && m<=mmax
                Tlm[l+1,m+1]=complex(tor_r[i,k], tor_i[i,k])
            end
        end
        ζ = SHTnsKit.vorticity_grid(cfg, Tlm)  # unit-sphere normal vorticity
        rr = (rvec === nothing || k > length(rvec)) ? 1.0 : rvec[k]
        ωr[:,:,k] = rr > eps() ? (ζ ./ rr) : zeros(nlat, nlon)
    end
    return ωr
end

function volume_average_helicity(h::Array{<:Real,3}, cfg::SHTnsKit.SHTConfig, r::Union{Nothing,Vector{<:Real}})
    nlat, nlon, nr = size(h)
    # Theta weights (include sinθ): cfg.w
    wθ = cfg.w
    Δφ = 2π / nlon
    # Radial weights via trapezoidal on r (if available), else assume unit spacing
    if r === nothing || length(r) != nr
        r = collect(1:nr)
    else
        r = Float64.(r)
    end
    Δr = similar(r)
    for k in 1:nr
        if k == 1
            Δr[k] = (r[min(2,nr)] - r[1])
        elseif k == nr
            Δr[k] = (r[nr] - r[nr-1])
        else
            Δr[k] = 0.5*(r[k+1] - r[k-1])
        end
    end
    vol = 0.0
    integral = 0.0
    @inbounds for k in 1:nr
        rk2 = r[k]^2
        vol += Δφ * Δr[k] * rk2 * sum(wθ)
        # integrate over θ,φ for each k
        # Sum over j: Δφ, over i: wθ[i]
        shell_int = 0.0
        for i in 1:nlat
            wi = wθ[i]
            # sum over φ at fixed (i,k)
            sφ = 0.0
            for j in 1:nlon
                sφ += h[i,j,k]
            end
            shell_int += wi * sφ * Δφ
        end
        integral += rk2 * Δr[k] * shell_int
    end
    return integral / max(vol, eps())
end

function time_average_helicity(dir::String, t0::Float64, t1::Float64, prefix::String; mode::String="avg")
    times = scan_times(dir, prefix)
    sel = [t for t in times if t0 <= t <= t1]
    isempty(sel) && error("No files found in $dir for prefix '$prefix' and time range [$t0, $t1]")

    sum_h = nothing
    sum_hr = nothing
    sum_hθ = nothing
    sum_hφ = nothing
    sum_hz = nothing
    count = 0
    coords = Dict{String,Any}()
    meta = Dict{String,Any}()
    last_cfg = Ref{Union{Nothing,SHTnsKit.SHTConfig}}(nothing)

    for t in sel
        fname = build_filename(dir, prefix, t)
        isempty(fname) && (@warn "No file for time $t"; continue)
        nc = NetCDF.open(fname, NC_NOWRITE)
        try
            vtor_r = read_var(nc, "velocity_toroidal_real"); vtor_i = read_var(nc, "velocity_toroidal_imag")
            vpol_r = read_var(nc, "velocity_poloidal_real"); vpol_i = read_var(nc, "velocity_poloidal_imag")
            if any(x->x===nothing, (vtor_r, vtor_i, vpol_r, vpol_i))
                @warn "Velocity spectral variables missing in $(basename(fname)); skipping"
                continue
            end
            cfg, lvals, mvals, (θ, φ, r) = build_sht_from_nc(nc)
            coords["theta"] = θ; coords["phi"] = φ; coords["r"] = r
            try meta["geometry"] = NetCDF.getatt(nc, NetCDF.NC_GLOBAL, "geometry") catch end

            # Build Geodynamo config and fields; compute vorticity spectrally, synthesize to grid
            lmax = maximum(lvals); mmax = maximum(mvals)
            nlat = (θ === nothing) ? size(vtor_r,1) : length(θ)
            nlon = (φ === nothing) ? max(2*lmax+1, 4) : length(φ)
            nr = (r === nothing) ? size(vtor_r,2) : length(r)

            gcfg = Geodynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon)
            last_cfg[] = gcfg.sht_config
            pencils_nt = Geodynamo.create_pencil_topology(gcfg)
            pencils = (pencils_nt.θ, pencils_nt.φ, pencils_nt.r)
            pencil_spec = pencils_nt.spec
            domain = Geodynamo.create_radial_domain(nr)
            fields = Geodynamo.create_shtns_velocity_fields(Float64, gcfg, domain, pencils, pencil_spec)

            # Load spectral coefficients (single-rank layout)
            spec_tor_r = parent(fields.toroidal.data_real); spec_tor_i = parent(fields.toroidal.data_imag)
            spec_pol_r = parent(fields.poloidal.data_real); spec_pol_i = parent(fields.poloidal.data_imag)
            nlm_local = min(size(spec_tor_r,1), size(vtor_r,1))
            for i2 in 1:nlm_local, k2 in 1:nr
                spec_tor_r[i2,1,k2] = Float64(vtor_r[i2,k2])
                spec_tor_i[i2,1,k2] = Float64(vtor_i[i2,k2])
                spec_pol_r[i2,1,k2] = Float64(vpol_r[i2,k2])
                spec_pol_i[i2,1,k2] = Float64(vpol_i[i2,k2])
            end

            Geodynamo.compute_vorticity_spectral_full!(fields, domain)
            Geodynamo.shtnskit_vector_synthesis!(fields.toroidal, fields.poloidal, fields.velocity)
            Geodynamo.shtnskit_vector_synthesis!(fields.vort_toroidal, fields.vort_poloidal, fields.vorticity)

            u_r = parent(fields.velocity.r_component.data)
            u_t = parent(fields.velocity.θ_component.data)
            u_p = parent(fields.velocity.φ_component.data)
            w_r = parent(fields.vorticity.r_component.data)
            w_t = parent(fields.vorticity.θ_component.data)
            w_p = parent(fields.vorticity.φ_component.data)
            hr = u_r .* w_r
            hθ = u_t .* w_t
            hφ = u_p .* w_p
            # z-component: u_z = u_r cosθ - u_θ sinθ; ω_z = ω_r cosθ - ω_θ sinθ
            θvec = θ
            if θvec === nothing && (cfg !== nothing)
                try
                    θvec = acos.(cfg.x)
                catch
                    θvec = range(0, stop=π, length=size(u_r,1)) |> collect
                end
            end
            cθ = cos.(θvec); sθ = sin.(θvec)
            cθ_mat = reshape(cθ, size(u_r,1), 1, 1)
            sθ_mat = reshape(sθ, size(u_r,1), 1, 1)
            u_z = u_r .* cθ_mat .- u_t .* sθ_mat
            w_z = w_r .* cθ_mat .- w_t .* sθ_mat
            hz = u_z .* w_z
            h = hr .+ hθ .+ hφ

            if sum_h === nothing
                sum_h = zero(h); sum_hr = zero(hr); sum_hθ = zero(hθ); sum_hφ = zero(hφ); sum_hz = zero(hz)
            end
            if mode == "avg"
                sum_h .+= h; sum_hr .+= hr; sum_hθ .+= hθ; sum_hφ .+= hφ; sum_hz .+= hz
            elseif mode == "abs"
                sum_h .+= abs.(h); sum_hr .+= abs.(hr); sum_hθ .+= abs.(hθ); sum_hφ .+= abs.(hφ); sum_hz .+= abs.(hz)
            elseif mode == "rms"
                sum_h .+= h.^2; sum_hr .+= hr.^2; sum_hθ .+= hθ.^2; sum_hφ .+= hφ.^2; sum_hz .+= hz.^2
            end
            count += 1
        finally
            NetCDF.close(nc)
        end
    end

    count == 0 && error("No samples accumulated in range [$t0,$t1]")
    if mode == "avg" || mode == "abs"
        h_avg = sum_h ./ count
        hr_avg = sum_hr ./ count
        hθ_avg = sum_hθ ./ count
        hφ_avg = sum_hφ ./ count
        hz_avg = sum_hz ./ count
    else
        # rms
        h_avg = sqrt.(sum_h ./ count)
        hr_avg = sqrt.(sum_hr ./ count)
        hθ_avg = sqrt.(sum_hθ ./ count)
        hφ_avg = sqrt.(sum_hφ ./ count)
        hz_avg = sqrt.(sum_hz ./ count)
    end
    return (h_avg, hr_avg, hθ_avg, hφ_avg, hz_avg), count, sel, coords, meta, last_cfg[]
end

function main()
    if !MPI.Initialized(); MPI.Init(); end
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    try
        outdir, t0, t1, prefix, outpath, mode = parse_args(copy(ARGS))
        times = rank == 0 ? scan_times(outdir, prefix) : Float64[]
        nt = MPI.bcast(length(times), 0, comm)
        if rank != 0
            times = Vector{Float64}(undef, nt)
        end
        if nt > 0
            MPI.Bcast!(times, 0, comm)
        end
        sel = [t for t in times if t0 <= t <= t1]
        if isempty(sel)
            if rank == 0
                @warn "No times found in $outdir for prefix '$prefix' and range [$t0,$t1]"
            end
            return
        end

        # Each rank processes its subset
        sum_h = nothing; sum_hr = nothing; sum_hθ = nothing; sum_hφ = nothing; sum_hz = nothing
        count = 0
        coords0 = Dict{String,Any}(); meta0 = Dict{String,Any}(); cfg0 = nothing
        for (idx, t) in enumerate(sel)
            if (idx - 1) % nprocs != rank; continue; end
            (h_avg, hr_avg, hθ_avg, hφ_avg, hz_avg), c, times_used, coords, meta, cfg = time_average_helicity(outdir, t, t, prefix; mode=mode)
            if sum_h === nothing
                sum_h = zero(h_avg); sum_hr = zero(hr_avg); sum_hθ = zero(hθ_avg); sum_hφ = zero(hφ_avg); sum_hz = zero(hz_avg)
                coords0 = coords; meta0 = meta; cfg0 = cfg
            end
            sum_h .+= h_avg .* c; sum_hr .+= hr_avg .* c; sum_hθ .+= hθ_avg .* c; sum_hφ .+= hφ_avg .* c; sum_hz .+= hz_avg .* c
            count += c
        end

        # Write per-rank temporary result
        tmpfile = joinpath(outdir, @sprintf("timeavg_helicity_tmp_rank_%04d.jld2", rank))
        jldopen(tmpfile, "w"; compress=true) do f
            write(f, "sum_h", sum_h); write(f, "sum_hr", sum_hr); write(f, "sum_hθ", sum_hθ); write(f, "sum_hφ", sum_hφ); write(f, "sum_hz", sum_hz)
            write(f, "count", count); write(f, "coords", coords0); write(f, "meta", meta0)
        end
        MPI.Barrier(comm)

        if rank == 0
            total_h = nothing; total_hr = nothing; total_hθ = nothing; total_hφ = nothing; total_hz = nothing; total_count = 0
            coordsA = Dict{String,Any}(); metaA = Dict{String,Any}()
            for r in 0:(nprocs-1)
                tf = joinpath(outdir, @sprintf("timeavg_helicity_tmp_rank_%04d.jld2", r))
                if !isfile(tf); continue; end
                data = JLD2.load(tf)
                if total_h === nothing
                    total_h = zero(data["sum_h"]); total_hr = zero(data["sum_hr"]); total_hθ = zero(data["sum_hθ"]); total_hφ = zero(data["sum_hφ"]); total_hz = zero(data["sum_hz"])
                    coordsA = data["coords"]; metaA = data["meta"]
                end
                total_h .+= data["sum_h"]; total_hr .+= data["sum_hr"]; total_hθ .+= data["sum_hθ"]; total_hφ .+= data["sum_hφ"]; total_hz .+= data["sum_hz"]
                total_count += data["count"]
            end
            if total_count == 0
                @warn "No helicity samples accumulated"
                return
            end
            h_avg = total_h ./ total_count
            hr_avg = total_hr ./ total_count
            hθ_avg = total_hθ ./ total_count
            hφ_avg = total_hφ ./ total_count
            hz_avg = total_hz ./ total_count
            # Volume average (approximate weights)
            Hvol = NaN
            if !isempty(coordsA) && haskey(coordsA, "theta")
                nlat = size(h_avg,1); nlon = size(h_avg,2)
                lmax = nlat - 2
                tmpcfg = SHTnsKit.create_gauss_config(lmax, nlat; mmax=lmax, nlon=nlon, norm=:orthonormal)
                Hvol = volume_average_helicity(h_avg, tmpcfg, get(coordsA, "r", nothing))
            end
            if isempty(outpath)
                ttag = string(replace(@sprintf("%.6f", t0), "."=>"p"), "_", replace(@sprintf("%.6f", t1), "."=>"p"))
                outpath = joinpath(outdir, "timeavg_helicity_$(prefix)_$(ttag).jld2")
            end
            isdir(dirname(outpath)) || mkpath(dirname(outpath))
            jldopen(outpath, "w"; compress=true) do f
                write(f, "helicity", h_avg)
                write(f, "helicity_r", hr_avg)
                write(f, "helicity_theta", hθ_avg)
                write(f, "helicity_phi", hφ_avg)
                write(f, "helicity_z", hz_avg)
                write(f, "count", total_count)
                write(f, "times", sel)
                write(f, "coords", coordsA)
                write(f, "metadata", metaA)
                write(f, "time_range", (t0, t1))
                write(f, "prefix", prefix)
                write(f, "volume_avg_helicity", Hvol)
                write(f, "avg_mode", mode)
            end
            # cleanup temps
            for r in 0:(nprocs-1)
                tf = joinpath(outdir, @sprintf("timeavg_helicity_tmp_rank_%04d.jld2", r))
                isfile(tf) && rm(tf; force=true)
            end
            println(@sprintf("Saved time-averaged helicity to %s (samples=%d)", outpath, total_count))
        end
        MPI.Barrier(comm)
    finally
        try MPI.Barrier(MPI.COMM_WORLD) catch end
        if MPI.Initialized() && !MPI.Finalized(); MPI.Finalize(); end
    end
end

isinteractive() || main()
