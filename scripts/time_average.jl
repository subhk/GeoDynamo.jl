#!/usr/bin/env julia

"""
Time-average selected quantities from merged NetCDF outputs and save to a compressed JLD2 file.

Assumptions:
- Inputs are merged (global) NetCDF files produced by Geodynamo's combiner/output system.
- Variables are consistent across the files in shape and meaning.

Usage examples:

  # Average temperature from t=0.0 to t=2.0 and save default JLD2
  julia --project=. scripts/time_average.jl ./output --start=0.0 --end=2.0 --vars=temperature

  # Average multiple quantities and write to an explicit file
  julia --project=. scripts/time_average.jl ./output --start=1.0 --end=3.0 \
       --vars=temperature,velocity_toroidal_real,velocity_toroidal_imag \
       --out=./output/avg_1p0_3p0.jld2

  # Limit to files with a particular prefix (default combined_global)
  julia --project=. scripts/time_average.jl ./output --start=0 --end=5 --vars=temperature --prefix=combined_global

Notes:
- The script scans for files matching "<prefix>_time_<time>.nc" and averages over times in [start,end].
- The output JLD2 contains one dataset per requested variable plus metadata.
"""

using Printf
using NetCDF
using JLD2
using SHTnsKit
using MPI

function usage()
    println("Usage: time_average.jl <output_dir> --start=<t0> --end=<t1> --vars=v1[,v2,...] [--prefix=name] [--out=path.jld2]")
end

function parse_args(args)
    if isempty(args)
        usage(); error("missing arguments")
    end
    outdir = abspath(args[1])
    t0 = nothing
    t1 = nothing
    varlist = String[]
    prefix = "combined_global"
    outpath = ""
    for a in args[2:end]
        if startswith(a, "--start="); t0 = parse(Float64, split(a, "=", limit=2)[2]);
        elseif startswith(a, "--end="); t1 = parse(Float64, split(a, "=", limit=2)[2]);
        elseif startswith(a, "--vars=")
            v = split(split(a, "=", limit=2)[2], ",")
            append!(varlist, v)
        elseif startswith(a, "--prefix=")
            prefix = split(a, "=", limit=2)[2]
        elseif startswith(a, "--out=")
            outpath = split(a, "=", limit=2)[2]
        else
            @warn "Unknown argument $a (ignored)"
        end
    end
    (t0 === nothing || t1 === nothing || isempty(varlist)) && (usage(); error("--start, --end, and --vars are required"))
    return outdir, t0, t1, varlist, prefix, outpath
end

format_time_str(t::Float64) = replace(@sprintf("%.6f", t), "." => "p")

function scan_times(dir::String, prefix::String)
    files = filter(f -> endswith(f, ".nc") && occursin(prefix, f), readdir(dir))
    times = Float64[]
    pattern = r"time_(\d+p\d+)"
    for f in files
        if (m = match(pattern, f)) !== nothing
            push!(times, parse(Float64, replace(m.captures[1], "p" => ".")))
        end
    end
    return sort(unique(times))
end

function build_filename(dir::String, prefix::String, t::Float64)
    ts = format_time_str(t)
    # Accept both naming patterns used in outputs
    candidates = [
        joinpath(dir, "$(prefix)_time_$(ts).nc"),
        joinpath(dir, "$(prefix)_output_time_$(ts)_rank_0000.nc"),
        joinpath(dir, "$(prefix)_output_time_$(ts).nc"),
    ]
    for c in candidates
        if isfile(c); return c; end
    end
    # Fallback: any file in dir with the correct time token
    for f in readdir(dir)
        full = joinpath(dir, f)
        if endswith(f, ".nc") && occursin("time_$(ts)", f) && occursin(prefix, f)
            return full
        end
    end
    return ""
end

function read_var_if_exists(nc, name)
    varid = NetCDF.varid(nc, name)
    varid == -1 && return nothing
    return NetCDF.readvar(nc, name)
end

function build_sht_config_from_file(nc)
    l_values = Int.(read_var_if_exists(nc, "l_values"))
    m_values = Int.(read_var_if_exists(nc, "m_values"))
    lmax = maximum(l_values)
    mmax = maximum(m_values)
    nlat = (NetCDF.varid(nc, "theta") != -1) ? length(read_var_if_exists(nc, "theta")) : (lmax+2)
    nlon = (NetCDF.varid(nc, "phi") != -1) ? length(read_var_if_exists(nc, "phi")) : max(2*lmax+1, 4)
    cfg = SHTnsKit.create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon, norm=:orthonormal)
    rvec = nothing
    if NetCDF.varid(nc, "r") != -1
        rvec = Float64.(read_var_if_exists(nc, "r"))
    end
    return cfg, l_values, m_values, rvec
end

function vector_synthesis_theta_phi(cfg::SHTnsKit.SHTConfig,
                                    l_values::Vector{Int}, m_values::Vector{Int},
                                    tor_r::AbstractMatrix, tor_i::AbstractMatrix,
                                    pol_r::AbstractMatrix, pol_i::AbstractMatrix,
                                    rvec::Union{Vector{Float64},Nothing}=nothing)
    lmax = cfg.lmax; mmax = cfg.mmax
    nlat = cfg.nlat; nlon = cfg.nlon
    nlm, nr = size(tor_r)
    vt = Array{Float64}(undef, nlat, nlon, nr)
    vp = Array{Float64}(undef, nlat, nlon, nr)
    vr = Array{Float64}(undef, nlat, nlon, nr)
    # Temporary coefficient matrices
    tor = zeros(ComplexF64, lmax+1, mmax+1)
    pol = zeros(ComplexF64, lmax+1, mmax+1)
    for r in 1:nr
        fill!(tor, 0); fill!(pol, 0)
        for i in 1:nlm
            l = l_values[i]; m = m_values[i]
            if l <= lmax && m <= mmax
                tor[l+1, m+1] = complex(tor_r[i,r], tor_i[i,r])
                pol[l+1, m+1] = complex(pol_r[i,r], pol_i[i,r])
            end
        end
        vt_slice, vp_slice = SHTnsKit.SHsphtor_to_spat(cfg, pol, tor; real_output=true)
        vt[:,:,r] = vt_slice
        vp[:,:,r] = vp_slice
        # Radial component from poloidal only: v_r = (l(l+1)/r) * P_lm Y_lm
        if rvec !== nothing
            rr = rvec[min(r, length(rvec))]
            if rr > eps()
                pol_rad = zeros(ComplexF64, lmax+1, mmax+1)
                for i in 1:nlm
                    l = l_values[i]; m = m_values[i]
                    if l <= lmax && m <= mmax
                        coeff = pol[l+1, m+1]
                        pol_rad[l+1, m+1] = coeff * (l*(l+1)/rr)
                    end
                end
                vr_slice = SHTnsKit.synthesis(cfg, pol_rad; real_output=true)
                vr[:,:,r] = vr_slice
            else
                vr[:,:,r] .= 0
            end
        else
            vr[:,:,r] .= 0
        end
    end
    return vt, vp, vr
end

function time_average(dir::String, t0::Float64, t1::Float64, vars::Vector{String}, prefix::String)
    times = scan_times(dir, prefix)
    sel = [t for t in times if t0 <= t <= t1]
    isempty(sel) && error("No files found in $dir for prefix '$prefix' and time range [$t0, $t1]")

    # Initialize accumulators on first successful read per var
    sums = Dict{String, Any}()
    counts = Dict{String, Int}()
    shapes = Dict{String, Tuple}()
    sample_meta = Dict{String,Any}()

    for t in sel
        fname = build_filename(dir, prefix, t)
        if isempty(fname)
            @warn "No file found for time $t (skipping)"; continue
        end
        nc = NetCDF.open(fname, NC_NOWRITE)
        try
            # Grab time/step/geometry once
            if isempty(sample_meta)
                try sample_meta["time"] = read_var_if_exists(nc, "time")[1] catch end
                try sample_meta["step"] = read_var_if_exists(nc, "step")[1] catch end
                try sample_meta["geometry"] = NetCDF.getatt(nc, NetCDF.NC_GLOBAL, "geometry") catch end
            end
            # Handle vector conversion if requested
            convert_velocity = any(v -> v in ("velocity","velocity_vector"), vars)
            convert_magnetic = any(v -> v in ("magnetic","magnetic_vector"), vars)

            if convert_velocity
                # Load spectral pairs; skip if missing
                vtor_r = read_var_if_exists(nc, "velocity_toroidal_real"); vtor_i = read_var_if_exists(nc, "velocity_toroidal_imag")
                vpol_r = read_var_if_exists(nc, "velocity_poloidal_real"); vpol_i = read_var_if_exists(nc, "velocity_poloidal_imag")
                if vtor_r !== nothing && vtor_i !== nothing && vpol_r !== nothing && vpol_i !== nothing
                    cfg, l_values, m_values, rvec = build_sht_config_from_file(nc)
                    vt, vp, vr = vector_synthesis_theta_phi(cfg, l_values, m_values, Float64.(vtor_r), Float64.(vtor_i), Float64.(vpol_r), Float64.(vpol_i), rvec)
                    for (name, A) in zip(("velocity_theta","velocity_phi","velocity_r"), (vt, vp, vr))
                        if !haskey(sums, name)
                            sums[name] = zero(A); shapes[name] = size(A); counts[name] = 0
                        end
                        if size(A) != shapes[name]
                            @warn "Shape mismatch for '$name' in $(basename(fname)) (skipping)"
                        else
                            sums[name] .+= A; counts[name] += 1
                        end
                    end
                else
                    @warn "Velocity spectral variables not found in $(basename(fname)); skipping conversion"
                end
            end

            if convert_magnetic
                mtor_r = read_var_if_exists(nc, "magnetic_toroidal_real"); mtor_i = read_var_if_exists(nc, "magnetic_toroidal_imag")
                mpol_r = read_var_if_exists(nc, "magnetic_poloidal_real"); mpol_i = read_var_if_exists(nc, "magnetic_poloidal_imag")
                if mtor_r !== nothing && mtor_i !== nothing && mpol_r !== nothing && mpol_i !== nothing
                    cfg, l_values, m_values, rvec = build_sht_config_from_file(nc)
                    bt, bp, br = vector_synthesis_theta_phi(cfg, l_values, m_values, Float64.(mtor_r), Float64.(mtor_i), Float64.(mpol_r), Float64.(mpol_i), rvec)
                    for (name, A) in zip(("magnetic_theta","magnetic_phi","magnetic_r"), (bt, bp, br))
                        if !haskey(sums, name)
                            sums[name] = zero(A); shapes[name] = size(A); counts[name] = 0
                        end
                        if size(A) != shapes[name]
                            @warn "Shape mismatch for '$name' in $(basename(fname)) (skipping)"
                        else
                            sums[name] .+= A; counts[name] += 1
                        end
                    end
                else
                    @warn "Magnetic spectral variables not found in $(basename(fname)); skipping conversion"
                end
            end

            # Average plain variables (non-converted)
            for v in vars
                if v in ("velocity","velocity_vector","magnetic","magnetic_vector")
                    continue
                end
                data = read_var_if_exists(nc, v)
                if data === nothing
                    @warn "Variable '$v' not in $(basename(fname)) (skipping this file for the var)"; continue
                end
                A = Float64.(data)
                if !haskey(sums, v)
                    sums[v] = zero(A); shapes[v] = size(A); counts[v] = 0
                elseif size(A) != shapes[v]
                    @warn "Shape mismatch for '$v' in $(basename(fname)): got $(size(A)), expected $(shapes[v]) (skipping)"; continue
                end
                sums[v] .+= A; counts[v] += 1
            end
        finally
            NetCDF.close(nc)
        end
    end

    # Compute averages
    avgs = Dict{String, Any}()
    # Include converted variable names if requested
    allkeys = copy(vars)
    if any(v -> v in ("velocity","velocity_vector"), vars)
        append!(allkeys, ["velocity_theta","velocity_phi","velocity_r"])
    end
    if any(v -> v in ("magnetic","magnetic_vector"), vars)
        append!(allkeys, ["magnetic_theta","magnetic_phi","magnetic_r"])
    end

    for v in allkeys
        if !haskey(counts, v) || counts[v] == 0
            @warn "No samples accumulated for variable '$v'"
            continue
        end
        avgs[v] = sums[v] ./ counts[v]
    end

    return avgs, counts, sel, sample_meta
end

function main()
    if !MPI.Initialized(); MPI.Init(); end
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    try
        outdir, t0, t1, varlist, prefix, outpath = parse_args(copy(ARGS))
        # Determine times on rank 0 and broadcast
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

        # Each rank processes its assigned subset (round-robin)
        local_sums = Dict{String, Any}()
        local_counts = Dict{String, Int}()
        local_meta = Dict{String,Any}()
        for (idx, t) in enumerate(sel)
            if (idx - 1) % nprocs != rank; continue; end
            avgs_part, counts_part, times_used_part, meta_part = time_average(outdir, t, t, varlist, prefix)
            # accumulate
            for (k, A) in avgs_part
                if !haskey(local_sums, k)
                    local_sums[k] = zero(A)
                    local_counts[k] = 0
                end
                local_sums[k] .+= A .* counts_part[k]  # convert back to sums
                local_counts[k] += counts_part[k]
            end
            if isempty(local_meta)
                local_meta = meta_part
            end
        end

        # Write per-rank temp file
        isdir(outdir) || mkpath(outdir)
        tmpfile = joinpath(outdir, @sprintf("timeavg_tmp_rank_%04d.jld2", rank))
        jldopen(tmpfile, "w"; compress=true) do f
            write(f, "sums", local_sums)
            write(f, "counts", local_counts)
            write(f, "meta", local_meta)
            write(f, "times", sel)
            write(f, "vars", varlist)
        end
        MPI.Barrier(comm)

        # Rank 0 aggregates temp files
        if rank == 0
            global_sums = Dict{String, Any}()
            global_counts = Dict{String, Int}()
            for r in 0:(nprocs-1)
                tf = joinpath(outdir, @sprintf("timeavg_tmp_rank_%04d.jld2", r))
                if !isfile(tf); continue; end
                data = JLD2.load(tf)
                sums_r = data["sums"]; counts_r = data["counts"]
                for (k, S) in sums_r
                    if !haskey(global_sums, k)
                        global_sums[k] = zero(S)
                        global_counts[k] = 0
                    end
                    global_sums[k] .+= S
                    global_counts[k] += get(counts_r, k, 0)
                end
            end
            # Compute averages
            avgs = Dict{String, Any}()
            for (k, S) in global_sums
                c = global_counts[k]
                if c > 0
                    avgs[k] = S ./ c
                end
            end
            if isempty(outpath)
                ttag = string(replace(@sprintf("%.6f", t0), "."=>"p"), "_", replace(@sprintf("%.6f", t1), "."=>"p"))
                outpath = joinpath(outdir, "timeavg_$(prefix)_$(ttag).jld2")
            end
            isdir(dirname(outpath)) || mkpath(dirname(outpath))
            jldopen(outpath, "w"; compress=true) do f
                write(f, "variables", varlist)
                write(f, "averages", avgs)
                write(f, "counts", global_counts)
                write(f, "times", sel)
                write(f, "metadata", local_meta)
                write(f, "time_range", (t0, t1))
                write(f, "prefix", prefix)
            end
            # Cleanup tmp files
            for r in 0:(nprocs-1)
                tf = joinpath(outdir, @sprintf("timeavg_tmp_rank_%04d.jld2", r))
                isfile(tf) && rm(tf; force=true)
            end
            println(@sprintf("Saved time-averaged variables to %s", outpath))
        end
        MPI.Barrier(comm)
    finally
        try MPI.Barrier(MPI.COMM_WORLD) catch end
        if MPI.Initialized() && !MPI.Finalized(); MPI.Finalize(); end
    end
end

isinteractive() || main()
