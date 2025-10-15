"""
Merge per-rank Geodynamo NetCDF outputs into a single global file.

Usage examples:
  # Merge a single time and keep originals
  julia --project=. scripts/merge_outputs.jl ./output --time=1.250000

  # Merge all available times and delete per-rank files afterward
  julia --project=. scripts/merge_outputs.jl ./output --all --delete

Optional flags:
  --time=<float>       Merge only the specified time (e.g., --time=2.0)
  --all                Merge all times found in directory
  --prefix=<name>      Filename prefix (default: geodynamo)
  --outdir=<path>      Directory for combined files (default: same as input)
  --basename=<name>    Base name for combined files (default: combined_global)
  --delete             Delete per-rank files after successful merge
"""

using Geodynamo
using Printf
using Dates
using Random
using MPI

# Combiner APIs are available under Geodynamo module

function parse_args(args::Vector{String})
    if isempty(args)
        error("Usage: merge_outputs.jl <output_dir> [--time=<float>|--all] [--delete] [--prefix=...] [--outdir=...] [--basename=...]")
    end
    output_dir = abspath(args[1])
    time_val = nothing
    merge_all = false
    delete_old = false
    prefix = "geodynamo"
    outdir = output_dir
    basename = "combined_global"
    for arg in args[2:end]
        if startswith(arg, "--time=")
            time_val = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg == "--all"
            merge_all = true
        elseif arg == "--delete"
            delete_old = true
        elseif startswith(arg, "--prefix=")
            prefix = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--outdir=")
            outdir = abspath(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--basename=")
            basename = split(arg, "=", limit=2)[2]
        else
            @warn "Unknown argument $arg (ignored)"
        end
    end
    return output_dir, time_val, merge_all, delete_old, prefix, outdir, basename
end

function format_time_str(t::Float64)
    ts = @sprintf("%.6f", t)
    return replace(ts, "." => "p")
end

function find_rank_files(output_dir::String, time::Float64, prefix::String)
    time_str = format_time_str(time)
    files = readdir(output_dir)
    return [joinpath(output_dir, f) for f in files if endswith(f, ".nc") && occursin("$(prefix)_output_", f) && occursin("_time_$(time_str)", f) && occursin("rank_", f)]
end

function merge_time(output_dir::String, t::Float64; delete_old::Bool=false, prefix::String="geodynamo", outdir::String=output_dir, basename::String="combined_global")
    cfg = Geodynamo.create_combiner_config(verbose=true, include_diagnostics=true, save_combined=true, combined_filename=basename, output_dir=outdir)
    combiner = Geodynamo.combine_distributed_time(output_dir, t; config=cfg)
    if delete_old
        rank_files = find_rank_files(output_dir, t, prefix)
        for f in rank_files
            try
                rm(f)
                @info "Deleted $f"
            catch e
                @warn "Could not delete $f: $e"
            end
        end
    end
    return combiner
end

function main()
    if !MPI.Initialized(); MPI.Init(); end
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    try
        output_dir, time_val, merge_all, delete_old, prefix, outdir, basename = parse_args(copy(ARGS))
        isdir(outdir) || mkpath(outdir)

        if merge_all
            times = rank == 0 ? Geodynamo.list_available_times(output_dir, prefix) : Float64[]
            # Broadcast count then values for robustness
            nt = MPI.bcast(length(times), 0, comm)
            if rank != 0
                times = Vector{Float64}(undef, nt)
            end
            if nt > 0
                MPI.Bcast!(times, 0, comm)
            end
            if isempty(times)
                if rank == 0
                    @warn "No times found in $output_dir"
                end
                return
            end
            if rank == 0
                @info "MPI ranks: $nprocs — merging $(length(times)) time points in parallel"
            end
            # Distribute times across ranks in round-robin
            for (idx, t) in enumerate(times)
                if (idx - 1) % nprocs == rank
                    try
                        merge_time(output_dir, t; delete_old=delete_old, prefix=prefix, outdir=outdir, basename=basename)
                    catch e
                        @error "Rank $rank failed merging time $t" exception=e
                    end
                end
            end
            MPI.Barrier(comm)
        elseif time_val !== nothing
            if rank == 0
                merge_time(output_dir, time_val; delete_old=delete_old, prefix=prefix, outdir=outdir, basename=basename)
            end
            MPI.Barrier(comm)
        else
            error("Specify either --time=<float> or --all")
        end
    finally
        # Strict finalization
        try
            MPI.Barrier(comm)
        catch
        end
        if MPI.Initialized() && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end

isinteractive() || main()
