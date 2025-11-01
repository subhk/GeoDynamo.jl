#!/usr/bin/env julia

using Pkg
using MPI

const ROOT = normpath(joinpath(@__DIR__, ".."))

# Ensure GeoDynamo is properly available in the test environment
cd(ROOT) do
    Pkg.activate(".")
    Pkg.instantiate()
end

const TEST_SCRIPT = normpath(joinpath(ROOT, "test", "runtests.jl"))

function candidate_launchers()
    candidates = String[]

    if haskey(ENV, "MPIEXEC")
        push!(candidates, ENV["MPIEXEC"])
    end

    if isdefined(MPI, :mpiexecjl)
        try
            launcher = MPI.mpiexecjl()
            if launcher isa AbstractString
                push!(candidates, String(launcher))
            elseif launcher isa Cmd
                !isempty(launcher.exec) && push!(candidates, launcher.exec[1])
            end
        catch
            # ignore and fall through to other candidates
        end
    end

    append!(candidates, ["mpiexecjl", "mpiexec", "mpirun"])
    return candidates
end

function resolve_launcher()
    for cand in candidate_launchers()
        exe = Sys.which(cand)
        if exe !== nothing
            return exe
        end
    end
    error("No MPI launcher (`mpiexecjl`, `mpiexec`, or `mpirun`) found in PATH.")
end

mpiexec_path = resolve_launcher()

julia_exe = Base.julia_cmd().exec[1]
cmd = `$mpiexec_path -n 1 $julia_exe --project=$ROOT --check-bounds=yes --compiled-modules=no $TEST_SCRIPT`

env_overrides = Dict{String,String}()
env_overrides["JULIA_PROJECT"] = ROOT
env_overrides["JULIA_NUM_THREADS"] = get(ENV, "JULIA_NUM_THREADS", "auto")

launcher_name = lowercase(basename(mpiexec_path))
if !haskey(ENV, "JULIA_MPI_BINARY") && launcher_name != "mpiexecjl"
    env_overrides["JULIA_MPI_BINARY"] = "system"
end

cd(ROOT) do
    withenv(env_overrides...) do
        run(cmd)
    end
end
