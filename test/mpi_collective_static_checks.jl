# Collective discipline: no file under src/ except parallel/collectives.jl may call
# an MPI collective directly. Every collective goes through a named helper whose
# docstring states its contract, so "a rank-local decision gates a collective" has
# one place to audit. See docs/superpowers/specs/2026-09-05-collective-discipline-design.md.
using Test

const COLLECTIVE_STATIC_ROOT = normpath(joinpath(@__DIR__, "..", "src"))
const COLLECTIVE_STATIC_HOME = joinpath("parallel", "collectives.jl")
const COLLECTIVE_RE = r"\bMPI\.(Allreduce!?|Bcast!|bcast|Allgather\w*|Alltoall\w*|Reduce!?|Gather\w*|Scatter\w*|Barrier)\b"

# Comments are not calls. Strip from the first unquoted `#` to end of line.
function _strip_line_comment(line::AbstractString)
    in_str = false
    prev = ' '
    for (i, c) in pairs(line)
        if c == '"' && prev != '\\'
            in_str = !in_str
        elseif c == '#' && !in_str
            return line[1:prevind(line, i)]
        end
        prev = c
    end
    return line
end

function _raw_collective_sites(root)
    hits = String[]
    for (dir, _, files) in walkdir(root), file in files
        endswith(file, ".jl") || continue
        path = joinpath(dir, file)
        rel = relpath(path, root)
        rel == COLLECTIVE_STATIC_HOME && continue
        for (n, line) in enumerate(eachline(path))
            occursin(COLLECTIVE_RE, _strip_line_comment(line)) || continue
            push!(hits, "$rel:$n: $(strip(line))")
        end
    end
    return hits
end

@testset "MPI collective static contract" begin
    hits = _raw_collective_sites(COLLECTIVE_STATIC_ROOT)
    isempty(hits) || println(stderr, "raw MPI collectives outside $(COLLECTIVE_STATIC_HOME):\n  " *
                                     join(hits, "\n  "))
    @test isempty(hits)
    # the home itself must still contain the real calls (guards against an over-eager regex)
    home = read(joinpath(COLLECTIVE_STATIC_ROOT, COLLECTIVE_STATIC_HOME), String)
    @test occursin(COLLECTIVE_RE, home)
end
