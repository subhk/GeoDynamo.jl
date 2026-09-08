# Collective discipline: no file under src/ except parallel/collectives.jl may call
# an MPI collective directly. Every collective goes through a named helper whose
# docstring states its contract, so "a rank-local decision gates a collective" has
# one place to audit. See docs/superpowers/specs/2026-09-05-collective-discipline-design.md.
using Test

const COLLECTIVE_STATIC_ROOT = normpath(joinpath(@__DIR__, "..", "src"))
const COLLECTIVE_STATIC_HOME = joinpath("parallel", "collectives.jl")
const COLLECTIVE_RE = r"\bMPI\.(Allreduce!?|Bcast!|bcast|Allgather\w*|Alltoall\w*|Reduce!?|Gather\w*|Scatter\w*|Barrier)\b"

# Parsing ignores comments and string contents, while still visiting executable
# expressions in interpolated strings and calls split across multiple lines.
function _collective_call_sites!(hits, expr, rel, line=1)
    expr isa Expr || return hits
    if expr.head === :call
        callee = expr.args[1]
        if callee isa Expr && callee.head === :. && callee.args[1] === :MPI
            name = callee.args[2]
            if name isa QuoteNode && occursin(COLLECTIVE_RE, "MPI.$(name.value)")
                push!(hits, "$rel:$line: MPI.$(name.value)")
            end
        end
    end
    for arg in expr.args
        if arg isa LineNumberNode
            line = arg.line
        else
            _collective_call_sites!(hits, arg, rel, line)
        end
    end
    return hits
end

function _raw_collective_sites(root)
    hits = String[]
    for (dir, _, files) in walkdir(root), file in files
        endswith(file, ".jl") || continue
        path = joinpath(dir, file)
        rel = relpath(path, root)
        rel == COLLECTIVE_STATIC_HOME && continue
        _collective_call_sites!(hits, Meta.parseall(read(path, String)), rel)
    end
    return hits
end

@testset "MPI collective static contract" begin
    @testset "Only executable calls count" begin
        mktempdir() do dir
            path = joinpath(dir, "probe.jl")
            write(path, join([
                "# MPI.Allreduce(x, comm)",
                "\"\"\"", "Explains MPI.bcast and MPI.Barrier(comm).", "\"\"\"",
                "function documented() end",
                "println(\"MPI.Allgather(x, comm)\")",
                "#= MPI.Reduce(x, comm)", "MPI.Scatter(x, comm) =#",
            ], "\n"))
            @test isempty(_raw_collective_sites(dir))
            open(path, "a") do io
                write(io, "\nMPI.Allreduce(\n x, +, comm)\nMPI.Barrier(comm)\n")
            end
            @test length(_raw_collective_sites(dir)) == 2
        end
    end
    hits = _raw_collective_sites(COLLECTIVE_STATIC_ROOT)
    isempty(hits) || println(stderr, "raw MPI collectives outside $(COLLECTIVE_STATIC_HOME):\n  " *
                                     join(hits, "\n  "))
    @test isempty(hits)
    # the home itself must still contain the real calls (guards against an over-eager regex)
    home = read(joinpath(COLLECTIVE_STATIC_ROOT, COLLECTIVE_STATIC_HOME), String)
    @test occursin(COLLECTIVE_RE, home)
end
