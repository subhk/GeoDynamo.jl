using Test
using GeoDynamo

# Guard: every `GeoDynamo.<binding>` referenced inside a Documenter `@docs` or
# `@autodocs` block must resolve. Documenter EVALUATES these bindings — a `@docs`
# block documents each one, and an `@autodocs` `Filter = t -> !(t in (GeoDynamo.X,…))`
# evaluates each symbol in its exclusion tuple — and FAILS HARD (UndefVarError, NOT a
# warning — `warnonly` does not catch it) on a missing one, breaking the docs deploy.
# A refactor that removes/renames an exported symbol without updating the docs has
# broken the deploy more than once (both a `@docs` list and an `@autodocs` Filter);
# this test catches it at suite time, naming the exact file:binding.

@testset "docs @docs/@autodocs bindings are all defined" begin
    docs_dir = normpath(joinpath(@__DIR__, "..", "docs", "src"))
    if !isdir(docs_dir)
        @test_skip "docs/src not present"
    else
        # Resolve a dotted path like GeoDynamo.bcs.foo against the module tree.
        function resolved(parts)
            cur = GeoDynamo
            for p in parts
                sym = Symbol(p)
                isdefined(cur, sym) || return false
                cur = getfield(cur, sym)
            end
            return true
        end

        undefined = String[]
        for (root, _, files) in walkdir(docs_dir)
            for fn in files
                endswith(fn, ".md") || continue
                path = joinpath(root, fn)
                in_block = false
                for line in eachline(path)
                    s = strip(line)
                    if startswith(s, "```@docs") || startswith(s, "```@autodocs")
                        in_block = true
                        continue
                    elseif startswith(s, "```")     # closing (or other) fence ends the block
                        in_block = false
                        continue
                    end
                    in_block || continue
                    # match every `GeoDynamo.a.b.func` token in the line (indented /
                    # comma-separated @autodocs Filter entries, or bare @docs entries).
                    for m in eachmatch(r"GeoDynamo((?:\.[A-Za-z_][A-Za-z0-9_!]*)+)", s)
                        parts = split(m.captures[1][2:end], ".")   # drop leading '.'
                        resolved(parts) ||
                            push!(undefined, "$(relpath(path, docs_dir)): GeoDynamo.$(join(parts, '.'))")
                    end
                end
            end
        end
        for u in undefined
            @warn "Undefined @docs/@autodocs binding (will break the docs deploy)" binding = u
        end
        @test isempty(undefined)
    end
end
