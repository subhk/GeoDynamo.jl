using Test
using GeoDynamo

# Guard: every `GeoDynamo.<binding>` referenced in a docs `@docs` block must be
# defined. Documenter expands `@docs` by evaluating each binding and FAILS HARD
# (UndefVarError, not a warning — `warnonly` does not catch it) if one is missing,
# which breaks the docs deploy. A refactor that removes/renames an exported symbol
# without updating the docs has broken the deploy more than once; this test catches
# it at suite time, pointing to the exact file:binding.

@testset "docs @docs bindings are all defined" begin
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
                in_docs = false
                for line in eachline(path)
                    s = strip(line)
                    if startswith(s, "```@docs")
                        in_docs = true
                        continue
                    elseif startswith(s, "```")     # any fence (incl ``` or ```@autodocs) ends a @docs list
                        in_docs = false
                        continue
                    end
                    in_docs || continue
                    # a binding line: GeoDynamo.a.b.func  (optionally followed by a method signature)
                    m = match(r"^GeoDynamo((?:\.[A-Za-z_][A-Za-z0-9_!]*)+)", s)
                    m === nothing && continue
                    parts = split(m.captures[1][2:end], ".")   # drop leading '.'
                    resolved(parts) || push!(undefined, "$(relpath(path, docs_dir)): GeoDynamo.$(join(parts, '.'))")
                end
            end
        end
        for u in undefined
            @warn "Undefined @docs binding (will break the docs deploy)" binding = u
        end
        @test isempty(undefined)
    end
end
