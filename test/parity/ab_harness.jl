"""
    ParityAB

Mechanism B of the parity harness: in-tree A/B.

For a sub-project that BREAKS the public API (SP-2 scalar collapse, SP-3 geometry
type), a single driver script cannot reference a type that exists on only one
side of the change. So both implementations stay live in `src/` for the duration
of the sub-project, this harness asserts they agree bit-for-bit, and the legacy
path is deleted in that sub-project's final commit.

Usage from a sub-project's temporary test file:

    ParityAB.assert_ab_parity(
        case -> build_with_legacy_scalar_field(case),
        case -> build_with_collapsed_scalar_field(case);
        compare_names = false)

CRITICAL: `digests_equal` (ParityDigest) compares fields POSITIONALLY via `zip`,
even when `compare_names = false`. That means the legacy and new implementations
must walk their field trees in IDENTICAL RELATIVE ORDER — same count, same
order, same shapes — even though the names legitimately differ across a clean
break. If a collapsed struct reorders its field declarations relative to the
legacy struct, this harness will report a spurious difference, or worse, will
silently compare two unrelated fields that happen to share an index. Preserve
field declaration order across the break.

A `"field count differs: N vs M"` (or shape-mismatch) message from `digests_equal`
is STRUCTURAL, not a numerical finding: it means the two implementations expose
different field trees (e.g. SP-2's scalar-field collapse removing a field
outright). Resolve it by aligning the trees to match, not by chasing it as a
physics bug.
"""
module ParityAB

using Test
using ..ParityDigest
using ..ParityFixtures

export assert_ab_parity, compare_ab

struct ABResult
    case::ParityFixtures.ParityCase
    ok::Bool
    message::String
end

"""
    compare_ab(legacy_build, new_build;
        cases, compare_names, nsteps, legacy_step, new_step) -> Vector{ABResult}

Build, evolve, and digest both sides of every case. Returns results without
asserting, so a caller can inspect them — used by the harness's own self-test to
prove it can report a difference.

`legacy_step` / `new_step` are the two step functions the module docstring
promises ("two constructor functions and two step functions"). Each has
signature `(state; nsteps) -> state` and both default to
`ParityFixtures.evolve!`, so existing call sites are unaffected. Mechanism B
exists for clean-break sub-projects where one symbol cannot serve both sides
of the change — a sub-project needing a different step entry point on its new
side (e.g. because the refactor renames or splits the step function itself)
supplies `new_step` instead of patching this harness mid-flight.

Throws if `cases` is empty: an empty case list runs zero comparisons and cannot
demonstrate parity, but a caller-supplied `cases` that is accidentally emptied
(e.g. by an over-eager filter while debugging) would otherwise pass through
silently.
"""
function compare_ab(legacy_build, new_build;
        cases = ParityFixtures.select_matrix(),
        compare_names::Bool = false,
        nsteps::Int = 4,
        legacy_step = ParityFixtures.evolve!,
        new_step = ParityFixtures.evolve!)
    isempty(cases) &&
        error("compare_ab: `cases` is empty — an empty case list cannot demonstrate parity")
    results = ABResult[]
    for case in cases
        a = legacy_step(legacy_build(case); nsteps = nsteps)
        b = new_step(new_build(case); nsteps = nsteps)
        ok, msg = ParityDigest.digests_equal(
            ParityDigest.digest_state(a), ParityDigest.digest_state(b);
            compare_names = compare_names)
        push!(results, ABResult(case, ok, msg))
    end
    return results
end

"""
    assert_ab_parity(legacy_build, new_build;
        cases, compare_names, nsteps, legacy_step, new_step)

Assert every case agrees bit-for-bit, one `@test` per case so a failure names the
configuration that diverged.

`compare_names` defaults to `false` because this mechanism exists for clean
breaks, where the two implementations legitimately expose different field names
for the same quantity. Order, shape, and bits must still match exactly.

`legacy_step` / `new_step` are the two step functions, `(state; nsteps) ->
state`, each defaulting to `ParityFixtures.evolve!` — see `compare_ab`'s
docstring for why this exists and when to override one side.

REMINDER: comparison is POSITIONAL (see the module docstring). Both builders
must walk their field trees in identical relative order, or this can report a
spurious mismatch — or compare unrelated fields at the same index — even when
the two sides are otherwise equivalent.

Throws if `cases` is empty: an empty case list would otherwise emit zero
`@test`s and report a green, zero-comparison testset — see `compare_ab`.
"""
function assert_ab_parity(legacy_build, new_build;
        cases = ParityFixtures.select_matrix(),
        compare_names::Bool = false,
        nsteps::Int = 4,
        legacy_step = ParityFixtures.evolve!,
        new_step = ParityFixtures.evolve!)
    isempty(cases) &&
        error("assert_ab_parity: `cases` is empty — an empty case list cannot demonstrate parity")
    for r in compare_ab(legacy_build, new_build;
        cases = cases, compare_names = compare_names, nsteps = nsteps,
        legacy_step = legacy_step, new_step = new_step)
        @testset "$(r.case)" begin
            @test r.ok
            r.ok || @info "A/B parity failure" case = r.case detail = r.message
        end
    end
    return nothing
end

end # module
