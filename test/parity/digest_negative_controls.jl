using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

include(joinpath(@__DIR__, "state_digest.jl"))
using .ParityDigest

# A digest built by hand, so these controls test the COMPARATOR in isolation
# from the walker. The walker is exercised by fixtures_test.jl in Task 2.
function _ctl_digest(values::Vector{Float64}; name = "a.b", dims = [length(values)])
    env = Dict{String, Any}("nthreads" => 1, "nranks" => 1,
        "word_size" => 64, "julia" => "1.11.1")
    fb = ParityDigest.FieldBits(name, dims, copy(values))
    return ParityDigest.StateDigest(env, Dict{String, Any}(), [fb],
        ParityDigest._hash_fields([fb]))
end

# A fieldless struct with no arrays of its own and no fields to recurse
# into: previously fell through every branch of `_walk!` and vanished with
# zero signal. `struct` must be defined at top level, not inside a testset.
struct _CtlUnclassifiedLeaf end

@testset "digest comparator negative controls" begin
    base = [1.0, 2.0, 3.0]

    @testset "identical digests compare equal" begin
        ok, msg = ParityDigest.digests_equal(_ctl_digest(base), _ctl_digest(base))
        @test ok
        @test isempty(msg)
    end

    @testset "1 ULP difference is detected" begin
        perturbed = copy(base)
        perturbed[2] = nextfloat(perturbed[2])
        ok, msg = ParityDigest.digests_equal(_ctl_digest(base), _ctl_digest(perturbed))
        @test !ok
        @test occursin("a.b", msg)
        @test occursin("index 2", msg)
    end

    @testset "signed zero is detected" begin
        z = [0.0, 0.0]
        nz = [0.0, -0.0]
        ok, msg = ParityDigest.digests_equal(_ctl_digest(z), _ctl_digest(nz))
        @test !ok
        @test occursin("index 2", msg)
        # Regression: a naive reinterpret(Int64,·)-subtract wraps to
        # typemin(Int64) for exactly this pair (0.0 reinterprets as 0,
        # -0.0 as typemin(Int64)), so this pins the *correct* signed
        # distance, not just that a difference was reported.
        @test occursin("(1 ULP)", msg)
    end

    @testset "matching NaNs in the same slot compare equal" begin
        n = [1.0, NaN, 3.0]
        ok, _ = ParityDigest.digests_equal(_ctl_digest(n), _ctl_digest(n))
        @test ok
    end

    @testset "environment mismatch reports non-comparable, not physics" begin
        a = _ctl_digest(base)
        b = _ctl_digest(base)
        b.env["nthreads"] = 4
        ok, msg = ParityDigest.digests_equal(a, b)
        @test !ok
        @test occursin("not comparable", msg)
        @test !occursin("index", msg)
    end

    @testset "shape mismatch is detected" begin
        a = _ctl_digest(base; dims = [3])
        b = _ctl_digest(base; dims = [1, 3])
        ok, msg = ParityDigest.digests_equal(a, b)
        @test !ok
        @test occursin("dims", msg)
    end

    @testset "name mismatch honours compare_names" begin
        a = _ctl_digest(base; name = "old.temperature")
        b = _ctl_digest(base; name = "new.payload")
        ok_strict, _ = ParityDigest.digests_equal(a, b; compare_names = true)
        @test !ok_strict
        ok_loose, _ = ParityDigest.digests_equal(a, b; compare_names = false)
        @test ok_loose
    end

    @testset "hash agreeing does not alone produce a pass" begin
        # Raw values are always confirmed, so a forged matching hash must still fail.
        a = _ctl_digest([1.0, 2.0])
        b = _ctl_digest([1.0, 3.0])
        forged = ParityDigest.StateDigest(b.env, b.info, b.fields, a.hash)
        ok, _ = ParityDigest.digests_equal(a, forged)
        @test !ok
    end

    @testset "unclassified shapes fail loud instead of vanishing" begin
        @testset "array with an unclassifiable eltype throws" begin
            out = ParityDigest.FieldBits[]
            seen = Base.IdSet{Any}()
            err = try
                ParityDigest._walk!(out, seen, "x.badarray", Any[1.0, "two"])
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("x.badarray", err.msg)
            @test occursin("Any", err.msg)
        end

        @testset "fieldless leaf type throws" begin
            out = ParityDigest.FieldBits[]
            seen = Base.IdSet{Any}()
            err = try
                ParityDigest._walk!(out, seen, "x.badleaf", _CtlUnclassifiedLeaf())
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("x.badleaf", err.msg)
            @test occursin("_CtlUnclassifiedLeaf", err.msg)
        end

        @testset "Dict is walked by value, not silently skipped nor thrown" begin
            # Documents the classification decision for the metadata::Dict
            # case: values are walked (a float payload is captured), keys
            # are not separately digested.
            out = ParityDigest.FieldBits[]
            seen = Base.IdSet{Any}()
            ParityDigest._walk!(out, seen, "x.meta",
                Dict{String, Any}("weight" => [1.0, 2.0]))
            @test length(out) == 1
            @test out[1].name == "x.meta[weight]"
            @test out[1].values == [1.0, 2.0]
        end

        @testset "Dict holding an unclassifiable value still throws, not swallows" begin
            out = ParityDigest.FieldBits[]
            seen = Base.IdSet{Any}()
            err = try
                ParityDigest._walk!(out, seen, "x.meta", Dict{String, Any}("odd" => Any[1, 2]))
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("x.meta[odd]", err.msg)
        end
    end
end
