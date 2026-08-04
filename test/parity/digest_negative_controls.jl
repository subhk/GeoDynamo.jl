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
end
