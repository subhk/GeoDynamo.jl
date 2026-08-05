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

# Mirrors the real regression: bcs.BoundaryConditionSet.field_type::FieldType
# is an @enum reachable from state.fields.*.boundary_condition_set whenever
# file-based boundary conditions are loaded. `@enum`/`struct` must be defined
# at top level, not inside a testset.
@enum _CtlFieldType _CtlVelocityField _CtlTemperatureField
struct _CtlEnumHolder
    field_type::_CtlFieldType
end

# Mirrors the real regression: SHTnsMagneticFields.outer_domain::RadialDomain
# (src/physics/magnetic/field.jl:189) holds a RadialDomain whose
# dr_matrices::Vector{Matrix{Float64}} field is a float array of float
# arrays — an unclassified array shape that must never be reached because
# :outer_domain is skipped by name, exactly like :domain.
struct _CtlOuterDomainHolder
    outer_domain::Vector{Matrix{Float64}}
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

        @testset "@enum leaf does not throw (regression: real states reach FieldType)" begin
            out = ParityDigest.FieldBits[]
            seen = Base.IdSet{Any}()
            err = try
                ParityDigest._walk!(out, seen, "root", _CtlEnumHolder(_CtlVelocityField))
                nothing
            catch e
                e
            end
            @test err === nothing
            # The enum itself carries no float payload, so nothing is captured —
            # but critically, it must not have thrown getting here.
            @test isempty(out)
        end

        @testset "BC marker singleton (NoSlip) does not throw" begin
            # Regression: GeoDynamo.NoSlip/StressFree/InsulatingMagnetic/
            # ConductingMagnetic are fieldless <: AbstractVelocityBC /
            # AbstractThermalBC / AbstractMagneticBC singletons reachable
            # from every field's parameters.velocity_bcs.{inner,outer} etc.
            out = ParityDigest.FieldBits[]
            seen = Base.IdSet{Any}()
            err = try
                ParityDigest._walk!(out, seen, "root", GeoDynamo.NoSlip())
                nothing
            catch e
                e
            end
            @test err === nothing
            @test isempty(out)
        end

        @testset ":outer_domain is skipped like :domain" begin
            out = ParityDigest.FieldBits[]
            seen = Base.IdSet{Any}()
            holder = _CtlOuterDomainHolder([[1.0 2.0; 3.0 4.0]])
            err = try
                ParityDigest._walk!(out, seen, "root", holder)
                nothing
            catch e
                e
            end
            @test err === nothing
            @test isempty(out)
        end

        @testset "AbstractDict walk order is canonical, not insertion/deletion dependent" begin
            d1 = Dict{String, Any}("alpha" => [1.0], "beta" => [2.0], "gamma" => [3.0])

            # Same logical content as d1, but built via a very different
            # insertion/deletion history (200 keys inserted then deleted).
            # Dict's own iteration order is not guaranteed equal to d1's even
            # though the final key/value sets are identical.
            d2 = Dict{String, Any}()
            for i in 1:200
                d2["junk$i"] = i
            end
            d2["alpha"] = [1.0]
            d2["beta"] = [2.0]
            d2["gamma"] = [3.0]
            for i in 1:200
                delete!(d2, "junk$i")
            end
            @test Set(keys(d1)) == Set(keys(d2))

            out1 = ParityDigest.FieldBits[]
            ParityDigest._walk!(out1, Base.IdSet{Any}(), "root", d1)
            out2 = ParityDigest.FieldBits[]
            ParityDigest._walk!(out2, Base.IdSet{Any}(), "root", d2)

            @test [fb.name for fb in out1] == [fb.name for fb in out2]
            @test [fb.values for fb in out1] == [fb.values for fb in out2]
        end
    end
end
