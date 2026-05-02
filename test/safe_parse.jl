using Test

@testset "Safe Parameter Parsing" begin
    @testset "safe_parse_value: literals" begin
        d = Dict{Symbol, Any}()

        # Booleans
        @test GeoDynamo.safe_parse_value("true", d) === true
        @test GeoDynamo.safe_parse_value("false", d) === false

        # Integers
        @test GeoDynamo.safe_parse_value("42", d) === 42
        @test GeoDynamo.safe_parse_value("-7", d) === -7
        @test GeoDynamo.safe_parse_value("0", d) === 0

        # Floats (including scientific notation)
        @test GeoDynamo.safe_parse_value("3.14", d) === 3.14
        @test GeoDynamo.safe_parse_value("1e-4", d) === 1e-4
        @test GeoDynamo.safe_parse_value("-2.5e3", d) === -2.5e3

        # Symbols
        @test GeoDynamo.safe_parse_value(":cnab2", d) === :cnab2
        @test GeoDynamo.safe_parse_value(":float64", d) === :float64

        # Strings
        @test GeoDynamo.safe_parse_value("\"hello\"", d) == "hello"
        @test GeoDynamo.safe_parse_value("\"\"", d) == ""

        # Mathematical constants
        @test GeoDynamo.safe_parse_value("π", d) === π
        @test GeoDynamo.safe_parse_value("pi", d) === π
    end

    @testset "safe_parse_value: arithmetic expressions" begin
        d = Dict{Symbol, Any}(:i_L => 32, :i_M => 16)

        # Simple arithmetic referencing known parameters
        @test GeoDynamo.safe_parse_value("i_L + 1", d) == 33
        @test GeoDynamo.safe_parse_value("i_M * 2", d) == 32
        @test GeoDynamo.safe_parse_value("i_L ÷ 2", d) == 16

        # Nested expression
        @test GeoDynamo.safe_parse_value("(i_L + 1) * (i_L + 2) ÷ 2", d) == (33 * 34) ÷ 2
    end

    @testset "safe_eval_expr: rejects disallowed operations" begin
        d = Dict{Symbol, Any}()

        # Arbitrary function call should be rejected
        expr = Meta.parse("open(\"file\")")
        @test_throws ArgumentError GeoDynamo.safe_eval_expr(expr, d)

        # Unknown symbol should be rejected
        expr = Meta.parse("unknown_var")
        @test_throws ArgumentError GeoDynamo.safe_eval_expr(expr, d)
    end

    @testset "safe_eval_expr: allowed operations" begin
        d = Dict{Symbol, Any}(:x => 4.0, :y => 9.0)

        @test GeoDynamo.safe_eval_expr(Meta.parse("sqrt(x)"), d) == 2.0
        @test GeoDynamo.safe_eval_expr(Meta.parse("abs(-3)"), d) == 3
        @test GeoDynamo.safe_eval_expr(Meta.parse("min(x, y)"), d) == 4.0
        @test GeoDynamo.safe_eval_expr(Meta.parse("max(x, y)"), d) == 9.0
        @test GeoDynamo.safe_eval_expr(Meta.parse("x ^ 2"), d) == 16.0
    end

    @testset "safe_eval_expr: literal passthrough" begin
        d = Dict{Symbol, Any}()
        @test GeoDynamo.safe_eval_expr(42, d) === 42
        @test GeoDynamo.safe_eval_expr(3.14, d) === 3.14
    end

    @testset "safe_eval_expr: QuoteNode for symbols" begin
        d = Dict{Symbol, Any}()
        qn = QuoteNode(:shell)
        @test GeoDynamo.safe_eval_expr(qn, d) === :shell
    end
end
