using Test
using GeoDynamo
using Logging

@testset "Parameter file parsing is loud about bad input" begin
    @testset "unknown parameter name warns and is dropped" begin
        tmpfile = tempname() * ".jl"
        try
            write(tmpfile, "nr = 32\nnot_a_real_parameter = 5\n")
            loaded = @test_logs (:warn, r"unknown solver parameter"i) match_mode = :any GeoDynamo.load_parameters_from_file(tmpfile)
            @test loaded.nr == 32          # the valid line still applied
        finally
            rm(tmpfile, force = true)
        end
    end

    @testset "unparseable value for a known parameter warns" begin
        tmpfile = tempname() * ".jl"
        try
            # `bogus_token` is an undefined symbol -> safe_eval_expr rejects it
            write(tmpfile, "nr = bogus_token\n")
            loaded = @test_logs (:warn, r"could not parse"i) match_mode = :any GeoDynamo.load_parameters_from_file(tmpfile)
            @test loaded.nr == 64          # default retained, not silently corrupted
        finally
            rm(tmpfile, force = true)
        end
    end

    @testset "a clean parameter file emits no warnings" begin
        tmpfile = tempname() * ".jl"
        try
            write(tmpfile, "# header comment\nnr = 48\nlmax = 16\n")
            loaded = @test_logs min_level = Logging.Warn GeoDynamo.load_parameters_from_file(tmpfile)
            @test loaded.nr == 48
            @test loaded.lmax == 16
        finally
            rm(tmpfile, force = true)
        end
    end
end
