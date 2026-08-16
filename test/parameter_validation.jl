using Test

const TIMESTEPPER_CONTRACT_ROOT = normpath(joinpath(@__DIR__, ".."))
const TIMESTEPPER_API_SOURCE = read(
    joinpath(TIMESTEPPER_CONTRACT_ROOT, "src", "api", "timesteppers.jl"), String)
const TIMESTEPPER_CONFIG_DOC = read(
    joinpath(TIMESTEPPER_CONTRACT_ROOT, "docs", "src", "configuration.md"), String)
const TIMESTEPPER_GUIDE_DOC = read(
    joinpath(TIMESTEPPER_CONTRACT_ROOT, "docs", "src", "timestepping.md"), String)
const TIMESTEPPER_INDEX_DOC = read(
    joinpath(TIMESTEPPER_CONTRACT_ROOT, "docs", "src", "index.md"), String)

@testset "Parameter Validation" begin
    @testset "Valid default parameters" begin
        params = GeoDynamo.SolverParameters()
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict = false)
        @test is_valid
        @test isempty(errors)
    end

    @testset "Invalid physical parameters" begin
        for (kwargs,
            needle) in (
            ((Ek = -1.0,), "Ek"),
            ((Ra = -100.0,), "Ra"),
            ((Pr = -1.0,), "Pr"),
            ((Pm = -1.0,), "Pm")
        )
            params = GeoDynamo.SolverParameters(; kwargs...)
            is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict = false)
            @test !is_valid
            @test any(contains(e, needle) for e in errors)
        end
    end

    @testset "Invalid grid parameters" begin
        params = GeoDynamo.SolverParameters(nr = 2)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict = false)
        @test !is_valid
        @test any(contains(e, "nr") for e in errors)

        params = GeoDynamo.SolverParameters(lmax = 0)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict = false)
        @test !is_valid
        @test any(contains(e, "lmax") for e in errors)
    end

    @testset "Invalid timestepping" begin
        params = GeoDynamo.SolverParameters(timestep = -0.001)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict = false)
        @test !is_valid
        @test any(contains(e, "timestep") for e in errors)

        params = GeoDynamo.SolverParameters(stop_iteration = 0)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict = false)
        @test !is_valid
        @test any(contains(e, "stop_iteration") for e in errors)
    end

    @testset "End time must exceed start time" begin
        params = GeoDynamo.SolverParameters(start_time = 1.0, end_time = 0.5)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict = false)
        @test !is_valid
        @test any(contains(e, "end_time") for e in errors)
    end

    @testset "Ball geometry requires radius_ratio == 0" begin
        params = GeoDynamo.SolverParameters(geometry = :ball, radius_ratio = 0.35)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict = false)
        @test !is_valid
        @test any(contains(e, "radius_ratio") for e in errors)
    end

    @testset "Invalid geometry" begin
        params = GeoDynamo.SolverParameters(geometry = :cube)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict = false)
        @test !is_valid
        @test any(contains(e, "geometry") for e in errors)
    end

    @testset "Invalid architecture" begin
        params = GeoDynamo.SolverParameters(architecture = :cuda)
        is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict = false)
        @test !is_valid
        @test any(contains(e, "architecture") for e in errors)
    end

    @testset "Invalid timestepper object" begin
        # `timestepper` must be an AbstractTimestepper struct (e.g. CNAB2()); a
        # bare Symbol is rejected by the field-type conversion.
        @test_throws MethodError GeoDynamo.SolverParameters(timestepper = :cnab2)
        @test_throws MethodError GeoDynamo.SolverParameters(timestepper = :rk4)
    end

    @testset "Unsupported solver timesteppers fail validation" begin
        for timestepper in (
            GeoDynamo.ExponentialAdamsBashforth2(),
            GeoDynamo.ETD(),
            GeoDynamo.ThetaMethod(),
        )
            params = GeoDynamo.SolverParameters(timestepper = timestepper)
            is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict = false)
            @test !is_valid
            @test any(contains(error, "not supported by the solver") for error in errors)
        end

        for timestepper in (
            GeoDynamo.CNAB2(),
            GeoDynamo.ExponentialRungeKutta2(),
            GeoDynamo.RungeKutta3(),
        )
            params = GeoDynamo.SolverParameters(timestepper = timestepper)
            is_valid, errors, _ = GeoDynamo.validate_parameters(params; strict = false)
            @test is_valid
            @test isempty(errors)
        end

        # Retain the planned descriptor types for source compatibility, but the
        # API reference and user guides must not advertise them as runnable.
        unsupported_notice = "Not currently supported by `Simulation`"
        @test length(collect(eachmatch(
            Regex(unsupported_notice), TIMESTEPPER_API_SOURCE))) == 3
        @test occursin("Currently supported", TIMESTEPPER_CONFIG_DOC)
        normalized_config_doc = replace(lowercase(TIMESTEPPER_CONFIG_DOC), r"\s+" => " ")
        @test occursin("not currently supported", normalized_config_doc)
        @test occursin("not currently supported", lowercase(TIMESTEPPER_GUIDE_DOC))
        @test !occursin("three production-grade", lowercase(TIMESTEPPER_GUIDE_DOC))
        @test occursin("RungeKutta3", TIMESTEPPER_GUIDE_DOC)
        @test occursin("RungeKutta3", TIMESTEPPER_INDEX_DOC)
    end

    @testset "Simulation timestepper accepts a scheme symbol" begin
        # The high-level `Simulation`/`_resolve_timestepper` path is lenient: a
        # bare scheme Symbol passed as `timestepper` is converted to its struct
        # so the result carries an AbstractTimestepper (not a Symbol) into
        # SolverParameters.
        params = GeoDynamo.SolverParameters()
        opts = GeoDynamo._resolve_timestepper(:cnab2, nothing, nothing, nothing, nothing, params)
        @test opts.timestepper isa GeoDynamo.CNAB2
        @test opts.timestepper isa GeoDynamo.AbstractTimestepper
        @test opts.timestep_scheme === :cnab2

        opts_erk = GeoDynamo._resolve_timestepper(:erk2, nothing, nothing, nothing, nothing, params)
        @test opts_erk.timestepper isa GeoDynamo.ExponentialRungeKutta2

        # An explicit struct still works unchanged.
        opts_struct = GeoDynamo._resolve_timestepper(GeoDynamo.ExponentialRungeKutta2(), nothing, nothing, nothing, nothing, params)
        @test opts_struct.timestepper isa GeoDynamo.ExponentialRungeKutta2

        # An unknown scheme symbol is still rejected with a clear error.
        @test_throws ArgumentError GeoDynamo._resolve_timestepper(:rk4, nothing, nothing, nothing, nothing, params)
    end

    @testset "Strict mode throws on invalid params" begin
        params = GeoDynamo.SolverParameters(Ek = -1.0)
        @test_throws ErrorException GeoDynamo.validate_parameters(params; strict = true)
    end

    @testset "CFL warning for large timestep" begin
        params = GeoDynamo.SolverParameters(timestep = 1.0, lmax = 32, Ek = 1e-4)
        _, _, warnings = GeoDynamo.validate_parameters(params; strict = false)
        @test any(contains(w, "CFL") for w in warnings)
    end
end
