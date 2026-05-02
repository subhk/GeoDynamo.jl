using Test

mutable struct DummyCompatibilityField
    config
end

@testset "Boundary Condition Utilities" begin
    @testset "determine_field_type_from_name" begin
        @test GeoDynamo.bcs.determine_field_type_from_name("temperature") == GeoDynamo.TEMPERATURE
        @test GeoDynamo.bcs.determine_field_type_from_name("Temperature") == GeoDynamo.TEMPERATURE
        @test GeoDynamo.bcs.determine_field_type_from_name("thermal_field") == GeoDynamo.TEMPERATURE

        @test GeoDynamo.bcs.determine_field_type_from_name("composition") == GeoDynamo.COMPOSITION
        @test GeoDynamo.bcs.determine_field_type_from_name("concentration") == GeoDynamo.COMPOSITION
        @test GeoDynamo.bcs.determine_field_type_from_name("xi") == GeoDynamo.COMPOSITION

        @test GeoDynamo.bcs.determine_field_type_from_name("velocity") == GeoDynamo.VELOCITY
        @test GeoDynamo.bcs.determine_field_type_from_name("flow") == GeoDynamo.VELOCITY
        @test GeoDynamo.bcs.determine_field_type_from_name("u") == GeoDynamo.VELOCITY

        @test GeoDynamo.bcs.determine_field_type_from_name("magnetic") == GeoDynamo.MAGNETIC
        @test GeoDynamo.bcs.determine_field_type_from_name("b") == GeoDynamo.MAGNETIC

        @test_throws ArgumentError GeoDynamo.bcs.determine_field_type_from_name("unknown_field")
    end

    @testset "get_default_units" begin
        @test GeoDynamo.bcs.get_default_units(GeoDynamo.TEMPERATURE) == "K"
        @test GeoDynamo.bcs.get_default_units(GeoDynamo.COMPOSITION) == "dimensionless"
        @test GeoDynamo.bcs.get_default_units(GeoDynamo.VELOCITY) == "m/s"
        @test GeoDynamo.bcs.get_default_units(GeoDynamo.MAGNETIC) == "T"
    end

    @testset "get_default_boundary_type" begin
        @test GeoDynamo.bcs.get_default_boundary_type(GeoDynamo.TEMPERATURE, GeoDynamo.INNER_BOUNDARY) == GeoDynamo.DIRICHLET
        @test GeoDynamo.bcs.get_default_boundary_type(GeoDynamo.COMPOSITION, GeoDynamo.INNER_BOUNDARY) == GeoDynamo.NEUMANN
        @test GeoDynamo.bcs.get_default_boundary_type(GeoDynamo.VELOCITY, GeoDynamo.OUTER_BOUNDARY) == GeoDynamo.DIRICHLET
        @test GeoDynamo.bcs.get_default_boundary_type(GeoDynamo.MAGNETIC, GeoDynamo.OUTER_BOUNDARY) == GeoDynamo.DIRICHLET
    end

    @testset "create_boundary_data: 2D (scalar, time-independent)" begin
        values = rand(Float64, 32, 64)
        bd = GeoDynamo.bcs.create_boundary_data(values, "temperature")
        @test bd.nlat == 32
        @test bd.nlon == 64
        @test bd.ntime == 1
        @test bd.ncomponents == 1
        @test bd.is_time_dependent == false
        @test bd.field_type == "temperature"
        @test bd.file_path == "programmatic"
    end

    @testset "create_boundary_data: 3D with time" begin
        values = rand(Float64, 16, 32, 10)
        time = collect(range(0.0, 1.0, length=10))
        bd = GeoDynamo.bcs.create_boundary_data(values, "temperature"; time=time)
        @test bd.nlat == 16
        @test bd.nlon == 32
        @test bd.ntime == 10
        @test bd.ncomponents == 1
        @test bd.is_time_dependent == true
    end

    @testset "create_boundary_data: 3D without time (vector)" begin
        values = rand(Float64, 16, 32, 3)
        bd = GeoDynamo.bcs.create_boundary_data(values, "velocity")
        @test bd.nlat == 16
        @test bd.nlon == 32
        @test bd.ntime == 1
        @test bd.ncomponents == 3
        @test bd.is_time_dependent == false
    end

    @testset "create_boundary_data: 4D (vector, time-dependent)" begin
        values = rand(Float64, 16, 32, 5, 3)
        bd = GeoDynamo.bcs.create_boundary_data(values, "velocity")
        @test bd.nlat == 16
        @test bd.nlon == 32
        @test bd.ntime == 5
        @test bd.ncomponents == 3
        @test bd.is_time_dependent == true
    end

    @testset "create_boundary_data: 1D throws" begin
        values = rand(Float64, 10)
        @test_throws ArgumentError GeoDynamo.bcs.create_boundary_data(values, "bad")
    end

    @testset "BoundaryCache lifecycle" begin
        cache = GeoDynamo.bcs.BoundaryCache{Float64}()
        @test isempty(cache.cache_keys)
        @test cache.last_update_time == 0.0

        # Store data
        data = rand(Float64, 4, 8)
        GeoDynamo.bcs.cache_boundary_data!(cache, "inner_t0", data)
        @test "inner_t0" in cache.cache_keys
        @test length(cache.cache_keys) == 1

        # Retrieve data
        retrieved = GeoDynamo.bcs.get_cached_data(cache, "inner_t0")
        @test retrieved === data

        # Missing key returns nothing
        @test GeoDynamo.bcs.get_cached_data(cache, "nonexistent") === nothing

        # Storing same key again doesn't duplicate
        GeoDynamo.bcs.cache_boundary_data!(cache, "inner_t0", rand(4, 8))
        @test length(cache.cache_keys) == 1

        # Clear
        GeoDynamo.bcs.clear_boundary_cache!(cache)
        @test isempty(cache.cache_keys)
        @test GeoDynamo.bcs.get_cached_data(cache, "inner_t0") === nothing
    end

    @testset "find_boundary_time_index" begin
        # Build a minimal BoundaryConditionSet with time coordinates
        time_coords = Float64[0.0, 0.5, 1.0, 1.5, 2.0]
        inner_vals = rand(Float64, 4, 8, 5)
        outer_vals = rand(Float64, 4, 8, 5)

        inner = GeoDynamo.bcs.create_boundary_data(inner_vals, "temperature"; time=time_coords)
        outer = GeoDynamo.bcs.create_boundary_data(outer_vals, "temperature"; time=time_coords)

        bc_set = GeoDynamo.bcs.BoundaryConditionSet{Float64}(
            inner, outer, "temperature", GeoDynamo.TEMPERATURE, 0.0
        )

        # Exact match
        @test GeoDynamo.bcs.find_boundary_time_index(bc_set, 0.0) == 1
        @test GeoDynamo.bcs.find_boundary_time_index(bc_set, 1.0) == 3
        @test GeoDynamo.bcs.find_boundary_time_index(bc_set, 2.0) == 5

        # Before first time
        @test GeoDynamo.bcs.find_boundary_time_index(bc_set, -1.0) == 1

        # After last time
        @test GeoDynamo.bcs.find_boundary_time_index(bc_set, 10.0) == 5

        # Between times — should return closest
        idx = GeoDynamo.bcs.find_boundary_time_index(bc_set, 0.2)
        @test idx == 1  # closer to 0.0 than 0.5

        idx = GeoDynamo.bcs.find_boundary_time_index(bc_set, 0.4)
        @test idx == 2  # closer to 0.5 than 0.0
    end

    @testset "find_boundary_time_index: time-independent" begin
        inner_vals = rand(Float64, 4, 8)
        outer_vals = rand(Float64, 4, 8)
        inner = GeoDynamo.bcs.create_boundary_data(inner_vals, "temperature")
        outer = GeoDynamo.bcs.create_boundary_data(outer_vals, "temperature")

        bc_set = GeoDynamo.bcs.BoundaryConditionSet{Float64}(
            inner, outer, "temperature", GeoDynamo.TEMPERATURE, 0.0
        )
        # Always returns 1 for time-independent
        @test GeoDynamo.bcs.find_boundary_time_index(bc_set, 0.0) == 1
        @test GeoDynamo.bcs.find_boundary_time_index(bc_set, 999.0) == 1
    end

    @testset "validate_field_boundary_compatibility checks both boundaries" begin
        cfg = GeoDynamo.create_shtnskit_config(lmax=4, mmax=4, nlat=10, nlon=16, nr=6)
        field = DummyCompatibilityField(cfg)

        inner = GeoDynamo.bcs.create_boundary_data(zeros(Float64, cfg.nlat, cfg.nlon), "temperature")
        outer = GeoDynamo.bcs.create_boundary_data(zeros(Float64, cfg.nlat, cfg.nlon + 1), "temperature")

        bc_set = GeoDynamo.bcs.BoundaryConditionSet{Float64}(
            inner, outer, "temperature", GeoDynamo.TEMPERATURE, 0.0
        )

        @test_throws ArgumentError GeoDynamo.bcs.validate_field_boundary_compatibility(
            field, GeoDynamo.TEMPERATURE, bc_set
        )
    end

    @testset "validate_boundary_files validates tuple specs and rejects bad input" begin
        cfg = GeoDynamo.create_shtnskit_config(lmax=4, mmax=4, nlat=10, nlon=16, nr=6)

        @test GeoDynamo.bcs.validate_boundary_files(
            GeoDynamo.TEMPERATURE,
            Dict(:inner => (:uniform, 1.0), :outer => (:dirichlet, 0.0)),
            cfg,
        )

        @test_throws ArgumentError GeoDynamo.bcs.validate_boundary_files(
            GeoDynamo.TEMPERATURE,
            Dict(:left => (:uniform, 1.0)),
            cfg,
        )

        @test_throws ArgumentError GeoDynamo.bcs.validate_boundary_files(
            GeoDynamo.TEMPERATURE,
            Dict(:inner => "definitely_missing_boundary_file.nc"),
            cfg,
        )
    end

    @testset "load_boundary_conditions! applies scalar programmatic boundaries" begin
        cfg = GeoDynamo.create_shtnskit_config(lmax=4, mmax=4, nlat=10, nlon=16, nr=6)
        temp_field = GeoDynamo.GeoDynamoShell.create_shell_temperature_field(Float64, cfg; nr=6)

        GeoDynamo.bcs.load_boundary_conditions!(
            temp_field,
            GeoDynamo.TEMPERATURE,
            Dict(:inner => (:uniform, 1.0), :outer => (:uniform, 0.5)),
        )

        @test temp_field.boundary_condition_set !== nothing
        @test all(temp_field.bc_type_inner .== Int(GeoDynamo.bcs.DIRICHLET))
        @test all(temp_field.bc_type_outer .== Int(GeoDynamo.bcs.DIRICHLET))
        @test count(!iszero, temp_field.boundary_values[1, :]) > 0
        @test count(!iszero, temp_field.boundary_values[2, :]) > 0
    end

    @testset "update_time_dependent_boundaries! refreshes scalar boundary values" begin
        cfg = GeoDynamo.create_shtnskit_config(lmax=4, mmax=4, nlat=10, nlon=16, nr=6)
        temp_field = GeoDynamo.GeoDynamoShell.create_shell_temperature_field(Float64, cfg; nr=6)

        time_coords = Float64[0.0, 1.0]
        inner_vals = zeros(Float64, cfg.nlat, cfg.nlon, 2)
        outer_vals = zeros(Float64, cfg.nlat, cfg.nlon, 2)
        inner_vals[:, :, 1] .= 1.0
        inner_vals[:, :, 2] .= 2.0
        outer_vals[:, :, 1] .= 3.0
        outer_vals[:, :, 2] .= 4.0

        inner = GeoDynamo.bcs.create_boundary_data(inner_vals, "temperature"; time=time_coords)
        outer = GeoDynamo.bcs.create_boundary_data(outer_vals, "temperature"; time=time_coords)
        temp_field.boundary_condition_set = GeoDynamo.bcs.BoundaryConditionSet{Float64}(
            inner, outer, "temperature", GeoDynamo.TEMPERATURE, 0.0
        )

        GeoDynamo.bcs.update_time_dependent_boundaries!(temp_field, GeoDynamo.TEMPERATURE, 1.0)

        expected_inner = GeoDynamo.bcs.shtns_physical_to_spectral(inner_vals[:, :, 2], cfg)
        expected_outer = GeoDynamo.bcs.shtns_physical_to_spectral(outer_vals[:, :, 2], cfg)

        @test temp_field.boundary_time_index[] == 2
        @test temp_field.boundary_values[1, :] ≈ expected_inner
        @test temp_field.boundary_values[2, :] ≈ expected_outer
    end
end
