#!/usr/bin/env julia

"""
NetCDF boundary-condition demo for the current GeoDynamo.jl boundary API.

This example shows how to:
1. generate sample NetCDF boundary files
2. load file-backed temperature and composition boundaries
3. validate the inner/outer files form a consistent boundary set
4. inspect summary statistics and interpolate onto another grid
5. inspect a time-dependent NetCDF boundary file directly

Run with:
    julia --project examples/netcdf_boundary_demo.jl
"""

using GeoDynamo
using GeoDynamo.bcs: get_boundary_statistics,
                     interpolate_boundary_to_grid,
                     load_composition_boundaries_from_files,
                     load_temperature_boundaries_from_files,
                     print_boundary_info,
                     read_netcdf_boundary_data,
                     validate_boundary_compatibility
using Printf

example_file(name) = joinpath(@__DIR__, name)

function load_sample_boundary_generator()
    generator = Module(:NetCDFBoundarySampleGenerator)
    Base.include(generator, joinpath(@__DIR__, "create_sample_netcdf_boundaries.jl"))
    return generator
end

function ensure_sample_files!()
    required = (
        example_file("cmb_temp.nc"),
        example_file("surface_temp.nc"),
        example_file("cmb_temp_timedep.nc"),
        example_file("surface_temp_timedep.nc"),
        example_file("cmb_composition.nc"),
        example_file("surface_composition.nc")
    )
    all(isfile, required) && return nothing

    generator = load_sample_boundary_generator()
    cd(@__DIR__) do
        Base.invokelatest(getfield(generator, :create_sample_temperature_boundaries);
            nlat = 64, nlon = 128, time_dependent = false)
        Base.invokelatest(getfield(generator, :create_sample_temperature_boundaries);
            nlat = 64, nlon = 128, time_dependent = true)
        Base.invokelatest(getfield(generator, :create_sample_composition_boundaries); nlat = 64, nlon = 128)
    end
    return nothing
end

function print_stats(label, data)
    stats = get_boundary_statistics(data)
    println(label)
    println("  range: [$(round(stats["min"], digits=3)), $(round(stats["max"], digits=3))] $(stats["units"])")
    println("  mean : $(round(stats["mean"], digits=3)) $(stats["units"])")
    println("  grid : $(stats["shape"])")
end

function demo_file_boundaries()
    println("=== File-backed boundary sets ===")

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = 21, mmax = 21, nlat = 64, nlon = 128, nr = 32)
    temp_boundaries = load_temperature_boundaries_from_files(
        example_file("cmb_temp.nc"), example_file("surface_temp.nc"), cfg)
    comp_boundaries = load_composition_boundaries_from_files(
        example_file("cmb_composition.nc"), example_file("surface_composition.nc"), cfg)

    validate_boundary_compatibility(
        temp_boundaries.boundary_set.inner_boundary,
        temp_boundaries.boundary_set.outer_boundary,
        "temperature"
    )
    validate_boundary_compatibility(
        comp_boundaries.boundary_set.inner_boundary,
        comp_boundaries.boundary_set.outer_boundary,
        "composition"
    )

    print_boundary_info(temp_boundaries.boundary_set)
    print_boundary_info(comp_boundaries.boundary_set)

    print_stats("Temperature inner boundary", temp_boundaries.boundary_set.inner_boundary)
    print_stats("Temperature outer boundary", temp_boundaries.boundary_set.outer_boundary)
    print_stats("Composition inner boundary", comp_boundaries.boundary_set.inner_boundary)
    print_stats("Composition outer boundary", comp_boundaries.boundary_set.outer_boundary)

    return cfg, temp_boundaries, comp_boundaries
end

function demo_interpolation(temp_boundaries)
    println("\n=== Interpolation to a coarser grid ===")

    target_theta = collect(range(0, π, length = 32))
    target_phi = collect(range(0, 2π, length = 65))[1:(end - 1)]

    inner = temp_boundaries.boundary_set.inner_boundary
    interpolated = interpolate_boundary_to_grid(inner, target_theta, target_phi)

    println("Original grid : $(inner.nlat) × $(inner.nlon)")
    println("Target grid   : $(length(target_theta)) × $(length(target_phi))")
    println("Interpolated range: [$(round(minimum(interpolated), digits=3)), $(round(maximum(interpolated), digits=3))] $(inner.units)")
end

function demo_time_dependent_file()
    println("\n=== Time-dependent NetCDF boundary ===")

    time_dependent = read_netcdf_boundary_data(example_file("cmb_temp_timedep.nc"))
    stats = get_boundary_statistics(time_dependent)

    println("File        : $(time_dependent.file_path)")
    println("Time-dependent: $(time_dependent.is_time_dependent)")
    println("Time steps  : $(time_dependent.ntime)")
    println("Grid        : $(time_dependent.nlat) × $(time_dependent.nlon)")
    println("Value range : [$(round(stats["min"], digits=3)), $(round(stats["max"], digits=3))] $(time_dependent.units)")

    if time_dependent.time !== nothing
        println("Time span   : $(first(time_dependent.time)) → $(last(time_dependent.time))")
    end
end

function main()
    ensure_sample_files!()

    cfg, temp_boundaries, _ = demo_file_boundaries()
    demo_interpolation(temp_boundaries)
    demo_time_dependent_file()

    println("\n=== Current workflow summary ===")
    println("Use `load_temperature_boundaries_from_files` / `load_composition_boundaries_from_files`")
    println("to build boundary sets, then inspect them with `print_boundary_info` and")
    println("`get_boundary_statistics`. File-backed boundary data can also be read")
    println("directly with `read_netcdf_boundary_data` for custom preprocessing.")
    println("Demo config nlm: $(cfg.nlm)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
