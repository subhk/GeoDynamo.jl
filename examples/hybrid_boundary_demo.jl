#!/usr/bin/env julia

"""
Programmatic and hybrid boundary demo for the current GeoDynamo.jl API.

This example demonstrates:
1. fully programmatic temperature boundaries
2. file + programmatic hybrid temperature boundaries
3. file + programmatic hybrid composition boundaries
4. direct spherical-harmonic boundary patterns with `Ylm`
5. time-dependent programmatic boundary data

Run with:
    julia --project examples/hybrid_boundary_demo.jl
"""

using GeoDynamo
using GeoDynamo.bcs:
    Ylm,
    create_hybrid_composition_boundaries,
    create_hybrid_temperature_boundaries,
    create_programmatic_boundary,
    create_programmatic_temperature_boundaries,
    create_time_dependent_programmatic_boundary,
    get_boundary_statistics,
    print_boundary_info

example_file(name) = joinpath(@__DIR__, name)

function load_sample_boundary_generator()
    generator = Module(:HybridBoundarySampleGenerator)
    Base.include(generator, joinpath(@__DIR__, "create_sample_netcdf_boundaries.jl"))
    return generator
end

function ensure_sample_files!()
    required = (
        example_file("cmb_temp.nc"),
        example_file("surface_temp.nc"),
        example_file("cmb_composition.nc"),
        example_file("surface_composition.nc"),
    )
    all(isfile, required) && return nothing

    generator = load_sample_boundary_generator()
    cd(@__DIR__) do
        Base.invokelatest(getfield(generator, :create_sample_temperature_boundaries); nlat=64, nlon=128, time_dependent=false)
        Base.invokelatest(getfield(generator, :create_sample_composition_boundaries); nlat=64, nlon=128)
    end
    return nothing
end

function summarize_programmatic_set(label, boundaries)
    println(label)
    print_boundary_info(boundaries.boundary_set)
    println("  inner BC type: $(boundaries.inner_bc_type)")
    println("  outer BC type: $(boundaries.outer_bc_type)")
end

function demo_hybrid_sets(cfg)
    println("=== Programmatic and hybrid boundary sets ===")

    programmatic_temp = create_programmatic_temperature_boundaries(
        (:dirichlet, 4000.0),
        (:dirichlet, 300.0),
        cfg,
    )

    hybrid_temp = create_hybrid_temperature_boundaries(
        example_file("cmb_temp.nc"),
        (:dirichlet, 300.0),
        cfg,
    )

    swapped_temp = create_hybrid_temperature_boundaries(
        example_file("surface_temp.nc"),
        (:dirichlet, 4000.0),
        cfg;
        swap_boundaries=true,
    )

    hybrid_comp = create_hybrid_composition_boundaries(
        example_file("cmb_composition.nc"),
        (:neumann, 0.0),
        cfg,
    )

    summarize_programmatic_set("Programmatic temperature boundaries", programmatic_temp)
    summarize_programmatic_set("\nHybrid temperature boundaries (file inner, uniform outer)", hybrid_temp)
    summarize_programmatic_set("\nHybrid temperature boundaries (uniform inner, file outer)", swapped_temp)
    summarize_programmatic_set("\nHybrid composition boundaries", hybrid_comp)
end

function demo_spherical_harmonic_patterns(cfg)
    println("\n=== Direct spherical-harmonic boundary data ===")

    y20 = create_programmatic_boundary(Ylm(2, 0), cfg, 0.5; field_type="temperature")
    y32 = create_programmatic_boundary(Ylm(3, 2), cfg, 0.25; field_type="temperature")

    print_stats("Y₂⁰ temperature pattern", y20)
    print_stats("Y₃² temperature pattern", y32)
end

function demo_time_dependent_pattern(cfg)
    println("\n=== Time-dependent programmatic boundary data ===")

    rotating = create_time_dependent_programmatic_boundary(
        Ylm(3, 2),
        cfg,
        (0.0, 1.0),
        6;
        amplitude=0.4,
        time_factor=2π,
        field_type="temperature",
    )

    stats = get_boundary_statistics(rotating)
    println("Time steps  : $(rotating.ntime)")
    println("Grid        : $(rotating.nlat) × $(rotating.nlon)")
    println("Value range : [$(round(stats["min"], digits=3)), $(round(stats["max"], digits=3))] $(rotating.units)")
end

function print_stats(label, data)
    stats = get_boundary_statistics(data)
    println(label)
    println("  range: [$(round(stats["min"], digits=3)), $(round(stats["max"], digits=3))] $(stats["units"])")
    println("  mean : $(round(stats["mean"], digits=3)) $(stats["units"])")
end

function main()
    ensure_sample_files!()

    cfg = GeoDynamo.create_shtnskit_config(lmax=32, mmax=32, nlat=64, nlon=128, nr=32)
    demo_hybrid_sets(cfg)
    demo_spherical_harmonic_patterns(cfg)
    demo_time_dependent_pattern(cfg)

    println("\n=== Current workflow summary ===")
    println("Use `create_programmatic_temperature_boundaries` for simple analytic")
    println("Dirichlet/Neumann boundary sets, `create_hybrid_*_boundaries` when one")
    println("boundary comes from file data, and `create_programmatic_boundary` when")
    println("you need standalone `Ylm`-driven boundary datasets.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
