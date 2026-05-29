#!/usr/bin/env julia

"""
Initial-conditions demo using the current public GeoDynamo.jl API.

This example demonstrates:
1. creating real field containers
2. random initial conditions for scalar, velocity, and magnetic fields
3. analytical initial conditions for temperature, magnetic, velocity, and composition
4. the file-load/save entry points exposed by `GeoDynamo.InitialConditions`

Run with:
    julia --project examples/initial_conditions_example.jl
"""

using GeoDynamo
using LinearAlgebra
using Random

function summarize_scalar(label, field)
    real_part = parent(field.spectral.data_real)
    println(label)
    println("  max amplitude: $(round(maximum(abs.(real_part)), sigdigits=4))")
    println("  total energy : $(round(sum(abs2, real_part), sigdigits=4))")
end

function summarize_vector(label, field)
    tor = parent(field.𝒯.data_real)
    pol = parent(field.𝒫.data_real)
    println(label)
    println("  toroidal max : $(round(maximum(abs.(tor)), sigdigits=4))")
    println("  poloidal max : $(round(maximum(abs.(pol)), sigdigits=4))")
    println("  total energy : $(round(sum(abs2, tor) + sum(abs2, pol), sigdigits=4))")
end

function main(; lmax = 24, mmax = 24, nlat = 48, nlon = 96, nr = 32)
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    domain = create_shell_radial_domain(nr)

    temp_field = create_shtns_temperature_field(Float64, cfg, domain)
    magnetic_field = create_shtns_magnetic_fields(Float64, cfg, domain, domain)
    velocity_field = create_shtns_velocity_fields(Float64, cfg, domain)
    composition_field = create_shtns_composition_field(Float64, cfg, domain)

    println("=== Random initial conditions ===")
    Random.seed!(42)
    generate_random_initial_conditions!(
        temp_field, :temperature; amplitude = 0.1, modes_range = 1:10, seed = 42)
    generate_random_initial_conditions!(
        magnetic_field, :magnetic; amplitude = 0.01, modes_range = 1:8, seed = 7)
    generate_random_initial_conditions!(
        velocity_field, :velocity; amplitude = 0.001, modes_range = 1:12, seed = 9)
    generate_random_initial_conditions!(
        composition_field, :composition; amplitude = 0.05, modes_range = 1:6, seed = 11)

    summarize_scalar("Temperature random state", temp_field)
    summarize_vector("Magnetic random state", magnetic_field)
    summarize_vector("Velocity random state", velocity_field)
    summarize_scalar("Composition random state", composition_field)

    println("\n=== Analytical initial conditions ===")
    set_analytical_initial_conditions!(temp_field, :temperature, :conductive; amplitude = 1.0)
    set_analytical_initial_conditions!(magnetic_field, :magnetic, :dipole; amplitude = 1.0)
    set_analytical_initial_conditions!(velocity_field, :velocity, :convective; amplitude = 0.01)
    set_analytical_initial_conditions!(composition_field, :composition, :stratified;
        bottom_composition = 0.3, top_composition = 0.1)

    summarize_scalar("Temperature conductive profile", temp_field)
    summarize_vector("Magnetic dipole state", magnetic_field)
    summarize_vector("Velocity convective seed", velocity_field)
    summarize_scalar("Composition stratified profile", composition_field)

    println("\n=== File-based API entry points ===")
    println("Use `save_initial_conditions(field, :temperature, \\\"my_temperature_ic.nc\\\")`")
    println("to save a generated field, or `load_initial_conditions!(field, :temperature, path)`")
    println("to restore it later. The current loader still falls back to analytical")
    println("profiles when NetCDF content is not implemented for a field type.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
