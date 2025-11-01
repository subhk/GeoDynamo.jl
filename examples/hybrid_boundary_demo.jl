#!/usr/bin/env julia

"""
Hybrid Boundary Conditions Demo for GeoDynamo.jl

This example demonstrates all the different ways to mix NetCDF and programmatic
boundary conditions, showing the full flexibility of the hybrid system.

Key Features Demonstrated:
1. NetCDF + Programmatic combinations
2. Different programmatic patterns
3. Custom function boundaries
4. Time-dependent hybrid boundaries
5. Performance comparison

Run with:
julia hybrid_boundary_demo.jl
"""

using GeoDynamo
using Printf

function demo_hybrid_combinations()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║                Hybrid Boundary Combinations Demo             ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    # Create sample NetCDF files if needed
    if !isfile("cmb_temp.nc")
        println("Creating sample NetCDF files...")
        include("create_sample_netcdf_boundaries.jl")
        println()
    end
    
    # Setup configuration
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    domain = create_radial_domain(0.35, 1.0, 32)
    
    println("Configuration: $(config.nlat) × $(config.nlon) grid, lmax=$(config.lmax)\n")
    
    # Example 1: NetCDF inner + Programmatic outer
    println("1. NetCDF Inner + Programmatic Outer")
    println("   Use case: Complex CMB data + simple surface conditions")
    temp_boundaries_1 = create_hybrid_temperature_boundaries(
        "cmb_temp.nc",         # Complex CMB from NetCDF
        (:uniform, 300.0),     # Simple uniform surface
        config
    )
    print_boundary_summary(temp_boundaries_1, "Example 1")
    println()
    
    # Example 2: Programmatic inner + NetCDF outer
    println("2. Programmatic Inner + NetCDF Outer")
    println("   Use case: Simple analytical CMB + complex surface data")
    temp_boundaries_2 = create_hybrid_temperature_boundaries(
        (:y11, 4000.0, Dict("amplitude" => 200.0)),  # Y₁₁ pattern at CMB
        "surface_temp.nc",                           # Complex surface from NetCDF
        config
    )
    print_boundary_summary(temp_boundaries_2, "Example 2")
    println()
    
    # Example 3: Both programmatic (different patterns)
    println("3. Both Programmatic (Different Patterns)")
    println("   Use case: Analytical study with controlled boundary conditions")
    temp_boundaries_3 = create_hybrid_temperature_boundaries(
        (:plume, 4200.0, Dict(
            "center_theta" => π/3,
            "center_phi" => π/4,
            "width" => π/8
        )),                    # Hot plume at CMB
        (:hemisphere, 250.0, Dict("axis" => "z")),  # Hemispherical cooling at surface
        config
    )
    print_boundary_summary(temp_boundaries_3, "Example 3")
    println()
    
    return config, domain
end

function demo_programmatic_patterns()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║              Programmatic Pattern Showcase                   ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    
    patterns_demo = [
        ("Uniform", (:uniform, 4000.0), "Constant temperature everywhere"),
        ("Y₁₁ Harmonic", (:y11, 4000.0, Dict("amplitude" => 300.0)), "Degree-1 spherical harmonic"),
        ("Y₂₀ Zonal", (:y20, 1000.0), "Zonal (latitude-dependent) pattern"),
        ("Gaussian Plume", (:plume, 4500.0, Dict("width" => π/6, "center_theta" => π/2)), "Hot upwelling plume"),
        ("Hemisphere", (:hemisphere, 800.0, Dict("axis" => "z")), "Northern hemisphere anomaly"),
        ("Dipole", (:dipole, 1200.0), "Dipolar Y₁₀ pattern"),
        ("Quadrupole", (:quadrupole, 600.0), "Quadrupolar pattern"),
        ("Checkerboard", (:checkerboard, 400.0, Dict("nblocks_theta" => 4, "nblocks_phi" => 8)), "Alternating hot/cold blocks")
    ]
    
    for (i, (name, pattern_spec, description)) in enumerate(patterns_demo)
        println("$(i). $name Pattern")
        println("   Description: $description")
        
        # Create boundary with this pattern
        boundary_data = create_programmatic_boundary(
            pattern_spec[1], config,
            amplitude=pattern_spec[2],
            parameters=length(pattern_spec) > 2 ? pattern_spec[3] : Dict(),
            units="K",
            description="$name boundary pattern"
        )
        
        # Show statistics
        stats = get_boundary_statistics(boundary_data)
        println("   Range: [$(round(stats["min"], digits=1)), $(round(stats["max"], digits=1))] K")
        println("   Mean: $(round(stats["mean"], digits=1)) K")
        println()
    end
end

function demo_custom_functions()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║                Custom Function Boundaries                    ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    
    # Example 1: Mathematical expression
    println("1. Mathematical Expression")
    math_function(theta, phi) = sin(3*theta) * cos(2*phi) + 0.5*cos(theta)^2
    
    temp_boundaries_math = create_hybrid_temperature_boundaries(
        (:custom, 4000.0, Dict("function" => math_function)),
        (:uniform, 300.0),
        config
    )
    
    inner_stats = get_boundary_statistics(temp_boundaries_math.inner_boundary)
    println("   Function: sin(3θ)cos(2φ) + 0.5cos²(θ)")
    println("   Temperature range: [$(round(inner_stats["min"], digits=1)), $(round(inner_stats["max"], digits=1))] K")
    println()
    
    # Example 2: Physical process simulation
    println("2. Physical Process Simulation")
    
    function subduction_zones(theta, phi)
        # Simulate subduction zone cooling pattern
        # Multiple cold zones at specific locations
        cooling = 0.0
        
        # Pacific Ring of Fire locations (simplified)
        subduction_sites = [
            (π/6, 0.0),       # North Pacific
            (π/2, π),         # West Pacific  
            (2π/3, 3π/2),     # South Pacific
        ]
        
        for (zone_theta, zone_phi) in subduction_sites
            # Distance from subduction zone
            distance = acos(cos(theta)*cos(zone_theta) + 
                           sin(theta)*sin(zone_theta)*cos(phi - zone_phi))
            
            # Gaussian cooling around subduction zone
            cooling += exp(-(distance/(π/8))^2)
        end
        
        return cooling
    end
    
    temp_boundaries_subduction = create_hybrid_temperature_boundaries(
        (:uniform, 4000.0),   # Uniform hot CMB
        (:custom, 300.0, Dict("function" => subduction_zones)),  # Subduction cooling at surface
        config
    )
    
    outer_stats = get_boundary_statistics(temp_boundaries_subduction.outer_boundary)
    println("   Function: Subduction zone cooling pattern")
    println("   Surface temperature range: [$(round(outer_stats["min"], digits=1)), $(round(outer_stats["max"], digits=1))] K")
    println()
    
    # Example 3: Data-driven function
    println("3. Data-Driven Function")
    
    function observational_pattern(theta, phi)
        # Simulate loading from observational data
        # This could read from arrays loaded from real data
        
        # Simplified representation of seismic tomography-inspired pattern
        degree_2_component = 0.5 * (3*cos(theta)^2 - 1)  # Y₂₀
        degree_4_component = sin(theta)^2 * cos(2*phi)    # Y₂₂-like
        
        return degree_2_component + 0.3 * degree_4_component
    end
    
    temp_boundaries_obs = create_hybrid_temperature_boundaries(
        (:custom, 4000.0, Dict("function" => observational_pattern)),
        "surface_temp.nc",  # NetCDF for surface
        config
    )
    
    obs_stats = get_boundary_statistics(temp_boundaries_obs.inner_boundary)
    println("   Function: Seismic tomography-inspired pattern")
    println("   CMB temperature range: [$(round(obs_stats["min"], digits=1)), $(round(obs_stats["max"], digits=1))] K")
end

function demo_time_dependent_hybrid()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║            Time-Dependent Hybrid Boundaries                  ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    domain = create_radial_domain(0.35, 1.0, 32)
    
    # Create time-dependent rotating plume
    println("Creating rotating plume boundary (10 time steps)...")
    rotating_plume = create_time_dependent_programmatic_boundary(
        :plume, config, (0.0, 1.0), 10,  # 10 time steps over 1 time unit
        amplitude=4200.0,
        parameters=Dict(
            "width" => π/6,
            "center_theta" => π/3,
            "center_phi" => 0.0,
            "time_factor" => 2π  # One full rotation
        )
    )
    
    # Create hybrid boundary set: time-dependent inner + static outer
    temp_boundaries = BoundaryConditionSet(
        rotating_plume,                               # Time-dependent inner
        create_programmatic_boundary(:uniform, config, amplitude=300.0),  # Static outer
        "temperature",
        time()
    )
    
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    
    println("\nSimulating time evolution:")
    for time_step in 1:5
        current_time = (time_step - 1) * 0.25  # Quarter intervals
        
        # Apply time-dependent boundaries
        apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries, current_time)
        
        # Show how l=0,m=0 mode (mean) changes
        mean_temp = temp_field.boundary_values[1, 1]  # l=0,m=0 mode, inner boundary
        println("   Time $time_step (t=$(current_time)): Mean CMB temperature = $(round(mean_temp, digits=1)) K")
    end
    
    println("\nTime-dependent hybrid boundary demonstration complete")
end

function demo_composition_hybrid()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║              Compositional Hybrid Boundaries                 ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    domain = create_radial_domain(0.35, 1.0, 32)
    
    # Light element release at CMB + uniform surface
    println("1. Light Element Release Pattern")
    comp_boundaries_1 = create_hybrid_composition_boundaries(
        (:plume, 0.8, Dict(
            "center_theta" => π/4,
            "center_phi" => 3π/4,
            "width" => π/10
        )),                    # Concentrated light element release
        (:uniform, 0.0),       # Zero composition at surface
        config
    )
    
    print_boundary_summary(comp_boundaries_1, "Composition Example 1")
    
    # Multiple release sites
    println("\n2. Multiple Release Sites (Custom Function)")
    
    function multiple_light_sources(theta, phi)
        # Simulate multiple light element release sites
        sources = [
            (π/6, π/3),      # Source 1
            (π/2, 4π/3),     # Source 2  
            (5π/6, 0.0),     # Source 3
        ]
        
        total_release = 0.0
        for (src_theta, src_phi) in sources
            distance = acos(cos(theta)*cos(src_theta) + 
                           sin(theta)*sin(src_theta)*cos(phi - src_phi))
            total_release += exp(-(distance/(π/12))^2)  # Narrow release zones
        end
        
        return min(total_release, 1.0)  # Cap at 100% light element fraction
    end
    
    comp_boundaries_2 = create_hybrid_composition_boundaries(
        (:custom, 0.9, Dict("function" => multiple_light_sources)),
        "surface_composition.nc",  # Surface composition from NetCDF
        config
    )
    
    print_boundary_summary(comp_boundaries_2, "Composition Example 2")
    
    # Create fields and apply
    comp_field = create_shtns_composition_field(Float64, config, domain)
    apply_netcdf_composition_boundaries!(comp_field, comp_boundaries_1)
    
    println("\nCompositional hybrid boundaries demonstrated")
end

function print_boundary_summary(boundaries, example_name)
    inner_stats = get_boundary_statistics(boundaries.inner_boundary)
    outer_stats = get_boundary_statistics(boundaries.outer_boundary)
    
    println("   $(example_name):")
    println("     Inner: $(inner_stats["description"]) ($(inner_stats["file_path"]))")
    println("       Range: [$(round(inner_stats["min"], digits=1)), $(round(inner_stats["max"], digits=1))] $(inner_stats["units"])")
    println("     Outer: $(outer_stats["description"]) ($(outer_stats["file_path"]))")
    println("       Range: [$(round(outer_stats["min"], digits=1)), $(round(outer_stats["max"], digits=1))] $(outer_stats["units"])")
end

function demo_performance_comparison()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║                Performance Comparison                        ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    config = create_optimized_config(64, 64, nlat=128, nlon=256)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    
    # Test different boundary approaches
    boundary_types = [
        ("Programmatic Only", create_hybrid_temperature_boundaries((:uniform, 4000.0), (:uniform, 300.0), config)),
        ("NetCDF Only", load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc")),
        ("Hybrid (NetCDF+Prog)", create_hybrid_temperature_boundaries("cmb_temp.nc", (:uniform, 300.0), config))
    ]
    
    println("Boundary application performance (10 iterations each):")
    println("Configuration: $(config.nlat)×$(config.nlon) grid, $(config.lmax) modes")
    println()
    
    for (name, boundaries) in boundary_types
        reset_performance_stats!()
        
        @timed_transform begin
            for i in 1:10
                apply_netcdf_temperature_boundaries!(temp_field, boundaries)
            end
        end
        
        stats = get_performance_summary()
        avg_time_ms = stats.total_time_ns / (1_000_000 * stats.total_transforms)
        
        println("$name:")
        println("  Average time per application: $(round(avg_time_ms, digits=2)) ms")
        println("  Total memory allocated: $(round(stats.allocation_bytes / (1024^2), digits=1)) MB")
        println()
    end
    
    println("Performance Tips:")
    println("  • Programmatic boundaries are fastest (no file I/O)")
    println("  • NetCDF boundaries have file loading overhead")
    println("  • Hybrid approaches balance flexibility and performance")
    println("  • Grid interpolation adds computational cost")
end

function main()
    println("Hybrid Boundary Conditions Demo for GeoDynamo.jl")
    println("=" * 60)
    println("This demo shows all the ways to mix NetCDF and programmatic boundaries\n")
    
    try
        # Main demonstrations
        demo_hybrid_combinations()
        demo_programmatic_patterns()
        demo_custom_functions()
        demo_time_dependent_hybrid()
        demo_composition_hybrid()
        demo_performance_comparison()
        
        println("\n" * "=" * 60)
        println("Hybrid Boundary Conditions Demo Completed Successfully!")
        println()
        println("Summary of Hybrid Approaches:")
        println()
        println("1. **NetCDF + Programmatic**: Best for real data + simple boundaries")
        println("   Example: `create_hybrid_temperature_boundaries(\"data.nc\", (:uniform, 300.0), config)`")
        println()
        println("2. **Programmatic + NetCDF**: Best for analytical + observational boundaries")
        println("   Example: `create_hybrid_temperature_boundaries((:y11, 4000.0), \"surface.nc\", config)`")
        println()
        println("3. **Both Programmatic**: Best for controlled analytical studies")
        println("   Example: `create_hybrid_temperature_boundaries((:plume, 4200.0), (:hemisphere, 250.0), config)`")
        println()
        println("4. **Custom Functions**: Best for complex mathematical patterns")
        println("   Example: `(:custom, 100.0, Dict(\"function\" => my_func))`")
        println()
        println("5. **Time-Dependent**: Best for evolving boundary conditions")
        println("   Example: `create_time_dependent_programmatic_boundary(:plume, config, (0,10), 50)`")
        println()
        println("**Next Steps**:")
        println("• Try modifying the examples with your own parameters")
        println("• Create custom boundary functions for your specific physics")
        println("• Scale up to higher resolution for production simulations")
        println("• Combine with velocity and magnetic field modules")
        
    catch e
        println("Demo failed with error: $e")
        return 1
    end
    
    return 0
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end