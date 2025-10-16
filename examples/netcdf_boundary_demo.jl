#!/usr/bin/env julia

"""
NetCDF Boundary Conditions Demo for GeoDynamo.jl

This example demonstrates:
1. Loading temperature and compositional boundary conditions from NetCDF files
2. Validating boundary compatibility with SHTns configuration
3. Applying boundaries to temperature and composition fields
4. Time-dependent boundary updates during simulation
5. Inspecting boundary statistics and metadata

Run with:
julia netcdf_boundary_demo.jl
"""

# First, create sample NetCDF files if they don't exist
if !isfile("cmb_temp.nc")
    include("create_sample_netcdf_boundaries.jl")
    println("Sample NetCDF files created.\n")
end

using GeoDynamo
using Printf

function demo_netcdf_boundary_loading()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║                NetCDF Boundary Loading Demo                  ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    println("\n1. Loading temperature boundary conditions...")
    try
        # Load temperature boundaries from NetCDF files
        temp_boundaries = load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc")
        println("   Temperature boundaries loaded successfully")
        
        # Display detailed information
        print_boundary_info(temp_boundaries)
        
        # Get statistical summary
        inner_stats = get_boundary_statistics(temp_boundaries.inner_boundary)
        outer_stats = get_boundary_statistics(temp_boundaries.outer_boundary)
        
        println("\n   Temperature Statistics:")
        println("   Inner boundary (CMB):")
        println("     Range: [$(round(inner_stats["min"], digits=1)), $(round(inner_stats["max"], digits=1))] $(inner_stats["units"])")
        println("     Mean:  $(round(inner_stats["mean"], digits=1)) $(inner_stats["units"])")
        
        println("   Outer boundary (Surface):")
        println("     Range: [$(round(outer_stats["min"], digits=1)), $(round(outer_stats["max"], digits=1))] $(outer_stats["units"])")
        println("     Mean:  $(round(outer_stats["mean"], digits=1)) $(outer_stats["units"])")
        
    catch e
        println("   Failed to load temperature boundaries: $e")
        return false
    end
    
    println("\n2. Loading compositional boundary conditions...")
    try
        # Load composition boundaries from NetCDF files
        comp_boundaries = load_composition_boundaries("cmb_composition.nc", "surface_composition.nc")
        println("   Compositional boundaries loaded successfully")
        
        # Display information
        print_boundary_info(comp_boundaries)
        
    catch e
        println("   Failed to load compositional boundaries: $e")
        return false
    end
    
    return true
end

function demo_boundary_compatibility()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║              Boundary Compatibility Validation               ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    # Create SHTns configuration matching the NetCDF grid
    lmax, mmax = 21, 21  # Reduced for demo
    nlat, nlon = 64, 128  # Matching NetCDF files
    
    println("\nCreating SHTns configuration:")
    println("  Grid: $(nlat) × $(nlon)")
    println("  Spectral: lmax=$(lmax), mmax=$(mmax)")
    
    try
        config = create_optimized_config(lmax, mmax; 
                                       use_threading=true,
                                       use_simd=true,
                                       nlat=nlat, 
                                       nlon=nlon)
        println("  SHTns configuration created")
        
        # Load boundaries
        temp_boundaries = load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc")
        comp_boundaries = load_composition_boundaries("cmb_composition.nc", "surface_composition.nc")
        
        # Validate compatibility
        println("\nValidating temperature boundary compatibility...")
        validate_netcdf_temperature_compatibility(temp_boundaries, config)
        
        println("Validating composition boundary compatibility...")
        validate_netcdf_composition_compatibility(comp_boundaries, config)
        
        return config, temp_boundaries, comp_boundaries
        
    catch e
        println("  Configuration or validation failed: $e")
        return nothing
    end
end

function demo_boundary_application(config, temp_boundaries, comp_boundaries)
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║                Boundary Application Demo                     ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    try
        # Create radial domain (simplified)
        println("\nCreating simulation fields...")
        ri, ro = 0.35, 1.0  # Inner and outer radius
        N = 64  # Radial resolution
        domain = create_radial_domain(ri, ro, N)
        
        # Create temperature and composition fields
        temp_field = create_shtns_temperature_field(Float64, config, domain)
        comp_field = create_shtns_composition_field(Float64, config, domain)
        
        println("  Fields created successfully")
        
        # Apply boundary conditions at t=0
        println("\nApplying NetCDF boundary conditions...")
        current_time = 0.0
        
        apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries, current_time)
        apply_netcdf_composition_boundaries!(comp_field, comp_boundaries, current_time)
        
        # Display applied boundary values (first few modes)
        println("\nApplied boundary values (first 5 spectral modes):")
        println("Temperature boundaries:")
        for i in 1:min(5, length(temp_field.boundary_values[1, :]))
            inner_val = temp_field.boundary_values[1, i]
            outer_val = temp_field.boundary_values[2, i]
            println("  Mode $i: Inner = $(round(inner_val, digits=3)), Outer = $(round(outer_val, digits=3))")
        end
        
        println("\nComposition boundaries:")
        for i in 1:min(5, length(comp_field.boundary_values[1, :]))
            inner_val = comp_field.boundary_values[1, i]
            outer_val = comp_field.boundary_values[2, i]
            println("  Mode $i: Inner = $(round(inner_val, digits=3)), Outer = $(round(outer_val, digits=3))")
        end
        
        return temp_field, comp_field
        
    catch e
        println("  Boundary application failed: $e")
        return nothing
    end
end

function demo_time_dependent_boundaries()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║              Time-Dependent Boundaries Demo                 ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    # Check if time-dependent files exist
    if !isfile("cmb_temp_timedep.nc") || !isfile("surface_temp_timedep.nc")
        println("Time-dependent NetCDF files not found. Run create_sample_netcdf_boundaries.jl first.")
        return
    end
    
    try
        # Load time-dependent boundaries
        println("\nLoading time-dependent temperature boundaries...")
        temp_boundaries_td = load_temperature_boundaries("cmb_temp_timedep.nc", "surface_temp_timedep.nc")
        
        print_boundary_info(temp_boundaries_td)
        
        # Demonstrate time evolution
        println("\nDemonstrating time-dependent boundary updates:")
        
        # Create minimal configuration
        lmax, mmax = 10, 10
        config = create_optimized_config(lmax, mmax, nlat=64, nlon=128)
        domain = create_radial_domain(0.35, 1.0, 32)
        temp_field = create_shtns_temperature_field(Float64, config, domain)
        
        # Simulate time stepping
        dt = 0.1
        nsteps = 5
        
        for timestep in 1:nsteps
            current_time = (timestep - 1) * dt
            println("  Time step $timestep (t = $(current_time)):")
            
            # Update boundaries
            update_temperature_boundaries_from_netcdf!(temp_field, temp_boundaries_td, timestep, dt)
            
            # Show how l=0,m=0 mode (mean temperature) changes with time
            mean_inner = temp_field.boundary_values[1, 1]  # l=0,m=0 mode, inner boundary
            mean_outer = temp_field.boundary_values[2, 1]  # l=0,m=0 mode, outer boundary
            println("    Mean temperatures: Inner = $(round(mean_inner, digits=1)) K, Outer = $(round(mean_outer, digits=1)) K")
        end
        
    catch e
        println("  Time-dependent boundary demo failed: $e")
    end
end

function demo_advanced_interpolation()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║                Advanced Interpolation Demo                   ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    try
        # Load boundary data
        temp_boundaries = load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc")
        
        # Create different target grids to demonstrate interpolation
        println("\nDemonstrating grid interpolation:")
        
        # Original grid
        orig_data = temp_boundaries.inner_boundary
        println("  Original grid: $(orig_data.nlat) × $(orig_data.nlon)")
        
        # Create a smaller target grid
        target_nlat, target_nlon = 32, 64
        target_theta = collect(range(0, π, length=target_nlat))
        target_phi = collect(range(0, 2π, length=target_nlon+1))[1:end-1]
        
        # Interpolate to new grid
        println("  Interpolating to: $(target_nlat) × $(target_nlon)")
        interpolated = interpolate_boundary_to_grid(orig_data, target_theta, target_phi)
        
        println("  Interpolation successful")
        println("  Interpolated data range: [$(round(minimum(interpolated), digits=1)), $(round(maximum(interpolated), digits=1))] K")
        
        # Compare statistics
        orig_mean = sum(orig_data.values) / length(orig_data.values)
        interp_mean = sum(interpolated) / length(interpolated)
        println("  Mean preservation: Original = $(round(orig_mean, digits=1)) K, Interpolated = $(round(interp_mean, digits=1)) K")
        
    catch e
        println("  Interpolation demo failed: $e")
    end
end

function main()
    println("NetCDF Boundary Conditions Demo for GeoDynamo.jl")
    println("=" ^ 60)
    
    # Demo 1: Loading boundary conditions
    success = demo_netcdf_boundary_loading()
    if !success
        println("\nWarning: Could not proceed - boundary loading failed")
        return
    end
    
    # Demo 2: Compatibility validation
    result = demo_boundary_compatibility()
    if result === nothing
        println("\nWarning: Could not proceed - compatibility validation failed")
        return
    end
    config, temp_boundaries, comp_boundaries = result
    
    # Demo 3: Boundary application
    fields = demo_boundary_application(config, temp_boundaries, comp_boundaries)
    
    # Demo 4: Time-dependent boundaries
    demo_time_dependent_boundaries()
    
    # Demo 5: Advanced interpolation
    demo_advanced_interpolation()
    
    println("\n" * "=" ^ 60)
    println("NetCDF Boundary Conditions Demo Completed Successfully!")
    println()
    println("Key takeaways:")
    println("• NetCDF boundary files can be easily loaded and validated")
    println("• Both time-independent and time-dependent boundaries are supported")
    println("• Automatic interpolation handles grid mismatches")
    println("• Boundary conditions integrate seamlessly with SHTns transforms")
    println("• Statistical analysis and metadata inspection is straightforward")
    println()
    println("For production simulations:")
    println("• Ensure NetCDF files have proper coordinates and metadata")
    println("• Validate grid compatibility before long simulations")
    println("• Consider memory usage for large time-dependent boundary datasets")
    println("• Use appropriate interpolation for different grid resolutions")
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
