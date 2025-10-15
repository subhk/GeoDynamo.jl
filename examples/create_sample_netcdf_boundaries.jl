#!/usr/bin/env julia

"""
Example script to create sample NetCDF boundary condition files for temperature and composition.

This script demonstrates:
1. Creating NetCDF files with proper structure for boundary conditions
2. Both time-independent and time-dependent boundary examples
3. Proper coordinate specification and metadata

Run with:
julia create_sample_netcdf_boundaries.jl
"""

using NCDatasets
using Printf

"""
    create_sample_temperature_boundaries(; nlat=64, nlon=128, time_dependent=false)

Create sample NetCDF files for temperature boundary conditions.
"""
function create_sample_temperature_boundaries(; nlat=64, nlon=128, time_dependent=false)
    println("Creating sample temperature boundary files...")
    
    # Create coordinate arrays
    lat = collect(range(-90, 90, length=nlat))          # Latitude [-90, 90]
    lon = collect(range(0, 360, length=nlon+1))[1:end-1] # Longitude [0, 360)
    theta = (90 .- lat) .* π/180                         # Colatitude in radians [0, π]
    phi = lon .* π/180                                   # Longitude in radians [0, 2π)
    
    if time_dependent
        # Create time-dependent boundary conditions
        ntime = 10
        time = collect(0.0:0.1:0.9)  # 10 time steps
        
        println("  Creating time-dependent temperature boundaries with $ntime time steps")
        
        # Inner boundary (CMB) - time-dependent hot plume
        create_timedep_temperature_inner_bc("cmb_temp_timedep.nc", theta, phi, time, nlat, nlon, ntime)
        
        # Outer boundary (surface) - time-dependent cooling
        create_timedep_temperature_outer_bc("surface_temp_timedep.nc", theta, phi, time, nlat, nlon, ntime)
    else
        # Create time-independent boundary conditions
        println("  Creating time-independent temperature boundaries")
        
        # Inner boundary (CMB) - hot with Y₁₁ perturbation
        create_temperature_inner_bc("cmb_temp.nc", theta, phi, nlat, nlon)
        
        # Outer boundary (surface) - cold with heterogeneity
        create_temperature_outer_bc("surface_temp.nc", theta, phi, nlat, nlon)
    end
    
    println("  Temperature boundary files created successfully")
end

"""
    create_temperature_inner_bc(filename, theta, phi, nlat, nlon)

Create time-independent CMB temperature boundary condition.
"""
function create_temperature_inner_bc(filename, theta, phi, nlat, nlon)
    NCDataset(filename, "c") do ds
        # Define dimensions
        defDim(ds, "lat", nlat)
        defDim(ds, "lon", nlon)
        
        # Define coordinate variables
        defVar(ds, "theta", Float64, ("lat",), attrib=Dict(
            "long_name" => "Colatitude",
            "units" => "radians",
            "description" => "Colatitude coordinate (0 to π)"
        ))
        
        defVar(ds, "phi", Float64, ("lon",), attrib=Dict(
            "long_name" => "Longitude", 
            "units" => "radians",
            "description" => "Longitude coordinate (0 to 2π)"
        ))
        
        # Define temperature variable
        temp_var = defVar(ds, "temperature", Float64, ("lat", "lon"), attrib=Dict(
            "long_name" => "Temperature boundary condition",
            "units" => "K",
            "description" => "CMB temperature boundary condition",
            "boundary_type" => "inner",
            "boundary_location" => "core-mantle boundary"
        ))
        
        # Write coordinate data
        ds["theta"][:] = theta
        ds["phi"][:] = phi
        
        # Create CMB temperature pattern: hot with Y₁₁ perturbation
        temperature = zeros(nlat, nlon)
        T_base = 4000.0  # Base CMB temperature [K]
        T_pert = 200.0   # Perturbation amplitude [K]
        
        for (i, th) in enumerate(theta)
            for (j, ph) in enumerate(phi)
                # Base temperature plus Y₁₁ spherical harmonic perturbation
                Y11_real = sin(th) * cos(ph)  # Real part of Y₁₁
                temperature[i, j] = T_base + T_pert * Y11_real
            end
        end
        
        # Write temperature data
        temp_var[:] = temperature
        
        # Global attributes
        ds.attrib["title"] = "CMB Temperature Boundary Condition"
        ds.attrib["description"] = "Sample inner boundary temperature for geodynamo simulation"
        ds.attrib["created_by"] = "Geodynamo.jl sample script"
        ds.attrib["creation_date"] = string(Dates.now())
        ds.attrib["grid_type"] = "gaussian"
        ds.attrib["nlat"] = nlat
        ds.attrib["nlon"] = nlon
    end
end

"""
    create_temperature_outer_bc(filename, theta, phi, nlat, nlon)

Create time-independent surface temperature boundary condition.
"""
function create_temperature_outer_bc(filename, theta, phi, nlat, nlon)
    NCDataset(filename, "c") do ds
        # Define dimensions
        defDim(ds, "lat", nlat)
        defDim(ds, "lon", nlon)
        
        # Define coordinate variables
        defVar(ds, "theta", Float64, ("lat",))
        defVar(ds, "phi", Float64, ("lon",))
        
        # Define temperature variable
        temp_var = defVar(ds, "temperature", Float64, ("lat", "lon"), attrib=Dict(
            "long_name" => "Surface temperature boundary condition",
            "units" => "K",
            "description" => "Surface temperature boundary condition",
            "boundary_type" => "outer",
            "boundary_location" => "surface"
        ))
        
        # Write coordinates
        ds["theta"][:] = theta
        ds["phi"][:] = phi
        
        # Create surface temperature pattern: cold with continents/oceans
        temperature = zeros(nlat, nlon)
        T_base = 300.0   # Base surface temperature [K]
        T_pert = 50.0    # Temperature variation [K]
        
        for (i, th) in enumerate(theta)
            for (j, ph) in enumerate(phi)
                # Latitudinal temperature variation
                lat_factor = cos(th)  # Warmer at equator, colder at poles
                
                # Add some longitudinal heterogeneity (simplified continents)
                lon_factor = 0.3 * cos(3*ph) + 0.2 * sin(2*ph)
                
                temperature[i, j] = T_base + T_pert * lat_factor + 10.0 * lon_factor
            end
        end
        
        # Write temperature data
        temp_var[:] = temperature
        
        # Global attributes
        ds.attrib["title"] = "Surface Temperature Boundary Condition"
        ds.attrib["description"] = "Sample outer boundary temperature for geodynamo simulation"
        ds.attrib["created_by"] = "Geodynamo.jl sample script"
        ds.attrib["creation_date"] = string(Dates.now())
    end
end

"""
    create_timedep_temperature_inner_bc(filename, theta, phi, time, nlat, nlon, ntime)

Create time-dependent CMB temperature boundary condition.
"""
function create_timedep_temperature_inner_bc(filename, theta, phi, time, nlat, nlon, ntime)
    NCDataset(filename, "c") do ds
        # Define dimensions
        defDim(ds, "lat", nlat)
        defDim(ds, "lon", nlon)
        defDim(ds, "time", ntime)
        
        # Define coordinate variables
        defVar(ds, "theta", Float64, ("lat",))
        defVar(ds, "phi", Float64, ("lon",))
        defVar(ds, "time", Float64, ("time",), attrib=Dict(
            "long_name" => "Time",
            "units" => "dimensionless_time",
            "description" => "Simulation time"
        ))
        
        # Define temperature variable (3D: lat, lon, time)
        temp_var = defVar(ds, "temperature", Float64, ("lat", "lon", "time"), attrib=Dict(
            "long_name" => "Time-dependent CMB temperature",
            "units" => "K", 
            "description" => "Time-varying CMB temperature boundary condition"
        ))
        
        # Write coordinates
        ds["theta"][:] = theta
        ds["phi"][:] = phi
        ds["time"][:] = time
        
        # Create time-dependent temperature pattern
        temperature = zeros(nlat, nlon, ntime)
        T_base = 4000.0
        
        for (k, t) in enumerate(time)
            for (i, th) in enumerate(theta)
                for (j, ph) in enumerate(phi)
                    # Rotating hot plume pattern
                    phase = 2π * t  # Rotation with time
                    Y11_real = sin(th) * cos(ph + phase)
                    T_pert = 200.0 * (1 + 0.5 * sin(π * t))  # Time-varying amplitude
                    
                    temperature[i, j, k] = T_base + T_pert * Y11_real
                end
            end
        end
        
        # Write temperature data
        temp_var[:] = temperature
        
        ds.attrib["title"] = "Time-dependent CMB Temperature"
        ds.attrib["time_dependent"] = true
    end
end

"""
    create_timedep_temperature_outer_bc(filename, theta, phi, time, nlat, nlon, ntime)

Create time-dependent surface temperature boundary condition.
"""
function create_timedep_temperature_outer_bc(filename, theta, phi, time, nlat, nlon, ntime)
    NCDataset(filename, "c") do ds
        defDim(ds, "lat", nlat)
        defDim(ds, "lon", nlon)
        defDim(ds, "time", ntime)
        
        defVar(ds, "theta", Float64, ("lat",))
        defVar(ds, "phi", Float64, ("lon",))
        defVar(ds, "time", Float64, ("time",))
        
        temp_var = defVar(ds, "temperature", Float64, ("lat", "lon", "time"), attrib=Dict(
            "long_name" => "Time-dependent surface temperature",
            "units" => "K"
        ))
        
        ds["theta"][:] = theta
        ds["phi"][:] = phi
        ds["time"][:] = time
        
        # Time-dependent surface cooling/heating pattern
        temperature = zeros(nlat, nlon, ntime)
        
        for (k, t) in enumerate(time)
            for (i, th) in enumerate(theta)
                for (j, ph) in enumerate(phi)
                    T_base = 300.0
                    # Seasonal-like variations
                    seasonal_factor = 1.0 + 0.1 * sin(4π * t)
                    lat_factor = cos(th)
                    
                    temperature[i, j, k] = T_base * seasonal_factor + 50.0 * lat_factor
                end
            end
        end
        
        temp_var[:] = temperature
        ds.attrib["title"] = "Time-dependent Surface Temperature"
    end
end

"""
    create_sample_composition_boundaries(; nlat=64, nlon=128)

Create sample NetCDF files for compositional boundary conditions.
"""
function create_sample_composition_boundaries(; nlat=64, nlon=128)
    println("Creating sample composition boundary files...")
    
    # Create coordinates
    theta = collect(range(0, π, length=nlat))
    phi = collect(range(0, 2π, length=nlon+1))[1:end-1]
    
    # Inner boundary (CMB) - compositionally light regions
    create_composition_inner_bc("cmb_composition.nc", theta, phi, nlat, nlon)
    
    # Outer boundary (surface) - uniform composition
    create_composition_outer_bc("surface_composition.nc", theta, phi, nlat, nlon)
    
    println("  Composition boundary files created successfully")
end

"""
    create_composition_inner_bc(filename, theta, phi, nlat, nlon)

Create CMB compositional boundary condition.
"""
function create_composition_inner_bc(filename, theta, phi, nlat, nlon)
    NCDataset(filename, "c") do ds
        defDim(ds, "lat", nlat)
        defDim(ds, "lon", nlon)
        
        defVar(ds, "theta", Float64, ("lat",))
        defVar(ds, "phi", Float64, ("lon",))
        
        comp_var = defVar(ds, "composition", Float64, ("lat", "lon"), attrib=Dict(
            "long_name" => "Compositional boundary condition",
            "units" => "dimensionless",
            "description" => "CMB compositional boundary (light element fraction)",
            "valid_range" => [0.0, 1.0]
        ))
        
        ds["theta"][:] = theta
        ds["phi"][:] = phi
        
        # Create compositional pattern: light element release regions
        composition = zeros(nlat, nlon)
        
        for (i, th) in enumerate(theta)
            for (j, ph) in enumerate(phi)
                # Base composition
                base_comp = 0.1
                
                # Add plume-like light element release regions
                # Y₂₀ pattern (equatorial band)
                Y20 = 0.5 * (3*cos(th)^2 - 1)
                comp_pert = 0.3 * Y20
                
                # Y₂₂ pattern for more complexity
                Y22 = sin(th)^2 * cos(2*ph)
                comp_pert += 0.2 * Y22
                
                composition[i, j] = base_comp + comp_pert
                
                # Ensure valid range [0, 1]
                composition[i, j] = max(0.0, min(1.0, composition[i, j]))
            end
        end
        
        comp_var[:] = composition
        
        ds.attrib["title"] = "CMB Compositional Boundary Condition"
        ds.attrib["description"] = "Light element release pattern at CMB"
    end
end

"""
    create_composition_outer_bc(filename, theta, phi, nlat, nlon)

Create surface compositional boundary condition.
"""
function create_composition_outer_bc(filename, theta, phi, nlat, nlon)
    NCDataset(filename, "c") do ds
        defDim(ds, "lat", nlat)
        defDim(ds, "lon", nlon)
        
        defVar(ds, "theta", Float64, ("lat",))
        defVar(ds, "phi", Float64, ("lon",))
        
        comp_var = defVar(ds, "composition", Float64, ("lat", "lon"), attrib=Dict(
            "long_name" => "Surface compositional boundary condition",
            "units" => "dimensionless",
            "description" => "Surface composition (usually uniform)"
        ))
        
        ds["theta"][:] = theta
        ds["phi"][:] = phi
        
        # Uniform surface composition (typically zero for heavy elements)
        composition = zeros(nlat, nlon)
        
        comp_var[:] = composition
        
        ds.attrib["title"] = "Surface Compositional Boundary Condition"
        ds.attrib["description"] = "Uniform surface composition"
    end
end

"""
    main()

Main function to create all sample boundary condition files.
"""
function main()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║              NetCDF Boundary Condition Generator             ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    # Default grid resolution
    nlat, nlon = 64, 128
    
    println("Grid resolution: $nlat × $nlon")
    println()
    
    # Create temperature boundary files
    create_sample_temperature_boundaries(nlat=nlat, nlon=nlon, time_dependent=false)
    create_sample_temperature_boundaries(nlat=nlat, nlon=nlon, time_dependent=true)
    
    println()
    
    # Create composition boundary files  
    create_sample_composition_boundaries(nlat=nlat, nlon=nlon)
    
    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║                    Files Created Successfully                ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║ Temperature boundaries (time-independent):                  ║")
    println("║   • cmb_temp.nc                                             ║")
    println("║   • surface_temp.nc                                         ║")
    println("║                                                              ║")
    println("║ Temperature boundaries (time-dependent):                    ║")
    println("║   • cmb_temp_timedep.nc                                     ║")
    println("║   • surface_temp_timedep.nc                                 ║")
    println("║                                                              ║")
    println("║ Compositional boundaries:                                    ║")
    println("║   • cmb_composition.nc                                      ║")
    println("║   • surface_composition.nc                                  ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    println()
    println("Usage in Geodynamo.jl:")
    println("```julia")
    println("using Geodynamo")
    println()
    println("# Load temperature boundaries")
    println("temp_bc = load_temperature_boundaries(\"cmb_temp.nc\", \"surface_temp.nc\")")
    println("print_boundary_info(temp_bc)")
    println()
    println("# Load composition boundaries") 
    println("comp_bc = load_composition_boundaries(\"cmb_composition.nc\", \"surface_composition.nc\")")
    println("print_boundary_info(comp_bc)")
    println()
    println("# Apply during simulation")
    println("apply_netcdf_temperature_boundaries!(temp_field, temp_bc)")
    println("apply_netcdf_composition_boundaries!(comp_field, comp_bc)")
    println("```")
end

# Run the script if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    using Dates  # For creation timestamps
    main()
end