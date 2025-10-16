# ================================================================================
# NetCDF I/O Functionality for Boundary Conditions
# ================================================================================

using NCDatasets
using Statistics

"""
    read_netcdf_boundary_data(filename::String; precision::Type{T}=Float64) where T

Read boundary condition data from a NetCDF file.

Returns a BoundaryData structure containing all boundary information.
"""
function read_netcdf_boundary_data(filename::String; precision::Type{T}=Float64) where T
    if !isfile(filename)
        throw(ArgumentError("Boundary condition file not found: $filename"))
    end
    
    NCDataset(filename, "r") do ds
        # Read coordinate variables
        theta = haskey(ds, "theta") ? Array(ds["theta"]) : nothing
        phi = haskey(ds, "phi") ? Array(ds["phi"]) : nothing
        time_var = haskey(ds, "time") ? Array(ds["time"]) : nothing
        
        # Determine field variable name (look for temperature, velocity, etc.)
        field_vars = ["temperature", "velocity", "magnetic", "composition", "u", "v", "w", "b", "xi"]
        field_name = nothing
        field_data = nothing
        
        for var_name in field_vars
            if haskey(ds, var_name)
                field_name = var_name
                field_data = Array(ds[var_name])
                break
            end
        end
        
        if field_data === nothing
            throw(ArgumentError("No recognized field variable found in $filename"))
        end
        
        # Get metadata
        units = haskey(ds[field_name].attrib, "units") ? ds[field_name].attrib["units"] : ""
        description = haskey(ds[field_name].attrib, "long_name") ? ds[field_name].attrib["long_name"] : field_name
        
        # Convert to requested precision
        field_data = convert(Array{precision}, field_data)
        if theta !== nothing
            theta = convert(Vector{precision}, theta)
        end
        if phi !== nothing
            phi = convert(Vector{precision}, phi)
        end
        if time_var !== nothing
            time_var = convert(Vector{precision}, time_var)
        end
        
        # Create BoundaryData structure
        return create_boundary_data(
            field_data, field_name;
            theta=theta, phi=phi, time=time_var,
            units=units, description=description, file_path=filename
        )
    end
end

"""
    write_netcdf_boundary_data(filename::String, boundary_data::BoundaryData)

Write boundary condition data to a NetCDF file.
"""
function write_netcdf_boundary_data(filename::String, boundary_data::BoundaryData)
    NCDataset(filename, "c") do ds
        # Define dimensions
        defDim(ds, "theta", boundary_data.nlat)
        defDim(ds, "phi", boundary_data.nlon)
        
        if boundary_data.is_time_dependent
            defDim(ds, "time", boundary_data.ntime)
        end
        
        if boundary_data.ncomponents > 1
            defDim(ds, "component", boundary_data.ncomponents)
        end
        
        # Define coordinate variables
        if boundary_data.theta !== nothing
            defVar(ds, "theta", eltype(boundary_data.theta), ("theta",))
            ds["theta"][:] = boundary_data.theta
            ds["theta"].attrib["units"] = "radians"
            ds["theta"].attrib["long_name"] = "colatitude"
        end
        
        if boundary_data.phi !== nothing
            defVar(ds, "phi", eltype(boundary_data.phi), ("phi",))
            ds["phi"][:] = boundary_data.phi
            ds["phi"].attrib["units"] = "radians"
            ds["phi"].attrib["long_name"] = "longitude"
        end
        
        if boundary_data.time !== nothing
            defVar(ds, "time", eltype(boundary_data.time), ("time",))
            ds["time"][:] = boundary_data.time
            ds["time"].attrib["units"] = "dimensionless"
            ds["time"].attrib["long_name"] = "time"
        end
        
        # Define field variable
        field_dims = ("theta", "phi")
        if boundary_data.is_time_dependent
            field_dims = (field_dims..., "time")
        end
        if boundary_data.ncomponents > 1
            field_dims = (field_dims..., "component")
        end
        
        defVar(ds, boundary_data.field_type, eltype(boundary_data.values), field_dims)
        ds[boundary_data.field_type][:] = boundary_data.values
        ds[boundary_data.field_type].attrib["units"] = boundary_data.units
        ds[boundary_data.field_type].attrib["long_name"] = boundary_data.description
        
        # Global attributes
        ds.attrib["title"] = "Boundary condition data for $(boundary_data.field_type)"
        ds.attrib["created_by"] = "GeoDynamo.jl BoundaryConditions module"
        ds.attrib["creation_time"] = string(now())
    end
end

"""
    validate_netcdf_boundary_file(filename::String, required_vars::Vector{String}=[])

Validate that a NetCDF boundary condition file has the required structure.
"""
function validate_netcdf_boundary_file(filename::String, required_vars::Vector{String}=String[])
    if !isfile(filename)
        throw(ArgumentError("File not found: $filename"))
    end
    
    errors = String[]
    
    NCDataset(filename, "r") do ds
        # Check for coordinate variables
        if !haskey(ds, "theta")
            push!(errors, "Missing required variable: theta")
        end
        
        if !haskey(ds, "phi")
            push!(errors, "Missing required variable: phi")
        end
        
        # Check for field variables
        field_vars = ["temperature", "velocity", "magnetic", "composition", "u", "v", "w", "b", "xi"]
        has_field = any(var -> haskey(ds, var), field_vars)
        
        if !has_field
            push!(errors, "No recognized field variable found. Expected one of: $(join(field_vars, ", "))")
        end
        
        # Check additional required variables
        for var in required_vars
            if !haskey(ds, var)
                push!(errors, "Missing required variable: $var")
            end
        end
        
        # Check coordinate ranges
        if haskey(ds, "theta")
            theta = Array(ds["theta"])
            if minimum(theta) < 0.0 || maximum(theta) > π
                push!(errors, "Theta coordinates out of range [0, π]: [$(minimum(theta)), $(maximum(theta))]")
            end
        end
        
        if haskey(ds, "phi")
            phi = Array(ds["phi"])
            if minimum(phi) < 0.0 || maximum(phi) >= 2π
                push!(errors, "Phi coordinates out of range [0, 2π): [$(minimum(phi)), $(maximum(phi))]")
            end
        end
    end
    
    if !isempty(errors)
        error_msg = "NetCDF validation failed for $filename:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
    
    return true
end

"""
    get_netcdf_file_info(filename::String)

Get information about a NetCDF boundary condition file.
"""
function get_netcdf_file_info(filename::String)
    if !isfile(filename)
        throw(ArgumentError("File not found: $filename"))
    end
    
    info = Dict{String, Any}()
    
    NCDataset(filename, "r") do ds
        info["filename"] = filename
        info["variables"] = collect(keys(ds))
        info["dimensions"] = Dict(name => size(ds.dim[name]) for name in keys(ds.dim))
        
        # Get coordinate info
        if haskey(ds, "theta")
            theta = Array(ds["theta"])
            info["theta_range"] = [minimum(theta), maximum(theta)]
            info["nlat"] = length(theta)
        end
        
        if haskey(ds, "phi")
            phi = Array(ds["phi"])
            info["phi_range"] = [minimum(phi), maximum(phi)]
            info["nlon"] = length(phi)
        end
        
        if haskey(ds, "time")
            time_var = Array(ds["time"])
            info["time_range"] = [minimum(time_var), maximum(time_var)]
            info["ntime"] = length(time_var)
            info["is_time_dependent"] = true
        else
            info["is_time_dependent"] = false
            info["ntime"] = 1
        end
        
        # Get field info
        field_vars = ["temperature", "velocity", "magnetic", "composition", "u", "v", "w", "b", "xi"]
        for var_name in field_vars
            if haskey(ds, var_name)
                field_data = ds[var_name]
                info["field_variable"] = var_name
                info["field_shape"] = size(field_data)
                info["field_type"] = eltype(field_data)
                
                if haskey(field_data.attrib, "units")
                    info["field_units"] = field_data.attrib["units"]
                end
                
                if haskey(field_data.attrib, "long_name")
                    info["field_description"] = field_data.attrib["long_name"]
                end
                
                break
            end
        end
        
        # Global attributes
        info["global_attributes"] = Dict(name => ds.attrib[name] for name in keys(ds.attrib))
    end
    
    return info
end

export read_netcdf_boundary_data, write_netcdf_boundary_data
export validate_netcdf_boundary_file, get_netcdf_file_info