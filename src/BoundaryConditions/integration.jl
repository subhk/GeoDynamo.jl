# ================================================================================
# Integration with Field Structures
# ================================================================================

using SHTnsKit

"""
    initialize_boundary_conditions!(field, field_type::FieldType, config)

Initialize boundary condition support for a field structure.

Adds the necessary boundary condition fields to existing field structures
without breaking compatibility with existing code.
"""
function initialize_boundary_conditions!(field, field_type::FieldType, config)
    
    # Add boundary condition fields if they don't exist
    if !hasfield(typeof(field), :boundary_condition_set)
        field.boundary_condition_set = nothing
    end
    
    if !hasfield(typeof(field), :boundary_interpolation_cache)
        field.boundary_interpolation_cache = Dict{String, Any}()
    end
    
    if !hasfield(typeof(field), :boundary_time_index)
        field.boundary_time_index = Ref{Int}(1)
    end
    
    # Initialize boundary condition type arrays if needed
    if field_type == TEMPERATURE || field_type == COMPOSITION
        # Scalar fields - initialize boundary type arrays
        if !hasfield(typeof(field), :bc_type_inner)
            nlm = SHTnsKit.get_num_modes(config.lmax)
            field.bc_type_inner = ones(Int, nlm)  # Default to Dirichlet
            field.bc_type_outer = ones(Int, nlm)
        end
        
        if !hasfield(typeof(field), :boundary_values)
            nlm = SHTnsKit.get_num_modes(config.lmax)
            field.boundary_values = zeros(config.T, 2, nlm)  # [inner/outer, modes]
        end
        
    elseif field_type == VELOCITY || field_type == MAGNETIC
        # Vector fields - initialize for toroidal and poloidal components
        if !hasfield(typeof(field), :toroidal)
            throw(ArgumentError("Vector field must have toroidal component"))
        end
        
        if !hasfield(typeof(field), :poloidal)
            throw(ArgumentError("Vector field must have poloidal component"))
        end
        
        # Initialize boundary arrays for toroidal component
        if !hasfield(typeof(field.toroidal), :bc_type_inner)
            nlm = SHTnsKit.get_num_modes(config.lmax)
            field.toroidal.bc_type_inner = ones(Int, nlm)
            field.toroidal.bc_type_outer = ones(Int, nlm)
            field.toroidal.boundary_values = zeros(config.T, 2, nlm)
        end
        
        # Initialize boundary arrays for poloidal component  
        if !hasfield(typeof(field.poloidal), :bc_type_inner)
            nlm = SHTnsKit.get_num_modes(config.lmax)
            field.poloidal.bc_type_inner = ones(Int, nlm)
            field.poloidal.bc_type_outer = ones(Int, nlm)
            field.poloidal.boundary_values = zeros(config.T, 2, nlm)
        end
    end
    
    return field
end

"""
    apply_boundary_conditions!(field, field_type::FieldType, solver_state)

Apply boundary conditions during solver operations.

This function integrates boundary conditions with the timestepping and solving process.
"""
function apply_boundary_conditions!(field, field_type::FieldType, solver_state)
    
    if field.boundary_condition_set === nothing
        return field  # No boundary conditions to apply
    end
    
    # Update time-dependent boundaries if needed
    current_time = get_current_simulation_time(solver_state)
    if field.boundary_condition_set.inner_boundary.is_time_dependent || 
       field.boundary_condition_set.outer_boundary.is_time_dependent
        
        update_time_dependent_boundaries!(field, field_type, current_time)
    end
    
    # Apply boundary conditions based on field type
    if field_type == TEMPERATURE
        apply_temperature_boundary_conditions!(field)
    elseif field_type == COMPOSITION
        apply_composition_boundary_conditions!(field)
    elseif field_type == VELOCITY
        apply_velocity_boundary_conditions!(field)
    elseif field_type == MAGNETIC
        apply_magnetic_boundary_conditions!(field)
    end
    
    return field
end

"""
    get_current_simulation_time(solver_state)

Extract current simulation time from solver state.
"""
function get_current_simulation_time(solver_state)
    
    # Try different possible time sources in solver state
    if hasfield(typeof(solver_state), :time)
        return solver_state.time
    elseif hasfield(typeof(solver_state), :t)
        return solver_state.t
    elseif hasfield(typeof(solver_state), :current_time)
        return solver_state.current_time
    elseif hasfield(typeof(solver_state), :timestep_state)
        ts_state = solver_state.timestep_state
        if hasfield(typeof(ts_state), :time)
            return ts_state.time
        elseif hasfield(typeof(ts_state), :step)
            # Estimate time from step number and dt
            dt = get(ts_state, :dt, 1.0)
            return ts_state.step * dt
        end
    end
    
    # Fallback to zero if no time information found
    return 0.0
end

"""
    validate_field_boundary_compatibility(field, field_type::FieldType, boundary_set::BoundaryConditionSet)

Validate that a field structure is compatible with boundary conditions.
"""
function validate_field_boundary_compatibility(field, field_type::FieldType, boundary_set::BoundaryConditionSet)
    
    errors = String[]
    
    # Check field type matches boundary condition type
    if boundary_set.field_type != field_type
        push!(errors, "Field type mismatch: field=$field_type, boundary=$(boundary_set.field_type)")
    end
    
    # Check grid compatibility
    if hasfield(typeof(field), :config)
        config = field.config
        
        if boundary_set.inner_boundary.nlat != config.nlat
            push!(errors, "Grid size mismatch: inner boundary nlat=$(boundary_set.inner_boundary.nlat), config nlat=$(config.nlat)")
        end
        
        if boundary_set.inner_boundary.nlon != config.nlon
            push!(errors, "Grid size mismatch: inner boundary nlon=$(boundary_set.inner_boundary.nlon), config nlon=$(config.nlon)")
        end
    end
    
    # Field-specific validation
    if field_type == VELOCITY || field_type == MAGNETIC
        # Vector fields must have toroidal and poloidal components
        if !hasfield(typeof(field), :toroidal) || !hasfield(typeof(field), :poloidal)
            push!(errors, "Vector field must have toroidal and poloidal components")
        end
        
        # Check vector component count
        if boundary_set.inner_boundary.ncomponents != 3
            push!(errors, "Vector boundary conditions require 3 components, got $(boundary_set.inner_boundary.ncomponents)")
        end
    elseif field_type == TEMPERATURE || field_type == COMPOSITION
        # Scalar fields
        if boundary_set.inner_boundary.ncomponents != 1
            push!(errors, "Scalar boundary conditions require 1 component, got $(boundary_set.inner_boundary.ncomponents)")
        end
    end
    
    if !isempty(errors)
        error_msg = "Field-boundary compatibility validation failed:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
    
    return true
end

"""
    copy_boundary_conditions!(dest_field, src_field, field_type::FieldType)

Copy boundary conditions from one field to another.
"""
function copy_boundary_conditions!(dest_field, src_field, field_type::FieldType)
    
    if src_field.boundary_condition_set === nothing
        return dest_field
    end
    
    # Copy boundary condition set
    dest_field.boundary_condition_set = src_field.boundary_condition_set
    
    # Copy interpolation cache
    dest_field.boundary_interpolation_cache = deepcopy(src_field.boundary_interpolation_cache)
    
    # Copy time index
    dest_field.boundary_time_index[] = src_field.boundary_time_index[]
    
    # Copy boundary condition arrays
    if field_type == TEMPERATURE || field_type == COMPOSITION
        if hasfield(typeof(src_field), :boundary_values)
            dest_field.boundary_values .= src_field.boundary_values
        end
        
        if hasfield(typeof(src_field), :bc_type_inner)
            dest_field.bc_type_inner .= src_field.bc_type_inner
            dest_field.bc_type_outer .= src_field.bc_type_outer
        end
        
    elseif field_type == VELOCITY || field_type == MAGNETIC
        # Copy toroidal boundary conditions
        if hasfield(typeof(src_field.toroidal), :boundary_values)
            dest_field.toroidal.boundary_values .= src_field.toroidal.boundary_values
            dest_field.toroidal.bc_type_inner .= src_field.toroidal.bc_type_inner
            dest_field.toroidal.bc_type_outer .= src_field.toroidal.bc_type_outer
        end
        
        # Copy poloidal boundary conditions
        if hasfield(typeof(src_field.poloidal), :boundary_values)
            dest_field.poloidal.boundary_values .= src_field.poloidal.boundary_values
            dest_field.poloidal.bc_type_inner .= src_field.poloidal.bc_type_inner
            dest_field.poloidal.bc_type_outer .= src_field.poloidal.bc_type_outer
        end
    end
    
    return dest_field
end

"""
    reset_boundary_conditions!(field, field_type::FieldType)

Reset/clear boundary conditions for a field.
"""
function reset_boundary_conditions!(field, field_type::FieldType)
    
    # Clear boundary condition set
    field.boundary_condition_set = nothing
    
    # Clear interpolation cache
    empty!(field.boundary_interpolation_cache)
    
    # Reset time index
    field.boundary_time_index[] = 1
    
    # Reset boundary arrays to zero
    if field_type == TEMPERATURE || field_type == COMPOSITION
        if hasfield(typeof(field), :boundary_values)
            fill!(field.boundary_values, 0.0)
        end
        
    elseif field_type == VELOCITY || field_type == MAGNETIC
        # Reset toroidal boundary conditions
        if hasfield(typeof(field.toroidal), :boundary_values)
            fill!(field.toroidal.boundary_values, 0.0)
        end
        
        # Reset poloidal boundary conditions
        if hasfield(typeof(field.poloidal), :boundary_values)
            fill!(field.poloidal.boundary_values, 0.0)
        end
    end
    
    return field
end

"""
    get_boundary_condition_summary(field, field_type::FieldType)

Get a summary of the current boundary condition state.
"""
function get_boundary_condition_summary(field, field_type::FieldType)
    
    summary = Dict{String, Any}()
    
    summary["field_type"] = string(field_type)
    summary["has_boundary_conditions"] = field.boundary_condition_set !== nothing
    
    if field.boundary_condition_set !== nothing
        boundary_set = field.boundary_condition_set
        
        summary["boundary_field_name"] = boundary_set.field_name
        summary["creation_time"] = boundary_set.creation_time
        summary["current_time_index"] = field.boundary_time_index[]
        
        # Inner boundary info
        summary["inner_boundary"] = Dict(
            "file_path" => boundary_set.inner_boundary.file_path,
            "is_time_dependent" => boundary_set.inner_boundary.is_time_dependent,
            "ntime" => boundary_set.inner_boundary.ntime,
            "ncomponents" => boundary_set.inner_boundary.ncomponents,
            "units" => boundary_set.inner_boundary.units,
            "description" => boundary_set.inner_boundary.description
        )
        
        # Outer boundary info
        summary["outer_boundary"] = Dict(
            "file_path" => boundary_set.outer_boundary.file_path,
            "is_time_dependent" => boundary_set.outer_boundary.is_time_dependent,
            "ntime" => boundary_set.outer_boundary.ntime,
            "ncomponents" => boundary_set.outer_boundary.ncomponents,
            "units" => boundary_set.outer_boundary.units,
            "description" => boundary_set.outer_boundary.description
        )
        
        # Cache info
        summary["interpolation_cache"] = Dict(
            "inner_cached" => haskey(field.boundary_interpolation_cache, "inner"),
            "outer_cached" => haskey(field.boundary_interpolation_cache, "outer"),
            "cache_size" => length(field.boundary_interpolation_cache)
        )
        
        # Field-specific boundary condition info
        if field_type == TEMPERATURE || field_type == COMPOSITION
            if hasfield(typeof(field), :boundary_values)
                summary["boundary_spectral_coefficients"] = Dict(
                    "inner_nonzero" => count(!iszero, field.boundary_values[1, :]),
                    "outer_nonzero" => count(!iszero, field.boundary_values[2, :]),
                    "total_modes" => size(field.boundary_values, 2)
                )
            end
            
        elseif field_type == VELOCITY || field_type == MAGNETIC
            summary["boundary_spectral_coefficients"] = Dict()
            
            if hasfield(typeof(field.toroidal), :boundary_values)
                summary["boundary_spectral_coefficients"]["toroidal"] = Dict(
                    "inner_nonzero" => count(!iszero, field.toroidal.boundary_values[1, :]),
                    "outer_nonzero" => count(!iszero, field.toroidal.boundary_values[2, :]),
                    "total_modes" => size(field.toroidal.boundary_values, 2)
                )
            end
            
            if hasfield(typeof(field.poloidal), :boundary_values)
                summary["boundary_spectral_coefficients"]["poloidal"] = Dict(
                    "inner_nonzero" => count(!iszero, field.poloidal.boundary_values[1, :]),
                    "outer_nonzero" => count(!iszero, field.poloidal.boundary_values[2, :]),
                    "total_modes" => size(field.poloidal.boundary_values, 2)
                )
            end
        end
    else
        summary["reason"] = "No boundary conditions loaded"
    end
    
    return summary
end

export initialize_boundary_conditions!, apply_boundary_conditions!
export validate_field_boundary_compatibility, copy_boundary_conditions!
export reset_boundary_conditions!, get_boundary_condition_summary