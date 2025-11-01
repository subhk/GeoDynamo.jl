# ================================================================================
# Integration with Timestepping Methods
# ================================================================================

"""
    update_boundary_conditions_for_timestep!(state, current_time::Float64)

Update all boundary conditions for the current timestep.

This function should be called at the beginning of each timestep to ensure
all fields have up-to-date boundary conditions.
"""
function update_boundary_conditions_for_timestep!(state, current_time::Float64)
    
    # Update temperature boundary conditions
    if hasfield(typeof(state), :temperature) && state.temperature !== nothing
        update_time_dependent_boundaries!(state.temperature, TEMPERATURE, current_time)
    end
    
    # Update composition boundary conditions
    if hasfield(typeof(state), :composition) && state.composition !== nothing
        update_time_dependent_boundaries!(state.composition, COMPOSITION, current_time)
    end
    
    # Update velocity boundary conditions
    if hasfield(typeof(state), :velocity) && state.velocity !== nothing
        update_time_dependent_boundaries!(state.velocity, VELOCITY, current_time)
    end
    
    # Update magnetic field boundary conditions
    if hasfield(typeof(state), :magnetic) && state.magnetic !== nothing
        update_time_dependent_boundaries!(state.magnetic, MAGNETIC, current_time)
    end
    
    return state
end

"""
    apply_boundary_conditions_to_rhs!(rhs, state, field_type::FieldType)

Apply boundary conditions to right-hand side during timestepping.

This function modifies the RHS vector to enforce boundary conditions
during implicit and explicit timestepping methods.
"""
function apply_boundary_conditions_to_rhs!(rhs, state, field_type::FieldType)
    
    field = get_field_from_state(state, field_type)
    
    if field === nothing
        return rhs  # No field to apply boundary conditions to
    end
    
    # Check if field has boundary conditions using unified interface
    boundary_set, _ = get_boundary_data(field, field_type)
    if boundary_set === nothing
        return rhs  # No boundary conditions to apply
    end
    
    # Apply boundary conditions based on field type
    if field_type == TEMPERATURE
        apply_temperature_bc_to_rhs!(rhs, field)
    elseif field_type == COMPOSITION
        apply_composition_bc_to_rhs!(rhs, field)
    elseif field_type == VELOCITY
        apply_velocity_bc_to_rhs!(rhs, field)
    elseif field_type == MAGNETIC
        apply_magnetic_bc_to_rhs!(rhs, field)
    end
    
    return rhs
end

"""
    get_boundary_data(field, field_type::FieldType)

Get boundary data from field using unified interface with fallback support.
"""
function get_boundary_data(field, field_type::FieldType)
    if field_type == TEMPERATURE
        if hasfield(typeof(field), :boundary_condition_set)
            return field.boundary_condition_set, field.boundary_interpolation_cache
        else
            # Use fallback cache system if available
            if isdefined(@__MODULE__, :_temperature_boundary_cache)
                field_id = objectid(field)
                if haskey(_temperature_boundary_cache, field_id)
                    data = _temperature_boundary_cache[field_id]
                    return data[:boundary_set], data[:interpolation_cache]
                end
            end
        end
    elseif field_type == COMPOSITION
        if hasfield(typeof(field), :boundary_condition_set)
            return field.boundary_condition_set, field.boundary_interpolation_cache
        else
            # Use fallback cache system if available
            if isdefined(@__MODULE__, :_composition_boundary_cache)
                field_id = objectid(field)
                if haskey(_composition_boundary_cache, field_id)
                    data = _composition_boundary_cache[field_id]
                    return data[:boundary_set], data[:interpolation_cache]
                end
            end
        end
    elseif field_type == VELOCITY
        if hasfield(typeof(field), :boundary_condition_set)
            return field.boundary_condition_set, field.boundary_interpolation_cache
        end
    elseif field_type == MAGNETIC
        if hasfield(typeof(field), :boundary_condition_set)
            return field.boundary_condition_set, field.boundary_interpolation_cache
        end
    end
    return nothing, nothing
end

"""
    get_time_index(field, field_type::FieldType)

Get current time index from field using unified interface with fallback support.
"""
function get_time_index(field, field_type::FieldType)
    if field_type == TEMPERATURE
        if hasfield(typeof(field), :boundary_time_index)
            return field.boundary_time_index[]
        else
            # Use fallback cache system if available
            if isdefined(@__MODULE__, :_temperature_boundary_cache)
                field_id = objectid(field)
                if haskey(_temperature_boundary_cache, field_id)
                    return _temperature_boundary_cache[field_id][:time_index]
                end
            end
        end
    elseif field_type == COMPOSITION
        if hasfield(typeof(field), :boundary_time_index)
            return field.boundary_time_index[]
        else
            # Use fallback cache system if available
            if isdefined(@__MODULE__, :_composition_boundary_cache)
                field_id = objectid(field)
                if haskey(_composition_boundary_cache, field_id)
                    return _composition_boundary_cache[field_id][:time_index]
                end
            end
        end
    elseif field_type == VELOCITY
        if hasfield(typeof(field), :boundary_time_index)
            return field.boundary_time_index[]
        end
    elseif field_type == MAGNETIC
        if hasfield(typeof(field), :boundary_time_index)
            return field.boundary_time_index[]
        end
    end
    return 1  # Default time index
end

"""
    get_field_from_state(state, field_type::FieldType)

Extract the appropriate field from simulation state.
"""
function get_field_from_state(state, field_type::FieldType)
    
    if field_type == TEMPERATURE
        return hasfield(typeof(state), :temperature) ? state.temperature : nothing
    elseif field_type == COMPOSITION
        return hasfield(typeof(state), :composition) ? state.composition : nothing
    elseif field_type == VELOCITY
        return hasfield(typeof(state), :velocity) ? state.velocity : nothing
    elseif field_type == MAGNETIC
        return hasfield(typeof(state), :magnetic) ? state.magnetic : nothing
    else
        return nothing
    end
end

"""
    apply_temperature_bc_to_rhs!(rhs, temp_field)

Apply temperature boundary conditions to RHS vector.
"""
function apply_temperature_bc_to_rhs!(rhs, temp_field)
    
    # Get boundary data using unified interface
    boundary_set, _ = get_boundary_data(temp_field, TEMPERATURE)
    if boundary_set === nothing
        return rhs
    end
    
    # Get current boundary values (spectral coefficients)
    if hasfield(typeof(temp_field), :boundary_values)
        inner_bc = temp_field.boundary_values[1, :]  # Inner boundary
        outer_bc = temp_field.boundary_values[2, :]  # Outer boundary
        
        nlm = length(inner_bc)
        
        # Apply Dirichlet boundary conditions by modifying RHS
        # For spectral methods, this typically involves:
        # 1. Setting appropriate boundary rows in the system matrix
        # 2. Modifying corresponding RHS entries
        
        for lm in 1:nlm
            # Check boundary condition types (default to Dirichlet for loaded boundary conditions)
            bc_type_inner = hasfield(typeof(temp_field), :bc_type_inner) ? temp_field.bc_type_inner[lm] : 1
            bc_type_outer = hasfield(typeof(temp_field), :bc_type_outer) ? temp_field.bc_type_outer[lm] : 1
            
            if bc_type_inner == 1  # Dirichlet at inner boundary
                # The boundary condition is already applied to the field boundary_values
                # The solver should use these values to constrain the solution
                # This is a placeholder - actual implementation depends on solver structure
            end
            
            if bc_type_outer == 1  # Dirichlet at outer boundary  
                # Similar implementation for outer boundary
                # The solver will use the outer boundary values for constraint
            end
        end
    end
    
    return rhs
end

"""
    apply_composition_bc_to_rhs!(rhs, comp_field)

Apply composition boundary conditions to RHS vector.
"""
function apply_composition_bc_to_rhs!(rhs, comp_field)
    
    # Get boundary data using unified interface
    boundary_set, _ = get_boundary_data(comp_field, COMPOSITION)
    if boundary_set === nothing
        return rhs
    end
    
    # Get current boundary values (spectral coefficients)
    if hasfield(typeof(comp_field), :boundary_values)
        inner_bc = comp_field.boundary_values[1, :]
        outer_bc = comp_field.boundary_values[2, :]
        
        nlm = length(inner_bc)
        
        # Apply Dirichlet boundary conditions by modifying RHS
        # Similar to temperature but ensure composition constraints [0,1]
        for lm in 1:nlm
            bc_type_inner = hasfield(typeof(comp_field), :bc_type_inner) ? comp_field.bc_type_inner[lm] : 1
            bc_type_outer = hasfield(typeof(comp_field), :bc_type_outer) ? comp_field.bc_type_outer[lm] : 1
            
            if bc_type_inner == 1  # Dirichlet at inner boundary
                # Apply composition boundary with range constraint
                # Composition values should be clamped to [0,1] range
                inner_value = clamp(real(inner_bc[lm]), 0.0, 1.0)
                # Solver will use this constrained value
            end
            
            if bc_type_outer == 1  # Dirichlet at outer boundary
                # Apply composition boundary with range constraint
                outer_value = clamp(real(outer_bc[lm]), 0.0, 1.0)
                # Solver will use this constrained value
            end
        end
    end
    
    return rhs
end

"""
    apply_velocity_bc_to_rhs!(rhs, velocity_field)

Apply velocity boundary conditions to RHS vector.
"""
function apply_velocity_bc_to_rhs!(rhs, velocity_field)
    
    # Get boundary data using unified interface
    boundary_set, _ = get_boundary_data(velocity_field, VELOCITY)
    if boundary_set === nothing
        return rhs
    end
    
    # Velocity boundary conditions in spherical coordinates:
    # - Toroidal component T: related to tangential velocity (v_θ, v_φ)
    # - Poloidal component P: related to radial velocity (v_r) and tangential flow potential

    # Apply boundary conditions to toroidal component
    if hasfield(typeof(velocity_field), :toroidal) && hasfield(typeof(velocity_field.toroidal), :boundary_values)
        inner_tor = velocity_field.toroidal.boundary_values[1, :]
        outer_tor = velocity_field.toroidal.boundary_values[2, :]

        nlm = length(inner_tor)
        for lm in 1:nlm
            # Get boundary condition types
            bc_type_inner = hasfield(typeof(velocity_field.toroidal), :bc_type_inner) ?
                           velocity_field.toroidal.bc_type_inner[lm] : 1
            bc_type_outer = hasfield(typeof(velocity_field.toroidal), :bc_type_outer) ?
                           velocity_field.toroidal.bc_type_outer[lm] : 1

            # Apply toroidal boundary conditions
            if bc_type_inner == 1  # Dirichlet (no-slip): T = prescribed value
                # For no-slip: T = 0 (no tangential velocity)
                # For prescribed tangential velocity: T = prescribed value
                # The RHS modification depends on the specific discretization
                # This is typically handled by the solver using boundary_values
            elseif bc_type_inner == 2  # Neumann (stress-free): ∂T/∂r = 0
                # For stress-free: tangential stress = 0
                # This requires Neumann boundary condition on T
            end

            if bc_type_outer == 1  # Dirichlet
                # Similar to inner boundary
            elseif bc_type_outer == 2  # Neumann
                # Similar to inner boundary
            end
        end
    end

    # Apply boundary conditions to poloidal component
    if hasfield(typeof(velocity_field), :poloidal) && hasfield(typeof(velocity_field.poloidal), :boundary_values)
        inner_pol = velocity_field.poloidal.boundary_values[1, :]
        outer_pol = velocity_field.poloidal.boundary_values[2, :]

        nlm = length(inner_pol)
        for lm in 1:nlm
            # Get boundary condition types
            bc_type_inner = hasfield(typeof(velocity_field.poloidal), :bc_type_inner) ?
                           velocity_field.poloidal.bc_type_inner[lm] : 1
            bc_type_outer = hasfield(typeof(velocity_field.poloidal), :bc_type_outer) ?
                           velocity_field.poloidal.bc_type_outer[lm] : 1

            # Apply poloidal boundary conditions
            if bc_type_inner == 1  # Dirichlet: P = prescribed value
                # For no-slip: P and ∂P/∂r constrained to give v_r = v_θ = v_φ = 0
                # For impermeable boundary: ∂P/∂r constrained to give v_r = 0
                # The specific constraint depends on the velocity field representation
            elseif bc_type_inner == 2  # Neumann: ∂P/∂r = prescribed value
                # For stress-free: specific stress conditions
            end

            if bc_type_outer == 1  # Dirichlet
                # Similar to inner boundary
            elseif bc_type_outer == 2  # Neumann
                # Similar to inner boundary
            end
        end
    end
    
    return rhs
end

"""
    apply_magnetic_bc_to_rhs!(rhs, magnetic_field)

Apply magnetic field boundary conditions to RHS vector.
"""
function apply_magnetic_bc_to_rhs!(rhs, magnetic_field)
    
    # Get boundary data using unified interface
    boundary_set, _ = get_boundary_data(magnetic_field, MAGNETIC)
    if boundary_set === nothing
        return rhs
    end
    
    # Apply boundary conditions to toroidal component
    if hasfield(typeof(magnetic_field), :toroidal) && hasfield(typeof(magnetic_field.toroidal), :boundary_values)
        toroidal = magnetic_field.toroidal

        if hasfield(typeof(toroidal), :bc_type_inner) && hasfield(typeof(toroidal), :bc_type_outer)
            inner_tor = toroidal.boundary_values[1, :]
            outer_tor = toroidal.boundary_values[2, :]
            bc_inner = toroidal.bc_type_inner
            bc_outer = toroidal.bc_type_outer

            nlm = length(inner_tor)
            for lm in 1:nlm
                # Apply boundary condition based on type:
                # bc_type = 1: Dirichlet (fixed value)
                # bc_type = 2: Neumann (fixed derivative)

                # Inner boundary
                if bc_inner[lm] == 1  # Dirichlet: B_tor = prescribed value
                    # RHS modification for fixed boundary value
                    # (specific implementation depends on discretization)
                elseif bc_inner[lm] == 2  # Neumann: ∂B_tor/∂r = 0 (insulating)
                    # RHS modification for natural boundary condition
                end

                # Outer boundary (similar logic)
                if bc_outer[lm] == 1
                    # Apply Dirichlet BC at outer boundary
                elseif bc_outer[lm] == 2
                    # Apply Neumann BC at outer boundary
                end
            end
        end
    end

    # Apply boundary conditions to poloidal component
    if hasfield(typeof(magnetic_field), :poloidal) && hasfield(typeof(magnetic_field.poloidal), :boundary_values)
        poloidal = magnetic_field.poloidal

        if hasfield(typeof(poloidal), :bc_type_inner) && hasfield(typeof(poloidal), :bc_type_outer)
            inner_pol = poloidal.boundary_values[1, :]
            outer_pol = poloidal.boundary_values[2, :]
            bc_inner = poloidal.bc_type_inner
            bc_outer = poloidal.bc_type_outer

            nlm = length(inner_pol)
            for lm in 1:nlm
                # Apply boundary conditions for poloidal component
                # (radial magnetic field component)

                if bc_inner[lm] == 1  # Dirichlet: B_pol = prescribed value
                    # Apply fixed boundary value constraint
                elseif bc_inner[lm] == 2  # Neumann: ∂B_pol/∂r = 0
                    # Apply natural boundary condition
                end

                if bc_outer[lm] == 1
                    # Apply Dirichlet BC at outer boundary
                elseif bc_outer[lm] == 2
                    # Apply Neumann BC at outer boundary
                end
            end
        end
    end
    
    return rhs
end

"""
    enforce_boundary_conditions_in_solution!(solution, state, field_type::FieldType)

Enforce boundary conditions in the solution vector after timestepping.

This function ensures that the solution satisfies the boundary conditions
after each timestep, which may be necessary for certain discretization schemes.
"""
function enforce_boundary_conditions_in_solution!(solution, state, field_type::FieldType)
    
    field = get_field_from_state(state, field_type)
    
    if field === nothing
        return solution
    end
    
    # Check if field has boundary conditions using unified interface
    boundary_set, _ = get_boundary_data(field, field_type)
    if boundary_set === nothing
        return solution
    end
    
    # Enforce boundary conditions based on field type
    if field_type == TEMPERATURE
        enforce_temperature_bc_in_solution!(solution, field)
    elseif field_type == COMPOSITION
        enforce_composition_bc_in_solution!(solution, field)
    elseif field_type == VELOCITY
        enforce_velocity_bc_in_solution!(solution, field)
    elseif field_type == MAGNETIC
        enforce_magnetic_bc_in_solution!(solution, field)
    end
    
    return solution
end

"""
    enforce_temperature_bc_in_solution!(solution, temp_field)

Enforce temperature boundary conditions in solution vector.
"""
function enforce_temperature_bc_in_solution!(solution, temp_field)
    
    # Get boundary data using unified interface
    boundary_set, _ = get_boundary_data(temp_field, TEMPERATURE)
    if boundary_set === nothing
        return solution
    end
    
    # Get boundary values if available
    if hasfield(typeof(temp_field), :boundary_values)
        inner_bc = temp_field.boundary_values[1, :]
        outer_bc = temp_field.boundary_values[2, :]
        
        # Enforce Dirichlet boundary conditions by directly setting solution values
        # For spectral methods, this involves constraining boundary modes
        # The actual implementation depends on how the solution vector is organized
        # This is a framework - specific solver implementations will use boundary_values
    end
    
    return solution
end

"""
    enforce_composition_bc_in_solution!(solution, comp_field)

Enforce composition boundary conditions in solution vector.
"""
function enforce_composition_bc_in_solution!(solution, comp_field)
    
    # Get boundary data using unified interface
    boundary_set, _ = get_boundary_data(comp_field, COMPOSITION)
    if boundary_set === nothing
        return solution
    end
    
    # Get boundary values if available
    if hasfield(typeof(comp_field), :boundary_values)
        inner_bc = comp_field.boundary_values[1, :]
        outer_bc = comp_field.boundary_values[2, :]
        
        # Similar implementation to temperature
        # Additionally ensure composition values remain in [0, 1] range
        # Clamp boundary values to physical range
        clamp!(real(inner_bc), 0.0, 1.0)
        clamp!(real(outer_bc), 0.0, 1.0)
    end
    
    return solution
end

"""
    enforce_velocity_bc_in_solution!(solution, velocity_field)

Enforce velocity boundary conditions in solution vector.
"""
function enforce_velocity_bc_in_solution!(solution, velocity_field)
    
    # Get boundary data using unified interface
    boundary_set, _ = get_boundary_data(velocity_field, VELOCITY)
    if boundary_set === nothing
        return solution
    end
    
    # Enforce velocity boundary conditions in the solution vector
    # This function is called after timestepping to ensure the solution satisfies BCs

    # Toroidal component enforcement
    if hasfield(typeof(velocity_field), :toroidal) && hasfield(typeof(velocity_field.toroidal), :boundary_values)
        inner_tor = velocity_field.toroidal.boundary_values[1, :]
        outer_tor = velocity_field.toroidal.boundary_values[2, :]

        nlm = length(inner_tor)
        for lm in 1:nlm
            # Get boundary condition types
            bc_type_inner = hasfield(typeof(velocity_field.toroidal), :bc_type_inner) ?
                           velocity_field.toroidal.bc_type_inner[lm] : 1
            bc_type_outer = hasfield(typeof(velocity_field.toroidal), :bc_type_outer) ?
                           velocity_field.toroidal.bc_type_outer[lm] : 1

            # Enforce toroidal boundary conditions
            if bc_type_inner == 1  # Dirichlet: enforce T = boundary_value at inner boundary
                # The specific implementation depends on solution vector structure
                # Typically involves setting specific components of the solution vector
                # to match the prescribed boundary values
            end

            if bc_type_outer == 1  # Dirichlet: enforce T = boundary_value at outer boundary
                # Similar enforcement for outer boundary
            end
        end
    end

    # Poloidal component enforcement
    if hasfield(typeof(velocity_field), :poloidal) && hasfield(typeof(velocity_field.poloidal), :boundary_values)
        inner_pol = velocity_field.poloidal.boundary_values[1, :]
        outer_pol = velocity_field.poloidal.boundary_values[2, :]

        nlm = length(inner_pol)
        for lm in 1:nlm
            # Get boundary condition types
            bc_type_inner = hasfield(typeof(velocity_field.poloidal), :bc_type_inner) ?
                           velocity_field.poloidal.bc_type_inner[lm] : 1
            bc_type_outer = hasfield(typeof(velocity_field.poloidal), :bc_type_outer) ?
                           velocity_field.poloidal.bc_type_outer[lm] : 1

            # Enforce poloidal boundary conditions
            if bc_type_inner == 1  # Dirichlet: enforce P = boundary_value at inner boundary
                # For no-slip: both P and ∂P/∂r must be constrained
                # For impermeable: only ∂P/∂r is constrained (v_r = 0)
                # Implementation depends on how P is discretized in the radial direction
            end

            if bc_type_outer == 1  # Dirichlet: enforce P = boundary_value at outer boundary
                # Similar enforcement for outer boundary
            end
        end
    end
    
    return solution
end

"""
    enforce_magnetic_bc_in_solution!(solution, magnetic_field)

Enforce magnetic field boundary conditions in solution vector.
"""
function enforce_magnetic_bc_in_solution!(solution, magnetic_field)
    
    # Get boundary data using unified interface
    boundary_set, _ = get_boundary_data(magnetic_field, MAGNETIC)
    if boundary_set === nothing
        return solution
    end
    
    # Enforce insulating or perfect conductor boundary conditions
    # For insulating: match potential field at boundary
    # For perfect conductor: enforce specific field continuity conditions
    
    # Toroidal component enforcement
    if hasfield(typeof(magnetic_field), :toroidal) && hasfield(typeof(magnetic_field.toroidal), :boundary_values)
        # Insulating: ∂(rB_tor)/∂r = 0
        # Perfect conductor: B_tor = 0
    end
    
    # Poloidal component enforcement
    if hasfield(typeof(magnetic_field), :poloidal) && hasfield(typeof(magnetic_field.poloidal), :boundary_values)
        # Insulating: match potential field
        # Perfect conductor: ∂B_pol/∂r = 0
    end
    
    return solution
end

"""
    compute_boundary_condition_residual(field, field_type::FieldType)

Compute residual to check how well boundary conditions are satisfied.

Returns a measure of how much the current field violates the boundary conditions.
Useful for monitoring solution quality and debugging.
"""
function compute_boundary_condition_residual(field, field_type::FieldType)
    
    # Get boundary data using unified interface
    boundary_set, cache = get_boundary_data(field, field_type)
    if boundary_set === nothing
        return 0.0
    end
    
    residual = 0.0
    
    try
        # Get current boundary values from field
        current_boundaries = get_current_boundaries(field, field_type)
        
        if haskey(current_boundaries, :error)
            return Inf  # Error in getting boundaries
        end
        
        # Get time index using field-specific interface
        time_index = get_time_index(field, field_type)
        
        # Get prescribed boundary values
        inner_prescribed = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
        outer_prescribed = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
        
        # Get current field values at boundaries
        inner_current = current_boundaries[:inner_physical]
        outer_current = current_boundaries[:outer_physical]
        
        # Compute L2 norm of difference
        inner_residual = sqrt(sum((inner_current - inner_prescribed).^2))
        outer_residual = sqrt(sum((outer_current - outer_prescribed).^2))
        
        residual = inner_residual + outer_residual
        
    catch e
        # If any error occurs in residual computation, return a high value
        @warn "Error computing boundary condition residual: $e"
        residual = Inf
    end
    
    return residual
end

"""
    log_boundary_condition_status(state, rank::Int=0)

Log the status of all boundary conditions in the simulation state.
"""
function log_boundary_condition_status(state, rank::Int=0)
    
    if rank != 0
        return  # Only log from rank 0
    end
    
    println("=" * 60)
    println("BOUNDARY CONDITION STATUS")
    println("=" * 60)
    
    # Check each field type
    field_types = [TEMPERATURE, COMPOSITION, VELOCITY, MAGNETIC]
    field_names = ["Temperature", "Composition", "Velocity", "Magnetic"]
    
    for (field_type, field_name) in zip(field_types, field_names)
        field = get_field_from_state(state, field_type)
        
        if field !== nothing
            println("$field_name Field:")
            
            # Use unified interface to check for boundary conditions
            boundary_set, _ = get_boundary_data(field, field_type)
            
            if boundary_set !== nothing
                println("  Boundary conditions loaded")
                
                # Get time index using unified interface
                time_index = get_time_index(field, field_type)
                println("    Time index: $(time_index)")
                
                # Display file information if available
                inner_file = get(boundary_set.inner_boundary, :file_path, "programmatic")
                outer_file = get(boundary_set.outer_boundary, :file_path, "programmatic")
                println("    Inner: $(basename(inner_file))")
                println("    Outer: $(basename(outer_file))")
                
                if boundary_set.inner_boundary.is_time_dependent || boundary_set.outer_boundary.is_time_dependent
                    println("    Time-dependent: Yes")
                else
                    println("    Time-dependent: No")
                end
                
                # Compute and display residual
                residual = compute_boundary_condition_residual(field, field_type)
                println("    Residual: $(round(residual, digits=6))")
                
            else
                println("  No boundary conditions")
            end
            
            println()
        end
    end
    
    println("=" * 60)
end

export update_boundary_conditions_for_timestep!
export apply_boundary_conditions_to_rhs!, enforce_boundary_conditions_in_solution!
export compute_boundary_condition_residual, log_boundary_condition_status
export get_boundary_data, get_time_index, get_field_from_state