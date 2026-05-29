# ================================================================================
# Integration with Field Structures
# ================================================================================

# Note: This file is included within the bcs module
# SHTnsKit is imported at the module level

"""
    initialize_boundary_conditions!(𝔽, field_type::FieldType)

Initialize boundary condition support for a field structure.

IMPORTANT: The field structure must already have the following fields defined:
- boundary_condition_set (can be nothing)
- boundary_interpolation_cache (BoundaryInterpolationCache)
- boundary_time_index (Ref{Int})

For scalar fields (TEMPERATURE, COMPOSITION):
- bc_type_inner, bc_type_outer (Vector{Int})
- boundary_values (Matrix)

For vector fields (VELOCITY, MAGNETIC):
- toroidal and poloidal components, each with bc_type_inner, bc_type_outer, boundary_values

Note: The field struct must be a mutable struct since this function modifies its fields.
"""
function initialize_boundary_conditions!(𝔽, field_type::FieldType)
    # Validate required boundary condition fields exist
    # Note: In Julia, struct fields cannot be added at runtime - they must be pre-defined
    required_fields = [:boundary_condition_set, :boundary_interpolation_cache, :boundary_time_index]

    for field_name in required_fields
        if !hasfield(typeof(𝔽), field_name)
            throw(ArgumentError("Field structure missing required field: $field_name. " *
                "Ensure your field struct includes boundary condition fields."))
        end
    end

    # Reset time index to initial value
    𝔽.boundary_time_index[] = 1

    # Initialization is split by scalar vs vector storage layout because the
    # runtime keeps scalar boundary values in one matrix and vector values on
    # toroidal/poloidal subfields.
    # Initialize boundary condition type arrays
    if field_type == TEMPERATURE || field_type == COMPOSITION
        # Scalar fields - validate and initialize boundary type arrays
        if !hasfield(typeof(𝔽), :bc_type_inner) || !hasfield(typeof(𝔽), :bc_type_outer)
            throw(ArgumentError("Scalar field must have bc_type_inner and bc_type_outer fields"))
        end

        if !hasfield(typeof(𝔽), :boundary_values)
            throw(ArgumentError("Scalar field must have boundary_values field"))
        end

        # Initialize arrays to default values
        fill!(𝔽.bc_type_inner, Int(DIRICHLET))
        fill!(𝔽.bc_type_outer, Int(DIRICHLET))
        fill!(𝔽.boundary_values, zero(eltype(𝔽.boundary_values)))

    elseif field_type == VELOCITY || field_type == MAGNETIC
        # Vector fields - validate toroidal and poloidal components
        if !hasfield(typeof(𝔽), :toroidal)
            throw(ArgumentError("Vector field must have toroidal component"))
        end

        if !hasfield(typeof(𝔽), :poloidal)
            throw(ArgumentError("Vector field must have poloidal component"))
        end

        # Validate and initialize toroidal component
        if !hasfield(typeof(𝔽.toroidal), :bc_type_inner)
            throw(ArgumentError("Toroidal component must have bc_type_inner, bc_type_outer, boundary_values fields"))
        end

        fill!(𝔽.toroidal.bc_type_inner, Int(DIRICHLET))
        fill!(𝔽.toroidal.bc_type_outer, Int(DIRICHLET))
        fill!(𝔽.toroidal.boundary_values, zero(eltype(𝔽.toroidal.boundary_values)))

        # Validate and initialize poloidal component
        if !hasfield(typeof(𝔽.poloidal), :bc_type_inner)
            throw(ArgumentError("Poloidal component must have bc_type_inner, bc_type_outer, boundary_values fields"))
        end

        fill!(𝔽.poloidal.bc_type_inner, Int(DIRICHLET))
        fill!(𝔽.poloidal.bc_type_outer, Int(DIRICHLET))
        fill!(𝔽.poloidal.boundary_values, zero(eltype(𝔽.poloidal.boundary_values)))
    end

    return 𝔽
end

"""
    apply_boundary_conditions!(𝔽, field_type::FieldType, solver_state)

Apply boundary conditions during solver operations.

This function integrates boundary conditions with the timestepping and solving process.
"""
function apply_boundary_conditions!(𝔽, field_type::FieldType, solver_state)
    # Check if field has boundary condition support
    if !hasfield(typeof(𝔽), :boundary_condition_set)
        return 𝔽  # Field doesn't support boundary conditions
    end

    if 𝔽.boundary_condition_set === nothing
        return 𝔽  # No boundary conditions to apply
    end

    # Update time-dependent boundaries if needed
    current_time = get_current_simulation_time(solver_state)
    if 𝔽.boundary_condition_set.inner_boundary.is_time_dependent ||
       𝔽.boundary_condition_set.outer_boundary.is_time_dependent

        update_time_dependent_boundaries!(𝔽, field_type, current_time)
    end

    # This is the integration hook between boundary metadata and the solver.
    # For the current discretization, most work is delegated to implicit matrix
    # assembly, but the boundary lifecycle still passes through this function.
    # Apply boundary conditions based on field type
    if field_type == TEMPERATURE
        # Temperature BCs are embedded in the implicit matrix system (see bcs/thermal_bc.jl)
        # No post-processing needed here
    elseif field_type == COMPOSITION
        # Composition BCs are embedded in the implicit matrix system (see bcs/compositional_bc.jl)
        # No post-processing needed here
    elseif field_type == VELOCITY
        # Velocity BCs are embedded in the implicit matrix system (see bcs/velocity_bc.jl)
        # No post-processing needed here
    elseif field_type == MAGNETIC
        # Magnetic BCs are embedded in the implicit matrix system (see bcs/magnetic_bc.jl)
        # No post-processing needed here
    end

    return 𝔽
end

"""
    get_current_simulation_time(solver_state)

Extract current simulation time from solver state.
"""
function get_current_simulation_time(solver_state)
    # Try different possible time sources in solver state
    # Different drivers expose time under slightly different names. Keep the
    # fallback chain here so BC integration code does not duplicate that logic.
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
            dt = hasfield(typeof(ts_state), :dt) ? ts_state.dt : 1.0
            return ts_state.step * dt
        end
    end

    # Fallback to zero if no time information found
    return 0.0
end

"""
    validate_field_boundary_compatibility(𝔽, field_type::FieldType, boundary_set::BoundaryConditionSet)

Validate that a field structure is compatible with boundary conditions.
"""
function validate_field_boundary_compatibility(𝔽, field_type::FieldType, boundary_set::BoundaryConditionSet)
    errors = String[]

    # Check field type matches boundary condition type
    if boundary_set.field_type != field_type
        push!(errors, "Field type mismatch: field=$field_type, boundary=$(boundary_set.field_type)")
    end

    # Check grid compatibility
    if hasfield(typeof(𝔽), :config)
        config = 𝔽.config

        if boundary_set.inner_boundary.nlat != config.nlat
            push!(errors, "Grid size mismatch: inner boundary nlat=$(boundary_set.inner_boundary.nlat), config nlat=$(config.nlat)")
        end

        if boundary_set.inner_boundary.nlon != config.nlon
            push!(errors, "Grid size mismatch: inner boundary nlon=$(boundary_set.inner_boundary.nlon), config nlon=$(config.nlon)")
        end

        if boundary_set.outer_boundary.nlat != config.nlat
            push!(errors, "Grid size mismatch: outer boundary nlat=$(boundary_set.outer_boundary.nlat), config nlat=$(config.nlat)")
        end

        if boundary_set.outer_boundary.nlon != config.nlon
            push!(errors, "Grid size mismatch: outer boundary nlon=$(boundary_set.outer_boundary.nlon), config nlon=$(config.nlon)")
        end
    end

    # Field-specific validation checks semantic compatibility, not just shape:
    # scalar boundaries must stay scalar and vector boundaries must map onto the
    # toroidal/poloidal storage that the solver expects.
    # Field-specific validation
    if field_type == VELOCITY || field_type == MAGNETIC
        # Vector fields must have toroidal and poloidal components
        if !hasfield(typeof(𝔽), :toroidal) || !hasfield(typeof(𝔽), :poloidal)
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
    copy_boundary_conditions!(dest_𝔽, src_𝔽, field_type::FieldType)

Copy boundary conditions from one field to another.
Both fields must have boundary condition fields already defined.
"""
function copy_boundary_conditions!(dest_𝔽, src_𝔽, field_type::FieldType)
    # Validate required fields exist on both src and dest
    if !hasfield(typeof(src_𝔽), :boundary_condition_set)
        throw(ArgumentError("Source field missing boundary_condition_set field"))
    end

    if !hasfield(typeof(dest_𝔽), :boundary_condition_set)
        throw(ArgumentError("Destination field missing boundary_condition_set field"))
    end

    if src_𝔽.boundary_condition_set === nothing
        return dest_𝔽
    end

    # Copy boundary condition set
    dest_𝔽.boundary_condition_set = src_𝔽.boundary_condition_set

    # Copy interpolation cache (if both fields have it)
    if hasfield(typeof(src_𝔽), :boundary_interpolation_cache) &&
       hasfield(typeof(dest_𝔽), :boundary_interpolation_cache)
        for (k, v) in src_𝔽.boundary_interpolation_cache
            dest_𝔽.boundary_interpolation_cache[k] = deepcopy(v)
        end
    end

    # Copy time index (if both fields have it)
    if hasfield(typeof(src_𝔽), :boundary_time_index) &&
       hasfield(typeof(dest_𝔽), :boundary_time_index)
        dest_𝔽.boundary_time_index[] = src_𝔽.boundary_time_index[]
    end

    # Copy metadata first, then storage arrays. That ordering keeps partially
    # initialized destination fields from observing stale time indices or caches
    # with mismatched boundary data.
    # Copy boundary condition arrays
    if field_type == TEMPERATURE || field_type == COMPOSITION
        if hasfield(typeof(src_𝔽), :boundary_values) &&
           hasfield(typeof(dest_𝔽), :boundary_values)
            dest_𝔽.boundary_values .= src_𝔽.boundary_values
        end

        if hasfield(typeof(src_𝔽), :bc_type_inner) &&
           hasfield(typeof(dest_𝔽), :bc_type_inner)
            dest_𝔽.bc_type_inner .= src_𝔽.bc_type_inner
            dest_𝔽.bc_type_outer .= src_𝔽.bc_type_outer
        end

    elseif field_type == VELOCITY || field_type == MAGNETIC
        # Copy toroidal boundary conditions
        if hasfield(typeof(src_𝔽), :toroidal) && hasfield(typeof(dest_𝔽), :toroidal)
            if hasfield(typeof(src_𝔽.toroidal), :boundary_values) &&
               hasfield(typeof(dest_𝔽.toroidal), :boundary_values)
                dest_𝔽.toroidal.boundary_values .= src_𝔽.toroidal.boundary_values
                dest_𝔽.toroidal.bc_type_inner .= src_𝔽.toroidal.bc_type_inner
                dest_𝔽.toroidal.bc_type_outer .= src_𝔽.toroidal.bc_type_outer
            end
        end

        # Copy poloidal boundary conditions
        if hasfield(typeof(src_𝔽), :poloidal) && hasfield(typeof(dest_𝔽), :poloidal)
            if hasfield(typeof(src_𝔽.poloidal), :boundary_values) &&
               hasfield(typeof(dest_𝔽.poloidal), :boundary_values)
                dest_𝔽.poloidal.boundary_values .= src_𝔽.poloidal.boundary_values
                dest_𝔽.poloidal.bc_type_inner .= src_𝔽.poloidal.bc_type_inner
                dest_𝔽.poloidal.bc_type_outer .= src_𝔽.poloidal.bc_type_outer
            end
        end
    end

    return dest_𝔽
end

"""
    reset_boundary_conditions!(𝔽, field_type::FieldType)

Reset/clear boundary conditions for a field.
"""
function reset_boundary_conditions!(𝔽, field_type::FieldType)
    # Validate and clear boundary condition set
    if hasfield(typeof(𝔽), :boundary_condition_set)
        𝔽.boundary_condition_set = nothing
    end

    # Clear interpolation cache
    if hasfield(typeof(𝔽), :boundary_interpolation_cache)
        empty!(𝔽.boundary_interpolation_cache)
    end

    # Reset time index
    if hasfield(typeof(𝔽), :boundary_time_index)
        𝔽.boundary_time_index[] = 1
    end

    # Resetting arrays as well as metadata makes the field safe to reuse in
    # tests, restarts, and solver reinitialization without carrying old BC data.
    # Reset boundary arrays to zero
    if field_type == TEMPERATURE || field_type == COMPOSITION
        if hasfield(typeof(𝔽), :boundary_values)
            fill!(𝔽.boundary_values, zero(eltype(𝔽.boundary_values)))
        end

    elseif field_type == VELOCITY || field_type == MAGNETIC
        # Reset toroidal boundary conditions
        if hasfield(typeof(𝔽), :toroidal) && hasfield(typeof(𝔽.toroidal), :boundary_values)
            fill!(𝔽.toroidal.boundary_values, zero(eltype(𝔽.toroidal.boundary_values)))
        end

        # Reset poloidal boundary conditions
        if hasfield(typeof(𝔽), :poloidal) && hasfield(typeof(𝔽.poloidal), :boundary_values)
            fill!(𝔽.poloidal.boundary_values, zero(eltype(𝔽.poloidal.boundary_values)))
        end
    end

    return 𝔽
end

"""
    get_boundary_condition_summary(𝔽, field_type::FieldType)

Get a summary of the current boundary condition state.
"""
function get_boundary_condition_summary(𝔽, field_type::FieldType)

    summary = Dict{String, Any}()

    summary["field_type"] = string(field_type)

    # Check if field has boundary condition fields
    has_bc_field = hasfield(typeof(𝔽), :boundary_condition_set)
    summary["has_boundary_fields"] = has_bc_field

    if !has_bc_field
        summary["reason"] = "Field structure does not have boundary condition fields"
        return summary
    end

    has_boundary_conditions = 𝔽.boundary_condition_set !== nothing
    summary["has_boundary_conditions"] = has_boundary_conditions

    if has_boundary_conditions
        boundary_set = 𝔽.boundary_condition_set

        summary["boundary_field_name"] = boundary_set.field_name
        summary["creation_time"] = boundary_set.creation_time

        # Time index (if available)
        if hasfield(typeof(𝔽), :boundary_time_index)
            summary["current_time_index"] = 𝔽.boundary_time_index[]
        end

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

        # Cache info (if available)
        if hasfield(typeof(𝔽), :boundary_interpolation_cache)
            summary["interpolation_cache"] = Dict(
                "inner_cached" => haskey(𝔽.boundary_interpolation_cache, "inner"),
                "outer_cached" => haskey(𝔽.boundary_interpolation_cache, "outer"),
                "cache_size" => length(𝔽.boundary_interpolation_cache)
            )
        end

        # Field-specific boundary condition info
        if field_type == TEMPERATURE || field_type == COMPOSITION
            if hasfield(typeof(𝔽), :boundary_values)
                summary["boundary_spectral_coefficients"] = Dict(
                    "inner_nonzero" => count(!iszero, 𝔽.boundary_values[1, :]),
                    "outer_nonzero" => count(!iszero, 𝔽.boundary_values[2, :]),
                    "total_modes" => size(𝔽.boundary_values, 2)
                )
            end

        elseif field_type == VELOCITY || field_type == MAGNETIC
            summary["boundary_spectral_coefficients"] = Dict{String, Any}()

            if hasfield(typeof(𝔽), :toroidal) && hasfield(typeof(𝔽.toroidal), :boundary_values)
                summary["boundary_spectral_coefficients"]["toroidal"] = Dict(
                    "inner_nonzero" => count(!iszero, 𝔽.toroidal.boundary_values[1, :]),
                    "outer_nonzero" => count(!iszero, 𝔽.toroidal.boundary_values[2, :]),
                    "total_modes" => size(𝔽.toroidal.boundary_values, 2)
                )
            end

            if hasfield(typeof(𝔽), :poloidal) && hasfield(typeof(𝔽.poloidal), :boundary_values)
                summary["boundary_spectral_coefficients"]["poloidal"] = Dict(
                    "inner_nonzero" => count(!iszero, 𝔽.poloidal.boundary_values[1, :]),
                    "outer_nonzero" => count(!iszero, 𝔽.poloidal.boundary_values[2, :]),
                    "total_modes" => size(𝔽.poloidal.boundary_values, 2)
                )
            end
        end
    else
        summary["reason"] = "No boundary conditions loaded"
    end

    return summary
end

# ================================================================================
# Apply Programmatic Boundaries to Fields
# ================================================================================

"""
    apply_temperature_boundaries!(temp_field, pbs::ProgrammaticBoundarySet; time=0.0)

Apply programmatic temperature boundary conditions to a temperature field.

Transforms physical-space boundary data to spectral coefficients and stores them
in the field's `boundary_values` matrix, along with setting BC types.

# Arguments
- `temp_field`: SHTnsTemperatureField to apply boundaries to
- `pbs`: ProgrammaticBoundarySet containing boundary data and BC types
- `time`: Current simulation time (for time-dependent boundaries)
"""
function apply_temperature_boundaries!(temp_field, pbs::ProgrammaticBoundarySet; time=0.0)
    cfg = temp_field.config

    # Store the boundary condition set on the field
    temp_field.boundary_condition_set = pbs.boundary_set

    # Set BC types for all modes
    fill!(temp_field.bc_type_inner, Int(pbs.inner_bc_type))
    fill!(temp_field.bc_type_outer, Int(pbs.outer_bc_type))

    # Programmatic boundaries are authored in physical space, but the solver
    # consumes spectral boundary rows. The transform happens once here so the
    # timestep path can stay entirely in spectral storage.
    # Transform inner boundary from physical to spectral
    inner_coeffs = shtns_physical_to_spectral(pbs.boundary_set.inner_boundary.values, cfg)
    nlm = min(length(inner_coeffs), size(temp_field.boundary_values, 2))
    for idx in 1:nlm
        temp_field.boundary_values[1, idx] = inner_coeffs[idx]
    end

    # Transform outer boundary from physical to spectral
    outer_coeffs = shtns_physical_to_spectral(pbs.boundary_set.outer_boundary.values, cfg)
    for idx in 1:nlm
        temp_field.boundary_values[2, idx] = outer_coeffs[idx]
    end

    return temp_field
end

"""
    apply_composition_boundaries!(comp_field, pbs::ProgrammaticBoundarySet; time=0.0)

Apply programmatic composition boundary conditions to a composition field.
Same interface as `apply_temperature_boundaries!`.
"""
function apply_composition_boundaries!(comp_field, pbs::ProgrammaticBoundarySet; time=0.0)
    cfg = comp_field.config

    # Store the boundary condition set on the field
    comp_field.boundary_condition_set = pbs.boundary_set

    # Set BC types for all modes
    fill!(comp_field.bc_type_inner, Int(pbs.inner_bc_type))
    fill!(comp_field.bc_type_outer, Int(pbs.outer_bc_type))

    # Composition follows the same physical-to-spectral embedding path as
    # temperature so the implicit solve sees one consistent boundary format.
    # Transform inner boundary from physical to spectral
    inner_coeffs = shtns_physical_to_spectral(pbs.boundary_set.inner_boundary.values, cfg)
    nlm = min(length(inner_coeffs), size(comp_field.boundary_values, 2))
    for idx in 1:nlm
        comp_field.boundary_values[1, idx] = inner_coeffs[idx]
    end

    # Transform outer boundary from physical to spectral
    outer_coeffs = shtns_physical_to_spectral(pbs.boundary_set.outer_boundary.values, cfg)
    for idx in 1:nlm
        comp_field.boundary_values[2, idx] = outer_coeffs[idx]
    end

    return comp_field
end

export initialize_boundary_conditions!, apply_boundary_conditions!
export validate_field_boundary_compatibility, copy_boundary_conditions!
export reset_boundary_conditions!, get_boundary_condition_summary
export apply_temperature_boundaries!, apply_composition_boundaries!
