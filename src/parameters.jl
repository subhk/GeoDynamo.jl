# ================================================================================
# Parameter Loading System
# ================================================================================

using Dates
using Printf

"""
    GeoDynamoParameters

Structure to hold all simulation parameters. This replaces the global constants
from the old params.jl file with a more flexible parameter system.
"""
Base.@kwdef mutable struct GeoDynamoParameters
    # Grid parameters
    i_N::Int = 64        # Number of radial points
    i_Nic::Int = 16      # Number of inner core radial points
    i_L::Int = 32        # Maximum spherical harmonic degree
    i_M::Int = 32        # Maximum azimuthal wavenumber
    i_Th::Int = 64       # Number of theta points (must be compatible with SHTnsKit)
    i_Ph::Int = 128      # Number of phi points (must be compatible with SHTnsKit)
    i_KL::Int = 4        # Bandwidth for finite differences
    
    # Derived parameters (computed automatically)
    i_L1::Int = i_L
    i_M1::Int = i_M
    i_H1::Int = (i_L + 1) * (i_L + 2) ÷ 2 - 1
    i_pH1::Int = i_H1
    i_Ma::Int = i_M ÷ 2
    
    # Physical parameters
    d_rratio::Float64 = 0.35      # Inner/outer core radius ratio
    d_R_outer::Float64 = 1.0      # Ball outer radius (unit length by default)
    d_Ra::Float64 = 1e6           # Rayleigh number
    d_E::Float64 = 1e-4           # Ekman number
    d_Pr::Float64 = 1.0           # Prandtl number
    d_Pm::Float64 = 1.0           # Magnetic Prandtl number
    d_Ro::Float64 = 1e-4          # Rossby number
    d_q::Float64 = 1.0            # Thermal diffusivity ratio
    
    # Timestepping parameters
    d_timestep::Float64 = 1e-4    # Time step size
    d_time::Float64 = 0.0         # Initial time
    d_implicit::Float64 = 0.5     # Crank-Nicolson parameter
    d_dterr::Float64 = 1e-8       # Error tolerance
    d_courant::Float64 = 0.5      # CFL factor
    i_maxtstep::Int = 10000       # Maximum timesteps
    i_save_rate2::Int = 100       # Output frequency
    ts_scheme::Symbol = :cnab2    # :cnab2 or :theta (legacy)
    # ETD/Krylov controls
    i_etd_m::Int = 20             # Max Arnoldi dimension for exp/phi actions
    d_krylov_tol::Float64 = 1e-8  # Residual tolerance for adaptive Arnoldi
    
    # Output control
    output_precision::Symbol = :float64      # :float32 or :float64 for NetCDF data
    independent_output_files::Bool = true    # Each rank writes its own files without barriers
    
    # Boundary condition flags
    i_vel_bc::Int = 1             # Velocity BC: 1=no-slip, 2=stress-free
    i_tmp_bc::Int = 1             # Temperature BC
    i_cmp_bc::Int = 1             # Composition BC
    
    # BC tuning parameters
    i_poloidal_stress_iters::Int = 2  # Iterations for poloidal stress-free correction
    
    # Boolean flags
    b_mag_impose::Bool = false    # Imposed magnetic field
    
    # Additional parameters for compatibility
    i_B::Int = 0                  # Magnetic field flag
    d_Ra_C::Float64 = 1e6         # Compositional Rayleigh number
    d_Sc::Float64 = 1.0           # Schmidt number
    
    # Geometry selection (:shell or :ball)
    geometry::Symbol = :shell
end

function print_section(io::IO, title::AbstractString)
    println(io, "\n" * title)
    println(io, repeat('-', length(title)))
end

function print_entry(io::IO, name::Symbol, value)
    println(io, @sprintf("  %-20s %s", String(name), string(value)))
end

function Base.show(io::IO, ::MIME"text/plain", params::GeoDynamoParameters)
    println(io, "GeoDynamoParameters")

    print_section(io, "Grid")
    for key in (:geometry, :i_N, :i_Nic, :i_L, :i_M, :i_Th, :i_Ph, :i_KL)
        print_entry(io, key, getfield(params, key))
    end

    print_section(io, "Derived")
    for key in (:i_L1, :i_M1, :i_H1, :i_pH1, :i_Ma)
        print_entry(io, key, getfield(params, key))
    end

    print_section(io, "Physical")
    for key in (:d_rratio, :d_R_outer, :d_Ra, :d_E, :d_Pr, :d_Pm, :d_Ro, :d_q, :d_Ra_C, :d_Sc)
        print_entry(io, key, getfield(params, key))
    end

    print_section(io, "Timestepping")
    for key in (:d_timestep, :d_time, :d_implicit, :d_dterr, :d_courant,
                :i_maxtstep, :i_save_rate2, :ts_scheme, :i_etd_m, :d_krylov_tol)
        print_entry(io, key, getfield(params, key))
    end

    print_section(io, "Boundary Conditions")
    for key in (:i_vel_bc, :i_tmp_bc, :i_cmp_bc, :i_poloidal_stress_iters)
        print_entry(io, key, getfield(params, key))
    end

    print_section(io, "Flags")
    for key in (:b_mag_impose, :i_B)
        print_entry(io, key, getfield(params, key))
    end
end

"""
    update_derived_parameters!(params::GeoDynamoParameters)

Update derived parameters based on primary parameters.
"""
function update_derived_parameters!(params::GeoDynamoParameters)
    params.i_L1 = params.i_L
    params.i_M1 = params.i_M
    params.i_H1 = (params.i_L + 1) * (params.i_L + 2) ÷ 2 - 1
    params.i_pH1 = params.i_H1
    params.i_Ma = params.i_M ÷ 2
    return params
end

"""
    load_parameters(config_file::String = "")

Load parameters from a configuration file. If no file is specified,
loads from the default config/default_params.jl file.

# Arguments
- `config_file::String`: Path to parameter file (optional)

# Returns
- `GeoDynamoParameters`: Loaded parameters
"""
function load_parameters(config_file::String = "")
    # Determine config file path
    if isempty(config_file)
        # Find the package root by looking for Project.toml
        pkg_root = find_package_root()
        config_file = joinpath(pkg_root, "config", "default_params.jl")
    end
    
    if !isfile(config_file)
        @warn "Parameter file not found: $config_file. Using default parameters."
        params = GeoDynamoParameters()
        update_derived_parameters!(params)
        return params
    end
    
    # Load the parameter file in a safe way
    params = load_parameters_from_file(config_file)
    update_derived_parameters!(params)
    
    return params
end

"""
    find_package_root()

Find the root directory of the GeoDynamo.jl package.
"""
function find_package_root()
    current_dir = @__DIR__
    
    # Walk up the directory tree looking for Project.toml
    while current_dir != "/"
        project_file = joinpath(current_dir, "Project.toml")
        if isfile(project_file)
            # Check if this is the GeoDynamo.jl project
            try
                content = read(project_file, String)
                if contains(content, "GeoDynamo") || contains(content, "name = \"GeoDynamo\"")
                    return current_dir
                end
            catch
                # Continue searching if we can't read the file
            end
        end
        current_dir = dirname(current_dir)
    end
    
    # If we can't find it, assume current directory
    @warn "Could not find GeoDynamo.jl package root. Using current directory."
    return dirname(@__DIR__)
end

"""
    load_parameters_from_file(config_file::String)

Load parameters from a Julia file containing parameter definitions.
"""
function load_parameters_from_file(config_file::String)
    # Create a safe environment to evaluate the parameter file
    param_dict = Dict{Symbol, Any}()

    try
        # Read and parse the file
        content = read(config_file, String)

        # Extract parameter definitions using regex
        # Match lines like: const i_N = 64
        for line in split(content, '\n')
            line = strip(line)
            if startswith(line, "const ") && contains(line, " = ")
                # Parse: const i_N = 64  # comment
                match_result = match(r"const\s+(\w+)\s*=\s*([^#]+)", line)
                if match_result !== nothing
                    param_name = Symbol(match_result.captures[1])
                    param_value_str = strip(match_result.captures[2])

                    # Evaluate the parameter value safely
                    # Create a module-level context with already-parsed parameters
                    try
                        # First, try direct evaluation (for simple literals)
                        param_value = eval(Meta.parse(param_value_str))
                        param_dict[param_name] = param_value
                    catch e
                        # If that fails, try evaluating with previously loaded parameters in scope
                        # This handles derived parameters like i_L1 = i_L
                        try
                            # Create a temporary module with the parameters we've loaded so far
                            expr = Meta.parse(param_value_str)
                            # Substitute known parameter names with their values
                            param_value = eval_with_context(expr, param_dict)
                            param_dict[param_name] = param_value
                        catch e2
                            @debug "Could not parse parameter $param_name = $param_value_str: $e2"
                            # Skip this parameter - it will use the default value
                        end
                    end
                end
            end
        end
    catch e
        @error "Error reading parameter file $config_file: $e"
        return GeoDynamoParameters()
    end

    # Create parameters struct with loaded values
    params = GeoDynamoParameters()

    # Update parameters with loaded values
    for field in fieldnames(GeoDynamoParameters)
        if haskey(param_dict, field)
            value = param_dict[field]
            # Skip if the value is nothing (failed to evaluate)
            if value === nothing
                @debug "Skipping parameter $field with value nothing (using default)"
                continue
            end
            try
                setfield!(params, field, value)
            catch e
                @warn "Could not set parameter $field: $e"
            end
        end
    end

    return params
end

"""
    eval_with_context(expr, param_dict::Dict{Symbol, Any})

Evaluate an expression by substituting symbols with values from param_dict.
"""
function eval_with_context(expr, param_dict::Dict{Symbol, Any})
    # If expr is a symbol, look it up in param_dict
    if expr isa Symbol
        return get(param_dict, expr, nothing)
    end

    # If expr is not an expression, just evaluate it normally
    if !isa(expr, Expr)
        return eval(expr)
    end

    # Recursively substitute symbols in the expression
    new_args = []
    for arg in expr.args
        if arg isa Symbol && haskey(param_dict, arg)
            push!(new_args, param_dict[arg])
        elseif arg isa Expr
            push!(new_args, eval_with_context(arg, param_dict))
        else
            push!(new_args, arg)
        end
    end

    # Create new expression with substituted values and evaluate
    new_expr = Expr(expr.head, new_args...)
    return eval(new_expr)
end

"""
    save_parameters(params::GeoDynamoParameters, filename::String)

Save parameters to a Julia file.
"""
function save_parameters(params::GeoDynamoParameters, filename::String)
    open(filename, "w") do io
        println(io, "# GeoDynamo.jl Parameters")
        println(io, "# Generated on $(now())")
        println(io)
        
        println(io, "# Grid parameters")
        println(io, "const i_N   = $(params.i_N)        # Number of radial points")
        println(io, "const i_Nic = $(params.i_Nic)      # Number of inner core radial points")
        println(io, "const i_L   = $(params.i_L)        # Maximum spherical harmonic degree")
        println(io, "const i_M   = $(params.i_M)        # Maximum azimuthal wavenumber")
        println(io, "const i_Th  = $(params.i_Th)       # Number of theta points")
        println(io, "const i_Ph  = $(params.i_Ph)       # Number of phi points")
        println(io, "const i_KL  = $(params.i_KL)        # Bandwidth for finite differences")
        println(io)
        
        println(io, "# Derived parameters")
        println(io, "const i_L1 = i_L")
        println(io, "const i_M1 = i_M")
        println(io, "const i_H1 = (i_L + 1) * (i_L + 2) ÷ 2 - 1")
        println(io, "const i_pH1 = i_H1")
        println(io, "const i_Ma = i_M ÷ 2")
        println(io)
        
        println(io, "# Physical parameters")
        println(io, "const d_rratio = $(params.d_rratio)         # Inner/outer core radius ratio")
        println(io, "const d_R_outer = $(params.d_R_outer)       # Ball outer radius (1.0 by default)")
        println(io, "const d_Ra = $(params.d_Ra)              # Rayleigh number")
        println(io, "const d_E = $(params.d_E)              # Ekman number")
        println(io, "const d_Pr = $(params.d_Pr)              # Prandtl number")
        println(io, "const d_Pm = $(params.d_Pm)              # Magnetic Prandtl number")
        println(io, "const d_Ro = $(params.d_Ro)              # Rossby number")
        println(io, "const d_q = $(params.d_q)               # Thermal diffusivity ratio")
        println(io)
        
        println(io, "# Timestepping parameters")
        println(io, "const d_timestep = $(params.d_timestep)")
        println(io, "const d_time = $(params.d_time)")
        println(io, "const d_implicit = $(params.d_implicit)        # Crank-Nicolson parameter")
        println(io, "const d_dterr = $(params.d_dterr)          # Error tolerance")
        println(io, "const d_courant = $(params.d_courant)         # CFL factor")
        println(io, "const i_maxtstep = $(params.i_maxtstep)      # Maximum timesteps")
        println(io, "const i_save_rate2 = $(params.i_save_rate2)      # Output frequency")
        println(io, "const ts_scheme = :$(params.ts_scheme)        # :cnab2, :theta, or :eab2")
        println(io, "const i_etd_m = $(params.i_etd_m)            # Krylov max subspace size")
        println(io, "const d_krylov_tol = $(params.d_krylov_tol)   # Krylov residual tolerance")
        println(io, "const output_precision = :$(params.output_precision)   # :float32 or :float64")
        println(io, "const independent_output_files = $(params.independent_output_files)   # Each rank writes its own files")
        println(io)
        
        println(io, "# Boundary condition flags")
        println(io, "const i_vel_bc = $(params.i_vel_bc)            # Velocity BC: 1=no-slip, 2=stress-free")
        println(io, "const i_tmp_bc = $(params.i_tmp_bc)            # Temperature BC")
        println(io, "const i_cmp_bc = $(params.i_cmp_bc)            # Composition BC")
        println(io, "const i_poloidal_stress_iters = $(params.i_poloidal_stress_iters)  # Iterations for poloidal stress-free correction")
        println(io)
        
        println(io, "# Boolean flags")
        println(io, "const b_mag_impose = $(params.b_mag_impose)    # Imposed magnetic field")
        println(io)
        
        println(io, "# Geometry selection")
        println(io, "const geometry = :$(params.geometry)   # :shell or :ball")
    end
    
    @info "Parameters saved to $filename"
end

"""
    create_parameter_template(filename::String)

Create a template parameter file for users to customize.
"""
function create_parameter_template(filename::String)
    params = GeoDynamoParameters()  # Default parameters
    save_parameters(params, filename)
    @info "Parameter template created at $filename"
end

# Global parameter instance (will be set during module initialization)
const GEODYNAMO_PARAMS = Ref{Union{GeoDynamoParameters, Nothing}}(nothing)

"""
    get_parameters()

Get the current global parameters. If not set, loads default parameters.
"""
function get_parameters()
    if GEODYNAMO_PARAMS[] === nothing
        GEODYNAMO_PARAMS[] = load_parameters()
    end
    return GEODYNAMO_PARAMS[]
end

"""
    set_parameters!(params::GeoDynamoParameters)

Set the global parameters.
"""
function set_parameters!(params::GeoDynamoParameters)
    update_derived_parameters!(params)
    GEODYNAMO_PARAMS[] = params
    update_global_parameters!()  # Update global variables
    return params
end

"""
    initialize_parameters(config_file::String = "")

Initialize the global parameter system.
"""
function initialize_parameters(config_file::String = "")
    params = load_parameters(config_file)
    set_parameters!(params)
    return params
end

# Convenience macros for backward compatibility (deprecated - use direct variable access)
macro param(name)
    quote
        $(esc(name))  # Just return the variable directly
    end
end

# Define global parameter variables for direct access
for param_name in fieldnames(GeoDynamoParameters)
    @eval begin
        global $(param_name)
        $(param_name) = nothing  # Will be initialized later
    end
end

"""
    update_global_parameters!()

Update all global parameter variables with values from the current parameter struct.
"""
function update_global_parameters!()
    params = get_parameters()
    for param_name in fieldnames(GeoDynamoParameters)
        @eval begin
            global $(param_name) = $params.$(param_name)
        end
    end
end
