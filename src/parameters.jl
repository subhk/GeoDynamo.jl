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
    
    # Derived parameters — always recomputed by update_derived_parameters!()
    # Defaults here use the default i_L=32/i_M=32 values; any custom construction
    # must call update_derived_parameters!() to reconcile (done automatically by
    # load_parameters and set_parameters!).
    i_L1::Int = 32
    i_M1::Int = 32
    i_H1::Int = (32 + 1) * (32 + 2) ÷ 2 - 1
    i_pH1::Int = (32 + 1) * (32 + 2) ÷ 2 - 1
    i_Ma::Int = 32 ÷ 2
    
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
    d_t_end::Float64 = 1.0       # Simulation end time
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
    i_vel_bc::Int = 1             # Velocity BC: 1=no-slip both, 2=no-slip inner/stress-free outer, 3=stress-free inner/no-slip outer, 4=stress-free both
    i_tmp_bc::Int = 1             # Temperature BC
    i_cmp_bc::Int = 1             # Composition BC
    
    # File-based boundary conditions
    s_tmp_bc_file::String = ""              # Temperature BC file path (empty = homogeneous)
    s_cmp_bc_file::String = ""              # Composition BC file path (empty = homogeneous)
    s_tmp_bc_format::Symbol = :physical     # :physical or :spectral
    s_cmp_bc_format::Symbol = :physical     # :physical or :spectral

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

    # ================================================================================
    # Topography Coupling Parameters
    # ================================================================================
    # Enable boundary topography effects on BCs (CMB and/or ICB)
    b_topography_enabled::Bool = false        # Master switch for topography coupling
    d_topo_epsilon::Float64 = 0.01            # Topography amplitude parameter ε
    i_topo_lmax::Int = -1                     # Max degree for topography (-1 = use i_L)

    # Topography coupling flags
    b_topo_velocity::Bool = true              # Enable velocity BC topography correction
    b_topo_magnetic::Bool = true              # Enable magnetic BC topography correction
    b_topo_thermal::Bool = true               # Enable thermal BC topography correction
    b_topo_slope_terms::Bool = true           # Include slope (∇h) terms
    b_topo_shift_terms::Bool = true           # Include shift (h) terms

    # Stefan condition for ICB phase change (optional)
    b_stefan_enabled::Bool = false            # Enable Stefan condition for ICB evolution
    d_stefan_number::Float64 = 1.0            # Stefan number St = c_p ΔT / L
    d_lambda_ic::Float64 = 1.0                # Conductivity ratio λ = k_ic / k_oc
    d_latent_heat::Float64 = 1.0              # Latent heat (nondimensional)

    # Topography source specifications
    s_topo_icb_file::String = ""              # ICB topography file (NetCDF), empty = no ICB topo
    s_topo_cmb_file::String = ""              # CMB topography file (NetCDF), empty = no CMB topo

    # ================================================================================
    # Restart Parameters
    # ================================================================================
    s_restart_file::String = ""               # Path to restart NetCDF file (empty = fresh start)
    s_restart_dir::String = ""                # Directory containing restart files (alternative to s_restart_file)
    d_restart_time::Float64 = 0.0             # Target restart time (used with s_restart_dir)
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
    for key in (:i_vel_bc, :i_tmp_bc, :i_cmp_bc, :i_poloidal_stress_iters,
                :s_tmp_bc_file, :s_cmp_bc_file, :s_tmp_bc_format, :s_cmp_bc_format)
        print_entry(io, key, getfield(params, key))
    end

    print_section(io, "Flags")
    for key in (:b_mag_impose, :i_B)
        print_entry(io, key, getfield(params, key))
    end

    print_section(io, "Topography Coupling")
    for key in (:b_topography_enabled, :d_topo_epsilon, :i_topo_lmax,
                :b_topo_velocity, :b_topo_magnetic, :b_topo_thermal,
                :b_topo_slope_terms, :b_topo_shift_terms)
        print_entry(io, key, getfield(params, key))
    end

    print_section(io, "Stefan Condition")
    for key in (:b_stefan_enabled, :d_stefan_number, :d_lambda_ic, :d_latent_heat)
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
    validate_parameters(params::GeoDynamoParameters; strict::Bool=false)

Validate all simulation parameters for physical correctness and numerical stability.

# Arguments
- `params::GeoDynamoParameters`: Parameters to validate
- `strict::Bool=false`: If true, throw errors on invalid params; if false, issue warnings

# Returns
- `(is_valid::Bool, errors::Vector{String}, warnings::Vector{String})`
"""
function validate_parameters(params::GeoDynamoParameters; strict::Bool=false)
    errors = String[]
    warnings = String[]

    # Grid parameters validation
    if params.i_N < 8
        push!(errors, "i_N (radial points) = $(params.i_N) is too small (minimum 8)")
    elseif params.i_N < 16
        push!(warnings, "i_N = $(params.i_N) is very coarse; consider i_N >= 32 for accuracy")
    end

    if params.i_L < 1
        push!(errors, "i_L (max spherical harmonic degree) = $(params.i_L) must be >= 1")
    end

    if params.i_M < 0 || params.i_M > params.i_L
        push!(errors, "i_M = $(params.i_M) must be in range [0, i_L=$(params.i_L)]")
    end

    if params.i_Th < 2 * params.i_L
        push!(warnings, "i_Th = $(params.i_Th) < 2*i_L = $(2*params.i_L); may cause aliasing")
    end

    if params.i_Ph < 2 * params.i_M
        push!(warnings, "i_Ph = $(params.i_Ph) < 2*i_M = $(2*params.i_M); may cause aliasing")
    end

    # Physical parameters validation
    if params.d_rratio < 0.0 || params.d_rratio >= 1.0
        push!(errors, "d_rratio = $(params.d_rratio) must be in range [0, 1) for shell geometry")
    end

    if params.d_Ra <= 0.0
        push!(errors, "d_Ra (Rayleigh number) = $(params.d_Ra) must be positive")
    elseif params.d_Ra > 1e10
        push!(warnings, "d_Ra = $(params.d_Ra) is very large; ensure numerical stability")
    end

    if params.d_E <= 0.0
        push!(errors, "d_E (Ekman number) = $(params.d_E) must be positive")
    elseif params.d_E < 1e-8
        push!(warnings, "d_E = $(params.d_E) is very small; may require fine resolution")
    end

    if params.d_Pr <= 0.0
        push!(errors, "d_Pr (Prandtl number) = $(params.d_Pr) must be positive")
    end

    if params.d_Pm <= 0.0
        push!(errors, "d_Pm (Magnetic Prandtl number) = $(params.d_Pm) must be positive")
    end

    if params.d_Sc <= 0.0
        push!(errors, "d_Sc (Schmidt number) = $(params.d_Sc) must be positive")
    end

    # Timestepping validation
    if params.d_timestep <= 0.0
        push!(errors, "d_timestep = $(params.d_timestep) must be positive")
    elseif params.d_timestep > 1.0
        push!(warnings, "d_timestep = $(params.d_timestep) is very large; check CFL condition")
    end

    if params.i_maxtstep < 1
        push!(errors, "i_maxtstep = $(params.i_maxtstep) must be >= 1")
    end

    # CFL condition estimate (rough check)
    # For spectral methods: dt < C / (l_max^2 * diffusivity)
    # In magnetic diffusion time scaling: magnetic κ=1, thermal κ=Pm/Pr,
    # compositional κ=Pm/Sc, viscous κ=E (Ekman number)
    max_diffusivity = max(1.0, params.d_Pm / params.d_Pr, params.d_Pm / params.d_Sc, params.d_E)
    cfl_limit = 0.1 / (params.i_L^2 * max_diffusivity)
    if params.d_timestep > cfl_limit
        push!(warnings, "d_timestep = $(params.d_timestep) may violate CFL condition " *
                        "(estimated limit: $(cfl_limit) for spectral stability with " *
                        "max diffusivity = $(max_diffusivity))")
    end

    # Timestepping scheme validation
    valid_schemes = [:cnab2, :theta, :erk2, :etd]
    if !(params.ts_scheme in valid_schemes)
        push!(errors, "ts_scheme = $(params.ts_scheme) not recognized. " *
                      "Valid schemes: $(valid_schemes)")
    end

    # Output precision validation
    if !(params.output_precision in [:float32, :float64])
        push!(errors, "output_precision = $(params.output_precision) must be :float32 or :float64")
    end

    # Geometry validation
    if !(params.geometry in [:shell, :ball])
        push!(errors, "geometry = $(params.geometry) must be :shell or :ball")
    end

    if params.geometry == :ball && params.d_rratio != 0.0
        push!(warnings, "geometry = :ball but d_rratio = $(params.d_rratio) != 0; should be 0 for full ball")
    end

    # Report results
    is_valid = isempty(errors)

    if get_rank() == 0  # Only print on rank 0
        if !isempty(errors)
            println("\n⚠️  PARAMETER VALIDATION ERRORS:")
            for (i, err) in enumerate(errors)
                println("  $i. $err")
            end
        end

        if !isempty(warnings)
            println("\n⚠️  PARAMETER VALIDATION WARNINGS:")
            for (i, warn) in enumerate(warnings)
                println("  $i. $warn")
            end
        end

        if is_valid && isempty(warnings)
            println("\n✅ All parameters validated successfully")
        end
    end

    # Strict mode: throw error if invalid
    if strict && !is_valid
        error("Parameter validation failed with $(length(errors)) error(s). " *
              "Fix parameters or set strict=false to proceed with warnings.")
    end

    return (is_valid, errors, warnings)
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

                    # Parse the parameter value safely — no eval/Meta.parse execution
                    try
                        param_value = safe_parse_value(param_value_str, param_dict)
                        param_dict[param_name] = param_value
                    catch e
                        @debug "Could not parse parameter $param_name = $param_value_str: $e"
                        # Skip this parameter - it will use the default value
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

    # Always recompute derived parameters from the (potentially overridden) primaries
    update_derived_parameters!(params)

    return params
end

"""
    safe_parse_value(value_str::AbstractString, param_dict::Dict{Symbol, Any})

Parse a parameter value string without using `eval`. Supports:
- Integer and float literals (including scientific notation)
- Boolean literals (`true`, `false`)
- Symbol literals (`:name`)
- String literals (`"..."`)
- Mathematical constants (`π`)
- Simple arithmetic expressions referencing previously-defined parameters
"""
function safe_parse_value(value_str::AbstractString, param_dict::Dict{Symbol, Any})
    s = strip(value_str)

    # Boolean literals
    s == "true"  && return true
    s == "false" && return false

    # Mathematical constants
    (s == "π" || s == "pi") && return π

    # Symbol literal :name
    if startswith(s, ':')
        return Symbol(s[2:end])
    end

    # String literal
    if startswith(s, '"') && endswith(s, '"')
        return s[2:end-1]
    end

    # Try integer (including negative)
    int_val = tryparse(Int, s)
    int_val !== nothing && return int_val

    # Try float (including scientific notation like 1e-4)
    float_val = tryparse(Float64, s)
    float_val !== nothing && return float_val

    # Try evaluating as a safe arithmetic expression referencing known parameters
    expr = Meta.parse(s)
    return safe_eval_expr(expr, param_dict)
end

# Allowed binary and unary operations for safe arithmetic evaluation
const _SAFE_OPS = Set{Symbol}([:+, :-, :*, :/, :÷, :^, :div, :mod, :min, :max, :sqrt, :abs])

"""
    safe_eval_expr(expr, param_dict::Dict{Symbol, Any})

Evaluate a parsed expression using only safe arithmetic operations and known parameter values.
No arbitrary code execution is possible.
"""
function safe_eval_expr(expr, param_dict::Dict{Symbol, Any})
    # Literal values pass through
    if expr isa Number
        return expr
    end

    # Symbol lookup — must be a known parameter or constant
    if expr isa Symbol
        expr === :π  && return π
        expr === :pi && return π
        haskey(param_dict, expr) && return param_dict[expr]
        throw(ArgumentError("Unknown parameter reference: $expr"))
    end

    # QuoteNode for Symbol literals like :float64
    if expr isa QuoteNode
        return expr.value
    end

    if !(expr isa Expr)
        throw(ArgumentError("Unsupported expression type: $(typeof(expr))"))
    end

    if expr.head === :call
        op = expr.args[1]
        if !(op isa Symbol) || op ∉ _SAFE_OPS
            throw(ArgumentError("Disallowed operation in parameter file: $op"))
        end
        args = [safe_eval_expr(a, param_dict) for a in expr.args[2:end]]
        return getfield(Base, op)(args...)
    elseif expr.head === :parens || expr.head === :block
        # Parenthesized expression — evaluate the last statement
        return safe_eval_expr(expr.args[end], param_dict)
    else
        throw(ArgumentError("Unsupported expression form in parameter file: $(expr.head)"))
    end
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
        println(io, "const i_vel_bc = $(params.i_vel_bc)            # Velocity BC: 1=no-slip both, 2=NS inner/SF outer, 3=SF inner/NS outer, 4=SF both")
        println(io, "const i_tmp_bc = $(params.i_tmp_bc)            # Temperature BC")
        println(io, "const i_cmp_bc = $(params.i_cmp_bc)            # Composition BC")
        println(io, "const i_poloidal_stress_iters = $(params.i_poloidal_stress_iters)  # Iterations for poloidal stress-free correction")
        println(io)

        println(io, "# File-based boundary conditions")
        println(io, "const s_tmp_bc_file = \"$(params.s_tmp_bc_file)\"    # Temperature BC file (empty = homogeneous)")
        println(io, "const s_cmp_bc_file = \"$(params.s_cmp_bc_file)\"    # Composition BC file (empty = homogeneous)")
        println(io, "const s_tmp_bc_format = :$(params.s_tmp_bc_format)  # :physical or :spectral")
        println(io, "const s_cmp_bc_format = :$(params.s_cmp_bc_format)  # :physical or :spectral")
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

# Global parameter instance with thread-safe initialization
# Using a lock to ensure thread-safe lazy initialization
const GEODYNAMO_PARAMS = Ref{Union{GeoDynamoParameters, Nothing}}(nothing)
const PARAMS_LOCK = ReentrantLock()

"""
    get_parameters()

Get the current global parameters. If not set, loads default parameters.
Thread-safe lazy initialization with type-stable return.
"""
function get_parameters()::GeoDynamoParameters
    # Fast path: already initialized (no lock needed for read)
    params = GEODYNAMO_PARAMS[]
    if params !== nothing
        return params::GeoDynamoParameters
    end

    # Slow path: need to initialize (acquire lock)
    lock(PARAMS_LOCK) do
        # Double-check after acquiring lock (another thread may have initialized)
        params = GEODYNAMO_PARAMS[]
        if params !== nothing
            return params::GeoDynamoParameters
        end

        # Initialize parameters
        params = load_parameters()
        GEODYNAMO_PARAMS[] = params
        return params::GeoDynamoParameters
    end
end

"""
    set_parameters!(params::GeoDynamoParameters; validate::Bool=true, strict::Bool=false)

Set the global parameters (thread-safe) with optional validation.

# Arguments
- `params::GeoDynamoParameters`: Parameters to set
- `validate::Bool=true`: Whether to validate parameters before setting
- `strict::Bool=false`: If true, throw error on invalid parameters; if false, proceed with warnings
"""
function set_parameters!(params::GeoDynamoParameters; validate::Bool=true, strict::Bool=false)
    # Validate parameters if requested
    if validate
        is_valid, errors, warnings = validate_parameters(params; strict=strict)
        if !is_valid && get_rank() == 0
            @warn "Setting parameters despite validation errors. Set strict=true to enforce validation."
        end
    end

    lock(PARAMS_LOCK) do
        update_derived_parameters!(params)
        GEODYNAMO_PARAMS[] = params
        update_global_parameters!()  # Update global variables
    end
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
# Initialize with default values from GeoDynamoParameters() to ensure type stability
# (avoids Union{Nothing, T} which causes dynamic dispatch on every access)
let _defaults = GeoDynamoParameters()
    for param_name in fieldnames(GeoDynamoParameters)
        val = getfield(_defaults, param_name)
        @eval begin
            global $(param_name)
            $(param_name) = $(QuoteNode(val))
        end
    end
end

"""
    update_global_parameters!()

Update all global parameter variables with values from the current parameter struct.
Generated at module load time to avoid runtime `@eval` (which is non-reentrant,
tears state across threads, and prevents JIT specialization).
"""
@eval function update_global_parameters!()
    params = get_parameters()
    $(Expr(:block, [
        :(global $(param_name) = getfield(params, $(QuoteNode(param_name))))
        for param_name in fieldnames(GeoDynamoParameters)
    ]...))
    return nothing
end
