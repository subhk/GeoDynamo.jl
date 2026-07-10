# ================================================================================
# Parameter Loading System
# ================================================================================

using Dates
using Printf

"""
    SolverParameters

Internal solver configuration using the same names accepted by the public model
constructors: `nr`, `lmax`, `mmax`, `Ek`, `Pr`, `Pm`, `Sc`, and related
descriptive keywords.
"""
Base.@kwdef struct SolverParameters
    architecture::Symbol = :cpu
    geometry::Symbol = :shell

    nr::Int = 64
    nr_inner::Int = 16
    lmax::Int = 32
    mmax::Int = 32
    nlat::Int = 64
    nlon::Int = 128
    radial_bandwidth::Int = 4
    radius_ratio::Float64 = 0.35
    r_outer::Float64 = 1.0

    Ek::Float64 = 1e-4
    Ra::Float64 = 1e6
    RaC::Float64 = 1e6
    Pr::Float64 = 1.0
    Pm::Float64 = 1.0
    Sc::Float64 = 1.0

    start_time::Float64 = 0.0
    timestep::Float64 = 5e-5
    timestep_error::Float64 = 1e-8
    courant::Float64 = 0.5
    end_time::Float64 = 1.0
    stop_iteration::Int = 10_000
    timestepper::AbstractTimestepper = CNAB2()

    output_precision::Symbol = :float64
    independent_output_files::Bool = true
    output_interval::Float64 = 1.0
    restart_interval::Float64 = 0.0

    include_magnetic::Bool = false
    include_composition::Bool = true
    impose_magnetic_field::Bool = false
    magnetic_inner_bc::Symbol = :insulating   # :insulating | :conducting_inner_core

    velocity_bcs::BoundaryConditions = BoundaryConditions(inner = NoSlip(), outer = NoSlip())
    temperature_bcs::BoundaryConditions = BoundaryConditions(
        inner = FixedTemperature(1.0), outer = FixedTemperature(0.0))
    composition_bcs::BoundaryConditions = BoundaryConditions(
        inner = FixedTemperature(0.0), outer = FixedTemperature(0.0))
    poloidal_stress_iterations::Int = 2
    temperature_bc_file::String = ""
    composition_bc_file::String = ""
    temperature_bc_format::Symbol = :physical
    composition_bc_format::Symbol = :physical

    topography_enabled::Bool = false
    topography_epsilon::Float64 = 0.01
    topography_degree::Int = -1
    include_topography_velocity::Bool = true
    include_topography_magnetic::Bool = true
    include_topography_thermal::Bool = true
    include_topography_slope_terms::Bool = true
    include_topography_shift_terms::Bool = true

    stefan_enabled::Bool = false
    stefan_number::Float64 = 1.0
    inner_core_conductivity_ratio::Float64 = 1.0
    latent_heat::Float64 = 1.0

    icb_topography_file::String = ""
    ocb_topography_file::String = ""

    restart_file::String = ""
    restart_dir::String = ""
    restart_time::Float64 = 0.0

    internal_heating::Union{Nothing, Float64, Function} = nothing
    compositional_source::Union{Nothing, Float64, Function} = nothing
end

function _parameter_rank0()
    return !isdefined(@__MODULE__, :get_rank) || get_rank() == 0
end

function print_section(io::IO, title::AbstractString)
    println(io, "\n" * title)
    println(io, repeat('-', length(title)))
end

function print_entry(io::IO, name::Symbol, value)
    println(io, @sprintf("  %-28s %s", String(name), string(value)))
end

function Base.show(io::IO, ::MIME"text/plain", params::SolverParameters)
    println(io, "SolverParameters")

    print_section(io, "Grid")
    for key in (:architecture, :geometry, :nr, :nr_inner, :lmax, :mmax, :nlat,
        :nlon, :radial_bandwidth, :radius_ratio, :r_outer)
        print_entry(io, key, getfield(params, key))
    end

    print_section(io, "Physics")
    for key in (:Ek, :Ra, :RaC, :Pr, :Pm, :Sc, :include_magnetic,
        :include_composition, :impose_magnetic_field,
        :internal_heating, :compositional_source)
        print_entry(io, key, getfield(params, key))
    end

    print_section(io, "Timestepping")
    for key in (:timestepper, :timestep, :start_time, :end_time,
        :stop_iteration, :timestep_error, :courant)
        print_entry(io, key, getfield(params, key))
    end

    print_section(io, "Boundary Conditions")
    for key in (:velocity_bcs, :temperature_bcs, :composition_bcs,
        :temperature_bc_file, :composition_bc_file,
        :temperature_bc_format, :composition_bc_format)
        print_entry(io, key, getfield(params, key))
    end

    print_section(io, "Topography")
    for key in (:topography_enabled, :topography_epsilon, :topography_degree,
        :include_topography_velocity, :include_topography_magnetic,
        :include_topography_thermal, :include_topography_slope_terms,
        :include_topography_shift_terms, :stefan_enabled,
        :stefan_number, :inner_core_conductivity_ratio, :latent_heat,
        :icb_topography_file, :ocb_topography_file)
        print_entry(io, key, getfield(params, key))
    end
end

function _parameter_errors_warnings(params::SolverParameters)
    errors = String[]
    warnings = String[]

    if !(params.architecture in (:cpu, :gpu))
        push!(errors, "architecture = $(params.architecture) must be :cpu or :gpu")
    end

    if params.nr < 8
        push!(errors, "nr (radial points) = $(params.nr) is too small (minimum 8)")
    elseif params.nr < 16
        push!(warnings, "nr = $(params.nr) is very coarse; consider nr >= 32 for accuracy")
    end

    if params.geometry === :shell && params.nr_inner < 2
        push!(errors, "nr_inner = $(params.nr_inner) must be >= 2 for shell geometry")
    end
    if params.geometry === :shell && params.nr_inner >= params.nr
        push!(errors,
            "nr_inner = $(params.nr_inner) must be < nr = $(params.nr) " *
            "(inner-core points are a strict subset of the radial grid; " *
            "matches the SphericalShellGrid constructor)")
    end

    if params.courant <= 0.0
        push!(errors, "courant = $(params.courant) must be positive")
    elseif params.courant > 1.0
        push!(warnings, "courant = $(params.courant) > 1; CFL safety factor is usually <= 1")
    end

    if !(params.magnetic_inner_bc in (:insulating, :conducting_inner_core))
        push!(errors,
            "magnetic_inner_bc = $(params.magnetic_inner_bc) must be :insulating or :conducting_inner_core")
    end
    if params.magnetic_inner_bc === :conducting_inner_core && params.geometry !== :shell
        push!(errors, "magnetic_inner_bc=:conducting_inner_core requires geometry=:shell")
    end

    if params.lmax < 1
        push!(errors, "lmax = $(params.lmax) must be >= 1")
    end

    if params.mmax < 0 || params.mmax > params.lmax
        push!(errors, "mmax = $(params.mmax) must be in range [0, lmax=$(params.lmax)]")
    end

    # The transform itself requires nlat >= lmax+1 and nlon >= 2*mmax+1 (Nyquist);
    # below that the grid is invalid (not merely aliased) and initialize_solver_state
    # would throw deep in the transform setup. Report it as an error here.
    if params.nlat < params.lmax + 1
        push!(errors,
            "nlat = $(params.nlat) must be >= lmax+1 = $(params.lmax + 1) for the transform")
    elseif params.nlat < 2 * params.lmax
        push!(warnings, "nlat = $(params.nlat) < 2*lmax = $(2 * params.lmax); may cause aliasing")
    end

    if params.nlon < 2 * params.mmax + 1
        push!(errors,
            "nlon = $(params.nlon) must be >= 2*mmax+1 = $(2 * params.mmax + 1) (Nyquist) for the transform")
    end

    if !(params.geometry in (:shell, :ball))
        push!(errors, "geometry = $(params.geometry) must be :shell or :ball")
    elseif params.geometry === :shell
        if !(0.0 < params.radius_ratio < 1.0)
            push!(errors,
                "radius_ratio = $(params.radius_ratio) must be in range (0, 1) for shell geometry")
        end
    elseif params.geometry === :ball && params.radius_ratio != 0.0
        push!(errors,
            "geometry = :ball but radius_ratio = $(params.radius_ratio) != 0; use radius_ratio=0 for full ball")
    end

    if params.Ra <= 0.0
        push!(errors, "Ra = $(params.Ra) must be positive")
    elseif params.Ra > 1e10
        push!(warnings, "Ra = $(params.Ra) is very large; ensure numerical stability")
    end

    if params.Ek <= 0.0
        push!(errors, "Ek = $(params.Ek) must be positive")
    elseif params.Ek < 1e-8
        push!(warnings, "Ek = $(params.Ek) is very small; may require fine resolution")
    end

    params.Pr > 0.0 || push!(errors, "Pr = $(params.Pr) must be positive")
    params.Pm > 0.0 || push!(errors, "Pm = $(params.Pm) must be positive")
    params.Sc > 0.0 || push!(errors, "Sc = $(params.Sc) must be positive")

    if params.timestep <= 0.0
        push!(errors, "timestep = $(params.timestep) must be positive")
    elseif params.timestep > 1.0
        push!(warnings, "timestep = $(params.timestep) is very large; check CFL condition")
    end

    if !(params.timestepper isa Union{CNAB2, ExponentialRungeKutta2, RungeKutta3})
        supported = "CNAB2, ExponentialRungeKutta2, or RungeKutta3"
        push!(errors,
            "timestepper = $(nameof(typeof(params.timestepper))) is not supported by the solver; " *
            "use $supported")
    end

    params.stop_iteration >= 1 ||
        push!(errors, "stop_iteration = $(params.stop_iteration) must be >= 1")

    if params.end_time <= params.start_time
        push!(errors,
            "end_time = $(params.end_time) must be greater than start_time = $(params.start_time)")
    end

    max_diffusivity = max(1.0, params.Pm / params.Pr, params.Pm / params.Sc, params.Ek)
    cfl_limit = 0.1 / (params.lmax^2 * max_diffusivity)
    if params.timestep > cfl_limit
        push!(warnings,
            "timestep = $(params.timestep) may violate CFL condition " *
            "(estimated limit: $(cfl_limit) for spectral stability with " *
            "max diffusivity = $(max_diffusivity))")
    end

    if params.output_precision ∉ (:float32, :float64)
        push!(errors, "output_precision = $(params.output_precision) must be :float32 or :float64")
    end

    return errors, warnings
end

"""
    validate_parameters(params; strict=false)

Validate a `SolverParameters` object.
"""
function validate_parameters(params::SolverParameters; strict::Bool = false)
    errors, warnings = _parameter_errors_warnings(params)
    is_valid = isempty(errors)

    if _parameter_rank0()
        if !isempty(errors)
            println("\nPARAMETER VALIDATION ERRORS:")
            for (i, err) in enumerate(errors)
                println("  $i. $err")
            end
        end

        if !isempty(warnings)
            println("\nPARAMETER VALIDATION WARNINGS:")
            for (i, warn) in enumerate(warnings)
                println("  $i. $warn")
            end
        end

        if is_valid && isempty(warnings)
            println("\nAll parameters validated successfully")
        end
    end

    if strict && !is_valid
        error("Parameter validation failed with $(length(errors)) error(s). " *
              "Fix parameters or set strict=false to proceed with warnings.")
    end

    return (is_valid, errors, warnings)
end

"""
    find_package_root()

Find the root directory of the GeoDynamo.jl package.
"""
function find_package_root()
    current_dir = @__DIR__

    while current_dir != "/"
        project_file = joinpath(current_dir, "Project.toml")
        if isfile(project_file)
            try
                content = read(project_file, String)
                if contains(content, "GeoDynamo") ||
                   contains(content, "name = \"GeoDynamo\"")
                    return current_dir
                end
            catch
            end
        end
        current_dir = dirname(current_dir)
    end

    @warn "Could not find GeoDynamo.jl package root. Using current directory."
    return dirname(@__DIR__)
end

function _default_parameter_file()
    return joinpath(find_package_root(), "config", "default_params.jl")
end

"""
    safe_parse_value(value_str, param_dict)

Parse a parameter value without `eval`.
"""
function safe_parse_value(value_str::AbstractString, param_dict::Dict{Symbol, Any})
    s = strip(value_str)

    s == "true" && return true
    s == "false" && return false
    s == "nothing" && return nothing
    (s == "π" || s == "pi") && return π

    if startswith(s, ':')
        return Symbol(s[2:end])
    end

    if startswith(s, '"') && endswith(s, '"')
        return s[2:(end - 1)]
    end

    int_val = tryparse(Int, s)
    int_val !== nothing && return int_val

    float_val = tryparse(Float64, s)
    float_val !== nothing && return float_val

    expr = Meta.parse(s)
    return safe_eval_expr(expr, param_dict)
end

const _SAFE_OPS = Set{Symbol}([:+, :-, :*, :/, :÷, :^, :div, :mod, :min, :max, :sqrt, :abs])
const _SAFE_PARAMETER_CONSTRUCTORS = Dict{Symbol, Any}(
    :CNAB2 => CNAB2,
    :ExponentialAdamsBashforth2 => ExponentialAdamsBashforth2,
    :EAB2 => ExponentialAdamsBashforth2,
    :ExponentialRungeKutta2 => ExponentialRungeKutta2,
    :ERK2 => ExponentialRungeKutta2,
    :RungeKutta3 => RungeKutta3,
    :CB3 => RungeKutta3,
    :ETD => ETD,
    :ThetaMethod => ThetaMethod,
    :NoSlip => NoSlip,
    :StressFree => StressFree,
    :FixedTemperature => FixedTemperature,
    :FixedFlux => FixedFlux,
    :ValueBoundaryCondition => ValueBoundaryCondition,
    :FluxBoundaryCondition => FluxBoundaryCondition,
    :InsulatingMagnetic => InsulatingMagnetic,
    :ConductingMagnetic => ConductingMagnetic,
    :BoundaryConditions => BoundaryConditions,
    :FieldBoundaryConditions => FieldBoundaryConditions
)

function safe_eval_expr(expr, param_dict::Dict{Symbol, Any})
    expr isa Number && return expr

    if expr isa Symbol
        expr === :π && return π
        expr === :pi && return π
        haskey(param_dict, expr) && return param_dict[expr]
        throw(ArgumentError("Unknown parameter reference: $expr"))
    end

    expr isa QuoteNode && return expr.value

    if !(expr isa Expr)
        throw(ArgumentError("Unsupported expression type: $(typeof(expr))"))
    end

    if expr.head === :call
        op = expr.args[1]
        if op isa Symbol && haskey(_SAFE_PARAMETER_CONSTRUCTORS, op)
            args = Any[]
            kwargs = Pair{Symbol, Any}[]
            for arg in expr.args[2:end]
                if arg isa Expr && arg.head === :kw
                    push!(kwargs, arg.args[1] => safe_eval_expr(arg.args[2], param_dict))
                else
                    push!(args, safe_eval_expr(arg, param_dict))
                end
            end
            return _SAFE_PARAMETER_CONSTRUCTORS[op](args...; kwargs...)
        end

        if !(op isa Symbol) || op ∉ _SAFE_OPS
            throw(ArgumentError("Disallowed operation in parameter file: $op"))
        end
        args = [safe_eval_expr(a, param_dict) for a in expr.args[2:end]]
        return getfield(Base, op)(args...)
    elseif expr.head === :parens || expr.head === :block
        return safe_eval_expr(expr.args[end], param_dict)
    else
        throw(ArgumentError("Unsupported expression form in parameter file: $(expr.head)"))
    end
end

const _LEGACY_PARAM_ALIASES = Dict{Symbol, Symbol}(:max_steps => :stop_iteration)

function _parameter_assignments_from_file(config_file::String)
    param_dict = Dict{Symbol, Any}()
    content = read(config_file, String)

    for line in split(content, '\n')
        line = strip(line)
        isempty(line) && continue
        startswith(line, "#") && continue

        match_result = match(r"^(?:const\s+)?([A-Za-z]\w*)\s*=\s*([^#]+)", line)
        match_result === nothing && continue

        param_name = Symbol(match_result.captures[1])
        param_value_str = strip(match_result.captures[2])

        if haskey(_LEGACY_PARAM_ALIASES, param_name)
            new_name = _LEGACY_PARAM_ALIASES[param_name]
            @warn "Parameter `$param_name` is deprecated; use `$new_name`."
            param_name = new_name
        end

        if param_name ∉ fieldnames(SolverParameters)
            @warn "Ignoring unknown solver parameter `$param_name` in $config_file"
            continue
        end

        try
            param_dict[param_name] = safe_parse_value(param_value_str, param_dict)
        catch e
            @warn "Could not parse parameter `$param_name = $param_value_str` in $config_file (kept default): $e"
        end
    end

    return param_dict
end

"""
    load_parameters_from_file(config_file)

Load a Julia parameter file containing assignments that match
`fieldnames(SolverParameters)`.
"""
function load_parameters_from_file(config_file::String)
    if !isfile(config_file)
        @warn "Parameter file not found: $config_file. Using default parameters."
        return SolverParameters()
    end

    try
        kwargs = _parameter_assignments_from_file(config_file)
        return SolverParameters(; kwargs...)
    catch e
        @error "Error reading parameter file $config_file: $e"
        return SolverParameters()
    end
end

"""
    load_parameters([config_file])

Load solver parameters from a file. With no file, loads
`config/default_params.jl` when present, otherwise uses `SolverParameters()`.
"""
function load_parameters(config_file::String = "")
    # An explicitly named file that does not exist is a hard error: silently
    # falling back to defaults hides a typo or a missing config. Only the
    # implicit default-file path is allowed to fall back when absent.
    if !isempty(config_file) && !isfile(config_file)
        throw(ArgumentError("load_parameters: parameter file not found: $config_file"))
    end
    path = isempty(config_file) ? _default_parameter_file() : config_file
    params = load_parameters_from_file(path)
    # strict=true throws when the loaded values are invalid, instead of
    # returning known-bad parameters that fail later inside the solver.
    validate_parameters(params; strict = true)
    return params
end

@inline _parameter_literal(value::Symbol) = ":" * String(value)
@inline _parameter_literal(value::AbstractString) = repr(value)
@inline _parameter_literal(value::AbstractTimestepper) = string(value)
@inline _parameter_literal(value::BoundaryConditions) = string(value)
@inline _parameter_literal(value) = repr(value)

"""
    save_parameters(params, filename)

Write a `SolverParameters` object as plain Julia assignments.
"""
function save_parameters(params::SolverParameters, filename::String)
    open(filename, "w") do io
        println(io, "# GeoDynamo.jl solver parameters")
        println(io, "# Generated on $(now())")
        println(io)
        for name in fieldnames(SolverParameters)
            println(io, name, " = ", _parameter_literal(getfield(params, name)))
        end
    end
    @info "Parameters saved to $filename"
    return filename
end

function create_parameter_template(filename::String)
    save_parameters(SolverParameters(), filename)
    @info "Parameter template created at $filename"
    return filename
end

const GEODYNAMO_PARAMS = Ref{Union{SolverParameters, Nothing}}(nothing)
const PARAMS_LOCK = ReentrantLock()

function get_parameters()::SolverParameters
    params = GEODYNAMO_PARAMS[]
    params !== nothing && return params

    lock(PARAMS_LOCK) do
        params = GEODYNAMO_PARAMS[]
        params !== nothing && return params

        params = load_parameters()
        GEODYNAMO_PARAMS[] = params
        return params
    end
end

@inline active_parameters()::SolverParameters = get_parameters()

function set_parameters!(params::SolverParameters; validate::Bool = true, strict::Bool = false)
    if validate
        is_valid, _, _ = validate_parameters(params; strict = strict)
        if !is_valid && _parameter_rank0()
            @warn "Setting parameters despite validation errors. Set strict=true to enforce validation."
        end
    end

    lock(PARAMS_LOCK) do
        GEODYNAMO_PARAMS[] = params
    end
    return params
end

function initialize_parameters(config_file::String = "")
    params = load_parameters(config_file)
    set_parameters!(params; validate = false)
    return params
end
