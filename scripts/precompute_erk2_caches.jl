#!/usr/bin/env julia

using Logging
using Geodynamo
using MPI

const FIELD_ALIASES = Dict(
    "temperature" => :temperature,
    "temp" => :temperature,
    "velocity_toroidal" => :velocity_toroidal,
    "vel_tor" => :velocity_toroidal,
    "velocity_poloidal" => :velocity_poloidal,
    "vel_pol" => :velocity_poloidal,
    "magnetic_toroidal" => :magnetic_toroidal,
    "mag_tor" => :magnetic_toroidal,
    "magnetic_poloidal" => :magnetic_poloidal,
    "mag_pol" => :magnetic_poloidal,
    "composition" => :composition,
    "all" => :all,
)

struct ScriptOptions
    params_path::Union{Nothing,String}
    dt::Union{Nothing,Float64}
    output::String
    fields::Vector{Symbol}
    precision::DataType
    include_composition::Bool
end

function usage()
    println("""
    Usage: julia --project scripts/precompute_erk2_caches.jl [options]

    Options:
      --params=FILE           Parameter file to load (defaults to config/default_params.jl)
      --dt=VALUE              Time step used for the ERK2 cache (defaults to d_timestep)
      --output=FILE           Output JLD2 file (defaults to ./erk2_caches.jld2)
      --fields=list           Comma-separated list of fields (temperature,vel_tor,mag_pol,...).
                              Use 'all' to precompute every available field.
      --precision=TYPE        Floating-point precision (Float64 or Float32, default Float64)
      --no-composition        Skip composition cache even if enabled in parameters
      -h, --help              Show this help message

    Example:
      julia --project scripts/precompute_erk2_caches.jl --dt=1.0e-4 --fields=temperature,vel_tor --output=caches.jld2
    """)
end

function canonical_field(name::String)
    key = replace(lowercase(strip(name)), '-' => '_')
    haskey(FIELD_ALIASES, key) || error("Unknown field alias '$name'")
    return FIELD_ALIASES[key]
end

function parse_precision(value::String)
    key = lowercase(strip(value))
    key in ("float64", "float", "double", "f64") && return Float64
    key in ("float32", "single", "f32") && return Float32
    error("Unsupported precision '$value'. Use Float64 or Float32.")
end

function parse_args(args)::ScriptOptions
    params_path = nothing
    dt = nothing
    output = abspath("erk2_caches.jld2")
    fields = Symbol[]
    precision = Float64
    include_composition = true

    for arg in args
        if arg in ("-h", "--help")
            usage()
            exit(0)
        elseif startswith(arg, "--params=")
            params_path = abspath(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--dt=")
            dt = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--output=")
            output = abspath(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--fields=")
            field_list = split(split(arg, "=", limit=2)[2], ",")
            fields = [canonical_field(f) for f in field_list if !isempty(strip(f))]
        elseif startswith(arg, "--precision=")
            precision = parse_precision(split(arg, "=", limit=2)[2])
        elseif arg == "--no-composition"
            include_composition = false
        else
            error("Unknown argument '$arg'. Use --help for usage.")
        end
    end

    return ScriptOptions(params_path, dt, output, fields, precision, include_composition)
end

function preferred_fields(state, include_composition::Bool)
    specs = Dict{Symbol,Float64}()
    specs[:temperature] = 1.0 / Geodynamo.d_Pr
    specs[:velocity_toroidal] = Geodynamo.d_E
    specs[:velocity_poloidal] = Geodynamo.d_E

    if Geodynamo.i_B == 1 && state.magnetic !== nothing
        specs[:magnetic_toroidal] = 1.0 / Geodynamo.d_Pm
        specs[:magnetic_poloidal] = 1.0 / Geodynamo.d_Pm
    end

    if include_composition && state.composition !== nothing
        specs[:composition] = 1.0 / Geodynamo.d_Sc
    end

    return specs
end

function main()
    opts = parse_args(ARGS)

    finalize_mpi = false
    if !MPI.Initialized()
        MPI.Init()
        finalize_mpi = true
    end

    try
        if opts.params_path !== nothing
            Geodynamo.initialize_parameters(opts.params_path)
        else
            Geodynamo.initialize_parameters()
        end

        dt = opts.dt === nothing ? Geodynamo.d_timestep : opts.dt
        dt <= 0 && error("Time step must be positive, got $dt")

        state = Geodynamo.initialize_simulation(opts.precision; include_composition=opts.include_composition)
        T = opts.precision

        specs = preferred_fields(state, opts.include_composition)
        available_fields = collect(keys(specs))
        selected_fields = if isempty(opts.fields) || (:all in opts.fields)
            available_fields
        else
            opts.fields
        end

        for field in selected_fields
            haskey(specs, field) || error("Requested field '$field' is not available with current parameters")
        end

        bundle = Dict{Symbol,Geodynamo.ERK2Cache{T}}()
        for field in selected_fields
            ν = specs[field]
            cache = Geodynamo.get_erk2_cache!(state.erk2_caches, field, ν, opts.precision,
                                              state.shtns_config, state.oc_domain, dt;
                                              use_krylov=false, m=Geodynamo.i_etd_m, tol=Geodynamo.d_krylov_tol)
            bundle[field] = cache
            @info "Computed ERK2 cache" field step_dt=dt diffusivity=ν radial_resolution=state.oc_domain.N
        end

        metadata = Dict{String,Any}(
            "dt" => dt,
            "precision" => string(opts.precision),
            "fields" => String.(selected_fields),
            "geometry" => string(state.geometry),
            "nr" => state.oc_domain.N,
            "created_rank" => Geodynamo.get_rank(),
            "i_etd_m" => Geodynamo.i_etd_m,
            "d_krylov_tol" => Geodynamo.d_krylov_tol,
        )

        if opts.params_path !== nothing
            metadata["params_path"] = opts.params_path
        end

        Geodynamo.save_erk2_cache_bundle(opts.output, bundle; metadata=metadata)
        @info "Saved ERK2 caches" output=opts.output metadata=metadata
    finally
        finalize_mpi && MPI.Finalize()
    end
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
