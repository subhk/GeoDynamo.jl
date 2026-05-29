# ================================================================================
# Field Information
# ================================================================================

"""
    FieldInfo

Compact description of the field dimensions, coordinates, and optional pencil
metadata needed by the NetCDF writer.

`extract_field_info(...)` builds this from a field snapshot and optional
transform/pencil metadata.
"""
struct FieldInfo
    # Physical dimensions (for temperature/composition)
    nlat::Int
    nlon::Int
    nr::Int

    # Spectral dimensions (for velocity/magnetic)
    nlm::Int

    # Coordinate arrays
    theta::Vector{Float64}
    phi::Vector{Float64}
    r::Vector{Float64}
    l_values::Vector{Int}
    m_values::Vector{Int}

    # Pencil decomposition information
    has_pencils::Bool
    pencils::NamedTuple
    has_config::Bool
    config::Union{SHTnsKitConfig, Nothing}

    # Local axis range information
    local_ranges::Dict{Symbol, UnitRange{Int}}

    # Owned spectral mode indices for mapped 2D spectral pencils
    local_spectral_modes::Vector{Int}
end

"""
    FieldInfo()

Construct an empty `FieldInfo` placeholder for tests or call paths that fill
dimensions later.
"""
function FieldInfo()
    return FieldInfo(0, 0, 0, 0, Float64[], Float64[], Float64[],
        Int[], Int[], false, NamedTuple(), false, nothing,
        Dict{Symbol, UnitRange{Int}}(), Int[])
end

"""
    FieldInfo(nlat, nlon, nr, nlm, theta, phi, r, l_values, m_values, pencils, config, local_ranges, local_spectral_modes)

Construct a fully populated `FieldInfo` when both SHTns configuration and pencil
decomposition metadata are available.
"""
function FieldInfo(nlat::Int, nlon::Int, nr::Int, nlm::Int,
        theta::Vector{Float64}, phi::Vector{Float64}, r::Vector{Float64},
        l_values::Vector{Int}, m_values::Vector{Int},
        pencils::NamedTuple, config::SHTnsKitConfig,
        local_ranges::Dict{Symbol, UnitRange{Int}},
        local_spectral_modes::Vector{Int})
    return FieldInfo(nlat, nlon, nr, nlm, theta, phi, r, l_values, m_values,
        true, pencils, true, config, local_ranges, local_spectral_modes)
end

"""
    extract_field_info(fields[, config, pencils])

Infer a `FieldInfo` description from a dictionary-style field snapshot.

This bridges the simulation-side field containers and the output writer’s more
uniform metadata needs.
"""
function extract_field_info(
        fields::Dict{String, Any}, config::Union{SHTnsKitConfig, Nothing} = nothing,
        pencils::Union{NamedTuple, Nothing} = nothing;
        radius_ratio::Float64 = 0.35,
        radial_grid::Union{AbstractVector{<:Real}, Nothing} = nothing)
    nlat = 0
    nlon = 0
    nr = 0
    nlm = 0

    # Get physical dimensions from temperature
    if haskey(fields, "temperature")
        temp_dims = size(fields["temperature"])
        nlat, nlon, nr = temp_dims[1], temp_dims[2], temp_dims[3]
    end

    # Get physical dimensions from composition if temperature not available
    if nlat == 0 && haskey(fields, "composition")
        comp_dims = size(fields["composition"])
        nlat, nlon, nr = comp_dims[1], comp_dims[2], comp_dims[3]
    end

    # Get spectral dimensions
    if haskey(fields, "velocity_toroidal") && haskey(fields["velocity_toroidal"], "real")
        spec_dims = size(fields["velocity_toroidal"]["real"])
        nlm = spec_dims[1]
        if nr == 0
            nr = spec_dims[end]
        end
    elseif haskey(fields, "magnetic_toroidal") &&
           haskey(fields["magnetic_toroidal"], "real")
        spec_dims = size(fields["magnetic_toroidal"]["real"])
        nlm = spec_dims[1]
        if nr == 0
            nr = spec_dims[end]
        end
    elseif haskey(fields, "temperature_spectral") &&
           haskey(fields["temperature_spectral"], "real")
        spec_dims = size(fields["temperature_spectral"]["real"])
        nlm = spec_dims[1]
        if nr == 0
            nr = spec_dims[end]
        end
    elseif haskey(fields, "composition_spectral") &&
           haskey(fields["composition_spectral"], "real")
        spec_dims = size(fields["composition_spectral"]["real"])
        nlm = spec_dims[1]
        if nr == 0
            nr = spec_dims[end]
        end
    end

    # Create coordinate arrays
    theta = nlat > 0 ? collect(range(0, π, length = nlat)) : Float64[]
    phi = nlon > 0 ? collect(range(0, 2π, length = nlon)) : Float64[]
    r = nr > 0 ? collect(range(radius_ratio, 1.0, length = nr)) : Float64[]

    # Create l,m values for spectral modes
    l_values = Int[]
    m_values = Int[]
    if nlm > 0
        # Invert nlm = (lmax+1)(lmax+2)/2 via quadratic formula
        lmax = Int(floor((-3 + sqrt(1 + 8*nlm)) / 2))
        for l in 0:lmax
            for m in 0:l
                if length(l_values) < nlm
                    push!(l_values, l)
                    push!(m_values, m)
                end
            end
        end
    end

    # Use config information if available
    if config !== nothing
        nlat = config.nlat
        nlon = config.nlon
        nlm = config.nlm
        l_values = config.l_values[1:min(length(config.l_values), nlm)]
        m_values = config.m_values[1:min(length(config.m_values), nlm)]
        theta = config.theta_grid
        phi = config.phi_grid
    end

    # Prefer the true radial collocation grid when the caller supplies it. The
    # equispaced range fabricated above does not match the Chebyshev-clustered
    # nodes the solver actually integrates on, so without this the NetCDF "r"
    # coordinate misplaces every radial data point.
    if radial_grid !== nothing && length(radial_grid) == nr
        r = collect(Float64, radial_grid)
    end

    # Extract local ownership metadata if pencils are provided. Spectral modes
    # are not a simple axis range for rectangular 2D (l, m, r) pencils.
    local_ranges = Dict{Symbol, UnitRange{Int}}()
    local_spectral_modes = Int[]
    if pencils !== nothing
        try
            local_ranges[:r] = range_local(pencils.r, 3)
            local_ranges[:θ] = range_local(pencils.θ, 1)
            local_ranges[:φ] = range_local(pencils.φ, 2)
            if config !== nothing
                local_spectral_modes = local_spectral_mode_indices(config)
            else
                legacy_spec_range = range_local(pencils.spec, 1)
                local_ranges[:spec] = legacy_spec_range
                local_spectral_modes = collect(legacy_spec_range)
            end
        catch
            # Fallback if pencil ranges not available
        end
    elseif config !== nothing
        local_spectral_modes = local_spectral_mode_indices(config)
    end

    # Create FieldInfo with type-stable constructor
    if pencils !== nothing && config !== nothing
        return FieldInfo(nlat, nlon, nr, nlm, theta, phi, r, l_values, m_values,
            pencils, config, local_ranges, local_spectral_modes)
    else
        dummy_pencils = NamedTuple()

        return FieldInfo(nlat, nlon, nr, nlm, theta, phi, r, l_values, m_values,
            pencils !== nothing, pencils !== nothing ? pencils : dummy_pencils,
            config !== nothing, config,
            local_ranges, local_spectral_modes)
    end
end
