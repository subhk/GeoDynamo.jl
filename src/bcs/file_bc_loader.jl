# ================================================================================
# File-Based Boundary Condition Loader
# ================================================================================
#
# Loads boundary conditions from NetCDF files (physical-space or pre-computed
# spectral format), transforms to spectral coefficients, and stores them for
# use during implicit solve steps.
#
# Pattern:
#   load boundary data -> extract Re/Im components -> set BCs
#
# Supported NetCDF formats:
#   :physical  - Physical-space data on (nlat, nlon) grid, transformed via SHTnsKit
#   :spectral  - Pre-computed spectral coefficients Cbc_Re[2,nlm] / Cbc_Im[2,nlm]
#
# Note: This file is included within the bcs module.
# All necessary packages are imported at the module level.
# ================================================================================

"""
    SpectralBoundaryCoefficients{T}

Stores spectral boundary condition coefficients for inner and outer boundaries.
Row 1 = inner boundary, Row 2 = outer boundary (matching Fortran convention).

# Fields
- `bc_real::Matrix{T}`: Real part of spectral BCs [2, nlm]
- `bc_imag::Matrix{T}`: Imaginary part of spectral BCs [2, nlm]
- `nlm::Int`: Number of spectral modes
- `source_file::String`: Path to the source file
- `source_format::Symbol`: Format used to load (:physical or :spectral)
"""
struct SpectralBoundaryCoefficients{T <: AbstractFloat}
    bc_real::Matrix{T}   # [2, nlm] - row 1=inner, row 2=outer
    bc_imag::Matrix{T}   # [2, nlm] - row 1=inner, row 2=outer
    nlm::Int
    source_file::String
    source_format::Symbol
end

"""
    BoundaryInterpolationCache{T}

Typed storage for spectral boundary condition coefficients owned by a field.
The string-key accessors keep legacy boundary helper code working, while hot
solver paths can read the concrete matrix fields directly.
"""
mutable struct BoundaryInterpolationCache{T <: AbstractFloat}
    bc_real::Union{Matrix{T}, Nothing}
    bc_imag::Union{Matrix{T}, Nothing}
    bc_loaded::Bool
    bc_source_file::String
    bc_source_format::Symbol
    bc_nlm::Int
    metadata::Dict{String, Any}
end

"""
    active_boundary_arrays(field) -> (real_array, imag_array)

The boundary-value arrays the solver will ACTUALLY read for `field`, as `(2, nlm)`
matrices (row 1 = inner, row 2 = outer). Either slot may be `nothing`.

A field can hold two different sets of boundary values: the parameter-derived
`boundary_values` / `boundary_values_imag`, and — once a spectral BC file is loaded —
the interpolation cache's `bc_real` / `bc_imag`. The cache WINS when it is loaded, so
anything that means to influence the implicit solve has to write to whichever set is
live. Writing unconditionally to `boundary_values` is a silent no-op in every run that
loads a spectral BC file, which is exactly how the topography couplings were losing
their corrections.

This is the single place that decides which set is live; `get_bc_vectors`
(solver/numerics.jl) reads through it and the topography couplings write through it, so
the two cannot drift apart.
"""
function active_boundary_arrays(field)
    cache = hasfield(typeof(field), :boundary_interpolation_cache) ?
            getfield(field, :boundary_interpolation_cache) : nothing
    if cache isa BoundaryInterpolationCache
        bc_real = cache.bc_real
        bc_imag = cache.bc_imag
        if cache.bc_loaded && bc_real !== nothing && bc_imag !== nothing
            return (bc_real, bc_imag)
        end
    end
    values = hasfield(typeof(field), :boundary_values) ?
             getfield(field, :boundary_values) : nothing
    values_imag = hasfield(typeof(field), :boundary_values_imag) ?
                  getfield(field, :boundary_values_imag) : nothing
    return (values, values_imag)
end

function BoundaryInterpolationCache(::Type{T} = Float64) where {T <: AbstractFloat}
    BoundaryInterpolationCache{T}(
        nothing, nothing, false, "", :unknown, 0, Dict{String, Any}())
end

function Base.empty!(cache::BoundaryInterpolationCache)
    cache.bc_real = nothing
    cache.bc_imag = nothing
    cache.bc_loaded = false
    cache.bc_source_file = ""
    cache.bc_source_format = :unknown
    cache.bc_nlm = 0
    empty!(cache.metadata)
    return cache
end

function Base.haskey(cache::BoundaryInterpolationCache, key::AbstractString)
    if key == "bc_real"
        return cache.bc_real !== nothing
    elseif key == "bc_imag"
        return cache.bc_imag !== nothing
    elseif key == "bc_loaded"
        return true
    elseif key == "bc_source_file"
        return !isempty(cache.bc_source_file)
    elseif key == "bc_source_format"
        return cache.bc_source_format !== :unknown
    elseif key == "bc_nlm"
        return cache.bc_nlm != 0
    end
    return haskey(cache.metadata, String(key))
end

function Base.getindex(cache::BoundaryInterpolationCache, key::AbstractString)
    if key == "bc_real"
        cache.bc_real === nothing && throw(KeyError(key))
        return cache.bc_real
    elseif key == "bc_imag"
        cache.bc_imag === nothing && throw(KeyError(key))
        return cache.bc_imag
    elseif key == "bc_loaded"
        return cache.bc_loaded
    elseif key == "bc_source_file"
        return cache.bc_source_file
    elseif key == "bc_source_format"
        return cache.bc_source_format
    elseif key == "bc_nlm"
        return cache.bc_nlm
    end
    return cache.metadata[String(key)]
end

function Base.setindex!(cache::BoundaryInterpolationCache{T}, value, key::AbstractString) where {T}
    if key == "bc_real"
        cache.bc_real = value isa Matrix{T} ? value : T.(value)
    elseif key == "bc_imag"
        cache.bc_imag = value isa Matrix{T} ? value : T.(value)
    elseif key == "bc_loaded"
        cache.bc_loaded = Bool(value)
    elseif key == "bc_source_file"
        cache.bc_source_file = String(value)
    elseif key == "bc_source_format"
        cache.bc_source_format = Symbol(value)
    elseif key == "bc_nlm"
        cache.bc_nlm = Int(value)
    else
        cache.metadata[String(key)] = value
    end
    return cache
end

function Base.get(cache::BoundaryInterpolationCache, key::AbstractString, default)
    haskey(cache, key) ? cache[key] : default
end

function _boundary_cache_pairs(cache::BoundaryInterpolationCache)
    pairs = Pair{String, Any}[]
    cache.bc_real === nothing || push!(pairs, "bc_real" => cache.bc_real)
    cache.bc_imag === nothing || push!(pairs, "bc_imag" => cache.bc_imag)
    cache.bc_loaded && push!(pairs, "bc_loaded" => true)
    isempty(cache.bc_source_file) || push!(pairs, "bc_source_file" => cache.bc_source_file)
    cache.bc_source_format === :unknown ||
        push!(pairs, "bc_source_format" => cache.bc_source_format)
    cache.bc_nlm == 0 || push!(pairs, "bc_nlm" => cache.bc_nlm)
    append!(pairs, cache.metadata)
    return pairs
end

Base.length(cache::BoundaryInterpolationCache) = length(_boundary_cache_pairs(cache))
Base.isempty(cache::BoundaryInterpolationCache) = length(cache) == 0
Base.iterate(cache::BoundaryInterpolationCache) = iterate(_boundary_cache_pairs(cache))
function Base.iterate(cache::BoundaryInterpolationCache, state)
    iterate(_boundary_cache_pairs(cache), state)
end

"""
    load_spectral_bc_from_file(filename::String, config; format::Symbol=:physical, T::Type=Float64)

Load boundary conditions from a NetCDF file and return spectral coefficients.

# Arguments
- `filename`: Path to the NetCDF file
- `config`: SHTnsKitConfig (or any object with `lmax`, `mmax`, `nlm` fields)
- `format`: `:physical` for physical-space data, `:spectral` for pre-computed coefficients
- `T`: Numeric type for the coefficients (default Float64)

# NetCDF File Formats

**Physical format** (`:physical`):
- Variables: `"inner"` with shape `(nlat, nlon)` and/or `"outer"` with shape `(nlat, nlon)`
- Or a single variable with 3rd dimension of size 2: `"temperature"` with shape `(nlat, nlon, 2)`
  (`slice 1 = inner, slice 2 = outer`)

**Spectral format** (`:spectral`, Fortran-compatible):
- Variables: `"Cbc_Re"` [2, nlm] and `"Cbc_Im"` [2, nlm]
- Row 1 = inner boundary, Row 2 = outer boundary

# Returns
- `SpectralBoundaryCoefficients{T}` with spectral BC data
"""
function load_spectral_bc_from_file(filename::String, config;
        format::Symbol = :physical,
        T::Type{<:AbstractFloat} = Float64)
    isfile(filename) || throw(ArgumentError("BC file not found: $filename"))
    nlm = config.nlm

    if format === :physical
        return _load_physical_format(filename, config, T)
    elseif format === :spectral
        return _load_spectral_format(filename, nlm, T)
    else
        throw(ArgumentError("Unknown BC format: $format. Use :physical or :spectral."))
    end
end

"""
Load physical-space boundary data from NetCDF, transform to spectral via SHTnsKit.
"""
function _load_physical_format(filename::String, config, ::Type{T}) where {T <:
                                                                           AbstractFloat}
    nlm = config.nlm
    bc_real = zeros(T, 2, nlm)
    bc_imag = zeros(T, 2, nlm)

    NCDatasets.Dataset(filename, "r") do ds
        # Try separate inner/outer variables first
        if haskey(ds, "inner") || haskey(ds, "outer")
            if haskey(ds, "inner")
                inner_phys = T.(Array(ds["inner"]))
                coeffs = shtns_physical_to_spectral(inner_phys, config; return_complex = true)
                for i in 1:min(length(coeffs), nlm)
                    bc_real[1, i] = T(real(coeffs[i]))
                    bc_imag[1, i] = T(imag(coeffs[i]))
                end
            end
            if haskey(ds, "outer")
                outer_phys = T.(Array(ds["outer"]))
                coeffs = shtns_physical_to_spectral(outer_phys, config; return_complex = true)
                for i in 1:min(length(coeffs), nlm)
                    bc_real[2, i] = T(real(coeffs[i]))
                    bc_imag[2, i] = T(imag(coeffs[i]))
                end
            end
        elseif haskey(ds, "temperature")
            # Single variable with 3rd dim: [nlat, nlon, 2]
            data = T.(Array(ds["temperature"]))
            if ndims(data) == 3 && size(data, 3) >= 2
                inner_phys = data[:, :, 1]
                coeffs = shtns_physical_to_spectral(inner_phys, config; return_complex = true)
                for i in 1:min(length(coeffs), nlm)
                    bc_real[1, i] = T(real(coeffs[i]))
                    bc_imag[1, i] = T(imag(coeffs[i]))
                end
                outer_phys = data[:, :, 2]
                coeffs = shtns_physical_to_spectral(outer_phys, config; return_complex = true)
                for i in 1:min(length(coeffs), nlm)
                    bc_real[2, i] = T(real(coeffs[i]))
                    bc_imag[2, i] = T(imag(coeffs[i]))
                end
            else
                @warn "Variable 'temperature' in $filename does not have expected shape [nlat, nlon, 2]"
            end
        elseif haskey(ds, "composition")
            # Composition variant of the single-variable format
            data = T.(Array(ds["composition"]))
            if ndims(data) == 3 && size(data, 3) >= 2
                inner_phys = data[:, :, 1]
                coeffs = shtns_physical_to_spectral(inner_phys, config; return_complex = true)
                for i in 1:min(length(coeffs), nlm)
                    bc_real[1, i] = T(real(coeffs[i]))
                    bc_imag[1, i] = T(imag(coeffs[i]))
                end
                outer_phys = data[:, :, 2]
                coeffs = shtns_physical_to_spectral(outer_phys, config; return_complex = true)
                for i in 1:min(length(coeffs), nlm)
                    bc_real[2, i] = T(real(coeffs[i]))
                    bc_imag[2, i] = T(imag(coeffs[i]))
                end
            else
                @warn "Variable 'composition' in $filename does not have expected shape [nlat, nlon, 2]"
            end
        else
            throw(ArgumentError("NetCDF file $filename does not contain expected variables " *
                                "(\"inner\"/\"outer\", \"temperature\", or \"composition\")"))
        end
    end

    return SpectralBoundaryCoefficients{T}(bc_real, bc_imag, nlm, filename, :physical)
end

"""
Load pre-computed spectral boundary coefficients from NetCDF (Fortran-compatible format).
"""
function _load_spectral_format(filename::String, nlm::Int, ::Type{T}) where {T <:
                                                                             AbstractFloat}
    bc_real = zeros(T, 2, nlm)
    bc_imag = zeros(T, 2, nlm)

    NCDatasets.Dataset(filename, "r") do ds
        haskey(ds, "Cbc_Re") || throw(ArgumentError(
            "NetCDF file $filename missing 'Cbc_Re' variable for spectral format"))
        haskey(ds, "Cbc_Im") || throw(ArgumentError(
            "NetCDF file $filename missing 'Cbc_Im' variable for spectral format"))

        raw_re = T.(Array(ds["Cbc_Re"]))  # Expected shape [2, nlm]
        raw_im = T.(Array(ds["Cbc_Im"]))  # Expected shape [2, nlm]

        # Handle both [2, nlm] and [nlm, 2] orientations
        if size(raw_re) == (2, nlm) || (size(raw_re, 1) == 2 && size(raw_re, 2) >= nlm)
            n = min(size(raw_re, 2), nlm)
            bc_real[:, 1:n] .= raw_re[:, 1:n]
            bc_imag[:, 1:n] .= raw_im[:, 1:n]
        elseif size(raw_re, 2) == 2 && size(raw_re, 1) >= nlm
            # Transposed: [nlm, 2] -> need to transpose
            n = min(size(raw_re, 1), nlm)
            bc_real[:, 1:n] .= permutedims(raw_re[1:n, :])
            bc_imag[:, 1:n] .= permutedims(raw_im[1:n, :])
        else
            throw(ArgumentError(
                "Cbc_Re has unexpected shape $(size(raw_re)); expected (2, $nlm) or ($nlm, 2)"))
        end
    end

    return SpectralBoundaryCoefficients{T}(bc_real, bc_imag, nlm, filename, :spectral)
end

"""
    store_bc_in_field!(field, bc_coeffs::SpectralBoundaryCoefficients)

Store spectral boundary coefficients into a field's `boundary_interpolation_cache`.
The field must have a `boundary_interpolation_cache` attribute.

After calling this, `get_bc_vectors_from_field(field)` will return the stored vectors.
"""
function store_bc_in_field!(field, bc_coeffs::SpectralBoundaryCoefficients{T}) where {T}
    cache = field.boundary_interpolation_cache
    _store_bc_in_cache!(cache, bc_coeffs)
    return field
end

function _store_bc_in_cache!(
        cache::BoundaryInterpolationCache{T},
        bc_coeffs::SpectralBoundaryCoefficients{S}
) where {T, S}
    cache.bc_real = T === S ? bc_coeffs.bc_real : T.(bc_coeffs.bc_real)
    cache.bc_imag = T === S ? bc_coeffs.bc_imag : T.(bc_coeffs.bc_imag)
    cache.bc_loaded = true
    cache.bc_source_file = bc_coeffs.source_file
    cache.bc_source_format = bc_coeffs.source_format
    cache.bc_nlm = bc_coeffs.nlm
    return cache
end

function _store_bc_in_cache!(cache::AbstractDict, bc_coeffs::SpectralBoundaryCoefficients)
    cache["bc_real"] = bc_coeffs.bc_real
    cache["bc_imag"] = bc_coeffs.bc_imag
    cache["bc_loaded"] = true
    cache["bc_source_file"] = bc_coeffs.source_file
    cache["bc_source_format"] = bc_coeffs.source_format
    cache["bc_nlm"] = bc_coeffs.nlm
    return cache
end

"""
    get_bc_vectors_from_field(field)

Extract boundary condition vectors from a field's cache for use in implicit solves.

# Returns
A named tuple `(inner_real, outer_real, inner_imag, outer_imag)` where each element
is either a `Vector{T}` of length nlm or `nothing` if no file BCs are loaded.
"""
function get_bc_vectors_from_field(field)
    cache = field.boundary_interpolation_cache
    return _get_bc_vectors_from_cache(cache)
end

function _get_bc_vectors_from_cache(cache::BoundaryInterpolationCache)
    bc_real = cache.bc_real
    bc_imag = cache.bc_imag
    if !cache.bc_loaded || bc_real === nothing || bc_imag === nothing
        return (inner_real = nothing, outer_real = nothing,
            inner_imag = nothing, outer_imag = nothing)
    end

    return (inner_real = view(bc_real, 1, :), outer_real = view(bc_real, 2, :),
        inner_imag = view(bc_imag, 1, :), outer_imag = view(bc_imag, 2, :))
end

function _get_bc_vectors_from_cache(cache)
    if !get(cache, "bc_loaded", false)
        return (inner_real = nothing, outer_real = nothing,
            inner_imag = nothing, outer_imag = nothing)
    end

    bc_real = cache["bc_real"]::Matrix
    bc_imag = cache["bc_imag"]::Matrix

    # Return views into the stored matrices to avoid per-call allocations
    return (inner_real = view(bc_real, 1, :), outer_real = view(bc_real, 2, :),
        inner_imag = view(bc_imag, 1, :), outer_imag = view(bc_imag, 2, :))
end

# Export from bcs module
export SpectralBoundaryCoefficients, BoundaryInterpolationCache
export load_spectral_bc_from_file, store_bc_in_field!, get_bc_vectors_from_field
