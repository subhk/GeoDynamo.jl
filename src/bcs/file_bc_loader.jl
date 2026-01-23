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
struct SpectralBoundaryCoefficients{T<:AbstractFloat}
    bc_real::Matrix{T}   # [2, nlm] - row 1=inner, row 2=outer
    bc_imag::Matrix{T}   # [2, nlm] - row 1=inner, row 2=outer
    nlm::Int
    source_file::String
    source_format::Symbol
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
- Variables: `"inner"` [nlat, nlon] and/or `"outer"` [nlat, nlon]
- Or a single variable with 3rd dimension of size 2: `"temperature"` [nlat, nlon, 2]
  (slice 1 = inner, slice 2 = outer)

**Spectral format** (`:spectral`, Fortran-compatible):
- Variables: `"Cbc_Re"` [2, nlm] and `"Cbc_Im"` [2, nlm]
- Row 1 = inner boundary, Row 2 = outer boundary

# Returns
- `SpectralBoundaryCoefficients{T}` with spectral BC data
"""
function load_spectral_bc_from_file(filename::String, config;
                                     format::Symbol=:physical,
                                     T::Type{<:AbstractFloat}=Float64)
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
function _load_physical_format(filename::String, config, ::Type{T}) where T<:AbstractFloat
    nlm = config.nlm
    bc_real = zeros(T, 2, nlm)
    bc_imag = zeros(T, 2, nlm)

    NCDatasets.Dataset(filename, "r") do ds
        # Try separate inner/outer variables first
        if haskey(ds, "inner") || haskey(ds, "outer")
            if haskey(ds, "inner")
                inner_phys = T.(Array(ds["inner"]))
                coeffs = shtns_physical_to_spectral(inner_phys, config; return_complex=true)
                for i in 1:min(length(coeffs), nlm)
                    bc_real[1, i] = T(real(coeffs[i]))
                    bc_imag[1, i] = T(imag(coeffs[i]))
                end
            end
            if haskey(ds, "outer")
                outer_phys = T.(Array(ds["outer"]))
                coeffs = shtns_physical_to_spectral(outer_phys, config; return_complex=true)
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
                coeffs = shtns_physical_to_spectral(inner_phys, config; return_complex=true)
                for i in 1:min(length(coeffs), nlm)
                    bc_real[1, i] = T(real(coeffs[i]))
                    bc_imag[1, i] = T(imag(coeffs[i]))
                end
                outer_phys = data[:, :, 2]
                coeffs = shtns_physical_to_spectral(outer_phys, config; return_complex=true)
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
                coeffs = shtns_physical_to_spectral(inner_phys, config; return_complex=true)
                for i in 1:min(length(coeffs), nlm)
                    bc_real[1, i] = T(real(coeffs[i]))
                    bc_imag[1, i] = T(imag(coeffs[i]))
                end
                outer_phys = data[:, :, 2]
                coeffs = shtns_physical_to_spectral(outer_phys, config; return_complex=true)
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
function _load_spectral_format(filename::String, nlm::Int, ::Type{T}) where T<:AbstractFloat
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
The field must have a `boundary_interpolation_cache::Dict{String, Any}` attribute.

After calling this, `get_bc_vectors_from_field(field)` will return the stored vectors.
"""
function store_bc_in_field!(field, bc_coeffs::SpectralBoundaryCoefficients{T}) where T
    cache = field.boundary_interpolation_cache
    cache["bc_real"] = bc_coeffs.bc_real
    cache["bc_imag"] = bc_coeffs.bc_imag
    cache["bc_loaded"] = true
    cache["bc_source_file"] = bc_coeffs.source_file
    cache["bc_source_format"] = bc_coeffs.source_format
    cache["bc_nlm"] = bc_coeffs.nlm
    return field
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
    if !get(cache, "bc_loaded", false)
        return (inner_real=nothing, outer_real=nothing,
                inner_imag=nothing, outer_imag=nothing)
    end

    bc_real = cache["bc_real"]::Matrix
    bc_imag = cache["bc_imag"]::Matrix
    T = eltype(bc_real)
    nlm = size(bc_real, 2)

    inner_real = Vector{T}(undef, nlm)
    outer_real = Vector{T}(undef, nlm)
    inner_imag = Vector{T}(undef, nlm)
    outer_imag = Vector{T}(undef, nlm)

    @inbounds for i in 1:nlm
        inner_real[i] = bc_real[1, i]
        outer_real[i] = bc_real[2, i]
        inner_imag[i] = bc_imag[1, i]
        outer_imag[i] = bc_imag[2, i]
    end

    return (inner_real=inner_real, outer_real=outer_real,
            inner_imag=inner_imag, outer_imag=outer_imag)
end

# Export from bcs module
export SpectralBoundaryCoefficients
export load_spectral_bc_from_file, store_bc_in_field!, get_bc_vectors_from_field
