# ================================================================================
# Topography Data Structures
# ================================================================================
#
# This file defines data structures for storing and manipulating boundary
# topography in spectral (spherical harmonic) form:
#
#   h(θ, φ) = Σ_{L,M} h_{LM} Y_L^M(θ, φ)
#
# The topography is defined at both the inner-core boundary (ICB) and
# the core-mantle boundary (CMB).
#
# ================================================================================

using NCDatasets

# ================================================================================
# Topography Field Structure
# ================================================================================

"""
    TopographyField{T}

Spectral representation of topography at a single boundary.

# Fields
- `radius::T`: Reference radius of the boundary (r_i or r_o)
- `lmax::Int`: Maximum spherical harmonic degree
- `mmax::Int`: Maximum azimuthal order
- `nlm::Int`: Total number of spectral coefficients
- `coeffs_real::Vector{T}`: Real parts of h_{LM} coefficients
- `coeffs_imag::Vector{T}`: Imaginary parts of h_{LM} coefficients
- `location::BoundaryLocation`: INNER_BOUNDARY (ICB) or OUTER_BOUNDARY (CMB)
- `rms_amplitude::T`: RMS amplitude of topography
- `max_amplitude::T`: Maximum |h| value
- `description::String`: Description of topography source
"""
mutable struct TopographyField{T<:AbstractFloat}
    radius::T                          # Reference radius r_b
    lmax::Int                          # Maximum degree L
    mmax::Int                          # Maximum order M
    nlm::Int                           # Number of spectral modes
    coeffs_real::Vector{T}             # Real parts of h_{LM}
    coeffs_imag::Vector{T}             # Imaginary parts of h_{LM}
    location::BoundaryLocation         # ICB or CMB
    rms_amplitude::T                   # RMS topography amplitude
    max_amplitude::T                   # Maximum amplitude
    description::String                # Source description
end

"""
    TopographyField{T}(lmax::Int, mmax::Int, radius::T, location::BoundaryLocation) where T

Create an empty TopographyField with zero coefficients.
"""
function TopographyField{T}(lmax::Int, mmax::Int, radius::T, location::BoundaryLocation) where T
    mmax = min(mmax, lmax)  # mmax cannot exceed lmax
    nlm = (lmax + 1) * (mmax + 1) - mmax * (mmax + 1) ÷ 2  # Approximate

    # More accurate nlm calculation for m <= l
    nlm = 0
    for l in 0:lmax
        for m in 0:min(l, mmax)
            nlm += 1
        end
    end

    return TopographyField{T}(
        radius, lmax, mmax, nlm,
        zeros(T, nlm),
        zeros(T, nlm),
        location,
        zero(T), zero(T),
        "zero topography"
    )
end

@doc """
    TopographyField(lmax, mmax, radius, location)
    TopographyField{T}

Spectral representation of boundary topography h(θ, φ) expanded in spherical harmonics.

The topography is represented as:
```math
h(\\theta, \\phi) = \\sum_{L=0}^{L_{max}} \\sum_{M=-L}^{L} h_{LM} Y_L^M(\\theta, \\phi)
```

# Arguments
- `lmax::Int`: Maximum spherical harmonic degree L
- `mmax::Int`: Maximum azimuthal order M (capped at lmax)
- `radius::Float64`: Reference boundary radius
- `location::BoundaryLocation`: `INNER_BOUNDARY` (ICB) or `OUTER_BOUNDARY` (CMB)

# Fields
- `radius`: Reference radius of the boundary
- `lmax`, `mmax`: Maximum degree and order
- `nlm`: Total number of spectral coefficients
- `coeffs_real`, `coeffs_imag`: Real and imaginary parts of h_{LM}
- `location`: Boundary location enum
- `rms_amplitude`, `max_amplitude`: Topography statistics

# Example
```julia
topo = TopographyField(32, 32, 0.35, INNER_BOUNDARY)
```

See also: [`TopographyData`](@ref), [`set_topography_coefficients!`](@ref)
""" TopographyField

TopographyField(lmax::Int, mmax::Int, radius::Float64, location::BoundaryLocation) =
    TopographyField{Float64}(lmax, mmax, radius, location)

# ================================================================================
# Main Topography Data Structure
# ================================================================================

"""
    TopographyData{T}

Complete topography data for both boundaries.

# Fields
- `icb::Union{TopographyField{T}, Nothing}`: Inner core boundary topography
- `cmb::Union{TopographyField{T}, Nothing}`: Core-mantle boundary topography
- `gaunt_cache::Union{GauntTensorCache{T}, Nothing}`: Cached Gaunt tensors
- `epsilon::T`: Topography amplitude parameter ε
- `is_time_dependent::Bool`: Whether topography evolves in time
"""
mutable struct TopographyData{T<:AbstractFloat}
    icb::Union{TopographyField{T}, Nothing}      # Inner boundary
    cmb::Union{TopographyField{T}, Nothing}      # Outer boundary
    gaunt_cache::Union{GauntTensorCache{T}, Nothing}  # Gaunt tensors
    epsilon::T                                   # Amplitude parameter
    is_time_dependent::Bool                      # Time evolution flag
end

"""
    TopographyData{T}() where T

Create empty TopographyData with no topography.
"""
function TopographyData{T}() where T
    return TopographyData{T}(nothing, nothing, nothing, T(0.01), false)
end

@doc """
    TopographyData
    TopographyData{T}

Complete topography data container for both ICB and CMB boundaries.

This structure holds the spectral topography fields for inner and outer boundaries,
along with pre-computed Gaunt tensor caches for efficient mode coupling calculations.

# Fields
- `icb::Union{TopographyField, Nothing}`: Inner core boundary (ICB) topography
- `cmb::Union{TopographyField, Nothing}`: Core-mantle boundary (CMB) topography
- `gaunt_cache::Union{GauntTensorCache, Nothing}`: Pre-computed Gaunt tensors
- `epsilon::Float64`: Topography amplitude parameter ε
- `is_time_dependent::Bool`: Whether topography evolves (e.g., Stefan condition)

# Example
```julia
topo_data = TopographyData{Float64}(icb_field, cmb_field, gaunt, 0.02, false)
```

See also: [`TopographyField`](@ref), [`create_topography_data`](@ref), [`GauntTensorCache`](@ref)
""" TopographyData

TopographyData() = TopographyData{Float64}()

# ================================================================================
# Spectral Index Utilities
# ================================================================================

"""
    lm_to_index(l::Int, m::Int, lmax::Int) -> Int

Convert (l, m) to linear index for spectral coefficient storage.
Uses the convention: index = l(l+1)/2 + m + 1 for m >= 0
"""
function lm_to_index(l::Int, m::Int, lmax::Int)
    @assert l >= 0 && l <= lmax "l must be in [0, lmax]"
    @assert abs(m) <= l "m must satisfy |m| <= l"

    if m >= 0
        # Standard indexing for m >= 0
        idx = l * (l + 1) ÷ 2 + m + 1
    else
        # Negative m uses conjugate relation
        # Map to positive m index and flag for conjugate
        idx = l * (l + 1) ÷ 2 + (-m) + 1
    end
    return idx
end

"""
    index_to_lm(idx::Int, lmax::Int) -> (l, m)

Convert linear index back to (l, m).
"""
function index_to_lm(idx::Int, lmax::Int)
    # Inverse of lm_to_index
    # Find l such that l(l+1)/2 < idx <= (l+1)(l+2)/2
    l = 0
    while (l + 1) * (l + 2) ÷ 2 < idx
        l += 1
    end
    m = idx - l * (l + 1) ÷ 2 - 1
    return (l, m)
end

# ================================================================================
# Topography Creation Functions
# ================================================================================

"""
    create_topography_data(; icb_coeffs=nothing, cmb_coeffs=nothing,
                            icb_radius=nothing, cmb_radius=nothing,
                            lmax=32, mmax=32, epsilon=0.01) -> TopographyData

Create TopographyData from spectral coefficients.

# Arguments
- `icb_coeffs`: Complex vector of ICB topography coefficients h_{LM}
- `cmb_coeffs`: Complex vector of CMB topography coefficients h_{LM}
- `icb_radius`: Inner boundary radius (default: from parameters)
- `cmb_radius`: Outer boundary radius (default: from parameters)
- `lmax, mmax`: Maximum spherical harmonic degree/order
- `epsilon`: Topography amplitude parameter

# Example
```julia
# Create CMB topography with degree-2 pattern
cmb_h = zeros(ComplexF64, 6)
cmb_h[lm_to_index(2, 0, 32)] = 0.1  # h_{2,0}
topo = create_topography_data(cmb_coeffs=cmb_h, cmb_radius=1.0, lmax=32)
```
"""
function create_topography_data(;
    icb_coeffs::Union{Vector{<:Number}, Nothing}=nothing,
    cmb_coeffs::Union{Vector{<:Number}, Nothing}=nothing,
    icb_radius::Union{Float64, Nothing}=nothing,
    cmb_radius::Union{Float64, Nothing}=nothing,
    lmax::Int=32,
    mmax::Int=-1,
    epsilon::Float64=0.01
)
    mmax = mmax < 0 ? lmax : mmax
    T = Float64

    topo = TopographyData{T}()
    topo.epsilon = epsilon

    # Create ICB topography field
    if icb_coeffs !== nothing
        r_i = icb_radius !== nothing ? icb_radius : 0.35  # Default from typical rratio
        icb = TopographyField{T}(lmax, mmax, r_i, INNER_BOUNDARY)
        set_topography_coefficients!(icb, icb_coeffs)
        topo.icb = icb
    end

    # Create CMB topography field
    if cmb_coeffs !== nothing
        r_o = cmb_radius !== nothing ? cmb_radius : 1.0
        cmb = TopographyField{T}(lmax, mmax, r_o, OUTER_BOUNDARY)
        set_topography_coefficients!(cmb, cmb_coeffs)
        topo.cmb = cmb
    end

    return topo
end

"""
    set_topography_coefficients!(field::TopographyField{T}, coeffs::Vector) where T

Set spectral coefficients for a topography field.

Accepts either:
- Complex vector: splits into real and imaginary parts
- Real vector: sets real parts only (imaginary = 0)
"""
function set_topography_coefficients!(field::TopographyField{T}, coeffs::Vector) where T
    n = min(length(coeffs), field.nlm)

    if eltype(coeffs) <: Complex
        for i in 1:n
            field.coeffs_real[i] = real(coeffs[i])
            field.coeffs_imag[i] = imag(coeffs[i])
        end
    else
        field.coeffs_real[1:n] .= T.(coeffs[1:n])
        fill!(field.coeffs_imag, zero(T))
    end

    # Update amplitude statistics
    update_topography_statistics!(field)

    return field
end

"""
    get_topography_coefficients(field::TopographyField{T}) -> Vector{Complex{T}}

Get complex spectral coefficients from topography field.
"""
function get_topography_coefficients(field::TopographyField{T}) where T
    return complex.(field.coeffs_real, field.coeffs_imag)
end

"""
    get_coefficient(field::TopographyField{T}, l::Int, m::Int) where T

Get the h_{l,m} coefficient from the topography field.
"""
function get_coefficient(field::TopographyField{T}, l::Int, m::Int) where T
    if l > field.lmax || abs(m) > min(l, field.mmax)
        return zero(Complex{T})
    end

    idx = lm_to_index(l, abs(m), field.lmax)

    if m >= 0
        return complex(field.coeffs_real[idx], field.coeffs_imag[idx])
    else
        # Negative m: use conjugate relation h_{l,-m} = (-1)^m conj(h_{l,m})
        coeff = complex(field.coeffs_real[idx], field.coeffs_imag[idx])
        return iseven(m) ? conj(coeff) : -conj(coeff)
    end
end

"""
    update_topography_statistics!(field::TopographyField{T}) where T

Update RMS and max amplitude statistics for the topography field.
"""
function update_topography_statistics!(field::TopographyField{T}) where T
    # RMS amplitude: sqrt(Σ |h_{LM}|²)
    sum_sq = zero(T)
    max_amp = zero(T)

    for i in 1:field.nlm
        amp_sq = field.coeffs_real[i]^2 + field.coeffs_imag[i]^2
        sum_sq += amp_sq
        max_amp = max(max_amp, sqrt(amp_sq))
    end

    field.rms_amplitude = sqrt(sum_sq)
    field.max_amplitude = max_amp

    return field
end

# ================================================================================
# Programmatic Topography Generation
# ================================================================================

"""
    create_uniform_topography(amplitude::Float64, radius::Float64,
                              location::BoundaryLocation; lmax::Int=32) -> TopographyField

Create uniform (degree-0) topography h = amplitude everywhere.
"""
function create_uniform_topography(amplitude::Float64, radius::Float64,
                                   location::BoundaryLocation; lmax::Int=32)
    field = TopographyField(lmax, lmax, radius, location)

    # Y_0^0 = 1/√(4π), so h_{0,0} = amplitude * √(4π)
    field.coeffs_real[1] = amplitude * sqrt(4π)

    update_topography_statistics!(field)
    field.description = "uniform topography, amplitude=$amplitude"

    return field
end

"""
    create_spherical_harmonic_topography(l::Int, m::Int, amplitude::Float64,
                                         radius::Float64, location::BoundaryLocation;
                                         lmax::Int=32) -> TopographyField

Create single spherical harmonic topography h = amplitude * Y_l^m(θ, φ).
"""
function create_spherical_harmonic_topography(l::Int, m::Int, amplitude::Float64,
                                              radius::Float64, location::BoundaryLocation;
                                              lmax::Int=32)
    @assert l <= lmax "l must be <= lmax"
    @assert abs(m) <= l "m must satisfy |m| <= l"

    field = TopographyField(lmax, lmax, radius, location)

    idx = lm_to_index(l, abs(m), lmax)
    if m >= 0
        field.coeffs_real[idx] = amplitude
    else
        # For m < 0, handle conjugate relation
        phase = iseven(-m) ? 1.0 : -1.0
        field.coeffs_real[idx] = phase * amplitude
    end

    update_topography_statistics!(field)
    field.description = "Y_$l^$m topography, amplitude=$amplitude"

    return field
end

"""
    create_random_topography(spectrum::Function, radius::Float64,
                             location::BoundaryLocation; lmax::Int=32,
                             seed::Int=0) -> TopographyField

Create random topography with prescribed power spectrum.

# Arguments
- `spectrum`: Function l -> power at degree l (e.g., l -> l^(-2) for red spectrum)
- `radius`: Boundary radius
- `location`: INNER_BOUNDARY or OUTER_BOUNDARY
- `lmax`: Maximum degree
- `seed`: Random seed (0 for no seed)

# Example
```julia
# Create topography with l^(-2) spectrum
topo = create_random_topography(l -> 1.0/max(l,1)^2, 1.0, OUTER_BOUNDARY)
```
"""
function create_random_topography(spectrum::Function, radius::Float64,
                                  location::BoundaryLocation;
                                  lmax::Int=32, seed::Int=0)
    if seed > 0
        Random.seed!(seed)
    end

    field = TopographyField(lmax, lmax, radius, location)

    for l in 0:lmax
        power = spectrum(l)
        for m in 0:l
            idx = lm_to_index(l, m, lmax)
            # Random amplitude with prescribed power
            amp = sqrt(power / (2l + 1)) * randn()
            phase = 2π * rand()
            field.coeffs_real[idx] = amp * cos(phase)
            field.coeffs_imag[idx] = amp * sin(phase)
        end
    end

    update_topography_statistics!(field)
    field.description = "random topography, lmax=$lmax"

    return field
end

# ================================================================================
# File I/O
# ================================================================================

"""
    load_topography_from_file(filename::String, location::BoundaryLocation;
                              radius::Union{Float64, Nothing}=nothing) -> TopographyField

Load topography from a NetCDF file.

Expected NetCDF structure:
- Dimensions: l, m (or lm for 1D storage)
- Variables: h_real, h_imag (or h for complex)
- Attributes: lmax, radius (optional)
"""
function load_topography_from_file(filename::String, location::BoundaryLocation;
                                   radius::Union{Float64, Nothing}=nothing)
    if !isfile(filename)
        throw(ArgumentError("Topography file not found: $filename"))
    end

    NCDataset(filename, "r") do ds
        # Read lmax
        lmax = haskey(ds.attrib, "lmax") ? ds.attrib["lmax"] : 32

        # Read radius
        r = radius !== nothing ? radius :
            (haskey(ds.attrib, "radius") ? ds.attrib["radius"] : 1.0)

        field = TopographyField(lmax, lmax, r, location)

        # Read coefficients
        if haskey(ds, "h_real") && haskey(ds, "h_imag")
            h_real = Array(ds["h_real"])
            h_imag = Array(ds["h_imag"])
            n = min(length(h_real), field.nlm)
            field.coeffs_real[1:n] .= h_real[1:n]
            field.coeffs_imag[1:n] .= h_imag[1:n]
        elseif haskey(ds, "h")
            h = Array(ds["h"])
            if eltype(h) <: Complex
                n = min(length(h), field.nlm)
                field.coeffs_real[1:n] .= real.(h[1:n])
                field.coeffs_imag[1:n] .= imag.(h[1:n])
            else
                n = min(length(h), field.nlm)
                field.coeffs_real[1:n] .= h[1:n]
            end
        else
            throw(ArgumentError("NetCDF file must contain 'h' or 'h_real'/'h_imag' variables"))
        end

        update_topography_statistics!(field)
        field.description = "loaded from $filename"

        return field
    end
end

"""
    load_topography_from_array(h_physical::Matrix{T}, radius::Float64,
                               location::BoundaryLocation, config;
                               lmax::Int=-1) where T

Load topography from a physical space array h(θ, φ) by transforming to spectral space.

# Arguments
- `h_physical`: 2D array of topography values on (θ, φ) grid
- `radius`: Boundary radius
- `location`: INNER_BOUNDARY or OUTER_BOUNDARY
- `config`: SHTnsKit configuration for transform
- `lmax`: Maximum degree to retain (-1 for config.lmax)
"""
function load_topography_from_array(h_physical::Matrix{T}, radius::Float64,
                                    location::BoundaryLocation, config;
                                    lmax::Int=-1) where T
    lmax_use = lmax > 0 ? lmax : config.lmax

    field = TopographyField{Float64}(lmax_use, lmax_use, radius, location)

    # Transform to spectral space using SHTnsKit
    try
        # Perform forward transform
        coeffs = SHTnsKit.analysis(config, h_physical)

        # Store coefficients
        idx = 0
        for l in 0:min(lmax_use, config.lmax)
            for m in 0:min(l, config.mmax)
                idx += 1
                if idx <= field.nlm && l < size(coeffs, 1) && m < size(coeffs, 2)
                    field.coeffs_real[idx] = real(coeffs[l+1, m+1])
                    field.coeffs_imag[idx] = imag(coeffs[l+1, m+1])
                end
            end
        end
    catch e
        @warn "SHTnsKit transform failed, using mean value only: $e"
        # Fallback: just use mean (l=0 mode)
        mean_val = sum(h_physical) / length(h_physical)
        field.coeffs_real[1] = mean_val * sqrt(4π)
    end

    update_topography_statistics!(field)
    field.description = "loaded from physical array"

    return field
end

"""
    save_topography_to_file(field::TopographyField, filename::String)

Save topography field to NetCDF file.
"""
function save_topography_to_file(field::TopographyField, filename::String)
    NCDataset(filename, "c") do ds
        # Define dimensions
        defDim(ds, "nlm", field.nlm)

        # Define variables
        h_real_var = defVar(ds, "h_real", Float64, ("nlm",))
        h_imag_var = defVar(ds, "h_imag", Float64, ("nlm",))

        # Write data
        h_real_var[:] = field.coeffs_real
        h_imag_var[:] = field.coeffs_imag

        # Write attributes
        ds.attrib["lmax"] = field.lmax
        ds.attrib["mmax"] = field.mmax
        ds.attrib["radius"] = field.radius
        ds.attrib["location"] = string(field.location)
        ds.attrib["rms_amplitude"] = field.rms_amplitude
        ds.attrib["description"] = field.description
    end

    if get_rank() == 0
        @info "Topography saved to $filename"
    end

    return nothing
end

# ================================================================================
# Topography Evaluation in Physical Space
# ================================================================================

"""
    evaluate_topography(field::TopographyField{T}, config) where T -> Matrix{T}

Evaluate topography in physical space using SHTnsKit synthesis.

# Arguments
- `field`: TopographyField containing spectral coefficients
- `config`: SHTnsKit configuration for the transform

Returns h(θ, φ) as a nlat × nlon matrix on the Gauss-Legendre grid.
"""
function evaluate_topography(field::TopographyField{T}, config) where T
    # Build spectral coefficient array in SHTnsKit format
    coeffs_sht = zeros(Complex{T}, field.lmax + 1, field.mmax + 1)

    idx = 0
    for l in 0:field.lmax
        for m in 0:min(l, field.mmax)
            idx += 1
            coeffs_sht[l+1, m+1] = complex(field.coeffs_real[idx], field.coeffs_imag[idx])
        end
    end

    # Use SHTnsKit synthesis for spectral→physical transform
    try
        h_physical = SHTnsKit.synthesis(config, coeffs_sht)
        return real.(h_physical)
    catch e
        @warn "SHTnsKit synthesis failed, using fallback method: $e"
        return evaluate_topography_fallback(field, config)
    end
end

"""
    evaluate_topography(field::TopographyField{T}, theta::Vector{T},
                        phi::Vector{T}) where T -> Matrix{T}

Evaluate topography in physical space on an arbitrary (θ, φ) grid.
This uses direct summation and is slower than the SHTnsKit version.

Returns h(θ_i, φ_j) as a nlat × nlon matrix.
"""
function evaluate_topography(field::TopographyField{T}, theta::Vector{T},
                             phi::Vector{T}) where T
    nlat = length(theta)
    nlon = length(phi)
    h = zeros(T, nlat, nlon)

    # For arbitrary grids, use direct summation with Legendre functions
    for i in 1:nlat
        θ = theta[i]
        cosθ = cos(θ)

        for j in 1:nlon
            φ = phi[j]

            # Sum over spectral coefficients
            val = zero(T)
            idx = 0
            for l in 0:field.lmax
                # Compute P_l^0(cos θ)
                plm_0 = compute_associated_legendre(l, 0, cosθ)

                for m in 0:min(l, field.mmax)
                    idx += 1

                    # Compute P_l^m(cos θ)
                    plm = m == 0 ? plm_0 : compute_associated_legendre(l, m, cosθ)

                    # Y_l^m contribution (real part)
                    # Y_l^m = N_l^m * P_l^m(cos θ) * exp(im*φ)
                    nlm = sqrt((2l + 1) / (4π) * factorial(big(l - m)) / factorial(big(l + m)))

                    coeff_re = field.coeffs_real[idx]
                    coeff_im = field.coeffs_imag[idx]

                    # m = 0 term
                    if m == 0
                        val += Float64(nlm) * plm * coeff_re
                    else
                        # m > 0: include both +m and -m contributions
                        # h = Σ h_lm Y_lm = Σ [h_lm Y_lm + h_l,-m Y_l,-m]
                        # For real h: h_l,-m = (-1)^m conj(h_lm)
                        cos_mφ = cos(m * φ)
                        sin_mφ = sin(m * φ)

                        val += 2 * Float64(nlm) * plm * (coeff_re * cos_mφ - coeff_im * sin_mφ)
                    end
                end
            end

            h[i, j] = val
        end
    end

    return h
end

"""
    evaluate_topography_fallback(field::TopographyField{T}, config) where T

Fallback evaluation when SHTnsKit synthesis fails.
"""
function evaluate_topography_fallback(field::TopographyField{T}, config) where T
    # Get grid information from config
    nlat = config.nlat
    nlon = config.nlon

    # Build theta/phi grids (Gauss-Legendre)
    theta = zeros(T, nlat)
    phi = zeros(T, nlon)

    # Gauss-Legendre points for theta
    nodes, _ = gauss_legendre_nodes(nlat)
    theta .= acos.(nodes)

    # Uniform grid for phi
    for j in 1:nlon
        phi[j] = 2π * (j - 1) / nlon
    end

    return evaluate_topography(field, theta, phi)
end

"""
    compute_associated_legendre(l::Int, m::Int, x::T) where T

Compute the associated Legendre polynomial P_l^m(x).
Uses the standard normalization (not Schmidt or fully normalized).
"""
function compute_associated_legendre(l::Int, m::Int, x::T) where T
    if m < 0 || m > l
        return zero(T)
    end

    # Start with P_m^m using the formula:
    # P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^(m/2)
    pmm = one(T)
    if m > 0
        somx2 = sqrt((1 - x) * (1 + x))
        fact = one(T)
        for i in 1:m
            pmm *= -fact * somx2
            fact += 2
        end
    end

    if l == m
        return pmm
    end

    # Compute P_{m+1}^m using:
    # P_{m+1}^m(x) = x(2m+1) P_m^m(x)
    pmmp1 = x * (2m + 1) * pmm

    if l == m + 1
        return pmmp1
    end

    # Use recurrence relation for higher l:
    # (l-m) P_l^m = x(2l-1) P_{l-1}^m - (l+m-1) P_{l-2}^m
    pll = zero(T)
    for ll in (m+2):l
        pll = (x * (2ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    end

    return pll
end

"""
    gauss_legendre_nodes(n::Int) -> (nodes, weights)

Compute Gauss-Legendre quadrature nodes and weights.
Returns nodes in [-1, 1] and corresponding weights.
"""
function gauss_legendre_nodes(n::Int)
    nodes = zeros(Float64, n)
    weights = zeros(Float64, n)

    m = div(n + 1, 2)

    for i in 1:m
        # Initial guess for root
        z = cos(π * (i - 0.25) / (n + 0.5))

        # Newton's method to find root
        for _ in 1:100
            p1 = 1.0
            p2 = 0.0

            for j in 1:n
                p3 = p2
                p2 = p1
                p1 = ((2j - 1) * z * p2 - (j - 1) * p3) / j
            end

            # Derivative
            pp = n * (z * p1 - p2) / (z^2 - 1)

            z1 = z
            z = z1 - p1 / pp

            if abs(z - z1) < 1e-15
                break
            end
        end

        nodes[i] = -z
        nodes[n+1-i] = z
        weights[i] = 2.0 / ((1 - z^2) * (n * (z * p1 - p2) / (z^2 - 1))^2)
        weights[n+1-i] = weights[i]
    end

    return nodes, weights
end

# ================================================================================
# Initialize Gaunt Cache for TopographyData
# ================================================================================

"""
    initialize_gaunt_cache!(topo::TopographyData{T}, lmax_field::Int;
                            precompute::Bool=true) where T

Initialize and optionally precompute Gaunt tensor cache for topography coupling.
"""
function initialize_gaunt_cache!(topo::TopographyData{T}, lmax_field::Int;
                                 precompute::Bool=true) where T
    # Determine topography lmax
    lmax_topo = 0
    if topo.icb !== nothing
        lmax_topo = max(lmax_topo, topo.icb.lmax)
    end
    if topo.cmb !== nothing
        lmax_topo = max(lmax_topo, topo.cmb.lmax)
    end

    if lmax_topo == 0
        @warn "No topography defined, Gaunt cache not needed"
        return nothing
    end

    # Create cache
    topo.gaunt_cache = GauntTensorCache{T}(lmax_field, lmax_topo)

    # Precompute if requested
    if precompute
        precompute_gaunt_tensors!(topo.gaunt_cache)
    end

    return topo.gaunt_cache
end
