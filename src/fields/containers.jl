# ================================================================================
# Variable Types with SHTnsKit Integration
# ================================================================================
#
# This file defines the core data structures for representing physical fields
# in the geodynamo simulation. Fields can be represented in either:
# - Spectral space: Spherical harmonic coefficients f_l^m(r)
# - Physical space: Grid point values f(θ, φ, r)
#
# All field types use PencilArrays for MPI-distributed storage, enabling
# efficient parallel computation on distributed memory systems.
#
# FIELD REPRESENTATION OVERVIEW:
# ------------------------------
# Scalar fields (temperature, composition):
#   - Spectral: SHTnsSpecField storing (l,m) coefficients
#   - Physical: SHTnsPhysField storing grid point values
#
# Vector fields (velocity, magnetic field):
#   - Physical: SHTnsVectorField with (r, θ, φ) components
#   - Spectral: SHTnsTorPolField with toroidal (T) and poloidal (P) scalars
#     The toroidal-poloidal decomposition: v = ∇×(T r̂) + ∇×∇×(P r̂)
#     ensures ∇·v = 0 by construction (useful for incompressible flow)
#
# ================================================================================

import .bcs: BoundaryType, DIRICHLET, NEUMANN

# ================================================================================
# Spectral Field Type
# ================================================================================

"""
    SHTnsSpecField{T}

A scalar field represented in spectral space as spherical harmonic coefficients.

# Storage Layout
Coefficients are stored as separate real and imaginary parts in 3D arrays:
- Dimension 1: Combined (l,m) spectral index (1 to nlm)
- Dimension 2: Dummy dimension (size 1) for PencilArrays compatibility
- Dimension 3: Radial level index

# Boundary Conditions
Each (l,m) mode can have independent boundary conditions at the inner
and outer radial boundaries. Common types:
- DIRICHLET: Specified value at boundary
- NEUMANN: Specified derivative at boundary

# Fields
- `config`: SHTnsKit configuration (grid and transform parameters)
- `nlm`: Total number of spectral modes
- `data_real`, `data_imag`: Real/imaginary parts of coefficients
- `pencil`: PencilArrays pencil defining data distribution
- `bc_type_inner/outer`: Boundary condition type for each mode
- `boundary_values`: Boundary values [2, nlm] for inner/outer
"""
mutable struct SHTnsSpecField{T<:Number}
    config::SHTnsKitConfig
    nlm::Int
    data_real::PencilArray{T,3}   # Real part of f_l^m(r)
    data_imag::PencilArray{T,3}   # Imaginary part of f_l^m(r)
    pencil::Pencil{3}             # Data distribution specification
    bc_type_inner::Vector{Int}    # BC type at inner boundary for each mode
    bc_type_outer::Vector{Int}    # BC type at outer boundary for each mode
    boundary_values::Matrix{T}    # [2, nlm] boundary values (row 1=inner, row 2=outer)
end

# ================================================================================
# Physical Field Type
# ================================================================================

"""
    SHTnsPhysField{T}

A scalar field represented on the physical (θ, φ, r) grid.

# Storage Layout
Field values stored in a 3D array:
- Dimension 1: Latitude index (1 to nlat), Gauss-Legendre points
- Dimension 2: Longitude index (1 to nlon), uniform spacing
- Dimension 3: Radial level index

# Pencil Orientation
The `pencil` field specifies how data is distributed across MPI processes.
Different pencil orientations are optimal for different operations:
- phi pencil: Optimal for FFTs (all longitudes local)
- theta pencil: Optimal for Legendre transforms (all latitudes local)
- r pencil: Optimal for radial operations (all radii local)
"""
struct SHTnsPhysField{T<:Number}
    config::SHTnsKitConfig
    nlat::Int                     # Number of latitude points
    nlon::Int                     # Number of longitude points
    data::PencilArray{T,3}        # Field values f(θ, φ, r)
    pencil::Pencil{3}             # Current data distribution
end

# ================================================================================
# Vector Field Types
# ================================================================================

"""
    SHTnsVectorField{T}

A vector field represented by its three spherical components in physical space.

Components:
- `r_component`: Radial component v_r(θ, φ, r)
- `θ_component`: Latitudinal component v_θ(θ, φ, r)
- `φ_component`: Longitudinal component v_φ(θ, φ, r)

Each component is a SHTnsPhysField, potentially with different pencil
orientations for optimal computation of different operations.
"""
struct SHTnsVectorField{T<:Number}
    r_component::SHTnsPhysField{T}
    θ_component::SHTnsPhysField{T}
    φ_component::SHTnsPhysField{T}
end

"""
    SHTnsTorPolField{T}

Vector field in toroidal-poloidal spectral representation.

# The Toroidal-Poloidal Decomposition
Any solenoidal (divergence-free) vector field can be written as:
    v = ∇ × (T r̂) + ∇ × ∇ × (P r̂)

where T and P are scalar functions called the toroidal and poloidal potentials.

# Why This Representation?
1. Automatically satisfies ∇·v = 0 (mass conservation for incompressible flow)
2. Decouples certain physical processes in the equations
3. Reduces storage: 2 scalars instead of 3 vector components
4. Natural for spherical geometry and spectral methods

# Fields
- `toroidal`: Toroidal potential T(l,m,r) in spectral space
- `poloidal`: Poloidal potential P(l,m,r) in spectral space
"""
struct SHTnsTorPolField{T<:Number}
    toroidal::SHTnsSpecField{T}
    poloidal::SHTnsSpecField{T}
end

# ================================================================================
# Radial Domain
# ================================================================================

"""
    RadialDomain

Specification of the radial discretization for the spherical shell.

# Radial Grid
The simulation domain is a spherical shell between inner radius r_i (e.g., inner
core boundary) and outer radius r_o (e.g., core-mantle boundary). The radial
direction is discretized using Chebyshev collocation for spectral accuracy.

# Fields
- `N`: Number of radial points
- `local_range`: Indices of radial points local to this MPI process
- `r`: Radial coordinate values [1, N]
- `dr_matrices`: Radial derivative matrices (1st, 2nd order, etc.)
- `radial_laplacian`: Matrix for radial part of Laplacian
- `integration_weights`: Quadrature weights for radial integration
"""
struct RadialDomain
    N::Int                                  # Number of radial points
    local_range::UnitRange{Int}             # Local radial indices for this process
    r::Matrix{Float64}                      # Radial coordinates [1, N]
    dr_matrices::Vector{Matrix{Float64}}    # Derivative matrices
    radial_laplacian::Matrix{Float64}       # Radial Laplacian operator
    integration_weights::Vector{Float64}    # Integration quadrature weights
end

radial_bandwidth(domain::RadialDomain) = (size(domain.radial_laplacian, 1) - 1) ÷ 2

# ================================================================================
# Field Constructor Functions
# ================================================================================

"""
    create_shtns_spectral_field(T, config, 𝒟ᵒᶜ, pencil_spec) -> SHTnsSpecField{T}

Create a new spectral field initialized to zero with default Dirichlet BCs.

# Arguments
- `T`: Element type (typically Float64)
- `config`: SHTnsKit configuration providing nlm and other parameters
- `𝒟ᵒᶜ`: RadialDomain specifying the radial discretization
- `pencil_spec`: PencilArrays Pencil defining the data distribution

# Returns
A new SHTnsSpecField with:
- All spectral coefficients initialized to zero
- Dirichlet boundary conditions on all modes
- Zero boundary values
"""
function create_shtns_spectral_field(::Type{T}, config::AbstractSHTnsConfig,
                                    𝒟ᵒᶜ::RadialDomain,
                                    pencil_spec::Pencil{3}) where T
    nlm = config.nlm

    # Allocate distributed arrays for real and imaginary parts
    data_real = PencilArray{T}(undef, pencil_spec)
    data_imag = PencilArray{T}(undef, pencil_spec)

    # Initialize coefficients to zero
    fill!(parent(data_real), zero(T))
    fill!(parent(data_imag), zero(T))

    # Default boundary conditions: Dirichlet (specified value) on all modes
    bc_inner = fill(Int(DIRICHLET), nlm)
    bc_outer = fill(Int(DIRICHLET), nlm)
    boundary_vals = zeros(T, 2, nlm)  # Row 1: inner, Row 2: outer

    return SHTnsSpecField{T}(config, nlm,
                        data_real, data_imag, pencil_spec,
                        bc_inner, bc_outer, boundary_vals)
end

"""
    create_shtns_physical_field(T, config, 𝒟ᵒᶜ, pencil) -> SHTnsPhysField{T}

Create a new physical space field initialized to zero.

# Arguments
- `T`: Element type (typically Float64)
- `config`: SHTnsKit configuration providing grid dimensions
- `𝒟ᵒᶜ`: RadialDomain (for consistency with spectral field API)
- `pencil`: PencilArrays Pencil defining the data distribution

# Returns
A new SHTnsPhysField with all grid values initialized to zero.
"""
function create_shtns_physical_field(::Type{T}, config::AbstractSHTnsConfig,
                                    𝒟ᵒᶜ::RadialDomain,
                                    pencil::Pencil{3}) where T
    nlat = config.nlat
    nlon = config.nlon

    # Allocate distributed array for field values
    data = PencilArray{T}(undef, pencil)
    fill!(parent(data), zero(T))

    return SHTnsPhysField{T}(config, nlat, nlon, data, pencil)
end

"""
    create_shtns_vector_field(T, config, 𝒟ᵒᶜ, pencils) -> SHTnsVectorField{T}

Create a new vector field with three physical space components.

# Arguments
- `T`: Element type
- `config`: SHTnsKit configuration
- `𝒟ᵒᶜ`: RadialDomain specification
- `pencils`: Either a NamedTuple with :theta/:θ, :phi/:φ, :r keys,
             or a tuple (pencil_θ, pencil_φ, pencil_r)

# Returns
A new SHTnsVectorField with all components initialized to zero.
Each component uses a potentially different pencil orientation for
optimal computation of different operations.
"""
function create_shtns_vector_field(::Type{T}, config::AbstractSHTnsConfig,
                                    𝒟ᵒᶜ::RadialDomain,
                                    pencils) where T
    # Handle both NamedTuple and plain tuple input for pencils
    if pencils isa NamedTuple
        # Support both Unicode (θ, φ) and ASCII (theta, phi) names
        pencil_θ = hasproperty(pencils, Symbol("θ")) ? getproperty(pencils, Symbol("θ")) : getproperty(pencils, :theta)
        pencil_φ = hasproperty(pencils, Symbol("φ")) ? getproperty(pencils, Symbol("φ")) : getproperty(pencils, :phi)
        pencil_r = getproperty(pencils, :r)
    else
        pencil_θ, pencil_φ, pencil_r = pencils
    end
    
    # Create each component with the r-pencil (contiguous in r)
    r_comp = create_shtns_physical_field(T, config, 𝒟ᵒᶜ, pencil_r)
    θ_comp = create_shtns_physical_field(T, config, 𝒟ᵒᶜ, pencil_r)
    φ_comp = create_shtns_physical_field(T, config, 𝒟ᵒᶜ, pencil_r)
    
    return SHTnsVectorField{T}(r_comp, θ_comp, φ_comp)
end


"""
    create_radial_domain(nr=nothing; radius_ratio=nothing, radial_bandwidth=nothing) -> RadialDomain

Create a radial domain for a spherical shell geometry.

# Arguments
- `nr`: Number of radial points (required)
- `radius_ratio`: Inner/outer radius ratio (defaults to 0.35)
- `radial_bandwidth`: Bandwidth for finite differences (defaults to 4)

# Returns
A RadialDomain with Chebyshev-like radial spacing optimized for spectral accuracy.
"""
function create_radial_domain(nr::Union{Int,Nothing}=nothing;
                              radius_ratio::Union{Real,Nothing}=nothing,
                              radial_bandwidth::Union{Int,Nothing}=nothing)
    N = nr !== nothing ? nr : error("create_radial_domain: nr (number of radial points) must be provided")
    ratio = radius_ratio !== nothing ? Float64(radius_ratio) : 0.35
    bandwidth = radial_bandwidth !== nothing ? radial_bandwidth : 4

    # Validate inputs
    if N < 2
        throw(ArgumentError("Number of radial points must be at least 2, got N=$N"))
    end
    if ratio <= 0.0 || ratio >= 1.0
        throw(ArgumentError("radius_ratio must be in (0, 1), got radius_ratio=$ratio"))
    end

    # Create radial coordinate array with powers r^p for p = -3...+3
    # Column 4 stores r itself, other columns store r^(p-4)
    r = zeros(N, 7)
    for n in 1:N
        # Chebyshev-like spacing: clustered near boundaries
        r[n, 4] = 0.5 * (1.0 + cos(π * (N - n) / (N - 1)))
    end

    # Shift to shell geometry: r ∈ [ri, ro] where ri = radius_ratio/(1-radius_ratio)
    ri = ratio / (1.0 - ratio)
    r[:, 4] .+= ri

    # Compute powers of r for efficient evaluation of r-dependent terms
    for p in 1:7
        if p != 4
            power = p - 4
            r[:, p] = r[:, 4] .^ power
        end
    end

    # Allocate derivative matrices and integration weights
    dr_matrices         = [zeros(2*bandwidth+1, N) for _ in 1:3]
    radial_laplacian    = zeros(2*bandwidth+1, N)

    # Clenshaw-Curtis integration weights for Chebyshev-Lobatto points
    # Grid: x_n = cos(π(N-n)/(N-1)) in [-1,1], mapped to [ri, ri+1] with h = 0.5
    # Weights satisfy: ∫_{-1}^{1} f(x) dx ≈ Σ w_n f(x_n)
    integration_weights = zeros(N)
    M = N - 1  # number of intervals
    for n in 1:N
        theta_n = π * (N - n) / M
        w = 0.0
        for k in 0:div(M, 2)
            bk = (k == 0 || k == div(M, 2) && iseven(M)) ? 1.0 : 2.0
            w += bk * cos(2 * k * theta_n) / (1 - 4 * k^2)
        end
        w *= 2.0 / M
        if n == 1 || n == N
            w *= 0.5  # endpoint correction
        end
        integration_weights[n] = w * 0.5  # scale by half-interval h = 0.5
    end

    return RadialDomain(N, 1:N, r, dr_matrices, radial_laplacian, integration_weights)
end


# range_local helper mirroring PencilArrays API but returning the logical-order
# ranges directly from the pencil metadata.
function range_local(pencil::Pencil{3}, dim::Int)
    return pencil.axes_local[dim]
end

# Alias for compatibility - delegates to range_local
get_local_range(pencil::Pencil{3}, dim::Int) = range_local(pencil, dim)

function range_local(pencil::Pencil{3})
    return pencil.axes_local
end

function get_local_indices(pencil::Pencil{3})
    return range_local(pencil)
end

# Access patterns for PencilArrays
function local_data_size(field::SHTnsSpecField{T}) where T
    return size_local(field.pencil)
end

function local_data_size(field::SHTnsPhysField{T}) where T
    return size_local(field.pencil)
end

# Safe accessors that respect PencilArray's local data
function get_local_data(field::SHTnsSpecField{T}) where T
    return (real=parent(field.data_real), imag=parent(field.data_imag))
end

function get_local_data(field::SHTnsPhysField{T}) where T
    return parent(field.data)
end

# ------------------------------------------------------------------------------
# Base interface implementations for PencilArray-backed field types
# ------------------------------------------------------------------------------

function Base.similar(field::SHTnsSpecField{T}) where T
    return Base.similar(field, T)
end

function Base.similar(field::SHTnsSpecField{T}, ::Type{S}) where {T,S<:Number}
    data_real = PencilArray{S}(undef, field.pencil)
    data_imag = PencilArray{S}(undef, field.pencil)
    fill!(parent(data_real), zero(S))
    fill!(parent(data_imag), zero(S))
    bc_inner = copy(field.bc_type_inner)
    bc_outer = copy(field.bc_type_outer)
    boundary_values = zeros(S, size(field.boundary_values, 1), size(field.boundary_values, 2))
    return SHTnsSpecField{S}(field.config, field.nlm,
                                 data_real, data_imag, field.pencil,
                                 bc_inner, bc_outer, boundary_values)
end

function Base.copy(field::SHTnsSpecField{T}) where T
    duplicate = similar(field)
    parent(duplicate.data_real) .= parent(field.data_real)
    parent(duplicate.data_imag) .= parent(field.data_imag)
    duplicate.bc_type_inner .= field.bc_type_inner
    duplicate.bc_type_outer .= field.bc_type_outer
    duplicate.boundary_values .= field.boundary_values
    return duplicate
end

function Base.similar(field::SHTnsPhysField{T}) where T
    return Base.similar(field, T)
end

function Base.similar(field::SHTnsPhysField{T}, ::Type{S}) where {T,S<:Number}
    data = PencilArray{S}(undef, field.pencil)
    fill!(parent(data), zero(S))
    return SHTnsPhysField{S}(field.config, field.nlat, field.nlon, data, field.pencil)
end

function Base.copy(field::SHTnsPhysField{T}) where T
    duplicate = similar(field)
    parent(duplicate.data) .= parent(field.data)
    return duplicate
end

# export SHTnsSpecField, SHTnsPhysField, SHTnsVectorField, SHTnsTorPolField
# export RadialDomain, create_shtns_spectral_field, create_shtns_physical_field
# export create_shtns_vector_field, create_radial_domain
# end
