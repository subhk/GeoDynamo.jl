# ================================================================================
# Variable Types with SHTnsKit Integration
# ================================================================================
    
# Field types that work with PencilArrays using SHTnsKit
mutable struct SHTnsSpectralField{T<:Number}
    config::AbstractSHTnsConfig
    nlm::Int
    data_real::PencilArray{T,3}
    data_imag::PencilArray{T,3}
    pencil::Pencil{3}  # Store pencil for local range info
    bc_type_inner::Vector{Int}
    bc_type_outer::Vector{Int}
    boundary_values::Matrix{T}          # [2, nlm] inner/outer boundary spectral values
end

# Physical field on SHTnsKit grid
struct SHTnsPhysicalField{T<:Number}
    config::AbstractSHTnsConfig
    nlat::Int
    nlon::Int
    data::PencilArray{T,3}  # Single array, transpose as needed
    pencil::Pencil{3}       # Current pencil orientation
end


# Vector field with SHTns
struct SHTnsVectorField{T<:Number}
    r_component::SHTnsPhysicalField{T}
    θ_component::SHTnsPhysicalField{T}
    φ_component::SHTnsPhysicalField{T}
end

# Toroidal-Poloidal decomposition with SHTns
struct SHTnsTorPolField{T<:Number}
    toroidal::SHTnsSpectralField{T}
    poloidal::SHTnsSpectralField{T}
end

# Radial domain (unchanged)
struct RadialDomain
    N::Int
    local_range::UnitRange{Int}
    r::Matrix{Float64}
    dr_matrices::Vector{Matrix{Float64}}
    radial_laplacian::Matrix{Float64}
    integration_weights::Vector{Float64}
end

# Constructor functions compatible with PencilArrays using SHTnsKit
function create_shtns_spectral_field(::Type{T}, config::AbstractSHTnsConfig, 
                                    oc_domain::RadialDomain,
                                    pencil_spec::Pencil{3}) where T
    nlm = config.nlm
    
    # Create PencilArrays with the given pencil
    data_real = PencilArray{T}(undef, pencil_spec)
    data_imag = PencilArray{T}(undef, pencil_spec)

    # Initialize to zero
    fill!(parent(data_real), zero(T))
    fill!(parent(data_imag), zero(T))

    bc_inner = ones(Int, nlm)
    bc_outer = ones(Int, nlm)
    boundary_vals = zeros(T, 2, nlm)

    return SHTnsSpectralField{T}(config, nlm,
                        data_real, data_imag, pencil_spec,
                        bc_inner, bc_outer, boundary_vals)
end


function create_shtns_physical_field(::Type{T}, config::AbstractSHTnsConfig,
                                    oc_domain::RadialDomain,
                                    pencil::Pencil{3}) where T
    nlat = config.nlat
    nlon = config.nlon
    
    # Create a single PencilArray
    data = PencilArray{T}(undef, pencil)
    fill!(parent(data), zero(T))
    
    return SHTnsPhysicalField{T}(config, nlat, nlon, data, pencil)
end


function create_shtns_vector_field(::Type{T}, config::AbstractSHTnsConfig,
                                    oc_domain::RadialDomain,
                                    pencils) where T
    if pencils isa NamedTuple
        pencil_θ = hasproperty(pencils, Symbol("θ")) ? getproperty(pencils, Symbol("θ")) : getproperty(pencils, :theta)
        pencil_φ = hasproperty(pencils, Symbol("φ")) ? getproperty(pencils, Symbol("φ")) : getproperty(pencils, :phi)
        pencil_r = getproperty(pencils, :r)
    else
        pencil_θ, pencil_φ, pencil_r = pencils
    end
    
    # Create each component with the r-pencil (contiguous in r)
    r_comp = create_shtns_physical_field(T, config, oc_domain, pencil_r)
    θ_comp = create_shtns_physical_field(T, config, oc_domain, pencil_r)
    φ_comp = create_shtns_physical_field(T, config, oc_domain, pencil_r)
    
    return SHTnsVectorField{T}(r_comp, θ_comp, φ_comp)
end


function create_radial_domain(nr::Int=i_N)
    N = nr
    
    r = zeros(N, 7)
    for n in 1:N
        r[n, 4] = 0.5 * (1.0 + cos(π * (N - n) / (N - 1)))
    end
    
    ri = d_rratio / (1.0 - d_rratio)
    r[:, 4] .+= ri
    
    for p in 1:7
        if p != 4
            power = p - 4
            r[:, p] = r[:, 4] .^ power
        end
    end
    
    dr_matrices         = [zeros(2*i_KL+1, N) for _ in 1:3]
    radial_laplacian    = zeros(2*i_KL+1, N)
    integration_weights = zeros(N)
    
    return RadialDomain(N, 1:N, r, dr_matrices, radial_laplacian, integration_weights)
end


# Helper functions for working with local portions of PencilArrays
function get_local_range(pencil::Pencil{3}, dim::Int)
    return pencil.axes_local[dim]
end

# range_local helper mirroring PencilArrays API but returning the logical-order
# ranges directly from the pencil metadata.
function range_local(pencil::Pencil{3}, dim::Int)
    return pencil.axes_local[dim]
end

function range_local(pencil::Pencil{3})
    return pencil.axes_local
end

function get_local_indices(pencil::Pencil{3})
    return range_local(pencil)
end

# Access patterns for PencilArrays
function local_data_size(field::SHTnsSpectralField{T}) where T
    return size_local(field.pencil)
end

function local_data_size(field::SHTnsPhysicalField{T}) where T
    return size_local(field.pencil)
end

# Safe accessors that respect PencilArray's local data
function get_local_data(field::SHTnsSpectralField{T}) where T
    return (real=parent(field.data_real), imag=parent(field.data_imag))
end

function get_local_data(field::SHTnsPhysicalField{T}) where T
    return parent(field.data)
end

# ------------------------------------------------------------------------------
# Base interface implementations for PencilArray-backed field types
# ------------------------------------------------------------------------------

function Base.similar(field::SHTnsSpectralField{T}) where T
    return Base.similar(field, T)
end

function Base.similar(field::SHTnsSpectralField{T}, ::Type{S}) where {T,S<:Number}
    data_real = PencilArray{S}(undef, field.pencil)
    data_imag = PencilArray{S}(undef, field.pencil)
    fill!(parent(data_real), zero(S))
    fill!(parent(data_imag), zero(S))
    bc_inner = copy(field.bc_type_inner)
    bc_outer = copy(field.bc_type_outer)
    boundary_values = zeros(S, size(field.boundary_values, 1), size(field.boundary_values, 2))
    return SHTnsSpectralField{S}(field.config, field.nlm,
                                 data_real, data_imag, field.pencil,
                                 bc_inner, bc_outer, boundary_values)
end

function Base.copy(field::SHTnsSpectralField{T}) where T
    duplicate = similar(field)
    parent(duplicate.data_real) .= parent(field.data_real)
    parent(duplicate.data_imag) .= parent(field.data_imag)
    duplicate.bc_type_inner .= field.bc_type_inner
    duplicate.bc_type_outer .= field.bc_type_outer
    duplicate.boundary_values .= field.boundary_values
    return duplicate
end

function Base.similar(field::SHTnsPhysicalField{T}) where T
    return Base.similar(field, T)
end

function Base.similar(field::SHTnsPhysicalField{T}, ::Type{S}) where {T,S<:Number}
    data = PencilArray{S}(undef, field.pencil)
    fill!(parent(data), zero(S))
    return SHTnsPhysicalField{S}(field.config, field.nlat, field.nlon, data, field.pencil)
end

# export SHTnsSpectralField, SHTnsPhysicalField, SHTnsVectorField, SHTnsTorPolField
# export RadialDomain, create_shtns_spectral_field, create_shtns_physical_field
# export create_shtns_vector_field, create_radial_domain
# end
