#!/usr/bin/env julia

# m=0 (axisymmetric) r-θ plot reconstructed from spectral data.
#
# Reads spectral coefficients (e.g., velocity_toroidal_real/imag and velocity_poloidal_real/imag)
# along with l_values, m_values from a NetCDF output. Selects m=0 modes, performs a
# scalar spherical-harmonic synthesis Y_{l0}(θ) to transform toroidal/poloidal scalars
# into physical space over θ for all radii, then plots r–θ heatmaps.
#
# Note (vector fields): This script reconstructs physical vector components
# u_r, u_θ, u_φ for m=0 by combining toroidal and poloidal scalars via
#   u_r   = Σ [ℓ(ℓ+1)/r^2] P_l0(r) Y_l0(θ)
#   u_θ   = Σ [ (1/r) ∂(r P_l0)/∂r ] ∂Y_l0/∂θ
#   u_φ   = Σ [ -(1/r) T_l0(r) ] ∂Y_l0/∂θ
# using Y_l0(θ)=√((2ℓ+1)/(4π)) P_l(cosθ) by default. Disable normalization with --no-norm.
#
# Note (scalar fields): For temperature/composition, if spectral variables
# temperature_spectral_* or composition_spectral_* exist, synthesis is used.
# Otherwise, the script falls back to computing the zonal mean (phi-average) of the
# physical variable temperature/ composition to obtain m=0.
#
# Usage examples:
#   # Velocity toroidal m=0 (default normalization)
#   julia --project=. scripts/plot_m0_from_torpol.jl out.nc velocity toroidal
#   # Magnetic poloidal m=0 with custom clim
#   julia --project=. scripts/plot_m0_from_torpol.jl out.nc magnetic poloidal --clim -1 1
#   # Show both toroidal and poloidal (two subplots)
#   julia --project=. scripts/plot_m0_from_torpol.jl out.nc velocity both --cmap balance
#   # Scalars: temperature m=0 from spectral coeffs (or fall back to phi-average)
#   julia --project=. scripts/plot_m0_from_torpol.jl out.nc temperature --backend shtns
#   # Scalars: composition m=0
#   julia --project=. scripts/plot_m0_from_torpol.jl out.nc composition --backend analytic --cmap magma
#   # Disable Ylm normalization (use raw P_l):
#   julia --project=. scripts/plot_m0_from_torpol.jl out.nc velocity toroidal --no-norm

using NCDatasets

mutable struct Opts
    cmap::Union{Nothing,Symbol}
    clim::Union{Nothing,Tuple{Float64,Float64}}
    degrees::Bool
    nonorm::Bool
    veccomp::Union{Nothing,Symbol}  # :ur | :utheta | :uphi for vectors
    backend::Symbol                 # :shtns or :analytic for vector synthesis
    strict::Bool                    # when true, disable scalar physical fallback
end

Opts() = Opts(nothing, nothing, true, false, :ur, :shtns, false)

function parse_args()
    path = get(ARGS, 1, "")
    family = lowercase(get(ARGS, 2, "velocity"))   # "velocity" | "magnetic" | "temperature" | "composition"
    # For vectors, the component is selected with --vec ur|utheta|uphi; we still synthesize using both toroidal+poloidal as needed.
    which  = lowercase(get(ARGS, 3, family in ("temperature","composition") ? "scalar" : "vector"))
    extras = ARGS[4:end]
    opts = Opts()
    i = 1
    while i <= length(extras)
        a = extras[i]
        if a == "--cmap" && i+1 <= length(extras)
            opts.cmap = Symbol(extras[i+1]); i += 2
        elseif a == "--clim" && i+2 <= length(extras)
            vmin = parse(Float64, extras[i+1]); vmax = parse(Float64, extras[i+2])
            opts.clim = (vmin, vmax); i += 3
        elseif a == "--degrees"
            opts.degrees = true; i += 1
        elseif a == "--radians"
            opts.degrees = false; i += 1
        elseif a == "--no-norm"
            opts.nonorm = true; i += 1
        elseif a == "--vec" && i+1 <= length(extras)
            v = lowercase(extras[i+1])
            v ∈ ("ur","utheta","uphi") || error("--vec must be one of ur|utheta|uphi")
            opts.veccomp = Symbol(v); i += 2
        elseif a == "--backend" && i+1 <= length(extras)
            b = lowercase(extras[i+1])
            b ∈ ("shtns","analytic") || error("--backend must be shtns|analytic")
            opts.backend = Symbol(b); i += 2
        elseif a == "--no-fallback" || a == "--strict"
            opts.strict = true; i += 1
        else
            error("Unknown/malformed option '$a'. Supported: --cmap NAME --clim MIN MAX --degrees|--radians --no-norm")
        end
    end
    family ∈ ("velocity","magnetic","temperature","composition") || error("family must be 'velocity','magnetic','temperature', or 'composition'")
    which = family in ("temperature","composition") ? "scalar" : "vector"
    return path, family, which, opts
end

function ensure_backend()
    local backend
    try
        @eval using GLMakie
        backend = :gl
    catch
        try
            @eval using CairoMakie
            backend = :cairo
        catch
            error("No Makie backend found. Install GLMakie or CairoMakie: import Pkg; Pkg.add(\"GLMakie\")")
        end
    end
    return backend
end

function read_spectral(path::AbstractString, family::String)
    isfile(path) || error("File not found: $path")
    ds = NCDataset(path)
    try
        haskey(ds, "theta") || error("theta not found")
        haskey(ds, "phi") || error("phi not found")
        haskey(ds, "r") || error("r not found")
        haskey(ds, "l_values") || error("l_values not found (spectral metadata)")
        haskey(ds, "m_values") || error("m_values not found (spectral metadata)")
        theta = vec(ds["theta"][:])
        rvals = vec(ds["r"][:])
        phi   = vec(ds["phi"][:])
        lvals = vec(ds["l_values"][:])
        mvals = vec(ds["m_values"][:])
        if family in ("velocity","magnetic")
            tor_r = "$(family)_toroidal_real"; tor_i = "$(family)_toroidal_imag"
            pol_r = "$(family)_poloidal_real"; pol_i = "$(family)_poloidal_imag"
            haskey(ds, tor_r) || error("$tor_r not found in file")
            haskey(ds, tor_i) || error("$tor_i not found in file")
            haskey(ds, pol_r) || error("$pol_r not found in file")
            haskey(ds, pol_i) || error("$pol_i not found in file")
            tor_real = Array(ds[tor_r][:])
            tor_imag = Array(ds[tor_i][:])
            pol_real = Array(ds[pol_r][:])
            pol_imag = Array(ds[pol_i][:])
            size(tor_real,1) == length(lvals) || @warn "nlm mismatch for toroidal"
            size(pol_real,1) == length(lvals) || @warn "nlm mismatch for poloidal"
            return (mode=:vector, theta=theta, phi=phi, r=rvals, l=lvals, m=mvals,
                    tor_real=tor_real, tor_imag=tor_imag,
                    pol_real=pol_real, pol_imag=pol_imag, ds=ds)
        else
            base = family == "temperature" ? "temperature_spectral" : "composition_spectral"
            sca_r = "$(base)_real"; sca_i = "$(base)_imag"
            haskey(ds, sca_r) || error("$sca_r not found in file (no spectral scalars)")
            haskey(ds, sca_i) || error("$sca_i not found in file (no spectral scalars)")
            sca_real = Array(ds[sca_r][:])
            sca_imag = Array(ds[sca_i][:])
            size(sca_real,1) == length(lvals) || @warn "nlm mismatch for scalar"
            return (mode=:scalar, theta=theta, phi=phi, r=rvals, l=lvals, m=mvals,
                    sca_real=sca_real, sca_imag=sca_imag, ds=ds)
        end
    catch
        close(ds); rethrow()
    end
end

# Compute Legendre P_l(x) for l=0..Lmax at all x via recurrence
function legendre_P_all(x::AbstractVector{<:Real}, Lmax::Int)
    nx = length(x)
    P = Array{Float64}(undef, Lmax+1, nx)
    @inbounds for j in 1:nx
        P[1,j] = 1.0       # l=0
        if Lmax >= 1
            P[2,j] = x[j]  # l=1
        end
    end
    @inbounds for l in 1:(Lmax-1)
        α = (2l + 1) / (l + 1)
        β = l / (l + 1)
        for j in 1:nx
            P[l+2, j] = α * x[j] * P[l+1, j] - β * P[l, j]
        end
    end
    return P  # P[l+1, j] corresponds to P_l(x[j])
end

# Compute associated Legendre P_l^1(x) for l=0..Lmax via recurrence with Condon-Shortley phase.
function legendre_P1_all(x::AbstractVector{<:Real}, Lmax::Int)
    nx = length(x)
    P1 = Array{Float64}(undef, Lmax+1, nx)
    @inbounds for j in 1:nx
        P1[1,j] = 0.0                 # l=0, m=1 -> 0
        if Lmax >= 1
            P1[2,j] = -sqrt(max(0.0, 1 - x[j]^2))  # l=1, m=1: -sqrt(1-x^2)
        end
    end
    @inbounds for l in 2:Lmax
        for j in 1:nx
            # Recurrence: P_l^1 = (2l-1) x P_{l-1}^1 - (l-1) P_{l-2}^1
            P1[l+1, j] = (2l - 1) * x[j] * P1[l, j] - (l - 1) * P1[l-1, j]
        end
    end
    return P1  # P1[l+1, j] corresponds to P_l^1(x[j])
end

function synthesize_m0(theta::AbstractVector, lvals::AbstractVector{<:Integer}, mvals::AbstractVector{<:Integer},
                       coeffs::AbstractMatrix; normalize::Bool=true)
    # coeffs is (nlm, nr) for either toroidal or poloidal real-part (complex OK but imag ~ 0 for m=0)
    # Returns matrix F(θ, r) of size (length(theta), nr)
    idxs = findall(==(0), mvals)
    isempty(idxs) && error("No m=0 coefficients found")
    # Determine max l among m=0 modes to size Legendre table
    Lmax = maximum(lvals[idxs])
    x = cos.(theta)
    P = legendre_P_all(x, Lmax)   # (Lmax+1) × nθ
    norm = normalize ? (l -> sqrt((2l+1)/(4π))) : (l -> 1.0)
    nθ = length(theta)
    nr = size(coeffs, 2)
    F = zeros(Float64, nθ, nr)
    @inbounds for (k, i) in enumerate(idxs)
        l = lvals[i]
        Pl = view(P, l+1, :)  # row for degree l
        for j in 1:nr
            c = coeffs[i, j]
            @simd for t in 1:nθ
                F[t, j] += norm(l) * Pl[t] * c
            end
        end
    end
    return F
end

function synthesize_vector_m0(theta::AbstractVector, r::AbstractVector,
                               lvals::AbstractVector{<:Integer}, mvals::AbstractVector{<:Integer},
                               pol_coeffs::AbstractMatrix, tor_coeffs::AbstractMatrix;
                               component::Symbol=:ur, normalize::Bool=true)
    # coeff matrices are (nlm, nr) real parts; component ∈ (:ur, :utheta, :uphi)
    idxs = findall(==(0), mvals)
    isempty(idxs) && error("No m=0 coefficients found for vector synthesis")
    Lmax = maximum(lvals[idxs])
    x = cos.(theta)
    P = legendre_P_all(x, Lmax)
    P1 = legendre_P1_all(x, Lmax)
    norm = normalize ? (l -> sqrt((2l+1)/(4π))) : (l -> 1.0)
    nθ = length(theta)
    nr = size(pol_coeffs, 2)
    F = zeros(Float64, nθ, nr)
    # Precompute 1/r and 1/r^2, and d(r*P_l0)/dr
    invr = 1.0 ./ r
    invr2 = invr .^ 2
    # Compute d(r pol)/dr per (l index, r): numeric diff
    function dr_of_rP(row::AbstractVector, r::AbstractVector)
        nr = length(r)
        out = similar(row)
        out[1] = (r[2]*row[2] - r[1]*row[1]) / (r[2]-r[1])
        @inbounds for k in 2:(nr-1)
            out[k] = (r[k+1]*row[k+1] - r[k-1]*row[k-1]) / (r[k+1]-r[k-1])
        end
        out[end] = (r[end]*row[end] - r[end-1]*row[end-1]) / (r[end]-r[end-1])
        return out
    end
    if component == :ur
        @inbounds for (k, i) in enumerate(idxs)
            l = lvals[i]
            Yl = view(P, l+1, :)
            Nl = norm(l)
            for j in 1:nr
                cP = pol_coeffs[i, j]
                s = Nl * l*(l+1) * invr2[j] * cP
                @simd for t in 1:nθ
                    F[t, j] += s * Yl[t]
                end
            end
        end
    elseif component == :utheta
        # need d(rP)/dr and dY/dθ = Nl * P_l^1(cosθ)
        d_rP = Array{Float64}(undef, length(idxs), nr)
        for (k, i) in enumerate(idxs)
            d_rP[k, :] = dr_of_rP(view(pol_coeffs, i, :), r)
        end
        @inbounds for (k, i) in enumerate(idxs)
            l = lvals[i]
            dY = view(P1, l+1, :)
            Nl = norm(l)
            for j in 1:nr
                s = Nl * invr[j] * d_rP[k, j]
                @simd for t in 1:nθ
                    F[t, j] += s * dY[t]
                end
            end
        end
    elseif component == :uphi
        # depends only on toroidal, via - (1/r) ∂Y/∂θ = - (1/r) Nl P_l^1
        @inbounds for (k, i) in enumerate(idxs)
            l = lvals[i]
            dY = view(P1, l+1, :)
            Nl = norm(l)
            for j in 1:nr
                cT = tor_coeffs[i, j]
                s = -Nl * invr[j] * cT
                @simd for t in 1:nθ
                    F[t, j] += s * dY[t]
                end
            end
        end
    else
        error("Unknown vector component $(component)")
    end
    return F
end

# SHTnsKit-based synthesis for tangential components (uθ, uφ) using only m=0 modes.
function shtns_vector_m0_tangential(theta::AbstractVector, phi::AbstractVector,
                                    lvals::AbstractVector{<:Integer}, mvals::AbstractVector{<:Integer},
                                    pol_real::AbstractMatrix, pol_imag::AbstractMatrix,
                                    tor_real::AbstractMatrix, tor_imag::AbstractMatrix)
    @eval using SHTnsKit
    nlat = length(theta); nlon = length(phi); nr = size(pol_real, 2)
    lmax = maximum(lvals); mmax = maximum(mvals)
    # Build SHTnsKit config with orthonormal norm to match our Y_lm
    sht = SHTnsKit.create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon, norm=:orthonormal)
    SHTnsKit.prepare_plm_tables!(sht)
    # Precompute indices for m=0
    idxs = findall(==(0), mvals)
    # Prepare outputs
    uθ_m0 = zeros(Float64, nlat, nr)
    uφ_m0 = zeros(Float64, nlat, nr)
    for j in 1:nr
        # Build coeff matrices (l+1, m+1) with only m=0 filled
        Pol = zeros(ComplexF64, lmax+1, mmax+1)
        Tor = zeros(ComplexF64, lmax+1, mmax+1)
        @inbounds for i in idxs
            l = lvals[i]
            Pol[l+1, 1] = complex(pol_real[i, j], pol_imag[i, j])
            Tor[l+1, 1] = complex(tor_real[i, j], tor_imag[i, j])
        end
        vt, vp = SHTnsKit.SHsphtor_to_spat(sht, Pol, Tor; real_output=true)
        # Average over phi to get m=0 slice (should be phi-independent already)
        uθ_m0[:, j] .= sum(vt; dims=2)[:,1] ./ nlon
        uφ_m0[:, j] .= sum(vp; dims=2)[:,1] ./ nlon
    end
    return uθ_m0, uφ_m0
end

# SHTnsKit-based synthesis for radial component ur using only m=0 modes.
function shtns_vector_ur_m0(theta::AbstractVector, phi::AbstractVector, r::AbstractVector,
                            lvals::AbstractVector{<:Integer}, mvals::AbstractVector{<:Integer},
                            pol_real::AbstractMatrix, pol_imag::AbstractMatrix)
    @eval using SHTnsKit
    nlat = length(theta); nlon = length(phi); nr = size(pol_real, 2)
    lmax = maximum(lvals); mmax = maximum(mvals)
    sht = SHTnsKit.create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon, norm=:orthonormal)
    SHTnsKit.prepare_plm_tables!(sht)
    idxs = findall(==(0), mvals)
    ur_m0 = zeros(Float64, nlat, nr)
    for j in 1:nr
        C = zeros(ComplexF64, lmax+1, mmax+1)
        rinv2 = 1.0 / (r[j]^2)
        @inbounds for i in idxs
            l = lvals[i]
            # ur = Σ l(l+1)/r^2 * P_l0(r) Y_l0(θ); here pol_real holds P_l0(r)
            C[l+1, 1] = complex(l*(l+1)*rinv2 * pol_real[i, j], 0.0)
        end
        fθφ = SHTnsKit.synthesis(sht, C; real_output=true)
        ur_m0[:, j] .= sum(fθφ; dims=2)[:,1] ./ nlon
    end
    return ur_m0
end

# SHTnsKit-based synthesis for scalar m=0 using only m=0 modes.
function shtns_scalar_m0(theta::AbstractVector, phi::AbstractVector,
                         lvals::AbstractVector{<:Integer}, mvals::AbstractVector{<:Integer},
                         sca_real::AbstractMatrix, sca_imag::AbstractMatrix)
    @eval using SHTnsKit
    nlat = length(theta); nlon = length(phi); nr = size(sca_real, 2)
    lmax = maximum(lvals); mmax = maximum(mvals)
    sht = SHTnsKit.create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon, norm=:orthonormal)
    SHTnsKit.prepare_plm_tables!(sht)
    idxs = findall(==(0), mvals)
    m0 = zeros(Float64, nlat, nr)
    for j in 1:nr
        C = zeros(ComplexF64, lmax+1, mmax+1)
        @inbounds for i in idxs
            l = lvals[i]
            C[l+1, 1] = complex(sca_real[i, j], sca_imag[i, j])
        end
        fθφ = SHTnsKit.synthesis(sht, C; real_output=true)
        m0[:, j] .= sum(fθφ; dims=2)[:,1] ./ nlon
    end
    return m0
end

function main()
    path, family, which, opts = parse_args()
    # Try spectral path first; fall back to physical scalars for temperature/composition
    data = nothing
    try
        data = read_spectral(path, family)
    catch err
        if family in ("temperature","composition")
            # Fallback: use physical variable and zonal mean
            ds = NCDataset(path)
            try
                if opts.strict
                    rethrow(err)
                end
                haskey(ds, family) || rethrow(err)
                theta = vec(ds["theta"][:])
                rvals = vec(ds["r"][:])
                A = Array(ds[family][:])   # expect (theta,phi,r)
                ndims(A) == 3 || error("Expected $(family)[theta,phi,r] for fallback physical averaging")
                m0 = dropdims(mean(A; dims=2), dims=2)  # (theta,r)
                data = (mode=:scalar_phys, theta=theta, r=rvals, m0=m0, ds=ds)
            catch
                close(ds); rethrow()
            end
        else
            rethrow()
        end
    end

    # Prepare plotting
    backend = ensure_backend()
    if backend == :gl
        using GLMakie
    else
        using CairoMakie
    end

    θplot = opts.degrees ? data.theta .* (180 / pi) : data.theta
    xlab = opts.degrees ? "θ (deg)" : "θ (rad)"

    # Helper to plot a single heatmap
    function plot_hm(figcell, Xθr, title_suffix)
        ax = Axis(figcell; xlabel=xlab, ylabel="r", title="$(family) $(title_suffix) m=0")
        hm_kwargs = (; interpolate=false)
        if opts.cmap !== nothing
            hm_kwargs = merge(hm_kwargs, (; colormap = opts.cmap))
        end
        if opts.clim !== nothing
            hm_kwargs = merge(hm_kwargs, (; colorrange = opts.clim))
        end
        h = heatmap!(ax, θplot, data.r, Xθr; hm_kwargs...)
        Colorbar(parent(figcell), h, label="value")
        return ax
    end

    if data.mode == :vector
        V = nothing
        if opts.veccomp == :ur
            if opts.backend == :shtns
                V = shtns_vector_ur_m0(data.theta, data.phi, data.r, data.l, data.m,
                                       data.pol_real, data.pol_imag)
            else
                V = synthesize_vector_m0(data.theta, data.r, data.l, data.m,
                                         data.pol_real, data.tor_real;
                                         component=:ur, normalize=!opts.nonorm)
            end
        else
            if opts.backend == :shtns
                # Need phi grid for averaging; try to read from file (optional)
                dsphi = try
                    NCDataset(path)
                catch
                    nothing
                end
                phi = dsphi === nothing ? range(0, stop=2pi, length= max(4, 2*maximum(data.l)+1)+1)[1:end-1] |> collect : vec(dsphi["phi"][:])
                if dsphi !== nothing; close(dsphi); end
                uθ_m0, uφ_m0 = shtns_vector_m0_tangential(data.theta, phi, data.l, data.m,
                                                          data.pol_real, data.pol_imag,
                                                          data.tor_real, data.tor_imag)
                V = opts.veccomp == :utheta ? uθ_m0 : uφ_m0
            else
                V = synthesize_vector_m0(data.theta, data.r, data.l, data.m,
                                         data.pol_real, data.tor_real;
                                         component=opts.veccomp, normalize=!opts.nonorm)
            end
        end
        fig = Figure(resolution=(900, 650))
        compname = String(opts.veccomp)
        plot_hm(fig[1,1], V, compname)
        outpng = replace(basename(path), r"\.nc$" => "") * "_$(family)_$(compname)_m0_rtheta.png"
        save(outpng, fig); @info "Saved" outpng; display(fig)
    elseif data.mode == :scalar
        Sm0 = opts.backend == :shtns ?
              shtns_scalar_m0(data.theta, data.phi, data.l, data.m, data.sca_real, data.sca_imag) :
              synthesize_m0(data.theta, data.l, data.m, data.sca_real; normalize=!opts.nonorm)
        fig = Figure(resolution=(900, 650))
        plot_hm(fig[1,1], Sm0, family)
        outpng = replace(basename(path), r"\.nc$" => "") * "_$(family)_m0_rtheta.png"
        save(outpng, fig); @info "Saved" outpng; display(fig)
    else
        # scalar physical fallback already computed as (theta, r)
        fig = Figure(resolution=(900, 650))
        plot_hm(fig[1,1], data.m0, family)
        outpng = replace(basename(path), r"\.nc$" => "") * "_$(family)_m0_rtheta.png"
        save(outpng, fig); @info "Saved" outpng; display(fig)
    end

    close(data.ds)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
