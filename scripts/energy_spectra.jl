#!/usr/bin/env julia

"""
Compute kinetic and magnetic energy spectra E_k(l,m) and E_b(l,m) from a NetCDF output file,
and save results to a single JLD2 file.

Usage:
  # Single file, auto output name (<input>_spectra.jld2)
  julia --project=. scripts/energy_spectra.jl <input.nc>

  # Single file with explicit output path
  julia --project=. scripts/energy_spectra.jl <input.nc> <output.jld2>

  # Multiple files (each saves to <input>_spectra.jld2 or into --outdir)
  julia --project=. scripts/energy_spectra.jl <f1.nc> <f2.nc> ... [--outdir=dir]

Inputs are expected to follow Geodynamo.jl output structure with spectral variables:
  - velocity_toroidal_real/imag[spectral_mode, r]
  - velocity_poloidal_real/imag[spectral_mode, r]
  - magnetic_toroidal_real/imag[spectral_mode, r]
  - magnetic_poloidal_real/imag[spectral_mode, r]
  - l_values[spectral_mode], m_values[spectral_mode]

Output (JLD2):
  - l_values::Vector{Int}
  - m_values::Vector{Int}
  - Ek_lm::Vector{Float64}
  - Eb_lm::Vector{Float64}
  - Ek_l::Vector{Float64}
  - Eb_l::Vector{Float64}
  - Ek_m::Vector{Float64}
  - Eb_m::Vector{Float64}
  - metadata::Dict (if available): geometry, time, step
"""

using NetCDF
using Printf
using JLD2
using Statistics
using SHTnsKit

function read_var(nc, name)
    varid = NetCDF.varid(nc, name)
    varid == -1 && error("Variable '$name' not found in file")
    return NetCDF.readvar(nc, name)
end

function load_spectral_data(nc)
    l_values = Int.(read_var(nc, "l_values"))
    m_values = Int.(read_var(nc, "m_values"))

    has_velocity = (NetCDF.varid(nc, "velocity_toroidal_real") != -1)
    has_magnetic = (NetCDF.varid(nc, "magnetic_toroidal_real") != -1)

    if !has_velocity && !has_magnetic
        error("No spectral velocity or magnetic variables found in file")
    end

    # Helper to load a pair of real/imag arrays if present, else zeros
    function load_pair(prefix)
        if NetCDF.varid(nc, "$(prefix)_real") != -1 && NetCDF.varid(nc, "$(prefix)_imag") != -1
            real = Float64.(read_var(nc, "$(prefix)_real"))
            imag = Float64.(read_var(nc, "$(prefix)_imag"))
        else
            real = zeros(Float64, length(l_values), 1)
            imag = zeros(Float64, length(m_values), 1)
        end
        return real, imag
    end

    vtor_r, vtor_i = load_pair("velocity_toroidal")
    vpol_r, vpol_i = load_pair("velocity_poloidal")
    mtor_r, mtor_i = load_pair("magnetic_toroidal")
    mpol_r, mpol_i = load_pair("magnetic_poloidal")

    return (
        l_values = l_values,
        m_values = m_values,
        vtor = (vtor_r, vtor_i),
        vpol = (vpol_r, vpol_i),
        mtor = (mtor_r, mtor_i),
        mpol = (mpol_r, mpol_i),
    )
end

"""
    energy_per_mode_scalar(real, imag)

Compute per-(l,m) scalar energy by summing |a_lm(r)|^2 across radius.
Assumes SHTnsKit orthonormal spherical harmonics (norm=:orthonormal), so
∫ |f(θ,φ)|^2 dΩ = Σ_lm |a_lm|^2. Returns a vector of length nlm.
"""
function energy_per_mode_scalar(real::AbstractMatrix, imag::AbstractMatrix)
    @assert size(real) == size(imag)
    nlm, nr = size(real)
    e = zeros(Float64, nlm)
    @inbounds for i in 1:nlm
        s = 0.0
        for k in 1:nr
            s += real[i,k]^2 + imag[i,k]^2
        end
        e[i] = s
    end
    return e
end

"""
    energy_per_mode_vector(l_values, tor_real, tor_imag, pol_real, pol_imag)

Compute per-(l,m) vector energy using toroidal/poloidal coefficients under
SHTnsKit orthonormalization. For a unit sphere surface integral, the modal
energy density is proportional to l(l+1) (|T_lm|^2 + |P_lm|^2).
We sum over radius index to obtain a depth-integrated spectrum.
"""
function energy_per_mode_vector(l_values::AbstractVector{<:Integer},
                                tor_real::AbstractMatrix, tor_imag::AbstractMatrix,
                                pol_real::AbstractMatrix, pol_imag::AbstractMatrix)
    @assert size(tor_real) == size(tor_imag) == size(pol_real) == size(pol_imag)
    nlm, nr = size(tor_real)
    e = zeros(Float64, nlm)
    @inbounds for i in 1:nlm
        l = l_values[i]
        lfac = float(l*(l+1))
        s = 0.0
        for k in 1:nr
            t2 = tor_real[i,k]^2 + tor_imag[i,k]^2
            p2 = pol_real[i,k]^2 + pol_imag[i,k]^2
            s += lfac * (t2 + p2)
        end
        e[i] = s
    end
    return e
end

function compute_spectra(data)
    l = data.l_values
    m = data.m_values
    nlm = length(l)

    # Kinetic energy per (l,m) with SHTnsKit orthonormalization:
    # E_k(l,m) ∝ l(l+1)(|T_lm|^2 + |P_lm|^2), summed across radius
    Ek_lm = energy_per_mode_vector(l, data.vtor[1], data.vtor[2], data.vpol[1], data.vpol[2])
    # Magnetic energy per (l,m) with the same modal form
    Eb_lm = energy_per_mode_vector(l, data.mtor[1], data.mtor[2], data.mpol[1], data.mpol[2])

    # Aggregate by l and m
    lmax = maximum(l)
    mmax = maximum(m)
    Ek_l = zeros(Float64, lmax+1)
    Eb_l = zeros(Float64, lmax+1)
    Ek_m = zeros(Float64, mmax+1)
    Eb_m = zeros(Float64, mmax+1)

    @inbounds for i in 1:nlm
        ℓ = l[i]
        𝓶 = m[i]
        Ek_l[ℓ+1] += Ek_lm[i]
        Eb_l[ℓ+1] += Eb_lm[i]
        Ek_m[𝓶+1] += Ek_lm[i]
        Eb_m[𝓶+1] += Eb_lm[i]
    end

    return (Ek_lm=Ek_lm, Eb_lm=Eb_lm, Ek_l=Ek_l, Eb_l=Eb_l, Ek_m=Ek_m, Eb_m=Eb_m)
end

function main()
    if isempty(ARGS)
        println("Usage: julia --project=. scripts/energy_spectra.jl <input.nc> [output.jld2] | <f1.nc> <f2.nc> ... [--outdir=dir]")
        return
    end

    # Parse optional outdir flag
    inputs = String[]
    outdir_flag = ""
    for a in ARGS
        if startswith(a, "--outdir=")
            outdir_flag = abspath(split(a, "=", limit=2)[2])
        else
            push!(inputs, a)
        end
    end

    if length(inputs) == 1
        input = inputs[0+1]
        # Decide output path: if a jld2 is provided as second arg (legacy mode), handle below
        outpath = joinpath(isempty(outdir_flag) ? dirname(input) : outdir_flag,
                           string(splitext(basename(input))[1], "_spectra.jld2"))
        outdir = dirname(outpath)
        isdir(outdir) || mkpath(outdir)

        nc = NetCDF.open(input, NC_NOWRITE)
        try
            data = load_spectral_data(nc)
            spectra = compute_spectra(data)

            # Try to collect basic metadata
            meta = Dict{String,Any}()
            try
                if NetCDF.varid(nc, "time") != -1
                    meta["time"] = NetCDF.readvar(nc, "time")[1]
                end
                if NetCDF.varid(nc, "step") != -1
                    meta["step"] = NetCDF.readvar(nc, "step")[1]
                end
                # Global attribute "geometry" if present
                try
                    meta["geometry"] = NetCDF.getatt(nc, NetCDF.NC_GLOBAL, "geometry")
                catch
                end
            catch
            end

            @save outpath data.l_values data.m_values spectra.Ek_lm spectra.Eb_lm spectra.Ek_l spectra.Eb_l spectra.Ek_m spectra.Eb_m meta
            println(@sprintf("Wrote spectra to %s", outpath))
        finally
            NetCDF.close(nc)
        end
        return
    end

    # If two args and second is .jld2, treat as explicit output path (legacy path)
    if length(inputs) == 2 && endswith(lowercase(inputs[2]), ".jld2")
        input, outpath = inputs[1], inputs[2]
        outdir = dirname(outpath)
        isdir(outdir) || mkpath(outdir)
        nc = NetCDF.open(input, NC_NOWRITE)
        try
            data = load_spectral_data(nc)
            spectra = compute_spectra(data)
            meta = Dict{String,Any}()
            try
                if NetCDF.varid(nc, "time") != -1
                    meta["time"] = NetCDF.readvar(nc, "time")[1]
                end
                if NetCDF.varid(nc, "step") != -1
                    meta["step"] = NetCDF.readvar(nc, "step")[1]
                end
                try meta["geometry"] = NetCDF.getatt(nc, NetCDF.NC_GLOBAL, "geometry") catch end
            catch
            end
            @save outpath data.l_values data.m_values spectra.Ek_lm spectra.Eb_lm spectra.Ek_l spectra.Eb_l spectra.Ek_m spectra.Eb_m meta
            println(@sprintf("Wrote spectra to %s", outpath))
        finally
            NetCDF.close(nc)
        end
        return
    end

    # Multiple inputs: process each, derive per-file output path (optionally under --outdir)
    for input in inputs
        if !endswith(lowercase(input), ".nc")
            @warn "Skipping non-NC file $input"
            continue
        end
        outpath = joinpath(isempty(outdir_flag) ? dirname(input) : outdir_flag,
                           string(splitext(basename(input))[1], "_spectra.jld2"))
        outdir = dirname(outpath)
        isdir(outdir) || mkpath(outdir)
        nc = NetCDF.open(input, NC_NOWRITE)
        try
            data = load_spectral_data(nc)
            spectra = compute_spectra(data)
            meta = Dict{String,Any}()
            try
                if NetCDF.varid(nc, "time") != -1
                    meta["time"] = NetCDF.readvar(nc, "time")[1]
                end
                if NetCDF.varid(nc, "step") != -1
                    meta["step"] = NetCDF.readvar(nc, "step")[1]
                end
                try meta["geometry"] = NetCDF.getatt(nc, NetCDF.NC_GLOBAL, "geometry") catch end
            catch
            end
            @save outpath data.l_values data.m_values spectra.Ek_lm spectra.Eb_lm spectra.Ek_l spectra.Eb_l spectra.Ek_m spectra.Eb_m meta
            println(@sprintf("Wrote spectra to %s", outpath))
        finally
            NetCDF.close(nc)
        end
    end
end

isinteractive() || main()
