#!/usr/bin/env julia

# Time-average z-helicity (u_z * ω_z) separated into positive and negative contributions.
#
# - Reads a sequence of NetCDF outputs (one per time) with velocity spectral coefficients
#   `velocity_toroidal_real/imag` and `velocity_poloidal_real/imag`, plus `theta, phi, r`.
# - Uses SHTnsKit to synthesize tangential components (u_θ, u_φ) on the stored grid and
#   reconstructs u_r from poloidal coefficients via u_r = L2/r * P_lm (orthonormal Ylm).
# - Computes vorticity ω = ∇ × u in spherical coordinates, then u_z = u_r cosθ - u_θ sinθ,
#   ω_z = ω_r cosθ - ω_θ sinθ, and z-helicity h_z = u_z * ω_z.
# - Accumulates time-averages separately for h_z > 0 and h_z < 0 at each grid cell.
# - Writes a JLD2 file with fields: `avg_pos`, `avg_neg`, `count_pos`, `count_neg`, and axes.
#
# Usage:
#   # Average all files in a directory (prefers rank_0000) and write JLD2
#   julia --project=. scripts/timeavg_zhelicity.jl outdir --out zhel_timeavg.jld2
#   # Restrict to time range
#   julia --project=. scripts/timeavg_zhelicity.jl outdir --tmin 1.0 --tmax 5.0 --out zhel_1_5.jld2

using NCDatasets
using JLD2
using GeoDynamo

const HAVE_SHTNSKIT = try
    @eval using SHTnsKit
    true
catch
    false
end

mutable struct Opts
    dir::String
    out::String
    tmin::Union{Nothing,Float64}
    tmax::Union{Nothing,Float64}
end

function parse_args()
    dir = get(ARGS, 1, "")
    dir != "" || error("Provide directory containing NetCDF outputs as first argument")
    opts = Opts(dir, "zhelicity_timeavg.jld2", nothing, nothing)
    i = 2
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--out" && i+1 <= length(ARGS)
            opts.out = ARGS[i+1]; i += 2
        elseif a == "--tmin" && i+1 <= length(ARGS)
            opts.tmin = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--tmax" && i+1 <= length(ARGS)
            opts.tmax = parse(Float64, ARGS[i+1]); i += 2
        else
            error("Unknown option $a. Use --out FILE --tmin T --tmax T")
        end
    end
    HAVE_SHTNSKIT || error("SHTnsKit.jl required. Install with: import Pkg; Pkg.add(\"SHTnsKit\")")
    return opts
end

time_of_file(path::AbstractString) = begin
    ds = NCDataset(path)
    try
        if haskey(ds, "time"); t = ds["time"][:]; return length(t)==1 ? float(t[1]) : float(first(t)); end
        if haskey(ds.attrib, "current_time"); return float(ds.attrib["current_time"]); end
        m = match(r"_time_([0-9p]+)", basename(path))
        m === nothing && return NaN
        parse(Float64, replace(m.captures[1], 'p' => '.'))
    finally
        close(ds)
    end
end

function list_nc(dir::String)
    files = filter(f->endswith(f, ".nc"), joinpath.(dir, readdir(dir)))
    r0 = filter(f->occursin("rank_0000", f), files)
    files = isempty(r0) ? files : r0
    times = [(f, time_of_file(f)) for f in files]
    sort(times; by=x->isfinite(x[2]) ? x[2] : Inf)
end

function coeffs_at_r(ds::NCDataset, r_idx::Int)
    l = vec(ds["l_values"][:]); m = vec(ds["m_values"][:])
    T_re = Array(ds["velocity_toroidal_real"][:]); T_im = Array(ds["velocity_toroidal_imag"][:])
    P_re = Array(ds["velocity_poloidal_real"][:]); P_im = Array(ds["velocity_poloidal_imag"][:])
    lmax = maximum(l); mmax = maximum(m)
    Tor = zeros(ComplexF64, lmax+1, mmax+1)
    Pol = zeros(ComplexF64, lmax+1, mmax+1)
    @inbounds for i in eachindex(l)
        ell = l[i]; mm = m[i]
        Tor[ell+1, mm+1] = complex(T_re[i, r_idx], T_im[i, r_idx])
        Pol[ell+1, mm+1] = complex(P_re[i, r_idx], P_im[i, r_idx])
    end
    return l, m, Tor, Pol
end

function synthesize_velocity!(u_r, u_θ, u_φ, sht, theta, phi, rvals, r_idx, l::Vector{Int}, Pol::Matrix{ComplexF64}, Tor::Matrix{ComplexF64})
    # Tangential from SHTnsKit
    vt, vp = SHTnsKit.SHsphtor_to_spat(sht, Pol, Tor; real_output=true)
    u_θ[:, :, r_idx] .= vt
    u_φ[:, :, r_idx] .= vp
    # Radial from poloidal scalar: u_r = L2/r * P_lm Y_lm
    lmax = maximum(l); mmax = size(Pol, 2)-1
    C = zeros(ComplexF64, lmax+1, mmax+1)
    r = rvals[r_idx]
    @inbounds for i in 1:(lmax+1)
        C[i,1:end] .= ( (i-1)*(i) / r ) .* Pol[i,1:end]
    end
    u_r[:, :, r_idx] .= SHTnsKit.synthesis(sht, C; real_output=true)
end

"""
Synthesize velocity and vorticity using GeoDynamo/SHTnsKit (spectral vorticity), avoiding finite differences.
Returns (u_r,u_θ,u_φ, ω_r,ω_θ,ω_φ) on the θ–φ–r grid in the NetCDF file.
"""
function synthesize_vel_and_vort_from_nc(ncpath::AbstractString)
    ds = NCDataset(ncpath)
    try
        θ = vec(ds["theta"][:]); φ = vec(ds["phi"][:]); r = vec(ds["r"][:])
        lvals = vec(ds["l_values"][:]); mvals = vec(ds["m_values"][:])
        vtor_r = Array(ds["velocity_toroidal_real"][:]); vtor_i = Array(ds["velocity_toroidal_imag"][:])
        vpol_r = Array(ds["velocity_poloidal_real"][:]); vpol_i = Array(ds["velocity_poloidal_imag"][:])
        lmax = maximum(lvals); mmax = maximum(mvals)
        nlat = length(θ); nlon = length(φ); nr = length(r)
        gcfg = GeoDynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon)
        pencils_nt = GeoDynamo.create_pencil_topology(gcfg)
        pencils = pencils_nt
        pencil_spec = pencils_nt.spec
        domain = GeoDynamo.create_radial_domain(nr)
        fields = GeoDynamo.create_shtns_velocity_fields(Float64, gcfg, domain, pencils, pencil_spec)
        # Load spectral coefficients (single-rank layout)
        spec_tor_r = parent(fields.toroidal.data_real); spec_tor_i = parent(fields.toroidal.data_imag)
        spec_pol_r = parent(fields.poloidal.data_real); spec_pol_i = parent(fields.poloidal.data_imag)
        nlm_local = min(size(spec_tor_r,1), size(vtor_r,1))
        for i2 in 1:nlm_local, k2 in 1:nr
            spec_tor_r[i2,1,k2] = Float64(vtor_r[i2,k2])
            spec_tor_i[i2,1,k2] = Float64(vtor_i[i2,k2])
            spec_pol_r[i2,1,k2] = Float64(vpol_r[i2,k2])
            spec_pol_i[i2,1,k2] = Float64(vpol_i[i2,k2])
        end
        # Spectral vorticity, then synthesize both vector fields
        GeoDynamo.compute_vorticity_spectral_full!(fields, domain)
        GeoDynamo.shtnskit_vector_synthesis!(fields.toroidal, fields.poloidal, fields.velocity)
        GeoDynamo.shtnskit_vector_synthesis!(fields.vort_toroidal, fields.vort_poloidal, fields.vorticity)
        u_r = parent(fields.velocity.r_component.data)
        u_θ = parent(fields.velocity.θ_component.data)
        u_φ = parent(fields.velocity.φ_component.data)
        ω_r = parent(fields.vorticity.r_component.data)
        ω_θ = parent(fields.vorticity.θ_component.data)
        ω_φ = parent(fields.vorticity.φ_component.data)
        return θ, φ, r, u_r, u_θ, u_φ, ω_r, ω_θ, ω_φ
    finally
        close(ds)
    end
end

function main()
    opts = parse_args()
    files_times = list_nc(opts.dir)
    # Filter by time range
    if opts.tmin !== nothing || opts.tmax !== nothing
        files_times = [(f,t) for (f,t) in files_times if (opts.tmin === nothing || t >= opts.tmin) && (opts.tmax === nothing || t <= opts.tmax)]
    end
    isempty(files_times) && error("No .nc files found matching time range")
    # Read grid from first file and init SHTnsKit
    ds0 = NCDataset(files_times[1][1])
    theta = vec(ds0["theta"][:]); phi = vec(ds0["phi"][:]); rvals = vec(ds0["r"][:])
    l = vec(ds0["l_values"][:]); m = vec(ds0["m_values"][:])
    lmax = maximum(l); mmax = maximum(m)
    sht = SHTnsKit.create_gauss_config(lmax, length(theta); mmax=mmax, nlon=length(phi), norm=:orthonormal)
    SHTnsKit.prepare_plm_tables!(sht)
    close(ds0)

    nθ, nφ, nr = length(theta), length(phi), length(rvals)
    sum_pos = zeros(Float64, nθ, nφ, nr)
    sum_neg = zeros(Float64, nθ, nφ, nr)
    cnt_pos = zeros(Int, nθ, nφ, nr)
    cnt_neg = zeros(Int, nθ, nφ, nr)

    for (path, t) in files_times
        try
            # Use spectral vorticity pipeline consistent with time_average_helicity.jl
            θf, φf, rf, u_r, u_θ, u_φ, ω_r, ω_θ, ω_φ = synthesize_vel_and_vort_from_nc(path)
            @assert length(θf)==nθ && length(φf)==nφ && length(rf)==nr
            # z-components and z-helicity
            cosθ = cos.(theta); sinθ = sin.(theta)
            # Broadcast over phi,r dims via reshape
            Cos = reshape(cosθ, nθ, 1, 1); Sin = reshape(sinθ, nθ, 1, 1)
            u_z = u_r .* Cos .- u_θ .* Sin
            ω_z = ω_r .* Cos .- ω_θ .* Sin
            h = u_z .* ω_z
            # Accumulate conditional time averages
            posmask = h .> 0
            negmask = h .< 0
            sum_pos .+= h .* posmask
            sum_neg .+= h .* negmask
            cnt_pos .+= posmask
            cnt_neg .+= negmask
        end
    end

    avg_pos = Array{Float64}(undef, nθ, nφ, nr)
    avg_neg = Array{Float64}(undef, nθ, nφ, nr)
    @inbounds for k in 1:nr, j in 1:nφ, i in 1:nθ
        cp = cnt_pos[i,j,k]; cn = cnt_neg[i,j,k]
        avg_pos[i,j,k] = cp>0 ? sum_pos[i,j,k]/cp : NaN
        avg_neg[i,j,k] = cn>0 ? sum_neg[i,j,k]/cn : NaN
    end

    # Also produce theta-aggregated time-averages, both unweighted and volume-weighted
    sum_pos_theta = sum(sum_pos; dims=(2,3))[:,1,1]
    sum_neg_theta = sum(sum_neg; dims=(2,3))[:,1,1]
    cnt_pos_theta = sum(cnt_pos; dims=(2,3))[:,1,1]
    cnt_neg_theta = sum(cnt_neg; dims=(2,3))[:,1,1]
    avg_pos_theta = [cnt_pos_theta[i]>0 ? sum_pos_theta[i]/cnt_pos_theta[i] : NaN for i in 1:nθ]
    avg_neg_theta = [cnt_neg_theta[i]>0 ? sum_neg_theta[i]/cnt_neg_theta[i] : NaN for i in 1:nθ]

    # Volume-weighted theta profiles using Gauss weights (wθ) and radial Δr and r^2 factors
    wθ = try
        Vector{Float64}(SHTnsKit.get_gauss_weights(sht))
    catch
        sin.(theta)
    end
    Δφ = nφ > 1 ? (phi[2]-phi[1]) : 2π
    Δr = similar(rvals)
    for k in 1:nr
        if k == 1
            Δr[k] = (rvals[min(2,nr)] - rvals[1])
        elseif k == nr
            Δr[k] = (rvals[nr] - rvals[nr-1])
        else
            Δr[k] = 0.5*(rvals[k+1] - rvals[k-1])
        end
    end
    num_pos_wθ = zeros(Float64, nθ); den_pos_wθ = zeros(Float64, nθ)
    num_neg_wθ = zeros(Float64, nθ); den_neg_wθ = zeros(Float64, nθ)
    @inbounds for i in 1:nθ, j in 1:nφ, k in 1:nr
        w = wθ[i] * Δφ * (rvals[k]^2) * Δr[k]
        if cnt_pos[i,j,k] > 0
            num_pos_wθ[i] += sum_pos[i,j,k] * w
            den_pos_wθ[i] += cnt_pos[i,j,k] * w
        end
        if cnt_neg[i,j,k] > 0
            num_neg_wθ[i] += sum_neg[i,j,k] * w
            den_neg_wθ[i] += cnt_neg[i,j,k] * w
        end
    end
    avg_pos_theta_weighted = [den_pos_wθ[i]>0 ? num_pos_wθ[i]/den_pos_wθ[i] : NaN for i in 1:nθ]
    avg_neg_theta_weighted = [den_neg_wθ[i]>0 ? num_neg_wθ[i]/den_neg_wθ[i] : NaN for i in 1:nθ]

    jldsave(opts.out;
        theta=theta, phi=phi, r=rvals,
        avg_pos=avg_pos, avg_neg=avg_neg,
        count_pos=cnt_pos, count_neg=cnt_neg,
        avg_pos_theta=avg_pos_theta, avg_neg_theta=avg_neg_theta,
        count_pos_theta=cnt_pos_theta, count_neg_theta=cnt_neg_theta,
        avg_pos_theta_weighted=avg_pos_theta_weighted, avg_neg_theta_weighted=avg_neg_theta_weighted,
        description="Time-averaged z-helicity (u_z*ω_z) separated by sign")
    println("Saved ", opts.out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
