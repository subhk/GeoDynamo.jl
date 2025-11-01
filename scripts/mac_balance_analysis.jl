#!/usr/bin/env julia

"""
MAC Balance Analysis for GeoDynamo Simulations

This script computes the Magnetic, Archimedes, and Coriolis (MAC) force balance 
from GeoDynamo.jl simulation data. The analysis can be performed on:
1. Constant radius surfaces (r = constant)
2. Phi-z meridional planes (φ = constant)

The MAC balance equation in the rotating frame is:
∂u/∂t + (u·∇)u = -2Ω×u - ∇p/ρ + Ra*T*r̂ + (∇×B)×B/μ₀ρ + ν∇²u

Where the dominant force balance terms are:
- Coriolis force: -2Ω×u  
- Pressure gradient: -∇p/ρ
- Archimedes force: Ra*T*r̂ (thermal buoyancy)
- Magnetic force: (∇×B)×B/μ₀ρ (Lorentz force)
- Inertial force: (u·∇)u (nonlinear advection)

The script computes each term and analyzes their relative magnitudes and balance.

Usage:
    julia mac_balance_analysis.jl [options]

Options:
    --data_file <path>          Path to simulation data file (HDF5/NetCDF)
    --output_dir <path>         Output directory for results
    --analysis_type <type>      Analysis type: "constant_radius", "phi_z_plane"
    --radius <r>                Radial location for constant radius analysis
    --phi_degree <deg>          Azimuthal angle in degrees for phi-z plane
    --time_index <int>          Time index to analyze (default: last available)
    --save_format <format>      Save format: "hdf5", "netcdf", "jld2"
    --rayleigh <Ra>            Rayleigh number (default: auto-detect or 1e6)
    --magnetic_prandtl <Pm>    Magnetic Prandtl number (default: 1.0)
    --prandtl <Pr>             Prandtl number (default: 1.0)
    --ekman <Ek>               Ekman number (default: 1e-3)
    --plot_format <fmt>        Plot format: "png", "pdf", "svg" (default: png)

Examples:
    # MAC balance at outer core boundary
    julia mac_balance_analysis.jl --data_file=simulation.h5 \
          --analysis_type=constant_radius --radius=0.95

    # MAC balance in meridional plane
    julia mac_balance_analysis.jl --data_file=simulation.h5 \
          --analysis_type=phi_z_plane --phi_degree=0

    # Custom parameters
    julia mac_balance_analysis.jl --data_file=simulation.h5 \
          --analysis_type=constant_radius --radius=0.8 \
          --rayleigh=1e7 --ekman=1e-4 --magnetic_prandtl=0.5
"""

using HDF5
using JLD2
using NetCDF
using Statistics
using LinearAlgebra
using ArgParse
using Dates
using Printf

# Optional plotting packages
try
    using Plots
    using PlotlyJS
    plotlyjs()
    global HAS_PLOTS = true
catch
    global HAS_PLOTS = false
    @warn "Plots.jl not available. Visualization disabled."
end

# Physical constants and simulation parameters
struct SimulationParams
    Ra::Float64        # Rayleigh number
    Pr::Float64        # Prandtl number  
    Pm::Float64        # Magnetic Prandtl number
    Ek::Float64        # Ekman number
    μ₀ρ::Float64       # Magnetic permeability × density
    rotation_rate::Float64  # Ω (typically = 1 in non-dimensional units)
end

struct AnalysisConfig
    analysis_type::Symbol  # :constant_radius or :phi_z_plane
    radius::Float64        # For constant radius analysis
    phi_deg::Float64       # For phi-z plane analysis (degrees)
    time_index::Int        # Time slice to analyze
end

struct MACBalanceResult
    # Grid coordinates
    coords::Dict{String, Any}
    
    # Velocity and magnetic field components
    velocity::Dict{String, Array}
    magnetic::Dict{String, Array}
    
    # MAC balance terms
    coriolis_force::Dict{String, Array}      # -2Ω×u
    pressure_gradient::Dict{String, Array}   # -∇p/ρ
    archimedes_force::Dict{String, Array}    # Ra*T*r̂
    magnetic_force::Dict{String, Array}      # (∇×B)×B/μ₀ρ
    inertial_force::Dict{String, Array}      # (u·∇)u
    viscous_force::Dict{String, Array}       # ν∇²u
    
    # Balance diagnostics
    total_force::Dict{String, Array}         # Sum of all forces
    force_magnitudes::Dict{String, Float64}  # RMS magnitudes
    balance_residual::Dict{String, Array}    # Residual of force balance
    
    # Pressure reconstruction (if applicable)
    pressure_reconstructed::Union{Array, Nothing}  # Reconstructed pressure field
    pressure_divergence::Union{Array, Nothing}     # ∇·F for pressure Poisson equation
    
    # Configuration
    params::SimulationParams
    config::AnalysisConfig
    metadata::Dict{String, Any}
end

function parse_commandline()
    s = ArgParseSettings(
        description = "MAC balance analysis for geodynamo simulations",
        version = "1.0.0"
    )

    @add_arg_table! s begin
        "--data_file"
            help = "Path to simulation data file"
            required = true
        "--output_dir"
            help = "Output directory for results"
            default = "./mac_balance_results"
        "--analysis_type"
            help = "Analysis type: constant_radius, phi_z_plane"
            default = "constant_radius"
        "--radius"
            help = "Radial location for constant radius analysis"
            arg_type = Float64
            default = 0.8
        "--phi_degree"
            help = "Azimuthal angle in degrees for phi-z plane"
            arg_type = Float64
            default = 0.0
        "--time_index"
            help = "Time index to analyze (-1 for last)"
            arg_type = Int
            default = -1
        "--save_format"
            help = "Save format: hdf5, netcdf, jld2"
            default = "hdf5"
        "--rayleigh"
            help = "Rayleigh number"
            arg_type = Float64
            default = 1e6
        "--magnetic_prandtl"
            help = "Magnetic Prandtl number"
            arg_type = Float64
            default = 1.0
        "--prandtl"
            help = "Prandtl number"
            arg_type = Float64
            default = 1.0
        "--ekman"
            help = "Ekman number"
            arg_type = Float64
            default = 1e-3
        "--plot_format"
            help = "Plot format: png, pdf, svg"
            default = "png"
    end

    return parse_args(s)
end

function load_geodynamo_fields(filename::String, time_idx::Int)
    """Load velocity, magnetic field, and temperature from geodynamo data"""
    
    println("Loading geodynamo fields from: $filename")
    
    if endswith(filename, ".h5") || endswith(filename, ".hdf5")
        data = h5open(filename, "r") do file
            # Load coordinate grids
            r_coords = read(file, "r")
            theta_coords = read(file, "theta") 
            phi_coords = read(file, "phi")
            times = read(file, "time")
            
            # Determine time index
            if time_idx == -1
                t_idx = length(times)
            else
                t_idx = min(time_idx, length(times))
            end
            
            println("Using time index $t_idx (t = $(times[t_idx]))")
            
            # Load velocity field components (r, θ, φ)
            u_r = read(file, "velocity_r")[:, :, :, t_idx]
            u_theta = read(file, "velocity_theta")[:, :, :, t_idx]  
            u_phi = read(file, "velocity_phi")[:, :, :, t_idx]
            
            # Load magnetic field components
            B_r = read(file, "magnetic_r")[:, :, :, t_idx]
            B_theta = read(file, "magnetic_theta")[:, :, :, t_idx]
            B_phi = read(file, "magnetic_phi")[:, :, :, t_idx]
            
            # Load temperature
            temperature = read(file, "temperature")[:, :, :, t_idx]
            
            # Try to load pressure if available
            pressure = nothing
            try
                pressure = read(file, "pressure")[:, :, :, t_idx]
            catch
                @warn "Pressure field not found - will compute from other terms"
            end
            
            Dict("r" => r_coords, "theta" => theta_coords, "phi" => phi_coords,
                 "time" => times[t_idx],
                 "u_r" => u_r, "u_theta" => u_theta, "u_phi" => u_phi,
                 "B_r" => B_r, "B_theta" => B_theta, "B_phi" => B_phi,
                 "temperature" => temperature, "pressure" => pressure)
        end
        
    elseif endswith(filename, ".nc")
        # Load coordinate grids
        r_coords = ncread(filename, "r")
        theta_coords = ncread(filename, "theta")
        phi_coords = ncread(filename, "phi") 
        times = ncread(filename, "time")
        
        # Determine time index
        if time_idx == -1
            t_idx = length(times)
        else
            t_idx = min(time_idx, length(times))
        end
        
        println("Using time index $t_idx (t = $(times[t_idx]))")
        
        # Load fields at specified time
        u_r = ncread(filename, "velocity_r", start=[1,1,1,t_idx], count=[-1,-1,-1,1])[:,:,:,1]
        u_theta = ncread(filename, "velocity_theta", start=[1,1,1,t_idx], count=[-1,-1,-1,1])[:,:,:,1]
        u_phi = ncread(filename, "velocity_phi", start=[1,1,1,t_idx], count=[-1,-1,-1,1])[:,:,:,1]
        
        B_r = ncread(filename, "magnetic_r", start=[1,1,1,t_idx], count=[-1,-1,-1,1])[:,:,:,1]
        B_theta = ncread(filename, "magnetic_theta", start=[1,1,1,t_idx], count=[-1,-1,-1,1])[:,:,:,1]
        B_phi = ncread(filename, "magnetic_phi", start=[1,1,1,t_idx], count=[-1,-1,-1,1])[:,:,:,1]
        
        temperature = ncread(filename, "temperature", start=[1,1,1,t_idx], count=[-1,-1,-1,1])[:,:,:,1]
        
        pressure = nothing
        try
            pressure = ncread(filename, "pressure", start=[1,1,1,t_idx], count=[-1,-1,-1,1])[:,:,:,1]
        catch
            @warn "Pressure field not found - will compute from other terms"
        end
        
        data = Dict("r" => r_coords, "theta" => theta_coords, "phi" => phi_coords,
                   "time" => times[t_idx],
                   "u_r" => u_r, "u_theta" => u_theta, "u_phi" => u_phi,
                   "B_r" => B_r, "B_theta" => B_theta, "B_phi" => B_phi,
                   "temperature" => temperature, "pressure" => pressure)
    else
        error("Unsupported file format. Use .h5, .hdf5, or .nc files")
    end
    
    println("Fields loaded successfully:")
    println("  Grid size: $(size(data["u_r"]))")
    println("  Time: $(data["time"])")
    println("  Max velocity: $(round(maximum(sqrt.(data["u_r"].^2 .+ data["u_theta"].^2 .+ data["u_phi"].^2)), digits=4))")
    println("  Max magnetic field: $(round(maximum(sqrt.(data["B_r"].^2 .+ data["B_theta"].^2 .+ data["B_phi"].^2)), digits=4))")
    
    return data
end

function extract_constant_radius_slice(data::Dict, target_radius::Float64)
    """Extract fields at constant radius surface"""
    
    r_coords = data["r"]
    theta_coords = data["theta"]
    phi_coords = data["phi"]
    
    # Find closest radius index
    r_idx = argmin(abs.(r_coords .- target_radius))
    actual_radius = r_coords[r_idx]
    
    println("Extracting constant radius slice:")
    println("  Target radius: $target_radius")
    println("  Actual radius: $actual_radius (index: $r_idx)")
    
    # Extract 2D slices [θ, φ]
    coords = Dict(
        "radius" => actual_radius,
        "theta" => theta_coords,
        "phi" => phi_coords
    )
    
    fields = Dict(
        "u_r" => data["u_r"][r_idx, :, :],
        "u_theta" => data["u_theta"][r_idx, :, :],
        "u_phi" => data["u_phi"][r_idx, :, :],
        "B_r" => data["B_r"][r_idx, :, :],
        "B_theta" => data["B_theta"][r_idx, :, :], 
        "B_phi" => data["B_phi"][r_idx, :, :],
        "temperature" => data["temperature"][r_idx, :, :],
        "pressure" => data["pressure"] === nothing ? nothing : data["pressure"][r_idx, :, :]
    )
    
    return coords, fields
end

function extract_phi_z_slice(data::Dict, phi_deg::Float64)
    """Extract fields in phi-z meridional plane"""
    
    r_coords = data["r"]
    theta_coords = data["theta"]
    phi_coords = data["phi"]
    
    # Convert phi to radians and find closest index
    phi_rad = deg2rad(phi_deg)
    phi_idx = argmin(abs.(phi_coords .- phi_rad))
    actual_phi = phi_coords[phi_idx]
    
    println("Extracting phi-z meridional plane:")
    println("  Target phi: $(phi_deg)° ($(round(phi_rad, digits=3)) rad)")
    println("  Actual phi: $(round(rad2deg(actual_phi), digits=1))° (index: $phi_idx)")
    
    # Extract 2D slices [r, θ] 
    coords = Dict(
        "r" => r_coords,
        "theta" => theta_coords,
        "phi" => actual_phi,
        "z" => r_coords' .* cos.(theta_coords)  # Cylindrical z-coordinate
    )
    
    fields = Dict(
        "u_r" => data["u_r"][:, :, phi_idx],
        "u_theta" => data["u_theta"][:, :, phi_idx],
        "u_phi" => data["u_phi"][:, :, phi_idx],
        "B_r" => data["B_r"][:, :, phi_idx],
        "B_theta" => data["B_theta"][:, :, phi_idx],
        "B_phi" => data["B_phi"][:, :, phi_idx],
        "temperature" => data["temperature"][:, :, phi_idx],
        "pressure" => data["pressure"] === nothing ? nothing : data["pressure"][:, :, phi_idx]
    )
    
    return coords, fields
end

function compute_derivatives_spherical(field::Array{T,2}, r, theta, phi=nothing; 
                                     analysis_type=:constant_radius) where T
    """Compute derivatives in spherical coordinates"""
    
    nr, ntheta = size(field)
    
    # Initialize derivative arrays
    ∂f_∂r = zeros(T, nr, ntheta)
    ∂f_∂θ = zeros(T, nr, ntheta)
    ∂f_∂φ = analysis_type == :constant_radius ? zeros(T, nr, ntheta) : nothing
    
    if analysis_type == :constant_radius
        # At constant radius: field[θ, φ]
        # ∂f/∂θ using centered differences
        for j in 2:ntheta-1
            for i in 1:nr
                dtheta = theta[j+1] - theta[j-1]
                ∂f_∂θ[i, j] = (field[i, j+1] - field[i, j-1]) / dtheta
            end
        end
        
        # Boundary conditions for θ derivatives
        ∂f_∂θ[:, 1] = ∂f_∂θ[:, 2]      # Copy from interior
        ∂f_∂θ[:, end] = ∂f_∂θ[:, end-1]
        
        # ∂f/∂φ using centered differences (φ is periodic)
        for i in 1:nr
            for j in 1:ntheta
                if i == 1
                    dphi = phi[2] - phi[1]
                    ∂f_∂φ[i, j] = (field[i, 2] - field[i, end]) / (2*dphi)  # Periodic
                elseif i == nr
                    dphi = phi[end] - phi[end-1]
                    ∂f_∂φ[i, j] = (field[i, 1] - field[i, end-1]) / (2*dphi)  # Periodic
                else
                    dphi = phi[i+1] - phi[i-1]
                    ∂f_∂φ[i, j] = (field[i+1, j] - field[i-1, j]) / dphi
                end
            end
        end
        
    else  # phi_z_plane
        # In meridional plane: field[r, θ]
        # ∂f/∂r using centered differences
        for i in 2:nr-1
            for j in 1:ntheta
                dr = r[i+1] - r[i-1]
                ∂f_∂r[i, j] = (field[i+1, j] - field[i-1, j]) / dr
            end
        end
        
        # Boundary conditions for r derivatives
        ∂f_∂r[1, :] = ∂f_∂r[2, :]        # Copy from interior
        ∂f_∂r[end, :] = ∂f_∂r[end-1, :]
        
        # ∂f/∂θ using centered differences
        for j in 2:ntheta-1
            for i in 1:nr
                dtheta = theta[j+1] - theta[j-1]
                ∂f_∂θ[i, j] = (field[i, j+1] - field[i, j-1]) / dtheta
            end
        end
        
        # Boundary conditions for θ derivatives  
        ∂f_∂θ[:, 1] = ∂f_∂θ[:, 2]
        ∂f_∂θ[:, end] = ∂f_∂θ[:, end-1]
    end
    
    return ∂f_∂r, ∂f_∂θ, ∂f_∂φ
end

function compute_curl_spherical(Br, Btheta, Bphi, coords, analysis_type)
    """Compute curl in spherical coordinates: ∇×B"""
    
    if analysis_type == :constant_radius
        r = coords["radius"]  # Single radius value
        theta = coords["theta"]
        phi = coords["phi"]
        
        # At constant radius, curl components are:
        # (∇×B)_r = (1/r sin θ)[∂(B_φ sin θ)/∂θ - ∂B_θ/∂φ]
        # (∇×B)_θ = (1/r)[∂B_r/∂φ/(sin θ) - ∂(rB_φ)/∂r]/r = -∂B_φ/∂r (since r is constant)
        # (∇×B)_φ = (1/r)[∂(rB_θ)/∂r - ∂B_r/∂θ]/r = ∂B_θ/∂r (since r is constant)
        
        _, ∂Btheta_∂θ, ∂Btheta_∂φ = compute_derivatives_spherical(Btheta, [r], theta, phi; analysis_type=analysis_type)
        _, ∂Bphi_∂θ, ∂Bphi_∂φ = compute_derivatives_spherical(Bphi, [r], theta, phi; analysis_type=analysis_type)
        _, ∂Br_∂θ, ∂Br_∂φ = compute_derivatives_spherical(Br, [r], theta, phi; analysis_type=analysis_type)
        
        # Curl components at constant radius
        curl_r = zeros(size(Br))
        curl_theta = zeros(size(Br))
        curl_phi = zeros(size(Br))
        
        for j in 1:length(theta), i in 1:length(phi)
            sin_th = sin(theta[j])
            
            # Radial component
            curl_r[i, j] = (1/(r*sin_th)) * (∂Bphi_∂θ[i, j] * sin_th + Bphi[i, j] * cos(theta[j]) - ∂Btheta_∂φ[i, j])
            
            # Note: θ and φ components require ∂/∂r which is zero at constant r
            # These represent the curl's projection onto θ,φ directions
            curl_theta[i, j] = ∂Br_∂φ[i, j] / (r * sin_th)
            curl_phi[i, j] = -∂Br_∂θ[i, j] / r
        end
        
    else  # phi_z_plane
        r_coords = coords["r"]
        theta = coords["theta"]
        
        # In meridional plane, compute available curl components
        ∂Br_∂r, ∂Br_∂θ, _ = compute_derivatives_spherical(Br, r_coords, theta; analysis_type=analysis_type)
        ∂Btheta_∂r, ∂Btheta_∂θ, _ = compute_derivatives_spherical(Btheta, r_coords, theta; analysis_type=analysis_type)
        ∂Bphi_∂r, ∂Bphi_∂θ, _ = compute_derivatives_spherical(Bphi, r_coords, theta; analysis_type=analysis_type)
        
        curl_r = zeros(size(Br))
        curl_theta = zeros(size(Br))
        curl_phi = zeros(size(Br))
        
        for j in 1:length(theta), i in 1:length(r_coords)
            r = r_coords[i]
            
            # Only φ-component can be fully computed in meridional plane
            curl_phi[i, j] = (1/r) * (∂Btheta_∂r[i, j] - ∂Br_∂θ[i, j]/r + Btheta[i, j]/r)
            
            # Other components require ∂/∂φ which is unavailable
            curl_r[i, j] = 0.0  # Approximation
            curl_theta[i, j] = 0.0  # Approximation
        end
    end
    
    return curl_r, curl_theta, curl_phi
end

function reconstruct_pressure_field(coords, fields, coriolis_force, archimedes_force, 
                                   magnetic_force, inertial_force, viscous_force, 
                                   params::SimulationParams, analysis_type)
    """Reconstruct pressure field from momentum balance using Poisson equation approach
    
    This solves: ∇²p = ρ∇·[2Ω×u + Ra*T*r̂ + (∇×B)×B/μ₀ρ + (u·∇)u - ν∇²u]
    """
    
    println("Reconstructing pressure field from momentum balance...")
    
    # Compute divergence of force terms (RHS of Poisson equation)
    # ∇²p = ρ∇·F_total where F_total is sum of all non-pressure forces
    
    total_force_r = coriolis_force["r"] + archimedes_force["r"] + magnetic_force["r"] + 
                   inertial_force["r"] + viscous_force["r"]
    total_force_theta = coriolis_force["theta"] + archimedes_force["theta"] + magnetic_force["theta"] + 
                       inertial_force["theta"] + viscous_force["theta"]  
    total_force_phi = coriolis_force["phi"] + archimedes_force["phi"] + magnetic_force["phi"] + 
                     inertial_force["phi"] + viscous_force["phi"]
    
    if analysis_type == :constant_radius
        r = coords["radius"]
        theta = coords["theta"]
        phi = coords["phi"]
        
        # At constant radius, divergence is:
        # ∇·F = (1/r²sin θ)[∂(F_θ sin θ)/∂θ + ∂F_φ/∂φ]
        
        # Compute derivatives
        _, ∂Ftheta_∂θ, ∂Ftheta_∂φ = compute_derivatives_spherical(total_force_theta, [r], theta, phi; analysis_type=analysis_type)
        _, ∂Fphi_∂θ, ∂Fphi_∂φ = compute_derivatives_spherical(total_force_phi, [r], theta, phi; analysis_type=analysis_type)
        
        # Divergence calculation
        div_F = zeros(size(total_force_r))
        for j in 1:length(theta), i in 1:length(phi)
            sin_th = sin(theta[j])
            cos_th = cos(theta[j])
            
            # ∇·F = (1/r²sin θ)[∂(F_θ sin θ)/∂θ + ∂F_φ/∂φ]
            term1 = (∂Ftheta_∂θ[i, j] * sin_th + total_force_theta[i, j] * cos_th)
            term2 = ∂Fphi_∂φ[i, j]
            
            div_F[i, j] = (term1 + term2) / (r^2 * sin_th + 1e-10)
        end
        
        # Simple integration to get pressure (this is approximate)
        # In practice, you'd solve the Poisson equation ∇²p = ρ*div_F
        pressure_reconstructed = cumulative_integrate_2d(div_F, theta, phi)
        
    else  # phi_z_plane  
        r_coords = coords["r"]
        theta = coords["theta"]
        
        # In meridional plane, divergence is:
        # ∇·F = (1/r²)[∂(r²F_r)/∂r] + (1/r sin θ)[∂(F_θ sin θ)/∂θ]
        
        ∂Fr_∂r, ∂Fr_∂θ, _ = compute_derivatives_spherical(total_force_r, r_coords, theta; analysis_type=analysis_type)
        ∂Ftheta_∂r, ∂Ftheta_∂θ, _ = compute_derivatives_spherical(total_force_theta, r_coords, theta; analysis_type=analysis_type)
        
        div_F = zeros(size(total_force_r))
        for j in 1:length(theta), i in 1:length(r_coords)
            r = r_coords[i]
            sin_th = sin(theta[j])
            cos_th = cos(theta[j])
            
            # Radial term: (1/r²)[∂(r²F_r)/∂r]
            term1 = (2*r*total_force_r[i, j] + r^2*∂Fr_∂r[i, j]) / (r^2 + 1e-10)
            
            # Angular term: (1/r sin θ)[∂(F_θ sin θ)/∂θ]  
            term2 = (∂Ftheta_∂θ[i, j] * sin_th + total_force_theta[i, j] * cos_th) / (r * sin_th + 1e-10)
            
            div_F[i, j] = term1 + term2
        end
        
        pressure_reconstructed = cumulative_integrate_2d_rtheta(div_F, r_coords, theta)
    end
    
    println("  Pressure field reconstructed from momentum balance")
    println("  RHS (∇·F) RMS: $(round(sqrt(mean(div_F.^2)), digits=6))")
    println("  Reconstructed pressure RMS: $(round(sqrt(mean(pressure_reconstructed.^2)), digits=6))")
    
    return pressure_reconstructed, div_F
end

function cumulative_integrate_2d(field, coord1, coord2)
    """Simple cumulative integration for pressure reconstruction"""
    integrated = similar(field)
    
    # Integrate along first dimension
    integrated[1, :] = field[1, :] * (coord1[2] - coord1[1])
    for i in 2:size(field, 1)
        dc = coord1[i] - coord1[i-1]
        integrated[i, :] = integrated[i-1, :] + field[i, :] * dc
    end
    
    # Integrate along second dimension
    for i in 1:size(field, 1)
        for j in 2:size(field, 2)
            dc = coord2[j] - coord2[j-1]  
            integrated[i, j] += integrated[i, j-1] + field[i, j] * dc
        end
    end
    
    return integrated
end

function cumulative_integrate_2d_rtheta(field, r_coords, theta_coords)
    """Cumulative integration in r-θ coordinates with proper weighting"""
    integrated = similar(field)
    
    # Integrate in r direction first
    integrated[1, :] = field[1, :] * (r_coords[2] - r_coords[1])
    for i in 2:length(r_coords)
        dr = r_coords[i] - r_coords[i-1]
        integrated[i, :] = integrated[i-1, :] + field[i, :] * dr
    end
    
    # Integrate in θ direction
    for i in 1:length(r_coords)
        for j in 2:length(theta_coords)
            dtheta = theta_coords[j] - theta_coords[j-1]
            integrated[i, j] += integrated[i, j-1] + field[i, j] * r_coords[i] * dtheta
        end
    end
    
    return integrated
end

function compute_mac_forces(coords, fields, params::SimulationParams, analysis_type)
    """Compute all MAC balance force terms"""
    
    println("Computing MAC balance forces...")
    
    u_r, u_theta, u_phi = fields["u_r"], fields["u_theta"], fields["u_phi"]
    B_r, B_theta, B_phi = fields["B_r"], fields["B_theta"], fields["B_phi"]
    temperature = fields["temperature"]
    
    # 1. CORIOLIS FORCE: -2Ω×u
    println("  Computing Coriolis force...")
    Ω = params.rotation_rate
    
    coriolis_r = -2 * Ω * u_phi  # -2Ω × u in spherical coords
    coriolis_theta = zeros(size(u_theta))  # Ω is in z-direction
    coriolis_phi = 2 * Ω * u_r
    
    coriolis_force = Dict(
        "r" => coriolis_r,
        "theta" => coriolis_theta, 
        "phi" => coriolis_phi
    )
    
    # 2. ARCHIMEDES FORCE: Ra*T*r̂ 
    println("  Computing Archimedes (buoyancy) force...")
    archimedes_r = params.Ra * temperature
    archimedes_theta = zeros(size(temperature))
    archimedes_phi = zeros(size(temperature))
    
    archimedes_force = Dict(
        "r" => archimedes_r,
        "theta" => archimedes_theta,
        "phi" => archimedes_phi
    )
    
    # 3. MAGNETIC FORCE: (∇×B)×B/μ₀ρ
    println("  Computing magnetic (Lorentz) force...")
    curl_B_r, curl_B_theta, curl_B_phi = compute_curl_spherical(B_r, B_theta, B_phi, coords, analysis_type)
    
    # (∇×B)×B = curl_B × B
    magnetic_r = (curl_B_theta * B_phi - curl_B_phi * B_theta) / params.μ₀ρ
    magnetic_theta = (curl_B_phi * B_r - curl_B_r * B_phi) / params.μ₀ρ
    magnetic_phi = (curl_B_r * B_theta - curl_B_theta * B_r) / params.μ₀ρ
    
    magnetic_force = Dict(
        "r" => magnetic_r,
        "theta" => magnetic_theta,
        "phi" => magnetic_phi
    )
    
    # 4. PRESSURE GRADIENT: -∇p/ρ (computed from momentum balance)
    println("  Computing pressure gradient from momentum balance...")
    
    # The momentum equation is: ∂u/∂t + (u·∇)u = -2Ω×u - ∇p/ρ + Ra*T*r̂ + (∇×B)×B/μ₀ρ + ν∇²u
    # Rearranging: ∇p/ρ = -∂u/∂t - (u·∇)u + 2Ω×u + Ra*T*r̂ + (∇×B)×B/μ₀ρ + ν∇²u
    # For steady state: ∇p/ρ ≈ -(u·∇)u + 2Ω×u + Ra*T*r̂ + (∇×B)×B/μ₀ρ + ν∇²u
    
    if fields["pressure"] !== nothing
        pressure = fields["pressure"]
        
        if analysis_type == :constant_radius
            r = coords["radius"]
            theta = coords["theta"]
            phi = coords["phi"]
            
            _, ∂p_∂θ, ∂p_∂φ = compute_derivatives_spherical(pressure, [r], theta, phi; analysis_type=analysis_type)
            
            pressure_r = zeros(size(pressure))  # ∂p/∂r = 0 at constant r
            pressure_theta = -∂p_∂θ / r
            pressure_phi = -∂p_∂φ / (r * sin.(theta'))
            
        else  # phi_z_plane
            r_coords = coords["r"]
            theta = coords["theta"]
            
            ∂p_∂r, ∂p_∂θ, _ = compute_derivatives_spherical(pressure, r_coords, theta; analysis_type=analysis_type)
            
            pressure_r = -∂p_∂r
            pressure_theta = -∂p_∂θ ./ r_coords'
            pressure_phi = zeros(size(pressure))  # ∂p/∂φ = 0 in meridional plane
        end
        
        println("    Using explicit pressure field from data")
        
    else
        # Compute pressure gradient from momentum balance
        println("    Computing pressure gradient from momentum balance (geostrophic + ageostrophic)")
        
        # Geostrophic balance: ∇p/ρ ≈ 2Ω×u + Ra*T*r̂ + (∇×B)×B/μ₀ρ
        # This is the dominant balance in rapidly rotating systems
        
        geostrophic_pressure_r = coriolis_r + archimedes_r + magnetic_r
        geostrophic_pressure_theta = coriolis_theta + archimedes_theta + magnetic_theta  
        geostrophic_pressure_phi = coriolis_phi + archimedes_phi + magnetic_phi
        
        # For more complete balance, add inertial and viscous terms (ageostrophic)
        # ∇p/ρ = 2Ω×u + Ra*T*r̂ + (∇×B)×B/μ₀ρ + (u·∇)u - ν∇²u
        
        pressure_r = geostrophic_pressure_r + inertial_force["r"] - viscous_force["r"]
        pressure_theta = geostrophic_pressure_theta + inertial_force["theta"] - viscous_force["theta"]
        pressure_phi = geostrophic_pressure_phi + inertial_force["phi"] - viscous_force["phi"]
        
        println("    Pressure gradient computed from force balance")
        println("    Geostrophic component RMS: $(round(sqrt(mean(geostrophic_pressure_r.^2 + geostrophic_pressure_theta.^2 + geostrophic_pressure_phi.^2)), digits=6))")
    end
    
    pressure_gradient = Dict(
        "r" => pressure_r,
        "theta" => pressure_theta,
        "phi" => pressure_phi
    )
    
    # 5. INERTIAL FORCE: (u·∇)u (simplified estimate)
    println("  Computing inertial (advection) force...")
    # Simplified calculation - magnitude estimate
    u_mag = sqrt.(u_r.^2 .+ u_theta.^2 .+ u_phi.^2)
    
    if analysis_type == :constant_radius
        r = coords["radius"]
        _, ∂ur_∂θ, ∂ur_∂φ = compute_derivatives_spherical(u_r, [r], coords["theta"], coords["phi"]; analysis_type=analysis_type)
        inertial_scale = u_mag / r  # Rough scaling
    else
        ∂ur_∂r, ∂ur_∂θ, _ = compute_derivatives_spherical(u_r, coords["r"], coords["theta"]; analysis_type=analysis_type)
        inertial_scale = u_mag .* maximum(abs.(∂ur_∂r))  # Rough scaling
    end
    
    inertial_force = Dict(
        "r" => inertial_scale .* u_r ./ (u_mag .+ 1e-10),
        "theta" => inertial_scale .* u_theta ./ (u_mag .+ 1e-10),
        "phi" => inertial_scale .* u_phi ./ (u_mag .+ 1e-10)
    )
    
    # 6. VISCOUS FORCE: ν∇²u (simplified estimate)
    println("  Computing viscous force...")
    ν = params.Pr * params.Ek  # Kinematic viscosity
    
    # Simplified Laplacian estimate
    viscous_scale = ν * u_mag * 10  # Rough estimate
    
    viscous_force = Dict(
        "r" => -viscous_scale .* u_r,
        "theta" => -viscous_scale .* u_theta,
        "phi" => -viscous_scale .* u_phi
    )
    
    return coriolis_force, pressure_gradient, archimedes_force, magnetic_force, inertial_force, viscous_force
end

function analyze_force_balance(coriolis_force, pressure_gradient, archimedes_force, 
                              magnetic_force, inertial_force, viscous_force)
    """Analyze the relative magnitudes and balance of MAC forces"""
    
    println("Analyzing force balance...")
    
    # Compute RMS magnitudes for each force component
    forces = Dict(
        "Coriolis" => coriolis_force,
        "Pressure" => pressure_gradient,
        "Archimedes" => archimedes_force,
        "Magnetic" => magnetic_force,
        "Inertial" => inertial_force,
        "Viscous" => viscous_force
    )
    
    force_magnitudes = Dict()
    
    for (name, force) in forces
        # Total magnitude: sqrt(Fr² + Fθ² + Fφ²)
        total_mag = sqrt.(force["r"].^2 .+ force["theta"].^2 .+ force["phi"].^2)
        rms_mag = sqrt(mean(total_mag.^2))
        force_magnitudes[name] = rms_mag
        
        println("  $name force RMS: $(round(rms_mag, digits=6))")
    end
    
    # Total force (should be small if balance is good)
    total_force = Dict(
        "r" => coriolis_force["r"] + pressure_gradient["r"] + archimedes_force["r"] + 
               magnetic_force["r"] + inertial_force["r"] + viscous_force["r"],
        "theta" => coriolis_force["theta"] + pressure_gradient["theta"] + archimedes_force["theta"] + 
                   magnetic_force["theta"] + inertial_force["theta"] + viscous_force["theta"],
        "phi" => coriolis_force["phi"] + pressure_gradient["phi"] + archimedes_force["phi"] + 
                 magnetic_force["phi"] + inertial_force["phi"] + viscous_force["phi"]
    )
    
    # Balance residual magnitude
    balance_residual = Dict(
        "r" => total_force["r"],
        "theta" => total_force["theta"],
        "phi" => total_force["phi"]
    )
    
    residual_mag = sqrt.(total_force["r"].^2 .+ total_force["theta"].^2 .+ total_force["phi"].^2)
    residual_rms = sqrt(mean(residual_mag.^2))
    
    println("  Force balance residual RMS: $(round(residual_rms, digits=6))")
    
    # Identify dominant forces
    sorted_forces = sort(collect(force_magnitudes), by=x->x[2], rev=true)
    println("  Force ranking (by RMS magnitude):")
    for (i, (name, mag)) in enumerate(sorted_forces)
        percentage = 100 * mag / sum(values(force_magnitudes))
        println("    $i. $name: $(round(percentage, digits=1))%")
    end
    
    return total_force, force_magnitudes, balance_residual
end

function create_visualization(result::MACBalanceResult, output_dir::String, plot_format::String)
    """Create visualization plots of MAC balance"""
    
    if !HAS_PLOTS
        @warn "Plotting not available - skipping visualization"
        return
    end
    
    println("Creating visualization plots...")
    
    mkpath(output_dir)
    analysis_type = result.config.analysis_type
    
    # Plot 1: Force magnitudes comparison
    force_names = collect(keys(result.force_magnitudes))
    force_values = collect(values(result.force_magnitudes))
    
    p1 = bar(force_names, force_values, 
             title="MAC Force Balance - RMS Magnitudes",
             xlabel="Force Type", ylabel="RMS Magnitude",
             yscale=:log10, rotation=45)
    
    savefig(p1, joinpath(output_dir, "mac_force_magnitudes.$plot_format"))
    
    # Plot 2: Spatial distribution of dominant forces
    coords = result.coords
    
    if analysis_type == :constant_radius
        # Constant radius: plot on θ-φ grid
        theta = coords["theta"]
        phi = coords["phi"]
        
        # Convert to Cartesian for plotting
        X = sin.(theta') .* cos.(phi)
        Y = sin.(theta') .* sin.(phi)
        
        # Plot Coriolis and magnetic forces
        p2 = contourf(X, Y, result.coriolis_force["r"], 
                     title="Coriolis Force (radial) at r=$(round(coords["radius"], digits=2))",
                     aspect_ratio=:equal)
        
        p3 = contourf(X, Y, result.magnetic_force["r"],
                     title="Magnetic Force (radial) at r=$(round(coords["radius"], digits=2))",  
                     aspect_ratio=:equal)
        
    else  # phi_z_plane
        # Phi-z plane: plot on r-θ grid  
        r = coords["r"]
        theta = coords["theta"]
        
        # Convert to cylindrical coordinates
        R = r' .* sin.(theta)  # Cylindrical radius
        Z = r' .* cos.(theta)  # Cylindrical height
        
        p2 = contourf(R, Z, result.coriolis_force["phi"],
                     title="Coriolis Force (φ) in φ=$(round(rad2deg(coords["phi"]), digits=1))° plane",
                     xlabel="Cylindrical radius", ylabel="Height")
        
        p3 = contourf(R, Z, result.magnetic_force["phi"],
                     title="Magnetic Force (φ) in φ=$(round(rad2deg(coords["phi"]), digits=1))° plane",
                     xlabel="Cylindrical radius", ylabel="Height")
    end
    
    savefig(p2, joinpath(output_dir, "mac_coriolis_distribution.$plot_format"))
    savefig(p3, joinpath(output_dir, "mac_magnetic_distribution.$plot_format"))
    
    # Plot 3: Force balance residual
    residual_mag = sqrt.(result.balance_residual["r"].^2 .+ 
                        result.balance_residual["theta"].^2 .+ 
                        result.balance_residual["phi"].^2)
    
    if analysis_type == :constant_radius
        p4 = contourf(X, Y, residual_mag,
                     title="Force Balance Residual at r=$(round(coords["radius"], digits=2))",
                     aspect_ratio=:equal)
    else
        p4 = contourf(R, Z, residual_mag,
                     title="Force Balance Residual in φ=$(round(rad2deg(coords["phi"]), digits=1))° plane",
                     xlabel="Cylindrical radius", ylabel="Height")
    end
    
    savefig(p4, joinpath(output_dir, "mac_balance_residual.$plot_format"))
    
    println("  Plots saved to: $output_dir")
end

function save_mac_results(result::MACBalanceResult, output_dir::String, 
                         save_format::String, data_file::String)
    """Save MAC balance analysis results"""
    
    mkpath(output_dir)
    
    # Create filename
    if result.config.analysis_type == :constant_radius
        location_str = "r$(replace(string(round(result.coords["radius"], digits=3)), "." => "p"))"
    else
        location_str = "phi$(Int(round(rad2deg(result.coords["phi"]))))"
    end
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    base_filename = "mac_balance_$(location_str)_$(timestamp)"
    
    println("\nSaving MAC balance results to: $output_dir")
    
    if save_format == "hdf5"
        filename = joinpath(output_dir, "$base_filename.h5")
        
        h5open(filename, "w") do file
            # Coordinates
            coords_group = create_group(file, "coordinates")
            for (key, val) in result.coords
                write(coords_group, key, val)
            end
            
            # Force components
            forces = ["coriolis_force", "pressure_gradient", "archimedes_force", 
                     "magnetic_force", "inertial_force", "viscous_force", "total_force", "balance_residual"]
            force_data = [result.coriolis_force, result.pressure_gradient, result.archimedes_force,
                         result.magnetic_force, result.inertial_force, result.viscous_force, 
                         result.total_force, result.balance_residual]
            
            for (force_name, force_dict) in zip(forces, force_data)
                force_group = create_group(file, force_name)
                for (component, data) in force_dict
                    write(force_group, component, data)
                end
            end
            
            # Velocity and magnetic fields
            vel_group = create_group(file, "velocity")
            for (component, data) in result.velocity
                write(vel_group, component, data)
            end
            
            mag_group = create_group(file, "magnetic")
            for (component, data) in result.magnetic
                write(mag_group, component, data)
            end
            
            # Force magnitudes
            mag_group = create_group(file, "force_magnitudes")
            for (name, magnitude) in result.force_magnitudes
                attrs(mag_group)[name] = magnitude
            end
            
            # Pressure reconstruction (if available)
            if result.pressure_reconstructed !== nothing
                write(file, "pressure_reconstructed", result.pressure_reconstructed)
                write(file, "pressure_divergence", result.pressure_divergence)
                attrs(file)["pressure_method"] = "reconstructed_from_momentum_balance"
            else
                attrs(file)["pressure_method"] = "explicit_from_data"
            end
            
            # Configuration and metadata
            attrs(file)["analysis_type"] = String(result.config.analysis_type)
            attrs(file)["radius"] = result.config.radius
            attrs(file)["phi_degree"] = result.config.phi_deg
            attrs(file)["time_index"] = result.config.time_index
            attrs(file)["rayleigh_number"] = result.params.Ra
            attrs(file)["prandtl_number"] = result.params.Pr
            attrs(file)["magnetic_prandtl_number"] = result.params.Pm
            attrs(file)["ekman_number"] = result.params.Ek
            attrs(file)["source_data_file"] = data_file
            attrs(file)["creation_time"] = string(now())
        end
        
    elseif save_format == "jld2"
        filename = joinpath(output_dir, "$base_filename.jld2")
        
        jldsave(filename;
               result=result,
               source_file=data_file,
               creation_time=now())
    else
        error("NetCDF format not yet implemented for MAC balance results. Use hdf5 or jld2.")
    end
    
    println("Results saved to: $filename")
    
    # Save summary report
    summary_file = joinpath(output_dir, "$(base_filename)_summary.txt")
    open(summary_file, "w") do f
        println(f, "MAC Balance Analysis Summary")
        println(f, "="^50)
        println(f, "Source data file: $data_file")
        println(f, "Analysis timestamp: $(now())")
        println(f, "")
        println(f, "Configuration:")
        println(f, "  Analysis type: $(result.config.analysis_type)")
        if result.config.analysis_type == :constant_radius
            println(f, "  Radius: $(result.coords["radius"])")
        else
            println(f, "  Phi angle: $(round(rad2deg(result.coords["phi"]), digits=1))°")
        end
        println(f, "  Time index: $(result.config.time_index)")
        println(f, "")
        println(f, "Physical Parameters:")
        println(f, "  Rayleigh number: $(result.params.Ra)")
        println(f, "  Prandtl number: $(result.params.Pr)")
        println(f, "  Magnetic Prandtl number: $(result.params.Pm)")
        println(f, "  Ekman number: $(result.params.Ek)")
        println(f, "")
        println(f, "Force Balance Results (RMS magnitudes):")
        sorted_forces = sort(collect(result.force_magnitudes), by=x->x[2], rev=true)
        for (name, magnitude) in sorted_forces
            percentage = 100 * magnitude / sum(values(result.force_magnitudes))
            println(f, "  $(rpad(name, 12)): $(round(magnitude, digits=6)) ($(round(percentage, digits=1))%)")
        end
        
        residual_mag = sqrt(mean(result.balance_residual["r"].^2 .+ 
                               result.balance_residual["theta"].^2 .+ 
                               result.balance_residual["phi"].^2))
        println(f, "")
        println(f, "Force balance residual: $(round(residual_mag, digits=6))")
        
        total_force_mag = maximum(values(result.force_magnitudes))
        balance_quality = 100 * (1 - residual_mag / total_force_mag)
        println(f, "Balance quality: $(round(balance_quality, digits=1))%")
    end
    
    return filename
end

function main()
    println("MAC Balance Analysis for GeoDynamo Simulations")
    println("="^60)
    
    # Parse command line arguments
    args = parse_commandline()
    
    # Create configuration objects
    params = SimulationParams(
        args["rayleigh"],
        args["prandtl"],
        args["magnetic_prandtl"],
        args["ekman"],
        1.0,  # μ₀ρ normalized
        1.0   # Ω normalized
    )
    
    config = AnalysisConfig(
        Symbol(args["analysis_type"]),
        args["radius"],
        args["phi_degree"],
        args["time_index"]
    )
    
    # Load simulation data
    data = load_geodynamo_fields(args["data_file"], config.time_index)
    
    # Extract analysis slice
    if config.analysis_type == :constant_radius
        coords, fields = extract_constant_radius_slice(data, config.radius)
    else  # phi_z_plane
        coords, fields = extract_phi_z_slice(data, config.phi_deg)
    end
    
    # Compute MAC forces
    coriolis_force, pressure_gradient, archimedes_force, magnetic_force, 
    inertial_force, viscous_force = compute_mac_forces(coords, fields, params, config.analysis_type)
    
    # If pressure wasn't available, reconstruct it from momentum balance
    pressure_reconstructed = nothing
    div_F = nothing
    if fields["pressure"] === nothing
        pressure_reconstructed, div_F = reconstruct_pressure_field(
            coords, fields, coriolis_force, archimedes_force, magnetic_force,
            inertial_force, viscous_force, params, config.analysis_type)
    end
    
    # Analyze force balance
    total_force, force_magnitudes, balance_residual = analyze_force_balance(
        coriolis_force, pressure_gradient, archimedes_force, 
        magnetic_force, inertial_force, viscous_force)
    
    # Create result structure
    result = MACBalanceResult(
        coords, 
        Dict("u_r" => fields["u_r"], "u_theta" => fields["u_theta"], "u_phi" => fields["u_phi"]),
        Dict("B_r" => fields["B_r"], "B_theta" => fields["B_theta"], "B_phi" => fields["B_phi"]),
        coriolis_force, pressure_gradient, archimedes_force, magnetic_force, 
        inertial_force, viscous_force,
        total_force, force_magnitudes, balance_residual,
        pressure_reconstructed, div_F,
        params, config,
        Dict("analysis_time" => now(), "source_file" => args["data_file"])
    )
    
    # Create visualization
    create_visualization(result, args["output_dir"], args["plot_format"])
    
    # Save results
    output_file = save_mac_results(result, args["output_dir"], args["save_format"], args["data_file"])
    
    println("\n" * "="^60)
    println("MAC Balance Analysis completed successfully!")
    println("Results saved to: $(dirname(output_file))")
    println("="^60)
    
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end