#!/usr/bin/env julia

using Geodynamo
using MPI
using Random
using LinearAlgebra

function main()
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = Geodynamo.get_comm()
    rank = Geodynamo.get_rank()

    # Small, fast test sizes
    lmax = 8
    mmax = 8
    nlat = max(lmax + 2, 16)
    nlon = max(2lmax + 1, 32)
    nr   = 8

    # Configure SHTnsKit and domain
    cfg = Geodynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon)
    dom = Geodynamo.create_radial_domain(nr)

    # Helper to compute global L2 error
    global_l2(x) = sqrt(MPI.Allreduce(sum(abs2, x), MPI.SUM, comm))

    # -----------------------------
    # Scalar transform roundtrip
    # -----------------------------
    spec1 = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    spec2 = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    phys  = Geodynamo.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)

    # Fill spectral coefficients with reproducible random values
    Random.seed!(1234 + rank)
    r_real = parent(spec1.data_real)
    r_imag = parent(spec1.data_imag)
    r_real .= randn.(Float64)
    r_imag .= randn.(Float64)

    # Enforce m=0 imaginary = 0 (common convention)
    # We approximate by zeroing every nlm index corresponding to m==0
    for lm_idx in 1:cfg.nlm
        if cfg.m_values[lm_idx] == 0
            local_lm = lm_idx - first(range_local(cfg.pencils.spec, 1)) + 1
            if 1 <= local_lm <= size(r_imag, 1)
                r_imag[local_lm, 1, :] .= 0.0
            end
        end
    end

    # Synthesis and analysis
    Geodynamo.shtnskit_spectral_to_physical!(spec1, phys)
    Geodynamo.shtnskit_physical_to_spectral!(phys, spec2)

    # Compute spectral error
    e_real = parent(spec2.data_real) .- parent(spec1.data_real)
    e_imag = parent(spec2.data_imag) .- parent(spec1.data_imag)
    err = sqrt(global_l2(e_real)^2 + global_l2(e_imag)^2)
    ref = sqrt(global_l2(parent(spec1.data_real))^2 + global_l2(parent(spec1.data_imag))^2)
    rel_err_scalar = err / max(ref, eps())

    # -----------------------------
    # Vector transform roundtrip
    # -----------------------------
    tor1 = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    pol1 = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    tor2 = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    pol2 = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    vec  = Geodynamo.create_shtns_vector_field(Float64, cfg, dom, (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))

    # Fill toroidal/poloidal
    parent(tor1.data_real) .= randn.(Float64)
    parent(tor1.data_imag) .= randn.(Float64)
    parent(pol1.data_real) .= randn.(Float64)
    parent(pol1.data_imag) .= randn.(Float64)

    # Vector synthesis and analysis
    Geodynamo.shtnskit_vector_synthesis!(tor1, pol1, vec)
    Geodynamo.shtnskit_vector_analysis!(vec, tor2, pol2)

    e_tor_r = parent(tor2.data_real) .- parent(tor1.data_real)
    e_tor_i = parent(tor2.data_imag) .- parent(tor1.data_imag)
    e_pol_r = parent(pol2.data_real) .- parent(pol1.data_real)
    e_pol_i = parent(pol2.data_imag) .- parent(pol1.data_imag)
    err_vec = sqrt(global_l2(e_tor_r)^2 + global_l2(e_tor_i)^2 + global_l2(e_pol_r)^2 + global_l2(e_pol_i)^2)
    ref_vec = sqrt(global_l2(parent(tor1.data_real))^2 + global_l2(parent(tor1.data_imag))^2 +
                   global_l2(parent(pol1.data_real))^2 + global_l2(parent(pol1.data_imag))^2)
    rel_err_vector = err_vec / max(ref_vec, eps())

    if rank == 0
        println("\nSHTnsKit smoke test results:")
        println("  Scalar transform rel. error: ", @sprintf("%.3e", rel_err_scalar))
        println("  Vector  transform rel. error: ", @sprintf("%.3e", rel_err_vector))
        tol = 1e-8
        println("  Status: ", (rel_err_scalar < tol && rel_err_vector < tol) ? "PASS" : "WARN (check tolerances)")
    end

    MPI.Barrier(comm)
    if !MPI.Is_finalized()
        MPI.Finalize()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
