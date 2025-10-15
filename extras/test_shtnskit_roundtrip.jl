using Test
using Geodynamo
using MPI
using Random

@testset "SHTnsKit scalar and vector roundtrip" begin
    if !MPI.Initialized(); MPI.Init(); end
    comm = Geodynamo.get_comm()
    rank = Geodynamo.get_rank()

    lmax = 6; mmax = 6
    nlat = max(lmax + 2, 12)
    nlon = max(2lmax + 1, 24)
    nr   = 6

    cfg = Geodynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon)
    dom = Geodynamo.create_radial_domain(nr)

    # Scalar roundtrip
    spec1 = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    spec2 = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    phys  = Geodynamo.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)

    Random.seed!(1234 + rank)
    parent(spec1.data_real) .= randn.(Float64)
    parent(spec1.data_imag) .= randn.(Float64)

    Geodynamo.shtnskit_spectral_to_physical!(spec1, phys)
    Geodynamo.shtnskit_physical_to_spectral!(phys, spec2)

    e_r = parent(spec2.data_real) .- parent(spec1.data_real)
    e_i = parent(spec2.data_imag) .- parent(spec1.data_imag)
    local_err = sum(abs2, e_r) + sum(abs2, e_i)
    err = MPI.Allreduce(local_err, MPI.SUM, comm)
    @test err / max(MPI.Allreduce(sum(abs2, parent(spec1.data_real)) + sum(abs2, parent(spec1.data_imag)), MPI.SUM, comm), eps()) < 1e-7

    # Vector roundtrip
    tor1 = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    pol1 = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    tor2 = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    pol2 = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    vec  = Geodynamo.create_shtns_vector_field(Float64, cfg, dom, (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))

    parent(tor1.data_real) .= randn.(Float64)
    parent(tor1.data_imag) .= randn.(Float64)
    parent(pol1.data_real) .= randn.(Float64)
    parent(pol1.data_imag) .= randn.(Float64)

    Geodynamo.shtnskit_vector_synthesis!(tor1, pol1, vec)
    Geodynamo.shtnskit_vector_analysis!(vec, tor2, pol2)

    e = sum(abs2, parent(tor2.data_real) .- parent(tor1.data_real)) +
        sum(abs2, parent(tor2.data_imag) .- parent(tor1.data_imag)) +
        sum(abs2, parent(pol2.data_real) .- parent(pol1.data_real)) +
        sum(abs2, parent(pol2.data_imag) .- parent(pol1.data_imag))
    err_vec = MPI.Allreduce(e, MPI.SUM, comm)

    ref = sum(abs2, parent(tor1.data_real)) + sum(abs2, parent(tor1.data_imag)) +
          sum(abs2, parent(pol1.data_real)) + sum(abs2, parent(pol1.data_imag))
    ref_vec = MPI.Allreduce(ref, MPI.SUM, comm)

    @test err_vec / max(ref_vec, eps()) < 1e-7

    MPI.Barrier(comm)
    if !MPI.Is_finalized(); MPI.Finalize(); end
end
