using Test
using GeoDynamo
using MPI
using Random

const FINALIZE_MPI = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "SHTnsKit scalar and vector roundtrip" begin
    if MPI.Finalized()
        @warn "MPI already finalized before SHTnsKit roundtrip tests; skipping"
        return
    end

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = GeoDynamo.get_comm()
    rank = GeoDynamo.get_rank()

    lmax = 6; mmax = 6
    nlat = max(lmax + 2, 12)
    nlon = max(2lmax + 1, 24)
    nr   = 6

    cfg = GeoDynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = GeoDynamo.create_radial_domain(nr)

    # Scalar roundtrip
    # Spectral fields use spec pencil (nlm×1×nr), physical fields use physical pencils (nlat×nlon×nr)
    spec1 = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    spec2 = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    phys  = GeoDynamo.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.phi)

    Random.seed!(1234 + rank)
    parent(spec1.data_real) .= randn.(Float64)
    parent(spec1.data_imag) .= randn.(Float64)

    # Enforce spherical harmonic constraint: m=0 modes must have zero imaginary part
    for idx in eachindex(IndexLinear(), view(parent(spec1.data_real), :, 1, 1))
        l, m = GeoDynamo.index_to_lm_shtnskit(idx, cfg.lmax, cfg.mmax)
        if m == 0
            parent(spec1.data_imag)[idx, :, :] .= 0.0
        end
    end

    GeoDynamo.shtnskit_spectral_to_physical!(spec1, phys)
    GeoDynamo.shtnskit_physical_to_spectral!(phys, spec2)

    e_r = parent(spec2.data_real) .- parent(spec1.data_real)
    e_i = parent(spec2.data_imag) .- parent(spec1.data_imag)
    local_err = sum(abs2, e_r) + sum(abs2, e_i)
    err = MPI.Allreduce(local_err, MPI.SUM, comm)
    @test err / max(MPI.Allreduce(sum(abs2, parent(spec1.data_real)) + sum(abs2, parent(spec1.data_imag)), MPI.SUM, comm), eps()) < 1e-7

    # Vector roundtrip
    # NOTE: Vector transforms currently have issues in SHTnsKit.jl (see VECTOR_TRANSFORM_ISSUE.md)
    # - Threading bug causes BoundsError
    # - Coefficient recovery doesn't work correctly even with single thread
    # Skipping vector test until upstream SHTnsKit issues are resolved
    @test_skip begin  # Mark as expected failure for now
    # Spectral fields (toroidal/poloidal) use spec pencil, vector components use physical pencils
    tor1 = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    pol1 = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    tor2 = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    pol2 = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    vec  = GeoDynamo.create_shtns_vector_field(Float64, cfg, dom, (cfg.pencils.phi, cfg.pencils.phi, cfg.pencils.phi))

    parent(tor1.data_real) .= randn.(Float64)
    parent(tor1.data_imag) .= randn.(Float64)
    parent(pol1.data_real) .= randn.(Float64)
    parent(pol1.data_imag) .= randn.(Float64)

    # Enforce spherical harmonic constraints for vector fields:
    # 1. m=0 modes must have zero imaginary part
    # 2. l=0 modes are not valid for spheroidal-toroidal decomposition
    for idx in eachindex(IndexLinear(), view(parent(tor1.data_real), :, 1, 1))
        l, m = GeoDynamo.index_to_lm_shtnskit(idx, cfg.lmax, cfg.mmax)
        if m == 0
            parent(tor1.data_imag)[idx, :, :] .= 0.0
            parent(pol1.data_imag)[idx, :, :] .= 0.0
        end
        # Vector fields: l=0 mode is not physical in sph-tor decomposition
        if l == 0
            parent(tor1.data_real)[idx, :, :] .= 0.0
            parent(tor1.data_imag)[idx, :, :] .= 0.0
            parent(pol1.data_real)[idx, :, :] .= 0.0
            parent(pol1.data_imag)[idx, :, :] .= 0.0
        end
    end

    GeoDynamo.shtnskit_vector_synthesis!(tor1, pol1, vec)
    GeoDynamo.shtnskit_vector_analysis!(vec, tor2, pol2)

    e = sum(abs2, parent(tor2.data_real) .- parent(tor1.data_real)) +
        sum(abs2, parent(tor2.data_imag) .- parent(tor1.data_imag)) +
        sum(abs2, parent(pol2.data_real) .- parent(pol1.data_real)) +
        sum(abs2, parent(pol2.data_imag) .- parent(pol1.data_imag))
    err_vec = MPI.Allreduce(e, MPI.SUM, comm)

    ref = sum(abs2, parent(tor1.data_real)) + sum(abs2, parent(tor1.data_imag)) +
          sum(abs2, parent(pol1.data_real)) + sum(abs2, parent(pol1.data_imag))
    ref_vec = MPI.Allreduce(ref, MPI.SUM, comm)

    err_vec / max(ref_vec, eps()) < 1e-7
    end  # End @test_skip

    if MPI.Initialized()
        MPI.Barrier(comm)
        if FINALIZE_MPI && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
