using Test
using MPI
using Random

const FINALIZE_MPI = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# Restrict a randomly-seeded spectral field to the physically valid SH modes for
# this config's (l,m) storage layout. The spectral pencil is 2D (l_slot, m_slot, r),
# whose upper triangle (l < m) holds no degrees of freedom; the imaginary part of
# m=0 modes is also zero for real fields, and l=0 is invalid for the spheroidal-
# toroidal vector decomposition. The transform legitimately zeroes all of these,
# so they must not be seeded if a roundtrip is expected to be the identity.
# Uses the config's own (l,m)->global-mode map, so it is correct for any pencil
# layout (1D nlm or 2D l/m).
function sanitize_spectral_modes!(field, cfg; zero_l0::Bool=false)
    sr = parent(field.data_real)
    si = parent(field.data_imag)
    lm_map = GeoDynamo.local_spectral_lm_map(cfg)
    for slot in CartesianIndices(lm_map)
        gl = lm_map[slot]
        if gl == 0
            @views sr[slot, :] .= 0.0
            @views si[slot, :] .= 0.0
        else
            (cfg.m_values[gl] == 0) && (@views si[slot, :] .= 0.0)
            if zero_l0 && cfg.l_values[gl] == 0
                @views sr[slot, :] .= 0.0
                @views si[slot, :] .= 0.0
            end
        end
    end
    return field
end

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
    randn!(parent(spec1.data_real))
    randn!(parent(spec1.data_imag))

    # Keep only valid SH modes (zero invalid l<m slots and m=0 imaginary parts).
    sanitize_spectral_modes!(spec1, cfg)

    GeoDynamo.shtnskit_spectral_to_physical!(spec1, phys)
    GeoDynamo.shtnskit_physical_to_spectral!(phys, spec2)

    e_r = parent(spec2.data_real) .- parent(spec1.data_real)
    e_i = parent(spec2.data_imag) .- parent(spec1.data_imag)
    local_err = sum(abs2, e_r) + sum(abs2, e_i)
    err = MPI.Allreduce(local_err, MPI.SUM, comm)
    @test err / max(MPI.Allreduce(sum(abs2, parent(spec1.data_real)) + sum(abs2, parent(spec1.data_imag)), MPI.SUM, comm), eps()) < 1e-7

    # Vector roundtrip
    # Spectral fields (toroidal/poloidal) use spec pencil, vector components use physical pencils
    tor1 = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    pol1 = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    tor2 = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    pol2 = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    vec  = GeoDynamo.create_shtns_vector_field(Float64, cfg, dom, (cfg.pencils.phi, cfg.pencils.phi, cfg.pencils.phi))

    randn!(parent(tor1.data_real))
    randn!(parent(tor1.data_imag))
    randn!(parent(pol1.data_real))
    randn!(parent(pol1.data_imag))

    # Keep only valid SH modes; for the spheroidal-toroidal vector decomposition
    # l=0 is also invalid, so drop it too.
    sanitize_spectral_modes!(tor1, cfg; zero_l0=true)
    sanitize_spectral_modes!(pol1, cfg; zero_l0=true)

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

    @test err_vec / max(ref_vec, eps()) < 1e-7

    if MPI.Initialized()
        MPI.Barrier(comm)
        if FINALIZE_MPI && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
