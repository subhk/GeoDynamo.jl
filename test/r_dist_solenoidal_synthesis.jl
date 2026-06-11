# Storage-layout solenoidal coefficient helpers must reproduce the per-mode
# reference computation bit-exactly. The storage pencil keeps r fully local,
# so these helpers work on ANY process grid — this is what un-blocks
# r-distributed (1x4 / 2x2) solenoidal synthesis.
using Test
using MPI
using LinearAlgebra
using GeoDynamo

MPI.Initialized() || MPI.Init()

const RD_NR   = 16
const RD_LMAX = 8

function _rd_setup()
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = RD_LMAX, mmax = RD_LMAX,
        nlat = 2 * RD_LMAX + 4, nlon = 4 * RD_LMAX + 8,
        nr   = RD_NR)
    dom = GeoDynamo.create_radial_domain(RD_NR)
    return cfg, dom
end

_rd_spec(cfg, dom) =
    GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)

# Deterministic, rank-independent P: value depends only on (lm, r_idx).
function _rd_fill_poloidal!(P, cfg)
    pr = parent(P.data_real); pi_ = parent(P.data_imag)
    fill!(pr, 0.0); fill!(pi_, 0.0)
    for lm in 1:cfg.nlm
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        m = cfg.m_values[lm]
        for r_idx in 1:RD_NR
            GeoDynamo.set_local_spectral_value!(pr, slot, r_idx,
                sinpi(0.3 * (lm + 7 * r_idx)))
            if m > 0
                GeoDynamo.set_local_spectral_value!(pi_, slot, r_idx,
                    cospi(0.3 * (lm - 5 * r_idx)))
            end
        end
    end
    return P
end

@testset "storage-layout solenoidal coefficients" begin
    cfg, dom = _rd_setup()
    P = _rd_fill_poloidal!(_rd_spec(cfg, dom), cfg)
    pr = parent(P.data_real); pi_ = parent(P.data_imag)

    S  = _rd_spec(cfg, dom)
    Vr = _rd_spec(cfg, dom)
    sr = parent(S.data_real);  si = parent(S.data_imag)
    vr = parent(Vr.data_real); vi = parent(Vr.data_imag)

    GeoDynamo._storage_spheroidal_from_poloidal!(sr, si, pr, pi_, cfg, dom)
    GeoDynamo._storage_vr_coeffs!(vr, vi, pr, pi_, cfg, dom,
        GeoDynamo._solenoidal_vr_factor)

    # Reference: plain per-mode loops, same D1, same op order.
    D1   = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    prof = Vector{Float64}(undef, RD_NR)
    dpr  = Vector{Float64}(undef, RD_NR)
    rN   = dom.r[RD_NR, 4]
    @testset "S = (∂_r P)/r and Vr = l(l+1)·P/r², bit-exact" begin
        for lm in 1:cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]
            for (src, s_out, v_out) in ((pr, sr, vr), (pi_, si, vi))
                for r_idx in 1:RD_NR
                    prof[r_idx] = GeoDynamo.local_spectral_value(src, slot, r_idx)
                end
                mul!(dpr, D1, prof)
                for r_idx in 1:RD_NR
                    r = dom.r[r_idx, 4]
                    @test GeoDynamo.local_spectral_value(s_out, slot, r_idx) ==
                          dpr[r_idx] / r
                    vref = r > eps(Float64) * rN ?
                        prof[r_idx] * GeoDynamo._solenoidal_vr_factor(l, r) : 0.0
                    @test GeoDynamo.local_spectral_value(v_out, slot, r_idx) == vref
                end
            end
        end
    end

    @testset "vector scratch has storage-layout slabs" begin
        plan = GeoDynamo.get_disttranspose_plan(cfg)
        sc   = GeoDynamo._vector_scratch(cfg, plan)
        for f in (sc.Ssto_re, sc.Ssto_im, sc.Vrsto_re, sc.Vrsto_im)
            @test size(f) == size(pr)
            @test eltype(f) == Float64
        end
    end
end
