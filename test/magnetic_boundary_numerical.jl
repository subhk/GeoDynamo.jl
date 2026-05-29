using Test
using MPI
using LinearAlgebra

const FINALIZE_MPI_MAGBC = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# Materialize row `i` of a BandedMatrix as a dense length-N vector so the
# embedded boundary stencil can be dotted against analytic radial profiles.
function _magbc_banded_row(bm, i::Int)
    N = bm.size
    bw = bm.bandwidth
    row = zeros(eltype(bm.data), N)
    for j in max(1, i - bw):min(N, i + bw)
        band_row = bw + 1 + i - j
        if 1 <= band_row <= 2 * bw + 1
            row[j] = bm.data[band_row, j]
        end
    end
    return row
end

# Numerical contract for the matrix-embedded magnetic boundary conditions.
#
# Insulating magnetic BCs require, per spherical-harmonic degree l:
#   toroidal  : BT = 0 at both boundaries                       (identity rows)
#   poloidal  : (∂_r - l/r)    BP = 0 at the inner boundary     → matches r^l
#               (∂_r + (l+1)/r) BP = 0 at the outer boundary    → matches r^{-(l+1)}
#
# A potential (current-free) field has poloidal scalar ∝ r^l in the regular
# interior and ∝ r^{-(l+1)} in the decaying exterior. The embedded boundary row
# must therefore annihilate the *matching* potential while leaving the
# mismatched one untouched. The boundary derivative stencil is exact for
# polynomials up to the radial bandwidth (default 4), so r^l (l ≤ 4) is killed
# to round-off; r^{-(l+1)} is killed to finite-difference truncation order.
@testset "Magnetic boundary-condition numerical satisfaction" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping magnetic BC numerical tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    lmax = 4
    mmax = 4
    nlat = max(lmax + 2, 10)
    nlon = max(2lmax + 1, 16)
    nr = 24

    cfg = GeoDynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = GeoDynamo.create_radial_domain(nr)
    r = dom.r[:, 4]

    η = 1.0
    dt = 1.0e-3
    pol = GeoDynamo.create_magnetic_poloidal_matrices(cfg, dom, η, dt; theta=0.5, T=Float64)
    tor = GeoDynamo.create_magnetic_toroidal_matrices(cfg, dom, η, dt; theta=0.5, T=Float64)

    @testset "toroidal insulating rows are identity (BT = 0)" begin
        e_inner = zeros(nr); e_inner[1] = 1.0
        e_outer = zeros(nr); e_outer[nr] = 1.0
        for (idx, l) in enumerate(tor.l_values)
            A = tor.system_matrices[idx]
            @test _magbc_banded_row(A, 1) ≈ e_inner atol = 1e-14
            @test _magbc_banded_row(A, nr) ≈ e_outer atol = 1e-14
        end
    end

    @testset "poloidal insulating rows match potential-field decay" begin
        for (idx, l) in enumerate(pol.l_values)
            l == 0 && continue  # no magnetic monopole; skip the degenerate degree
            A = pol.system_matrices[idx]
            row_in = _magbc_banded_row(A, 1)
            row_out = _magbc_banded_row(A, nr)

            f_in = r .^ l              # regular interior potential   ∝ r^l
            f_out = r .^ (-(l + 1))    # decaying exterior potential  ∝ r^{-(l+1)}

            res_in_matched = abs(dot(row_in, f_in))
            res_in_wrong = abs(dot(row_in, f_out))
            res_out_matched = abs(dot(row_out, f_out))
            res_out_wrong = abs(dot(row_out, f_in))

            # Rows must be non-degenerate: they do NOT annihilate the mismatched
            # potential (also guards against an accidental all-zero boundary row).
            @test res_in_wrong > 1e-3
            @test res_out_wrong > 1e-3

            # Embedded rows annihilate the matching potential far better than the
            # mismatched one. A sign error on the l/r term collapses this gap.
            @test res_in_matched < 1e-8 * res_in_wrong
            @test res_out_matched < 1e-3 * res_out_wrong
        end
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_MAGBC && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
