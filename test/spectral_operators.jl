using Test
using MPI
using GeoDynamo
const G = GeoDynamo

# Unit coverage for the dense-alm spectral operator public API that was previously
# untested: horizontal Laplacian (+inverse), gradient magnitude, enstrophy, custom
# and exponential spectral filters, mode truncation, and field rotations.
#
# Assertions use convention-independent invariants (exact algebra for the pure-loop
# operators; scaling / identity / energy-invariance for the ops that may dispatch to
# a native SHTnsKit routine) so they hold regardless of the SHTnsKit build.

@testset "Spectral operators (dense alm)" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping"
        return
    end
    MPI.Initialized() || MPI.Init()

    lmax = 4;
    mmax = 4
    nlat = 10;
    nlon = 16;
    nr = 4
    cfg = G.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)

    newalm() = zeros(ComplexF64, lmax + 1, mmax + 1)
    set!(a, l, m, v) = (a[l + 1, m + 1] = v)
    get(a, l, m) = a[l + 1, m + 1]

    @testset "apply_horizontal_laplacian! scales by -l(l+1)" begin
        a = newalm()
        set!(a, 0, 0, 3.0 + 0im)
        set!(a, 1, 0, 1.0 + 0im)
        set!(a, 2, 1, 2.0 - 1.0im)
        set!(a, 4, 3, -0.5 + 0.25im)
        G.apply_horizontal_laplacian!(cfg, a)        # in place
        @test get(a, 0, 0) == 0.0 + 0im              # l=0 → factor 0
        @test get(a, 1, 0) ≈ (1.0 + 0im) * (-2)
        @test get(a, 2, 1) ≈ (2.0 - 1.0im) * (-6)
        @test get(a, 4, 3) ≈ (-0.5 + 0.25im) * (-20)
    end

    @testset "inverse Laplacian round-trips l≥1, zeros l=0" begin
        a0 = newalm()
        set!(a0, 0, 0, 7.0 + 0im)
        set!(a0, 1, 1, 1.5 + 0.5im)
        set!(a0, 3, 2, -2.0 + 1.0im)
        lap = newalm();
        G.apply_horizontal_laplacian!(cfg, a0; alm_out = lap)
        inv = newalm();
        G.apply_inverse_horizontal_laplacian!(cfg, lap; alm_out = inv)
        @test get(inv, 0, 0) == 0.0 + 0im            # l=0 regularized to 0
        @test get(inv, 1, 1) ≈ get(a0, 1, 1)
        @test get(inv, 3, 2) ≈ get(a0, 3, 2)
    end

    @testset "compute_horizontal_gradient_magnitude = Σ l(l+1)|a|² (×2 for m>0)" begin
        a = newalm()
        @test G.compute_horizontal_gradient_magnitude(cfg, a) == 0.0   # zero field
        set!(a, 0, 0, 5.0 + 0im)                                       # l=0 contributes 0
        @test G.compute_horizontal_gradient_magnitude(cfg, a) == 0.0
        b = newalm();
        set!(b, 2, 0, 3.0 + 0im)                         # 6·|3|²·1
        @test G.compute_horizontal_gradient_magnitude(cfg, b) ≈ 6 * 9
        c = newalm();
        set!(c, 2, 1, 3.0 + 0im)                         # 6·|3|²·2 (m>0)
        @test G.compute_horizontal_gradient_magnitude(cfg, c) ≈ 6 * 9 * 2
    end

    @testset "compute_enstrophy: zero, non-negative, quadratic scaling" begin
        z = newalm()
        @test G.compute_enstrophy(cfg, z) == 0.0
        a = newalm();
        set!(a, 1, 0, 1.0 + 0im);
        set!(a, 2, 1, 0.5 - 0.5im)
        e1 = G.compute_enstrophy(cfg, a)
        @test e1 > 0.0
        e2 = G.compute_enstrophy(cfg, 2.0 .* a)
        @test e2 ≈ 4 * e1                                              # quadratic form
        l0 = newalm();
        set!(l0, 0, 0, 4.0 + 0im)
        @test G.compute_enstrophy(cfg, l0) ≈ 0.0 atol=1e-12            # l=0 → no enstrophy
    end

    @testset "apply_spectral_filter! applies filter_func(l,m)" begin
        a = newalm()
        set!(a, 1, 0, 2.0 + 0im);
        set!(a, 2, 0, 4.0 + 0im);
        set!(a, 2, 1, 1.0 + 1im)
        ref = copy(a)
        G.apply_spectral_filter!(cfg, a, (l, m) -> 0.5)               # halve all
        @test a ≈ 0.5 .* ref
        b = newalm();
        set!(b, 1, 0, 2.0 + 0im);
        set!(b, 2, 0, 4.0 + 0im)
        G.apply_spectral_filter!(cfg, b, (l, m) -> l == 2 ? 0.0 : 1.0)
        @test get(b, 1, 0) ≈ 2.0 + 0im
        @test get(b, 2, 0) == 0.0 + 0im
    end

    @testset "apply_exponential_filter! is 1 at l=0, 0.5 at cutoff·lmax, monotone" begin
        # lmax=4, cutoff=0.5 ⇒ l=2 sits exactly at the half-power point.
        a = newalm()
        for l in 0:lmax
            set!(a, l, 0, 1.0 + 0im)
        end
        G.apply_exponential_filter!(cfg, a; order = 16, cutoff = 0.5)
        @test get(a, 0, 0) ≈ 1.0 + 0im                                # l=0 unchanged
        @test real(get(a, 2, 0)) ≈ 0.5 atol=1e-9                      # half power at l=2
        # monotone non-increasing in l along m=0
        vals = [real(get(a, l, 0)) for l in 0:lmax]
        @test all(vals[i] >= vals[i + 1] - 1e-12 for i in 1:lmax)
        @test_throws ArgumentError G.apply_exponential_filter!(cfg, newalm(); cutoff = 0.0)
        @test_throws ArgumentError G.apply_exponential_filter!(cfg, newalm(); cutoff = 1.5)
    end

    @testset "truncate_spectral_modes! zeros l>lmax_new or m>mmax_new" begin
        a = newalm()
        for l in 0:lmax, m in 0:min(l, mmax)

            set!(a, l, m, ComplexF64(l + 1, m))
        end
        out = newalm()
        G.truncate_spectral_modes!(cfg, a, 2, 1; alm_out = out)
        @test get(out, 1, 0) ≈ get(a, 1, 0)                           # kept
        @test get(out, 2, 1) ≈ get(a, 2, 1)                           # kept
        @test get(out, 3, 0) == 0.0 + 0im                             # l>2 dropped
        @test get(out, 2, 2) == 0.0 + 0im                             # m>1 dropped
    end

    @testset "rotate_field_z!: α=0 / α=2π identity, m=0 fixed, energy invariant" begin
        a = newalm()
        set!(a, 1, 0, 1.0 + 0im);
        set!(a, 2, 1, 0.7 - 0.3im);
        set!(a, 3, 2, -0.4 + 0.9im)
        e0 = sum(abs2, a)

        id0 = newalm();
        G.rotate_field_z!(cfg, a, 0.0; alm_out = id0)
        @test id0 ≈ a                                                 # zero rotation
        id2 = newalm();
        G.rotate_field_z!(cfg, a, 2π; alm_out = id2)
        @test id2 ≈ a                                                 # full turn
        rot = newalm();
        G.rotate_field_z!(cfg, a, 0.37; alm_out = rot)
        @test sum(abs2, rot) ≈ e0                                     # |phase|=1 ⇒ energy kept
        @test get(rot, 1, 0) ≈ get(a, 1, 0)                           # m=0 phase-invariant
    end

    @testset "rotations leave an l=0 monopole invariant (native or fallback)" begin
        mono = newalm();
        set!(mono, 0, 0, 2.5 + 0im)
        @test G.rotate_field_y!(cfg, mono, 0.9; alm_out = newalm()) ≈ mono
        @test G.rotate_field_90y!(cfg, mono; alm_out = newalm()) ≈ mono
        @test G.rotate_field_90x!(cfg, mono; alm_out = newalm()) ≈ mono
        @test G.rotate_field_euler!(cfg, mono, 0.3, 0.6, 0.9; alm_out = newalm()) ≈ mono
    end
end
