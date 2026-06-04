# Tier-3 extended kernel coverage for three numeric-kernel files:
#   * src/fields/transforms.jl   — spectral helpers, energy/enstrophy fallbacks,
#                                  rotations, in-place transforms, filters
#   * src/physics/nonlinear.jl   — scalar advection, internal sources, ball
#                                  regularity, geometric (1/r) factors
#   * src/physics/magnetic/solver.jl — conducting-inner-core ICB coupling helpers
#
# Every @test asserts a known-correct value: an analytic number, a round-trip
# identity, a conservation/continuity property, or a documented boundary value.
# All fixtures are deterministic (no rand()).

using Test
using GeoDynamo
using LinearAlgebra

const G = GeoDynamo

# A shared band-limited spectral configuration for the transform tests.
_t3_cfg() = G.create_shtnskit_config(lmax = 4, mmax = 4, nlat = 10, nlon = 16, nr = 6)

# ============================================================================
# transforms.jl
# ============================================================================
@testset "tier3 transforms.jl extended" begin
    cfg = _t3_cfg()

    # --- compute_scalar_energy_spectrum fallback branch -------------------
    # An undersized alm (2x2) makes SHTnsKit.energy_scalar_l_spectrum throw a
    # BoundsError, exercising the manual fallback (transforms.jl ~1440-1455).
    # Only l=0,m=0 is representable; the bounds-check zeros every higher degree.
    @testset "scalar energy spectrum manual fallback" begin
        small = zeros(ComplexF64, 2, 2)
        small[1, 1] = 5.0 + 0.0im                 # l=0, m=0
        spec = G.compute_scalar_energy_spectrum(cfg, small; real_field = true)
        @test length(spec) == cfg.lmax + 1
        @test spec[1] == abs2(5.0)                # 25.0
        @test all(==(0.0), spec[2:end])           # higher l bounds-checked out
    end

    # --- compute_enstrophy fallback branch --------------------------------
    # enstrophy() also BoundsErrors on a 2x2 Tlm -> manual l(l+1) fallback
    # (transforms.jl ~1519-1534).  l=1,m=0 coeff=2 -> |2|^2 * 1*2 = 8.
    @testset "enstrophy manual fallback" begin
        Tsmall = zeros(ComplexF64, 2, 2)
        Tsmall[2, 1] = 2.0 + 0.0im                # l=1, m=0
        @test G.compute_enstrophy(cfg, Tsmall; real_field = true) == 8.0
    end

    # --- compute_vector_energy_spectrum fallback branch -------------------
    # energy_vector_l_spectrum BoundsErrors on 2x2 -> fallback sums the scalar
    # spectra of S and T (transforms.jl ~1473-1476).  S(l=1,m=0)=2 -> 4,
    # T(l=1,m=0)=2 -> 4, total at l=1 is 8.
    @testset "vector energy spectrum manual fallback" begin
        Ssmall = zeros(ComplexF64, 2, 2)
        Tsmall = zeros(ComplexF64, 2, 2)
        Ssmall[2, 1] = 2.0 + 0.0im
        Tsmall[2, 1] = 2.0 + 0.0im
        vs = G.compute_vector_energy_spectrum(cfg, Ssmall, Tsmall; real_field = true)
        @test length(vs) == cfg.lmax + 1
        @test vs[2] == 8.0                        # l=1: 4 (S) + 4 (T)
        @test vs[1] == 0.0
    end

    # --- total energy equals sum of spectrum (native path consistency) ----
    # compute_total_scalar_energy must agree with sum(compute_scalar_energy_spectrum)
    # for a valid full-size coefficient matrix.
    @testset "total scalar energy = sum(spectrum)" begin
        alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        alm[3, 1] = 3.0 + 0.0im                   # l=2, m=0
        total = G.compute_total_scalar_energy(cfg, alm; real_field = true)
        summed = sum(G.compute_scalar_energy_spectrum(cfg, alm; real_field = true))
        @test total ≈ summed atol = 1e-12
        @test total > 0.0
    end

    # --- rotate_field_z! (manual phase-rotation fallback) -----------------
    # SHTnsKit.SH_Zrotate is defined but MethodErrors here, so the manual
    # phase rotation alm[l,m] *= exp(-i m alpha) runs (transforms.jl ~1869-1885,
    # incl. the in-place store at 1878).  Rotating m=1 by pi/2 -> factor -i;
    # m=0 is invariant.
    @testset "rotate_field_z! phase rotation" begin
        alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        alm[2, 2] = 1.0 + 0.0im                   # l=1, m=1
        alm[2, 1] = 4.0 + 0.0im                   # l=1, m=0 (must be unchanged)
        out = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        ret = G.rotate_field_z!(cfg, alm, π / 2; alm_out = out)
        @test ret === out
        @test out[2, 2] ≈ -1.0im atol = 1e-12     # exp(-i*pi/2)
        @test out[2, 1] ≈ 4.0 + 0.0im atol = 1e-12

        # In-place variant (alm_out === nothing): rotates alm itself (line 1878).
        alm_ip = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        alm_ip[2, 2] = 1.0 + 0.0im
        ret2 = G.rotate_field_z!(cfg, alm_ip, π / 2)
        @test ret2 === alm_ip
        @test alm_ip[2, 2] ≈ -1.0im atol = 1e-12
        # A full 2*pi rotation is the identity.
        alm_full = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        alm_full[3, 3] = 2.0 - 1.0im              # l=2, m=2
        G.rotate_field_z!(cfg, alm_full, 2π)
        @test alm_full[3, 3] ≈ 2.0 - 1.0im atol = 1e-12
    end

    # --- in-place synthesis/analysis round-trip ---------------------------
    # shtnskit_synthesis_inplace! ∘ shtnskit_analysis_inplace! ≈ identity on a
    # band-limited field (transforms.jl ~1800-1838).
    @testset "in-place synthesis/analysis round-trip" begin
        rcfg = G.create_shtnskit_config(lmax = 6, mmax = 6, nlat = 16, nlon = 24, nr = 4)
        alm = zeros(ComplexF64, rcfg.lmax + 1, rcfg.mmax + 1)
        alm[1, 1] = 1.0 + 0.0im
        alm[3, 1] = 0.4 + 0.0im
        alm[4, 2] = 0.2 + 0.1im
        f = zeros(Float64, rcfg.nlat, rcfg.nlon)
        G.shtnskit_synthesis_inplace!(rcfg, alm, f)
        @test all(isfinite, f)
        @test size(f) == (rcfg.nlat, rcfg.nlon)
        back = zeros(ComplexF64, rcfg.lmax + 1, rcfg.mmax + 1)
        G.shtnskit_analysis_inplace!(rcfg, f, back)
        @test maximum(abs, back .- alm) < 1e-12
    end

    # --- apply_inverse_horizontal_laplacian! ------------------------------
    # For l>0 the operator multiplies by -1/(l(l+1)); l=0 is regularized to 0
    # (default) or preserved when regularize_l0=false (transforms.jl ~2074-2099,
    # incl. the keep branch at 2088-2089).
    @testset "inverse horizontal laplacian" begin
        alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        alm[1, 1] = 7.0 + 0.0im                   # l=0, m=0
        alm[3, 1] = 6.0 + 0.0im                   # l=2, m=0
        out = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        G.apply_inverse_horizontal_laplacian!(cfg, alm; alm_out = out, regularize_l0 = true)
        @test out[3, 1] ≈ -6.0 / (2 * 3) atol = 1e-12   # -1.0
        @test out[1, 1] == 0.0 + 0.0im            # l=0 regularized

        out2 = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        G.apply_inverse_horizontal_laplacian!(cfg, alm; alm_out = out2, regularize_l0 = false)
        @test out2[1, 1] ≈ 7.0 + 0.0im atol = 1e-12     # l=0 preserved
        @test out2[3, 1] ≈ -1.0 + 0.0im atol = 1e-12
    end

    # --- apply_exponential_filter! degenerate lmax=0 path -----------------
    # With lmax=0 there is nothing to filter; the monopole is copied through
    # (transforms.jl ~2209-2219).
    @testset "exponential filter lmax=0 passthrough" begin
        cfg0 = G.create_shtnskit_config(lmax = 0, mmax = 0, nlat = 4, nlon = 4, nr = 4)
        alm0 = zeros(ComplexF64, 1, 1)
        alm0[1, 1] = 5.0 + 0.0im
        out0 = zeros(ComplexF64, 1, 1)
        r = G.apply_exponential_filter!(cfg0, alm0; alm_out = out0)
        @test r === out0
        @test out0[1, 1] == 5.0 + 0.0im

        r2 = G.apply_exponential_filter!(cfg0, alm0)   # no alm_out -> returns alm
        @test r2 === alm0
        @test alm0[1, 1] == 5.0 + 0.0im

        # cutoff out of range must error (validation branch).
        cfg2 = _t3_cfg()
        almv = zeros(ComplexF64, cfg2.lmax + 1, cfg2.mmax + 1)
        @test_throws ArgumentError G.apply_exponential_filter!(cfg2, almv; cutoff = 0.0)
    end

    # --- extract_coefficients_pair_for_shtnskit (serial local path) -------
    # On a single rank the spectral matrix is fully local, so the pair extractor
    # returns the two coefficient matrices directly (transforms.jl ~808-827).
    @testset "extract_coefficients_pair_for_shtnskit" begin
        dom = G.create_radial_domain(6)
        s1 = G.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
        s2 = G.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
        m20 = G.get_mode_index(cfg, 2, 0)
        slot20 = G.local_spectral_storage_slot(cfg, m20)
        G.set_local_spectral_value!(parent(s1.data_real), slot20, 3, 7.0)
        m11 = G.get_mode_index(cfg, 1, 1)
        slot11 = G.local_spectral_storage_slot(cfg, m11)
        G.set_local_spectral_value!(parent(s2.data_real), slot11, 3, 2.0)
        G.set_local_spectral_value!(parent(s2.data_imag), slot11, 3, -1.0)
        c1, c2 = G.extract_coefficients_pair_for_shtnskit(
            parent(s1.data_real), parent(s1.data_imag),
            parent(s2.data_real), parent(s2.data_imag), 3, cfg)
        @test size(c1) == (cfg.lmax + 1, cfg.mmax + 1)
        @test size(c2) == (cfg.lmax + 1, cfg.mmax + 1)
        @test c1[3, 1] ≈ 7.0 + 0.0im atol = 1e-12        # l=2, m=0
        @test c2[2, 2] ≈ 2.0 - 1.0im atol = 1e-12        # l=1, m=1
    end

    # --- index_to_lm_fast bounds (transforms.jl ~923-928) -----------------
    @testset "index_to_lm_fast" begin
        @test G.index_to_lm_fast(1, cfg) == (0, 0)
        @test G.index_to_lm_fast(cfg.nlm + 1, cfg) == (-1, -1)  # out of range high
        @test G.index_to_lm_fast(0, cfg) == (-1, -1)            # out of range low
    end
end

# ============================================================================
# nonlinear.jl
# ============================================================================
@testset "tier3 nonlinear.jl extended" begin
    cfg = _t3_cfg()
    dom = G.create_radial_domain(6)

    # A minimal velocity stand-in: solver_compute_scalar_advection_local! only
    # touches vel_fields.velocity.{r,θ,φ}_component.data, so a NamedTuple of
    # physical fields on the same grid is a faithful, lightweight fixture.
    _mkphys() = G.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)

    # --- solver_compute_scalar_advection_local! ---------------------------
    # advection = -(u_r ∇_r + u_θ ∇_θ + u_φ ∇_φ); with unit velocity and
    # gradient (2,3,4) every node is -(2+3+4) = -9 (nonlinear.jl ~854-858).
    @testset "scalar advection -(u·∇)" begin
        temp = G.create_shtns_temperature_field(Float64, cfg, dom)
        ur, uth, uph = _mkphys(), _mkphys(), _mkphys()
        fill!(parent(ur.data), 1.0)
        fill!(parent(uth.data), 1.0)
        fill!(parent(uph.data), 1.0)
        vel = (velocity = (r_component = ur, θ_component = uth, φ_component = uph),)
        fill!(parent(temp.gradient.r_component.data), 2.0)
        fill!(parent(temp.gradient.θ_component.data), 3.0)
        fill!(parent(temp.gradient.φ_component.data), 4.0)
        G.solver_compute_scalar_advection_local!(temp, vel)
        adv = parent(temp.advection_physical.data)
        @test all(==(-9.0), adv)
        @test all(isfinite, adv)
    end

    # --- solver_add_internal_sources_local! -------------------------------
    # A nonzero radial source profile is added uniformly over each shell
    # (nonlinear.jl ~863-890).  source[k] = 10k => advection at radial level k
    # equals 10k everywhere on that shell.
    @testset "add internal sources" begin
        temp = G.create_shtns_temperature_field(Float64, cfg, dom)
        fill!(parent(temp.advection_physical.data), 0.0)
        temp.internal_sources .= [10.0 * k for k in 1:dom.N]
        G.solver_add_internal_sources_local!(temp, dom)
        adv = parent(temp.advection_physical.data)
        nlat, nlon, nr = size(temp.advection_physical.data)
        @test adv[1, 1, 1] == 10.0
        @test adv[1, 1, 2] == 20.0
        @test all(==(10.0), adv[:, :, 1])         # uniform over shell 1
        @test all(==(20.0), adv[:, :, 2])

        # No-op fast path: an all-zero source profile leaves the field untouched.
        temp2 = G.create_shtns_temperature_field(Float64, cfg, dom)
        fill!(parent(temp2.advection_physical.data), 7.0)
        @test all(iszero, temp2.internal_sources)
        G.solver_add_internal_sources_local!(temp2, dom)
        @test all(==(7.0), parent(temp2.advection_physical.data))
    end

    # --- solver_enforce_ball_scalar_regularity! ---------------------------
    # At the geometric centre (radial index 1) every l>0 spectral mode must
    # vanish; l=0 is preserved (nonlinear.jl ~893-920).
    @testset "ball scalar regularity zeros l>0 at centre" begin
        spec = G.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
        for (l, m) in [(0, 0), (1, 0), (2, 0), (3, 0)]
            mi = G.get_mode_index(cfg, l, m)
            s = G.local_spectral_storage_slot(cfg, mi)
            G.set_local_spectral_value!(parent(spec.data_real), s, 1, 5.0)
            G.set_local_spectral_value!(parent(spec.data_real), s, 2, 9.0)
        end
        G.solver_enforce_ball_scalar_regularity!(spec)
        s00 = G.local_spectral_storage_slot(cfg, G.get_mode_index(cfg, 0, 0))
        @test G.local_spectral_value(parent(spec.data_real), s00, 1) == 5.0   # l=0 kept
        for l in (1, 2, 3)
            s = G.local_spectral_storage_slot(cfg, G.get_mode_index(cfg, l, 0))
            @test G.local_spectral_value(parent(spec.data_real), s, 1) == 0.0  # zeroed at r=1
            @test G.local_spectral_value(parent(spec.data_real), s, 2) == 9.0  # untouched off-centre
        end
    end

    # --- scalar_nonlinear_to_spectral! ball branch ------------------------
    # The :ball geometry path runs the analysis then enforces centre regularity,
    # so any l>0 mode at r=1 must be zero (nonlinear.jl ~923-940).
    @testset "scalar_nonlinear_to_spectral! ball enforces regularity" begin
        phys = G.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)
        fill!(parent(phys.data), 1.0)
        spec = G.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
        ret = G.scalar_nonlinear_to_spectral!(phys, spec, :ball)
        @test ret === spec
        for l in (1, 2, 3, 4)
            mi = G.get_mode_index(cfg, l, 0)
            s = G.local_spectral_storage_slot(cfg, mi)
            s === nothing && continue
            @test G.local_spectral_value(parent(spec.data_real), s, 1) == 0.0
        end
    end

    # --- apply_geometric_factors_spectral! --------------------------------
    # For r>0 the horizontal gradients are scaled by 1/r; at the ball centre
    # (r==0) they are forced to zero (nonlinear.jl ~230-254, incl. the r==0
    # branch at 235-243).
    @testset "geometric (1/r) factors + r=0 centre" begin
        # Private copy so zeroing a radius below does not mutate the shared `dom`
        # fixture (keeps this testset order-independent).
        dom_local = deepcopy(dom)
        tlp, tlm = G.build_theta_gradient_neighbors(cfg)
        nr_local = length(G.local_range(cfg.pencils.spec, 3))
        mkspec() = G.create_shtns_spectral_field(Float64, cfg, dom_local, cfg.pencils.spec)
        ws = G.SolverGradientWorkspace(
            mkspec(), mkspec(), mkspec(),
            zeros(Float64, cfg.nlm, nr_local), zeros(Float64, cfg.nlm, nr_local),
            tlp, tlm)
        temp = G.create_shtns_temperature_field(Float64, cfg, dom_local)
        m20 = G.get_mode_index(cfg, 2, 0)
        s = G.local_spectral_storage_slot(cfg, m20)
        for r in 1:dom_local.N
            G.set_local_spectral_value!(parent(ws.∇θ_spec.data_real), s, r, 4.0)
            G.set_local_spectral_value!(parent(ws.∇φ_spec.data_real), s, r, 6.0)
        end
        # Force a ball-centre at radial index 1 by zeroing its physical radius.
        dom_local.r[1, 4] = 0.0
        rinv2 = dom_local.r[2, 3]
        G.apply_geometric_factors_spectral!(ws, temp, dom_local)
        # centre: θ and φ gradients zeroed
        @test G.local_spectral_value(parent(ws.∇θ_spec.data_real), s, 1) == 0.0
        @test G.local_spectral_value(parent(ws.∇φ_spec.data_real), s, 1) == 0.0
        # interior: scaled by 1/r
        @test G.local_spectral_value(parent(ws.∇θ_spec.data_real), s, 2) ≈ 4.0 * rinv2 atol = 1e-12
        @test G.local_spectral_value(parent(ws.∇φ_spec.data_real), s, 2) ≈ 6.0 * rinv2 atol = 1e-12
    end
end

# ============================================================================
# magnetic/solver.jl
# ============================================================================
@testset "tier3 magnetic/solver.jl extended" begin
    cfg = G.create_shtnskit_config(lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16)
    oc = G.create_radial_domain(16)
    ic = G.create_radial_domain(8)   # inner-core ball domain (Nic = 8)

    # --- _magnetic_toroidal_inner_bc_increment ----------------------------
    # Without CONTINUITY_MAG modes the helper returns `nothing`; once enabled it
    # couples the toroidal inner-boundary RHS to -nl_poloidal at radial index 1
    # for every mode (magnetic/solver.jl ~101-138).
    @testset "toroidal inner-bc increment (CONTINUITY_MAG coupling)" begin
        mag = G.create_shtns_magnetic_fields(Float64, cfg, oc, ic)
        @test G._magnetic_toroidal_inner_bc_increment(mag) === nothing

        fill!(mag.toroidal.bc_type_inner, Int(G.CONTINUITY_MAG))
        m20 = G.get_mode_index(cfg, 2, 0)
        s = G.local_spectral_storage_slot(cfg, m20)
        G.set_local_spectral_value!(parent(mag.nl_poloidal.data_real), s, 1, 5.0)
        G.set_local_spectral_value!(parent(mag.prev_nl_poloidal.data_real), s, 1, 3.0)
        res = G._magnetic_toroidal_inner_bc_increment(mag)
        @test res !== nothing
        bc_real, prev_real, bc_imag, prev_imag = res
        @test length(bc_real) == mag.toroidal.nlm
        @test bc_real[m20] == -5.0                # -nl_poloidal[1]
        @test prev_real[m20] == -3.0              # -prev_nl_poloidal[1]
        @test bc_imag[m20] == 0.0
        @test prev_imag[m20] == 0.0
    end

    # Shared admittance for the conducting-inner-core helpers.
    luniq = sort(unique(cfg.l_values))
    filter!(>(0), luniq)
    adm = G.create_inner_core_admittance(Float64, luniq, ic, 1.0, 1e-3; theta = 0.5)

    # --- _magnetic_conducting_history_flux --------------------------------
    # A zero inner-core history gives φ0 = 0 (documented contract); a nonzero
    # history gives a finite nonzero flux (magnetic/solver.jl ~155-193).
    @testset "conducting history flux φ0" begin
        @test adm.Nic == 8
        mag = G.create_shtns_magnetic_fields(Float64, cfg, oc, ic)
        fill!(parent(mag.poloidal_ic.data_real), 0.0)
        fill!(parent(mag.poloidal_ic.data_imag), 0.0)
        φ0_real, φ0_imag = G._magnetic_conducting_history_flux(mag, mag.poloidal_ic, adm)
        @test length(φ0_real) == mag.poloidal_ic.nlm
        @test maximum(abs, φ0_real) == 0.0        # zero history -> zero flux
        @test maximum(abs, φ0_imag) == 0.0

        m20 = G.get_mode_index(cfg, 2, 0)
        s = G.local_spectral_storage_slot(cfg, m20)
        for r in 1:adm.Nic
            G.set_local_spectral_value!(parent(mag.poloidal_ic.data_real), s, r, Float64(r))
        end
        φ0r, φ0i = G._magnetic_conducting_history_flux(mag, mag.poloidal_ic, adm)
        @test isfinite(φ0r[m20])
        @test abs(φ0r[m20]) > 0.0                 # nonzero history -> nonzero flux
        @test φ0i[m20] == 0.0                     # imaginary part untouched
    end

    # --- _magnetic_conducting_reconstruct! --------------------------------
    # After the outer-core solve the inner-core profile is reconstructed with
    # regularity S(0)=0 and ICB continuity S(ri)=g, where g is the outer-core
    # value at radial index 1 (magnetic/solver.jl ~209-247).
    @testset "conducting inner-core reconstruction (S(0)=0, S(ri)=g)" begin
        mag = G.create_shtns_magnetic_fields(Float64, cfg, oc, ic)
        fill!(parent(mag.poloidal_ic.data_real), 0.0)
        fill!(parent(mag.poloidal_ic.data_imag), 0.0)
        m20 = G.get_mode_index(cfg, 2, 0)
        s = G.local_spectral_storage_slot(cfg, m20)
        G.set_local_spectral_value!(parent(mag.poloidal.data_real), s, 1, 2.5)  # ICB value g
        G._magnetic_conducting_reconstruct!(mag.poloidal, mag.poloidal_ic, adm)
        icr = parent(mag.poloidal_ic.data_real)
        @test G.local_spectral_value(icr, s, 1) == 0.0             # regularity at centre
        @test G.local_spectral_value(icr, s, adm.Nic) ≈ 2.5 atol = 1e-12  # ICB continuity
        for r in 1:adm.Nic
            @test isfinite(G.local_spectral_value(icr, s, r))
        end
    end
end
