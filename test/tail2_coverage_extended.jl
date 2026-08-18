# ================================================================================
# Extended tail coverage for bcs/topography + io
# ================================================================================
#
# Targets previously-uncovered branches in:
#   * src/bcs/topography/gaunt_tensors.jl   (numerical Gaunt path, selection rules)
#   * src/bcs/topography/topography_data.jl (file load, direct-summation evaluate,
#                                            associated-Legendre m>0, negative-m index)
#   * src/bcs/topography/stefan_condition.jl(base flux, diagnostics, dummy-field paths)
#   * src/bcs/topography/velocity_coupling.jl (impermeability / no-slip / stress-free
#                                            correction kernels, driven via directly
#                                            constructed BoundaryDerivativeCache —
#                                            no MPI/solver state required)
#   * src/bcs/file_bc_loader.jl             (cache dict-protocol, physical single-var
#                                            temperature/composition, dict cache path)
#   * src/bcs/netcdf_io.jl                  (time-dependent/components write+info,
#                                            validator out-of-range branches)
#   * src/io/restart.jl                     (pencils!==nothing read paths, physical
#                                            + spectral, at a single rank)
#
# Every @test asserts a known-correct value (closed-form coupling term, exact
# round-trip, analytic Legendre value, energy preservation, or size/finiteness).
# Deterministic: any randomness is seeded.

using Test
using GeoDynamo
using NCDatasets
using LinearAlgebra
using Random
using MPI

const TC2 = GeoDynamo.bcs.topography           # topography submodule
const BCS2 = GeoDynamo.bcs                      # bcs module

@testset "tail2 extended coverage" begin

    # =====================================================================
    # gaunt_tensors.jl
    # =====================================================================
    @testset "gaunt_tensors.jl" begin
        # --- selection-rule early returns in the Wigner-3j analytic path ---
        # m-sum violated (-m1+m2+M != 0): triangle ok but azimuthal fails.
        @test TC2.compute_gaunt_from_wigner3j(2, 1, 2, 0, 2, 0) == 0.0  # 1 != 0+0
        # triangle inequality violated: l1+l2 < L.
        @test TC2.compute_gaunt_from_wigner3j(0, 0, 0, 0, 2, 0) == 0.0
        # parity rule violated: l1+l2+L odd.
        @test TC2.compute_gaunt_from_wigner3j(1, 0, 1, 0, 1, 0) == 0.0

        # --- _wigner_3j selection-rule early returns ---
        @test TC2._wigner_3j(1, 1, 1, 1, 1, 1) == 0.0   # m1+m2+m3 != 0
        @test TC2._wigner_3j(1, 1, 2, 2, 0, -2) == 0.0  # |m1|>l1 (m1=2>l1=1)
        @test TC2._wigner_3j(1, 1, 5, 0, 0, 0) == 0.0   # l1+l2 < l3

        # --- evaluate_spherical_harmonics_grid / gradient out-of-range guards ---
        small = TC2.GauntTensorCache(2, 1)
        big_l = small.lmax + small.lmax_topo + 1
        Y_oob = TC2.evaluate_spherical_harmonics_grid(big_l, 0, small)
        @test size(Y_oob) == (small.nlat, small.nlon)
        @test all(iszero, Y_oob)
        g_th, g_ph = TC2.evaluate_spherical_harmonic_gradient_grid(big_l, 0, small)
        @test all(iszero, g_th) && all(iszero, g_ph)
        # l == 0 also returns zero gradient
        g0_th, g0_ph = TC2.evaluate_spherical_harmonic_gradient_grid(0, 0, small)
        @test all(iszero, g0_th) && all(iszero, g0_ph)

        # --- numerical compute_gaunt_tensor selection-rule returns ---
        @test TC2.compute_gaunt_tensor(2, 1, 2, 0, 2, 0, small) == 0.0  # azimuthal
        @test TC2.compute_gaunt_tensor(0, 0, 0, 0, 2, 0, small) == 0.0  # triangle
        @test TC2.compute_gaunt_tensor(1, 0, 1, 0, 1, 0, small) == 0.0  # parity

        # --- numerical Gaunt matches the analytic Wigner-3j value ---
        # G_{0,0,1,0,1,0}: a known nonzero coefficient.
        g_num = TC2.compute_gaunt_tensor(0, 0, 1, 0, 1, 0, small)
        g_wig = TC2.compute_gaunt_from_wigner3j(0, 0, 1, 0, 1, 0)
        @test isfinite(g_num)
        @test g_num≈g_wig atol=1e-9
        # Y_0^0 = 1/sqrt(4pi); ∫ Y_0^0 |Y_1^0|^2 dΩ = 1/sqrt(4π).
        @test g_num≈1 / sqrt(4π) atol=1e-9

        # --- compute_cross_gaunt_tensor: selection returns + a finite real value ---
        @test TC2.compute_cross_gaunt_tensor(1, 1, 1, 0, 1, 0, small) == 0.0  # azimuthal
        @test TC2.compute_cross_gaunt_tensor(1, 0, 0, 0, 1, 0, small) == 0.0  # l2==0
        xval = TC2.compute_cross_gaunt_tensor(1, 0, 1, 0, 1, 0, small)
        @test isfinite(xval)
        # Cross integral of identical real harmonics (A×A) integrates to ~0.
        @test abs(xval) < 1e-8

        # --- gaunt_on_the_fly numerical branch (use_wigner=false) ---
        otf = TC2.gaunt_on_the_fly(0, 0, 1, 0, 1, 0; use_wigner = false)
        @test otf≈1 / sqrt(4π) atol=1e-9

        # --- precompute_gaunt_tensors! with the numerical (non-Wigner) path ---
        # Tiny cache so the cross-Gaunt numerical integration stays cheap.
        gc = TC2.GauntTensorCache(1, 1)
        TC2.precompute_gaunt_tensors!(gc; verbose = false, use_wigner = false)
        @test gc.is_precomputed
        # The basic Gaunt G_{0,0,0,0,0,0} = 1/sqrt(4π) must be present.
        @test gc.G[(0, 0, 0, 0, 0, 0)]≈1 / sqrt(4π) atol=1e-9
        @test length(gc.G) > 0
    end

    # =====================================================================
    # topography_data.jl
    # =====================================================================
    @testset "topography_data.jl" begin
        # --- lm_to_index negative-m branch maps to |m| index ---
        @test TC2.lm_to_index(2, -1, 4) == TC2.lm_to_index(2, 1, 4)
        @test TC2.lm_to_index(3, -2, 4) == TC2.lm_to_index(3, 2, 4)

        # --- create_spherical_harmonic_topography for m < 0 (imag / sine part) ---
        f_neg = TC2.create_spherical_harmonic_topography(
            2, -1, 0.7, 1.0, GeoDynamo.OUTER_BOUNDARY; lmax = 4)
        idx = TC2.lm_to_index(2, 1, 4)
        # phase = iseven(-m)?1:-1 = iseven(1)?... = -1, so coeff = -0.7
        @test f_neg.coeffs_imag[idx] ≈ -0.7
        @test f_neg.coeffs_real[idx] == 0.0
        f_neg2 = TC2.create_spherical_harmonic_topography(
            2, -2, 0.5, 1.0, GeoDynamo.OUTER_BOUNDARY; lmax = 4)
        idx2 = TC2.lm_to_index(2, 2, 4)
        # -m = 2 even -> phase +1
        @test f_neg2.coeffs_imag[idx2] ≈ 0.5

        # --- load_topography_from_file: h_real/h_imag and complex h paths ---
        # NOTE: we deliberately omit the "lmax" attribute. NCDatasets reads an
        # integer attribute back as Int32, and TopographyField(::Int,...) only
        # has an Int64 method -> reading a file with an lmax attribute throws a
        # MethodError (suspected latent bug, see report). Without it, lmax
        # defaults to 32; we only wrote n=15 coefficients, so compare the first n.
        mktempdir() do dir
            field0 = TC2.TopographyField(4, 4, 1.0, GeoDynamo.OUTER_BOUNDARY)
            n = field0.nlm                       # 15 for lmax=4
            hr = collect(0.1:0.1:(0.1 * n))
            hi = collect(0.01:0.01:(0.01 * n))

            f1 = joinpath(dir, "topo_realimag.nc")
            NCDataset(f1, "c") do ds
                defDim(ds, "nlm", n)
                defVar(ds, "h_real", Float64, ("nlm",))
                defVar(ds, "h_imag", Float64, ("nlm",))
                ds["h_real"][:] = hr
                ds["h_imag"][:] = hi
                ds.attrib["radius"] = 1.0
            end
            loaded = TC2.load_topography_from_file(f1, GeoDynamo.OUTER_BOUNDARY)
            @test loaded.lmax == 32              # default (no lmax attribute)
            @test loaded.radius == 1.0           # read from radius attribute
            @test loaded.coeffs_real[1:n] ≈ hr
            @test loaded.coeffs_imag[1:n] ≈ hi
            @test all(iszero, loaded.coeffs_real[(n + 1):end])
            @test contains(loaded.description, "loaded from")
            # statistics recomputed on load. rms_amplitude is the spatial RMS
            # sqrt((1/4π)∫h²dΩ): the m>0 coefficients each count twice (the −m
            # partner shares the stored slot). The first n=15 coeffs follow the
            # lmax=4 packing (l ≤ 4 ⇒ min(l,mmax)=l), so build the matching m>0
            # weights independently of the production routine.
            rms_weights = Float64[m == 0 ? 1.0 : 2.0 for l in 0:4 for m in 0:l]
            expected_rms = sqrt(sum(rms_weights .* (hr .^ 2 .+ hi .^ 2)) / (4pi))
            @test loaded.rms_amplitude ≈ expected_rms

            # real single-variable "h" (NetCDF has no native complex type, so the
            # complex-"h" branch is unreachable via NCDatasets; the real branch
            # plus the radius override branch are covered here).
            f3 = joinpath(dir, "topo_realonly.nc")
            NCDataset(f3, "c") do ds
                defDim(ds, "nlm", n)
                defVar(ds, "h", Float64, ("nlm",))
                ds["h"][:] = hr
            end
            # radius override exercises the radius!==nothing branch
            loaded3 = TC2.load_topography_from_file(
                f3, GeoDynamo.INNER_BOUNDARY; radius = 0.35)
            @test loaded3.radius == 0.35
            @test loaded3.coeffs_real[1:n] ≈ hr
            @test all(iszero, loaded3.coeffs_imag)

            # missing variable -> ArgumentError
            f4 = joinpath(dir, "topo_bad.nc")
            NCDataset(f4, "c") do ds
                defDim(ds, "nlm", n)
                defVar(ds, "wrong", Float64, ("nlm",))
                ds["wrong"][:] = hr
            end
            @test_throws ArgumentError TC2.load_topography_from_file(
                f4, GeoDynamo.OUTER_BOUNDARY)

            # nonexistent file -> ArgumentError
            @test_throws ArgumentError TC2.load_topography_from_file(
                joinpath(dir, "nope.nc"), GeoDynamo.OUTER_BOUNDARY)
        end

        # --- evaluate_topography on an explicit (theta, phi) grid (direct sum) ---
        # Uniform topography: h_{0,0} = a*sqrt(4π) reconstructs to ~a everywhere
        # since Y_0^0 = 1/sqrt(4π).
        a = 0.25
        fu = TC2.create_uniform_topography(a, 1.0, GeoDynamo.OUTER_BOUNDARY; lmax = 4)
        theta = collect(range(0.3, π - 0.3; length = 5))
        phi = collect(range(0.0, 2π * (1 - 1 / 6); length = 6))
        h_grid = TC2.evaluate_topography(fu, theta, phi)
        @test size(h_grid) == (5, 6)
        @test all(x -> isapprox(x, a; atol = 1e-10), h_grid)

        # Add an m>0 mode to exercise the m!=0 direct-summation branch (cos/sin).
        f_lm = TC2.create_spherical_harmonic_topography(
            2, 1, 0.4, 1.0, GeoDynamo.OUTER_BOUNDARY; lmax = 4)
        h_lm = TC2.evaluate_topography(f_lm, theta, phi)
        @test size(h_lm) == (5, 6)
        @test all(isfinite, h_lm)
        # Real Y_2^1 ∝ sin(φ-pattern); has both positive and negative values.
        @test maximum(h_lm) > 0 && minimum(h_lm) < 0

        # --- compute_associated_legendre m>0 (P_1^1 and P_2^1) ---
        # P_1^1(x) = -sqrt(1-x^2)
        for x in (-0.5, 0.0, 0.6)
            @test TC2.compute_associated_legendre(1, 1, x) ≈ -sqrt(1 - x^2)
        end
        # P_2^1(x) = -3 x sqrt(1-x^2)
        for x in (-0.4, 0.3, 0.7)
            @test TC2.compute_associated_legendre(2, 1, x) ≈ -3 * x * sqrt(1 - x^2)
        end
        # P_2^2(x) = 3 (1-x^2)
        @test TC2.compute_associated_legendre(2, 2, 0.5) ≈ 3 * (1 - 0.5^2)
        # out-of-range m>l -> 0
        @test TC2.compute_associated_legendre(1, 2, 0.5) == 0.0

        # --- initialize_gaunt_cache! warns + returns nothing when no topography ---
        empty_td = TC2.TopographyData{Float64}()
        @test (@test_logs (:warn,) TC2.initialize_gaunt_cache!(empty_td, 4)) === nothing
        @test empty_td.gaunt_cache === nothing
    end

    # =====================================================================
    # stefan_condition.jl
    # =====================================================================
    @testset "stefan_condition.jl" begin
        lmax = 2
        st = TC2.StefanState(lmax = lmax, ri = 0.35, k_ic = 2.0, k_oc = 1.0)
        nlm = st.topography.nlm
        @test st.lambda ≈ 2.0
        @test !st.is_initialized

        # --- compute_stefan_flux: F = k_ic * flux_ic - k_oc * flux_oc ---
        st.heat_flux_ic .= [complex(Float64(i), 0.0) for i in 1:nlm]
        st.heat_flux_oc .= [complex(0.0, Float64(i)) for i in 1:nlm]
        flux = TC2.compute_stefan_flux(st)
        @test length(flux) == nlm
        for i in 1:nlm
            @test flux[i] ≈ complex(2.0 * i, -1.0 * i)  # 2*ic - 1*oc
        end

        # --- compute_stefan_flux_with_topography early returns ---
        # No coupling terms requested -> returns the base flux unchanged.
        td = TC2.TopographyData{Float64}()
        td.icb = st.topography
        gaunt = TC2.GauntTensorCache(lmax, lmax)
        cfg_off = TC2.TopographyCouplingConfig(
            enabled = true, include_slope_terms = false,
            include_shift_terms = false, epsilon = 0.02)
        flux_off = TC2.compute_stefan_flux_with_topography(
            st, st.topography, st.topography, td, gaunt, cfg_off)
        @test flux_off == flux

        # Coupling requested but ICB topography is nothing -> base flux.
        td_noicb = TC2.TopographyData{Float64}()
        cfg_on = TC2.TopographyCouplingConfig(
            enabled = true, include_slope_terms = true,
            include_shift_terms = true, epsilon = 0.02)
        flux_noicb = TC2.compute_stefan_flux_with_topography(
            st, st.topography, st.topography, td_noicb, gaunt, cfg_on)
        @test flux_noicb == flux

        # --- get_spectral_coefficient on a non-spectral object returns 0 ---
        @test TC2.get_spectral_coefficient(42, 1, 0) == zero(ComplexF64)

        # --- get_stefan_diagnostics: known RMS values ---
        st.normal_velocity .= [complex(1.0, 0.0) for _ in 1:nlm]
        diag = TC2.get_stefan_diagnostics(st)
        @test diag["lambda"] ≈ 2.0
        @test diag["is_initialized"] == false
        @test diag["normal_velocity_rms"] ≈ sqrt(Float64(nlm))  # |1|^2 summed
        @test diag["heat_flux_rms"] ≈ sqrt(sum(abs2, flux))
        @test diag["stefan_number"] ≈ st.Stefan

        # --- compute_boundary_heat_flux_spectral / compute_normal_velocity_spectral
        #     on objects without the expected fields return empty vectors ---
        @test TC2.compute_boundary_heat_flux_spectral(42, 0.35, :inner) == ComplexF64[]
        @test TC2.compute_normal_velocity_spectral(42, 0.35, GeoDynamo.INNER_BOUNDARY) ==
              ComplexF64[]
    end

    # =====================================================================
    # velocity_coupling.jl  (closed-form coupling-term assertions)
    # =====================================================================
    @testset "velocity_coupling.jl" begin
        T = Float64
        lmax = 2
        mmax = 2
        rb = 1.0

        # nlm in (l,m) triangular indexing for m>=0, lmax=2: (0,0)(1,0)(1,1)(2,0)(2,1)(2,2)=6
        nlm = 6
        @test TC2.lm_to_index(2, 2, lmax) == nlm

        # Build a BoundaryDerivativeCache directly with a single nonzero field mode.
        # Choose field mode (l'=1, m'=0); set P value and dP/dr at the OUTER boundary.
        make_cache = function (; value_outer, d1_outer)
            vi = zeros(Complex{T}, nlm)
            vo = zeros(Complex{T}, nlm)
            d1i = zeros(Complex{T}, nlm)
            d1o = zeros(Complex{T}, nlm)
            idx10 = TC2.lm_to_index(1, 0, lmax)
            vo[idx10] = value_outer
            d1o[idx10] = d1_outer
            return TC2.BoundaryDerivativeCache{T}(
                lmax, mmax, nlm, nothing, vi, vo, d1i, d1o, nothing, nothing)
        end

        # Topography with a single mode h_{1,0} = h_amp.
        h_amp = 0.3
        topo = TC2.TopographyField(lmax, mmax, rb, GeoDynamo.OUTER_BOUNDARY)
        tc = zeros(ComplexF64, topo.nlm)
        tc[TC2.lm_to_index(1, 0, lmax)] = h_amp
        TC2.set_topography_coefficients!(topo, tc)

        # Gaunt cache with exactly the entries needed for target mode (l=2,m=0)
        # coupling field (l'=1,m'=0) through topo (L=1,M=0).
        gaunt = TC2.GauntTensorCache(lmax, lmax)
        l, m = 2, 0
        lp, mp = 1, 0
        L, M = 1, 0
        Gb = TC2.gaunt_on_the_fly(l, m, lp, mp, L, M; use_wigner = true)
        Ggrad = TC2.gradient_gaunt_from_basic(l, lp, L, Gb)
        gaunt.G[(l, m, lp, mp, L, M)] = Gb
        gaunt.G_∇[(l, m, lp, mp, L, M)] = Ggrad
        @test abs(Gb) > 1e-12   # this coupling is non-trivial

        P_val = 0.5
        dP_dr = 0.2

        # ---- stress-free correction (slope term, poloidal only) ----
        cfg_slope = TC2.TopographyCouplingConfig(
            enabled = true, include_slope_terms = true,
            include_shift_terms = false, epsilon = 0.02)
        pc = make_cache(value_outer = complex(P_val, 0.0), d1_outer = complex(dP_dr, 0.0))
        sf = TC2.compute_stressfree_correction(
            l, m, pc, topo, gaunt, rb, GeoDynamo.OUTER_BOUNDARY, cfg_slope)
        ll = lp * (lp + 1)
        dur_dr = ll * (dP_dr / rb^2 - 2 * P_val / rb^3)
        sf_expected = h_amp * Ggrad * dur_dr / rb^2
        @test sf ≈ complex(sf_expected, 0.0)

        # slope terms disabled -> zero
        cfg_noslope = TC2.TopographyCouplingConfig(
            enabled = true, include_slope_terms = false, epsilon = 0.02)
        @test TC2.compute_stressfree_correction(
            l, m, pc, topo, gaunt, rb, GeoDynamo.OUTER_BOUNDARY, cfg_noslope) ==
              zero(ComplexF64)

        # ---- no-slip correction (toroidal shift via G * dT/dr) ----
        dT_dr = 0.4
        tc_cache = make_cache(value_outer = complex(0.0, 0.0), d1_outer = complex(dT_dr, 0.0))
        cfg_shift = TC2.TopographyCouplingConfig(
            enabled = true, include_shift_terms = true,
            include_slope_terms = false, epsilon = 0.02)
        Pcorr, Tcorr = TC2.compute_noslip_correction(
            l, m, pc, tc_cache, topo, gaunt, rb, GeoDynamo.OUTER_BOUNDARY, cfg_shift)
        @test Pcorr == zero(ComplexF64)   # no-slip never corrects poloidal
        @test Tcorr ≈ complex(h_amp * Gb * dT_dr, 0.0)

        # shift disabled -> both zero
        Pc2, Tc2 = TC2.compute_noslip_correction(
            l, m, pc, tc_cache, topo, gaunt, rb, GeoDynamo.OUTER_BOUNDARY,
            TC2.TopographyCouplingConfig(enabled = true, include_shift_terms = false))
        @test Pc2 == zero(ComplexF64) && Tc2 == zero(ComplexF64)

        # ---- impermeability correction: shift term only ----
        # shift_contrib = G * l'(l'+1) * (dP_dr/rb^2 - 2 P/rb^3); correction = h*shift.
        cfg_imp_shift = TC2.TopographyCouplingConfig(
            enabled = true, include_shift_terms = true,
            include_slope_terms = false, epsilon = 0.02)
        imp_shift = TC2.compute_impermeability_correction(
            l, m, pc, tc_cache, topo, gaunt, rb, GeoDynamo.OUTER_BOUNDARY, cfg_imp_shift)
        shift_contrib = Gb * ll * (dP_dr / rb^2 - 2 * P_val / rb^3)
        @test imp_shift ≈ complex(h_amp * shift_contrib, 0.0)

        # ---- impermeability correction: slope terms (poloidal + toroidal) ----
        # Need a cross-Gaunt entry for the toroidal contribution.
        Gcross = TC2.compute_cross_gaunt_tensor(l, m, lp, mp, L, M, gaunt)
        gaunt.G_cross[(l, m, lp, mp, L, M)] = Gcross
        # Put both P and T values into a single cache so both slope branches fire.
        idx10 = TC2.lm_to_index(1, 0, lmax)
        vi = zeros(Complex{T}, nlm)
        vo = zeros(Complex{T}, nlm)
        d1i = zeros(Complex{T}, nlm)
        d1o = zeros(Complex{T}, nlm)
        vo[idx10] = complex(P_val, 0.0)
        d1o[idx10] = complex(dP_dr, 0.0)
        p_only = TC2.BoundaryDerivativeCache{T}(
            lmax, mmax, nlm, nothing, vi, vo, d1i, d1o, nothing, nothing)
        T_val = 0.6
        vot = zeros(Complex{T}, nlm)
        vot[idx10] = complex(T_val, 0.0)
        t_only = TC2.BoundaryDerivativeCache{T}(
            lmax, mmax, nlm, nothing, zeros(Complex{T}, nlm), vot,
            zeros(Complex{T}, nlm), zeros(Complex{T}, nlm), nothing, nothing)
        cfg_imp_slope = TC2.TopographyCouplingConfig(
            enabled = true, include_shift_terms = false,
            include_slope_terms = true, epsilon = 0.02)
        imp_slope = TC2.compute_impermeability_correction(
            l, m, p_only, t_only, topo, gaunt, rb, GeoDynamo.OUTER_BOUNDARY, cfg_imp_slope)
        slope_P = -Ggrad * (dP_dr / rb^2)
        expected_slope = h_amp * slope_P
        if abs(Gcross) > 1e-15
            expected_slope += h_amp * (-Gcross * T_val / rb^2)
        end
        @test imp_slope ≈ complex(expected_slope, 0.0)

        # --- assemble_velocity_boundary_operator diagonal (impermeability) ---
        op = TC2.assemble_velocity_boundary_operator(
            2, topo, gaunt, cfg_imp_shift, :impermeability)
        @test op[(2, 0, :P)] ≈ complex(T(2 * 3) / rb^2, 0.0)
        @test all(haskey(op, (2, mm, :P)) for mm in -2:2)
    end

    # =====================================================================
    # file_bc_loader.jl
    # =====================================================================
    @testset "file_bc_loader.jl" begin
        # --- BoundaryInterpolationCache dict-protocol (per-key haskey/getindex) ---
        cache = BCS2.BoundaryInterpolationCache(Float64)
        @test isempty(cache)
        @test length(cache) == 0
        @test !haskey(cache, "bc_real")
        @test !haskey(cache, "bc_source_file")
        @test cache["bc_loaded"] == false          # special key always present
        @test get(cache, "missing", :dflt) == :dflt

        cache["bc_real"] = [1.0 2.0; 3.0 4.0]
        cache["bc_imag"] = [0.0 0.0; 0.0 0.0]
        cache["bc_loaded"] = true
        cache["bc_source_file"] = "src.nc"
        cache["bc_source_format"] = :spectral
        cache["bc_nlm"] = 2
        cache["custom"] = 99                        # routes to metadata dict

        @test haskey(cache, "bc_real")
        @test haskey(cache, "bc_imag")
        @test haskey(cache, "bc_source_file")
        @test haskey(cache, "bc_source_format")
        @test haskey(cache, "bc_nlm")
        @test haskey(cache, "custom")
        @test cache["bc_real"] == [1.0 2.0; 3.0 4.0]
        @test cache["bc_source_file"] == "src.nc"
        @test cache["bc_source_format"] == :spectral
        @test cache["bc_nlm"] == 2
        @test cache["custom"] == 99
        @test !isempty(cache)
        # length counts the populated special keys + metadata entries
        @test length(cache) == 7
        # iterate protocol returns Pairs
        collected = collect(cache)
        @test ("bc_loaded" => true) in collected
        @test ("custom" => 99) in collected

        # getindex on an absent matrix key throws KeyError
        empty2 = BCS2.BoundaryInterpolationCache(Float64)
        @test_throws KeyError empty2["bc_real"]

        # empty! resets everything
        empty!(cache)
        @test isempty(cache)
        @test cache["bc_loaded"] == false

        # --- _get_bc_vectors_from_cache on a plain Dict (AbstractDict path) ---
        nlm = 3
        dict_cache = Dict{String, Any}(
            "bc_loaded" => true,
            "bc_real" => reshape(collect(1.0:(2 * nlm)), 2, nlm),
            "bc_imag" => -reshape(collect(1.0:(2 * nlm)), 2, nlm))
        vecs = BCS2._get_bc_vectors_from_cache(dict_cache)
        @test vecs.inner_real == dict_cache["bc_real"][1, :]
        @test vecs.outer_real == dict_cache["bc_real"][2, :]
        @test vecs.inner_imag == dict_cache["bc_imag"][1, :]
        # not loaded -> all nothing
        novecs = BCS2._get_bc_vectors_from_cache(Dict{String, Any}("bc_loaded" => false))
        @test novecs.inner_real === nothing && novecs.outer_imag === nothing

        # --- physical single-variable "temperature" / "composition" formats ---
        cfg = GeoDynamo.create_shtnskit_config(
            lmax = 2, mmax = 2, nlat = 8, nlon = 16, nr = 4)
        mktempdir() do dir
            # temperature variable with 3rd dim of size 2 (zeros -> zero coeffs)
            f_t = joinpath(dir, "temp_single.nc")
            NCDataset(f_t, "c") do ds
                defDim(ds, "theta", cfg.nlat)
                defDim(ds, "phi", cfg.nlon)
                defDim(ds, "side", 2)
                defVar(ds, "temperature", Float64, ("theta", "phi", "side"))
                ds["temperature"][:, :, :] = zeros(cfg.nlat, cfg.nlon, 2)
            end
            coeffs_t = BCS2.load_spectral_bc_from_file(f_t, cfg; format = :physical)
            @test coeffs_t.source_format == :physical
            @test size(coeffs_t.bc_real) == (2, cfg.nlm)
            @test all(iszero, coeffs_t.bc_real)
            @test all(iszero, coeffs_t.bc_imag)

            # composition variable, same shape
            f_c = joinpath(dir, "comp_single.nc")
            NCDataset(f_c, "c") do ds
                defDim(ds, "theta", cfg.nlat)
                defDim(ds, "phi", cfg.nlon)
                defDim(ds, "side", 2)
                defVar(ds, "composition", Float64, ("theta", "phi", "side"))
                ds["composition"][:, :, :] = zeros(cfg.nlat, cfg.nlon, 2)
            end
            coeffs_c = BCS2.load_spectral_bc_from_file(f_c, cfg; format = :physical)
            @test all(iszero, coeffs_c.bc_real)

            # unknown format -> ArgumentError
            @test_throws ArgumentError BCS2.load_spectral_bc_from_file(
                f_t, cfg; format = :bogus)
            # missing file -> ArgumentError
            @test_throws ArgumentError BCS2.load_spectral_bc_from_file(
                joinpath(dir, "nope.nc"), cfg; format = :physical)

            # spectral file with a wrong shape -> ArgumentError
            f_bad = joinpath(dir, "spec_bad.nc")
            NCDataset(f_bad, "c") do ds
                defDim(ds, "a", 3)
                defDim(ds, "b", 3)
                defVar(ds, "Cbc_Re", Float64, ("a", "b"))
                defVar(ds, "Cbc_Im", Float64, ("a", "b"))
                ds["Cbc_Re"][:, :] = zeros(3, 3)
                ds["Cbc_Im"][:, :] = zeros(3, 3)
            end
            @test_throws ArgumentError BCS2.load_spectral_bc_from_file(
                f_bad, cfg; format = :spectral)
        end
    end

    # =====================================================================
    # netcdf_io.jl
    # =====================================================================
    @testset "netcdf_io.jl" begin
        mktempdir() do dir
            # --- time-dependent + multi-component write/read/info round-trip ---
            nlat, nlon, ntime, ncomp = 4, 5, 3, 2
            theta = collect(range(0.0, π; length = nlat))
            phi = collect(range(0.0, 2π * (1 - 1 / nlon); length = nlon))
            time = collect(0.0:1.0:(ntime - 1))
            vals = reshape(collect(Float64, 1:(nlat * nlon * ntime * ncomp)),
                nlat, nlon, ntime, ncomp)
            bdata = BCS2.create_boundary_data(
                vals, "velocity";
                theta = theta, phi = phi, time = time,
                units = "m/s", description = "td vector boundary")
            @test bdata.is_time_dependent
            @test bdata.ntime == ntime
            @test bdata.ncomponents == ncomp

            f_td = joinpath(dir, "td_velocity.nc")
            BCS2.write_netcdf_boundary_data(f_td, bdata)
            @test isfile(f_td)
            @test BCS2.validate_netcdf_boundary_file(f_td)

            loaded = BCS2.read_netcdf_boundary_data(f_td)
            @test loaded.field_type == "velocity"
            @test loaded.units == "m/s"
            @test loaded.is_time_dependent
            @test loaded.values == vals
            @test loaded.time ≈ time

            info = BCS2.get_netcdf_file_info(f_td)
            @test info["is_time_dependent"] == true
            @test info["ntime"] == ntime
            @test info["time_range"] == [minimum(time), maximum(time)]
            @test info["field_variable"] == "velocity"
            @test info["field_shape"] == size(vals)
            @test info["field_units"] == "m/s"
            @test info["field_description"] == "td vector boundary"

            # --- validator: phi out of range -> ArgumentError ---
            f_badphi = joinpath(dir, "badphi.nc")
            NCDataset(f_badphi, "c") do ds
                defDim(ds, "theta", 2)
                defDim(ds, "phi", 2)
                defVar(ds, "theta", Float64, ("theta",))
                defVar(ds, "phi", Float64, ("phi",))
                defVar(ds, "temperature", Float64, ("theta", "phi"))
                ds["theta"][:] = [0.0, π]
                ds["phi"][:] = [0.0, 2π + 0.5]      # >= 2π violates range
                ds["temperature"][:, :] = zeros(2, 2)
            end
            @test_throws ArgumentError BCS2.validate_netcdf_boundary_file(f_badphi)

            # --- validator: missing required extra variable -> ArgumentError ---
            f_ok = joinpath(dir, "ok.nc")
            NCDataset(f_ok, "c") do ds
                defDim(ds, "theta", 2)
                defDim(ds, "phi", 2)
                defVar(ds, "theta", Float64, ("theta",))
                defVar(ds, "phi", Float64, ("phi",))
                defVar(ds, "temperature", Float64, ("theta", "phi"))
                ds["theta"][:] = [0.0, π]
                ds["phi"][:] = [0.0, π]
                ds["temperature"][:, :] = zeros(2, 2)
            end
            @test BCS2.validate_netcdf_boundary_file(f_ok)
            @test_throws ArgumentError BCS2.validate_netcdf_boundary_file(
                f_ok, ["nonexistent_var"])

            # --- error paths: missing file ---
            @test_throws ArgumentError BCS2.read_netcdf_boundary_data(
                joinpath(dir, "nope.nc"))
            @test_throws ArgumentError BCS2.get_netcdf_file_info(
                joinpath(dir, "nope.nc"))
            @test_throws ArgumentError BCS2.validate_netcdf_boundary_file(
                joinpath(dir, "nope.nc"))
        end
    end

    # =====================================================================
    # io/restart.jl  (pencils !== nothing read paths at a single rank)
    # =====================================================================
    @testset "io/restart.jl pencils path" begin
        if !MPI.Initialized()
            MPI.Init()
        end
        comm = GeoDynamo.output_comm()

        parallel_ok = let probe_err = GeoDynamo.parallel_netcdf_probe(comm)
            probe_err === nothing ||
                @warn "Parallel NetCDF unavailable; skipping restart pencils round-trip" error = probe_err
            probe_err === nothing
        end

        if parallel_ok && MPI.Comm_size(comm) == 1
            lmax = 4
            mmax = 4
            nlat = max(lmax + 2, 10)
            nlon = max(2lmax + 1, 16)
            nr = 8

            cfg = GeoDynamo.create_shtnskit_config(
                lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
            dom = GeoDynamo.create_radial_domain(nr)
            radial_nodes = Float64.(dom.r[1:nr, 4])
            nlm = cfg.nlm

            Random.seed!(20260604)
            T_in = randn(cfg.nlat, cfg.nlon, nr)
            C_in = randn(cfg.nlat, cfg.nlon, nr)
            btor_real = randn(nlm, nr)
            btor_imag = randn(nlm, nr)

            fields = Dict{String, Any}(
                "temperature" => copy(T_in),
                "composition" => copy(C_in),
                "magnetic_toroidal" =>
                    Dict("real" => copy(btor_real), "imag" => copy(btor_imag)))

            tmpdir = mktempdir()
            base = GeoDynamo.default_config(Float64)
            config = GeoDynamo.OutputConfig(
                base.output_space, tmpdir, base.filename_prefix,
                base.include_metadata, base.include_grid, base.include_diagnostics,
                Float64, base.spectral_lmax_output, true,
                base.output_interval, base.restart_interval,
                base.max_output_time, base.time_tolerance)

            tracker = GeoDynamo.create_time_tracker(config, 0.0)
            tracker.last_output_time = 2.0
            tracker.output_count = 5
            tracker.restart_count = 0
            tracker.grid_file_written = true
            metadata = Dict{String, Any}(
                "current_time" => 3.0, "current_step" => 7)

            GeoDynamo.write_restart!(fields, tracker, metadata, config, cfg.pencils;
                shtns_config = cfg, geometry = :shell,
                radius_ratio = 0.35, radial_grid = radial_nodes)

            found = GeoDynamo.find_restart_files(tmpdir, 0.0)
            @test length(found) == 1

            # read_restart! with pencils + shtns_config -> spectral pencils branch
            reader = GeoDynamo.create_time_tracker(config, 0.0)
            rdata, md = GeoDynamo.read_restart!(
                reader, tmpdir, 3.0, config, cfg.pencils; shtns_config = cfg)

            @test md["current_step"] == 7
            @test reader.output_count == 5
            @test reader.last_restart_time ≈ 3.0

            # Physical fields round-trip exactly at a single rank (full ranges).
            @test haskey(rdata, "temperature")
            @test size(rdata["temperature"]) == size(T_in)
            @test maximum(abs.(rdata["temperature"] .- T_in)) == 0.0
            @test maximum(abs.(rdata["composition"] .- C_in)) == 0.0

            # Spectral field via the pencils+shtns_config unpack path: the returned
            # local storage is (nlm, 1, nr); at one rank it preserves every mode,
            # so the total spectral energy matches the written coefficients.
            # The unpack returns local spectral STORAGE (2D-spectral-pencil layout
            # × nr), so dim-1 is the slot axis, not nlm; the radial axis is nr.
            @test haskey(rdata, "magnetic_toroidal")
            mr = rdata["magnetic_toroidal"]["real"]
            mi = rdata["magnetic_toroidal"]["imag"]
            @test ndims(mr) == 3
            @test size(mr, 3) == nr
            @test all(isfinite, mr) && all(isfinite, mi)
            # At one rank every (l,m) mode is preserved, so the total spectral
            # energy equals that of the written coefficients (slot reordering only).
            @test sum(abs2, mr)≈sum(abs2, btor_real) rtol=1e-10
            @test sum(abs2, mi)≈sum(abs2, btor_imag) rtol=1e-10

            # _load_restart_file with explicit path + pencils + shtns_config.
            reader2 = GeoDynamo.create_time_tracker(config, 0.0)
            rdata2, md2 = GeoDynamo._load_restart_file(
                found[1], reader2, config; pencils = cfg.pencils, shtns_config = cfg)
            @test md2["current_step"] == 7
            @test maximum(abs.(rdata2["temperature"] .- T_in)) == 0.0
            @test maximum(abs.(rdata2["composition"] .- C_in)) == 0.0
            @test sum(abs2, rdata2["magnetic_toroidal"]["real"])≈sum(abs2, btor_real) rtol=1e-10
        else
            @info "Skipping restart pencils round-trip (needs parallel NetCDF, single rank)"
        end
    end
end
