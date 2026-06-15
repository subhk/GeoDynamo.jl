using Test
using MPI
using LinearAlgebra
using Statistics
using GeoDynamo
const G = GeoDynamo

# ============================================================================
# Regression: set_analytical_magnetic!(:uniform_field) must produce an actually
# uniform field in the requested Cartesian direction.
#
# Historical bug (core/initial_conditions.jl):
#   :z used a POLOIDAL l = 0 mode → B_r ∝ l(l+1) = 0 → identically ZERO field.
#   :x used the (l = 1, m = 0) mode → an AXIAL (z) field, not x.
#
# Correct construction (synthesis convention B_r = l(l+1)·P/r², S = ∂_r P / r):
#   a uniform field is the POLOIDAL l = 1 mode with radial profile P(r) ∝ r²
#   (then B_r and B_θ are radius-independent ⇒ a constant Cartesian field).
#     :z → (l, m) = (1, 0):  B_r ∝ cosθ
#     :x → (l, m) = (1, 1) real part:  B_r ∝ sinθ·cosφ
#
# These checks are normalization-INDEPENDENT (they test uniformity + axiality),
# except the :z magnitude check, which uses the derived C = (amp/2)·√(4π/3) that
# makes B_z = amplitude under the orthonormal Y_1^0 = √(3/4π)·cosθ convention.
# ============================================================================

# Convert a spherical-component vector field (B_r, B_θ, B_φ) on the LOCAL grid
# of `vec` into flat Cartesian (Bx, By, Bz) arrays, restricted to interior radii
# (the banded radial-derivative endpoint stencils degrade S = ∂_r P / r at the
# first/last levels). Uses the component pencil's axes_local so the global θ/φ/r
# indices are correct under MPI as well as serially.
function _cartesian_components(cfg, vec; r_skip = 2)
    Br = parent(vec.r_component.data)
    Bθ = parent(vec.θ_component.data)
    Bφ = parent(vec.φ_component.data)
    ax = vec.r_component.pencil.axes_local      # (θ_range, φ_range, r_range), global indices
    θs = cfg.theta_grid                          # colatitude θ ∈ [0, π]
    φs = cfg.phi_grid                            # longitude φ ∈ [0, 2π)
    nr_global = maximum(ax[3])                    # radial axis is r-local on current main

    Bx = Float64[]; By = Float64[]; Bz = Float64[]
    for (kl, kg) in enumerate(ax[3])
        # interior radii only (drop the first/last `r_skip` global levels)
        (kg <= r_skip || kg > nr_global - r_skip) && continue
        for (jl, jg) in enumerate(ax[2])
            φ = φs[jg]; sφ = sin(φ); cφ = cos(φ)
            for (il, ig) in enumerate(ax[1])
                θ = θs[ig]; sθ = sin(θ); cθ = cos(θ)
                br = Br[il, jl, kl]; bθ = Bθ[il, jl, kl]; bφ = Bφ[il, jl, kl]
                push!(Bx, br * sθ * cφ + bθ * cθ * cφ - bφ * sφ)
                push!(By, br * sθ * sφ + bθ * cθ * sφ + bφ * cφ)
                push!(Bz, br * cθ - bθ * sθ)
            end
        end
    end
    return Bx, By, Bz
end

@testset "uniform_field analytical magnetic IC" begin
    MPI.Initialized() || MPI.Init()

    lmax = 6; mmax = 6
    nlat = 2 * lmax + 4
    nlon = 4 * lmax + 8
    nr   = 24

    cfg   = G.create_shtnskit_config(lmax = lmax, mmax = mmax,
                                     nlat = nlat, nlon = nlon, nr = nr)
    shell = G.create_radial_domain(nr)           # shell, r ∈ [~0.54, ~1.54]

    # ---- :z (axial) ---------------------------------------------------------
    @testset "direction = :z is uniform and axial" begin
        B0  = 2.5
        mag = G.create_shtns_magnetic_fields(Float64, cfg, shell, shell)
        G.set_analytical_initial_conditions!(mag, :magnetic, :uniform_field;
            amplitude = B0, direction = :z, domain = shell)

        # The poloidal storage must be NONZERO (the old l=0 path left it all zero).
        @test maximum(abs, parent(mag.poloidal.data_real)) > 0.0

        G.vector_spectral_to_physical!(mag.toroidal, mag.poloidal, mag.magnetic;
            domain = shell)

        Bx, By, Bz = _cartesian_components(cfg, mag.magnetic)
        @test !isempty(Bz)
        bz_mean = mean(Bz)
        bscale  = abs(bz_mean)

        @test bscale > 1e-6                                  # field is nonzero (catches old l=0 ≡ 0 bug)
        @test maximum(abs, Bz .- bz_mean) < 1e-3 * bscale    # B_z uniform across all points
        @test maximum(abs, Bx) < 1e-3 * bscale               # axial: no x component
        @test maximum(abs, By) < 1e-3 * bscale               # axial: no y component
        @test isapprox(bz_mean, B0; rtol = 1e-2)             # magnitude matches amplitude
    end

    # ---- :x -----------------------------------------------------------------
    @testset "direction = :x is uniform and along x" begin
        B0  = 1.3
        mag = G.create_shtns_magnetic_fields(Float64, cfg, shell, shell)
        G.set_analytical_initial_conditions!(mag, :magnetic, :uniform_field;
            amplitude = B0, direction = :x, domain = shell)

        @test maximum(abs, parent(mag.poloidal.data_real)) > 0.0

        G.vector_spectral_to_physical!(mag.toroidal, mag.poloidal, mag.magnetic;
            domain = shell)

        Bx, By, Bz = _cartesian_components(cfg, mag.magnetic)
        @test !isempty(Bx)
        bx_mean = mean(Bx)
        bscale  = abs(bx_mean)

        # Magnitude is left normalization-free for the m=1 mode (a uniform field
        # along x is exactly the (1,1) real harmonic regardless of the SHTnsKit
        # m=1 synthesis constant); we assert uniformity + axiality, not |Bx|=B0.
        @test bscale > 1e-6                                  # nonzero
        @test maximum(abs, Bx .- bx_mean) < 1e-3 * bscale    # B_x uniform
        @test maximum(abs, By) < 1e-3 * bscale               # no y component
        @test maximum(abs, Bz) < 1e-3 * bscale               # no z component (old bug gave an axial field here)
    end

    # ---- unsupported direction is a hard error ------------------------------
    @testset "unsupported direction errors" begin
        mag = G.create_shtns_magnetic_fields(Float64, cfg, shell, shell)
        @test_throws ArgumentError G.set_analytical_initial_conditions!(
            mag, :magnetic, :uniform_field; amplitude = 1.0, direction = :y, domain = shell)
    end
end
