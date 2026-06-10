using Test
using MPI
using LinearAlgebra
using Random
using GeoDynamo

MPI.Initialized() || MPI.Init()

# ---------------------------------------------------------------------------
# Fixture constants (duplicated from force_projection_reference.jl)
# ---------------------------------------------------------------------------
const ST_NR   = 48
const ST_LMAX = 8

function _st_setup()
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = ST_LMAX, mmax = ST_LMAX,
        nlat = 2 * ST_LMAX + 4, nlon = 4 * ST_LMAX + 8,
        nr   = ST_NR)
    dom = GeoDynamo.create_radial_domain(ST_NR)   # shell domain, radius_ratio=0.35 default
    return cfg, dom
end

_st_spec(cfg, dom) = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
_st_vec(cfg, dom)  = GeoDynamo.create_shtns_vector_field(Float64, cfg, dom,
    (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))
_st_phys(cfg, dom) = GeoDynamo.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)

# ---------------------------------------------------------------------------
# Harness 1 (duplicated from _fp_angular_derivs): exact angular derivatives
# via sphtor spectral synthesis.
#
# Returns (dθ_bare, dφ_oversin):
#   dθ_bare    = ∂θ g   (bare colatitude derivative)
#   dφ_oversin = (1/sinθ)·∂φ g
#
# Mechanism: T=0, S=spectral(g) → vector_spectral_to_physical! (sphtor).
# Exact for band-limited fields.
# ---------------------------------------------------------------------------
function _st_angular_derivs(cfg, dom, g_phys::GeoDynamo.SHTnsPhysField)
    g_spec = _st_spec(cfg, dom)
    GeoDynamo.scalar_physical_to_spectral!(g_phys, g_spec)

    zero_spec = _st_spec(cfg, dom)   # all-zero toroidal
    V = _st_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(zero_spec, g_spec, V; domain = nothing)

    # V.θ_component = ∂θg  (bare)
    # V.φ_component = (1/sinθ)∂φg
    return V.θ_component, V.φ_component
end

# ---------------------------------------------------------------------------
# Harness 2 (duplicated from _fp_radial_deriv): radial derivative via banded
# finite-difference operator.  Assumes radial dimension is fully local.
# ---------------------------------------------------------------------------
function _st_radial_deriv(cfg, dom, g_phys)
    D1      = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    nr      = dom.N
    arr_in  = parent(g_phys.data)
    out     = _st_phys(cfg, dom)
    arr_out = parent(out.data)

    prof  = Vector{Float64}(undef, nr)
    dprof = Vector{Float64}(undef, nr)

    for j in 1:size(arr_in, 2), i in 1:size(arr_in, 1)
        for k in 1:nr
            prof[k] = arr_in[i, j, k]
        end
        mul!(dprof, D1, prof)
        for k in 1:nr
            arr_out[i, j, k] = dprof[k]
        end
    end

    return out
end

# ---------------------------------------------------------------------------
# Physical-space divergence reference
#
# ∇·u = (1/r²)∂_r(r²·u_r) + (1/(r·sinθ))∂θ(sinθ·u_θ) + (1/(r·sinθ))∂φ(u_φ)
#
# Steps:
#   (A) r²·u_r  → banded D1  → ∂_r(r²·u_r)
#   (B) sinθ·u_θ → sphtor synthesis → ∂_θ(sinθ·u_θ)
#   (C) u_φ     → sphtor synthesis → (1/sinθ)·∂_φ(u_φ)
#
# Combined: A/r² + (B/sinθ + C)/r
# ---------------------------------------------------------------------------
function _st_divergence(cfg, dom, V)
    r_range = GeoDynamo.range_local(cfg.pencils.r, 3)
    sinθ    = sin.(cfg.theta_grid)

    # (A) r²·u_r and its radial derivative
    r2ur = _st_phys(cfg, dom)
    a    = parent(r2ur.data)
    ur   = parent(V.r_component.data)
    for k in axes(a, 3), j in 1:cfg.nlon, i in 1:cfg.nlat
        r         = dom.r[k + first(r_range) - 1, 4]
        a[i, j, k] = r^2 * ur[i, j, k]
    end
    d_r2ur = _st_radial_deriv(cfg, dom, r2ur)

    # (B) sinθ·u_θ and its colatitude derivative
    sut = _st_phys(cfg, dom)
    b   = parent(sut.data)
    uθ  = parent(V.θ_component.data)
    for k in axes(b, 3), j in 1:cfg.nlon, i in 1:cfg.nlat
        b[i, j, k] = sinθ[i] * uθ[i, j, k]
    end
    dθ_sut, _ = _st_angular_derivs(cfg, dom, sut)

    # (C) (1/sinθ)·∂_φ(u_φ) directly from sphtor synthesis of u_φ
    _, dφ_uφ = _st_angular_derivs(cfg, dom, V.φ_component)

    # Assemble divergence
    out    = _st_phys(cfg, dom)
    o      = parent(out.data)
    A      = parent(d_r2ur.data)
    B      = parent(dθ_sut.data)
    C      = parent(dφ_uφ.data)
    for k in axes(o, 3), j in 1:cfg.nlon, i in 1:cfg.nlat
        r         = dom.r[k + first(r_range) - 1, 4]
        o[i, j, k] = A[i, j, k] / r^2 + (B[i, j, k] / sinθ[i] + C[i, j, k]) / r
    end

    return out
end

# ===========================================================================
# Gate test: compute_divergence_spectral is a real diagnostic
# ===========================================================================
@testset "compute_divergence_spectral is real (not a stub)" begin
    cfg, dom = _st_setup()
    Random.seed!(21)

    tor = _st_spec(cfg, dom)
    pol = _st_spec(cfg, dom)

    for spec in (tor, pol)
        sr = parent(spec.data_real)
        si = parent(spec.data_imag)
        for lm in 1:cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]
            m = cfg.m_values[lm]
            (1 <= l <= 6) || continue
            for r_idx in 1:dom.N
                x = (dom.r[r_idx, 4] - dom.r[1, 4]) / (dom.r[dom.N, 4] - dom.r[1, 4])
                v = sinpi(x) * randn() * 1e-2
                GeoDynamo.set_local_spectral_value!(sr, slot, r_idx, v)
                m > 0 && GeoDynamo.set_local_spectral_value!(si, slot, r_idx, 0.7v)
            end
        end
    end

    # Call the function under test
    l2, linf = GeoDynamo.compute_divergence_spectral(tor, pol, dom)

    # Build an independent physical-space reference
    V = _st_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom)
    div_phys  = _st_divergence(cfg, dom, V)
    dd        = parent(div_phys.data)
    ref_linf  = maximum(abs, dd[:, :, 2:(size(dd, 3) - 1)])   # interior (D1 endpoints)

    @info "divergence gate" l2 linf ref_linf

    # (a) must not be the stub when the field is genuinely non-solenoidal
    @test !(l2 == 0.0 && linf == 0.0) || ref_linf < 1e-10

    # (b) order-of-magnitude agreement with the physical-space reference
    @test isapprox(linf, ref_linf; rtol = 0.5) || (linf < 1e-10 && ref_linf < 1e-10)
end
