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
# ⚠ Grid-space divergence is ALIASED for m>0 content: scalar analysis of the
# tangential components (1/sinθ structure) spreads beyond lmax and truncates.
# Use ONLY on m=0-only fields (aliasing-free); the spectral helper below is the
# general-purpose reference.
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

# ---------------------------------------------------------------------------
# Spectral divergence reference: QST-analyze the field (Stage-1-verified
# machinery), then per-mode  div_lm(r) = (1/r²)∂_r(r²Q_lm) − l(l+1)·S_lm/r.
# Exact for band-limited fields (no grid-space aliasing); interior nodes only.
# Returns (l2, linf) over modes × interior radii.
# ---------------------------------------------------------------------------
function _st_spectral_divergence(cfg, dom, V)
    Q = _st_spec(cfg, dom); S = _st_spec(cfg, dom); T_ = _st_spec(cfg, dom)
    GeoDynamo.force_physical_to_qst!(V, Q, S, T_)
    D1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    nr = dom.N
    r2q = Vector{Float64}(undef, nr); d = Vector{Float64}(undef, nr)
    sumsq = 0.0; linf = 0.0; count = 0
    for lm in 1:cfg.nlm
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        l = cfg.l_values[lm]; λ = l * (l + 1)
        for (qsrc, ssrc) in ((parent(Q.data_real), parent(S.data_real)),
            (parent(Q.data_imag), parent(S.data_imag)))
            for r_idx in 1:nr
                r = dom.r[r_idx, 4]
                r2q[r_idx] = r^2 * GeoDynamo.local_spectral_value(qsrc, slot, r_idx)
            end
            mul!(d, D1, r2q)
            for r_idx in 2:(nr - 1)
                r = dom.r[r_idx, 4]
                dv = d[r_idx] / r^2 -
                     λ * GeoDynamo.local_spectral_value(ssrc, slot, r_idx) / r
                sumsq += dv^2; linf = max(linf, abs(dv)); count += 1
            end
        end
    end
    return (count > 0 ? sqrt(sumsq / count) : 0.0, linf)
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

    # Reference: same spectral formula computed in-test on the synthesized field
    V = _st_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom)
    ref_l2, ref_linf = _st_spectral_divergence(cfg, dom, V)

    @info "divergence gate" l2 linf ref_l2 ref_linf

    # (a) agreement with the in-test reference (pins the implementation)
    @test isapprox(linf, ref_linf; rtol = 1e-6, atol = 1e-14)
    @test isapprox(l2, ref_l2; rtol = 1e-6, atol = 1e-14)

    # (b) under the solenoidal convention any (T,P) synthesis is divergence-free:
    # the diagnostic must report ~0 — this is the Stage-2 acceptance through the
    # full synthesize→analyze chain.
    scale = maximum(abs, parent(V.r_component.data)) +
            maximum(abs, parent(V.θ_component.data)) +
            maximum(abs, parent(V.φ_component.data))
    @test linf < 1e-8 * scale
end

# ===========================================================================
# Stage-2 Task 2: the synthesis itself must be solenoidal under the new
# convention u_r = l(l+1)·P/r², S = (1/r)·∂_r(r·P).
# ===========================================================================
@testset "synthesized (T,P) field is solenoidal" begin
    cfg, dom = _st_setup()
    Random.seed!(23)
    tor = _st_spec(cfg, dom); pol = _st_spec(cfg, dom)
    for spec in (tor, pol)
        sr = parent(spec.data_real); si = parent(spec.data_imag)
        for lm in 1:cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]; m = cfg.m_values[lm]
            (1 <= l <= 6) || continue
            for r_idx in 1:dom.N
                x = (dom.r[r_idx, 4] - dom.r[1, 4]) / (dom.r[dom.N, 4] - dom.r[1, 4])
                v = sinpi(x) * randn() * 1e-2
                GeoDynamo.set_local_spectral_value!(sr, slot, r_idx, v)
                m > 0 && GeoDynamo.set_local_spectral_value!(si, slot, r_idx, 0.7v)
            end
        end
    end
    V = _st_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom)
    scale = maximum(abs, parent(V.r_component.data)) +
            maximum(abs, parent(V.θ_component.data)) +
            maximum(abs, parent(V.φ_component.data))
    _, maxdiv = _st_spectral_divergence(cfg, dom, V)
    @info "solenoidality" maxdiv scale
    @test maxdiv < 1e-8 * scale
end

@testset "verify_solenoidal rejects inconsistent radial content" begin
    cfg, dom = _st_setup()
    tor = _st_spec(cfg, dom)
    pol = _st_spec(cfg, dom)
    lm = findfirst(i -> cfg.l_values[i] == 2 && cfg.m_values[i] == 0, 1:cfg.nlm)
    slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
    if slot !== nothing
        for r_idx in 1:dom.N
            r = dom.r[r_idx, 4]
            GeoDynamo.set_local_spectral_value!(
                parent(pol.data_real), slot, r_idx, sinpi(r) * 1e-2)
        end
    end

    vec = _st_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(tor, pol, vec; domain = dom)
    tor_out = _st_spec(cfg, dom)
    pol_out = _st_spec(cfg, dom)

    # A field synthesized from (T,P) is accepted.
    @test GeoDynamo.vector_physical_to_spectral!(
        vec, tor_out, pol_out; domain = dom, verify_solenoidal = true) ==
          (tor_out, pol_out)

    # Change only u_r. Tangential S still describes the original P, so the
    # three components can no longer be one solenoidal vector field.
    radial = parent(vec.r_component.data)
    radial .*= 1.25
    @test_throws ArgumentError GeoDynamo.vector_physical_to_spectral!(
        vec, tor_out, pol_out; domain = dom, verify_solenoidal = true)
end

@testset "manufactured single-mode synthesis (l=2, P=r³)" begin
    cfg, dom = _st_setup()
    pol = _st_spec(cfg, dom); tor = _st_spec(cfg, dom)
    lm_seed = findfirst(i -> cfg.l_values[i] == 2 && cfg.m_values[i] == 0, 1:cfg.nlm)
    slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_seed)
    for r_idx in 1:dom.N
        GeoDynamo.set_local_spectral_value!(parent(pol.data_real), slot, r_idx,
            dom.r[r_idx, 4]^3)
    end
    V = _st_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom)
    # u_r must be λ·P/r² = 6·r³/r² = 6r times the Y-pattern
    expect_spec = _st_spec(cfg, dom)
    for r_idx in 1:dom.N
        GeoDynamo.set_local_spectral_value!(parent(expect_spec.data_real), slot, r_idx,
            6.0 * dom.r[r_idx, 4])
    end
    expect_phys = _st_phys(cfg, dom)
    GeoDynamo.scalar_spectral_to_physical!(expect_spec, expect_phys)
    @test isapprox(parent(V.r_component.data), parent(expect_phys.data);
        rtol = 1e-8, atol = 1e-12)
    # u_θ radial structure: S = (∂_r P)/r = 3r²/r = 3r for P = r³, so the
    # ratio across interior radii is linear in r. (S = P'/r is the consistent
    # spheroidal scalar for u_r = l(l+1)P/r²: ∇·u = λP'/r² − λ(P'/r)/r ≡ 0.)
    uθ = parent(V.θ_component.data)
    r_range = GeoDynamo.range_local(cfg.pencils.r, 3)
    k1, k2 = 10, 30
    r1 = dom.r[k1 + first(r_range) - 1, 4]; r2 = dom.r[k2 + first(r_range) - 1, 4]
    checked = 0
    for j in 1:cfg.nlon, i in 1:cfg.nlat
        if abs(uθ[i, j, k2]) > 1e-6 && checked < 8
            @test isapprox(uθ[i, j, k1] / uθ[i, j, k2], r1 / r2; rtol = 1e-5)
            checked += 1
        end
    end
    @test checked >= 8

    # m=0-only field ⇒ the GRID-space divergence reference is aliasing-free and
    # provides an independent (non-spectral) confirmation of solenoidality.
    div_phys = _st_divergence(cfg, dom, V)
    dd = parent(div_phys.data)
    interior = dd[:, :, 3:(size(dd, 3) - 2)]
    scale_m0 = maximum(abs, parent(V.r_component.data)) +
               maximum(abs, parent(V.θ_component.data))
    @info "m=0 grid-space divergence cross-check" maxdiv = maximum(abs, interior) scale_m0
    @test maximum(abs, interior) < 1e-5 * scale_m0
end

# ===========================================================================
# Stage-2 Task 3: Q-based analysis — synthesis→analysis roundtrip on (T,P).
# Analysis recovers P from the radial component (P = r²·Q/λ), T from the
# toroidal sphtor scalar. Profiles vanish at endpoints (sinpi fill) so the
# banded-D1 endpoint stencils in the synthesis don't pollute the comparison.
# ===========================================================================
@testset "synthesis→analysis roundtrip on (T,P)" begin
    cfg, dom = _st_setup()
    Random.seed!(29)
    tor = _st_spec(cfg, dom); pol = _st_spec(cfg, dom)
    for spec in (tor, pol)
        sr = parent(spec.data_real); si = parent(spec.data_imag)
        for lm in 1:cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]; m = cfg.m_values[lm]
            (1 <= l <= 6) || continue
            for r_idx in 1:dom.N
                x = (dom.r[r_idx, 4] - dom.r[1, 4]) / (dom.r[dom.N, 4] - dom.r[1, 4])
                v = sinpi(x) * randn() * 1e-2
                GeoDynamo.set_local_spectral_value!(sr, slot, r_idx, v)
                m > 0 && GeoDynamo.set_local_spectral_value!(si, slot, r_idx, 0.7v)
            end
        end
    end
    V = _st_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom)
    tor2 = _st_spec(cfg, dom); pol2 = _st_spec(cfg, dom)
    GeoDynamo.vector_physical_to_spectral!(V, tor2, pol2; domain = dom)
    for (a, b, name) in ((tor2, tor, "T"), (pol2, pol, "P"))
        x = vcat(vec(parent(a.data_real)), vec(parent(a.data_imag)))
        y = vcat(vec(parent(b.data_real)), vec(parent(b.data_imag)))
        @test isapprox(x, y; rtol = 1e-8, atol = 1e-12)
    end
end

# ===========================================================================
# Stage-2 Task 4: vorticity under the solenoidal convention.
#
# Derivation in the code's pinned basis (machine-verified pieces):
#   velocity u = 𝒫(P) + 𝒯(T):  u_r = λP/r²·Y, S_u = P'/r, T_u = T
#   ω = ∇×u  ⇒  stored potentials of ω:
#     P_ω = −r·T                      (curl of toroidal → poloidal)
#     T_ω = (P'' − λP/r²)/r           (curl of poloidal → toroidal)
#
# Non-circular check for T_ω: the Stage-1-verified projection identity gives
#   [r̂·∇×ω]_lm = [r̂·∇×∇×u]_lm = (λ/r²)·(Q_u − ∂_r(r·S_u))
# with (Q_u,S_u) from the RAW QST analysis of the synthesized u; the vorticity
# formula must satisfy −(λ/r)·T_ω == that, per mode and radius.
# P_ω is pinned by ω_r: λ·P_ω/r² == −(λ/r)·T_u (Stage-1 toroidal projection).
# ===========================================================================
@testset "spectral vorticity matches Stage-1 curl projections" begin
    cfg, dom = _st_setup()
    Random.seed!(31)
    vf = GeoDynamo.create_shtns_velocity_fields(Float64, cfg, dom)
    tor = vf.toroidal; pol = vf.poloidal
    # SMOOTH radial profiles (random amplitude per MODE, not per radius): the
    # check compares a direct D2 against a composed D1∘D1 discretization, which
    # only agree on resolved profiles — white noise in r would be a test
    # artifact, not a formula probe.
    for spec in (tor, pol)
        sr = parent(spec.data_real); si = parent(spec.data_imag)
        for lm in 1:cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]; m = cfg.m_values[lm]
            (1 <= l <= 6) || continue
            amp_r = randn() * 1e-2
            amp_i = m > 0 ? 0.7 * randn() * 1e-2 : 0.0
            for r_idx in 1:dom.N
                x = (dom.r[r_idx, 4] - dom.r[1, 4]) / (dom.r[dom.N, 4] - dom.r[1, 4])
                prof = sinpi(x) * (1.0 + 0.3x)
                GeoDynamo.set_local_spectral_value!(sr, slot, r_idx, amp_r * prof)
                m > 0 && GeoDynamo.set_local_spectral_value!(si, slot, r_idx, amp_i * prof)
            end
        end
    end

    GeoDynamo.compute_vorticity_spectral!(vf, dom)
    ζT = vf.ζᵀ; ζP = vf.ζᴾ

    # Reference projections from the RAW QST analysis of the synthesized u
    V = _st_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom)
    Q = _st_spec(cfg, dom); S = _st_spec(cfg, dom); T_ = _st_spec(cfg, dom)
    GeoDynamo.force_physical_to_qst!(V, Q, S, T_)
    Rtor = _st_spec(cfg, dom); Rpol = _st_spec(cfg, dom)
    GeoDynamo.force_curl_projections!(Rtor, Rpol, Q, S, T_, dom)

    # (a) T_ω: −(λ/r)·ζᵀ_lm(r) == Rpol_lm(r)  (interior radii)
    # (b) P_ω: λ·ζᴾ_lm(r)/r²   == Rtor_lm(r)
    failures_a = 0; failures_b = 0; checked = 0
    for lm in 1:cfg.nlm
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        l = cfg.l_values[lm]
        (1 <= l <= 6) || continue
        λ = l * (l + 1)
        for (zt, zp, rt, rp) in (
            (parent(ζT.data_real), parent(ζP.data_real),
             parent(Rtor.data_real), parent(Rpol.data_real)),
            (parent(ζT.data_imag), parent(ζP.data_imag),
             parent(Rtor.data_imag), parent(Rpol.data_imag)))
            for r_idx in 3:(dom.N - 2)
                r = dom.r[r_idx, 4]
                a_lhs = -λ / r * GeoDynamo.local_spectral_value(zt, slot, r_idx)
                a_rhs = GeoDynamo.local_spectral_value(rp, slot, r_idx)
                b_lhs = λ / r^2 * GeoDynamo.local_spectral_value(zp, slot, r_idx)
                b_rhs = GeoDynamo.local_spectral_value(rt, slot, r_idx)
                isapprox(a_lhs, a_rhs; rtol = 1e-5, atol = 1e-10) || (failures_a += 1)
                isapprox(b_lhs, b_rhs; rtol = 1e-5, atol = 1e-10) || (failures_b += 1)
                checked += 1
            end
        end
    end
    @info "vorticity check" checked failures_a failures_b
    @test checked > 0
    @test failures_a == 0
    @test failures_b == 0
end
