using Test
using GeoDynamo
using MPI
using LinearAlgebra
using Random

MPI.Initialized() || MPI.Init()

# The legacy scalar-gradient path (compute_all_gradients_spectral! +
# transform_field_and_gradients_to_physical! on a GradientWorkspace) must produce
# the TRUE horizontal gradient (∇f)_θ = (1/r)∂θf and (∇f)_φ = (1/(r sinθ))∂φf.
#
# Two coupled bugs: (1) the θ-recurrence used the wrong l-argument (A_±(l) — the
# coefficients of Y_{l±1} in sinθ∂θY_l — instead of A_+(l-1)/A_-(l+1), the
# coefficients of Y_l in sinθ∂θY_{l∓1}), producing garbage; (2) the physical
# tangential components were never divided by sinθ. Oracle: sphtor synthesis
# (vector_spectral_to_physical! with the scalar as the spheroidal potential) gives
# the exact bare angular derivatives ∂θf and (1/sinθ)∂φf for band-limited fields.

function _rel_l2(a, b)
    nlat = size(a, 1); nr = size(a, 3)
    θsel = 3:(nlat - 2)
    rsel = 3:(nr - 2)
    av = vec(a[θsel, :, rsel]); bv = vec(b[θsel, :, rsel])
    return norm(av .- bv) / max(norm(bv), eps())
end

@testset "legacy scalar gradient carries the 1/sinθ metric (matches sphtor gradient)" begin
    lmax = 8; nr = 24
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = lmax, nlat = 2lmax + 4, nlon = 4lmax + 8, nr = nr)
    dom = GeoDynamo.create_radial_domain(nr)
    temp = GeoDynamo.create_shtns_temperature_field(Float64, cfg, dom)
    ws = GeoDynamo.create_gradient_workspace(Float64, cfg, dom)

    Random.seed!(5)
    sr = parent(temp.spectral.data_real); si = parent(temp.spectral.data_imag)
    for lm in 1:cfg.nlm
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        l = cfg.l_values[lm]; m = cfg.m_values[lm]
        (1 <= l <= lmax - 2) || continue
        for r in 1:nr
            x = (dom.r[r, 4] - dom.r[1, 4]) / (dom.r[nr, 4] - dom.r[1, 4])
            v = sinpi(x) * randn() * 1e-2
            GeoDynamo.set_local_spectral_value!(sr, slot, r, v)
            m > 0 && GeoDynamo.set_local_spectral_value!(si, slot, r, 0.7v)
        end
    end

    GeoDynamo.zero_gradient_workspace!(ws)
    GeoDynamo.compute_all_gradients_spectral!(temp, dom, ws)
    GeoDynamo.transform_field_and_gradients_to_physical!(temp, ws)
    gθ = copy(parent(temp.gradient.θ_component.data))
    gφ = copy(parent(temp.gradient.φ_component.data))

    zero_spec = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    V = GeoDynamo.create_shtns_vector_field(Float64, cfg, dom,
        (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))
    GeoDynamo.vector_spectral_to_physical!(zero_spec, temp.spectral, V; domain = nothing)
    refθ = parent(V.θ_component.data); refφ = parent(V.φ_component.data)
    r_range = GeoDynamo.range_local(cfg.pencils.r, 3)
    expθ = similar(refθ); expφ = similar(refφ)
    for k in axes(refθ, 3)
        rinv = dom.r[k + first(r_range) - 1, 3]
        @views expθ[:, :, k] .= refθ[:, :, k] .* rinv
        @views expφ[:, :, k] .= refφ[:, :, k] .* rinv
    end

    @test _rel_l2(gθ, expθ) < 0.05
    @test _rel_l2(gφ, expφ) < 0.05
end
