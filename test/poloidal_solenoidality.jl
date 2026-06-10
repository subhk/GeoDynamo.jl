using Test
using MPI
using GeoDynamo
const G = GeoDynamo

# Solenoidality of synthesized poloidal fields — HARD GATE since Stage 2.
#
# Historical note: until the Stage-2 solenoidal-transform work
# (docs/superpowers/specs/2026-06-10-poloidal-momentum-double-curl-design.md)
# the synthesis fed the stored poloidal potential directly to the spheroidal
# slot (no radial-derivative coupling) and used two different u_r conventions,
# so the synthesized field was NOT divergence-free and this test was
# @test_broken. The synthesis now implements
#   u_r = l(l+1)·P/r² ,  tangential spheroidal S = (∂_r P)/r
# which is divergence-free by construction:
#   ∇·V|_lm = (1/r²)·d(r²Q)/dr − l(l+1)·S/r ≡ 0  with  Q = l(l+1)P/r².
#
# The test synthesizes a single poloidal mode with a non-constant radial
# profile, re-analyzes the ACTUAL physical field (Q from B_r, S from the raw
# tangential sphtor analysis), and requires the measured divergence to vanish.

function _setmode!(spec, l, m, domain, fn)
    rd = parent(spec.data_real)
    rr = G.range_local(spec.pencil, 3)
    idx = G.get_mode_index(spec.config, l, m);
    idx == 0 && return
    slot = G.local_spectral_storage_slot(spec.config, idx);
    slot === nothing && return
    for (lr, gr) in enumerate(rr)
        lr <= size(rd, 3) || continue
        G.set_local_spectral_value!(rd, slot, lr, fn(domain.r[gr, 4]))
    end
end

@testset "Synthesized poloidal field is solenoidal (∇·B = 0)" begin
    MPI.Finalized() && (@warn "MPI finalized; skipping"; return)
    MPI.Initialized() || MPI.Init()

    lmax = 6;
    mmax = 6;
    nlat = 12;
    nlon = 24;
    nr = 24
    cfg = G.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    dom = G.create_radial_domain(nr)
    rvals = dom.r[1:nr, 4]

    # Single poloidal mode (l=1,m=0), smooth non-constant radial profile, tor=0.
    vel = G.create_shtns_velocity_fields(Float64, cfg, dom;
        params = G.SolverParameters(
            geometry = :shell, nr = nr, lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon))
    fill!(parent(vel.toroidal.data_real), 0.0);
    fill!(parent(vel.toroidal.data_imag), 0.0)
    fill!(parent(vel.poloidal.data_real), 0.0);
    fill!(parent(vel.poloidal.data_imag), 0.0)
    _setmode!(vel.poloidal, 1, 0, dom, r -> r * (dom.r[nr, 4] - r))   # P(r)=r(r_o−r), nonconstant

    vec = G.create_shtns_vector_field(Float64, cfg, dom, (
        cfg.pencils.phi, cfg.pencils.phi, cfg.pencils.phi))
    G.shtnskit_vector_synthesis!(vel.toroidal, vel.poloidal, vec; domain = dom)

    # Re-analyze the ACTUAL physical field: Q from B_r (scalar analysis), S from
    # the raw tangential sphtor analysis (NOT the stored potential — under the
    # solenoidal convention the synthesized spheroidal scalar is (∂_r P)/r).
    qspec = G.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    sspec = G.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    tspec = G.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    G.force_physical_to_qst!(vec, qspec, sspec, tspec)

    m10 = G.get_mode_index(cfg, 1, 0)
    qslot = G.local_spectral_storage_slot(cfg, m10)
    Q = [G.local_spectral_value(parent(qspec.data_real), qslot, r) for r in 1:nr]
    S = [G.local_spectral_value(parent(sspec.data_real), qslot, r) for r in 1:nr]

    d1 = G.create_derivative_matrix(Float64, 1, dom)
    dr_r2Q = d1 * (rvals .^ 2 .* Q)                       # d(r²Q)/dr
    divB = (1.0 ./ rvals .^ 2) .* dr_r2Q .- (2.0 ./ rvals) .* S   # l(l+1)=2 for l=1

    interior = 3:(nr - 2)   # avoid one-sided endpoint stencils
    maxdiv = maximum(abs, divB[interior])
    relscale = maximum(abs, S[interior]) / minimum(rvals[interior])
    @info "poloidal ∇·B" maxdiv relscale ratio = maxdiv / relscale

    # HARD GATE since the Stage-2 solenoidal synthesis.
    @test maxdiv / relscale < 1e-6
end
