using Test
using MPI
using GeoDynamo
const G = GeoDynamo

# Solenoidality of synthesized poloidal fields.
#
# The vector synthesis reconstructs a poloidal field's radial component as
#   v_r = l(l+1)/r · pol        (src/fields/transforms.jl ~L590)
# while the tangential field is synthesized from `pol` passed directly to
# `synthesis_sphtor` as the spheroidal coefficient (no radial-derivative term),
# and the analysis recovers `pol` purely from the tangential field. For a true
# (Mie) poloidal field B = ∇×∇×(P r̂) the spheroidal coefficient is the radial
# derivative S = (1/r) d(rP)/dr and the radial component is v_r = l(l+1)/r² · P.
# The code's choices satisfy ∇·B = 0 only when the radial profile is constant;
# in general the synthesized poloidal field is NOT divergence-free:
#   ∇·V|_lm = (1/r²) d(r² Q)/dr − l(l+1)/r · S ,  Q = l(l+1)/r·pol , S = pol
#           = l(l+1)[ pol/r² + pol'/r − pol/r ]  ≠ 0 .
#
# This test synthesizes a single poloidal mode and measures ∇·B from the actual
# physical field (B_r analyzed back to spectral, plus the radial-derivative of
# r²B_r), so it is the exact oracle a reconstruction fix must satisfy. It is
# marked @test_broken because the current reconstruction fails it by design.
#
# SCOPED FIX (Mie-consistent; aligns synthesis with the already-Mie diffusion
# operator, src/bcs/scalar_bc.jl:104 uses −l(l+1)/r²):
#   synthesis:  v_r = l(l+1)/r² · P ;  tangential spheroidal = (1/r) d(rP)/dr
#   analysis :  recover P from v_r  (P = v_r · r²/l(l+1)), toroidal from tangential
# This changes how nonlinear terms project onto poloidal/toroidal, so it MUST be
# validated at the evolution level (Christensen Case 0) before merging — not at
# the transform level alone. See memory proj_geodynamo_poloidal_synthesis_nonsolenoidal.

function _setmode!(spec, l, m, domain, fn)
    rd = parent(spec.data_real)
    rr = G.range_local(spec.pencil, 3)
    idx = G.get_mode_index(spec.config, l, m); idx == 0 && return
    slot = G.local_spectral_storage_slot(spec.config, idx); slot === nothing && return
    for (lr, gr) in enumerate(rr)
        lr <= size(rd, 3) || continue
        G.set_local_spectral_value!(rd, slot, lr, fn(domain.r[gr, 4]))
    end
end

@testset "Synthesized poloidal field is solenoidal (∇·B = 0)" begin
    MPI.Finalized() && (@warn "MPI finalized; skipping"; return)
    MPI.Initialized() || MPI.Init()

    lmax = 6; mmax = 6; nlat = 12; nlon = 24; nr = 24
    cfg = G.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = G.create_radial_domain(nr)
    rvals = dom.r[1:nr, 4]

    # Single poloidal mode (l=1,m=0), smooth non-constant radial profile, tor=0.
    vel = G.create_shtns_velocity_fields(Float64, cfg, dom;
            params=G.SolverParameters(geometry=:shell, nr=nr, lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon))
    fill!(parent(vel.toroidal.data_real), 0.0); fill!(parent(vel.toroidal.data_imag), 0.0)
    fill!(parent(vel.poloidal.data_real), 0.0); fill!(parent(vel.poloidal.data_imag), 0.0)
    _setmode!(vel.poloidal, 1, 0, dom, r -> r * (dom.r[nr,4] - r))   # P(r)=r(r_o−r), nonconstant

    vec = G.create_shtns_vector_field(Float64, cfg, dom, (cfg.pencils.phi, cfg.pencils.phi, cfg.pencils.phi))
    G.shtnskit_vector_synthesis!(vel.toroidal, vel.poloidal, vec; domain=dom)

    # Analyze B_r (a physical scalar) back to spectral to get Q_lm(r).
    qspec = G.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    G.shtnskit_physical_to_spectral!(vec.r_component, qspec)

    m10 = G.get_mode_index(cfg, 1, 0)
    qslot = G.local_spectral_storage_slot(cfg, m10)
    pslot = G.local_spectral_storage_slot(vel.poloidal.config, m10)
    Q = [G.local_spectral_value(parent(qspec.data_real), qslot, r) for r in 1:nr]   # B_r coeff
    S = [G.local_spectral_value(parent(vel.poloidal.data_real), pslot, r) for r in 1:nr]   # spheroidal (pol)

    d1 = G.create_derivative_matrix(Float64, 1, dom)
    dr_r2Q = d1 * (rvals .^ 2 .* Q)                       # d(r²Q)/dr
    divB = (1.0 ./ rvals .^ 2) .* dr_r2Q .- (2.0 ./ rvals) .* S   # l(l+1)=2 for l=1

    interior = 3:(nr-2)   # avoid one-sided endpoint stencils
    maxdiv = maximum(abs, divB[interior])
    relscale = maximum(abs, S[interior]) / minimum(rvals[interior])
    @info "poloidal ∇·B" maxdiv relscale ratio=maxdiv/relscale

    # The synthesized poloidal field SHOULD be divergence-free. It is not, under
    # the current 2-component reconstruction — documented as broken until the
    # Mie-consistent fix (see header) lands and is validated against Case 0.
    @test_broken maxdiv / relscale < 1e-6
end
