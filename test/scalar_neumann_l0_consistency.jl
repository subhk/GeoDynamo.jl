using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

# Double-Neumann (FixedFlux/FixedFlux, scalar_bc_code == 4) consistency for the
# l = 0 mean mode.
#
# Under fixed-flux conditions at BOTH boundaries the inner wall condition is
# Neumann (∂_r Θ = flux) for EVERY degree l, including the spherically-symmetric
# mean l = 0. The implicit time-stepping operator (mass/dt) I − θ κ L is a
# positive-shifted Helmholtz operator and is NON-singular even with pure-Neumann
# rows (the (mass/dt) I shift removes the bare-Laplacian constant null space), so
# no Dirichlet "pin" is needed or correct for l = 0.
#
# Bug: _apply_scalar_boundary_rows! special-cased `scalar_bc_code == 4 && l == 0`
# by overwriting the inner Neumann derivative row with a Dirichlet identity row,
# which then imposes the inner FLUX datum as a prescribed VALUE on the mean mode.
#
# Contract: the inner boundary row of the l = 0 system matrix must equal the
# inner boundary row of every other degree (the shared, l-independent Neumann
# first-derivative stencil) — NOT the identity row [1, 0, 0, …].

@testset "double-Neumann scalar: l=0 inner row is Neumann, not a Dirichlet pin" begin
    lmax = 4; mmax = 4
    nlat = max(lmax + 2, 10); nlon = max(2lmax + 1, 16); nr = 16
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    dom = GeoDynamo.create_radial_domain(nr)

    dt = 1.0e-3; θ = 0.5; diffusivity = 0.7
    # temperature_bc_code = 4  ⇒  FixedFlux / FixedFlux (double Neumann)
    matrices = GeoDynamo.create_temperature_matrices(cfg, dom, diffusivity, dt;
        temperature_bc_code = 4, theta = θ)

    idx0 = matrices.lookup[0]
    idx2 = matrices.lookup[2]
    A0 = GeoDynamo.banded_to_dense(matrices.system_matrices[idx0])
    A2 = GeoDynamo.banded_to_dense(matrices.system_matrices[idx2])

    inner0 = A0[1, :]
    inner2 = A2[1, :]

    # The l-independent Neumann derivative stencil spans multiple radial nodes.
    @test count(!iszero, inner2) > 1
    # The inner flux BC must be applied identically for l = 0 and l = 2.
    @test inner0 ≈ inner2
    # And specifically must NOT be a Dirichlet identity pin on the mean mode.
    @test !(inner0[1] == 1.0 && all(iszero, inner0[2:end]))
end
