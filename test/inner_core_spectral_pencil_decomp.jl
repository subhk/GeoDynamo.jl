# Regression test for the MPI multi-rank inner-core magnetic coupling bug.
#
# The inner-core spectral pencil MUST share the (l, m) process-grid
# decomposition of the outer-core spectral pencil. `add_inner_core_rotation!`
# derives a single storage `slot` from the OUTER pencil and uses it to index
# BOTH the outer nonlinear arrays and the inner-core (ic_*) arrays. If the two
# pencils distribute the (l, m) axes onto opposite process-grid axes, that slot
# addresses the wrong mode (or out of bounds) on any run with nprocs > 1.
#
# decomp_dims is a property of the constructed Pencil object, so this check is
# meaningful even at nprocs == 1 (where range_local would be trivially equal).

using Test

@testset "inner-core spectral pencil matches outer-core decomposition" begin
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = 4, mmax = 4, nlat = 10, nlon = 16, nr = 6)

    outer_spec = cfg.pencils.spec
    nr_inner = 4
    inner_spec = GeoDynamo.create_inner_core_spectral_pencil(cfg, outer_spec, nr_inner)

    # Same (l, m) ownership => identical decomposition axes.
    @test inner_spec.decomp_dims == outer_spec.decomp_dims

    # Radial extent is the inner-core grid, not the outer-core nr.
    @test inner_spec.size_global[3] == nr_inner
    @test inner_spec.size_global[1] == outer_spec.size_global[1]  # lmax+1
    @test inner_spec.size_global[2] == outer_spec.size_global[2]  # mmax+1
end
