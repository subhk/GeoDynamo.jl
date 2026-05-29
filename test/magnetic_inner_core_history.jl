using Test, MPI, LinearAlgebra

@testset "inner-core history flux φ0" begin
    if !MPI.Initialized()
        ;
        MPI.Init();
    end

    N = 8
    radius_ratio = 0.35
    unit_ball = GeoDynamo.create_ball_radial_domain(N; radial_bandwidth = 4)
    icdom = GeoDynamo.scale_radial_domain(unit_ball, radius_ratio / (1.0 - radius_ratio))
    adm = GeoDynamo.create_inner_core_admittance(
        Float64, [1, 2, 3], icdom, 1.0, 1e-3; theta = 0.5)

    # Zero history → zero flux (exactly).
    @test GeoDynamo.inner_core_history_flux(adm, 1, zeros(N)) == 0.0

    # Non-trivial regular profile → nonzero flux.
    p = collect(range(0.0, 1.0; length = N))
    @test GeoDynamo.inner_core_history_flux(adm, 1, p) != 0.0
end
