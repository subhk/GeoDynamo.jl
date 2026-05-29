using Test, MPI, LinearAlgebra

@testset "inner-core reconstruction" begin
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

    # Reconstruct with zero history and ICB Dirichlet value g.
    g = 0.37
    S = GeoDynamo.reconstruct_inner_core(adm, 1, g, zeros(N))
    @test S[end] ≈ g atol=1e-12     # ICB value imposed
    @test S[1] ≈ 0.0 atol=1e-12     # regularity at r=0 (l ≥ 1)
end
