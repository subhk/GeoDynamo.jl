using Test, MPI, LinearAlgebra

@testset "inner-core admittance" begin
    if !MPI.Initialized()
        ;
        MPI.Init();
    end

    # Build inner-core ball domain (replicates create_inner_core_domain):
    # unit ball domain of N points, scaled by the inner-core radius.
    N = 8
    radius_ratio = 0.35
    unit_ball = GeoDynamo.create_ball_radial_domain(N; radial_bandwidth = 4)
    inner_core_radius = radius_ratio / (1.0 - radius_ratio)
    icdom = GeoDynamo.scale_radial_domain(unit_ball, inner_core_radius)

    η = 1.0;
    dt = 1e-3;
    θ = 0.5
    adm = GeoDynamo.create_inner_core_admittance(
        Float64, [1, 2, 3], icdom, η, dt; theta = θ)
    α1 = GeoDynamo.inner_core_alpha(adm, 1)
    α2 = GeoDynamo.inner_core_alpha(adm, 2)
    @test isfinite(α1) && isfinite(α2)
    @test α2 > α1 > 0     # diffusive admittance positive & increasing in l
end
