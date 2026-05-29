using Test
using MPI

const FINALIZE_MPI_RD = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Radial Domain" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping radial domain tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    @testset "Shell radial domain creation" begin
        nr = 8
        dom = GeoDynamo.create_radial_domain(nr)

        @test dom.N == nr
        @test length(dom.local_range) == nr
        @test size(dom.r) == (nr, 7)

        # r values should be monotonically increasing
        r_vals = dom.r[:, 4]
        @test all(diff(r_vals) .> 0)

        # r values should be within [r_inner, r_outer]
        r_inner = dom.r[1, 4]
        r_outer = dom.r[nr, 4]
        @test r_inner > 0
        @test r_outer > r_inner

        # Column 4 holds r, column 5 holds r^1 (= r), should match
        @test dom.r[:, 5] ≈ dom.r[:, 4]

        # Column 3 holds r^(-1)
        @test dom.r[:, 3] ≈ 1.0 ./ dom.r[:, 4] atol=1e-12

        # Radial operators must be populated at construction, not left zero
        # (regression guard against the silent-zeros footgun), and must equal a
        # fresh recompute.
        @test any(!iszero, dom.radial_laplacian)
        @test dom.radial_laplacian ≈ GeoDynamo.create_radial_laplacian(dom).data
        for order in 1:length(dom.dr_matrices)
            @test any(!iszero, dom.dr_matrices[order])
            @test dom.dr_matrices[order] ≈
                  GeoDynamo.create_derivative_matrix(order, dom).data
        end
    end

    @testset "Ball radial domain creation" begin
        Ball = GeoDynamo.GeoDynamoBall
        nr = 8
        dom = Ball.create_ball_radial_domain(nr)

        @test dom.N == nr
        @test size(dom.r) == (nr, 7)

        # Ball starts at r=0
        @test dom.r[1, 4] ≈ 0.0 atol=1e-15

        # Outer radius > 0
        @test dom.r[nr, 4] > 0

        # r values monotonically increasing
        @test all(diff(dom.r[:, 4]) .> 0)

        # At r=0, negative powers should be regularized to 0
        @test dom.r[1, 1] == 0.0  # r^(-3)
        @test dom.r[1, 2] == 0.0  # r^(-2)
        @test dom.r[1, 3] == 0.0  # r^(-1)

        # Radial operators populated at construction (regression: silent-zeros footgun)
        @test any(!iszero, dom.radial_laplacian)
        @test any(!iszero, dom.dr_matrices[1])
        @test all(isfinite, dom.radial_laplacian)  # r=0 regularization must not leak Inf/NaN
    end

    @testset "Ball domain requires nr >= 2" begin
        Ball = GeoDynamo.GeoDynamoBall
        @test_throws ErrorException Ball.create_ball_radial_domain(1)
    end

    @testset "Shell domain requires nr >= 2" begin
        @test_throws ArgumentError GeoDynamo.create_radial_domain(1)
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_RD && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
