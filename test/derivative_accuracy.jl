using Test
using MPI

const FINALIZE_MPI_DERIV = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Radial Derivative Accuracy" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping derivative tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    nr = 16
    dom = GeoDynamo.create_radial_domain(nr)
    r = dom.r[:, 4]

    @testset "First derivative of r^2 should be 2r" begin
        D1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
        f = r .^ 2
        df = D1 * f
        df_exact = 2.0 .* r

        # Interior points should be accurate; boundary stencils may have larger error
        interior = 3:nr-2
        @test df[interior] ≈ df_exact[interior] atol=1e-8
    end

    @testset "First derivative of r^3 should be 3r^2" begin
        D1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
        f = r .^ 3
        df = D1 * f
        df_exact = 3.0 .* r .^ 2

        interior = 3:nr-2
        @test df[interior] ≈ df_exact[interior] atol=1e-6
    end

    @testset "Second derivative of r^3 should be 6r" begin
        D2 = GeoDynamo.create_derivative_matrix(Float64, 2, dom)
        f = r .^ 3
        d2f = D2 * f
        d2f_exact = 6.0 .* r

        interior = 3:nr-2
        @test d2f[interior] ≈ d2f_exact[interior] atol=1e-4
    end

    @testset "apply_∂r! matches D * v" begin
        D1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
        f = sin.(r)
        result_mul = D1 * f
        result_apply = zeros(nr)
        GeoDynamo.apply_∂r!(result_apply, D1, f)
        @test result_mul ≈ result_apply atol=1e-14
    end

    @testset "Radial Laplacian on r^2 = 6" begin
        # ∇²_r(r^2) = d²/dr²(r^2) + (2/r)*d/dr(r^2) = 2 + 2/r * 2r = 2 + 4 = 6
        lap = GeoDynamo.create_radial_laplacian(Float64, dom)
        f = r .^ 2
        lapf = lap * f

        interior = 3:nr-2
        @test lapf[interior] ≈ fill(6.0, length(interior)) atol=1e-6
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_DERIV && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
