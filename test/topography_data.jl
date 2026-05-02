using Test

# Shorthand for the internal topography submodule
const topo = GeoDynamo.bcs.topography

@testset "Topography Data Structures" begin

    # ─── lm_to_index / index_to_lm ───────────────────────────────────────
    @testset "lm_to_index" begin
        lmax = 4
        # (0,0) → 1
        @test topo.lm_to_index(0, 0, lmax) == 1
        # (1,0) → 2, (1,1) → 3
        @test topo.lm_to_index(1, 0, lmax) == 2
        @test topo.lm_to_index(1, 1, lmax) == 3
        # (2,0) → 4, (2,1) → 5, (2,2) → 6
        @test topo.lm_to_index(2, 0, lmax) == 4
        @test topo.lm_to_index(2, 1, lmax) == 5
        @test topo.lm_to_index(2, 2, lmax) == 6
    end

    @testset "index_to_lm roundtrip" begin
        lmax = 8
        for l in 0:lmax
            for m in 0:l
                idx = topo.lm_to_index(l, m, lmax)
                l2, m2 = topo.index_to_lm(idx, lmax)
                @test l2 == l
                @test m2 == m
            end
        end
    end

    @testset "lm_to_index asserts on invalid inputs" begin
        @test_throws AssertionError topo.lm_to_index(-1, 0, 4)
        @test_throws AssertionError topo.lm_to_index(5, 0, 4)
        @test_throws AssertionError topo.lm_to_index(2, 3, 4)
    end

    # ─── TopographyField construction ─────────────────────────────────────
    @testset "TopographyField{Float64} construction" begin
        field = topo.TopographyField{Float64}(4, 4, 1.0, GeoDynamo.OUTER_BOUNDARY)
        @test field.lmax == 4
        @test field.mmax == 4
        @test field.radius == 1.0
        @test field.location == GeoDynamo.OUTER_BOUNDARY
        @test field.rms_amplitude == 0.0
        @test field.max_amplitude == 0.0
        @test field.description == "zero topography"
        # nlm for lmax=4: Σ (min(l, mmax)+1) for l=0..4 = 1+2+3+4+5 = 15
        @test field.nlm == 15
        @test length(field.coeffs_real) == field.nlm
        @test length(field.coeffs_imag) == field.nlm
        @test all(x -> x == 0.0, field.coeffs_real)
    end

    @testset "TopographyField Float64 shorthand" begin
        field = topo.TopographyField(8, 8, 0.35, GeoDynamo.INNER_BOUNDARY)
        @test field isa topo.TopographyField{Float64}
        @test field.radius == 0.35
        @test field.location == GeoDynamo.INNER_BOUNDARY
    end

    @testset "TopographyField mmax capped at lmax" begin
        field = topo.TopographyField{Float64}(4, 10, 1.0, GeoDynamo.OUTER_BOUNDARY)
        @test field.mmax == 4  # capped to lmax
    end

    # ─── TopographyData construction ──────────────────────────────────────
    @testset "TopographyData{Float64}() empty" begin
        td = topo.TopographyData{Float64}()
        @test td.icb === nothing
        @test td.cmb === nothing
        @test td.gaunt_cache === nothing
        @test td.epsilon == 0.01
        @test td.is_time_dependent == false
    end

    @testset "TopographyData() defaults to Float64" begin
        td = topo.TopographyData()
        @test td isa topo.TopographyData{Float64}
    end

    # ─── set/get topography coefficients ──────────────────────────────────
    @testset "set_topography_coefficients! with real vector" begin
        field = topo.TopographyField(4, 4, 1.0, GeoDynamo.OUTER_BOUNDARY)
        coeffs = collect(1.0:field.nlm)
        topo.set_topography_coefficients!(field, coeffs)
        @test field.coeffs_real ≈ coeffs
        @test all(x -> x == 0.0, field.coeffs_imag)
        @test field.rms_amplitude > 0
        @test field.max_amplitude > 0
    end

    @testset "set_topography_coefficients! with complex vector" begin
        field = topo.TopographyField(4, 4, 1.0, GeoDynamo.OUTER_BOUNDARY)
        coeffs = [complex(Float64(i), Float64(i+1)) for i in 1:field.nlm]
        topo.set_topography_coefficients!(field, coeffs)
        @test field.coeffs_real ≈ real.(coeffs)
        @test field.coeffs_imag ≈ imag.(coeffs)
    end

    @testset "get_topography_coefficients roundtrip" begin
        field = topo.TopographyField(4, 4, 1.0, GeoDynamo.OUTER_BOUNDARY)
        input = [complex(Float64(i), Float64(2i)) for i in 1:field.nlm]
        topo.set_topography_coefficients!(field, input)
        output = topo.get_topography_coefficients(field)
        @test output ≈ input
    end

    # ─── get_coefficient ──────────────────────────────────────────────────
    @testset "get_coefficient for specific (l,m)" begin
        field = topo.TopographyField(4, 4, 1.0, GeoDynamo.OUTER_BOUNDARY)
        coeffs = zeros(ComplexF64, field.nlm)
        idx_20 = topo.lm_to_index(2, 0, 4)
        coeffs[idx_20] = 3.0 + 4.0im
        topo.set_topography_coefficients!(field, coeffs)

        c = topo.get_coefficient(field, 2, 0)
        @test c ≈ 3.0 + 4.0im
    end

    @testset "get_coefficient returns zero for out-of-range" begin
        field = topo.TopographyField(4, 4, 1.0, GeoDynamo.OUTER_BOUNDARY)
        @test topo.get_coefficient(field, 10, 0) == 0.0 + 0.0im
    end

    # ─── update_topography_statistics! ────────────────────────────────────
    @testset "update_topography_statistics!" begin
        field = topo.TopographyField(2, 2, 1.0, GeoDynamo.OUTER_BOUNDARY)
        # Set known coefficients
        fill!(field.coeffs_real, 0.0)
        fill!(field.coeffs_imag, 0.0)
        field.coeffs_real[1] = 3.0
        field.coeffs_imag[1] = 4.0
        # amplitude of first coeff = sqrt(9+16) = 5
        topo.update_topography_statistics!(field)
        @test field.max_amplitude ≈ 5.0
        @test field.rms_amplitude ≈ 5.0  # only one nonzero coeff
    end

    # ─── create_topography_data ───────────────────────────────────────────
    @testset "create_topography_data with CMB coefficients" begin
        lmax = 4
        nlm_lmax4 = 15  # for lmax=4, mmax=4
        coeffs = zeros(ComplexF64, nlm_lmax4)
        coeffs[1] = 1.0
        td = topo.create_topography_data(
            cmb_coeffs=coeffs, cmb_radius=1.0, lmax=lmax, epsilon=0.05
        )
        @test td.cmb !== nothing
        @test td.icb === nothing
        @test td.epsilon == 0.05
        @test td.cmb.radius == 1.0
        @test td.cmb.lmax == lmax
    end

    @testset "create_topography_data with ICB coefficients" begin
        coeffs = [1.0, 0.5, 0.3]
        td = topo.create_topography_data(
            icb_coeffs=coeffs, icb_radius=0.35, lmax=4
        )
        @test td.icb !== nothing
        @test td.cmb === nothing
        @test td.icb.radius == 0.35
    end

    # ─── create_uniform_topography ────────────────────────────────────────
    @testset "create_uniform_topography" begin
        field = topo.create_uniform_topography(0.1, 1.0, GeoDynamo.OUTER_BOUNDARY; lmax=8)
        @test field.lmax == 8
        @test field.radius == 1.0
        @test field.coeffs_real[1] ≈ 0.1 * sqrt(4π)
        @test contains(field.description, "uniform")
    end

    # ─── create_spherical_harmonic_topography ─────────────────────────────
    @testset "create_spherical_harmonic_topography (l=2, m=0)" begin
        field = topo.create_spherical_harmonic_topography(
            2, 0, 0.5, 1.0, GeoDynamo.OUTER_BOUNDARY; lmax=8
        )
        idx = topo.lm_to_index(2, 0, 8)
        @test field.coeffs_real[idx] ≈ 0.5
        @test contains(field.description, "Y_2^0")
    end

    @testset "create_spherical_harmonic_topography (l=3, m=2)" begin
        field = topo.create_spherical_harmonic_topography(
            3, 2, 1.0, 0.35, GeoDynamo.INNER_BOUNDARY; lmax=8
        )
        idx = topo.lm_to_index(3, 2, 8)
        @test field.coeffs_real[idx] ≈ 1.0
        @test field.location == GeoDynamo.INNER_BOUNDARY
    end

    @testset "create_spherical_harmonic_topography asserts on bad inputs" begin
        @test_throws AssertionError topo.create_spherical_harmonic_topography(
            10, 0, 1.0, 1.0, GeoDynamo.OUTER_BOUNDARY; lmax=8
        )
    end

    # ─── Gauss-Legendre nodes ─────────────────────────────────────────────
    @testset "gauss_legendre_nodes" begin
        nodes, weights = topo.gauss_legendre_nodes(8)
        @test length(nodes) == 8
        @test length(weights) == 8
        # Gauss-Legendre nodes are in (-1, 1) and symmetric
        @test all(-1 .< nodes .< 1)
        for i in 1:length(nodes)
            @test nodes[i] ≈ -nodes[end - i + 1] atol=1e-12
        end
        # Weights should be positive and sum to 2
        @test all(weights .> 0)
        @test sum(weights) ≈ 2.0 atol=1e-12
    end

    @testset "gauss_legendre_nodes n=1" begin
        nodes, weights = topo.gauss_legendre_nodes(1)
        @test length(nodes) == 1
        @test nodes[1] ≈ 0.0 atol=1e-12
        @test weights[1] ≈ 2.0 atol=1e-12
    end

    # ─── compute_associated_legendre ──────────────────────────────────────
    @testset "compute_associated_legendre P_0^0" begin
        # P_0^0(x) = 1 for all x
        for x in [-0.5, 0.0, 0.5, 0.99]
            @test topo.compute_associated_legendre(0, 0, x) ≈ 1.0
        end
    end

    @testset "compute_associated_legendre P_1^0" begin
        # P_1^0(x) = x
        for x in [-0.5, 0.0, 0.5]
            @test topo.compute_associated_legendre(1, 0, x) ≈ x
        end
    end

    @testset "compute_associated_legendre P_2^0" begin
        # P_2^0(x) = (3x^2 - 1)/2
        for x in [-0.5, 0.0, 0.5, 0.8]
            expected = (3x^2 - 1) / 2
            @test topo.compute_associated_legendre(2, 0, x) ≈ expected atol=1e-12
        end
    end
end
