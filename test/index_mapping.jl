using Test

@testset "Spectral Index Mapping" begin
    @testset "index_to_lm_shtnskit roundtrip" begin
        lmax = 6; mmax = 6
        idx = 0
        for m in 0:mmax
            for l in m:lmax
                idx += 1
                l_out, m_out = GeoDynamo.index_to_lm_shtnskit(idx, lmax, mmax)
                @test l_out == l
                @test m_out == m
            end
        end
        # Total number of modes
        expected_nlm = sum(lmax - m + 1 for m in 0:mmax)
        @test idx == expected_nlm
    end

    @testset "index_to_lm_shtnskit invalid indices" begin
        lmax = 4; mmax = 4
        l, m = GeoDynamo.index_to_lm_shtnskit(0, lmax, mmax)
        @test l == -1 && m == -1

        l, m = GeoDynamo.index_to_lm_shtnskit(-1, lmax, mmax)
        @test l == -1 && m == -1

        # Too large index
        nlm = sum(lmax - mm + 1 for mm in 0:mmax)
        l, m = GeoDynamo.index_to_lm_shtnskit(nlm + 1, lmax, mmax)
        @test l == -1 && m == -1
    end

    @testset "l=0,m=0 is always index 1" begin
        for lmax in 1:8
            l, m = GeoDynamo.index_to_lm_shtnskit(1, lmax, lmax)
            @test l == 0
            @test m == 0
        end
    end

    @testset "m=0 modes are contiguous at start" begin
        lmax = 6; mmax = 6
        for l in 0:lmax
            l_out, m_out = GeoDynamo.index_to_lm_shtnskit(l + 1, lmax, mmax)
            @test l_out == l
            @test m_out == 0
        end
    end
end
