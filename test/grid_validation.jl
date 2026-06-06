using Test
using GeoDynamo

@testset "Grid constructor validation" begin
    SS = GeoDynamo.SphericalShellGrid
    SB = GeoDynamo.SphericalBallGrid
    cpu = GeoDynamo.CPU()

    @testset "SphericalShellGrid rejects invalid spectral/radial configs" begin
        # lmax must be >= 1
        @test_throws ArgumentError SS(cpu; lmax = 0, nr = 8)
        # mmax must satisfy 0 <= mmax <= lmax
        @test_throws ArgumentError SS(cpu; lmax = 4, mmax = 6, nr = 8)
        @test_throws ArgumentError SS(cpu; lmax = 4, mmax = -1, nr = 8)
        # nr must be >= 2 (a single radial point is degenerate)
        @test_throws ArgumentError SS(cpu; lmax = 4, nr = 1)
        # explicit nlat too small for lmax
        @test_throws ArgumentError SS(cpu; lmax = 8, nlat = 8, nr = 8)
        # explicit nlon too small for mmax (Nyquist 2*mmax+1)
        @test_throws ArgumentError SS(cpu; lmax = 8, mmax = 8, nlon = 8, nr = 8)
        # nr_inner must be < nr
        @test_throws ArgumentError SS(cpu; lmax = 4, nr = 4, nr_inner = 4)
    end

    @testset "SphericalShellGrid accepts valid configs" begin
        g = SS(cpu; lmax = 8, nr = 16)
        @test g.lmax == 8
        @test g.nlat >= g.lmax + 1
        @test g.nlon >= 2 * g.mmax + 1
        @test g.nr == 16
        g2 = SS(cpu; lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        @test g2.nr_inner == 4
        # small lmax: the default nlat self-corrects to >= lmax+1 and constructs
        g3 = SS(cpu; lmax = 1, nr = 8)
        @test g3.nlat >= g3.lmax + 1
    end

    @testset "SphericalBallGrid rejects invalid spectral/radial configs" begin
        @test_throws ArgumentError SB(cpu; lmax = 0, nr = 8)
        @test_throws ArgumentError SB(cpu; lmax = 4, mmax = 6, nr = 8)
        @test_throws ArgumentError SB(cpu; lmax = 4, mmax = -1, nr = 8)
        @test_throws ArgumentError SB(cpu; lmax = 4, nr = 1)
        @test_throws ArgumentError SB(cpu; lmax = 8, nlat = 8, nr = 8)
        @test_throws ArgumentError SB(cpu; lmax = 8, mmax = 8, nlon = 8, nr = 8)
    end

    @testset "SphericalBallGrid accepts valid configs" begin
        g = SB(cpu; lmax = 4, nr = 8)
        @test g.lmax == 4
        @test g.nlat >= g.lmax + 1
        @test g.nr == 8
        # small lmax: the default nlat self-corrects to >= lmax+1 and constructs
        g2 = SB(cpu; lmax = 1, nr = 8)
        @test g2.nlat >= g2.lmax + 1
    end
end
