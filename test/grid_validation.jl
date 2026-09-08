using Test
using GeoDynamo

@testset "Pencil decomposition is memoized per grid" begin
    # Every decomposition allocates MPI communicators (MPITopology -> MPI_Cart_create,
    # plus make_subcomms) and nothing frees them, so rebuilding it for each config
    # walks the suite into MPICH's 2048-communicator ceiling. Identical grids must
    # therefore share one decomposition; distinct grids must not.
    kw = (; lmax = 4, mmax = 4, nlat = 10, nlon = 20, nr = 8, optimize_decomp = false)
    a = GeoDynamo.create_shtnskit_config(; kw...)
    b = GeoDynamo.create_shtnskit_config(; kw...)
    for k in (:theta, :phi, :r, :spec, :theta_phys)
        @test getproperty(a.pencils, k) === getproperty(b.pencils, k)
    end
    @test a.pencils.θ_comm == b.pencils.θ_comm
    @test a.pencils.r_comm == b.pencils.r_comm

    # optimize_decomp is currently retained only for API compatibility and does
    # not affect topology. It must therefore not split the communicator cache.
    same_grid_other_flag = GeoDynamo.create_shtnskit_config(;
        kw..., optimize_decomp = true)
    for k in (:theta, :phi, :r, :spec, :theta_phys)
        @test getproperty(a.pencils, k) === getproperty(same_grid_other_flag.pencils, k)
    end

    # a different grid is a different decomposition
    c = GeoDynamo.create_shtnskit_config(; lmax = 4, mmax = 4, nlat = 12, nlon = 24,
        nr = 8, optimize_decomp = false)
    @test c.pencils.r !== a.pencils.r

    # ...and the cache is clearable, so a test that needs a fresh topology can get one
    @test GeoDynamo.clear_pencil_decomposition_cache!() isa Int
    d = GeoDynamo.create_shtnskit_config(; kw...)
    @test d.pencils.r !== a.pencils.r
end


@testset "SHTnsKit configuration summary is opt-in" begin
    function capture_stdout(f)
        path, io = mktemp()
        try
            redirect_stdout(f, io)
            flush(io)
            seekstart(io)
            return read(io, String)
        finally
            close(io)
            rm(path; force = true)
        end
    end

    default_output = capture_stdout() do
        GeoDynamo.create_shtnskit_config(
            lmax = 3, mmax = 3, nlat = 8, nlon = 12, nr = 8)
    end
    @test !contains(default_output, "SHTnsKit Configuration Summary")

    verbose_output = capture_stdout() do
        GeoDynamo.create_shtnskit_config(
            lmax = 3, mmax = 3, nlat = 8, nlon = 12, nr = 8, verbose = true)
    end
    @test contains(verbose_output, "SHTnsKit Configuration Summary") ==
          (GeoDynamo.get_rank() == 0)
end

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
