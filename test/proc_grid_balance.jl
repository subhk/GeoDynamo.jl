using Test
using GeoDynamo

# Process-grid balance detection: the spectral pencil splits l (over r_ranks) and m
# (over θ_ranks) into CONTIGUOUS slot blocks, but triangular SH truncation makes the
# mode COUNT per block uneven — and a square grid can leave whole ranks with zero
# modes (the high-m / low-l corner has no m<=l pairs). spectral_mode_counts reports
# the per-rank mode count so setup can warn about idle ranks / imbalance.

@testset "spectral_mode_counts: idle ranks + imbalance detection" begin
    lmax = mmax = 85
    nlm = (lmax + 1) * (lmax + 2) ÷ 2   # = 3741 for mmax == lmax

    # Serial: one rank owns every mode.
    @test GeoDynamo.spectral_mode_counts(1, 1, lmax, mmax) == [nlm]

    # 2x2: the high-m / low-l corner rank owns ZERO modes.
    c22 = GeoDynamo.spectral_mode_counts(2, 2, lmax, mmax)
    @test length(c22) == 4
    @test sum(c22) == nlm
    @test minimum(c22) == 0

    # 4x1: no idle rank, but a substantial (>1.5x) mode-load imbalance.
    c41 = GeoDynamo.spectral_mode_counts(4, 1, lmax, mmax)
    @test length(c41) == 4
    @test sum(c41) == nlm
    @test minimum(c41) > 0
    @test maximum(c41) / (sum(c41) / 4) > 1.5

    # 8x1: still no idle rank, larger imbalance.
    c81 = GeoDynamo.spectral_mode_counts(8, 1, lmax, mmax)
    @test sum(c81) == nlm
    @test minimum(c81) > 0
end

# A rank with zero SLOTS is not the same as a rank with zero MODES. Over-decomposing
# an axis past its global size gives PencilArrays an EMPTY local range (e.g. `1:0`),
# and the run then deadlocks: reproduced at 4 ranks with GEODYNAMO_PROC_GRID=4x1 on
# lmax=mmax=1, nlat=4, nlon=8, nr=8 — spec `axes_local` came back as (1:2, 1:0, 1:8)
# on rank 0 and (1:2, 2:1, 1:8) on rank 2, `time_step!` completed on every rank, and
# `run!` never returned (killed by MPIEXEC_TIMEOUT after 240 s, no error). The same
# grid runs to completion at 1 and 2 ranks. `validate_proc_grid` already names the
# condition exactly ("ranks with no m-modes"), so it must refuse rather than warn.
@testset "validate_proc_grid refuses a decomposition that empties an axis" begin
    # the reproduced deadlock: theta_ranks=4 > mmax+1=2
    @test_throws ArgumentError GeoDynamo.validate_proc_grid(
        4, 1; nlat = 4, nr = 8, lmax = 1, mmax = 1)
    # each empty-axis condition on its own
    @test_throws ArgumentError GeoDynamo.validate_proc_grid(   # > nlat
        16, 1; nlat = 4, nr = 8, lmax = 32, mmax = 32)
    @test_throws ArgumentError GeoDynamo.validate_proc_grid(   # > mmax+1
        6, 1; nlat = 10, nr = 8, lmax = 4, mmax = 4)
    @test_throws ArgumentError GeoDynamo.validate_proc_grid(   # > nr
        1, 16; nlat = 10, nr = 8, lmax = 32, mmax = 32)
    @test_throws ArgumentError GeoDynamo.validate_proc_grid(   # > lmax+1
        1, 6; nlat = 10, nr = 8, lmax = 4, mmax = 4)

    # every grid the MPI gates actually run must stay legal
    for (t, r) in ((4, 1), (1, 4), (2, 2))
        @test GeoDynamo.validate_proc_grid(t, r; nlat = 10, nr = 8, lmax = 4, mmax = 4) ===
              nothing
    end
    @test GeoDynamo.validate_proc_grid(1, 1; nlat = 1, nr = 1, lmax = 0, mmax = 0) === nothing

    # ... in particular 2x2 at lmax=mmax=4, which DOES leave the high-m/low-l corner
    # rank with zero valid modes and nevertheless runs bit-exact in the gates. A
    # zero-mode rank stays a warning; only an empty slot range is fatal.
    @test minimum(GeoDynamo.spectral_mode_counts(2, 2, 4, 4)) == 0
    @test GeoDynamo.validate_proc_grid(2, 2; nlat = 10, nr = 8, lmax = 4, mmax = 4) === nothing
end

# The check is worthless if the live decomposition path never calls it.
# `create_pencil_topology` (parallel/pencils.jl) does, but that is the Shell/Ball entry
# point; a `SphericalShellGrid` + `GeodynamoModel` run reaches
# `create_pencil_decomposition_shtnskit` (transforms/spectral.jl) instead — which read
# `read_proc_grid` and built the MPITopology without validating it. That is why the 4x1
# lmax=mmax=1 deadlock reproduced with no GeoDynamo grid warning in the log at all.
@testset "the live decomposition path validates its process grid" begin
    root = pkgdir(GeoDynamo)
    for (file, fn) in (
            (joinpath("src", "transforms", "spectral.jl"),
                "create_pencil_decomposition_shtnskit"),
            (joinpath("src", "parallel", "pencils.jl"), "create_pencil_topology"))
        src = read(joinpath(root, file), String)
        body = match(Regex("function $(fn)\\((?s:.*?)\\nend\\n"), src)
        @test body !== nothing
        @test occursin("read_proc_grid", body.match)
        @test occursin("validate_proc_grid", body.match)
    end
end
