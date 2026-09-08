using Test
using MPI
using GeoDynamo
using SHTnsKit

@testset "SHTnsKit v2 compatibility" begin
    @test v"2.0.2" <= Base.pkgversion(SHTnsKit) < v"3.0.0"

    removed_symbols = (
        "create_gauss_config_gpu",
        "get_config_device",
        "is_gpu_config",
        "CUDA_DEVICE",
        "SH_to_grad_spat",
        "SHqst_to_spat",
        "spat_to_SHqst",
        "shtns_use_threads",
    )
    # Walk every production file, not a hand-listed few: src/parallel,
    # src/physics, src/gpu and src/bcs all call SHTnsKit directly, and a v1 API
    # reintroduced there is exactly the regression this guard exists to catch.
    production_files = String[]
    for root_dir in (joinpath(@__DIR__, "..", "src"), joinpath(@__DIR__, "..", "ext"))
        for (root, _, files) in walkdir(root_dir)
            for f in files
                endswith(f, ".jl") && push!(production_files, joinpath(root, f))
            end
        end
    end
    @test length(production_files) > 5

    # Match whole words on code only: a bare `occursin` flags `_is_gpu_config`
    # and prose mentions of a removed symbol in comments.
    patterns = [symbol => Regex("\\b\\Q$symbol\\E\\b") for symbol in removed_symbols]
    violations = String[]
    for path in production_files
        for (lineno, line) in enumerate(eachline(path))
            code = first(split(line, '#'; limit = 2))
            for (symbol, pattern) in patterns
                occursin(pattern, code) &&
                    push!(violations, "$(relpath(path, @__DIR__)):$lineno: $symbol")
            end
        end
    end
    @test isempty(violations)

    if MPI.Finalized()
        @warn "MPI already finalized; skipping SHTnsKit v2 runtime checks"
        return
    end
    MPI.Initialized() || MPI.Init()

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = 3,
        mmax = 3,
        nlat = 6,
        nlon = 10,
        nr = 4,
    )
    @test cfg._buffers.transform_device === :cpu
    @test GeoDynamo.transform_arch(cfg) isa GeoDynamo.CPU
    @test GeoDynamo.get_shtnskit_version_info().has_qst_transforms

    coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    coefficients[2, 1] = 1.0
    expected_theta, expected_phi =
        SHTnsKit.synthesis_grad(cfg.sht_config, coefficients; real_output = true)
    gradient_theta = fill(NaN, cfg.nlat, cfg.nlon)
    gradient_phi = fill(NaN, cfg.nlat, cfg.nlon)
    GeoDynamo.spectral_gradient!(
        cfg, coefficients, gradient_theta, gradient_phi)
    @test gradient_theta ≈ expected_theta
    @test gradient_phi ≈ expected_phi
    @test maximum(abs, gradient_theta) > 0

    # Independent anchor: comparing the wrapper against the very call it makes
    # cannot detect a convention change inside SHTnsKit. The coefficient above is
    # a pure l=1, m=0 mode, so with orthonormal normalisation the field is
    # Y_1^0 = sqrt(3/4pi) cos(theta) and its surface gradient is analytic.
    analytic_theta = [-sqrt(3 / (4 * pi)) * sin(t) for t in cfg.sht_config.θ]
    for j in 1:cfg.nlon
        @test gradient_theta[:, j] ≈ analytic_theta
    end
    @test all(iszero, gradient_phi)

    # Zero-initialise: synthesis_qst reads every valid (l, m) slot, so `similar`
    # would feed uninitialised heap into the transform.
    qlm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    slm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    tlm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    qlm[1, 1] = 0.25
    qlm[2, 1] = 0.5
    slm[2, 1] = -0.75
    tlm[2, 1] = 0.4

    expected_vr, expected_vtheta, expected_vphi =
        SHTnsKit.synthesis_qst(
            cfg.sht_config, qlm, slm, tlm; real_output = true)
    vr = similar(expected_vr)
    vtheta = similar(expected_vtheta)
    vphi = similar(expected_vphi)
    GeoDynamo.shtnskit_qst_to_spatial!(
        cfg, qlm, slm, tlm, vr, vtheta, vphi)
    @test vr ≈ expected_vr
    @test vtheta ≈ expected_vtheta
    @test vphi ≈ expected_vphi
    # A NaN on both sides satisfies neither of the above nor `≈`, but it reads as
    # an opaque "NaN ≈ NaN" failure. Say what actually went wrong.
    @test all(isfinite, vr)
    @test all(isfinite, vtheta)
    @test all(isfinite, vphi)

    # Independent anchor: the fused QST transform must agree with the separate
    # scalar + sphtor calls, which is the composition v1 actually executed.
    composed_vr = SHTnsKit.synthesis(cfg.sht_config, qlm; real_output = true)
    composed_vtheta, composed_vphi =
        SHTnsKit.synthesis_sphtor(cfg.sht_config, slm, tlm; real_output = true)
    @test vr ≈ composed_vr
    @test vtheta ≈ composed_vtheta
    @test vphi ≈ composed_vphi

    expected_q, expected_s, expected_t =
        SHTnsKit.analysis_qst(cfg.sht_config, vr, vtheta, vphi)
    analyzed_q = similar(qlm)
    analyzed_s = similar(slm)
    analyzed_t = similar(tlm)
    GeoDynamo.shtnskit_spatial_to_qst!(
        cfg, vr, vtheta, vphi, analyzed_q, analyzed_s, analyzed_t)
    @test analyzed_q ≈ expected_q
    @test analyzed_s ≈ expected_s
    @test analyzed_t ≈ expected_t
    @test all(isfinite, analyzed_q)
    @test all(isfinite, analyzed_s)
    @test all(isfinite, analyzed_t)

    # Round-trip anchor: analysing the synthesised field must return the modes it
    # was built from, independently of which SHTnsKit entry point produced them.
    @test analyzed_q[2, 1] ≈ qlm[2, 1]
    @test analyzed_s[2, 1] ≈ slm[2, 1]
    @test analyzed_t[2, 1] ≈ tlm[2, 1]

    # SHTnsKit v2 promotes the Gauss grid vectors to advertised properties
    # (`Base.propertynames` lists θ/φ/w), so reading the private `_grid` field --
    # and arming a silent uniform-grid fallback against an internal name -- is no
    # longer necessary.
    @test cfg.theta_grid ≈ collect(cfg.sht_config.θ)
    @test cfg.phi_grid ≈ collect(cfg.sht_config.φ)
    @test cfg.gauss_weights ≈ collect(cfg.sht_config.w)
    @test sum(cfg.gauss_weights) ≈ 2
    spectral_source = read(
        joinpath(@__DIR__, "..", "src", "transforms", "spectral.jl"), String)
    spectral_code = join(
        (first(split(line, '#'; limit = 2)) for line in eachline(IOBuffer(spectral_source))),
        "\n")
    grid_field_reads = count(_ -> true, eachmatch(r"\._grid\b", spectral_code))
    @test grid_field_reads == 0

    # `device = :auto` was accepted on the v1 path (it reached
    # create_gauss_config_gpu, which picked CUDA or CPU); it must keep resolving
    # rather than aborting the caller.
    auto_cfg = GeoDynamo.create_shtnskit_config(
        lmax = 3, mmax = 3, nlat = 6, nlon = 10, nr = 4, device = :auto)
    @test auto_cfg._buffers.transform_device in (:cpu, :gpu, :cuda)
    @test auto_cfg._buffers.transform_device ==
          (GeoDynamo.gpu_functional() ? :cuda : :cpu)
    @test_throws ArgumentError GeoDynamo.create_shtnskit_config(
        lmax = 3, mmax = 3, nlat = 6, nlon = 10, nr = 4, device = :tpu)

    # SHTnsKit v2 cannot change the thread count, but a mismatched request is
    # the caller's information problem, not grounds for aborting their run.
    @test GeoDynamo.set_shtnskit_threads(Threads.nthreads()) === nothing
    @test GeoDynamo.set_shtnskit_threads(Threads.nthreads() + 1) === nothing
    @test_throws ArgumentError GeoDynamo.set_shtnskit_threads(0)
end
