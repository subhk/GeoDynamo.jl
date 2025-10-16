using Test
using GeoDynamo
const Shell = GeoDynamo.GeoDynamoShell

@testset "Shell programmatic boundary application" begin
    # Small config
    lmax = 6; mmax = 6
    nlat = max(lmax + 2, 12)
    nlon = max(2lmax + 1, 24)
    nr   = 6

    cfg = GeoDynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = Shell.create_shell_radial_domain(nr)
    temp = Shell.create_shell_temperature_field(Float64, cfg; nr=nr)

    # Programmatic uniform boundaries
    inner_val = 100.0
    outer_val = 250.0
    bset = Shell.create_shell_hybrid_temperature_boundaries((:uniform, inner_val), (:uniform, outer_val), cfg)
    Shell.apply_shell_temperature_boundaries!(temp, bset; time=0.0)

    # Check BC types are Dirichlet (1)
    @test all(temp.bc_type_inner .== 1)
    @test all(temp.bc_type_outer .== 1)

    # For uniform patterns, only l=0 should carry significant magnitude.
    # We compare entries relative to the maximum to avoid exact normalization issues.
    max_inner = maximum(abs.(temp.boundary_values[1, :]))
    max_outer = maximum(abs.(temp.boundary_values[2, :]))
    @test max_inner > 0
    @test max_outer > 0

    for (lm_idx, l) in enumerate(cfg.l_values)
        if l > 0
            @test abs(temp.boundary_values[1, lm_idx]) ≤ 1e-8 * max(1.0, max_inner)
            @test abs(temp.boundary_values[2, lm_idx]) ≤ 1e-8 * max(1.0, max_outer)
        end
    end
end
