using Test, GeoDynamo, SHTnsKit, MPI, PencilArrays

const FINALIZE_MPI_P3_TRANSPOSE = !MPI.Initialized()
FINALIZE_MPI_P3_TRANSPOSE && MPI.Init()

@testset "Alm <-> spec_solve identity (dealiased)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=24, nr=8)  # dealiased nlon=24
    plan = GeoDynamo.get_disttranspose_plan(cfg)
    Alm = SHTnsKit.allocate_spectral(plan)
    p = parent(Alm)
    for i in eachindex(p)
        p[i] = ComplexF64(i + 100 * (MPI.Comm_rank(MPI.COMM_WORLD) + 1))
    end
    a0 = copy(p)
    solve = GeoDynamo.to_spec_solve(cfg, Alm, plan)
    GeoDynamo.from_spec_solve!(cfg, Alm, solve, plan)
    @test parent(Alm) == a0
end

FINALIZE_MPI_P3_TRANSPOSE && MPI.Finalize()
