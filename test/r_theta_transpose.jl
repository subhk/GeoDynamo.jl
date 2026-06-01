using Test, GeoDynamo, MPI, PencilArrays

MPI.Initialized() || MPI.Init()

@testset "r<->lm transpose roundtrip is identity" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=6)

    # --- Create array in solve orientation (spec: l/θ_ranks, m/r_ranks, r local) ---
    a = GeoDynamo.create_pencil_array(ComplexF64, cfg.pencils.spec; init=:zero)
    rank_offset = ComplexF64(MPI.Comm_rank(MPI.COMM_WORLD) + 1)
    p = parent(a)
    for i in eachindex(p)
        p[i] = rank_offset + im * ComplexF64(i)
    end
    a0 = copy(parent(a))

    # --- spec_transform pencil must exist in cfg.pencils ---
    @test :spec_transform in keys(cfg.pencils)

    # --- spec_transform: m full (axis 2 local), r distributed (axis 2 of topology) ---
    bl = PencilArrays.size_local(cfg.pencils.spec_transform)
    @test bl[2] == PencilArrays.size_global(cfg.pencils.spec_transform)[2]   # m axis is full/local

    # --- Roundtrip via transpose helpers ---
    b = GeoDynamo.create_pencil_array(ComplexF64, cfg.pencils.spec_transform; init=:zero)
    GeoDynamo.transpose_solve_to_transform!(b, a)    # spec -> spec_transform
    GeoDynamo.transpose_transform_to_solve!(a, b)    # back to spec
    @test parent(a) == a0                             # exact identity (no floating-point error)
end
