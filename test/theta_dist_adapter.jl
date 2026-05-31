using Test, GeoDynamo, SHTnsKit, MPI
MPI.Initialized() || MPI.Init()
@testset "spectral adapter round-trip (dense seam)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=4)
    alm_mat = zeros(ComplexF64, 9, 9)
    alm_mat[3,1] = 1.0
    alm_mat[5,2] = 0.5 + 0.2im
    sht_alm = GeoDynamo.to_sht_spectral(cfg, alm_mat)
    back    = GeoDynamo.from_sht_spectral(cfg, sht_alm)
    @test back ≈ alm_mat
    @test eltype(sht_alm) == ComplexF64
end
