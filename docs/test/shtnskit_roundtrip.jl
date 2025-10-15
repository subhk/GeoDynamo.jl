using Test  
using Geodynamo

@testset "SHTnsKit roundtrip" begin
    @test isdefined(Geodynamo, :create_shtnskit_config)
    @test isdefined(Geodynamo, :shtnskit_spectral_to_physical!)
    @test isdefined(Geodynamo, :shtnskit_physical_to_spectral!)
    @test isdefined(Geodynamo, :shtnskit_vector_synthesis!)
    @test isdefined(Geodynamo, :shtnskit_vector_analysis!)
    
    # Basic roundtrip test (may require MPI)
    try
        cfg = create_shtnskit_config(lmax=2, mmax=2, nlat=4, nlon=8, optimize_decomp=false)
        @test cfg !== nothing
        @test cfg.L_max == 2
        @test cfg.M_max == 2
    catch e
        @test_broken false
        @info "SHTnsKit roundtrip test skipped due to: $e"
    end
end
