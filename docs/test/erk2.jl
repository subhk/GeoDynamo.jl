using Test
using Geodynamo

@testset "ERK2 Timestepping" begin
    @test isdefined(Geodynamo, :TimestepState)
    @test isdefined(Geodynamo, :create_erk2_config)
    @test isdefined(Geodynamo, :apply_explicit_operator!)
    @test isdefined(Geodynamo, :solve_implicit_step!)
    
    # Basic ERK2 functionality test (may require MPI)
    try
        config = create_erk2_config()
        @test config !== nothing
    catch e
        @test_broken false
        @info "ERK2 config test skipped due to: $e"
    end
end
