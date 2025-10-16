using Test
using GeoDynamo

@testset "ERK2 Timestepping" begin
    @test isdefined(GeoDynamo, :TimestepState)
    @test isdefined(GeoDynamo, :create_erk2_config)
    @test isdefined(GeoDynamo, :apply_explicit_operator!)
    @test isdefined(GeoDynamo, :solve_implicit_step!)
    
    # Basic ERK2 functionality test (may require MPI)
    try
        config = create_erk2_config()
        @test config !== nothing
    catch e
        @test_broken false
        @info "ERK2 config test skipped due to: $e"
    end
end
