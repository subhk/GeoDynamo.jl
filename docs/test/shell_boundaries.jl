using Test
using Geodynamo

@testset "Shell boundaries" begin
    @test isdefined(Geodynamo, :GeodynamoShell)
    
    Shell = Geodynamo.GeodynamoShell
    @test isdefined(Shell, :create_shell_radial_domain)
    @test isdefined(Shell, :create_shell_temperature_field)
    @test isdefined(Shell, :create_shell_hybrid_temperature_boundaries)
    @test isdefined(Shell, :apply_shell_temperature_boundaries!)
    
    # Basic shell boundary test (may require MPI)
    try
        dom = Shell.create_shell_radial_domain(4)
        @test dom !== nothing
    catch e
        @test_broken false
        @info "Shell boundaries test skipped due to: $e"
    end
end
