using Test

@testset "Reviewed legacy paths stay removed" begin
    src = normpath(joinpath(@__DIR__, "..", "src"))
    transforms_path = joinpath(src, "fields", "transforms.jl")
    transforms = read(transforms_path, String)

    for legacy in (
        "function perform_synthesis_phi_local!",
        "function perform_synthesis_with_transpose!",
        "function perform_synthesis_to_phi_pencil!",
        "function perform_analysis_phi_local!",
        "function perform_analysis_with_transpose!",
        "function perform_analysis_from_phi_pencil!",
    )
        @test !contains(transforms, legacy)
    end

    for relpath in (
        joinpath("physics", "velocity", "solver.jl"),
        joinpath("physics", "magnetic", "solver.jl"),
        joinpath("physics", "temperature", "solver.jl"),
        joinpath("physics", "composition", "solver.jl"),
    )
        @test !contains(read(joinpath(src, relpath), String), "Vector{Function}")
    end

    # The ERK2 helper must prepare real cached infrastructure, not re-enable
    # disabled PLM tables or time three fills plus barriers as "transforms".
    optimizer_start = findfirst("function optimize_erk2_transforms!", transforms)
    optimizer_stop = findnext("function create_erk2_config", transforms,
        last(optimizer_start) + 1)
    optimizer = transforms[first(optimizer_start):(first(optimizer_stop) - 1)]
    @test !contains(optimizer, "prepare_plm_tables!")
    @test !contains(optimizer, "MPI.Barrier")
    @test contains(optimizer, "get_disttranspose_plan(config)")
    @test !contains(transforms, "function optimize_fft_performance!")

    # The public single-step API must use the same rollback boundary as the
    # Simulation constructor when changing dt-dependent state.
    simulation = read(joinpath(src, "api", "simulation.jl"), String)
    step_start = findfirst("function time_step!", simulation)
    step_stop = findnext("\nend", simulation, last(step_start) + 1)
    step_body = simulation[first(step_start):last(step_stop)]
    @test contains(step_body, "_commit_run_controls!")
end
