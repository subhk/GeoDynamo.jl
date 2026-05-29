using Test

const LOOP_MACRO_STATIC_SRC = normpath(joinpath(@__DIR__, "..", "src"))

function __loop_macro_source(path_parts...)
    return read(joinpath(LOOP_MACRO_STATIC_SRC, path_parts...), String)
end

function __loop_macro_function_body(source::String, signature::String)
    start = findfirst(signature, source)
    start === nothing && error("Could not find function signature: $signature")
    next_function = findnext("\nfunction ", source, last(start) + 1)
    return next_function === nothing ? source[first(start):end] : source[first(start):first(next_function)-1]
end

@testset "Local spectral loop macro source checks" begin
    numerics = __loop_macro_source("solver", "numerics.jl")
    @test occursin("macro solver_local_spectral_modes", numerics)
    @test occursin("macro solver_threaded_local_spectral_modes", numerics)

    vorticity_body = __loop_macro_function_body(
        numerics,
        "function compute_vorticity_spectral!(",
    )
    @test occursin("@solver_threaded_local_spectral_modes", vorticity_body)
    @test !occursin("Threads.@threads for lm_idx in lm_range", vorticity_body)

    curl_body = __loop_macro_function_body(
        numerics,
        "function spectral_curl_torpol!(",
    )
    @test occursin("@solver_local_spectral_modes", curl_body)
    @test !occursin("for lm_idx in lm_range", curl_body)
end
