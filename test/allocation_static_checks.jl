using Test

const ALLOCATION_STATIC_SRC = normpath(joinpath(@__DIR__, "..", "src"))

function _allocation_static_source(path_parts...)
    return read(joinpath(ALLOCATION_STATIC_SRC, path_parts...), String)
end

function _allocation_static_function_body(source::String, signature::String)
    start = findfirst(signature, source)
    start === nothing && error("Could not find function signature: $signature")
    next_function = findnext("\nfunction ", source, last(start) + 1)
    return next_function === nothing ? source[first(start):end] : source[first(start):first(next_function)-1]
end

@testset "Hot-path allocation source checks" begin
    numerics = _allocation_static_source("solver", "numerics.jl")
    vorticity_body = _allocation_static_function_body(
        numerics,
        "function solver_compute_vorticity_spectral!(",
    )
    @test occursin("get_velocity_workspace(T)", vorticity_body)
    @test !occursin("[zeros(T, nr) for _ in 1:nthreads]", vorticity_body)

    backend = _allocation_static_source("solver", "backend.jl")
    @test occursin("theta_full_real::Vector{T}", backend)
    @test occursin("theta_full_imag::Vector{T}", backend)

    nonlinear = _allocation_static_source("physics", "nonlinear.jl")
    theta_body = _allocation_static_function_body(
        nonlinear,
        "function solver_compute_theta_gradient_spectral!(",
    )
    @test occursin("ws.theta_full_real", theta_body)
    @test !occursin("full_real = zeros(T, nlm)", theta_body)
    @test !occursin("full_imag = zeros(T, nlm)", theta_body)

    scalar_transform_body = _allocation_static_function_body(
        nonlinear,
        "function solver_transform_field_and_gradients_to_physical!(",
    )
    @test !occursin("spectral_fields = [", scalar_transform_body)
    @test !occursin("physical_fields = [", scalar_transform_body)

    state = _allocation_static_source("solver", "state.jl")
    imex = _allocation_static_source("timestep", "imex.jl")
    velocity_solver = _allocation_static_source("physics", "velocity", "solver.jl")
    @test occursin("struct SolverRadialWork{T}", state)
    @test occursin("radial_work::Dict{Symbol, SolverRadialWork{T}}", state)
    @test occursin("work::Union{SolverRadialWork{T}, Nothing}=nothing", imex)
    @test occursin("solver_get_radial_work!", velocity_solver)
end
