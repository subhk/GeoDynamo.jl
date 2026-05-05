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

    simulation = _allocation_static_source("api", "simulation.jl")
    @test !occursin("callbacks      :: Vector{Any}", simulation)
    @test !occursin("output_writers :: Vector{Any}", simulation)
    @test !occursin("collect(Any, callbacks)", simulation)
    @test !occursin("collect(Any, output_writers)", simulation)

    file_bc_loader = _allocation_static_source("bcs", "file_bc_loader.jl")
    @test occursin("mutable struct BoundaryInterpolationCache{T", file_bc_loader)

    solver_numerics = _allocation_static_source("solver", "numerics.jl")
    @test !occursin("cache[\"bc_real\"]::Matrix", solver_numerics)
    @test !occursin("cache[\"bc_imag\"]::Matrix", solver_numerics)

    temperature_field = _allocation_static_source("physics", "temperature", "field.jl")
    composition_field = _allocation_static_source("physics", "composition", "field.jl")
    @test !occursin("boundary_interpolation_cache::Dict{String, Any}", temperature_field)
    @test !occursin("boundary_interpolation_cache::Dict{String, Any}", composition_field)
    @test !occursin("sum(temp_data.^2)", temperature_field)
    @test !occursin("sum(comp_data.^2)", composition_field)

    spectral = _allocation_static_source("transforms", "spectral.jl")
    @test occursin("abstract type AbstractTransformWorkspace end", spectral)
    @test occursin("struct SHTnsKitConfig{T", spectral)
    @test occursin("T::Type{T}", spectral)
    @test !occursin("pencils::NamedTuple", spectral)
    @test !occursin("fft_plans::Dict{Symbol, Any}", spectral)
    @test !occursin("transpose_plans::Dict{Symbol, Any}", spectral)
    @test !occursin("solver_transform_workspace :: Any", spectral)
    @test !occursin("transform_device         :: Any", spectral)
    @test !occursin("spatial_scratch          :: Any", spectral)
    @test !occursin("fft_scratch              :: Any", spectral)
end
