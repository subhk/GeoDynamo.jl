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
    @test occursin("mutable struct SolverKrylovWork{T}", state)
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
    @test !occursin("get_cached_buffer!(create_func::Function", spectral)
    @test !occursin("field = get(_BUFFERS_FIELD_MAP", spectral)
    @test occursin("function get_cached_buffer!(create_func::F, config, ::Val{key}) where {F,key}", spectral)

    containers = _allocation_static_source("fields", "containers.jl")
    @test occursin("mutable struct SHTnsSpecField{", containers)
    @test occursin("C<:SHTnsKitConfig", containers)
    @test occursin("DR<:PencilArray{T,3}", containers)
    @test occursin("struct SHTnsPhysField{", containers)
    @test occursin("D<:PencilArray{T,3}", containers)
    @test !occursin("config::SHTnsKitConfig", containers)

    for source in (
        _allocation_static_source("physics", "temperature", "field.jl"),
        _allocation_static_source("physics", "composition", "field.jl"),
        _allocation_static_source("physics", "velocity", "field.jl"),
        _allocation_static_source("physics", "magnetic", "field.jl"),
    )
        @test !occursin("config::SHTnsKitConfig", source)
    end

    backend_state = _allocation_static_source("GeoDynamo.jl")
    @test occursin("struct GPUBackendState{", backend_state)
    @test !occursin("scalar_synthesis::Function", backend_state)
    @test !occursin("scratch_zeros::Function", backend_state)

    solver_backend = _allocation_static_source("solver", "backend.jl")
    @test occursin("struct SolverBackend{A<:AbstractArchitecture,C<:SHTnsConfigType", solver_backend)
    @test occursin("struct SolverRuntime{", solver_backend)
    @test occursin("C<:SHTnsConfigType", solver_backend)
    @test !occursin("shtns_config::SHTnsConfigType", solver_backend)

    imex = _allocation_static_source("timestep", "imex.jl")
    krylov_body = _allocation_static_function_body(
        imex,
        "function solver_eab2_update_krylov_cached!(",
    )
    @test occursin("krylov_work::Union{SolverRadialWork{T}, Nothing}=nothing", krylov_body)
    @test occursin("krylov_action_work = work_ok ? krylov_work.krylov : nothing", krylov_body)
    @test occursin("work=krylov_action_work", krylov_body)
    @test occursin("krylov_work=radial_work", velocity_solver)
    @test occursin("krylov_work=radial_work", _allocation_static_source("physics", "temperature", "solver.jl"))
    @test occursin("krylov_work=radial_work", _allocation_static_source("physics", "composition", "solver.jl"))
    @test occursin("krylov_work=radial_work", _allocation_static_source("physics", "magnetic", "solver.jl"))
    @test !occursin("u_real_next = exp_action_krylov", krylov_body)
    @test !occursin("nl_real_increment = solver_phi1_action_krylov", krylov_body)

    magnetic_field = _allocation_static_source("physics", "magnetic", "field.jl")
    @test occursin("curl_work::NTuple{6,Vector{T}}", magnetic_field)
    current_body = _allocation_static_function_body(
        magnetic_field,
        "function compute_current_density_spectral!(",
    )
    induction_body = _allocation_static_function_body(
        magnetic_field,
        "function compute_curl_of_induction!(",
    )
    @test occursin("_work=ℬ.curl_work", current_body)
    @test occursin("_work=ℬ.curl_work", induction_body)

    solver_current_body = _allocation_static_function_body(
        solver_numerics,
        "function solver_compute_current_density_spectral!(",
    )
    solver_induction_body = _allocation_static_function_body(
        solver_numerics,
        "function solver_compute_curl_of_induction!(",
    )
    @test occursin("_work=magnetic_fields.curl_work", solver_current_body)
    @test occursin("_work=magnetic_fields.curl_work", solver_induction_body)

    @test !occursin("solver_get_cached_buffer!(create_func::Function", nonlinear)
    @test !occursin("field = get(_SOLVER_BUFFERS_KEY_MAP", nonlinear)
    @test occursin("function solver_get_cached_buffer!(create_func::F, config, ::Val{key}) where {F,key}", nonlinear)

    scalar_ops = _allocation_static_source("fields", "scalar_operators.jl")
    flux_tau_body = _allocation_static_function_body(
        scalar_ops,
        "function apply_flux_bc_tau!(",
    )
    influence_body = _allocation_static_function_body(
        scalar_ops,
        "function apply_flux_bc_influence_matrix!(",
    )
    @test occursin("struct ScalarFluxBCWork{T}", scalar_ops)
    @test occursin("flux_work = ScalarFluxBCWork{T}(domain.N)", scalar_ops)
    @test occursin("work=nothing", flux_tau_body)
    @test occursin("work=nothing", influence_body)
    @test !occursin("profile_real = zeros(T, nr)", flux_tau_body)
    @test !occursin("profile_real = zeros(T, nr)", influence_body)

    solver_state = _allocation_static_source("solver", "state.jl")
    erk2 = _allocation_static_source("timestep", "erk2.jl")
    integrate_erk2_body = _allocation_static_function_body(
        erk2,
        "function integrate_solver_erk2_step!(",
    )
    finalize_erk2_body = _allocation_static_function_body(
        erk2,
        "function finalize_solver_erk2_field!(",
    )
    @test occursin("erk2_field_buffers::Dict{Symbol, SolverERK2FieldBuffers{T}}", solver_state)
    @test occursin("get_solver_erk2_field_buffers!", integrate_erk2_body)
    @test !occursin(" = SolverERK2FieldBuffers(", integrate_erk2_body)
    @test !occursin("result_real_profile = similar(result)", finalize_erk2_body)
    @test !occursin("get(buffers.cache_lookup, l, nothing)", erk2)

    temperature_solver = _allocation_static_source("physics", "temperature", "solver.jl")
    composition_solver = _allocation_static_source("physics", "composition", "solver.jl")
    magnetic_solver = _allocation_static_source("physics", "magnetic", "solver.jl")
    @test occursin("solver_solve_temperature_implicit_step!(\n            temperature.spectral,\n            temperature.nonlinear,\n            matrices;\n            bc_inner=bc.inner_real,\n            bc_outer=bc.outer_real,\n            bc_inner_imag=bc.inner_imag,\n            bc_outer_imag=bc.outer_imag,\n            work=radial_work,", temperature_solver)
    @test occursin("solver_solve_velocity_implicit_step!(\n            velocity.𝒯,\n            velocity.nlᵀ,\n            matrices,\n            :toroidal;\n            velocity_bc_code=velocity_bc,\n            domain=runtime.𝒟ᵒᶜ,\n            work=radial_work,", velocity_solver)
    @test occursin("solver_solve_velocity_implicit_step!(\n            velocity.𝒫,\n            velocity.nlᴾ,\n            matrices,\n            :poloidal;\n            velocity_bc_code=velocity_bc,\n            domain=runtime.𝒟ᵒᶜ,\n            work=radial_work,", velocity_solver)
    @test occursin("solver_solve_composition_implicit_step!(\n            composition.spectral,\n            composition.nonlinear,\n            matrices;\n            bc_inner=bc.inner_real,\n            bc_outer=bc.outer_real,\n            bc_inner_imag=bc.inner_imag,\n            bc_outer_imag=bc.outer_imag,\n            work=radial_work,", composition_solver)
    @test occursin("solver_solve_magnetic_implicit_step!(\n            magnetic.𝒯,\n            magnetic.nlᵀ,\n            matrices,\n            :toroidal;\n            mag_bc_inner=inner_bc === nothing ? nothing : inner_bc[1],\n            prev_bc_inner=inner_bc === nothing ? nothing : inner_bc[2],\n            mag_bc_inner_imag=inner_bc === nothing ? nothing : inner_bc[3],\n            prev_bc_inner_imag=inner_bc === nothing ? nothing : inner_bc[4],\n            work=radial_work,", magnetic_solver)
    @test occursin("solver_solve_magnetic_implicit_step!(\n            magnetic.𝒫,\n            magnetic.nlᴾ,\n            matrices,\n            :poloidal,\n            work=radial_work,", magnetic_solver)

    velocity_field = _allocation_static_source("physics", "velocity", "field.jl")
    @test !occursin("_compute_vorticity_spectral_threaded!", velocity_field)
end
