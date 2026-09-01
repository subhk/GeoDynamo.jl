using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
import KernelAbstractions

@testset "GPU Architecture (Oceananigans style)" begin
    @testset "Architecture type hierarchy" begin
        @test CPU() isa AbstractArchitecture
        @test GPU{Nothing} <: AbstractArchitecture

        # Old global state is gone (not in this task, just prep for future tests)
        @test Base.isexported(GeoDynamo, :AbstractArchitecture)
        @test Base.isexported(GeoDynamo, :CPU)
        @test Base.isexported(GeoDynamo, :GPU)
        @test Base.isexported(GeoDynamo, :arch_zeros)
        @test Base.isexported(GeoDynamo, :on_architecture)
        @test Base.isexported(GeoDynamo, :get_backend)
    end

    @testset "CUDA extension pins the vendor for vector transforms" begin
        # SHTnsKit v2's `GPU()` is vendor-neutral: with both CUDA.jl and
        # AMDGPU.jl functional, `_gpu_adapter(nothing)` throws
        # "multiple functional GPU adapters are loaded". `synthesis_sphtor` /
        # `analysis_sphtor` auto-select a vendor only from inputs that are
        # already device arrays -- true of the `::CUDA.CuArray` methods, but not
        # of the `register_gpu_backend!` hooks, which receive host arrays. Those
        # must name the vendor with `prototype=`.
        ext_source = read(
            joinpath(@__DIR__, "..", "ext", "GeoDynamoCUDAExt.jl"), String)
        reg_start = findfirst("vector_synthesis=", ext_source)
        reg_stop = findnext("scratch_zeros=", ext_source, last(reg_start))
        @test reg_start !== nothing && reg_stop !== nothing
        registration = ext_source[first(reg_start):first(reg_stop)]
        @test occursin("gpu_synthesis_sphtor", registration)
        @test occursin("gpu_analysis_sphtor", registration)
        @test count("prototype=", registration) == 2
    end

    @testset "arch_zeros on CPU" begin
        a = arch_zeros(CPU(), Float64, 3, 4)
        @test a == zeros(Float64, 3, 4)
        @test a isa Matrix{Float64}
    end

    @testset "on_architecture CPU returns Array" begin
        a = [1, 2, 3]
        @test on_architecture(CPU(), a) isa Array
        @test on_architecture(CPU(), a) == a
    end

    @testset "get_backend CPU" begin
        @test get_backend(CPU()) isa KernelAbstractions.CPU
    end

    @testset "Architecture symbol conversion honors GPU() contract" begin
        @test GeoDynamo.architecture_from_symbol(:cpu) isa CPU
        @test_throws ArgumentError GeoDynamo.architecture_from_symbol(:cuda)
        if GeoDynamo.gpu_functional()
            g = GeoDynamo.architecture_from_symbol(:gpu)
            @test g isa GPU
            @test g.backend !== nothing
        else
            # No CUDA: :gpu must error like GPU(), not silently return a
            # CPU-backed GPU(nothing) that downgrades the run to CPU.
            @test_throws Exception GeoDynamo.architecture_from_symbol(:gpu)
        end
    end

    @testset ":gpu backend never silently downgrades to CPU" begin
        params = GeoDynamo.SolverParameters(architecture = :gpu,
            lmax = 4, mmax = 4, nlat = 8, nlon = 16, nr = 4, nr_inner = 2,
            include_composition = false)
        if GeoDynamo.gpu_functional()
            be = GeoDynamo.create_solver_backend(params)
            @test be.architecture isa GPU
            @test be.architecture.backend !== nothing
        else
            @test_throws Exception GeoDynamo.create_solver_backend(params)
        end
    end

    @testset "SolverBackend show does not repeat the architecture" begin
        # `create_solver_runtime` overwrites `_buffers.transform_device` with the
        # backend's own architecture, so a "compute device" row printed from it
        # just restates the "architecture" row above it. Report something that
        # actually differs: whether the allocation-free transform plan was built.
        params = GeoDynamo.SolverParameters(architecture = :cpu,
            lmax = 4, mmax = 4, nlat = 8, nlon = 16, nr = 8, nr_inner = 2,
            include_composition = false)
        backend = GeoDynamo.create_solver_backend(params)
        rendered = sprint(show, MIME("text/plain"), backend)

        @test occursin("architecture", rendered)
        @test !occursin("compute device", rendered)
        @test occursin("transform plan", rendered)
    end

    @testset "backend build preserves concrete arch object" begin
        # A GPU arch carrying a sentinel backend the Symbol path cannot
        # reproduce (it can only build GPU() or the old GPU(nothing)). The
        # object-taking backend builder — which the model/init path now routes
        # through — must store it verbatim instead of rebuilding from the
        # `params.architecture` Symbol.
        sentinel = GPU(:sentinel_backend)
        params = GeoDynamo.SolverParameters(architecture = :gpu,
            lmax = 4, mmax = 4, nlat = 8, nlon = 16, nr = 8, nr_inner = 2,
            include_composition = false)
        backend = GeoDynamo.create_solver_backend(sentinel, params)
        @test backend.architecture === sentinel
        # initialize_solver_state must accept the `arch` passthrough that the
        # model build relies on (regression guard for the kwarg itself).
        @test hasmethod(GeoDynamo.initialize_solver_state, Tuple{Type{Float64}},
            (:params, :arch))
    end

    @testset "GeodynamoModel preserves grid GPU backend [GPU]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            grid = SphericalShellGrid(GPU(); lmax = 4, nr = 4)
            model = GeodynamoModel(grid; include_composition = false)
            @test model.state.backend.architecture isa GPU
            @test model.state.backend.architecture.backend !== nothing
        end
    end

    @testset "GPU{B} parametric" begin
        @test GPU{Nothing} <: AbstractArchitecture
        g = GPU(nothing)
        @test g isa GPU
        @test g.backend === nothing
    end
end
