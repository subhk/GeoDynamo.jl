# ================================================================================
# Collective discipline — serial contracts of src/parallel/collectives.jl and
# src/api/registry.jl. The multi-rank half lives in mpi_control_plane_invariants.jl.
# ================================================================================
using Test
using MPI
using GeoDynamo

@testset "collectives.jl serial fast paths" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    if MPI.Comm_size(comm) > 1
        @test_skip "serial contracts are exercised at one rank"
    else
        @test GeoDynamo._collective_active(nothing) == false
        @test GeoDynamo._collective_active(comm) == false

        @test GeoDynamo.global_sum(2.5, comm) === 2.5
        @test GeoDynamo.global_max(3, comm) === 3
        @test GeoDynamo.global_min(-1.0, comm) === -1.0
        @test GeoDynamo.global_sum(7) === 7            # default comm
        buf = [1.0, 2.0]
        @test GeoDynamo.global_sum!(buf, comm) === buf
        @test buf == [1.0, 2.0]

        @test GeoDynamo.any_rank(true, comm) === true
        @test GeoDynamo.any_rank(false, comm) === false
        @test GeoDynamo.all_ranks(true, comm) === true
        @test GeoDynamo.all_ranks(false, comm) === false

        @test GeoDynamo.root_value(() -> "tmp", comm) == "tmp"
        @test GeoDynamo.root_value(() -> 42) == 42
        err = try
            GeoDynamo.root_value(() -> error("boom"), comm, "probe")
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("probe failed on rank 0: boom", sprint(showerror, err))
        @test GeoDynamo.run_on_root!(() -> nothing, comm, "noop") === nothing

        flags = Int[1, 0]
        @test GeoDynamo.root_broadcast!(flags, comm) === flags
        @test GeoDynamo.gather_root(5, comm) == [5]
        @test GeoDynamo.allgather(Int32(9), comm) == Int32[9]

        send = ComplexF64[1, 2, 3]
        recv = zeros(ComplexF64, 3)
        vbuf = MPI.VBuffer(recv, [3])
        @test GeoDynamo.allgatherv!(send, vbuf, comm) === recv
        @test recv == send

        sbuf = MPI.VBuffer(ComplexF64[4, 5], [2])
        rbuf = MPI.VBuffer(zeros(ComplexF64, 2), [2])
        GeoDynamo.alltoallv!(sbuf, rbuf, comm)
        @test rbuf.data == ComplexF64[4, 5]

        @test GeoDynamo.barrier(comm) === nothing
        @test GeoDynamo.barrier(nothing) === nothing
    end
end

@testset "threaded-update guard lives with the collectives" begin
    # Serial fast path never trips the guard (no MPI op is about to be issued) …
    GeoDynamo._with_threaded_update_guard() do
        @test GeoDynamo._in_threaded_implicit_update()
        @test GeoDynamo.global_sum(1.0, nothing) === 1.0
    end
    @test !GeoDynamo._in_threaded_implicit_update()
    # … while the kernel-facing numerics delegates still assert up front, as before.
    GeoDynamo._with_threaded_update_guard() do
        @test_throws ErrorException GeoDynamo.allreduce_sum(1.0)
    end
end

@testset "control-plane sites use the collectives helpers" begin
    root = normpath(joinpath(@__DIR__, ".."))
    src(rel) = read(joinpath(root, "src", rel), String)
    # the pre-refactor names are gone (built from symbols so a blanket rename cannot
    # rewrite this assertion into its opposite)
    @test !isdefined(GeoDynamo, Symbol("_any_rank", "_flag"))
    @test !isdefined(GeoDynamo, Symbol("_all_ranks", "_flag"))
    @test !isdefined(GeoDynamo, Symbol("_run_on_root", "_collectively!"))
    @test occursin("root_broadcast!(buf, comm)", src("api/schedules.jl"))
    @test occursin("any_rank(missing_here, comm)", src("io/restart.jl"))
    @test occursin("root_value(", src("io/restart.jl"))
    @test occursin("root_value(", src("api/output_writers.jl"))
    @test occursin("root_broadcast!(flags, comm)", src("io/history.jl"))
    @test occursin("all_ranks(", src("io/netcdf.jl"))
    @test occursin("global_sum(local_energy", src("io/config.jl"))
    # `_collective_scan_output_count` and `_existing_grid_file` were the two
    # root-only I/O sites still hand-rolling the try/bcast pattern.
    @test !occursin("scan_error = MPI.bcast", src("io/restart.jl"))
    @test !occursin("MPI.Bcast!(buffer, 0, comm)", src("api/output_writers.jl"))
end

using OrderedCollections: OrderedDict

@testset "CollectiveRegistry" begin
    R = GeoDynamo.CollectiveRegistry{:callback}
    reg = R()
    @test isempty(reg) && length(reg) == 0
    @test !GeoDynamo.isfrozen(reg)

    GeoDynamo.register!(reg, :a, 1)
    GeoDynamo.register!(reg, :b, 2)
    @test reg[:a] == 1 && haskey(reg, :b) && get(reg, :c, nothing) === nothing
    @test collect(keys(reg)) == [:a, :b]
    @test collect(values(reg)) == [1, 2]
    @test collect(pairs(reg)) == [:a => 1, :b => 2]
    @test [k for (k, _) in reg] == [:a, :b]            # iterable like a dict
    @test length(reg) == 2
    GeoDynamo.register!(reg, :a, 10)                    # overwrite allowed before freeze
    @test reg[:a] == 10

    e = try; reg[:c] = 3; nothing; catch err; err; end
    @test e isa ArgumentError && occursin("add_callback!", sprint(showerror, e))
    e = try; delete!(reg, :a); nothing; catch err; err; end
    @test e isa ArgumentError && occursin("add_callback!", sprint(showerror, e))

    # one rank: validate_and_freeze! freezes without touching MPI
    calls = Ref(0)
    sig(r) = (calls[] += 1; repr(collect(keys(r))))
    GeoDynamo.validate_and_freeze!(reg, sig, "mismatch")
    @test GeoDynamo.isfrozen(reg)
    @test calls[] == 0 || calls[] == 1                  # serial path may skip the signature
    e = try; GeoDynamo.register!(reg, :late, 0); nothing; catch err; err; end
    @test e isa ArgumentError && occursin("frozen", sprint(showerror, e))
    @test !haskey(reg, :late)

    # built from an existing OrderedDict (what _to_ordered returns)
    seeded = GeoDynamo.CollectiveRegistry{:writer}(OrderedDict{Symbol, Any}(:w => 1))
    @test seeded[:w] == 1 && !GeoDynamo.isfrozen(seeded)
end

@testset "Simulation registries freeze on first step" begin
    MPI.Initialized() || MPI.Init()
    grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU(); lmax = 4, mmax = 4,
        nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
    mk() = GeoDynamo.Simulation(GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4);
        Δt = 1e-4, stop_iteration = 2)

    sim = mk()
    @test sim.callbacks isa GeoDynamo.CollectiveRegistry{:callback}
    @test sim.output_writers isa GeoDynamo.CollectiveRegistry{:writer}
    @test GeoDynamo.isfrozen(sim.output_writers)          # fixed at construction
    @test !GeoDynamo.isfrozen(sim.callbacks)
    @test haskey(sim.callbacks, :nan_checker)              # defaults survived the wrap
    @test sim._stop_needs_reduce == true                   # conservative until frozen

    fired = Ref(0)
    GeoDynamo.add_callback!(sim, s -> (fired[] += 1);
        schedule = GeoDynamo.IterationInterval(1), name = :counter)
    GeoDynamo.add_callback!(sim,
        GeoDynamo.Callback(s -> nothing, GeoDynamo.IterationInterval(1)); name = :prebuilt)
    @test haskey(sim.callbacks, :prebuilt)

    GeoDynamo.time_step!(sim)
    @test GeoDynamo.isfrozen(sim.callbacks)
    @test fired[] == 1
    @test sim._stop_needs_reduce == true                   # user callbacks may stop rank-locally
    e = try
        GeoDynamo.add_callback!(sim, s -> nothing; schedule = GeoDynamo.IterationInterval(1))
        nothing
    catch err
        err
    end
    @test e isa ArgumentError && occursin("frozen", sprint(showerror, e))
    e = try; sim.callbacks[:x] = 1; nothing; catch err; err; end
    @test e isa ArgumentError

    # run! freezes too, and a second run! on a frozen registry is fine
    sim2 = mk()
    GeoDynamo.run!(sim2)
    @test GeoDynamo.isfrozen(sim2.callbacks)
    @test sim2._stop_needs_reduce == false                 # default registry: no per-step Allreduce
    @test sim2.model.clock.iteration == 2
    GeoDynamo.run!(sim2)
    @test sim2.model.clock.iteration == 2
end
