using Test
using GeoDynamo
using NCDatasets

@testset "find_restart_files honors target_time" begin
    mktempdir() do dir
        function write_rf(name, t)
            p = joinpath(dir, name)
            NCDataset(p, "c") do ds
                defDim(ds, "time", 1)
                v = defVar(ds, "time", Float64, ("time",))
                v[1] = t
            end
            return p
        end

        # p1 is written first (older mtime) but holds sim-time 1.0;
        # p2 is newer by mtime and holds sim-time 10.0.
        p1 = write_rf("geodynamo_shell_restart_1.nc", 1.0)
        sleep(0.05)
        p2 = write_rf("geodynamo_shell_restart_2.nc", 10.0)

        # A meaningful target_time selects by closeness to the stored sim-time,
        # not by mtime.
        @test first(GeoDynamo.find_restart_files(dir, 1.5)) == p1
        @test first(GeoDynamo.find_restart_files(dir, 9.0)) == p2

        # The 0.0 sentinel keeps "restart from the latest checkpoint" semantics
        # (newest by mtime).
        @test first(GeoDynamo.find_restart_files(dir, 0.0)) == p2

        # No matching files -> empty.
        @test isempty(GeoDynamo.find_restart_files(mktempdir(), 1.0))
    end
end
