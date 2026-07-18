# r×θ equivalence cross-grid comparison
#
# Reads binary snapshots from RTHETA_TMPDIR (or /tmp) for each of the four
# grid layouts and asserts that all agree with the 1x1 reference to < 1e-10.
#
# Called by test/run_mpi_r_theta_equivalence.sh after all driver runs complete.
# Can also be run standalone:
#   RTHETA_TMPDIR=/path/to/snaps julia test/r_theta_compare_snapshots.jl

snap_dir = get(ENV, "RTHETA_TMPDIR", tempdir())

function read_snapshot(path)
    open(path, "r") do io
        nlm      = read(io, Int64)
        nr       = read(io, Int64)
        ntensors = read(io, Int64)
        tensors  = [read!(io, Array{Float64}(undef, nlm, nr)) for _ in 1:ntensors]
        return nlm, nr, tensors
    end
end

grids = ["1x1", "4x1", "1x4", "2x2"]
files = [joinpath(snap_dir, "rtheta_sig_$(g).bin") for g in grids]

for (g, f) in zip(grids, files)
    isfile(f) || error("Snapshot for $g not found: $f")
end

nlm_ref, nr_ref, refs = read_snapshot(files[1])
println("Reference (1x1): nlm=$nlm_ref, nr=$nr_ref")

tensor_names = ["temp_real", "temp_imag",
                "vel_tor_real", "vel_tor_imag",
                "vel_pol_real", "vel_pol_imag"]

all_pass = true
for (grid, file) in zip(grids[2:end], files[2:end])
    global all_pass
    nlm, nr, tensors = read_snapshot(file)
    @assert nlm == nlm_ref && nr == nr_ref "Dimension mismatch for $grid"
    local_pass = true
    for (name, ref_t, got_t) in zip(tensor_names, refs, tensors)
        maxdiff = maximum(abs.(ref_t .- got_t))
        ok      = maxdiff < 1e-10
        status  = ok ? "PASS" : "FAIL"
        println("  $(rpad(grid, 5))  $(rpad(name, 14))  maxdiff=$(rpad(string(maxdiff), 6))  [$status]")
        ok || (local_pass = false; all_pass = false)
    end
    local_pass && println("  → $grid vs 1x1: all tensors ≤ 1e-10 ✓")
    println()
end

if all_pass
    println("="^60)
    println("ALL GRIDS PASS: 2D-grid step == 1D-grid step to 1e-10")
    println("="^60)
else
    println("="^60)
    println("EQUIVALENCE FAILED — see FAIL lines above")
    println("="^60)
    exit(1)
end
