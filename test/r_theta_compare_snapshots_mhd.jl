# r×θ MHD equivalence cross-grid comparison
#
# Reads binary snapshots from RTHETA_TMPDIR (or /tmp) for each of the four
# grid layouts and asserts that all agree with the 1x1 reference to < 1e-10.
# Snapshots contain 12 tensors (6 field pairs):
#   temp, composition, vel_tor, vel_pol, mag_tor, mag_pol  (real+imag each)
#
# Called by test/run_mpi_r_theta_equivalence_mhd.sh after all driver runs complete.
# Can also be run standalone:
#   RTHETA_TMPDIR=/path/to/snaps julia test/r_theta_compare_snapshots_mhd.jl

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
files = [joinpath(snap_dir, "rtheta_mhd_sig_$(g).bin") for g in grids]

for (g, f) in zip(grids, files)
    isfile(f) || error("MHD snapshot for $g not found: $f")
end

nlm_ref, nr_ref, refs = read_snapshot(files[1])
println("Reference (1x1): nlm=$nlm_ref, nr=$nr_ref, ntensors=$(length(refs))")

tensor_names = [
    "temp_real",    "temp_imag",
    "comp_real",    "comp_imag",
    "vel_tor_real", "vel_tor_imag",
    "vel_pol_real", "vel_pol_imag",
    "mag_tor_real", "mag_tor_imag",
    "mag_pol_real", "mag_pol_imag",
]

all_pass = true
for (grid, file) in zip(grids[2:end], files[2:end])
    global all_pass
    nlm, nr, tensors = read_snapshot(file)
    @assert nlm == nlm_ref && nr == nr_ref "Dimension mismatch for $grid"
    @assert length(tensors) == length(refs) "Tensor count mismatch for $grid: got $(length(tensors)), want $(length(refs))"
    local_pass = true
    for (name, ref_t, got_t) in zip(tensor_names, refs, tensors)
        maxdiff = maximum(abs.(ref_t .- got_t))
        ok      = maxdiff < 1e-10
        status  = ok ? "PASS" : "FAIL"
        println("  $(rpad(grid, 5))  $(rpad(name, 14))  maxdiff=$(rpad(string(maxdiff), 22))  [$status]")
        ok || (local_pass = false; all_pass = false)
    end
    local_pass && println("  -> $grid vs 1x1: all 12 tensors <= 1e-10 (magnetic+composition)")
    println()
end

if all_pass
    println("="^70)
    println("ALL GRIDS PASS: MHD step (magnetic+composition) == 1D-grid to 1e-10")
    println("="^70)
else
    println("="^70)
    println("MHD EQUIVALENCE FAILED -- see FAIL lines above")
    println("="^70)
    exit(1)
end
