# Activate the GeoDynamoCUDAExt extension when CUDA is available, so the
# GPU-gated tests below actually execute on a GPU box (gpu_functional() can
# then return true). On CPU-only environments (Apple Silicon, CPU CI) this is
# a silent no-op and the GPU cases @test_skip exactly as before.
#
# CUDA is deliberately NOT a test-target dependency (it would force CUDA + the
# GPU extensions to precompile on every CPU CI run). So under sandboxed
# `Pkg.test()` this `using CUDA` won't resolve and is caught below. On a GPU
# box, run the suite in an environment that has CUDA available (e.g. `dev`/`add`
# CUDA into the active project) and this loads it, activating GeoDynamoCUDAExt
# so `gpu_functional()` flips true and the GPU gates actually execute.
try
    @eval using CUDA
catch err
    @info "CUDA not loaded; GPU-gated tests will skip" exception = (err, catch_backtrace())
end

# Tolerances for the [LOCAL] GPU≈CPU full-step parity gates.
#
# These compare the dense device path running on `Array` against the CPU solver:
# same machine, same arithmetic, deterministic seeds, so the only slack needed is
# for reduction reordering (the suite runs at 1 and 4 threads). Measured worst
# disagreement across every [LOCAL] gate — phase5n2 single step, phase6 4-step
# trajectory and gpu_run!(::SolverState) sync-back, phase5n all three gated
# configs — is max|diff| = 1.06e-15, max relative = 1.22e-15. These bounds leave
# ~5 orders of headroom on the largest field while still being ~3 orders tighter
# than the 1e-9/1e-7 they replaced, which passed anything down to 1e-7 relative.
#
# Do NOT reuse these for [GPU-BOX] gates: real CUDA reorders reductions and uses
# FMA, so those legitimately need the looser 1e-7/1e-5.
const GPU_LOCAL_ATOL = 1e-12
const GPU_LOCAL_RTOL = 1e-10
