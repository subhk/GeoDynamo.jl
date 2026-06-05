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
