# Activate the GeoDynamoCUDAExt extension when CUDA is available, so the
# GPU-gated tests below actually execute on a GPU box (gpu_functional() can
# then return true). On CPU-only environments (Apple Silicon, CPU CI) this is
# a silent no-op and the GPU cases @test_skip exactly as before.
#
# CUDA is a test-target dependency (Project.toml [extras] + targets.test) so it
# resolves under standard `Pkg.test()`; loading it here is what makes the
# extension load and the gate flip on hardware with a functional device.
try
    @eval using CUDA
catch err
    @info "CUDA not loaded; GPU-gated tests will skip" exception = (err, catch_backtrace())
end
