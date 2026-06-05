# =============================================================================
# GPU Phase 0 — device capability gate + constructor (core-side interface).
# CUDA-specific behaviour is provided by ext/GeoDynamoCUDAExt.jl when CUDA loads.
# =============================================================================

"""
    gpu_functional() -> Bool

`true` only when a CUDA-capable GPU is present AND the `GeoDynamoCUDAExt`
extension is loaded (i.e. `CUDA.functional()`).  `false` otherwise, including on
machines with no GPU.  Use this to gate GPU code paths and tests.
"""
gpu_functional() = false

"""
    gpu_synchronize()

Block until all queued GPU work completes.  No-op when no GPU backend is active.
"""
gpu_synchronize() = nothing

"""
    GPU() -> GPU

Construct a `GPU` architecture bound to the default functional GPU backend.
Errors if no GPU backend is available (CUDA extension not loaded / no device).
The CUDA extension overrides `_gpu_default_backend()` to return a `CUDABackend`.
"""
GPU() = GPU(_gpu_default_backend())

_gpu_default_backend() = error(
    "GPU() requires a functional CUDA GPU and the GeoDynamoCUDAExt extension " *
    "(load CUDA.jl on a machine with a CUDA device).")
