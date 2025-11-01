"""
GPU Backend Management for GeoDynamo.jl

This module provides vendor-agnostic GPU computing capabilities using:
- GPUArrays.jl: Universal GPU array interface
- GPUArraysCore.jl: Core GPU functionality
- KernelAbstractions.jl: Vendor-agnostic kernel programming
- CUDA.jl: NVIDIA GPU support
- AMDGPU.jl: AMD GPU support  
- Metal.jl: Apple GPU support

Features:
- Automatic device detection and selection
- User-configurable device preferences (CPU/GPU)
- Fallback to CPU for unsupported operations
- Performance monitoring and optimization hints
"""

using GPUArrays
using GPUArraysCore
using KernelAbstractions
using LinearAlgebra
using Printf

# Conditional imports for GPU backends
const HAS_CUDA = try
    import CUDA
    CUDA.functional()
catch
    false
end

const HAS_AMDGPU = try
    import AMDGPU
    AMDGPU.functional()
catch
    false
end

const HAS_METAL = try
    import Metal
    Metal.functional()
catch
    false
end

# Export main interface
export GeoDynamoDevice, select_device!, get_device, get_backend
export cpu_array, gpu_array, device_array, sync_device!
export @geodynamo_kernel, launch_kernel!
export GPU_AVAILABLE, list_available_devices, device_info

"""
Supported device types for GeoDynamo.jl computations
"""
@enum DeviceType begin
    CPU_DEVICE = 0
    CUDA_DEVICE = 1
    AMDGPU_DEVICE = 2
    METAL_DEVICE = 3
end

"""
GeoDynamoDevice

Manages computational device selection and array allocation for GeoDynamo.jl
"""
mutable struct GeoDynamoDevice
    device_type::DeviceType
    device_name::String
    backend::Any  # KernelAbstractions backend
    array_type::Type
    max_memory_gb::Float64
    compute_capability::String
end

# Global device state
const CURRENT_DEVICE = Ref{GeoDynamoDevice}()

"""
    GPU_AVAILABLE::Bool

Global constant indicating if any GPU backend is available
"""
const GPU_AVAILABLE = HAS_CUDA || HAS_AMDGPU || HAS_METAL

function __init__()
    # Initialize with CPU device by default
    CURRENT_DEVICE[] = create_cpu_device()
    
    if GPU_AVAILABLE
        println("GPU backends available:")
        HAS_CUDA && println("  ✓ CUDA.jl (NVIDIA)")
        HAS_AMDGPU && println("  ✓ AMDGPU.jl (AMD)")
        HAS_METAL && println("  ✓ Metal.jl (Apple)")
        println("Use select_device!(\"GPU\") to enable GPU acceleration")
    else
        println("No GPU backends available - using CPU only")
    end
end

"""
    create_cpu_device() -> GeoDynamoDevice

Create CPU device configuration
"""
function create_cpu_device()
    return GeoDynamoDevice(
        CPU_DEVICE,
        "CPU ($(Threads.nthreads()) threads)",
        CPU(),
        Array,
        get_system_memory_gb(),
        "N/A"
    )
end

"""
    create_cuda_device() -> GeoDynamoDevice

Create CUDA device configuration (if available)
"""
function create_cuda_device()
    if !HAS_CUDA
        error("CUDA.jl not available or GPU not functional")
    end
    
    dev = CUDA.device()
    mem_gb = CUDA.available_memory() / (1024^3)
    compute_cap = "$(CUDA.capability(dev).major).$(CUDA.capability(dev).minor)"
    
    return GeoDynamoDevice(
        CUDA_DEVICE,
        "CUDA: $(CUDA.name(dev))",
        CUDADevice(),
        CUDA.CuArray,
        mem_gb,
        compute_cap
    )
end

"""
    create_amdgpu_device() -> GeoDynamoDevice

Create AMDGPU device configuration (if available)
"""
function create_amdgpu_device()
    if !HAS_AMDGPU
        error("AMDGPU.jl not available or GPU not functional")
    end
    
    # AMDGPU device info
    mem_gb = AMDGPU.device_memory() / (1024^3)
    device_name = "AMD GPU"
    
    return GeoDynamoDevice(
        AMDGPU_DEVICE,
        "AMDGPU: $device_name",
        ROCDevice(),
        AMDGPU.ROCArray,
        mem_gb,
        "ROCm"
    )
end

"""
    create_metal_device() -> GeoDynamoDevice

Create Metal device configuration (if available)
"""
function create_metal_device()
    if !HAS_METAL
        error("Metal.jl not available or GPU not functional")
    end
    
    # Metal device info
    mem_gb = 8.0  # Placeholder - Metal.jl may not expose this directly
    
    return GeoDynamoDevice(
        METAL_DEVICE,
        "Metal: Apple GPU",
        MetalDevice(),
        Metal.MtlArray,
        mem_gb,
        "Metal"
    )
end

"""
    select_device!(device::String) -> GeoDynamoDevice
    select_device!(device_type::DeviceType) -> GeoDynamoDevice

Select computational device for GeoDynamo.jl operations.

# Arguments
- `device::String`: Device selection ("CPU", "GPU", "CUDA", "AMDGPU", "METAL")
- `device_type::DeviceType`: Explicit device type enum

# Examples
```julia
# User-friendly string interface
select_device!("CPU")     # Force CPU computation
select_device!("GPU")     # Auto-select best available GPU
select_device!("CUDA")    # Prefer NVIDIA GPU
select_device!("AMDGPU")  # Prefer AMD GPU
select_device!("METAL")   # Prefer Apple GPU

# Direct enum interface
select_device!(CUDA_DEVICE)
```
"""
function select_device!(device::String)
    device_upper = uppercase(device)
    
    if device_upper == "CPU"
        CURRENT_DEVICE[] = create_cpu_device()
        
    elseif device_upper == "GPU"
        # Auto-select best available GPU
        if HAS_CUDA
            CURRENT_DEVICE[] = create_cuda_device()
        elseif HAS_AMDGPU
            CURRENT_DEVICE[] = create_amdgpu_device()
        elseif HAS_METAL
            CURRENT_DEVICE[] = create_metal_device()
        else
            @warn "No GPU available - falling back to CPU"
            CURRENT_DEVICE[] = create_cpu_device()
        end
        
    elseif device_upper == "CUDA"
        if HAS_CUDA
            CURRENT_DEVICE[] = create_cuda_device()
        else
            error("CUDA not available. Install CUDA.jl and ensure GPU is functional.")
        end
        
    elseif device_upper == "AMDGPU"
        if HAS_AMDGPU
            CURRENT_DEVICE[] = create_amdgpu_device()
        else
            error("AMDGPU not available. Install AMDGPU.jl and ensure GPU is functional.")
        end
        
    elseif device_upper == "METAL"
        if HAS_METAL
            CURRENT_DEVICE[] = create_metal_device()
        else
            error("Metal not available. Install Metal.jl and ensure running on Apple Silicon.")
        end
        
    else
        error("Unknown device: $device. Use 'CPU', 'GPU', 'CUDA', 'AMDGPU', or 'METAL'")
    end
    
    println("Selected device: $(CURRENT_DEVICE[].device_name)")
    println("Available memory: $(round(CURRENT_DEVICE[].max_memory_gb, digits=1)) GB")
    
    return CURRENT_DEVICE[]
end

function select_device!(device_type::DeviceType)
    if device_type == CPU_DEVICE
        return select_device!("CPU")
    elseif device_type == CUDA_DEVICE
        return select_device!("CUDA")
    elseif device_type == AMDGPU_DEVICE
        return select_device!("AMDGPU")
    elseif device_type == METAL_DEVICE
        return select_device!("METAL")
    else
        error("Unknown device type: $device_type")
    end
end

"""
    get_device() -> GeoDynamoDevice

Get current computational device
"""
get_device() = CURRENT_DEVICE[]

"""
    get_backend() -> KernelAbstractions.Backend

Get current KernelAbstractions backend
"""
get_backend() = CURRENT_DEVICE[].backend

"""
    device_array(x::AbstractArray) -> AbstractArray

Convert array to current device array type
"""
function device_array(x::AbstractArray{T}) where T
    ArrayType = CURRENT_DEVICE[].array_type
    if x isa ArrayType
        return x
    else
        return ArrayType(x)
    end
end

"""
    cpu_array(x::AbstractArray) -> Array

Convert array to CPU array
"""
function cpu_array(x::AbstractArray)
    if x isa Array
        return x
    else
        return Array(x)
    end
end

"""
    gpu_array(x::AbstractArray) -> AbstractGPUArray

Convert array to GPU array (using current GPU backend)
"""
function gpu_array(x::AbstractArray{T}) where T
    device = get_device()
    
    if device.device_type == CPU_DEVICE
        @warn "Current device is CPU - consider select_device!(\"GPU\")"
        return x
    end
    
    ArrayType = device.array_type
    if x isa ArrayType
        return x
    else
        return ArrayType(x)
    end
end

"""
    sync_device!()

Synchronize current device (wait for GPU kernels to complete)
"""
function sync_device!()
    device = get_device()
    
    if device.device_type == CUDA_DEVICE && HAS_CUDA
        CUDA.synchronize()
    elseif device.device_type == AMDGPU_DEVICE && HAS_AMDGPU
        AMDGPU.synchronize()
    elseif device.device_type == METAL_DEVICE && HAS_METAL
        Metal.synchronize()
    end
    # CPU doesn't need explicit synchronization
end

"""
    @geodynamo_kernel expr

Macro for defining device-agnostic kernels using KernelAbstractions
"""
macro geodynamo_kernel(expr)
    return quote
        @kernel function $(esc(expr))
    end
end

"""
    launch_kernel!(kernel_func, ndrange, args...; workgroupsize=nothing)

Launch a kernel function on current device
"""
function launch_kernel!(kernel_func, ndrange, args...; workgroupsize=nothing)
    backend = get_backend()
    
    if workgroupsize === nothing
        # Auto-select workgroup size based on device
        device = get_device()
        if device.device_type == CUDA_DEVICE
            workgroupsize = min(256, ndrange)
        elseif device.device_type == AMDGPU_DEVICE
            workgroupsize = min(256, ndrange)
        elseif device.device_type == METAL_DEVICE
            workgroupsize = min(256, ndrange)
        else  # CPU
            workgroupsize = min(Threads.nthreads(), ndrange)
        end
    end
    
    kernel = kernel_func(backend, workgroupsize)
    kernel(args..., ndrange=ndrange)
    sync_device!()
end

"""
    list_available_devices() -> Vector{String}

List all available computational devices
"""
function list_available_devices()
    devices = ["CPU"]
    
    if HAS_CUDA
        push!(devices, "CUDA")
    end
    
    if HAS_AMDGPU
        push!(devices, "AMDGPU")
    end
    
    if HAS_METAL
        push!(devices, "METAL")
    end
    
    return devices
end

"""
    device_info()

Print detailed information about current device and available alternatives
"""
function device_info()
    device = get_device()
    
    println("Current Device Information:")
    println("=" ^ 50)
    println("Device: $(device.device_name)")
    println("Type: $(device.device_type)")
    println("Backend: $(typeof(device.backend))")
    println("Array Type: $(device.array_type)")
    println("Memory: $(round(device.max_memory_gb, digits=1)) GB")
    println("Compute: $(device.compute_capability)")
    
    println("\nAvailable Devices:")
    println("-" ^ 30)
    available = list_available_devices()
    for dev in available
        marker = dev == string(device.device_type)[1:end-7] ? " (current)" : ""
        println("  $dev$marker")
    end
    
    if GPU_AVAILABLE && device.device_type == CPU_DEVICE
        println("\n💡 Tip: Use select_device!(\"GPU\") for better performance")
    end
end

"""
    get_system_memory_gb() -> Float64

Get system memory in GB
"""
function get_system_memory_gb()
    try
        # Try to get system memory (Unix-like systems)
        if Sys.islinux() || Sys.isapple()
            mem_kb = parse(Int, readchomp(`sysctl -n hw.memsize`)) ÷ 1024
            return mem_kb / (1024^2)  # Convert to GB
        else
            return 8.0  # Default fallback
        end
    catch
        return 8.0  # Default fallback
    end
end

"""
    memory_usage_gb() -> Float64

Get current memory usage of active arrays on current device
"""
function memory_usage_gb()
    device = get_device()
    
    if device.device_type == CUDA_DEVICE && HAS_CUDA
        return (CUDA.pool_used_memory() + CUDA.pool_cached_memory()) / (1024^3)
    elseif device.device_type == AMDGPU_DEVICE && HAS_AMDGPU
        # AMDGPU memory usage (if available)
        return 0.0  # Placeholder
    elseif device.device_type == METAL_DEVICE && HAS_METAL
        # Metal memory usage (if available)
        return 0.0  # Placeholder
    else
        # CPU memory usage is harder to track precisely
        return 0.0
    end
end

"""
    optimize_memory!()

Trigger garbage collection and memory cleanup on current device
"""
function optimize_memory!()
    device = get_device()
    
    # CPU garbage collection
    GC.gc()
    
    # GPU-specific memory cleanup
    if device.device_type == CUDA_DEVICE && HAS_CUDA
        CUDA.reclaim()
    elseif device.device_type == AMDGPU_DEVICE && HAS_AMDGPU
        # AMDGPU.reclaim() # If available
    elseif device.device_type == METAL_DEVICE && HAS_METAL
        # Metal.reclaim() # If available
    end
end

"""
Performance monitoring for device operations
"""
mutable struct DevicePerformanceStats
    kernel_launches::Int
    total_kernel_time::Float64
    memory_transfers_host_to_device::Int
    memory_transfers_device_to_host::Int
    total_transfer_time::Float64
end

const PERF_STATS = Ref(DevicePerformanceStats(0, 0.0, 0, 0, 0.0))

"""
    reset_performance_stats!()

Reset device performance counters
"""
function reset_performance_stats!()
    PERF_STATS[] = DevicePerformanceStats(0, 0.0, 0, 0, 0.0)
end

"""
    get_performance_stats() -> DevicePerformanceStats

Get current device performance statistics
"""
get_performance_stats() = PERF_STATS[]

"""
    print_performance_report()

Print detailed device performance report
"""
function print_performance_report()
    stats = get_performance_stats()
    device = get_device()
    
    println("Device Performance Report")
    println("=" ^ 40)
    println("Device: $(device.device_name)")
    println("Kernel Launches: $(stats.kernel_launches)")
    println("Total Kernel Time: $(round(stats.total_kernel_time * 1000, digits=2)) ms")
    if stats.kernel_launches > 0
        avg_time = stats.total_kernel_time / stats.kernel_launches * 1000
        println("Average Kernel Time: $(round(avg_time, digits=2)) ms")
    end
    println("H→D Transfers: $(stats.memory_transfers_host_to_device)")
    println("D→H Transfers: $(stats.memory_transfers_device_to_host)")
    println("Total Transfer Time: $(round(stats.total_transfer_time * 1000, digits=2)) ms")
    println("Memory Usage: $(round(memory_usage_gb(), digits=2)) GB")
end