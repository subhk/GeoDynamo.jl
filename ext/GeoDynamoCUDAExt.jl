module GeoDynamoCUDAExt

using GeoDynamo
using CUDA
using GPUArrays
using GPUArraysCore
using KernelAbstractions
using SHTnsKit

host_scratch_zeros(::Type{T}, dims...) where {T} = zeros(T, dims...)

GeoDynamo.arch_zeros(g::GeoDynamo.GPU, FT::DataType, dims...) =
    KernelAbstractions.zeros(g.backend, FT, dims...)

GeoDynamo.on_architecture(::GeoDynamo.GPU, a) = CUDA.cu(a)

function host_fill_scalar_coeff_buffer(coeffs_buffer, spec_real, spec_imag, r_local, config)
    return GeoDynamo.Solver.solver_cpu_fill_scalar_coeff_buffer!(
        coeffs_buffer,
        spec_real,
        spec_imag,
        r_local,
        config,
    )
end

function host_store_scalar_coefficients(spec_real, spec_imag, coeffs_matrix, r_local, config)
    return GeoDynamo.Solver.solver_cpu_store_scalar_coefficients!(
        spec_real,
        spec_imag,
        coeffs_matrix,
        r_local,
        config,
    )
end

function host_extract_physical_slice(slice_buffer, phys_data, r_local, config; axes_local=nothing)
    return GeoDynamo.Solver.solver_cpu_extract_physical_slice!(
        slice_buffer,
        phys_data,
        r_local,
        config;
        axes_local=axes_local,
    )
end

function host_store_physical_slice(phys_data, phys_slice, r_local)
    return GeoDynamo.Solver.solver_cpu_store_physical_slice!(
        phys_data,
        phys_slice,
        r_local,
    )
end

function host_fill_vector_coeff_buffer(coeffs_buffer, spec_real, spec_imag, r_local, config)
    return GeoDynamo.Solver.solver_cpu_fill_vector_coeff_buffer!(
        coeffs_buffer,
        spec_real,
        spec_imag,
        r_local,
        config,
    )
end

function host_store_vector_coefficients(spec_real, spec_imag, coeffs_matrix, r_local, config)
    return GeoDynamo.Solver.solver_cpu_store_vector_coefficients!(
        spec_real,
        spec_imag,
        coeffs_matrix,
        r_local,
        config,
    )
end

function host_extract_vector_component(component_buffer, v_data, r_local, config; axes_local=nothing)
    return GeoDynamo.Solver.solver_cpu_extract_vector_component!(
        component_buffer,
        v_data,
        r_local,
        config;
        axes_local=axes_local,
    )
end

function host_store_vector_components(v_theta, v_phi, vt_field, vp_field, r_local, config; axes_local=nothing)
    return GeoDynamo.Solver.solver_cpu_store_vector_components!(
        v_theta,
        v_phi,
        vt_field,
        vp_field,
        r_local,
        config;
        axes_local=axes_local,
    )
end

function __init__()
    GeoDynamo.register_gpu_backend!(
        available=CUDA.functional(),
        device=:cuda,
        scalar_synthesis=(cfg, coeffs; real_output::Bool=true) ->
            SHTnsKit.gpu_synthesis_safe(
                cfg,
                coeffs;
                device=SHTnsKit.CUDA_DEVICE,
                real_output=real_output,
            ),
        scalar_analysis=(cfg, spatial; real_output::Bool=true) ->
            SHTnsKit.gpu_analysis_safe(
                cfg,
                spatial;
                device=SHTnsKit.CUDA_DEVICE,
                real_output=real_output,
            ),
        vector_synthesis=(cfg, sph_coeffs, tor_coeffs; real_output::Bool=true) ->
            SHTnsKit.gpu_synthesis_sphtor(
                cfg,
                sph_coeffs,
                tor_coeffs;
                device=SHTnsKit.CUDA_DEVICE,
                real_output=real_output,
            ),
        vector_analysis=(cfg, vtheta, vphi) ->
            SHTnsKit.gpu_analysis_sphtor(
                cfg,
                vtheta,
                vphi;
                device=SHTnsKit.CUDA_DEVICE,
            ),
        scratch_zeros=host_scratch_zeros,
        fill_scalar_coeff_buffer=host_fill_scalar_coeff_buffer,
        store_scalar_coefficients=host_store_scalar_coefficients,
        extract_physical_slice=host_extract_physical_slice,
        store_physical_slice=host_store_physical_slice,
        fill_vector_coeff_buffer=host_fill_vector_coeff_buffer,
        store_vector_coefficients=host_store_vector_coefficients,
        extract_vector_component=host_extract_vector_component,
        store_vector_components=host_store_vector_components,
    )

    return nothing
end

end
