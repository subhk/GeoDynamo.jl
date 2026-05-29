const TopographyAPI = getproperty(getproperty(GeoDynamo, :bcs), :topography)
const SOLVER_BACKEND_MPI = getproperty(GeoDynamo, :MPI)
const SOLVER_MPI = SOLVER_BACKEND_MPI
const SOLVER_LINEAR_ALGEBRA = getproperty(GeoDynamo, :LinearAlgebra)
const SOLVER_SHTNS_CONFIG_BUILDER = getproperty(GeoDynamo, :create_shtnskit_config)
const SOLVER_BALL_DOMAIN_BUILDER = getproperty(getproperty(GeoDynamo, :GeoDynamoBall), :create_ball_radial_domain)
const SOLVER_SHELL_DOMAIN_BUILDER = getproperty(getproperty(GeoDynamo, :GeoDynamoShell), :create_shell_radial_domain)
const SOLVER_VELOCITY_FIELD_BUILDER = getproperty(GeoDynamo, :create_shtns_velocity_fields)
const SOLVER_MAGNETIC_FIELD_BUILDER = getproperty(GeoDynamo, :create_shtns_magnetic_fields)
const SOLVER_TEMPERATURE_FIELD_BUILDER = getproperty(GeoDynamo, :create_shtns_temperature_field)
const SOLVER_COMPOSITION_FIELD_BUILDER = getproperty(GeoDynamo, :create_shtns_composition_field)
const SOLVER_SPECTRAL_FIELD_BUILDER = getproperty(GeoDynamo, :create_shtns_spectral_field)
const SOLVER_VELOCITY_TOROIDAL_MATRIX_BUILDER = getproperty(GeoDynamo, :create_velocity_toroidal_matrices)
const SOLVER_VELOCITY_POLOIDAL_MATRIX_BUILDER = getproperty(GeoDynamo, :create_velocity_poloidal_matrices)
const SOLVER_MAGNETIC_TOROIDAL_MATRIX_BUILDER = getproperty(GeoDynamo, :create_magnetic_toroidal_matrices)
const SOLVER_MAGNETIC_POLOIDAL_MATRIX_BUILDER = getproperty(GeoDynamo, :create_magnetic_poloidal_matrices)
const SOLVER_TEMPERATURE_MATRIX_BUILDER = getproperty(GeoDynamo, :create_temperature_matrices)
const SOLVER_COMPOSITION_MATRIX_BUILDER = getproperty(GeoDynamo, :create_composition_matrices)
const SOLVER_LOAD_SPECTRAL_BC = getproperty(GeoDynamo, :load_spectral_bc_from_file)
const SOLVER_STORE_BC_IN_FIELD! = getproperty(GeoDynamo, :store_bc_in_field!)
const SOLVER_SHARED_BUFFER_CACHE_LOCK = getproperty(GeoDynamo, :_BUFFER_CACHE_LOCK)
const SOLVER_SHARED_TIMING_FLAG = getproperty(GeoDynamo, :ENABLE_TIMING)
const SOLVER_SHARED_ERK2_DIAGNOSTICS_ENABLED = getproperty(GeoDynamo, :ERK2_DIAGNOSTICS_ENABLED)
const SOLVER_SHARED_ERK2_DIAGNOSTICS_INTERVAL = getproperty(GeoDynamo, :ERK2_DIAGNOSTICS_INTERVAL)
const SOLVER_GPU_BACKEND_LOADED = getproperty(GeoDynamo, :gpu_backend_loaded)
const SOLVER_GPU_BACKEND_AVAILABLE = getproperty(GeoDynamo, :gpu_backend_available)
const SOLVER_GPU_BACKEND_DEVICE = getproperty(GeoDynamo, :gpu_backend_device)
const SOLVER_GPU_SCALAR_SYNTHESIS = getproperty(GeoDynamo, :gpu_scalar_synthesis)
const SOLVER_GPU_SCALAR_ANALYSIS = getproperty(GeoDynamo, :gpu_scalar_analysis)
const SOLVER_GPU_VECTOR_SYNTHESIS = getproperty(GeoDynamo, :gpu_vector_synthesis)
const SOLVER_GPU_VECTOR_ANALYSIS = getproperty(GeoDynamo, :gpu_vector_analysis)
const SOLVER_GPU_SCRATCH_ZEROS = getproperty(GeoDynamo, :gpu_scratch_zeros)
const SOLVER_GPU_FILL_SCALAR_COEFF_BUFFER = getproperty(GeoDynamo, :gpu_fill_scalar_coeff_buffer)
const SOLVER_GPU_STORE_SCALAR_COEFFICIENTS = getproperty(GeoDynamo, :gpu_store_scalar_coefficients)
const SOLVER_GPU_EXTRACT_PHYSICAL_SLICE = getproperty(GeoDynamo, :gpu_extract_physical_slice)
const SOLVER_GPU_STORE_PHYSICAL_SLICE = getproperty(GeoDynamo, :gpu_store_physical_slice)
const SOLVER_GPU_FILL_VECTOR_COEFF_BUFFER = getproperty(GeoDynamo, :gpu_fill_vector_coeff_buffer)
const SOLVER_GPU_STORE_VECTOR_COEFFICIENTS = getproperty(GeoDynamo, :gpu_store_vector_coefficients)
const SOLVER_GPU_EXTRACT_VECTOR_COMPONENT = getproperty(GeoDynamo, :gpu_extract_vector_component)
const SOLVER_GPU_STORE_VECTOR_COMPONENTS = getproperty(GeoDynamo, :gpu_store_vector_components)
const SOLVER_GPU_NOT_LOADED_ERROR = getproperty(GeoDynamo, :_gpu_backend_not_loaded_error)
const SOLVER_GPU_UNAVAILABLE_ERROR = getproperty(GeoDynamo, :_gpu_backend_unavailable_error)

const RadialDomainType = getproperty(GeoDynamo, :RadialDomain)
const ScalarFieldType = getproperty(GeoDynamo, :AbstractScalarField)
const SpectralFieldType = getproperty(GeoDynamo, :SHTnsSpecField)
const PhysicalFieldType = getproperty(GeoDynamo, :SHTnsPhysField)
const VectorFieldType = getproperty(GeoDynamo, :SHTnsVectorField)
const VelocityFieldsType = getproperty(GeoDynamo, :SHTnsVelocityFields)
const TemperatureFieldType = getproperty(GeoDynamo, :SHTnsTemperatureField)
const MagneticFieldsType = getproperty(GeoDynamo, :SHTnsMagneticFields)
const CompositionFieldType = getproperty(GeoDynamo, :SHTnsCompositionField)
const SHTnsConfigType = getproperty(GeoDynamo, :SHTnsKitConfig)
const NaNConfigType = getproperty(GeoDynamo, :NaNDetectionConfig)
const OldBandedMatrix = getproperty(GeoDynamo, :BandedMatrix)
const OldBandedLU = getproperty(GeoDynamo, :BandedLU)
const OldImplicitMatrices = getproperty(GeoDynamo, :SHTnsImplicitMatrices)

@inline solver_backend_rank() = getproperty(GeoDynamo, :get_rank)()
@inline solver_backend_process_count() = getproperty(GeoDynamo, :get_nprocs)()
@inline solver_backend_mpi_initialized() = SOLVER_BACKEND_MPI.Initialized()
@inline solver_shared_timing_enabled() = SOLVER_SHARED_TIMING_FLAG[]
@inline solver_shared_erk2_diagnostics_enabled() = SOLVER_SHARED_ERK2_DIAGNOSTICS_ENABLED[]
@inline solver_shared_erk2_diagnostics_interval() = SOLVER_SHARED_ERK2_DIAGNOSTICS_INTERVAL[]
@inline solver_gpu_device() = SOLVER_GPU_BACKEND_DEVICE()
@inline solver_gpu_scalar_synthesis(cfg, coeffs; real_output::Bool=true) =
    SOLVER_GPU_SCALAR_SYNTHESIS(cfg, coeffs; real_output=real_output)
@inline solver_gpu_scalar_analysis(cfg, spatial; real_output::Bool=true) =
    SOLVER_GPU_SCALAR_ANALYSIS(cfg, spatial; real_output=real_output)
@inline solver_gpu_vector_synthesis(cfg, sph_coeffs, tor_coeffs; real_output::Bool=true) =
    SOLVER_GPU_VECTOR_SYNTHESIS(cfg, sph_coeffs, tor_coeffs; real_output=real_output)
@inline solver_gpu_vector_analysis(cfg, vtheta, vphi) =
    SOLVER_GPU_VECTOR_ANALYSIS(cfg, vtheta, vphi)
@inline function solver_gpu_scratch_zeros(::Type{T}, dims::Vararg{Int,N}) where {T,N}
    return SOLVER_GPU_SCRATCH_ZEROS(T, dims...)::AbstractArray{T,N}
end
@inline solver_gpu_fill_scalar_coeff_buffer(coeffs_buffer, spec_real, spec_imag, r_local, config) =
    SOLVER_GPU_FILL_SCALAR_COEFF_BUFFER(coeffs_buffer, spec_real, spec_imag, r_local, config)
@inline solver_gpu_store_scalar_coefficients(spec_real, spec_imag, coeffs_matrix, r_local, config) =
    SOLVER_GPU_STORE_SCALAR_COEFFICIENTS(spec_real, spec_imag, coeffs_matrix, r_local, config)
@inline solver_gpu_extract_physical_slice(slice_buffer, phys_data, r_local, config; axes_local=nothing) =
    SOLVER_GPU_EXTRACT_PHYSICAL_SLICE(slice_buffer, phys_data, r_local, config; axes_local=axes_local)
@inline solver_gpu_store_physical_slice(phys_data, phys_slice, r_local) =
    SOLVER_GPU_STORE_PHYSICAL_SLICE(phys_data, phys_slice, r_local)
@inline solver_gpu_fill_vector_coeff_buffer(coeffs_buffer, spec_real, spec_imag, r_local, config) =
    SOLVER_GPU_FILL_VECTOR_COEFF_BUFFER(coeffs_buffer, spec_real, spec_imag, r_local, config)
@inline solver_gpu_store_vector_coefficients(spec_real, spec_imag, coeffs_matrix, r_local, config) =
    SOLVER_GPU_STORE_VECTOR_COEFFICIENTS(spec_real, spec_imag, coeffs_matrix, r_local, config)
@inline solver_gpu_extract_vector_component(component_buffer, v_data, r_local, config; axes_local=nothing) =
    SOLVER_GPU_EXTRACT_VECTOR_COMPONENT(component_buffer, v_data, r_local, config; axes_local=axes_local)
@inline solver_gpu_store_vector_components(v_theta, v_phi, vt_field, vp_field, r_local, config; axes_local=nothing) =
    SOLVER_GPU_STORE_VECTOR_COMPONENTS(v_theta, v_phi, vt_field, vp_field, r_local, config; axes_local=axes_local)
@inline solver_gpu_not_loaded_error() = SOLVER_GPU_NOT_LOADED_ERROR()
@inline solver_gpu_unavailable_error() = SOLVER_GPU_UNAVAILABLE_ERROR()

function backend_ensure_mpi!()
    solver_backend_mpi_initialized() || SOLVER_BACKEND_MPI.Init()
    return nothing
end
