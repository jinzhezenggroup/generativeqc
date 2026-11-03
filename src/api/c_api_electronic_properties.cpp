#include <algorithm>
#include <mutex>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "dft/electronic_properties_cuda.hpp"
#include "generativeqc/electronic_properties.h"
#include "integrals/s_integrals.hpp"
#include "methods/dft_method.hpp"
#if GENERATIVEQC_HAS_CUDA
#include "tensor/cuda_error.hpp"
#endif

namespace {
generativeqc_status property_exception(std::string* detail) noexcept {
  try {
    throw;
#if GENERATIVEQC_HAS_CUDA
  } catch (const generativeqc_tensor::DeviceAllocationError& error) {
    *detail = error.what();
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const generativeqc_tensor::DeviceRuntimeError& error) {
    *detail = error.what();
    return GENERATIVEQC_STATUS_CUDA_ERROR;
#endif
  } catch (...) {
    return generativeqc::api::map_exception(detail);
  }
}
}  // namespace

struct generativeqc_electronic_properties {
  std::unique_ptr<generativeqc::dft::CudaElectronicProperties> cuda;
};

extern "C" generativeqc_status generativeqc_electronic_properties_cuda_create_v1(
    generativeqc_context* context, const generativeqc_system* system, const double* density,
    size_t density_count, size_t point_capacity, generativeqc_electronic_properties** output) {
  if (output) *output = nullptr;
  if (!context || !system || !density || !output || !point_capacity)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  if (context->state.requested_backend == generativeqc::core::kCpuReferenceBackend ||
      !system->data.ecp_terms.empty() ||
      std::any_of(system->data.atoms.begin(), system->data.atoms.end(),
                  [](const auto& atom) { return atom.ecp_core != 0; }))
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  try {
#if GENERATIVEQC_HAS_CUDA
    auto owner = std::make_unique<generativeqc_electronic_properties>();
    owner->cuda = generativeqc::dft::make_cuda_electronic_properties(
        system->data, context->state.device_id, density, density_count, point_capacity);
    *output = owner.release();
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  } catch (...) {
    return property_exception(&context->last_detail);
  }
}

extern "C" generativeqc_status generativeqc_electronic_properties_cuda_analyze_v1(
    generativeqc_context* context, generativeqc_electronic_properties* properties, double* dipole,
    double* mulliken, double* lowdin, size_t atom_count) {
  if (!context || !properties || (!dipole && !mulliken) || bool(mulliken) != bool(lowdin) ||
      atom_count != (mulliken ? properties->cuda->atoms() : 0) ||
      context->state.requested_backend == generativeqc::core::kCpuReferenceBackend ||
      context->state.device_id != properties->cuda->device())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    properties->cuda->analyze(dipole, mulliken, lowdin);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return property_exception(&context->last_detail);
  }
}

extern "C" generativeqc_status generativeqc_electronic_properties_cuda_evaluate_v1(
    generativeqc_context* context, generativeqc_electronic_properties* properties,
    const double* points, size_t point_count, int spin, double* output, size_t output_count) {
  if (!context || !properties || (point_count && (!points || !output)) ||
      point_count != output_count || (spin != 0 && spin != 1) ||
      context->state.requested_backend == generativeqc::core::kCpuReferenceBackend ||
      context->state.device_id != properties->cuda->device())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    properties->cuda->evaluate(points, point_count, spin != 0, output);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return property_exception(&context->last_detail);
  }
}

extern "C" void generativeqc_electronic_properties_destroy_v1(
    generativeqc_electronic_properties* properties) {
  delete properties;
}

extern "C" generativeqc_status generativeqc_ks_batch_copy_density_v1(generativeqc_batch* batch,
                                                                     size_t index, double* density,
                                                                     size_t density_count) {
  if (!batch || !density || !density_count || index >= batch->atom_counts.size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    generativeqc::dft::CudaKsFinalStateToken token;
    auto status = generativeqc::methods::detail::dft_final_state_token(*batch->plan, index, token,
                                                                       batch->context->last_detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    generativeqc::dft::VerifiedKsFinalState state;
    status = generativeqc::methods::detail::read_dft_final_state(
        *batch->plan, index, token, false, state, batch->context->last_detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    size_t required = 0;
    for (const auto& matrix : state.density) required += matrix.size();
    if (!required || required != density_count) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    for (const auto& matrix : state.density) {
      std::copy(matrix.begin(), matrix.end(), density);
      density += matrix.size();
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

extern "C" generativeqc_status generativeqc_system_position_integrals_cpu_v1(
    generativeqc_context* context, const generativeqc_system* system, double* output,
    size_t output_count) {
  if (!context || !system || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    generativeqc::integrals::position_integrals(system->data, {output, output_count});
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}
