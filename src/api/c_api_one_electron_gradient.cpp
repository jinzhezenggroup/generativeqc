#include <algorithm>
#include <limits>
#include <span>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/generativeqc.h"
#include "molecule/basis.hpp"
#include "scf/cuda_one_electron_gradient.hpp"

extern "C" generativeqc_status generativeqc_system_one_electron_gradient_cuda(
    generativeqc_context* context, const generativeqc_system* system, const double* overlap_weights,
    const double* kinetic_weights, const double* attraction_weights, size_t matrix_count,
    unsigned schedule, size_t maximum_bytes, double* gradient, size_t gradient_count,
    generativeqc_one_electron_gradient_resources* resources) {
  if (!context || !system || !gradient || schedule > 3 || !maximum_bytes)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (resources && !generativeqc::api::valid_descriptor(resources))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  const auto n = generativeqc::molecule::ao_count(system->data);
  if (!n || n > std::numeric_limits<std::size_t>::max() / n || matrix_count != n * n ||
      gradient_count != system->data.atoms.size() * 3)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (resources) {
    *resources = {sizeof(*resources), GENERATIVEQC_ABI_VERSION, 0, 0, 0, 0, 0, 0};
  }
  std::lock_guard<std::recursive_mutex> context_lock(context->mutex);
#if GENERATIVEQC_HAS_CUDA
  if (context->state.executed_backend != GENERATIVEQC_BACKEND_CUDA) {
    context->last_detail = "generic generated gradients require a CUDA context";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  try {
    auto weights = [&](const double* p) {
      return p ? std::span<const double>(p, matrix_count) : std::span<const double>();
    };
    std::vector<double> result;
    // Keep backend scratch diagnostics separate from the last public failure.
    std::string detail;
    generativeqc::scf::OneElectronGradientResources measured;
    auto status = generativeqc::scf::execute_cuda_one_electron_gradient(
        context->state.device_id, system->data, weights(overlap_weights), weights(kinetic_weights),
        weights(attraction_weights), schedule, maximum_bytes, result, detail, &measured);
    if (status != GENERATIVEQC_STATUS_SUCCESS) {
      context->last_detail = std::move(detail);
      return status;
    }
    std::copy(result.begin(), result.end(), gradient);
    if (resources) {
      resources->device_bytes = measured.device_bytes;
      resources->host_numeric_bytes = measured.host_numeric_bytes;
      resources->host_to_device_bytes = measured.host_to_device_bytes;
      resources->device_to_host_bytes = measured.device_to_host_bytes;
      resources->synchronous_uploads = measured.synchronous_uploads;
      resources->stream_synchronizations = measured.stream_synchronizations;
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
#else
  (void)overlap_weights;
  (void)kinetic_weights;
  (void)attraction_weights;
  context->last_detail = "CUDA one-electron gradients are unavailable in this build";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
}
