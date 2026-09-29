#include <algorithm>
#include <limits>
#include <span>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/generativeqc.h"
#include "molecule/basis.hpp"
#include "scf/cuda_df_gradient.hpp"

extern "C" generativeqc_status generativeqc_system_df_gradient_cuda(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* auxiliary, const double* bar_a, size_t count_a, const double* bar_m,
    size_t count_m, unsigned schedule, size_t maximum_bytes, size_t maximum_tile_elements,
    double* gradient, size_t gradient_count, generativeqc_df_gradient_resources* resources) {
  if (!context || !orbital || !auxiliary || !gradient || schedule > 1 || !maximum_bytes)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (resources && !generativeqc::api::valid_descriptor(resources))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  const auto n = generativeqc::molecule::ao_count(orbital->data),
             a = generativeqc::molecule::ao_count(auxiliary->data);
  const auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(double);
  if (!n || !a || n > maximum / n || a > maximum / a || n * n > maximum / a ||
      count_a != n * n * a || count_m != a * a || gradient_count != 3 * orbital->data.atoms.size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (resources)
    *resources = {sizeof(*resources), GENERATIVEQC_ABI_VERSION, 0, 0, 0, 0, 0, 0, 0, 0};
  std::lock_guard<std::recursive_mutex> context_lock(context->mutex);
#if GENERATIVEQC_HAS_CUDA
  if (context->state.executed_backend != GENERATIVEQC_BACKEND_CUDA) {
    context->last_detail = "generic DF gradients require a CUDA context";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  try {
    std::vector<double> result;
    // Backend diagnostics are scratch state; publish them only on failure so
    // a successful gradient does not invalidate a borrowed context detail.
    std::string detail;
    generativeqc::scf::DfGradientResources measured;
    auto span = [](const double* p, std::size_t n) {
      return p ? std::span<const double>(p, n) : std::span<const double>();
    };
    const auto status = generativeqc::scf::execute_cuda_df_gradient(
        context->state.device_id, orbital->data, auxiliary->data, span(bar_a, count_a),
        span(bar_m, count_m), schedule, maximum_bytes, maximum_tile_elements, result, detail,
        &measured);
    if (status != GENERATIVEQC_STATUS_SUCCESS) {
      context->last_detail = std::move(detail);
      return status;
    }
    std::copy(result.begin(), result.end(), gradient);
    if (resources)
      *resources = {sizeof(*resources),
                    GENERATIVEQC_ABI_VERSION,
                    measured.host_bytes,
                    measured.device_bytes,
                    measured.host_to_device_bytes,
                    measured.device_to_host_bytes,
                    measured.weight_tile_elements,
                    measured.tiles,
                    measured.uploads,
                    measured.stream_synchronizations};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
#else
  (void)bar_a;
  (void)bar_m;
  (void)maximum_tile_elements;
  context->last_detail = "CUDA DF gradients are unavailable in this build";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
}
