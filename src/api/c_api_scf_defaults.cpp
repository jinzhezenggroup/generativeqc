#include <algorithm>
#include <cmath>
#include <string_view>

#include "api/error.hpp"
#include "generativeqc/scf_defaults.h"
#include "methods/scf_defaults.hpp"

extern "C" generativeqc_status generativeqc_scf_defaults_query_v1(
    const generativeqc_scf_defaults_context_v1* context, const generativeqc_ks_options* ks_options,
    const char* model_identity, size_t model_identity_size, generativeqc_scf_defaults_v1* output) {
  using generativeqc::api::valid_descriptor;
  if (!context || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!valid_descriptor(context) || !valid_descriptor(output) ||
      (ks_options && !valid_descriptor(ks_options)))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if ((context->backend != GENERATIVEQC_BACKEND_CPU_REFERENCE &&
       context->backend != GENERATIVEQC_BACKEND_CUDA) ||
      context->has_ecp > 1 || context->reserved ||
      (model_identity_size && (!model_identity || model_identity_size != 64)))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const std::string_view identity(model_identity ? model_identity : "", model_identity_size);
  if (!std::all_of(identity.begin(), identity.end(),
                   [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (ks_options) {
    const auto& options = *ks_options;
    if (!options.grid_version || options.grid_version > 4 ||
        !std::isfinite(options.coincident_tolerance) || options.coincident_tolerance < 0 ||
        (options.element_radius_count &&
         (!options.element_radii || options.element_radius_count != 119)))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    for (std::uint32_t i = 0; i != options.element_radius_count; ++i)
      if (!std::isfinite(options.element_radii[i]) || options.element_radii[i] < 0)
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  *output = generativeqc::methods::scf_defaults::resolve(*context, ks_options, identity);
  return GENERATIVEQC_STATUS_SUCCESS;
}
