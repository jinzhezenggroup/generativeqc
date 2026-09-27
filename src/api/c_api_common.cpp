#include <cmath>\n#include <cstring>\n#include <limits>\n\n#include "api/error.hpp"
#include "generated_libxc_public_cpu.hpp"
#include "methods/method.hpp"
#include "vibeqc/vibeqc.h"

extern "C" {

uint32_t vibeqc_ks_options_version(void) { return 1; }

vibeqc_status vibeqc_libxc_semilocal_program_get(const char* component_id,
                                                 vibeqc_ks_semilocal_program* program) {
  if (component_id == nullptr || *component_id == '\0' || program == nullptr)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  const auto* native = vibeqc::dft::bulk_public::find(component_id);
  if (native == nullptr) return VIBEQC_STATUS_NOT_IMPLEMENTED;
  program->identifier = native->identifier;
  program->expression_identity = native->expression_identity;
  program->ingredient_mask = native->ingredient_mask;
  program->domain_version = native->domain_version;
  program->native_program = native;
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status vibeqc_libxc_semilocal_program_evaluate_v1(
    const vibeqc_ks_semilocal_program* program, const double* rho, const double* gradient,
    const double* tau, std::size_t point_count, double* values, std::size_t value_count) {
  constexpr std::size_t stride = 11;
  if (program == nullptr || program->identifier == nullptr || *program->identifier == '\0' ||
      program->expression_identity == nullptr || *program->expression_identity == '\0' ||
      program->native_program == nullptr || rho == nullptr || gradient == nullptr || tau == nullptr ||
      values == nullptr || point_count == 0 ||
      point_count > std::numeric_limits<std::size_t>::max() / stride ||
      value_count != stride * point_count)
    return VIBEQC_STATUS_INVALID_ARGUMENT;

  const auto* native = vibeqc::dft::bulk_public::find(program->identifier);
  if (native == nullptr) return VIBEQC_STATUS_NOT_IMPLEMENTED;
  if (native != program->native_program || native->ingredient_mask != program->ingredient_mask ||
      native->domain_version != program->domain_version ||
      std::strcmp(native->expression_identity, program->expression_identity) != 0)
    return VIBEQC_STATUS_INVALID_ARGUMENT;

  try {
    vibeqc::dft::validate_semilocal_point_program(*native);
    for (std::size_t point = 0; point < point_count; ++point) {
      double local_rho[2]{rho[point], rho[point_count + point]};
      double local_gradient[2][3]{};
      double local_tau[2]{tau[point], tau[point_count + point]};
      for (std::size_t spin = 0; spin < 2; ++spin)
        for (std::size_t axis = 0; axis < 3; ++axis)
          local_gradient[spin][axis] = gradient[(spin * point_count + point) * 3 + axis];

      const auto xc = native->evaluate(local_rho, local_gradient, local_tau);
      double packed[stride]{xc.energy, xc.rho[0], xc.rho[1],
                            xc.gradient[0][0], xc.gradient[0][1], xc.gradient[0][2],
                            xc.gradient[1][0], xc.gradient[1][1], xc.gradient[1][2],
                            xc.kinetic[0], xc.kinetic[1]};
      for (double value : packed)
        if (!std::isfinite(value)) return VIBEQC_STATUS_NUMERICAL_FAILURE;
      std::copy(std::begin(packed), std::end(packed), values + stride * point);
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
}

uint32_t vibeqc_get_abi_version(void) { return VIBEQC_ABI_VERSION; }

const char* vibeqc_status_message(vibeqc_status status) {
  switch (status) {
    case VIBEQC_STATUS_SUCCESS:
      return "success";
    case VIBEQC_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case VIBEQC_STATUS_ABI_MISMATCH:
      return "ABI mismatch";
    case VIBEQC_STATUS_NOT_IMPLEMENTED:
      return "requested capability is not implemented";
    case VIBEQC_STATUS_NOT_CONVERGED:
      return "SCF did not converge";
    case VIBEQC_STATUS_NUMERICAL_FAILURE:
      return "numerical failure";
    case VIBEQC_STATUS_CUDA_ERROR:
      return "CUDA runtime error";
    case VIBEQC_STATUS_OUT_OF_MEMORY:
      return "out of memory";
    case VIBEQC_STATUS_INTERNAL_ERROR:
      return "internal error";
    case VIBEQC_STATUS_PRECISION_UNAVAILABLE:
      return "precision provenance not yet populated";
  }
  return "unknown status";
}

vibeqc_status vibeqc_method_available(vibeqc_method method, int32_t* available) {
  if (available == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  const vibeqc::methods::Capabilities* capabilities = vibeqc::methods::find_capabilities(method);
  if (capabilities == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  *available = capabilities->available ? 1 : 0;
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status vibeqc_method_get_capabilities(vibeqc_method method,
                                             vibeqc_method_capabilities_descriptor* output) {
  if (!vibeqc::api::valid_descriptor(output)) {
    return output == nullptr ? VIBEQC_STATUS_INVALID_ARGUMENT : VIBEQC_STATUS_ABI_MISMATCH;
  }
  const vibeqc::methods::Capabilities* capabilities = vibeqc::methods::find_capabilities(method);
  if (capabilities == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  output->method = capabilities->method;
  output->family = capabilities->family;
  output->supported_properties = capabilities->supported_properties;
  output->available = capabilities->available ? 1 : 0;
  output->supports_batch = capabilities->supports_batch ? 1 : 0;
  return VIBEQC_STATUS_SUCCESS;
}

}  // extern "C"
