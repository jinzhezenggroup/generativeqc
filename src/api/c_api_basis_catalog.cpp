#include <mutex>
#include <span>
#include <stdexcept>
#include <string_view>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/generativeqc.h"
#include "molecule/basis_catalog.hpp"
#include "molecule/basis_template.hpp"

namespace {
generativeqc_status create_expanded(generativeqc_context* context,
                                    const generativeqc_system_descriptor& descriptor,
                                    const generativeqc::molecule::ExpandedBasis& expanded,
                                    generativeqc_system** system) {
  auto resolved = descriptor;
  resolved.shells = expanded.shells.data();
  resolved.shell_count = static_cast<std::uint32_t>(expanded.shells.size());
  resolved.primitives = expanded.primitives.data();
  resolved.primitive_count = static_cast<std::uint32_t>(expanded.primitives.size());
  if (!expanded.ecp_terms.empty())
    return generativeqc_system_create_ecp(context, &resolved, expanded.core_electrons.data(),
                                          expanded.ecp_terms.data(), expanded.ecp_terms.size(),
                                          system);
  return generativeqc_system_create(context, &resolved, system);
}
}  // namespace

extern "C" {

const char* generativeqc_bundled_basis_catalog_sha256(void) {
  return generativeqc::molecule::bundled_basis_catalog_sha256();
}

generativeqc_status generativeqc_system_create_bundled(
    generativeqc_context* context, const generativeqc_system_descriptor* descriptor,
    const char* basis_name, const char* expected_catalog_sha256, generativeqc_system** system) {
  if (context == nullptr || descriptor == nullptr || basis_name == nullptr || system == nullptr)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *system = nullptr;
  if (descriptor->abi_version != GENERATIVEQC_ABI_VERSION ||
      descriptor->struct_size < sizeof(generativeqc_system_descriptor))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (descriptor->atoms == nullptr || descriptor->atom_count == 0 ||
      descriptor->shells != nullptr || descriptor->shell_count != 0 ||
      descriptor->primitives != nullptr || descriptor->primitive_count != 0)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;

  std::lock_guard<std::recursive_mutex> context_lock(context->mutex);
  try {
    if (expected_catalog_sha256 != nullptr &&
        std::string_view(expected_catalog_sha256) !=
            generativeqc::molecule::bundled_basis_catalog_sha256())
      throw std::invalid_argument("bundled basis catalog checksum does not match caller snapshot");
    const auto expanded = generativeqc::molecule::expand_bundled_basis(
        basis_name, std::span(descriptor->atoms, descriptor->atom_count),
        static_cast<generativeqc_basis_representation>(descriptor->basis_representation));
    return create_expanded(context, *descriptor, expanded, system);
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

generativeqc_status generativeqc_system_create_basis_template_v1(
    generativeqc_context* context, const generativeqc_system_descriptor* descriptor,
    const generativeqc_basis_template_descriptor_v1* basis, generativeqc_system** system) {
  if (!context || !descriptor || !basis || !system) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *system = nullptr;
  if (descriptor->struct_size < sizeof(*descriptor) ||
      descriptor->abi_version != GENERATIVEQC_ABI_VERSION || basis->struct_size < sizeof(*basis) ||
      basis->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (!descriptor->atoms || !descriptor->atom_count || descriptor->shells ||
      descriptor->shell_count || descriptor->primitives || descriptor->primitive_count)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> context_lock(context->mutex);
  try {
    const auto expanded = generativeqc::molecule::expand_basis_template(
        *basis, std::span(descriptor->atoms, descriptor->atom_count),
        static_cast<generativeqc_basis_representation>(descriptor->basis_representation));
    return create_expanded(context, *descriptor, expanded, system);
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

}  // extern "C"
