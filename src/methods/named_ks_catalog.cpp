#include "methods/named_ks_catalog.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "native_method_catalog.hpp"

namespace generativeqc::methods::named_ks {

std::span<const Method> methods() noexcept { return generated::kMethods; }

const Method* find_method(std::string_view name) noexcept { return generated::find_method(name); }

std::span<const generativeqc_ks_semilocal_component> components(const Method& method) noexcept {
  return std::span(generated::kComponents).subspan(method.component_offset, method.component_count);
}

std::span<const generativeqc_ks_exchange_term> exchanges(const Method& method) noexcept {
  return std::span(generated::kExchanges).subspan(method.exchange_offset, method.exchange_count);
}

}  // namespace generativeqc::methods::named_ks

namespace {
bool disjoint(const void* first, std::size_t first_bytes, const void* second,
              std::size_t second_bytes) noexcept {
  if (!first_bytes || !second_bytes) return true;
  const auto first_address = reinterpret_cast<std::uintptr_t>(first);
  const auto second_address = reinterpret_cast<std::uintptr_t>(second);
  const auto maximum = std::numeric_limits<std::uintptr_t>::max();
  if (first_bytes > maximum - first_address || second_bytes > maximum - second_address)
    return false;
  return first_address + first_bytes <= second_address ||
         second_address + second_bytes <= first_address;
}
}  // namespace

/** Private additive named-program transport. A zero selection is an explicit
 * compiler/custom/old-catalog route, not failure or scientific fallback. An
 * actual refusal is transactional and must never be retried through Python.
 * Name/identity are complete counted strings; an empty name permits exact-IR
 * lookup, while an empty identity permits an explicit SDK named-default query.
 * Selected component/domain/exchange pointers are immutable library-owned data;
 * preserve that library until preparation has copied the descriptor. */
extern "C" GENERATIVEQC_API generativeqc_status generativeqc_ks_catalog_plan_v1(
    const char* name, std::size_t name_size, const char* identity, std::size_t identity_size,
    generativeqc_backend backend, generativeqc_ks_options* output, std::uint32_t* selected) {
  if (!output || !selected || output->struct_size < sizeof(*output) ||
      output->abi_version != GENERATIVEQC_ABI_VERSION || name_size > 256 || (name_size && !name) ||
      (identity_size && !identity) || (!name_size && !identity_size) ||
      (identity_size && identity_size != 64) ||
      (backend != GENERATIVEQC_BACKEND_CPU_REFERENCE && backend != GENERATIVEQC_BACKEND_CUDA) ||
      !disjoint(output, output->struct_size, selected, sizeof(*selected)) ||
      !disjoint(output, output->struct_size, name, name_size) ||
      !disjoint(output, output->struct_size, identity, identity_size) ||
      !disjoint(selected, sizeof(*selected), name, name_size) ||
      !disjoint(selected, sizeof(*selected), identity, identity_size))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;

  const std::string_view domain(output->scf_domain ? output->scf_domain : "");
  if ((output->scf_domain &&
       (!disjoint(output, output->struct_size, domain.data(), domain.size() + 1) ||
        !disjoint(selected, sizeof(*selected), domain.data(), domain.size() + 1))) ||
      output->element_radius_count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
      !disjoint(output, output->struct_size, output->element_radii,
                static_cast<std::size_t>(output->element_radius_count) * sizeof(double)) ||
      !disjoint(selected, sizeof(*selected), output->element_radii,
                static_cast<std::size_t>(output->element_radius_count) * sizeof(double)))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const std::string_view requested_name(name ? name : "", name_size);
  const std::string_view requested_identity(identity ? identity : "", identity_size);
  if (requested_name.find('\0') != std::string_view::npos ||
      !std::all_of(requested_identity.begin(), requested_identity.end(), [](char character) {
        return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
      }))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;

  using namespace generativeqc::methods::named_ks;
  const Method* entry = nullptr;
  if (!requested_name.empty()) {
    entry = find_method(requested_name);
  } else {
    for (const auto& candidate : methods()) {
      if (candidate.identity == requested_identity &&
          (backend == GENERATIVEQC_BACKEND_CUDA ? candidate.cuda : candidate.cpu)) {
        entry = &candidate;
        break;
      }
    }
  }
  if (!entry || (!requested_identity.empty() && entry->identity != requested_identity) ||
      !(backend == GENERATIVEQC_BACKEND_CUDA ? entry->cuda : entry->cpu) ||
      (output->scf_domain && entry->domain != domain)) {
    *selected = 0;
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  auto candidate = *output;
  candidate.scf_domain = entry->domain.data();
  candidate.spin_channels = entry->spin;
  candidate.semilocal_components = components(*entry).data();
  candidate.semilocal_component_count = entry->component_count;
  candidate.semilocal_range_omega = 0;
  candidate.exchange_terms = entry->exchange_count ? exchanges(*entry).data() : nullptr;
  candidate.exchange_term_count = entry->exchange_count;
  candidate.has_nonlocal_correlation = 0;
  candidate.nonlocal_variant = 0;
  candidate.nonlocal_b = candidate.nonlocal_c = candidate.nonlocal_coefficient = 0;
  candidate.nonlocal_maximum_bytes = 0;
  *output = candidate;
  *selected = 1;
  return GENERATIVEQC_STATUS_SUCCESS;
}
