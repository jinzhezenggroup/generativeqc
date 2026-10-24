#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string_view>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/generativeqc.h"
#include "molecule/basis_catalog.hpp"

namespace {
namespace catalog = generativeqc::molecule;

struct Range {
  std::uintptr_t begin{}, end{};
};

template <class Record>
bool aligned(const Record* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(Record) == 0;
}

Range range(const void* pointer, std::size_t count, std::size_t size, std::size_t alignment) {
  if (!count) return {};
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  if (!pointer || address % alignment || count > std::numeric_limits<std::size_t>::max() / size ||
      address > std::numeric_limits<std::uintptr_t>::max() - count * size)
    throw std::invalid_argument("invalid builtin basis buffer range or alignment");
  return {address, address + count * size};
}

bool disjoint(Range left, Range right) {
  return left.end <= right.begin || right.end <= left.begin;
}

struct Request {
  std::string_view name;
  std::span<const generativeqc_atom> atoms;
  catalog::BasisExpansionCapacity capacity;
};

/** Validate every borrowed extent before computing raw output capacity.
 * Include version-extension tails in alias checks even though this ABI reads
 * only its known prefix. Capacity and expansion share the general-template owner.
 */
Request request(const generativeqc_builtin_basis_descriptor_v1& descriptor, Range output) {
  if (!descriptor.name_count || descriptor.name_count > 256 || !descriptor.atom_count ||
      descriptor.atom_count > std::numeric_limits<std::uint32_t>::max() || descriptor.reserved ||
      descriptor.maximum_expansion_bytes > std::numeric_limits<std::size_t>::max() ||
      (descriptor.expected_pack_checksum_count != 0 &&
       descriptor.expected_pack_checksum_count != 64))
    throw std::invalid_argument("invalid builtin basis metadata dimensions");
  const Range inputs[] = {
      range(&descriptor, 1, descriptor.struct_size, alignof(decltype(descriptor))),
      range(descriptor.name, descriptor.name_count, 1, 1),
      range(descriptor.atoms, descriptor.atom_count, sizeof(generativeqc_atom),
            alignof(generativeqc_atom)),
      range(descriptor.expected_pack_checksum, descriptor.expected_pack_checksum_count, 1, 1)};
  for (const auto input : inputs)
    if (!disjoint(input, output)) throw std::invalid_argument("builtin basis output aliases input");
  if (descriptor.expected_pack_checksum_count &&
      std::string_view(descriptor.expected_pack_checksum, 64) !=
          catalog::bundled_basis_pack_checksum())
    throw std::invalid_argument("builtin basis pack checksum mismatch");
  Request result{{descriptor.name, static_cast<std::size_t>(descriptor.name_count)},
                 {descriptor.atoms, static_cast<std::size_t>(descriptor.atom_count)},
                 {}};
  result.capacity =
      catalog::bundled_basis_capacity(result.name, result.atoms, descriptor.representation);
  return result;
}
}  // namespace

extern "C" uint32_t generativeqc_builtin_basis_count_v1(void) {
  return static_cast<std::uint32_t>(catalog::bundled_basis_count());
}

extern "C" const char* generativeqc_builtin_basis_name_v1(uint64_t index) {
  if (index >= catalog::bundled_basis_count()) return nullptr;
  return catalog::bundled_basis_name(static_cast<std::size_t>(index)).data();
}

extern "C" generativeqc_status generativeqc_builtin_basis_query_v1(
    const generativeqc_builtin_basis_descriptor_v1* descriptor,
    generativeqc_builtin_basis_capacity_v1* output) {
  if (!descriptor || !output || !aligned(descriptor) || !aligned(output))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(descriptor) ||
      !generativeqc::api::valid_descriptor(output))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  try {
    const auto input =
        request(*descriptor, range(output, 1, output->struct_size, alignof(decltype(*output))));
    const auto& required = input.capacity;
    *output = {sizeof(*output),
               GENERATIVEQC_ABI_VERSION,
               required.shell_count,
               required.primitive_count,
               required.ao_count,
               required.buffer_bytes,
               catalog::canonical_basis_name(input.name).data(),
               catalog::bundled_basis_pack_checksum().data()};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception();
  }
}

extern "C" generativeqc_status generativeqc_system_create_builtin_basis_v1(
    generativeqc_context* context, const generativeqc_builtin_basis_descriptor_v1* descriptor,
    generativeqc_system** system) {
  if (!context || !descriptor || !system || !aligned(descriptor) || !aligned(system))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(descriptor)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    const auto input =
        request(*descriptor, range(system, 1, sizeof(*system), alignof(decltype(*system))));
    if (descriptor->maximum_expansion_bytes &&
        input.capacity.buffer_bytes > descriptor->maximum_expansion_bytes)
      throw std::bad_alloc{};
    const auto expanded =
        catalog::expand_bundled_basis(input.name, input.atoms, descriptor->representation);
    const generativeqc_system_descriptor native{
        sizeof(generativeqc_system_descriptor),
        GENERATIVEQC_ABI_VERSION,
        input.atoms.data(),
        static_cast<std::uint32_t>(input.atoms.size()),
        expanded.shells.data(),
        static_cast<std::uint32_t>(expanded.shells.size()),
        expanded.primitives.data(),
        static_cast<std::uint32_t>(expanded.primitives.size()),
        descriptor->charge,
        descriptor->multiplicity,
        descriptor->representation};
    generativeqc_system* created{};
    const auto status = generativeqc_system_create(context, &native, &created);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    *system = created;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}
