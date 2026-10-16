#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "generativeqc/generativeqc.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Record>
Record versioned() {
  Record result{};
  result.struct_size = sizeof(result);
  result.abi_version = GENERATIVEQC_ABI_VERSION;
  return result;
}

struct Context {
  generativeqc_context* value{};
  ~Context() { generativeqc_context_destroy(value); }
};

struct System {
  generativeqc_system* value{};
  ~System() { generativeqc_system_destroy(value); }
};

void run() {
  const auto count = generativeqc_builtin_basis_count_v1();
  require(
      count >= 3 && generativeqc_builtin_basis_name_v1(count) == nullptr &&
          generativeqc_builtin_basis_name_v1(std::numeric_limits<std::uint64_t>::max()) == nullptr,
      "installed discovery does not bound its ordinals");
  for (std::uint64_t index = 0; index < count; ++index)
    require(generativeqc_builtin_basis_name_v1(index) != nullptr &&
                (index == 0 || std::strcmp(generativeqc_builtin_basis_name_v1(index - 1),
                                           generativeqc_builtin_basis_name_v1(index)) < 0),
            "installed names are missing or not deterministically ordered");
  std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}}};
  auto descriptor = versioned<generativeqc_builtin_basis_descriptor_v1>();
  descriptor.name = "STO_3G";
  descriptor.name_count = std::strlen(descriptor.name);
  descriptor.atoms = atoms.data();
  descriptor.atom_count = atoms.size();
  descriptor.representation = GENERATIVEQC_BASIS_CARTESIAN;
  descriptor.multiplicity = 1;
  auto capacity = versioned<generativeqc_builtin_basis_capacity_v1>();
  require(
      generativeqc_builtin_basis_query_v1(&descriptor, &capacity) == GENERATIVEQC_STATUS_SUCCESS &&
          capacity.shell_count == 2 && capacity.primitive_count == 6 && capacity.ao_count == 2 &&
          capacity.expansion_bytes ==
              2 * sizeof(generativeqc_shell) + 6 * sizeof(generativeqc_primitive) &&
          std::strcmp(capacity.canonical_name, "sto-3g") == 0 &&
          std::strlen(capacity.pack_checksum) == 64,
      "installed shape query differs from the independent H2 basis");
  const auto published = capacity;
  alignas(generativeqc_builtin_basis_descriptor_v1)
      std::array<unsigned char, sizeof(descriptor) + 1>
          unaligned_descriptor{};
  std::memcpy(unaligned_descriptor.data() + 1, &descriptor, sizeof(descriptor));
  require(generativeqc_builtin_basis_query_v1(
              reinterpret_cast<const generativeqc_builtin_basis_descriptor_v1*>(
                  unaligned_descriptor.data() + 1),
              &capacity) == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              std::memcmp(&capacity, &published, sizeof(capacity)) == 0,
          "query dereferenced an unaligned descriptor");
  alignas(generativeqc_builtin_basis_capacity_v1) std::array<unsigned char, sizeof(capacity) + 1>
      unaligned_capacity{};
  std::memcpy(unaligned_capacity.data() + 1, &capacity, sizeof(capacity));
  require(generativeqc_builtin_basis_query_v1(
              &descriptor, reinterpret_cast<generativeqc_builtin_basis_capacity_v1*>(
                               unaligned_capacity.data() + 1)) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              std::memcmp(unaligned_capacity.data() + 1, &published, sizeof(published)) == 0,
          "query dereferenced or modified an unaligned capacity");
  const auto refused_query = [&](const auto& malformed, generativeqc_status expected) {
    auto output = published;
    require(generativeqc_builtin_basis_query_v1(&malformed, &output) == expected &&
                std::memcmp(&output, &published, sizeof(output)) == 0,
            "query refusal modified caller publication");
  };
  auto malformed = descriptor;
  malformed.abi_version += 1;
  refused_query(malformed, GENERATIVEQC_STATUS_ABI_MISMATCH);
  malformed = descriptor;
  malformed.atom_count = std::numeric_limits<std::uint64_t>::max();
  malformed.atoms = reinterpret_cast<const generativeqc_atom*>(1);
  refused_query(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  malformed = descriptor;
  malformed.reserved = 1;
  refused_query(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  malformed = descriptor;
  malformed.name_count = 257;
  refused_query(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  malformed = descriptor;
  malformed.expected_pack_checksum = nullptr;
  malformed.expected_pack_checksum_count = 64;
  refused_query(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  std::array<char, 64> wrong_checksum;
  wrong_checksum.fill('x');
  malformed.expected_pack_checksum = wrong_checksum.data();
  refused_query(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  struct Extended {
    generativeqc_builtin_basis_descriptor_v1 descriptor;
    generativeqc_builtin_basis_capacity_v1 aliased;
  } extended{descriptor, published};
  extended.descriptor.struct_size = sizeof(extended);
  require(generativeqc_builtin_basis_query_v1(&extended.descriptor, &extended.aliased) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "query accepted output inside a descriptor extension");
  struct ExtendedCapacity {
    generativeqc_builtin_basis_capacity_v1 capacity;
    std::uint64_t tail;
  } extended_capacity{published, 0x12345678};
  extended_capacity.capacity.struct_size = sizeof(extended_capacity);
  require(generativeqc_builtin_basis_query_v1(&descriptor, &extended_capacity.capacity) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              extended_capacity.tail == 0x12345678,
          "query overwrote a future capacity extension");

  Context context;
  auto context_descriptor = versioned<generativeqc_context_descriptor>();
  require(generativeqc_context_create(&context_descriptor, &context.value) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "CPU context creation failed");
  auto* const sentinel = reinterpret_cast<generativeqc_system*>(std::uintptr_t{0x100});
  generativeqc_system* output = sentinel;
  descriptor.expected_pack_checksum = capacity.pack_checksum;
  descriptor.expected_pack_checksum_count = 64;
  descriptor.maximum_expansion_bytes = capacity.expansion_bytes - 1;
  atoms[0].x = std::numeric_limits<double>::quiet_NaN();
  require(
      generativeqc_builtin_basis_query_v1(&descriptor, &capacity) == GENERATIVEQC_STATUS_SUCCESS,
      "metadata query read coordinate physics or enforced an execution budget");
  require(generativeqc_system_create_builtin_basis_v1(context.value, &descriptor, &output) ==
                  GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
              output == sentinel,
          "below-limit construction read coordinates or published a partial handle");
  descriptor.maximum_expansion_bytes = capacity.expansion_bytes;
  require(generativeqc_system_create_builtin_basis_v1(context.value, &descriptor, &output) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              output == sentinel,
          "physical normalization refusal modified the handle slot");
  atoms[0].x = 0;
  require(generativeqc_system_create_builtin_basis_v1(context.value, &descriptor, &output) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "exact-limit native built-in system construction failed");
  System built_in{output};
  struct AliasedSystem {
    generativeqc_builtin_basis_descriptor_v1 descriptor;
    generativeqc_system* slot;
  } aliased_system{descriptor, sentinel};
  aliased_system.descriptor.struct_size = sizeof(aliased_system);
  require(generativeqc_system_create_builtin_basis_v1(context.value, &aliased_system.descriptor,
                                                      &aliased_system.slot) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              aliased_system.slot == sentinel,
          "construction accepted output in a descriptor extension");

  const std::array<generativeqc_shell, 2> shells{{{0, 0, 0, 3}, {1, 0, 3, 3}}};
  const std::array<generativeqc_primitive, 6> primitives{{{3.425250914, 0.1543289673},
                                                          {0.6239137298, 0.5353281423},
                                                          {0.1688554040, 0.4446345422},
                                                          {3.425250914, 0.1543289673},
                                                          {0.6239137298, 0.5353281423},
                                                          {0.1688554040, 0.4446345422}}};
  const generativeqc_system_descriptor explicit_descriptor{sizeof(generativeqc_system_descriptor),
                                                           GENERATIVEQC_ABI_VERSION,
                                                           atoms.data(),
                                                           2,
                                                           shells.data(),
                                                           2,
                                                           primitives.data(),
                                                           6,
                                                           0,
                                                           1,
                                                           GENERATIVEQC_BASIS_CARTESIAN};
  System explicit_system;
  require(generativeqc_system_create(context.value, &explicit_descriptor, &explicit_system.value) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "independent explicit basis system failed");
  std::array<double, 4> cross{}, reference{};
  require(generativeqc_system_cross_overlap_cpu(context.value, built_in.value,
                                                explicit_system.value, cross.data(),
                                                cross.size()) == GENERATIVEQC_STATUS_SUCCESS &&
              generativeqc_system_cross_overlap_cpu(
                  context.value, explicit_system.value, explicit_system.value, reference.data(),
                  reference.size()) == GENERATIVEQC_STATUS_SUCCESS,
          "native-only installed overlap publication failed");
  for (std::size_t index = 0; index < cross.size(); ++index)
    require(std::abs(cross[index] - reference[index]) < 1e-14,
            "compiled catalog changed actual normalized overlap data");
}
}  // namespace

int main() {
  try {
    run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
