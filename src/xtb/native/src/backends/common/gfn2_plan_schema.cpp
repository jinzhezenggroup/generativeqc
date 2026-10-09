#include "backends/common/gfn2_plan_schema.hpp"
#include "runtime/ragged_topology.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>

namespace generativeqc::xtb::detail {

namespace {

namespace shared_topology = ::generativeqc::runtime::ragged;

// Native PODs stay unchanged because CUDA kernels and descriptors consume them.
// Field-wise adapters borrow the same pointers without type-punning objects.
static_assert(sizeof(Gfn2AtomPair) == sizeof(shared_topology::AtomPairLayout));
static_assert(alignof(Gfn2AtomPair) == alignof(shared_topology::AtomPairLayout));
static_assert(offsetof(Gfn2AtomPair, first) == offsetof(shared_topology::AtomPairLayout, first));
static_assert(offsetof(Gfn2AtomPair, second) == offsetof(shared_topology::AtomPairLayout, second));

Gfn2PlanSchemaDiagnostic native_diagnostic(shared_topology::Diagnostic diagnostic) noexcept {
  return {static_cast<Gfn2PlanSchemaError>(diagnostic.error),
          static_cast<Gfn2PlanSchemaField>(diagnostic.field), diagnostic.index};
}

shared_topology::TopologyView shared_view(const Gfn2RaggedTopologyView& source) noexcept {
  return {
      static_cast<shared_topology::MemorySpace>(source.memory_space),
      static_cast<shared_topology::PairMapKind>(source.pair_map_kind),
      source.plan_token,
      source.batch_size,
      source.total_atoms,
      source.total_shells,
      source.total_orbitals,
      source.total_matrix_elements,
      source.total_pairs,
      source.bucket_count,
      source.atom_offset_count,
      source.batch_shell_offset_count,
      source.batch_orbital_offset_count,
      source.matrix_offset_count,
      source.atom_shell_offset_count,
      source.shell_orbital_offset_count,
      source.shell_to_atom_count,
      source.orbital_to_shell_count,
      source.orbital_to_atom_count,
      source.pair_offset_count,
      source.atom_pair_count,
      source.bucket_offset_count,
      source.bucket_system_count,
      source.bucket_orbital_count,
      source.atom_offsets,
      source.batch_shell_offsets,
      source.batch_orbital_offsets,
      source.matrix_offsets,
      source.atom_shell_offsets,
      source.shell_orbital_offsets,
      source.shell_to_atom,
      source.orbital_to_shell,
      source.orbital_to_atom,
      source.pair_offsets,
      source.atom_pairs,
      source.bucket_offsets,
      source.bucket_systems,
      source.bucket_orbital_counts,
  };
}

shared_topology::AtomProjectionView shared_view(const Gfn2AtomProjectionView& source) noexcept {
  return {
      static_cast<shared_topology::MemorySpace>(source.memory_space),
      source.plan_token,
      source.batch_size,
      source.total_atoms,
      source.atom_offset_count,
      source.atom_offsets,
  };
}

void native_projection(const shared_topology::AtomProjectionView& source,
                       Gfn2AtomProjectionView& destination) noexcept {
  destination.memory_space = static_cast<Gfn2PlanMemorySpace>(source.memory_space);
  destination.plan_token = source.plan_token;
  destination.batch_size = source.batch_size;
  destination.total_atoms = source.total_atoms;
  destination.atom_offset_count = source.atom_offset_count;
  destination.atom_offsets = source.atom_offsets;
}

shared_topology::ShellOwnershipProjectionView shared_view(const Gfn2ShellOwnershipProjectionView& source) noexcept {
  return {
      static_cast<shared_topology::MemorySpace>(source.memory_space),
      source.plan_token,
      source.batch_size,
      source.total_atoms,
      source.total_shells,
      source.batch_shell_offset_count,
      source.atom_shell_offset_count,
      source.shell_to_atom_count,
      source.batch_shell_offsets,
      source.atom_shell_offsets,
      source.shell_to_atom,
  };
}

void native_projection(const shared_topology::ShellOwnershipProjectionView& source,
                       Gfn2ShellOwnershipProjectionView& destination) noexcept {
  destination.memory_space = static_cast<Gfn2PlanMemorySpace>(source.memory_space);
  destination.plan_token = source.plan_token;
  destination.batch_size = source.batch_size;
  destination.total_atoms = source.total_atoms;
  destination.total_shells = source.total_shells;
  destination.batch_shell_offset_count = source.batch_shell_offset_count;
  destination.atom_shell_offset_count = source.atom_shell_offset_count;
  destination.shell_to_atom_count = source.shell_to_atom_count;
  destination.batch_shell_offsets = source.batch_shell_offsets;
  destination.atom_shell_offsets = source.atom_shell_offsets;
  destination.shell_to_atom = source.shell_to_atom;
}

shared_topology::AOMatrixProjectionView shared_view(const Gfn2AOMatrixProjectionView& source) noexcept {
  return {
      static_cast<shared_topology::MemorySpace>(source.memory_space),
      source.plan_token,
      source.batch_size,
      source.total_shells,
      source.total_orbitals,
      source.total_matrix_elements,
      source.batch_orbital_offset_count,
      source.matrix_offset_count,
      source.shell_orbital_offset_count,
      source.orbital_to_shell_count,
      source.orbital_to_atom_count,
      source.batch_orbital_offsets,
      source.matrix_offsets,
      source.shell_orbital_offsets,
      source.orbital_to_shell,
      source.orbital_to_atom,
  };
}

void native_projection(const shared_topology::AOMatrixProjectionView& source,
                       Gfn2AOMatrixProjectionView& destination) noexcept {
  destination.memory_space = static_cast<Gfn2PlanMemorySpace>(source.memory_space);
  destination.plan_token = source.plan_token;
  destination.batch_size = source.batch_size;
  destination.total_shells = source.total_shells;
  destination.total_orbitals = source.total_orbitals;
  destination.total_matrix_elements = source.total_matrix_elements;
  destination.batch_orbital_offset_count = source.batch_orbital_offset_count;
  destination.matrix_offset_count = source.matrix_offset_count;
  destination.shell_orbital_offset_count = source.shell_orbital_offset_count;
  destination.orbital_to_shell_count = source.orbital_to_shell_count;
  destination.orbital_to_atom_count = source.orbital_to_atom_count;
  destination.batch_orbital_offsets = source.batch_orbital_offsets;
  destination.matrix_offsets = source.matrix_offsets;
  destination.shell_orbital_offsets = source.shell_orbital_offsets;
  destination.orbital_to_shell = source.orbital_to_shell;
  destination.orbital_to_atom = source.orbital_to_atom;
}

shared_topology::PackedAllPairProjectionView shared_view(const Gfn2PackedAllPairProjectionView& source) noexcept {
  return {
      static_cast<shared_topology::MemorySpace>(source.memory_space),
      source.plan_token,
      source.batch_size,
      source.total_pairs,
      source.pair_offset_count,
      source.pair_offsets,
  };
}

void native_projection(const shared_topology::PackedAllPairProjectionView& source,
                       Gfn2PackedAllPairProjectionView& destination) noexcept {
  destination.memory_space = static_cast<Gfn2PlanMemorySpace>(source.memory_space);
  destination.plan_token = source.plan_token;
  destination.batch_size = source.batch_size;
  destination.total_pairs = source.total_pairs;
  destination.pair_offset_count = source.pair_offset_count;
  destination.pair_offsets = source.pair_offsets;
}

shared_topology::AOBucketProjectionView shared_view(const Gfn2AOBucketProjectionView& source) noexcept {
  return {
      static_cast<shared_topology::MemorySpace>(source.memory_space),
      source.plan_token,
      source.batch_size,
      source.bucket_count,
      source.bucket_offset_count,
      source.bucket_system_count,
      source.bucket_orbital_count,
      source.bucket_offsets,
      source.bucket_systems,
      source.bucket_orbital_counts,
  };
}

void native_projection(const shared_topology::AOBucketProjectionView& source,
                       Gfn2AOBucketProjectionView& destination) noexcept {
  destination.memory_space = static_cast<Gfn2PlanMemorySpace>(source.memory_space);
  destination.plan_token = source.plan_token;
  destination.batch_size = source.batch_size;
  destination.bucket_count = source.bucket_count;
  destination.bucket_offset_count = source.bucket_offset_count;
  destination.bucket_system_count = source.bucket_system_count;
  destination.bucket_orbital_count = source.bucket_orbital_count;
  destination.bucket_offsets = source.bucket_offsets;
  destination.bucket_systems = source.bucket_systems;
  destination.bucket_orbital_counts = source.bucket_orbital_counts;
}

shared_topology::ElementIdentityProjectionView shared_view(const Gfn2ElementIdentityProjectionView& source) noexcept {
  return {
      static_cast<shared_topology::MemorySpace>(source.memory_space),
      source.plan_token,
      source.total_atoms,
      source.atomic_number_count,
      source.element_fingerprint,
      source.atomic_numbers,
  };
}

void native_projection(const shared_topology::ElementIdentityProjectionView& source,
                       Gfn2ElementIdentityProjectionView& destination) noexcept {
  destination.memory_space = static_cast<Gfn2PlanMemorySpace>(source.memory_space);
  destination.plan_token = source.plan_token;
  destination.total_atoms = source.total_atoms;
  destination.atomic_number_count = source.atomic_number_count;
  destination.element_fingerprint = source.element_fingerprint;
  destination.atomic_numbers = source.atomic_numbers;
}

}  // namespace


namespace {

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
  Gfn2PlanSchemaField field = Gfn2PlanSchemaField::kNone;
  bool active = false;
};

constexpr Gfn2PlanSchemaDiagnostic success() noexcept { return {}; }

constexpr Gfn2PlanSchemaDiagnostic failure(Gfn2PlanSchemaError error, Gfn2PlanSchemaField field,
                                           std::int64_t index = -1) noexcept {
  return {error, field, index};
}

constexpr bool known_memory_space(Gfn2PlanMemorySpace memory_space) noexcept {
  return memory_space == Gfn2PlanMemorySpace::kHost ||
         memory_space == Gfn2PlanMemorySpace::kCudaDevice ||
         memory_space == Gfn2PlanMemorySpace::kHipDevice;
}

template <typename T>
Gfn2PlanSchemaDiagnostic make_range(const T* pointer, std::int64_t count, Gfn2PlanSchemaField field,
                                    AddressRange& range) noexcept {
  if (count < 0) {
    return failure(Gfn2PlanSchemaError::kInvalidCount, field);
  }
  if (count == 0) {
    if (pointer != nullptr) {
      return failure(Gfn2PlanSchemaError::kInvalidCount, field);
    }
    range = {};
    return success();
  }
  if (pointer == nullptr) {
    return failure(Gfn2PlanSchemaError::kNullPointer, field);
  }
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
  if (address % alignof(T) != 0u) {
    return failure(Gfn2PlanSchemaError::kMisalignedPointer, field);
  }
  constexpr std::uintmax_t kMaximumSize = std::numeric_limits<std::size_t>::max();
  if (static_cast<std::uintmax_t>(count) > kMaximumSize / sizeof(T)) {
    return failure(Gfn2PlanSchemaError::kCountOverflow, field);
  }
  const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(T);
  if (address > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return failure(Gfn2PlanSchemaError::kAddressOverflow, field);
  }
  range = {address, address + bytes, field, true};
  return success();
}

bool overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return first.active && second.active && first.begin < second.end && second.begin < first.end;
}

template <std::size_t N>
Gfn2PlanSchemaDiagnostic validate_aliases(const std::array<AddressRange, N>& ranges) noexcept {
  for (std::size_t first = 0u; first < ranges.size(); ++first) {
    for (std::size_t second = first + 1u; second < ranges.size(); ++second) {
      if (overlap(ranges[first], ranges[second])) {
        return failure(Gfn2PlanSchemaError::kAliasedRange, ranges[second].field);
      }
    }
  }
  return success();
}

bool add_one(std::int64_t value, std::int64_t& result) noexcept {
  if (value == std::numeric_limits<std::int64_t>::max()) {
    return false;
  }
  result = value + 1;
  return true;
}

bool product(std::int64_t first, std::int64_t second, std::int64_t& result) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  result = first * second;
  return true;
}

std::uint64_t mix_hash(std::uint64_t value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31u);
}

void hash_append(std::uint64_t value, std::uint64_t& hash) noexcept {
  hash = mix_hash(hash ^ mix_hash(value + 0x9e3779b97f4a7c15ULL));
}

Gfn2PlanSchemaDiagnostic validate_offsets(const std::int64_t* offsets, std::int64_t partitions,
                                          std::int64_t endpoint,
                                          Gfn2PlanSchemaField field) noexcept {
  if (offsets[0] != 0) {
    return failure(Gfn2PlanSchemaError::kInvalidOffsets, field, 0);
  }
  for (std::int64_t index = 0; index < partitions; ++index) {
    if (offsets[index] < 0 || offsets[index] > offsets[index + 1] ||
        offsets[index + 1] > endpoint) {
      return failure(Gfn2PlanSchemaError::kInvalidOffsets, field, index);
    }
  }
  if (offsets[partitions] != endpoint) {
    return failure(Gfn2PlanSchemaError::kInvalidOffsets, field, partitions);
  }
  return success();
}

Gfn2PlanSchemaDiagnostic validate_no_topology_alias(const Gfn2RaggedTopologyView& topology,
                                                    const AddressRange& candidate) noexcept {
  if (!candidate.active) {
    return success();
  }
  const void* pointers[] = {
      topology.atom_offsets,
      topology.batch_shell_offsets,
      topology.batch_orbital_offsets,
      topology.matrix_offsets,
      topology.atom_shell_offsets,
      topology.shell_orbital_offsets,
      topology.shell_to_atom,
      topology.orbital_to_shell,
      topology.orbital_to_atom,
      topology.pair_offsets,
      topology.atom_pairs,
      topology.bucket_offsets,
      topology.bucket_systems,
      topology.bucket_orbital_counts,
  };
  const std::int64_t counts[] = {
      topology.atom_offset_count,
      topology.batch_shell_offset_count,
      topology.batch_orbital_offset_count,
      topology.matrix_offset_count,
      topology.atom_shell_offset_count,
      topology.shell_orbital_offset_count,
      topology.shell_to_atom_count,
      topology.orbital_to_shell_count,
      topology.orbital_to_atom_count,
      topology.pair_offset_count,
      topology.atom_pair_count,
      topology.bucket_offset_count,
      topology.bucket_system_count,
      topology.bucket_orbital_count,
  };
  const std::size_t element_sizes[] = {
      sizeof(std::int64_t), sizeof(std::int64_t), sizeof(std::int64_t), sizeof(std::int64_t),
      sizeof(std::int64_t), sizeof(std::int64_t), sizeof(std::int64_t), sizeof(std::int64_t),
      sizeof(std::int64_t), sizeof(std::int64_t), sizeof(Gfn2AtomPair), sizeof(std::int64_t),
      sizeof(std::int32_t), sizeof(std::int32_t),
  };
  for (std::size_t index = 0u; index < std::size(pointers); ++index) {
    if (counts[index] != 0) {
      const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointers[index]);
      const std::size_t bytes = static_cast<std::size_t>(counts[index]) * element_sizes[index];
      const AddressRange topology_range{begin, begin + bytes, Gfn2PlanSchemaField::kTopology, true};
      if (overlap(candidate, topology_range)) {
        return failure(Gfn2PlanSchemaError::kAliasedRange, candidate.field);
      }
    }
  }
  return success();
}

}  // namespace

std::uint64_t gfn2_wavefunction_layout_fingerprint_host(
    const Gfn2WavefunctionLayoutView& layout) noexcept {
  std::int64_t offset_count = 0;
  if (layout.memory_space != Gfn2PlanMemorySpace::kHost || layout.plan_token == 0u ||
      layout.batch_size <= 0 || !add_one(layout.batch_size, offset_count) ||
      layout.spin_channel_count != layout.batch_size ||
      layout.spin_channel_offset_count != offset_count ||
      layout.spin_orbital_offset_count != offset_count ||
      layout.spin_matrix_offset_count != offset_count ||
      layout.spin_shell_offset_count != offset_count ||
      layout.spin_atom_offset_count != offset_count || layout.spin_channels == nullptr ||
      layout.spin_channel_offsets == nullptr || layout.spin_orbital_offsets == nullptr ||
      layout.spin_matrix_offsets == nullptr || layout.spin_shell_offsets == nullptr ||
      layout.spin_atom_offsets == nullptr) {
    return 0u;
  }

  std::uint64_t hash = 0x6a09e667f3bcc909ULL;
  hash_append(1u, hash);  // Fingerprint schema version.
  hash_append(layout.plan_token, hash);
  hash_append(static_cast<std::uint64_t>(layout.batch_size), hash);
  hash_append(static_cast<std::uint64_t>(layout.total_spin_channels), hash);
  hash_append(static_cast<std::uint64_t>(layout.total_spin_orbitals), hash);
  hash_append(static_cast<std::uint64_t>(layout.total_spin_matrix_elements), hash);
  hash_append(static_cast<std::uint64_t>(layout.total_spin_shells), hash);
  hash_append(static_cast<std::uint64_t>(layout.total_spin_atoms), hash);

  for (std::int64_t system = 0; system < layout.batch_size; ++system) {
    hash_append(static_cast<std::uint32_t>(layout.spin_channels[system]), hash);
  }
  const auto append_offsets = [&hash, &layout](const std::int64_t* offsets,
                                               std::uint64_t field_tag) noexcept {
    hash_append(field_tag, hash);
    /* batch_size == INT64_MAX was rejected above, so incrementing index is
     * defined through the final batch endpoint. */
    for (std::int64_t index = 0; index < layout.batch_size + 1; ++index) {
      hash_append(static_cast<std::uint64_t>(offsets[index]), hash);
    }
  };
  append_offsets(layout.spin_channel_offsets, 1u);
  append_offsets(layout.spin_orbital_offsets, 2u);
  append_offsets(layout.spin_matrix_offsets, 3u);
  append_offsets(layout.spin_shell_offsets, 4u);
  append_offsets(layout.spin_atom_offsets, 5u);
  return hash == 0u ? 1u : hash;
}

Gfn2PlanSchemaDiagnostic validate_gfn2_topology_binding(
    const Gfn2RaggedTopologyView& topology, Gfn2PlanMemorySpace expected_memory_space) noexcept {
  return native_diagnostic(shared_topology::validate_topology_binding(
      shared_view(topology), static_cast<shared_topology::MemorySpace>(expected_memory_space)));
}

Gfn2PlanSchemaDiagnostic validate_gfn2_topology_host(
    const Gfn2RaggedTopologyView& topology) noexcept {
  return native_diagnostic(shared_topology::validate_topology_host(shared_view(topology)));
}

Gfn2PlanSchemaDiagnostic bind_gfn2_topology_host(const Gfn2RaggedTopologyView& candidate,
                                                 Gfn2RaggedTopologyView& binding) noexcept {
  binding = {};
  shared_topology::TopologyView result{};
  const auto diagnostic = shared_topology::bind_topology_host(shared_view(candidate), result);
  if (diagnostic.error == shared_topology::Error::kSuccess) binding = candidate;
  return native_diagnostic(diagnostic);
}

Gfn2PlanSchemaDiagnostic validate_gfn2_wavefunction_layout_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2WavefunctionLayoutView& layout,
    Gfn2PlanMemorySpace expected_memory_space) noexcept {
  Gfn2PlanSchemaDiagnostic diagnostic =
      validate_gfn2_topology_binding(topology, expected_memory_space);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  if (!known_memory_space(layout.memory_space) || layout.memory_space != expected_memory_space) {
    return failure(Gfn2PlanSchemaError::kInvalidMemorySpace, Gfn2PlanSchemaField::kSpinChannels);
  }
  if (layout.plan_token == 0u) {
    return failure(Gfn2PlanSchemaError::kInvalidPlanToken, Gfn2PlanSchemaField::kSpinChannels);
  }
  if (layout.layout_fingerprint == 0u) {
    return failure(Gfn2PlanSchemaError::kInvalidLayoutFingerprint,
                   Gfn2PlanSchemaField::kWavefunctionLayoutFingerprint);
  }
  if (layout.plan_token != topology.plan_token || layout.batch_size != topology.batch_size) {
    return failure(Gfn2PlanSchemaError::kCrossPlan, Gfn2PlanSchemaField::kSpinChannels);
  }
  if (layout.total_spin_channels < 0 || layout.total_spin_orbitals < 0 ||
      layout.total_spin_matrix_elements < 0 || layout.total_spin_shells < 0 ||
      layout.total_spin_atoms < 0) {
    return failure(Gfn2PlanSchemaError::kInvalidCount, Gfn2PlanSchemaField::kSpinChannels);
  }

  std::int64_t offset_count = 0;
  if (!add_one(topology.batch_size, offset_count)) {
    return failure(Gfn2PlanSchemaError::kCountOverflow, Gfn2PlanSchemaField::kSpinChannelOffsets);
  }
  if (layout.spin_channel_count != topology.batch_size ||
      layout.spin_channel_offset_count != offset_count ||
      layout.spin_orbital_offset_count != offset_count ||
      layout.spin_matrix_offset_count != offset_count ||
      layout.spin_shell_offset_count != offset_count ||
      layout.spin_atom_offset_count != offset_count) {
    return failure(Gfn2PlanSchemaError::kInvalidCount, Gfn2PlanSchemaField::kSpinChannels);
  }

  std::array<AddressRange, 6> ranges{};
  diagnostic = make_range(layout.spin_channels, layout.spin_channel_count,
                          Gfn2PlanSchemaField::kSpinChannels, ranges[0]);
  if (diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    diagnostic = make_range(layout.spin_channel_offsets, layout.spin_channel_offset_count,
                            Gfn2PlanSchemaField::kSpinChannelOffsets, ranges[1]);
  }
  if (diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    diagnostic = make_range(layout.spin_orbital_offsets, layout.spin_orbital_offset_count,
                            Gfn2PlanSchemaField::kSpinOrbitalOffsets, ranges[2]);
  }
  if (diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    diagnostic = make_range(layout.spin_matrix_offsets, layout.spin_matrix_offset_count,
                            Gfn2PlanSchemaField::kSpinMatrixOffsets, ranges[3]);
  }
  if (diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    diagnostic = make_range(layout.spin_shell_offsets, layout.spin_shell_offset_count,
                            Gfn2PlanSchemaField::kSpinShellOffsets, ranges[4]);
  }
  if (diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    diagnostic = make_range(layout.spin_atom_offsets, layout.spin_atom_offset_count,
                            Gfn2PlanSchemaField::kSpinAtomOffsets, ranges[5]);
  }
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  diagnostic = validate_aliases(ranges);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  for (const AddressRange& range : ranges) {
    diagnostic = validate_no_topology_alias(topology, range);
    if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
      return diagnostic;
    }
  }
  return success();
}

Gfn2PlanSchemaDiagnostic validate_gfn2_wavefunction_layout_host(
    const Gfn2RaggedTopologyView& topology, const Gfn2WavefunctionLayoutView& layout) noexcept {
  Gfn2PlanSchemaDiagnostic diagnostic =
      validate_gfn2_wavefunction_layout_binding(topology, layout, Gfn2PlanMemorySpace::kHost);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }

  const struct OffsetSet {
    const std::int64_t* values;
    std::int64_t endpoint;
    Gfn2PlanSchemaField field;
  } offsets[] = {
      {layout.spin_channel_offsets, layout.total_spin_channels,
       Gfn2PlanSchemaField::kSpinChannelOffsets},
      {layout.spin_orbital_offsets, layout.total_spin_orbitals,
       Gfn2PlanSchemaField::kSpinOrbitalOffsets},
      {layout.spin_matrix_offsets, layout.total_spin_matrix_elements,
       Gfn2PlanSchemaField::kSpinMatrixOffsets},
      {layout.spin_shell_offsets, layout.total_spin_shells, Gfn2PlanSchemaField::kSpinShellOffsets},
      {layout.spin_atom_offsets, layout.total_spin_atoms, Gfn2PlanSchemaField::kSpinAtomOffsets},
  };
  for (const OffsetSet& set : offsets) {
    diagnostic = validate_offsets(set.values, topology.batch_size, set.endpoint, set.field);
    if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
      return diagnostic;
    }
  }

  for (std::int64_t system = 0; system < topology.batch_size; ++system) {
    const std::int32_t spin_channels = layout.spin_channels[system];
    if (spin_channels != 1 && spin_channels != 2) {
      return failure(Gfn2PlanSchemaError::kInvalidSpinChannels, Gfn2PlanSchemaField::kSpinChannels,
                     system);
    }
    const std::int64_t atoms = topology.atom_offsets[system + 1] - topology.atom_offsets[system];
    const std::int64_t shells =
        topology.batch_shell_offsets[system + 1] - topology.batch_shell_offsets[system];
    const std::int64_t orbitals =
        topology.batch_orbital_offsets[system + 1] - topology.batch_orbital_offsets[system];
    const std::int64_t matrices =
        topology.matrix_offsets[system + 1] - topology.matrix_offsets[system];
    std::int64_t expected_orbitals = 0;
    std::int64_t expected_matrices = 0;
    std::int64_t expected_shells = 0;
    std::int64_t expected_atoms = 0;
    if (!product(spin_channels, orbitals, expected_orbitals) ||
        !product(spin_channels, matrices, expected_matrices) ||
        !product(spin_channels, shells, expected_shells) ||
        !product(spin_channels, atoms, expected_atoms)) {
      return failure(Gfn2PlanSchemaError::kCountOverflow, Gfn2PlanSchemaField::kSpinChannels,
                     system);
    }
    const auto extent = [system](const std::int64_t* values) noexcept {
      return values[system + 1] - values[system];
    };
    if (extent(layout.spin_channel_offsets) != spin_channels) {
      return failure(Gfn2PlanSchemaError::kInvalidWavefunctionExtent,
                     Gfn2PlanSchemaField::kSpinChannelOffsets, system);
    }
    if (extent(layout.spin_orbital_offsets) != expected_orbitals) {
      return failure(Gfn2PlanSchemaError::kInvalidWavefunctionExtent,
                     Gfn2PlanSchemaField::kSpinOrbitalOffsets, system);
    }
    if (extent(layout.spin_matrix_offsets) != expected_matrices) {
      return failure(Gfn2PlanSchemaError::kInvalidWavefunctionExtent,
                     Gfn2PlanSchemaField::kSpinMatrixOffsets, system);
    }
    if (extent(layout.spin_shell_offsets) != expected_shells) {
      return failure(Gfn2PlanSchemaError::kInvalidWavefunctionExtent,
                     Gfn2PlanSchemaField::kSpinShellOffsets, system);
    }
    if (extent(layout.spin_atom_offsets) != expected_atoms) {
      return failure(Gfn2PlanSchemaError::kInvalidWavefunctionExtent,
                     Gfn2PlanSchemaField::kSpinAtomOffsets, system);
    }
  }
  if (gfn2_wavefunction_layout_fingerprint_host(layout) != layout.layout_fingerprint) {
    return failure(Gfn2PlanSchemaError::kInvalidLayoutFingerprint,
                   Gfn2PlanSchemaField::kWavefunctionLayoutFingerprint);
  }
  return success();
}

Gfn2PlanSchemaDiagnostic bind_gfn2_wavefunction_layout_host(
    const Gfn2RaggedTopologyView& topology, const Gfn2WavefunctionLayoutView& candidate,
    Gfn2WavefunctionLayoutView& binding) noexcept {
  binding = {};
  const Gfn2PlanSchemaDiagnostic diagnostic =
      validate_gfn2_wavefunction_layout_host(topology, candidate);
  if (diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    binding = candidate;
  }
  return diagnostic;
}

Gfn2PlanSchemaDiagnostic validate_gfn2_geometry_provenance_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2GeometryCacheProvenanceView& provenance,
    Gfn2PlanMemorySpace expected_memory_space) noexcept {
  Gfn2PlanSchemaDiagnostic diagnostic =
      validate_gfn2_topology_binding(topology, expected_memory_space);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  if (provenance.memory_space != expected_memory_space) {
    return failure(Gfn2PlanSchemaError::kInvalidMemorySpace,
                   Gfn2PlanSchemaField::kGeometryProvenance);
  }
  if (provenance.plan_token == 0u || provenance.plan_token != topology.plan_token) {
    return failure(Gfn2PlanSchemaError::kCrossPlan, Gfn2PlanSchemaField::kGeometryProvenance);
  }
  if (provenance.batch_size != topology.batch_size) {
    return failure(Gfn2PlanSchemaError::kInvalidCount, Gfn2PlanSchemaField::kGeometryProvenance);
  }
  AddressRange generations{};
  if (provenance.generation_scope == Gfn2GenerationScope::kBatch) {
    if (provenance.geometry_generation == 0u || provenance.system_generation_count != 0 ||
        provenance.system_geometry_generations != nullptr) {
      return failure(Gfn2PlanSchemaError::kInvalidCount, Gfn2PlanSchemaField::kGeometryProvenance);
    }
  } else if (provenance.generation_scope == Gfn2GenerationScope::kPerSystem) {
    if (provenance.geometry_generation != 0u ||
        provenance.system_generation_count != topology.batch_size) {
      return failure(Gfn2PlanSchemaError::kInvalidCount,
                     Gfn2PlanSchemaField::kSystemGeometryGenerations);
    }
    diagnostic =
        make_range(provenance.system_geometry_generations, provenance.system_generation_count,
                   Gfn2PlanSchemaField::kSystemGeometryGenerations, generations);
    if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
      return diagnostic;
    }
  } else {
    return failure(Gfn2PlanSchemaError::kInvalidCount, Gfn2PlanSchemaField::kGeometryProvenance);
  }

  return validate_no_topology_alias(topology, generations);
}

Gfn2PlanSchemaDiagnostic validate_gfn2_geometry_provenance_host(
    const Gfn2RaggedTopologyView& topology, const Gfn2GeometryCacheProvenanceView& provenance,
    std::uint64_t expected_geometry_generation, const std::uint8_t* active_mask,
    std::int64_t active_mask_elements) noexcept {
  Gfn2PlanSchemaDiagnostic diagnostic =
      validate_gfn2_geometry_provenance_binding(topology, provenance, Gfn2PlanMemorySpace::kHost);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  if (expected_geometry_generation == 0u) {
    return failure(Gfn2PlanSchemaError::kStaleGeometry, Gfn2PlanSchemaField::kGeometryProvenance);
  }
  AddressRange active_range{};
  if (active_mask == nullptr) {
    if (active_mask_elements != 0) {
      return failure(Gfn2PlanSchemaError::kInvalidActiveMask, Gfn2PlanSchemaField::kActiveMask);
    }
  } else {
    if (active_mask_elements != topology.batch_size) {
      return failure(Gfn2PlanSchemaError::kInvalidActiveMask, Gfn2PlanSchemaField::kActiveMask);
    }
    diagnostic = make_range(active_mask, active_mask_elements, Gfn2PlanSchemaField::kActiveMask,
                            active_range);
    if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
      return diagnostic;
    }
    diagnostic = validate_no_topology_alias(topology, active_range);
    if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
      return diagnostic;
    }
    if (provenance.generation_scope == Gfn2GenerationScope::kPerSystem) {
      AddressRange generations{};
      diagnostic =
          make_range(provenance.system_geometry_generations, provenance.system_generation_count,
                     Gfn2PlanSchemaField::kSystemGeometryGenerations, generations);
      if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
        return diagnostic;
      }
      if (overlap(active_range, generations)) {
        return failure(Gfn2PlanSchemaError::kAliasedRange, Gfn2PlanSchemaField::kActiveMask);
      }
    }
  }
  for (std::int64_t system = 0; active_mask != nullptr && system < topology.batch_size; ++system) {
    if (active_mask[system] > 1u) {
      return failure(Gfn2PlanSchemaError::kInvalidActiveMask, Gfn2PlanSchemaField::kActiveMask,
                     system);
    }
  }
  if (provenance.generation_scope == Gfn2GenerationScope::kBatch) {
    return provenance.geometry_generation == expected_geometry_generation
               ? success()
               : failure(Gfn2PlanSchemaError::kStaleGeometry,
                         Gfn2PlanSchemaField::kGeometryProvenance);
  }
  for (std::int64_t system = 0; system < topology.batch_size; ++system) {
    const bool active = active_mask == nullptr || active_mask[system] == 1u;
    if (active && provenance.system_geometry_generations[system] != expected_geometry_generation) {
      return failure(Gfn2PlanSchemaError::kStaleGeometry,
                     Gfn2PlanSchemaField::kSystemGeometryGenerations, system);
    }
  }
  return success();
}

namespace {

constexpr bool known_pair_list_state(Gfn2PairListState state) noexcept {
  return state == Gfn2PairListState::kCandidate || state == Gfn2PairListState::kCommitted;
}

constexpr bool known_pair_list_role(Gfn2PairListRole role) noexcept {
  return role == Gfn2PairListRole::kCoordination || role == Gfn2PairListRole::kD4Coordination ||
         role == Gfn2PairListRole::kD4TwoBody || role == Gfn2PairListRole::kD4Atm;
}

constexpr double pair_list_role_cutoff(Gfn2PairListRole role) noexcept {
  switch (role) {
    case Gfn2PairListRole::kCoordination:
    case Gfn2PairListRole::kD4Atm:
      return 25.0;
    case Gfn2PairListRole::kD4Coordination:
      return 30.0;
    case Gfn2PairListRole::kD4TwoBody:
      return 50.0;
  }
  return 0.0;
}

}  // namespace

Gfn2PlanSchemaDiagnostic validate_gfn2_pair_list_consumer_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2PairListConsumerView& consumer,
    Gfn2PlanMemorySpace expected_memory_space) noexcept {
  Gfn2PlanSchemaDiagnostic diagnostic =
      validate_gfn2_topology_binding(topology, expected_memory_space);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  if (consumer.memory_space != expected_memory_space) {
    return failure(Gfn2PlanSchemaError::kInvalidMemorySpace,
                   Gfn2PlanSchemaField::kPairListConsumer);
  }
  if (consumer.plan_token == 0u || consumer.plan_token != topology.plan_token) {
    return failure(Gfn2PlanSchemaError::kCrossPlan, Gfn2PlanSchemaField::kPairListConsumer);
  }
  if (!known_pair_list_state(consumer.state)) {
    return failure(Gfn2PlanSchemaError::kInvalidPairListState,
                   Gfn2PlanSchemaField::kPairListConsumer);
  }
  if (consumer.state != Gfn2PairListState::kCommitted) {
    return failure(Gfn2PlanSchemaError::kInvalidPairListState,
                   Gfn2PlanSchemaField::kPairListConsumer);
  }
  if (!known_pair_list_role(consumer.role)) {
    return failure(Gfn2PlanSchemaError::kInvalidPairListRole,
                   Gfn2PlanSchemaField::kPairListConsumer);
  }
  if (consumer.pair_map_kind != Gfn2PairMapKind::kExplicit) {
    return failure(Gfn2PlanSchemaError::kInvalidPairMap, Gfn2PlanSchemaField::kPairListConsumer);
  }
  if (consumer.batch_size != topology.batch_size || consumer.total_atoms != topology.total_atoms) {
    return failure(Gfn2PlanSchemaError::kInvalidCount, Gfn2PlanSchemaField::kPairListConsumer);
  }
  const double required_cutoff = pair_list_role_cutoff(consumer.role);
  if (!(consumer.cutoff_bohr > 0.0) || !std::isfinite(consumer.cutoff_bohr) ||
      !(consumer.list_builder_cutoff_bohr > 0.0) ||
      !std::isfinite(consumer.list_builder_cutoff_bohr) ||
      consumer.cutoff_bohr != required_cutoff ||
      consumer.list_builder_cutoff_bohr < consumer.cutoff_bohr) {
    return failure(Gfn2PlanSchemaError::kInsufficientPairListCutoff,
                   Gfn2PlanSchemaField::kPairListConsumer);
  }
  if (consumer.max_pairs_per_system <= 0 || consumer.max_neighbors_per_atom <= 0) {
    return failure(Gfn2PlanSchemaError::kInvalidCount, Gfn2PlanSchemaField::kPairListConsumer);
  }
  if (consumer.pair_offset_count != consumer.batch_size + 1 ||
      consumer.neighbor_offset_count != consumer.total_atoms + 1) {
    return failure(Gfn2PlanSchemaError::kInvalidCount, Gfn2PlanSchemaField::kPairListOffsets);
  }
  if (consumer.pair_count < 0 || consumer.neighbor_count < 0 ||
      consumer.pair_count_elements != consumer.batch_size ||
      consumer.neighbor_count_elements != consumer.total_atoms) {
    return failure(Gfn2PlanSchemaError::kInvalidCount, Gfn2PlanSchemaField::kPairListPairs);
  }
  std::int64_t pair_capacity = 0;
  std::int64_t neighbor_capacity = 0;
  if (!product(consumer.max_pairs_per_system, consumer.batch_size, pair_capacity) ||
      !product(consumer.max_neighbors_per_atom, consumer.total_atoms, neighbor_capacity)) {
    return failure(Gfn2PlanSchemaError::kCountOverflow, Gfn2PlanSchemaField::kPairListConsumer);
  }
  if (consumer.pair_count > pair_capacity || consumer.neighbor_count > neighbor_capacity) {
    return failure(Gfn2PlanSchemaError::kInvalidCount, Gfn2PlanSchemaField::kPairListPairs);
  }
  if (consumer.committed_generation_count != consumer.batch_size ||
      consumer.eligible_mask_count != consumer.batch_size) {
    return failure(Gfn2PlanSchemaError::kInvalidCount, Gfn2PlanSchemaField::kPairListGenerations);
  }
  if (consumer.active_mask_count != 0 && consumer.active_mask_count != consumer.batch_size) {
    return failure(Gfn2PlanSchemaError::kInvalidActiveMask,
                   Gfn2PlanSchemaField::kPairListActiveMask);
  }
  if ((consumer.active_mask_count == 0) != (consumer.active_mask == nullptr)) {
    return failure(Gfn2PlanSchemaError::kInvalidActiveMask,
                   Gfn2PlanSchemaField::kPairListActiveMask);
  }

  std::array<AddressRange, 6> masks;
  AddressRange generations{};
  AddressRange eligible{};
  AddressRange active{};
  diagnostic = make_range(consumer.pair_offsets, consumer.pair_offset_count,
                          Gfn2PlanSchemaField::kPairListOffsets, masks[0]);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  diagnostic = make_range(consumer.pairs, consumer.pair_count, Gfn2PlanSchemaField::kPairListPairs,
                          masks[1]);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  diagnostic = make_range(consumer.pair_counts, consumer.pair_count_elements,
                          Gfn2PlanSchemaField::kPairListOffsets, masks[4]);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  diagnostic = make_range(consumer.neighbor_counts, consumer.neighbor_count_elements,
                          Gfn2PlanSchemaField::kPairListNeighborOffsets, masks[5]);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  diagnostic = make_range(consumer.neighbor_offsets, consumer.neighbor_offset_count,
                          Gfn2PlanSchemaField::kPairListNeighborOffsets, masks[2]);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  diagnostic = make_range(consumer.neighbors, consumer.neighbor_count,
                          Gfn2PlanSchemaField::kPairListNeighbors, masks[3]);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  diagnostic = make_range(consumer.committed_generations, consumer.committed_generation_count,
                          Gfn2PlanSchemaField::kPairListGenerations, generations);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  diagnostic = make_range(consumer.eligible_mask, consumer.eligible_mask_count,
                          Gfn2PlanSchemaField::kPairListEligibleMask, eligible);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  if (consumer.active_mask_count != 0) {
    diagnostic = make_range(consumer.active_mask, consumer.active_mask_count,
                            Gfn2PlanSchemaField::kPairListActiveMask, active);
    if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
      return diagnostic;
    }
  }
  for (std::size_t first = 0u; first < masks.size(); ++first) {
    for (std::size_t second = first + 1u; second < masks.size(); ++second) {
      if (overlap(masks[first], masks[second])) {
        return failure(Gfn2PlanSchemaError::kAliasedRange, masks[second].field);
      }
    }
    if (overlap(masks[first], generations) || overlap(masks[first], eligible) ||
        overlap(masks[first], active)) {
      return failure(Gfn2PlanSchemaError::kAliasedRange, masks[first].field);
    }
  }
  if (overlap(generations, eligible) || overlap(generations, active) || overlap(eligible, active)) {
    return failure(Gfn2PlanSchemaError::kAliasedRange, Gfn2PlanSchemaField::kPairListGenerations);
  }
  for (const AddressRange& range : masks) {
    diagnostic = validate_no_topology_alias(topology, range);
    if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
      return diagnostic;
    }
  }
  diagnostic = validate_no_topology_alias(topology, generations);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  diagnostic = validate_no_topology_alias(topology, eligible);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  diagnostic = validate_no_topology_alias(topology, active);
  return diagnostic.error != Gfn2PlanSchemaError::kSuccess ? diagnostic : success();
}

Gfn2PlanSchemaDiagnostic validate_gfn2_pair_list_consumer_host(
    const Gfn2RaggedTopologyView& topology, const Gfn2PairListConsumerView& consumer,
    std::uint64_t expected_geometry_generation) noexcept {
  Gfn2PlanSchemaDiagnostic diagnostic =
      validate_gfn2_pair_list_consumer_binding(topology, consumer, Gfn2PlanMemorySpace::kHost);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  if (expected_geometry_generation == 0u) {
    return failure(Gfn2PlanSchemaError::kStaleGeometry, Gfn2PlanSchemaField::kPairListGenerations);
  }
  if (consumer.committed_generations == nullptr || consumer.eligible_mask == nullptr) {
    return failure(Gfn2PlanSchemaError::kNullPointer, Gfn2PlanSchemaField::kPairListGenerations);
  }
  for (std::int64_t system = 0; system < consumer.batch_size; ++system) {
    if (consumer.eligible_mask[system] > 1u) {
      return failure(Gfn2PlanSchemaError::kInvalidActiveMask,
                     Gfn2PlanSchemaField::kPairListEligibleMask, system);
    }
    if (consumer.active_mask != nullptr && consumer.active_mask[system] > 1u) {
      return failure(Gfn2PlanSchemaError::kInvalidActiveMask,
                     Gfn2PlanSchemaField::kPairListActiveMask, system);
    }
  }
  if (consumer.pair_offsets[0] != 0 ||
      consumer.pair_offsets[consumer.batch_size] > consumer.pair_count) {
    return failure(Gfn2PlanSchemaError::kInvalidOffsets, Gfn2PlanSchemaField::kPairListOffsets, 0);
  }
  if (consumer.neighbor_offsets[0] != 0 ||
      consumer.neighbor_offsets[consumer.total_atoms] > consumer.neighbor_count) {
    return failure(Gfn2PlanSchemaError::kInvalidOffsets,
                   Gfn2PlanSchemaField::kPairListNeighborOffsets, 0);
  }
  std::int64_t published_pair_count = 0;
  for (std::int64_t system = 0; system < consumer.batch_size; ++system) {
    const std::int64_t atom_begin = topology.atom_offsets[system];
    const std::int64_t atom_end = topology.atom_offsets[system + 1];
    const std::int64_t pair_begin = consumer.pair_offsets[system];
    const std::int64_t next_pair_begin = consumer.pair_offsets[system + 1];
    const std::int64_t count = consumer.pair_counts[system];
    if (pair_begin < 0 || next_pair_begin < pair_begin || next_pair_begin > consumer.pair_count ||
        count < 0 || count > consumer.max_pairs_per_system ||
        count > next_pair_begin - pair_begin) {
      return failure(Gfn2PlanSchemaError::kInvalidOffsets, Gfn2PlanSchemaField::kPairListOffsets,
                     system);
    }
    if (count > consumer.pair_count - published_pair_count) {
      return failure(Gfn2PlanSchemaError::kCountOverflow, Gfn2PlanSchemaField::kPairListPairs,
                     system);
    }
    published_pair_count += count;
    const std::int64_t pair_end = pair_begin + count;
    Gfn2AtomPair previous{};
    bool have_previous = false;
    for (std::int64_t pair = pair_begin; pair < pair_end; ++pair) {
      const Gfn2AtomPair atom_pair = consumer.pairs[pair];
      if (atom_pair.first < atom_begin || atom_pair.second > atom_end - 1 ||
          atom_pair.first >= atom_pair.second) {
        return failure(Gfn2PlanSchemaError::kInvalidPairMap, Gfn2PlanSchemaField::kPairListPairs,
                       pair);
      }
      if (have_previous &&
          (atom_pair.second < previous.second ||
           (atom_pair.second == previous.second && atom_pair.first <= previous.first))) {
        return failure(Gfn2PlanSchemaError::kInvalidPairMap, Gfn2PlanSchemaField::kPairListPairs,
                       pair);
      }
      previous = atom_pair;
      have_previous = true;
    }
    if (consumer.eligible_mask[system] == 1u &&
        consumer.committed_generations[system] != expected_geometry_generation) {
      return failure(Gfn2PlanSchemaError::kStaleGeometry, Gfn2PlanSchemaField::kPairListGenerations,
                     system);
    }
  }
  std::int64_t published_neighbor_count = 0;
  for (std::int64_t atom = 0; atom < consumer.total_atoms; ++atom) {
    const std::int64_t begin = consumer.neighbor_offsets[atom];
    const std::int64_t next_begin = consumer.neighbor_offsets[atom + 1];
    const std::int64_t count = consumer.neighbor_counts[atom];
    if (begin < 0 || next_begin < begin || next_begin > consumer.neighbor_count || count < 0 ||
        count > consumer.max_neighbors_per_atom || count > next_begin - begin) {
      return failure(Gfn2PlanSchemaError::kInvalidOffsets,
                     Gfn2PlanSchemaField::kPairListNeighborOffsets, atom);
    }
    if (count > consumer.neighbor_count - published_neighbor_count) {
      return failure(Gfn2PlanSchemaError::kCountOverflow, Gfn2PlanSchemaField::kPairListNeighbors,
                     atom);
    }
    published_neighbor_count += count;
    const std::int64_t end = begin + count;
    std::int64_t system = 0;
    while (system + 1 < consumer.batch_size && atom >= topology.atom_offsets[system + 1]) {
      ++system;
    }
    const std::int64_t atom_begin = topology.atom_offsets[system];
    const std::int64_t atom_end = topology.atom_offsets[system + 1];
    std::int64_t previous = -1;
    for (std::int64_t index = begin; index < end; ++index) {
      const std::int64_t peer = consumer.neighbors[index];
      if (peer < atom_begin || peer >= atom_end || peer == atom || peer <= previous) {
        return failure(Gfn2PlanSchemaError::kInvalidPairMap,
                       Gfn2PlanSchemaField::kPairListNeighbors, index);
      }
      previous = peer;
    }
  }
  std::int64_t doubled_pairs = 0;
  if (!product(published_pair_count, 2, doubled_pairs) ||
      doubled_pairs != published_neighbor_count) {
    return failure(Gfn2PlanSchemaError::kInvalidPairMap, Gfn2PlanSchemaField::kPairListPairs);
  }
  for (std::int64_t system = 0; system < consumer.batch_size; ++system) {
    const std::int64_t pair_begin = consumer.pair_offsets[system];
    const std::int64_t pair_end = pair_begin + consumer.pair_counts[system];
    for (std::int64_t pair = pair_begin; pair < pair_end; ++pair) {
      const Gfn2AtomPair atom_pair = consumer.pairs[pair];
      const std::int64_t first_begin = consumer.neighbor_offsets[atom_pair.first];
      const std::int64_t first_end = first_begin + consumer.neighbor_counts[atom_pair.first];
      const std::int64_t second_begin = consumer.neighbor_offsets[atom_pair.second];
      const std::int64_t second_end = second_begin + consumer.neighbor_counts[atom_pair.second];
      if (!std::binary_search(consumer.neighbors + first_begin, consumer.neighbors + first_end,
                              atom_pair.second) ||
          !std::binary_search(consumer.neighbors + second_begin, consumer.neighbors + second_end,
                              atom_pair.first)) {
        return failure(Gfn2PlanSchemaError::kInvalidPairMap,
                       Gfn2PlanSchemaField::kPairListNeighbors, pair);
      }
    }
  }
  return success();
}

Gfn2PlanSchemaDiagnostic project_gfn2_pair_list_role_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2PairListConsumerView& source,
    Gfn2PairListRole target_role, Gfn2PlanMemorySpace expected_memory_space,
    Gfn2PairListConsumerView& projection) noexcept {
  projection = {};
  Gfn2PlanSchemaDiagnostic diagnostic =
      validate_gfn2_pair_list_consumer_binding(topology, source, expected_memory_space);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  if (!known_pair_list_role(target_role)) {
    return failure(Gfn2PlanSchemaError::kInvalidPairListRole,
                   Gfn2PlanSchemaField::kPairListConsumer);
  }
  const double target_cutoff = pair_list_role_cutoff(target_role);
  if (source.list_builder_cutoff_bohr < target_cutoff) {
    return failure(Gfn2PlanSchemaError::kInsufficientPairListCutoff,
                   Gfn2PlanSchemaField::kPairListConsumer);
  }

  /* The committed storage and publication metadata form one lease.  Preserve
   * it byte-for-byte at the field level and specialize only the consumer's
   * physical predicate. */
  projection = source;
  projection.role = target_role;
  projection.cutoff_bohr = target_cutoff;
  return success();
}


std::uint64_t gfn2_element_identity_fingerprint_host(
    const Gfn2ElementIdentityProjectionView& element) noexcept {
  return shared_topology::element_identity_fingerprint_host(shared_view(element));
}

Gfn2PlanSchemaDiagnostic validate_gfn2_atom_projection_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2AtomProjectionView& projection,
    Gfn2PlanMemorySpace expected_memory_space) noexcept {
  return native_diagnostic(shared_topology::validate_atom_projection_binding(
      shared_view(topology), shared_view(projection), static_cast<shared_topology::MemorySpace>(expected_memory_space)));
}

Gfn2PlanSchemaDiagnostic validate_gfn2_shell_ownership_projection_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2ShellOwnershipProjectionView& projection,
    Gfn2PlanMemorySpace expected_memory_space) noexcept {
  return native_diagnostic(shared_topology::validate_shell_ownership_projection_binding(
      shared_view(topology), shared_view(projection), static_cast<shared_topology::MemorySpace>(expected_memory_space)));
}

Gfn2PlanSchemaDiagnostic validate_gfn2_ao_matrix_projection_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2AOMatrixProjectionView& projection,
    Gfn2PlanMemorySpace expected_memory_space) noexcept {
  return native_diagnostic(shared_topology::validate_ao_matrix_projection_binding(
      shared_view(topology), shared_view(projection), static_cast<shared_topology::MemorySpace>(expected_memory_space)));
}

Gfn2PlanSchemaDiagnostic validate_gfn2_packed_all_pair_projection_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2PackedAllPairProjectionView& projection,
    Gfn2PlanMemorySpace expected_memory_space) noexcept {
  return native_diagnostic(shared_topology::validate_packed_all_pair_projection_binding(
      shared_view(topology), shared_view(projection), static_cast<shared_topology::MemorySpace>(expected_memory_space)));
}

Gfn2PlanSchemaDiagnostic validate_gfn2_ao_bucket_projection_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2AOBucketProjectionView& projection,
    Gfn2PlanMemorySpace expected_memory_space) noexcept {
  return native_diagnostic(shared_topology::validate_ao_bucket_projection_binding(
      shared_view(topology), shared_view(projection), static_cast<shared_topology::MemorySpace>(expected_memory_space)));
}

Gfn2PlanSchemaDiagnostic validate_gfn2_element_identity_projection_binding(
    const Gfn2ElementIdentityProjectionView& projection,
    Gfn2PlanMemorySpace expected_memory_space) noexcept {
  return native_diagnostic(shared_topology::validate_element_identity_projection_binding(
      shared_view(projection), static_cast<shared_topology::MemorySpace>(expected_memory_space)));
}

Gfn2PlanSchemaDiagnostic project_gfn2_atom_projection_host(
    const Gfn2RaggedTopologyView& topology, Gfn2AtomProjectionView& projection) noexcept {
  projection = {};
  shared_topology::AtomProjectionView result{};
  const auto diagnostic = shared_topology::project_atom_projection_host(shared_view(topology), result);
  native_projection(result, projection);
  return native_diagnostic(diagnostic);
}

Gfn2PlanSchemaDiagnostic project_gfn2_shell_ownership_projection_host(
    const Gfn2RaggedTopologyView& topology, Gfn2ShellOwnershipProjectionView& projection) noexcept {
  projection = {};
  shared_topology::ShellOwnershipProjectionView result{};
  const auto diagnostic = shared_topology::project_shell_ownership_projection_host(shared_view(topology), result);
  native_projection(result, projection);
  return native_diagnostic(diagnostic);
}

Gfn2PlanSchemaDiagnostic project_gfn2_ao_matrix_projection_host(
    const Gfn2RaggedTopologyView& topology, Gfn2AOMatrixProjectionView& projection) noexcept {
  projection = {};
  shared_topology::AOMatrixProjectionView result{};
  const auto diagnostic = shared_topology::project_ao_matrix_projection_host(shared_view(topology), result);
  native_projection(result, projection);
  return native_diagnostic(diagnostic);
}

Gfn2PlanSchemaDiagnostic project_gfn2_packed_all_pair_projection_host(
    const Gfn2RaggedTopologyView& topology, Gfn2PackedAllPairProjectionView& projection) noexcept {
  projection = {};
  shared_topology::PackedAllPairProjectionView result{};
  const auto diagnostic = shared_topology::project_packed_all_pair_projection_host(shared_view(topology), result);
  native_projection(result, projection);
  return native_diagnostic(diagnostic);
}

Gfn2PlanSchemaDiagnostic project_gfn2_ao_bucket_projection_host(
    const Gfn2RaggedTopologyView& topology, Gfn2AOBucketProjectionView& projection) noexcept {
  projection = {};
  shared_topology::AOBucketProjectionView result{};
  const auto diagnostic = shared_topology::project_ao_bucket_projection_host(shared_view(topology), result);
  native_projection(result, projection);
  return native_diagnostic(diagnostic);
}

Gfn2PlanSchemaDiagnostic project_gfn2_element_identity_projection_host(
    const std::int32_t* atomic_numbers, std::int64_t atomic_number_count, std::uint64_t plan_token,
    Gfn2ElementIdentityProjectionView& projection) noexcept {
  projection = {};
  shared_topology::ElementIdentityProjectionView result{};
  const auto diagnostic = shared_topology::project_element_identity_projection_host(atomic_numbers, atomic_number_count, plan_token, result);
  native_projection(result, projection);
  return native_diagnostic(diagnostic);
}

}  // namespace generativeqc::xtb::detail
