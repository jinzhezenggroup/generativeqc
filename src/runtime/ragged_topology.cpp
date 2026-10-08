// See ragged_topology.hpp for ownership and upstream licensing.
#include "ragged_topology.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include "bounded_workspace.hpp"

namespace generativeqc::runtime::ragged {
namespace {

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
  Field field = Field::kNone;
  bool active = false;
};

constexpr Diagnostic success() noexcept { return {}; }

constexpr Diagnostic failure(Error error, Field field, std::int64_t index = -1) noexcept {
  return {error, field, index};
}

constexpr bool known_memory_space(MemorySpace memory_space) noexcept {
  return memory_space == MemorySpace::kHost || memory_space == MemorySpace::kCudaDevice ||
         memory_space == MemorySpace::kHipDevice;
}

template <typename T>
Diagnostic make_range(const T* pointer, std::int64_t count, Field field,
                      AddressRange& range) noexcept {
  if (count < 0) {
    return failure(Error::kInvalidCount, field);
  }
  if (count == 0) {
    if (pointer != nullptr) {
      return failure(Error::kInvalidCount, field);
    }
    range = {};
    return success();
  }
  if (pointer == nullptr) {
    return failure(Error::kNullPointer, field);
  }
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
  if (address % alignof(T) != 0u) {
    return failure(Error::kMisalignedPointer, field);
  }
  constexpr std::uintmax_t kMaximumSize = std::numeric_limits<std::size_t>::max();
  if (static_cast<std::uintmax_t>(count) > kMaximumSize / sizeof(T)) {
    return failure(Error::kCountOverflow, field);
  }
  std::size_t bytes = 0;
  if (!checked_multiply(static_cast<std::size_t>(count), sizeof(T), bytes)) {
    return failure(Error::kCountOverflow, field);
  }
  if (address > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return failure(Error::kAddressOverflow, field);
  }
  range = {address, address + bytes, field, true};
  return success();
}

bool overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return first.active && second.active && first.begin < second.end && second.begin < first.end;
}

template <std::size_t N>
Diagnostic validate_aliases(const std::array<AddressRange, N>& ranges) noexcept {
  for (std::size_t first = 0u; first < ranges.size(); ++first) {
    for (std::size_t second = first + 1u; second < ranges.size(); ++second) {
      if (overlap(ranges[first], ranges[second])) {
        return failure(Error::kAliasedRange, ranges[second].field);
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

bool square(std::int64_t value, std::int64_t& result) noexcept {
  if (value < 0 || (value != 0 && value > std::numeric_limits<std::int64_t>::max() / value)) {
    return false;
  }
  result = value * value;
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

bool triangle(std::int64_t value, std::int64_t& result) noexcept {
  if (value < 0 || (value > 1 && value > std::numeric_limits<std::int64_t>::max() / (value - 1))) {
    return false;
  }
  result = (value & 1LL) == 0LL ? (value / 2LL) * (value - 1LL) : value * ((value - 1LL) / 2LL);
  return true;
}

Diagnostic validate_offsets(const std::int64_t* offsets, std::int64_t partitions,
                            std::int64_t endpoint, Field field) noexcept {
  if (offsets[0] != 0) {
    return failure(Error::kInvalidOffsets, field, 0);
  }
  for (std::int64_t index = 0; index < partitions; ++index) {
    if (offsets[index] < 0 || offsets[index] > offsets[index + 1] ||
        offsets[index + 1] > endpoint) {
      return failure(Error::kInvalidOffsets, field, index);
    }
  }
  if (offsets[partitions] != endpoint) {
    return failure(Error::kInvalidOffsets, field, partitions);
  }
  return success();
}

template <typename T>
Diagnostic add_range(std::array<AddressRange, 14>& ranges, std::size_t index, const T* pointer,
                     std::int64_t count, Field field) noexcept {
  return make_range(pointer, count, field, ranges[index]);
}

}  // namespace

Diagnostic validate_topology_binding(const TopologyView& topology,
                                     MemorySpace expected_memory_space) noexcept {
  if (!known_memory_space(expected_memory_space) ||
      topology.memory_space != expected_memory_space) {
    return failure(Error::kInvalidMemorySpace, Field::kTopology);
  }
  if (topology.plan_token == 0u) {
    return failure(Error::kInvalidPlanToken, Field::kTopology);
  }
  if (topology.batch_size <= 0 || topology.total_atoms < 0 || topology.total_shells < 0 ||
      topology.total_orbitals < 0 || topology.total_matrix_elements < 0 ||
      topology.total_pairs < 0 || topology.bucket_count < 0) {
    return failure(Error::kInvalidCount, Field::kTopology);
  }

  std::int64_t batch_offsets = 0;
  std::int64_t atom_offsets = 0;
  std::int64_t shell_offsets = 0;
  if (!add_one(topology.batch_size, batch_offsets) ||
      !add_one(topology.total_atoms, atom_offsets) ||
      !add_one(topology.total_shells, shell_offsets)) {
    return failure(Error::kCountOverflow, Field::kTopology);
  }
  if (topology.atom_offset_count != batch_offsets ||
      topology.batch_shell_offset_count != batch_offsets ||
      topology.batch_orbital_offset_count != batch_offsets ||
      topology.matrix_offset_count != batch_offsets ||
      topology.atom_shell_offset_count != atom_offsets ||
      topology.shell_orbital_offset_count != shell_offsets ||
      topology.shell_to_atom_count != topology.total_shells ||
      topology.orbital_to_shell_count != topology.total_orbitals ||
      topology.orbital_to_atom_count != topology.total_orbitals) {
    return failure(Error::kInvalidCount, Field::kTopology);
  }

  if (topology.pair_map_kind == PairMapKind::kNone) {
    if (topology.total_pairs != 0 || topology.pair_offset_count != 0 ||
        topology.atom_pair_count != 0 || topology.pair_offsets != nullptr ||
        topology.atom_pairs != nullptr) {
      return failure(Error::kInvalidPairMap, Field::kPairOffsets);
    }
  } else if (topology.pair_map_kind == PairMapKind::kPackedLowerTriangle) {
    if (topology.pair_offset_count != batch_offsets || topology.atom_pair_count != 0 ||
        topology.atom_pairs != nullptr) {
      return failure(Error::kInvalidPairMap, Field::kPairOffsets);
    }
  } else if (topology.pair_map_kind == PairMapKind::kExplicit) {
    if (topology.pair_offset_count != batch_offsets ||
        topology.atom_pair_count != topology.total_pairs) {
      return failure(Error::kInvalidPairMap, Field::kAtomPairs);
    }
  } else {
    return failure(Error::kInvalidPairMap, Field::kTopology);
  }

  std::int64_t bucket_offsets = 0;
  if (topology.bucket_count == 0) {
    if (topology.bucket_offset_count != 0 || topology.bucket_system_count != 0 ||
        topology.bucket_orbital_count != 0 || topology.bucket_offsets != nullptr ||
        topology.bucket_systems != nullptr || topology.bucket_orbital_counts != nullptr) {
      return failure(Error::kInvalidBucketMap, Field::kBucketOffsets);
    }
  } else {
    if (!add_one(topology.bucket_count, bucket_offsets)) {
      return failure(Error::kCountOverflow, Field::kBucketOffsets);
    }
    if (topology.bucket_count > topology.batch_size ||
        topology.batch_size > std::numeric_limits<std::int32_t>::max() ||
        topology.bucket_offset_count != bucket_offsets ||
        topology.bucket_system_count != topology.batch_size ||
        topology.bucket_orbital_count != topology.bucket_count) {
      return failure(Error::kInvalidBucketMap, Field::kBucketOffsets);
    }
  }

  std::array<AddressRange, 14> ranges{};
  Diagnostic diagnostic =
      add_range(ranges, 0u, topology.atom_offsets, topology.atom_offset_count, Field::kAtomOffsets);
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 1u, topology.batch_shell_offsets,
                           topology.batch_shell_offset_count, Field::kBatchShellOffsets);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 2u, topology.batch_orbital_offsets,
                           topology.batch_orbital_offset_count, Field::kBatchOrbitalOffsets);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 3u, topology.matrix_offsets, topology.matrix_offset_count,
                           Field::kMatrixOffsets);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 4u, topology.atom_shell_offsets,
                           topology.atom_shell_offset_count, Field::kAtomShellOffsets);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 5u, topology.shell_orbital_offsets,
                           topology.shell_orbital_offset_count, Field::kShellOrbitalOffsets);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 6u, topology.shell_to_atom, topology.shell_to_atom_count,
                           Field::kShellToAtom);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 7u, topology.orbital_to_shell, topology.orbital_to_shell_count,
                           Field::kOrbitalToShell);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 8u, topology.orbital_to_atom, topology.orbital_to_atom_count,
                           Field::kOrbitalToAtom);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 9u, topology.pair_offsets, topology.pair_offset_count,
                           Field::kPairOffsets);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 10u, static_cast<const AtomPairLayout*>(topology.atom_pairs),
                           topology.atom_pair_count, Field::kAtomPairs);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 11u, topology.bucket_offsets, topology.bucket_offset_count,
                           Field::kBucketOffsets);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 12u, topology.bucket_systems, topology.bucket_system_count,
                           Field::kBucketSystems);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = add_range(ranges, 13u, topology.bucket_orbital_counts,
                           topology.bucket_orbital_count, Field::kBucketOrbitalCounts);
  }
  return diagnostic.error == Error::kSuccess ? validate_aliases(ranges) : diagnostic;
}

Diagnostic validate_topology_host(const TopologyView& topology) noexcept {
  Diagnostic diagnostic = validate_topology_binding(topology, MemorySpace::kHost);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }

  diagnostic = validate_offsets(topology.atom_offsets, topology.batch_size, topology.total_atoms,
                                Field::kAtomOffsets);
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = validate_offsets(topology.batch_shell_offsets, topology.batch_size,
                                  topology.total_shells, Field::kBatchShellOffsets);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = validate_offsets(topology.batch_orbital_offsets, topology.batch_size,
                                  topology.total_orbitals, Field::kBatchOrbitalOffsets);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = validate_offsets(topology.matrix_offsets, topology.batch_size,
                                  topology.total_matrix_elements, Field::kMatrixOffsets);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = validate_offsets(topology.atom_shell_offsets, topology.total_atoms,
                                  topology.total_shells, Field::kAtomShellOffsets);
  }
  if (diagnostic.error == Error::kSuccess) {
    diagnostic = validate_offsets(topology.shell_orbital_offsets, topology.total_shells,
                                  topology.total_orbitals, Field::kShellOrbitalOffsets);
  }
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }

  for (std::int64_t system = 0; system < topology.batch_size; ++system) {
    const std::int64_t atom_begin = topology.atom_offsets[system];
    const std::int64_t atom_end = topology.atom_offsets[system + 1];
    const std::int64_t shell_begin = topology.batch_shell_offsets[system];
    const std::int64_t shell_end = topology.batch_shell_offsets[system + 1];
    const std::int64_t orbital_begin = topology.batch_orbital_offsets[system];
    const std::int64_t orbital_end = topology.batch_orbital_offsets[system + 1];
    const std::int64_t orbital_count = orbital_end - orbital_begin;
    std::int64_t matrix_count = 0;
    if (!square(orbital_count, matrix_count)) {
      return failure(Error::kCountOverflow, Field::kMatrixOffsets, system);
    }
    if (topology.matrix_offsets[system + 1] - topology.matrix_offsets[system] != matrix_count) {
      return failure(Error::kInvalidMatrixExtent, Field::kMatrixOffsets, system);
    }
    if (topology.atom_shell_offsets[atom_begin] != shell_begin ||
        topology.atom_shell_offsets[atom_end] != shell_end ||
        topology.shell_orbital_offsets[shell_begin] != orbital_begin ||
        topology.shell_orbital_offsets[shell_end] != orbital_end) {
      return failure(Error::kInvalidOffsets, Field::kAtomShellOffsets, system);
    }
    if ((atom_end == atom_begin) != (shell_end == shell_begin) ||
        (shell_end == shell_begin) != (orbital_end == orbital_begin)) {
      return failure(Error::kInvalidOffsets, Field::kBatchShellOffsets, system);
    }
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::int64_t owned_shell_begin = topology.atom_shell_offsets[atom];
      const std::int64_t owned_shell_end = topology.atom_shell_offsets[atom + 1];
      if (owned_shell_begin >= owned_shell_end) {
        return failure(Error::kInvalidShellMap, Field::kAtomShellOffsets, atom);
      }
      for (std::int64_t shell = owned_shell_begin; shell < owned_shell_end; ++shell) {
        if (topology.shell_to_atom[shell] != atom) {
          return failure(Error::kInvalidShellMap, Field::kShellToAtom, shell);
        }
      }
    }
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::int64_t owned_orbital_begin = topology.shell_orbital_offsets[shell];
      const std::int64_t owned_orbital_end = topology.shell_orbital_offsets[shell + 1];
      if (owned_orbital_begin >= owned_orbital_end) {
        return failure(Error::kInvalidOrbitalMap, Field::kShellOrbitalOffsets, shell);
      }
      const std::int64_t atom = topology.shell_to_atom[shell];
      if (atom < atom_begin || atom >= atom_end) {
        return failure(Error::kInvalidShellMap, Field::kShellToAtom, shell);
      }
      for (std::int64_t orbital = owned_orbital_begin; orbital < owned_orbital_end; ++orbital) {
        if (topology.orbital_to_shell[orbital] != shell) {
          return failure(Error::kInvalidOrbitalMap, Field::kOrbitalToShell, orbital);
        }
        if (topology.orbital_to_atom[orbital] != atom) {
          return failure(Error::kInvalidOrbitalMap, Field::kOrbitalToAtom, orbital);
        }
      }
    }
  }

  if (topology.pair_map_kind != PairMapKind::kNone) {
    diagnostic = validate_offsets(topology.pair_offsets, topology.batch_size, topology.total_pairs,
                                  Field::kPairOffsets);
    if (diagnostic.error != Error::kSuccess) {
      return diagnostic;
    }
    for (std::int64_t system = 0; system < topology.batch_size; ++system) {
      const std::int64_t pair_begin = topology.pair_offsets[system];
      const std::int64_t pair_end = topology.pair_offsets[system + 1];
      const std::int64_t atom_begin = topology.atom_offsets[system];
      const std::int64_t atom_end = topology.atom_offsets[system + 1];
      if (topology.pair_map_kind == PairMapKind::kPackedLowerTriangle) {
        std::int64_t expected_pairs = 0;
        if (!triangle(atom_end - atom_begin, expected_pairs)) {
          return failure(Error::kCountOverflow, Field::kPairOffsets, system);
        }
        if (pair_end - pair_begin != expected_pairs) {
          return failure(Error::kInvalidPairMap, Field::kPairOffsets, system);
        }
      } else {
        AtomPairLayout previous{-1, -1};
        for (std::int64_t pair = pair_begin; pair < pair_end; ++pair) {
          AtomPairLayout current{};
          std::memcpy(&current,
                      static_cast<const std::byte*>(topology.atom_pairs) +
                          static_cast<std::size_t>(pair) * sizeof(current),
                      sizeof(current));
          const bool in_system = current.first >= atom_begin && current.first < current.second &&
                                 current.second < atom_end;
          const bool sorted = pair == pair_begin || previous.first < current.first ||
                              (previous.first == current.first && previous.second < current.second);
          if (!in_system || !sorted) {
            return failure(Error::kInvalidPairMap, Field::kAtomPairs, pair);
          }
          previous = current;
        }
      }
    }
  }

  if (topology.bucket_count != 0) {
    diagnostic = validate_offsets(topology.bucket_offsets, topology.bucket_count,
                                  topology.batch_size, Field::kBucketOffsets);
    if (diagnostic.error != Error::kSuccess) {
      return diagnostic;
    }
    for (std::int64_t bucket = 0; bucket < topology.bucket_count; ++bucket) {
      if (topology.bucket_offsets[bucket] == topology.bucket_offsets[bucket + 1]) {
        return failure(Error::kInvalidBucketMap, Field::kBucketOffsets, bucket);
      }
      if (topology.bucket_orbital_counts[bucket] < 0) {
        return failure(Error::kInvalidBucketMap, Field::kBucketOrbitalCounts, bucket);
      }
      for (std::int64_t position = topology.bucket_offsets[bucket];
           position < topology.bucket_offsets[bucket + 1]; ++position) {
        const std::int64_t system = topology.bucket_systems[position];
        if (system < 0 || system >= topology.batch_size) {
          return failure(Error::kInvalidBucketMap, Field::kBucketSystems, position);
        }
        const std::int64_t orbitals =
            topology.batch_orbital_offsets[system + 1] - topology.batch_orbital_offsets[system];
        if (orbitals != topology.bucket_orbital_counts[bucket]) {
          return failure(Error::kInvalidBucketMap, Field::kBucketOrbitalCounts, bucket);
        }
        for (std::int64_t earlier = 0; earlier < position; ++earlier) {
          if (topology.bucket_systems[earlier] == system) {
            return failure(Error::kInvalidBucketMap, Field::kBucketSystems, position);
          }
        }
      }
    }
  }
  return success();
}

Diagnostic bind_topology_host(const TopologyView& candidate, TopologyView& binding) noexcept {
  binding = {};
  const Diagnostic diagnostic = validate_topology_host(candidate);
  if (diagnostic.error == Error::kSuccess) {
    binding = candidate;
  }
  return diagnostic;
}

namespace {

constexpr Diagnostic projection_failure(Error error, Field field,
                                        std::int64_t index = -1) noexcept {
  return failure(error, field, index);
}

Diagnostic validate_projection_memory_and_token(MemorySpace memory_space, std::uint64_t plan_token,
                                                std::uint64_t master_token,
                                                MemorySpace expected_memory_space,
                                                Field field) noexcept {
  if (!known_memory_space(expected_memory_space) || memory_space != expected_memory_space) {
    return projection_failure(Error::kInvalidMemorySpace, field);
  }
  if (plan_token == 0u || plan_token != master_token) {
    return projection_failure(Error::kCrossPlan, field);
  }
  return success();
}

template <typename T>
bool exact_pointer(const T* ours, const T* master, std::int64_t count,
                   std::int64_t master_count) noexcept {
  return ours == master && count == master_count;
}

}  // namespace

std::uint64_t element_identity_fingerprint_host(
    const ElementIdentityProjectionView& element) noexcept {
  if (element.memory_space != MemorySpace::kHost || element.plan_token == 0u ||
      element.total_atoms < 0 || element.atomic_number_count != element.total_atoms ||
      (element.total_atoms != 0 && element.atomic_numbers == nullptr)) {
    return 0u;
  }
  std::uint64_t hash = 0x6a09e667f3bcc909ULL;
  hash_append(2u, hash);  // Fingerprint schema version.
  hash_append(element.plan_token, hash);
  hash_append(static_cast<std::uint64_t>(element.total_atoms), hash);
  for (std::int64_t atom = 0; atom < element.total_atoms; ++atom) {
    hash_append(static_cast<std::uint32_t>(element.atomic_numbers[atom]), hash);
  }
  return hash == 0u ? 1u : hash;
}

Diagnostic validate_atom_projection_binding(const TopologyView& topology,
                                            const AtomProjectionView& projection,
                                            MemorySpace expected_memory_space) noexcept {
  Diagnostic diagnostic = validate_topology_binding(topology, expected_memory_space);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  diagnostic = validate_projection_memory_and_token(projection.memory_space, projection.plan_token,
                                                    topology.plan_token, expected_memory_space,
                                                    Field::kProjection);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  std::int64_t atom_offsets = 0;
  if (projection.batch_size != topology.batch_size ||
      projection.total_atoms != topology.total_atoms ||
      !add_one(topology.batch_size, atom_offsets) || projection.atom_offset_count != atom_offsets ||
      !exact_pointer(projection.atom_offsets, topology.atom_offsets, atom_offsets,
                     topology.atom_offset_count)) {
    return projection_failure(Error::kInvalidProjection, Field::kProjection);
  }
  return success();
}

Diagnostic validate_shell_ownership_projection_binding(
    const TopologyView& topology, const ShellOwnershipProjectionView& projection,
    MemorySpace expected_memory_space) noexcept {
  Diagnostic diagnostic = validate_topology_binding(topology, expected_memory_space);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  diagnostic = validate_projection_memory_and_token(projection.memory_space, projection.plan_token,
                                                    topology.plan_token, expected_memory_space,
                                                    Field::kProjection);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  std::int64_t batch_offsets = 0;
  std::int64_t atom_offsets = 0;
  if (projection.batch_size != topology.batch_size ||
      projection.total_atoms != topology.total_atoms ||
      projection.total_shells != topology.total_shells ||
      !add_one(topology.batch_size, batch_offsets) ||
      !add_one(topology.total_atoms, atom_offsets) ||
      projection.batch_shell_offset_count != batch_offsets ||
      projection.atom_shell_offset_count != atom_offsets ||
      projection.shell_to_atom_count != topology.total_shells ||
      !exact_pointer(projection.batch_shell_offsets, topology.batch_shell_offsets, batch_offsets,
                     topology.batch_shell_offset_count) ||
      !exact_pointer(projection.atom_shell_offsets, topology.atom_shell_offsets, atom_offsets,
                     topology.atom_shell_offset_count) ||
      !exact_pointer(projection.shell_to_atom, topology.shell_to_atom, topology.total_shells,
                     topology.shell_to_atom_count)) {
    return projection_failure(Error::kInvalidProjection, Field::kProjection);
  }
  return success();
}

Diagnostic validate_ao_matrix_projection_binding(const TopologyView& topology,
                                                 const AOMatrixProjectionView& projection,
                                                 MemorySpace expected_memory_space) noexcept {
  Diagnostic diagnostic = validate_topology_binding(topology, expected_memory_space);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  diagnostic = validate_projection_memory_and_token(projection.memory_space, projection.plan_token,
                                                    topology.plan_token, expected_memory_space,
                                                    Field::kProjection);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  std::int64_t batch_offsets = 0;
  std::int64_t shell_offsets = 0;
  if (projection.batch_size != topology.batch_size ||
      projection.total_shells != topology.total_shells ||
      projection.total_orbitals != topology.total_orbitals ||
      projection.total_matrix_elements != topology.total_matrix_elements ||
      !add_one(topology.batch_size, batch_offsets) ||
      !add_one(topology.total_shells, shell_offsets) ||
      projection.batch_orbital_offset_count != batch_offsets ||
      projection.matrix_offset_count != batch_offsets ||
      projection.shell_orbital_offset_count != shell_offsets ||
      projection.orbital_to_shell_count != topology.total_orbitals ||
      projection.orbital_to_atom_count != topology.total_orbitals ||
      !exact_pointer(projection.batch_orbital_offsets, topology.batch_orbital_offsets,
                     batch_offsets, topology.batch_orbital_offset_count) ||
      !exact_pointer(projection.matrix_offsets, topology.matrix_offsets, batch_offsets,
                     topology.matrix_offset_count) ||
      !exact_pointer(projection.shell_orbital_offsets, topology.shell_orbital_offsets,
                     shell_offsets, topology.shell_orbital_offset_count) ||
      !exact_pointer(projection.orbital_to_shell, topology.orbital_to_shell,
                     topology.total_orbitals, topology.orbital_to_shell_count) ||
      !exact_pointer(projection.orbital_to_atom, topology.orbital_to_atom, topology.total_orbitals,
                     topology.orbital_to_atom_count)) {
    return projection_failure(Error::kInvalidProjection, Field::kProjection);
  }
  return success();
}

Diagnostic validate_packed_all_pair_projection_binding(
    const TopologyView& topology, const PackedAllPairProjectionView& projection,
    MemorySpace expected_memory_space) noexcept {
  Diagnostic diagnostic = validate_topology_binding(topology, expected_memory_space);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  diagnostic = validate_projection_memory_and_token(projection.memory_space, projection.plan_token,
                                                    topology.plan_token, expected_memory_space,
                                                    Field::kProjection);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  if (topology.pair_map_kind != PairMapKind::kPackedLowerTriangle ||
      projection.batch_size != topology.batch_size ||
      projection.total_pairs != topology.total_pairs ||
      projection.pair_offset_count != topology.pair_offset_count ||
      !exact_pointer(projection.pair_offsets, topology.pair_offsets, topology.pair_offset_count,
                     topology.pair_offset_count)) {
    return projection_failure(Error::kInvalidProjection, Field::kProjection);
  }
  return success();
}

Diagnostic validate_ao_bucket_projection_binding(const TopologyView& topology,
                                                 const AOBucketProjectionView& projection,
                                                 MemorySpace expected_memory_space) noexcept {
  Diagnostic diagnostic = validate_topology_binding(topology, expected_memory_space);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  diagnostic = validate_projection_memory_and_token(projection.memory_space, projection.plan_token,
                                                    topology.plan_token, expected_memory_space,
                                                    Field::kProjection);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  std::int64_t bucket_offsets = 0;
  if (projection.batch_size != topology.batch_size ||
      projection.bucket_count != topology.bucket_count ||
      (topology.bucket_count != 0 && !add_one(topology.bucket_count, bucket_offsets)) ||
      projection.bucket_offset_count != bucket_offsets ||
      projection.bucket_system_count != topology.bucket_system_count ||
      projection.bucket_orbital_count != topology.bucket_orbital_count ||
      !exact_pointer(projection.bucket_offsets, topology.bucket_offsets, bucket_offsets,
                     topology.bucket_offset_count) ||
      !exact_pointer(projection.bucket_systems, topology.bucket_systems,
                     topology.bucket_system_count, topology.bucket_system_count) ||
      !exact_pointer(projection.bucket_orbital_counts, topology.bucket_orbital_counts,
                     topology.bucket_orbital_count, topology.bucket_orbital_count)) {
    return projection_failure(Error::kInvalidProjection, Field::kProjection);
  }
  return success();
}

Diagnostic validate_element_identity_projection_binding(
    const ElementIdentityProjectionView& projection, MemorySpace expected_memory_space) noexcept {
  if (projection.memory_space != expected_memory_space ||
      !known_memory_space(expected_memory_space)) {
    return projection_failure(Error::kInvalidMemorySpace, Field::kElementIdentity);
  }
  if (projection.plan_token == 0u) {
    return projection_failure(Error::kInvalidPlanToken, Field::kElementIdentity);
  }
  if (projection.total_atoms != projection.atomic_number_count || projection.total_atoms < 0) {
    return projection_failure(Error::kElementCountMismatch, Field::kElementIdentity);
  }
  AddressRange range{};
  Diagnostic diagnostic = make_range(projection.atomic_numbers, projection.atomic_number_count,
                                     Field::kElementIdentity, range);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  if (projection.element_fingerprint == 0u) {
    return projection_failure(Error::kInvalidElementFingerprint, Field::kElementFingerprint);
  }
  /* The fingerprint is a setup-time host seal copied unchanged into device
   * descriptors, exactly like the wavefunction layout fingerprint.  Only a
   * host descriptor can be re-verified by dereferencing its values; a CUDA
   * descriptor proves identity through the nonzero seal alone. */
  if (expected_memory_space == MemorySpace::kHost) {
    const ElementIdentityProjectionView host_view = projection;
    return element_identity_fingerprint_host(host_view) == projection.element_fingerprint
               ? success()
               : projection_failure(Error::kInvalidElementFingerprint, Field::kElementFingerprint);
  }
  return success();
}

Diagnostic project_atom_projection_host(const TopologyView& topology,
                                        AtomProjectionView& projection) noexcept {
  projection = {};
  Diagnostic diagnostic = validate_topology_host(topology);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  std::int64_t atom_offsets = 0;
  if (!add_one(topology.batch_size, atom_offsets)) {
    return projection_failure(Error::kCountOverflow, Field::kProjection);
  }
  projection.memory_space = topology.memory_space;
  projection.plan_token = topology.plan_token;
  projection.batch_size = topology.batch_size;
  projection.total_atoms = topology.total_atoms;
  projection.atom_offset_count = atom_offsets;
  projection.atom_offsets = topology.atom_offsets;
  return success();
}

Diagnostic project_shell_ownership_projection_host(
    const TopologyView& topology, ShellOwnershipProjectionView& projection) noexcept {
  projection = {};
  Diagnostic diagnostic = validate_topology_host(topology);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  std::int64_t batch_offsets = 0;
  std::int64_t atom_offsets = 0;
  if (!add_one(topology.batch_size, batch_offsets) ||
      !add_one(topology.total_atoms, atom_offsets)) {
    return projection_failure(Error::kCountOverflow, Field::kProjection);
  }
  projection.memory_space = topology.memory_space;
  projection.plan_token = topology.plan_token;
  projection.batch_size = topology.batch_size;
  projection.total_atoms = topology.total_atoms;
  projection.total_shells = topology.total_shells;
  projection.batch_shell_offset_count = batch_offsets;
  projection.atom_shell_offset_count = atom_offsets;
  projection.shell_to_atom_count = topology.total_shells;
  projection.batch_shell_offsets = topology.batch_shell_offsets;
  projection.atom_shell_offsets = topology.atom_shell_offsets;
  projection.shell_to_atom = topology.shell_to_atom;
  return success();
}

Diagnostic project_ao_matrix_projection_host(const TopologyView& topology,
                                             AOMatrixProjectionView& projection) noexcept {
  projection = {};
  Diagnostic diagnostic = validate_topology_host(topology);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  std::int64_t batch_offsets = 0;
  std::int64_t shell_offsets = 0;
  if (!add_one(topology.batch_size, batch_offsets) ||
      !add_one(topology.total_shells, shell_offsets)) {
    return projection_failure(Error::kCountOverflow, Field::kProjection);
  }
  projection.memory_space = topology.memory_space;
  projection.plan_token = topology.plan_token;
  projection.batch_size = topology.batch_size;
  projection.total_shells = topology.total_shells;
  projection.total_orbitals = topology.total_orbitals;
  projection.total_matrix_elements = topology.total_matrix_elements;
  projection.batch_orbital_offset_count = batch_offsets;
  projection.matrix_offset_count = batch_offsets;
  projection.shell_orbital_offset_count = shell_offsets;
  projection.orbital_to_shell_count = topology.total_orbitals;
  projection.orbital_to_atom_count = topology.total_orbitals;
  projection.batch_orbital_offsets = topology.batch_orbital_offsets;
  projection.matrix_offsets = topology.matrix_offsets;
  projection.shell_orbital_offsets = topology.shell_orbital_offsets;
  projection.orbital_to_shell = topology.orbital_to_shell;
  projection.orbital_to_atom = topology.orbital_to_atom;
  return success();
}

Diagnostic project_packed_all_pair_projection_host(
    const TopologyView& topology, PackedAllPairProjectionView& projection) noexcept {
  projection = {};
  Diagnostic diagnostic = validate_topology_host(topology);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  if (topology.pair_map_kind != PairMapKind::kPackedLowerTriangle) {
    return projection_failure(Error::kInvalidProjection, Field::kProjection);
  }
  projection.memory_space = topology.memory_space;
  projection.plan_token = topology.plan_token;
  projection.batch_size = topology.batch_size;
  projection.total_pairs = topology.total_pairs;
  projection.pair_offset_count = topology.pair_offset_count;
  projection.pair_offsets = topology.pair_offsets;
  return success();
}

Diagnostic project_ao_bucket_projection_host(const TopologyView& topology,
                                             AOBucketProjectionView& projection) noexcept {
  projection = {};
  Diagnostic diagnostic = validate_topology_host(topology);
  if (diagnostic.error != Error::kSuccess) {
    return diagnostic;
  }
  std::int64_t bucket_offsets = 0;
  if (topology.bucket_count != 0 && !add_one(topology.bucket_count, bucket_offsets)) {
    return projection_failure(Error::kCountOverflow, Field::kProjection);
  }
  projection.memory_space = topology.memory_space;
  projection.plan_token = topology.plan_token;
  projection.batch_size = topology.batch_size;
  projection.bucket_count = topology.bucket_count;
  projection.bucket_offset_count = bucket_offsets;
  projection.bucket_system_count = topology.bucket_system_count;
  projection.bucket_orbital_count = topology.bucket_orbital_count;
  projection.bucket_offsets = topology.bucket_offsets;
  projection.bucket_systems = topology.bucket_systems;
  projection.bucket_orbital_counts = topology.bucket_orbital_counts;
  return success();
}

Diagnostic project_element_identity_projection_host(
    const std::int32_t* atomic_numbers, std::int64_t atomic_number_count, std::uint64_t plan_token,
    ElementIdentityProjectionView& projection) noexcept {
  projection = {};
  if (plan_token == 0u) {
    return projection_failure(Error::kInvalidPlanToken, Field::kElementIdentity);
  }
  if (atomic_number_count < 0) {
    return projection_failure(Error::kInvalidCount, Field::kElementIdentity);
  }
  if (atomic_number_count != 0) {
    if (atomic_numbers == nullptr) {
      return projection_failure(Error::kNullPointer, Field::kElementIdentity);
    }
    AddressRange range{};
    Diagnostic diagnostic =
        make_range(atomic_numbers, atomic_number_count, Field::kElementIdentity, range);
    if (diagnostic.error != Error::kSuccess) {
      return diagnostic;
    }
  }
  projection.memory_space = MemorySpace::kHost;
  projection.plan_token = plan_token;
  projection.total_atoms = atomic_number_count;
  projection.atomic_number_count = atomic_number_count;
  projection.atomic_numbers = atomic_numbers;
  projection.element_fingerprint = element_identity_fingerprint_host(projection);
  if (projection.element_fingerprint == 0u) {
    return projection_failure(Error::kInvalidElementFingerprint, Field::kElementFingerprint);
  }
  return success();
}

}  // namespace generativeqc::runtime::ragged
