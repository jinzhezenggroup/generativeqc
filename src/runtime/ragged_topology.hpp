// Host-only validation of borrowed ragged atom/shell/AO storage.
// Derived from xTBloom common schema; see src/xtb/native/LICENSE and
// src/xtb/native/CUDA_MKL_LINKING_EXCEPTION for the retained licensing terms.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace generativeqc::runtime::ragged {

// Numeric values preserve the native compatibility diagnostic contract. Gaps
// belong to method policy and are deliberately absent from this shared owner.
enum class MemorySpace : std::uint32_t {
  kHost = 1u,
  kCudaDevice = 2u,
  kHipDevice = 3u,
};

enum class PairMapKind : std::uint32_t {
  kNone = 0u,
  kPackedLowerTriangle = 1u,
  kExplicit = 2u,
};

enum class Error : std::uint32_t {
  kSuccess = 0u,
  kInvalidMemorySpace = 1u,
  kInvalidCount = 2u,
  kCountOverflow = 3u,
  kInvalidPlanToken = 4u,
  kNullPointer = 5u,
  kMisalignedPointer = 6u,
  kAddressOverflow = 7u,
  kAliasedRange = 8u,
  kInvalidOffsets = 9u,
  kInvalidMatrixExtent = 10u,
  kInvalidShellMap = 11u,
  kInvalidOrbitalMap = 12u,
  kInvalidPairMap = 13u,
  kInvalidBucketMap = 14u,
  kCrossPlan = 15u,
  kInvalidElementFingerprint = 24u,
  kInvalidProjection = 25u,
  kElementCountMismatch = 26u,
};

enum class Field : std::uint32_t {
  kNone = 0u,
  kTopology = 1u,
  kAtomOffsets = 2u,
  kBatchShellOffsets = 3u,
  kBatchOrbitalOffsets = 4u,
  kMatrixOffsets = 5u,
  kAtomShellOffsets = 6u,
  kShellOrbitalOffsets = 7u,
  kShellToAtom = 8u,
  kOrbitalToShell = 9u,
  kOrbitalToAtom = 10u,
  kPairOffsets = 11u,
  kAtomPairs = 12u,
  kBucketOffsets = 13u,
  kBucketSystems = 14u,
  kBucketOrbitalCounts = 15u,
  kProjection = 34u,
  kElementIdentity = 35u,
  kElementFingerprint = 36u,
};

struct Diagnostic {
  Error error = Error::kSuccess;
  Field field = Field::kNone;
  /* Global array index, system index, or -1 when no narrower location exists. */
  std::int64_t index = -1;
};

struct AtomPairLayout {
  std::int64_t first = 0;
  std::int64_t second = 0;
};

// Immutable batch-major storage: every count is in elements. Empty systems are
// allowed; every nonempty atom owns shells and every shell owns orbitals. Each
// matrix slice is a complete row-major square. Optional explicit pairs are
// lexicographically first-endpoint-major; AO buckets partition all systems.
struct TopologyView {
  MemorySpace memory_space = MemorySpace::kHost;
  PairMapKind pair_map_kind = PairMapKind::kNone;
  std::uint64_t plan_token = 0u;

  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t total_pairs = 0;
  std::int64_t bucket_count = 0;

  std::int64_t atom_offset_count = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t batch_orbital_offset_count = 0;
  std::int64_t matrix_offset_count = 0;
  std::int64_t atom_shell_offset_count = 0;
  std::int64_t shell_orbital_offset_count = 0;
  std::int64_t shell_to_atom_count = 0;
  std::int64_t orbital_to_shell_count = 0;
  std::int64_t orbital_to_atom_count = 0;
  std::int64_t pair_offset_count = 0;
  std::int64_t atom_pair_count = 0;
  std::int64_t bucket_offset_count = 0;
  std::int64_t bucket_system_count = 0;
  std::int64_t bucket_orbital_count = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* batch_shell_offsets = nullptr;
  const std::int64_t* batch_orbital_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int64_t* shell_orbital_offsets = nullptr;
  const std::int64_t* shell_to_atom = nullptr;
  const std::int64_t* orbital_to_shell = nullptr;
  const std::int64_t* orbital_to_atom = nullptr;
  const std::int64_t* pair_offsets = nullptr;
  const void* atom_pairs = nullptr;
  const std::int64_t* bucket_offsets = nullptr;
  const std::int32_t* bucket_systems = nullptr;
  const std::int32_t* bucket_orbital_counts = nullptr;
};

struct AtomProjectionView {
  MemorySpace memory_space = MemorySpace::kHost;
  std::uint64_t plan_token = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t atom_offset_count = 0;
  const std::int64_t* atom_offsets = nullptr;
};

struct ShellOwnershipProjectionView {
  MemorySpace memory_space = MemorySpace::kHost;
  std::uint64_t plan_token = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t atom_shell_offset_count = 0;
  std::int64_t shell_to_atom_count = 0;
  const std::int64_t* batch_shell_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int64_t* shell_to_atom = nullptr;
};

struct AOMatrixProjectionView {
  MemorySpace memory_space = MemorySpace::kHost;
  std::uint64_t plan_token = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t batch_orbital_offset_count = 0;
  std::int64_t matrix_offset_count = 0;
  std::int64_t shell_orbital_offset_count = 0;
  std::int64_t orbital_to_shell_count = 0;
  std::int64_t orbital_to_atom_count = 0;
  const std::int64_t* batch_orbital_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* shell_orbital_offsets = nullptr;
  const std::int64_t* orbital_to_shell = nullptr;
  const std::int64_t* orbital_to_atom = nullptr;
};

struct PackedAllPairProjectionView {
  MemorySpace memory_space = MemorySpace::kHost;
  std::uint64_t plan_token = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_pairs = 0;
  std::int64_t pair_offset_count = 0;
  const std::int64_t* pair_offsets = nullptr;
};

struct AOBucketProjectionView {
  MemorySpace memory_space = MemorySpace::kHost;
  std::uint64_t plan_token = 0u;
  std::int64_t batch_size = 0;
  std::int64_t bucket_count = 0;
  std::int64_t bucket_offset_count = 0;
  std::int64_t bucket_system_count = 0;
  std::int64_t bucket_orbital_count = 0;
  const std::int64_t* bucket_offsets = nullptr;
  const std::int32_t* bucket_systems = nullptr;
  const std::int32_t* bucket_orbital_counts = nullptr;
};

// The order-sensitive setup seal excludes addresses and memory space. It is
// copied unchanged into device descriptors and only recomputed for host views.
struct ElementIdentityProjectionView {
  MemorySpace memory_space = MemorySpace::kHost;
  std::uint64_t plan_token = 0u;
  std::int64_t total_atoms = 0;
  std::int64_t atomic_number_count = 0;
  std::uint64_t element_fingerprint = 0u;
  const std::int32_t* atomic_numbers = nullptr;
};

// TopologyView borrows every array. Structural validation does not dereference
// storage; the host inspector requires host-readable arrays. atom_pairs names
// an aligned array whose elements have exactly AtomPairLayout's representation.
// Host inspection copies each pair's bytes, so unrelated compatible POD types
// never require an aliasing cast. The owner allocates and retains nothing.
static_assert(std::is_trivially_copyable_v<AtomPairLayout>);
static_assert(std::is_standard_layout_v<AtomPairLayout>);
static_assert(sizeof(AtomPairLayout) == 2 * sizeof(std::int64_t));
static_assert(offsetof(AtomPairLayout, second) == sizeof(std::int64_t));

// Structural validators inspect counts, addresses and exact borrowed identity
// only. Host validators inspect values. Builders clear their output before
// validation; all functions preserve the first failing diagnostic.
[[nodiscard]] Diagnostic validate_topology_binding(const TopologyView& topology,
                                                   MemorySpace expected_memory_space) noexcept;

[[nodiscard]] Diagnostic validate_topology_host(const TopologyView& topology) noexcept;

[[nodiscard]] Diagnostic bind_topology_host(const TopologyView& candidate,
                                            TopologyView& binding) noexcept;

[[nodiscard]] std::uint64_t element_identity_fingerprint_host(
    const ElementIdentityProjectionView& element) noexcept;

[[nodiscard]] Diagnostic validate_atom_projection_binding(
    const TopologyView& topology, const AtomProjectionView& projection,
    MemorySpace expected_memory_space) noexcept;

[[nodiscard]] Diagnostic validate_shell_ownership_projection_binding(
    const TopologyView& topology, const ShellOwnershipProjectionView& projection,
    MemorySpace expected_memory_space) noexcept;

[[nodiscard]] Diagnostic validate_ao_matrix_projection_binding(
    const TopologyView& topology, const AOMatrixProjectionView& projection,
    MemorySpace expected_memory_space) noexcept;

[[nodiscard]] Diagnostic validate_packed_all_pair_projection_binding(
    const TopologyView& topology, const PackedAllPairProjectionView& projection,
    MemorySpace expected_memory_space) noexcept;

[[nodiscard]] Diagnostic validate_ao_bucket_projection_binding(
    const TopologyView& topology, const AOBucketProjectionView& projection,
    MemorySpace expected_memory_space) noexcept;

[[nodiscard]] Diagnostic validate_element_identity_projection_binding(
    const ElementIdentityProjectionView& projection, MemorySpace expected_memory_space) noexcept;

[[nodiscard]] Diagnostic project_atom_projection_host(const TopologyView& topology,
                                                      AtomProjectionView& projection) noexcept;

[[nodiscard]] Diagnostic project_shell_ownership_projection_host(
    const TopologyView& topology, ShellOwnershipProjectionView& projection) noexcept;

[[nodiscard]] Diagnostic project_ao_matrix_projection_host(
    const TopologyView& topology, AOMatrixProjectionView& projection) noexcept;

[[nodiscard]] Diagnostic project_packed_all_pair_projection_host(
    const TopologyView& topology, PackedAllPairProjectionView& projection) noexcept;

[[nodiscard]] Diagnostic project_ao_bucket_projection_host(
    const TopologyView& topology, AOBucketProjectionView& projection) noexcept;

[[nodiscard]] Diagnostic project_element_identity_projection_host(
    const std::int32_t* atomic_numbers, std::int64_t atomic_number_count, std::uint64_t plan_token,
    ElementIdentityProjectionView& projection) noexcept;

}  // namespace generativeqc::runtime::ragged
