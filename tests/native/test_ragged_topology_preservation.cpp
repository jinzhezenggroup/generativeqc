// Compile this real compatibility-API probe against the frozen oracle and
// current wrappers separately. Traces include every diagnostic and ABI offset.
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>

#include "backends/common/gfn2_plan_schema.hpp"

namespace {
bool counting = false;
std::size_t allocations = 0;
void count_allocation() {
  if (counting) ++allocations;
}
}  // namespace
extern "C" {
void* __real_malloc(std::size_t);
void* __real_calloc(std::size_t, std::size_t);
void* __real_realloc(void*, std::size_t);
void* __real_aligned_alloc(std::size_t, std::size_t);
int __real_posix_memalign(void**, std::size_t, std::size_t);
void* __wrap_malloc(std::size_t n) {
  count_allocation();
  return __real_malloc(n);
}
void* __wrap_calloc(std::size_t n, std::size_t s) {
  count_allocation();
  return __real_calloc(n, s);
}
void* __wrap_realloc(void* p, std::size_t n) {
  count_allocation();
  return __real_realloc(p, n);
}
void* __wrap_aligned_alloc(std::size_t a, std::size_t n) {
  count_allocation();
  return __real_aligned_alloc(a, n);
}
int __wrap_posix_memalign(void** p, std::size_t a, std::size_t n) {
  count_allocation();
  return __real_posix_memalign(p, a, n);
}
}
void* operator new(std::size_t n) {
  count_allocation();
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc{};
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
  count_allocation();
  void* p = nullptr;
  if (posix_memalign(&p, static_cast<std::size_t>(a), n ? n : 1)) throw std::bad_alloc{};
  return p;
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {
using namespace generativeqc::xtb::detail;
using I = std::int64_t;
using E = Gfn2PlanSchemaError;
using F = Gfn2PlanSchemaField;
using M = Gfn2PlanMemorySpace;
using T = Gfn2RaggedTopologyView;
constexpr I maximum = std::numeric_limits<I>::max();
void require(bool condition, const char* name) {
  if (!condition) {
    std::fprintf(stderr, "ragged topology: %s\n", name);
    std::abort();
  }
}
template <class Function>
auto no_alloc(Function function) {
  const auto before = allocations;
  counting = true;
  const auto result = function();
  counting = false;
  require(allocations == before, "production allocated");
  return result;
}
void diagnostic(const char* label, Gfn2PlanSchemaDiagnostic d) {
  std::printf("%s %u %u %lld\n", label, static_cast<unsigned>(d.error),
              static_cast<unsigned>(d.field), static_cast<long long>(d.index));
}
template <class Function>
auto invoke(const char* name, Function function) {
  auto d = no_alloc(function);
  diagnostic(name, d);
  return d;
}
template <class V>
void corrupt(V& value) {
  if constexpr (std::is_pointer_v<V>)
    value = reinterpret_cast<V>(0x1008u);
  else if constexpr (std::is_enum_v<V>)
    value = static_cast<V>(99);
  else
    value = static_cast<V>(12345);
}

#define FIELDS_Gfn2RaggedTopologyView(X) \
  X(memory_space)                        \
  X(pair_map_kind)                       \
  X(plan_token)                          \
  X(batch_size)                          \
  X(total_atoms)                         \
  X(total_shells)                        \
  X(total_orbitals)                      \
  X(total_matrix_elements)               \
  X(total_pairs)                         \
  X(bucket_count)                        \
  X(atom_offset_count)                   \
  X(batch_shell_offset_count)            \
  X(batch_orbital_offset_count)          \
  X(matrix_offset_count)                 \
  X(atom_shell_offset_count)             \
  X(shell_orbital_offset_count)          \
  X(shell_to_atom_count)                 \
  X(orbital_to_shell_count)              \
  X(orbital_to_atom_count)               \
  X(pair_offset_count)                   \
  X(atom_pair_count)                     \
  X(bucket_offset_count)                 \
  X(bucket_system_count)                 \
  X(bucket_orbital_count)                \
  X(atom_offsets)                        \
  X(batch_shell_offsets)                 \
  X(batch_orbital_offsets)               \
  X(matrix_offsets)                      \
  X(atom_shell_offsets)                  \
  X(shell_orbital_offsets)               \
  X(shell_to_atom)                       \
  X(orbital_to_shell)                    \
  X(orbital_to_atom)                     \
  X(pair_offsets)                        \
  X(atom_pairs)                          \
  X(bucket_offsets)                      \
  X(bucket_systems)                      \
  X(bucket_orbital_counts)

#define FIELDS_Gfn2AtomProjectionView(X) \
  X(memory_space)                        \
  X(plan_token)                          \
  X(batch_size)                          \
  X(total_atoms)                         \
  X(atom_offset_count)                   \
  X(atom_offsets)

#define FIELDS_Gfn2ShellOwnershipProjectionView(X) \
  X(memory_space)                                  \
  X(plan_token)                                    \
  X(batch_size)                                    \
  X(total_atoms)                                   \
  X(total_shells)                                  \
  X(batch_shell_offset_count)                      \
  X(atom_shell_offset_count)                       \
  X(shell_to_atom_count)                           \
  X(batch_shell_offsets)                           \
  X(atom_shell_offsets)                            \
  X(shell_to_atom)

#define FIELDS_Gfn2AOMatrixProjectionView(X) \
  X(memory_space)                            \
  X(plan_token)                              \
  X(batch_size)                              \
  X(total_shells)                            \
  X(total_orbitals)                          \
  X(total_matrix_elements)                   \
  X(batch_orbital_offset_count)              \
  X(matrix_offset_count)                     \
  X(shell_orbital_offset_count)              \
  X(orbital_to_shell_count)                  \
  X(orbital_to_atom_count)                   \
  X(batch_orbital_offsets)                   \
  X(matrix_offsets)                          \
  X(shell_orbital_offsets)                   \
  X(orbital_to_shell)                        \
  X(orbital_to_atom)

#define FIELDS_Gfn2PackedAllPairProjectionView(X) \
  X(memory_space)                                 \
  X(plan_token)                                   \
  X(batch_size)                                   \
  X(total_pairs)                                  \
  X(pair_offset_count)                            \
  X(pair_offsets)

#define FIELDS_Gfn2AOBucketProjectionView(X) \
  X(memory_space)                            \
  X(plan_token)                              \
  X(batch_size)                              \
  X(bucket_count)                            \
  X(bucket_offset_count)                     \
  X(bucket_system_count)                     \
  X(bucket_orbital_count)                    \
  X(bucket_offsets)                          \
  X(bucket_systems)                          \
  X(bucket_orbital_counts)

#define FIELDS_Gfn2ElementIdentityProjectionView(X) \
  X(memory_space)                                   \
  X(plan_token)                                     \
  X(total_atoms)                                    \
  X(atomic_number_count)                            \
  X(element_fingerprint)                            \
  X(atomic_numbers)

bool equal(const Gfn2RaggedTopologyView& a, const Gfn2RaggedTopologyView& b) {
#define COMPARE(f) \
  if (a.f != b.f) return false;
  FIELDS_Gfn2RaggedTopologyView(COMPARE)
#undef COMPARE
      return true;
}
Gfn2RaggedTopologyView poisoned_Gfn2RaggedTopologyView() {
  Gfn2RaggedTopologyView value{};
#define POISON(f) corrupt(value.f);
  FIELDS_Gfn2RaggedTopologyView(POISON)
#undef POISON
      return value;
}
bool equal(const Gfn2AtomProjectionView& a, const Gfn2AtomProjectionView& b) {
#define COMPARE(f) \
  if (a.f != b.f) return false;
  FIELDS_Gfn2AtomProjectionView(COMPARE)
#undef COMPARE
      return true;
}
Gfn2AtomProjectionView poisoned_Gfn2AtomProjectionView() {
  Gfn2AtomProjectionView value{};
#define POISON(f) corrupt(value.f);
  FIELDS_Gfn2AtomProjectionView(POISON)
#undef POISON
      return value;
}
bool equal(const Gfn2ShellOwnershipProjectionView& a, const Gfn2ShellOwnershipProjectionView& b) {
#define COMPARE(f) \
  if (a.f != b.f) return false;
  FIELDS_Gfn2ShellOwnershipProjectionView(COMPARE)
#undef COMPARE
      return true;
}
Gfn2ShellOwnershipProjectionView poisoned_Gfn2ShellOwnershipProjectionView() {
  Gfn2ShellOwnershipProjectionView value{};
#define POISON(f) corrupt(value.f);
  FIELDS_Gfn2ShellOwnershipProjectionView(POISON)
#undef POISON
      return value;
}
bool equal(const Gfn2AOMatrixProjectionView& a, const Gfn2AOMatrixProjectionView& b) {
#define COMPARE(f) \
  if (a.f != b.f) return false;
  FIELDS_Gfn2AOMatrixProjectionView(COMPARE)
#undef COMPARE
      return true;
}
Gfn2AOMatrixProjectionView poisoned_Gfn2AOMatrixProjectionView() {
  Gfn2AOMatrixProjectionView value{};
#define POISON(f) corrupt(value.f);
  FIELDS_Gfn2AOMatrixProjectionView(POISON)
#undef POISON
      return value;
}
bool equal(const Gfn2PackedAllPairProjectionView& a, const Gfn2PackedAllPairProjectionView& b) {
#define COMPARE(f) \
  if (a.f != b.f) return false;
  FIELDS_Gfn2PackedAllPairProjectionView(COMPARE)
#undef COMPARE
      return true;
}
Gfn2PackedAllPairProjectionView poisoned_Gfn2PackedAllPairProjectionView() {
  Gfn2PackedAllPairProjectionView value{};
#define POISON(f) corrupt(value.f);
  FIELDS_Gfn2PackedAllPairProjectionView(POISON)
#undef POISON
      return value;
}
bool equal(const Gfn2AOBucketProjectionView& a, const Gfn2AOBucketProjectionView& b) {
#define COMPARE(f) \
  if (a.f != b.f) return false;
  FIELDS_Gfn2AOBucketProjectionView(COMPARE)
#undef COMPARE
      return true;
}
Gfn2AOBucketProjectionView poisoned_Gfn2AOBucketProjectionView() {
  Gfn2AOBucketProjectionView value{};
#define POISON(f) corrupt(value.f);
  FIELDS_Gfn2AOBucketProjectionView(POISON)
#undef POISON
      return value;
}
bool equal(const Gfn2ElementIdentityProjectionView& a, const Gfn2ElementIdentityProjectionView& b) {
#define COMPARE(f) \
  if (a.f != b.f) return false;
  FIELDS_Gfn2ElementIdentityProjectionView(COMPARE)
#undef COMPARE
      return true;
}
Gfn2ElementIdentityProjectionView poisoned_Gfn2ElementIdentityProjectionView() {
  Gfn2ElementIdentityProjectionView value{};
#define POISON(f) corrupt(value.f);
  FIELDS_Gfn2ElementIdentityProjectionView(POISON)
#undef POISON
      return value;
}

void abi() {
  std::printf("abi Gfn2PlanSchemaDiagnostic %zu %zu", sizeof(Gfn2PlanSchemaDiagnostic),
              alignof(Gfn2PlanSchemaDiagnostic));
  std::printf(" %zu", offsetof(Gfn2PlanSchemaDiagnostic, error));
  std::printf(" %zu", offsetof(Gfn2PlanSchemaDiagnostic, field));
  std::printf(" %zu", offsetof(Gfn2PlanSchemaDiagnostic, index));
  std::puts("");
  std::printf("abi Gfn2AtomPair %zu %zu", sizeof(Gfn2AtomPair), alignof(Gfn2AtomPair));
  std::printf(" %zu", offsetof(Gfn2AtomPair, first));
  std::printf(" %zu", offsetof(Gfn2AtomPair, second));
  std::puts("");
  std::printf("abi Gfn2RaggedTopologyView %zu %zu", sizeof(Gfn2RaggedTopologyView),
              alignof(Gfn2RaggedTopologyView));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, memory_space));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, pair_map_kind));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, plan_token));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, batch_size));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, total_atoms));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, total_shells));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, total_orbitals));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, total_matrix_elements));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, total_pairs));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, bucket_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, atom_offset_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, batch_shell_offset_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, batch_orbital_offset_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, matrix_offset_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, atom_shell_offset_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, shell_orbital_offset_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, shell_to_atom_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, orbital_to_shell_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, orbital_to_atom_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, pair_offset_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, atom_pair_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, bucket_offset_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, bucket_system_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, bucket_orbital_count));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, atom_offsets));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, batch_shell_offsets));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, batch_orbital_offsets));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, matrix_offsets));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, atom_shell_offsets));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, shell_orbital_offsets));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, shell_to_atom));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, orbital_to_shell));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, orbital_to_atom));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, pair_offsets));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, atom_pairs));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, bucket_offsets));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, bucket_systems));
  std::printf(" %zu", offsetof(Gfn2RaggedTopologyView, bucket_orbital_counts));
  std::puts("");
  std::printf("abi Gfn2WavefunctionLayoutView %zu %zu", sizeof(Gfn2WavefunctionLayoutView),
              alignof(Gfn2WavefunctionLayoutView));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, memory_space));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, plan_token));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, layout_fingerprint));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, batch_size));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, total_spin_channels));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, total_spin_orbitals));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, total_spin_matrix_elements));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, total_spin_shells));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, total_spin_atoms));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, spin_channel_count));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, spin_channel_offset_count));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, spin_orbital_offset_count));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, spin_matrix_offset_count));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, spin_shell_offset_count));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, spin_atom_offset_count));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, spin_channels));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, spin_channel_offsets));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, spin_orbital_offsets));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, spin_matrix_offsets));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, spin_shell_offsets));
  std::printf(" %zu", offsetof(Gfn2WavefunctionLayoutView, spin_atom_offsets));
  std::puts("");
  std::printf("abi Gfn2GeometryCacheProvenanceView %zu %zu",
              sizeof(Gfn2GeometryCacheProvenanceView), alignof(Gfn2GeometryCacheProvenanceView));
  std::printf(" %zu", offsetof(Gfn2GeometryCacheProvenanceView, memory_space));
  std::printf(" %zu", offsetof(Gfn2GeometryCacheProvenanceView, generation_scope));
  std::printf(" %zu", offsetof(Gfn2GeometryCacheProvenanceView, plan_token));
  std::printf(" %zu", offsetof(Gfn2GeometryCacheProvenanceView, geometry_generation));
  std::printf(" %zu", offsetof(Gfn2GeometryCacheProvenanceView, batch_size));
  std::printf(" %zu", offsetof(Gfn2GeometryCacheProvenanceView, system_generation_count));
  std::printf(" %zu", offsetof(Gfn2GeometryCacheProvenanceView, system_geometry_generations));
  std::puts("");
  std::printf("abi Gfn2PairListConsumerView %zu %zu", sizeof(Gfn2PairListConsumerView),
              alignof(Gfn2PairListConsumerView));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, memory_space));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, state));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, role));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, pair_map_kind));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, plan_token));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, cutoff_bohr));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, list_builder_cutoff_bohr));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, batch_size));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, total_atoms));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, max_pairs_per_system));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, max_neighbors_per_atom));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, pair_offset_count));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, neighbor_offset_count));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, pair_count));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, neighbor_count));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, pair_offsets));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, pairs));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, pair_count_elements));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, neighbor_count_elements));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, pair_counts));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, neighbor_counts));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, neighbor_offsets));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, neighbors));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, committed_generation_count));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, eligible_mask_count));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, active_mask_count));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, committed_generations));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, eligible_mask));
  std::printf(" %zu", offsetof(Gfn2PairListConsumerView, active_mask));
  std::puts("");
  std::printf("abi Gfn2AtomProjectionView %zu %zu", sizeof(Gfn2AtomProjectionView),
              alignof(Gfn2AtomProjectionView));
  std::printf(" %zu", offsetof(Gfn2AtomProjectionView, memory_space));
  std::printf(" %zu", offsetof(Gfn2AtomProjectionView, plan_token));
  std::printf(" %zu", offsetof(Gfn2AtomProjectionView, batch_size));
  std::printf(" %zu", offsetof(Gfn2AtomProjectionView, total_atoms));
  std::printf(" %zu", offsetof(Gfn2AtomProjectionView, atom_offset_count));
  std::printf(" %zu", offsetof(Gfn2AtomProjectionView, atom_offsets));
  std::puts("");
  std::printf("abi Gfn2ShellOwnershipProjectionView %zu %zu",
              sizeof(Gfn2ShellOwnershipProjectionView), alignof(Gfn2ShellOwnershipProjectionView));
  std::printf(" %zu", offsetof(Gfn2ShellOwnershipProjectionView, memory_space));
  std::printf(" %zu", offsetof(Gfn2ShellOwnershipProjectionView, plan_token));
  std::printf(" %zu", offsetof(Gfn2ShellOwnershipProjectionView, batch_size));
  std::printf(" %zu", offsetof(Gfn2ShellOwnershipProjectionView, total_atoms));
  std::printf(" %zu", offsetof(Gfn2ShellOwnershipProjectionView, total_shells));
  std::printf(" %zu", offsetof(Gfn2ShellOwnershipProjectionView, batch_shell_offset_count));
  std::printf(" %zu", offsetof(Gfn2ShellOwnershipProjectionView, atom_shell_offset_count));
  std::printf(" %zu", offsetof(Gfn2ShellOwnershipProjectionView, shell_to_atom_count));
  std::printf(" %zu", offsetof(Gfn2ShellOwnershipProjectionView, batch_shell_offsets));
  std::printf(" %zu", offsetof(Gfn2ShellOwnershipProjectionView, atom_shell_offsets));
  std::printf(" %zu", offsetof(Gfn2ShellOwnershipProjectionView, shell_to_atom));
  std::puts("");
  std::printf("abi Gfn2AOMatrixProjectionView %zu %zu", sizeof(Gfn2AOMatrixProjectionView),
              alignof(Gfn2AOMatrixProjectionView));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, memory_space));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, plan_token));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, batch_size));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, total_shells));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, total_orbitals));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, total_matrix_elements));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, batch_orbital_offset_count));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, matrix_offset_count));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, shell_orbital_offset_count));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, orbital_to_shell_count));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, orbital_to_atom_count));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, batch_orbital_offsets));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, matrix_offsets));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, shell_orbital_offsets));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, orbital_to_shell));
  std::printf(" %zu", offsetof(Gfn2AOMatrixProjectionView, orbital_to_atom));
  std::puts("");
  std::printf("abi Gfn2PackedAllPairProjectionView %zu %zu",
              sizeof(Gfn2PackedAllPairProjectionView), alignof(Gfn2PackedAllPairProjectionView));
  std::printf(" %zu", offsetof(Gfn2PackedAllPairProjectionView, memory_space));
  std::printf(" %zu", offsetof(Gfn2PackedAllPairProjectionView, plan_token));
  std::printf(" %zu", offsetof(Gfn2PackedAllPairProjectionView, batch_size));
  std::printf(" %zu", offsetof(Gfn2PackedAllPairProjectionView, total_pairs));
  std::printf(" %zu", offsetof(Gfn2PackedAllPairProjectionView, pair_offset_count));
  std::printf(" %zu", offsetof(Gfn2PackedAllPairProjectionView, pair_offsets));
  std::puts("");
  std::printf("abi Gfn2AOBucketProjectionView %zu %zu", sizeof(Gfn2AOBucketProjectionView),
              alignof(Gfn2AOBucketProjectionView));
  std::printf(" %zu", offsetof(Gfn2AOBucketProjectionView, memory_space));
  std::printf(" %zu", offsetof(Gfn2AOBucketProjectionView, plan_token));
  std::printf(" %zu", offsetof(Gfn2AOBucketProjectionView, batch_size));
  std::printf(" %zu", offsetof(Gfn2AOBucketProjectionView, bucket_count));
  std::printf(" %zu", offsetof(Gfn2AOBucketProjectionView, bucket_offset_count));
  std::printf(" %zu", offsetof(Gfn2AOBucketProjectionView, bucket_system_count));
  std::printf(" %zu", offsetof(Gfn2AOBucketProjectionView, bucket_orbital_count));
  std::printf(" %zu", offsetof(Gfn2AOBucketProjectionView, bucket_offsets));
  std::printf(" %zu", offsetof(Gfn2AOBucketProjectionView, bucket_systems));
  std::printf(" %zu", offsetof(Gfn2AOBucketProjectionView, bucket_orbital_counts));
  std::puts("");
  std::printf("abi Gfn2ElementIdentityProjectionView %zu %zu",
              sizeof(Gfn2ElementIdentityProjectionView),
              alignof(Gfn2ElementIdentityProjectionView));
  std::printf(" %zu", offsetof(Gfn2ElementIdentityProjectionView, memory_space));
  std::printf(" %zu", offsetof(Gfn2ElementIdentityProjectionView, plan_token));
  std::printf(" %zu", offsetof(Gfn2ElementIdentityProjectionView, total_atoms));
  std::printf(" %zu", offsetof(Gfn2ElementIdentityProjectionView, atomic_number_count));
  std::printf(" %zu", offsetof(Gfn2ElementIdentityProjectionView, element_fingerprint));
  std::printf(" %zu", offsetof(Gfn2ElementIdentityProjectionView, atomic_numbers));
  std::puts("");
  std::printf("enum Gfn2PlanMemorySpace::kHost %u\n",
              static_cast<unsigned>(Gfn2PlanMemorySpace::kHost));
  std::printf("enum Gfn2PlanMemorySpace::kCudaDevice %u\n",
              static_cast<unsigned>(Gfn2PlanMemorySpace::kCudaDevice));
  std::printf("enum Gfn2PlanMemorySpace::kHipDevice %u\n",
              static_cast<unsigned>(Gfn2PlanMemorySpace::kHipDevice));
  std::printf("enum Gfn2PairMapKind::kNone %u\n", static_cast<unsigned>(Gfn2PairMapKind::kNone));
  std::printf("enum Gfn2PairMapKind::kPackedLowerTriangle %u\n",
              static_cast<unsigned>(Gfn2PairMapKind::kPackedLowerTriangle));
  std::printf("enum Gfn2PairMapKind::kExplicit %u\n",
              static_cast<unsigned>(Gfn2PairMapKind::kExplicit));
  std::printf("enum Gfn2GenerationScope::kBatch %u\n",
              static_cast<unsigned>(Gfn2GenerationScope::kBatch));
  std::printf("enum Gfn2GenerationScope::kPerSystem %u\n",
              static_cast<unsigned>(Gfn2GenerationScope::kPerSystem));
  std::printf("enum Gfn2PairListRole::kCoordination %u\n",
              static_cast<unsigned>(Gfn2PairListRole::kCoordination));
  std::printf("enum Gfn2PairListRole::kD4Coordination %u\n",
              static_cast<unsigned>(Gfn2PairListRole::kD4Coordination));
  std::printf("enum Gfn2PairListRole::kD4TwoBody %u\n",
              static_cast<unsigned>(Gfn2PairListRole::kD4TwoBody));
  std::printf("enum Gfn2PairListRole::kD4Atm %u\n",
              static_cast<unsigned>(Gfn2PairListRole::kD4Atm));
  std::printf("enum Gfn2PairListState::kCandidate %u\n",
              static_cast<unsigned>(Gfn2PairListState::kCandidate));
  std::printf("enum Gfn2PairListState::kCommitted %u\n",
              static_cast<unsigned>(Gfn2PairListState::kCommitted));
  std::printf("enum Gfn2PlanSchemaError::kSuccess %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kSuccess));
  std::printf("enum Gfn2PlanSchemaError::kInvalidMemorySpace %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidMemorySpace));
  std::printf("enum Gfn2PlanSchemaError::kInvalidCount %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidCount));
  std::printf("enum Gfn2PlanSchemaError::kCountOverflow %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kCountOverflow));
  std::printf("enum Gfn2PlanSchemaError::kInvalidPlanToken %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidPlanToken));
  std::printf("enum Gfn2PlanSchemaError::kNullPointer %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kNullPointer));
  std::printf("enum Gfn2PlanSchemaError::kMisalignedPointer %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kMisalignedPointer));
  std::printf("enum Gfn2PlanSchemaError::kAddressOverflow %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kAddressOverflow));
  std::printf("enum Gfn2PlanSchemaError::kAliasedRange %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kAliasedRange));
  std::printf("enum Gfn2PlanSchemaError::kInvalidOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidOffsets));
  std::printf("enum Gfn2PlanSchemaError::kInvalidMatrixExtent %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidMatrixExtent));
  std::printf("enum Gfn2PlanSchemaError::kInvalidShellMap %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidShellMap));
  std::printf("enum Gfn2PlanSchemaError::kInvalidOrbitalMap %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidOrbitalMap));
  std::printf("enum Gfn2PlanSchemaError::kInvalidPairMap %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidPairMap));
  std::printf("enum Gfn2PlanSchemaError::kInvalidBucketMap %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidBucketMap));
  std::printf("enum Gfn2PlanSchemaError::kCrossPlan %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kCrossPlan));
  std::printf("enum Gfn2PlanSchemaError::kStaleGeometry %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kStaleGeometry));
  std::printf("enum Gfn2PlanSchemaError::kInvalidActiveMask %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidActiveMask));
  std::printf("enum Gfn2PlanSchemaError::kInvalidSpinChannels %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidSpinChannels));
  std::printf("enum Gfn2PlanSchemaError::kInvalidWavefunctionExtent %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidWavefunctionExtent));
  std::printf("enum Gfn2PlanSchemaError::kInvalidLayoutFingerprint %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidLayoutFingerprint));
  std::printf("enum Gfn2PlanSchemaError::kInvalidPairListRole %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidPairListRole));
  std::printf("enum Gfn2PlanSchemaError::kInvalidPairListState %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidPairListState));
  std::printf("enum Gfn2PlanSchemaError::kInsufficientPairListCutoff %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInsufficientPairListCutoff));
  std::printf("enum Gfn2PlanSchemaError::kInvalidElementFingerprint %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidElementFingerprint));
  std::printf("enum Gfn2PlanSchemaError::kInvalidProjection %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kInvalidProjection));
  std::printf("enum Gfn2PlanSchemaError::kElementCountMismatch %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaError::kElementCountMismatch));
  std::printf("enum Gfn2PlanSchemaField::kNone %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kNone));
  std::printf("enum Gfn2PlanSchemaField::kTopology %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kTopology));
  std::printf("enum Gfn2PlanSchemaField::kAtomOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kAtomOffsets));
  std::printf("enum Gfn2PlanSchemaField::kBatchShellOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kBatchShellOffsets));
  std::printf("enum Gfn2PlanSchemaField::kBatchOrbitalOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kBatchOrbitalOffsets));
  std::printf("enum Gfn2PlanSchemaField::kMatrixOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kMatrixOffsets));
  std::printf("enum Gfn2PlanSchemaField::kAtomShellOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kAtomShellOffsets));
  std::printf("enum Gfn2PlanSchemaField::kShellOrbitalOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kShellOrbitalOffsets));
  std::printf("enum Gfn2PlanSchemaField::kShellToAtom %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kShellToAtom));
  std::printf("enum Gfn2PlanSchemaField::kOrbitalToShell %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kOrbitalToShell));
  std::printf("enum Gfn2PlanSchemaField::kOrbitalToAtom %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kOrbitalToAtom));
  std::printf("enum Gfn2PlanSchemaField::kPairOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kPairOffsets));
  std::printf("enum Gfn2PlanSchemaField::kAtomPairs %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kAtomPairs));
  std::printf("enum Gfn2PlanSchemaField::kBucketOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kBucketOffsets));
  std::printf("enum Gfn2PlanSchemaField::kBucketSystems %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kBucketSystems));
  std::printf("enum Gfn2PlanSchemaField::kBucketOrbitalCounts %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kBucketOrbitalCounts));
  std::printf("enum Gfn2PlanSchemaField::kGeometryProvenance %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kGeometryProvenance));
  std::printf("enum Gfn2PlanSchemaField::kSystemGeometryGenerations %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kSystemGeometryGenerations));
  std::printf("enum Gfn2PlanSchemaField::kActiveMask %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kActiveMask));
  std::printf("enum Gfn2PlanSchemaField::kSpinChannels %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kSpinChannels));
  std::printf("enum Gfn2PlanSchemaField::kSpinChannelOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kSpinChannelOffsets));
  std::printf("enum Gfn2PlanSchemaField::kSpinOrbitalOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kSpinOrbitalOffsets));
  std::printf("enum Gfn2PlanSchemaField::kSpinMatrixOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kSpinMatrixOffsets));
  std::printf("enum Gfn2PlanSchemaField::kSpinShellOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kSpinShellOffsets));
  std::printf("enum Gfn2PlanSchemaField::kSpinAtomOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kSpinAtomOffsets));
  std::printf("enum Gfn2PlanSchemaField::kWavefunctionLayoutFingerprint %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kWavefunctionLayoutFingerprint));
  std::printf("enum Gfn2PlanSchemaField::kPairListConsumer %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kPairListConsumer));
  std::printf("enum Gfn2PlanSchemaField::kPairListOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kPairListOffsets));
  std::printf("enum Gfn2PlanSchemaField::kPairListPairs %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kPairListPairs));
  std::printf("enum Gfn2PlanSchemaField::kPairListNeighborOffsets %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kPairListNeighborOffsets));
  std::printf("enum Gfn2PlanSchemaField::kPairListNeighbors %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kPairListNeighbors));
  std::printf("enum Gfn2PlanSchemaField::kPairListGenerations %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kPairListGenerations));
  std::printf("enum Gfn2PlanSchemaField::kPairListEligibleMask %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kPairListEligibleMask));
  std::printf("enum Gfn2PlanSchemaField::kPairListActiveMask %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kPairListActiveMask));
  std::printf("enum Gfn2PlanSchemaField::kProjection %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kProjection));
  std::printf("enum Gfn2PlanSchemaField::kElementIdentity %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kElementIdentity));
  std::printf("enum Gfn2PlanSchemaField::kElementFingerprint %u\n",
              static_cast<unsigned>(Gfn2PlanSchemaField::kElementFingerprint));
}
struct Storage {
  std::array<I, 5> atom{0, 0, 2, 3, 5}, shell{0, 0, 3, 4, 6};
  std::array<I, 5> orbital{0, 0, 4, 5, 9}, matrix{0, 0, 16, 17, 33};
  std::array<I, 6> atom_shell{0, 1, 3, 4, 5, 6};
  std::array<I, 7> shell_orbital{0, 1, 2, 4, 5, 6, 9};
  std::array<I, 6> shell_atom{0, 1, 1, 2, 3, 4};
  std::array<I, 9> orbital_shell{0, 1, 2, 2, 3, 4, 5, 5, 5};
  std::array<I, 9> orbital_atom{0, 1, 1, 1, 2, 3, 4, 4, 4};
  std::array<I, 5> pair{0, 0, 1, 1, 2};
  std::array<Gfn2AtomPair, 3> pairs{{{0, 1}, {3, 4}, {3, 4}}};
  std::array<I, 4> bucket{0, 1, 2, 4};
  std::array<std::int32_t, 4> systems{0, 2, 1, 3};
  std::array<std::int32_t, 3> sizes{0, 1, 4};
  T view(Gfn2PairMapKind kind = Gfn2PairMapKind::kPackedLowerTriangle) const {
    T t{};
    t.plan_token = 0xfedcba9876543210ull;
    t.batch_size = 4;
    t.total_atoms = 5;
    t.total_shells = 6;
    t.total_orbitals = 9;
    t.total_matrix_elements = 33;
    t.atom_offset_count = t.batch_shell_offset_count = t.batch_orbital_offset_count = 5;
    t.matrix_offset_count = 5;
    t.atom_shell_offset_count = 6;
    t.shell_orbital_offset_count = 7;
    t.shell_to_atom_count = 6;
    t.orbital_to_shell_count = t.orbital_to_atom_count = 9;
    t.atom_offsets = atom.data();
    t.batch_shell_offsets = shell.data();
    t.batch_orbital_offsets = orbital.data();
    t.matrix_offsets = matrix.data();
    t.atom_shell_offsets = atom_shell.data();
    t.shell_orbital_offsets = shell_orbital.data();
    t.shell_to_atom = shell_atom.data();
    t.orbital_to_shell = orbital_shell.data();
    t.orbital_to_atom = orbital_atom.data();
    t.pair_map_kind = kind;
    if (kind != Gfn2PairMapKind::kNone) {
      t.total_pairs = 2;
      t.pair_offset_count = 5;
      t.pair_offsets = pair.data();
      if (kind == Gfn2PairMapKind::kExplicit) {
        t.atom_pair_count = 2;
        t.atom_pairs = pairs.data();
      }
    }
    t.bucket_count = 3;
    t.bucket_offset_count = 4;
    t.bucket_system_count = 4;
    t.bucket_orbital_count = 3;
    t.bucket_offsets = bucket.data();
    t.bucket_systems = systems.data();
    t.bucket_orbital_counts = sizes.data();
    return t;
  }
};

// Success must borrow the exact original arrays; every named scalar is copied.
// Mutation coverage is field-by-field, including pointers with valid contents
// but different identity, and records diagnostic precedence from the oracle.

void check_atom(const T& t, bool mutations) {
  auto out = poisoned_Gfn2AtomProjectionView();
  const auto d = invoke("project_atom", [&] { return project_gfn2_atom_projection_host(t, out); });
  Gfn2AtomProjectionView expected{};
  if (d.error == E::kSuccess) {
#define COPY(f) expected.f = t.f;
    FIELDS_Gfn2AtomProjectionView(COPY)
#undef COPY
  }
  require(equal(out, expected), "atom pointer/count/token identity or failure clearing");
  if (d.error == E::kSuccess) {
    require(invoke(
                "binding_atom",
                [&] {
                  return validate_gfn2_atom_projection_binding(t, out, M::kHost);
                }).error == E::kSuccess,
            "valid atom projection");
    if (mutations) {
#define MUTATE(f)                                                                     \
  {                                                                                   \
    auto changed = out;                                                               \
    corrupt(changed.f);                                                               \
    require(invoke(                                                                   \
                "atom." #f,                                                           \
                [&] {                                                                 \
                  return validate_gfn2_atom_projection_binding(t, changed, M::kHost); \
                }).error != E::kSuccess,                                              \
            "mutated atom." #f);                                                      \
  }
      FIELDS_Gfn2AtomProjectionView(MUTATE)
#undef MUTATE
    }
  }
}

void check_shell_ownership(const T& t, bool mutations) {
  auto out = poisoned_Gfn2ShellOwnershipProjectionView();
  const auto d = invoke("project_shell_ownership",
                        [&] { return project_gfn2_shell_ownership_projection_host(t, out); });
  Gfn2ShellOwnershipProjectionView expected{};
  if (d.error == E::kSuccess) {
#define COPY(f) expected.f = t.f;
    FIELDS_Gfn2ShellOwnershipProjectionView(COPY)
#undef COPY
  }
  require(equal(out, expected), "shell_ownership pointer/count/token identity or failure clearing");
  if (d.error == E::kSuccess) {
    require(invoke(
                "binding_shell_ownership",
                [&] {
                  return validate_gfn2_shell_ownership_projection_binding(t, out, M::kHost);
                }).error == E::kSuccess,
            "valid shell_ownership projection");
    if (mutations) {
#define MUTATE(f)                                                                                \
  {                                                                                              \
    auto changed = out;                                                                          \
    corrupt(changed.f);                                                                          \
    require(invoke(                                                                              \
                "shell_ownership." #f,                                                           \
                [&] {                                                                            \
                  return validate_gfn2_shell_ownership_projection_binding(t, changed, M::kHost); \
                }).error != E::kSuccess,                                                         \
            "mutated shell_ownership." #f);                                                      \
  }
      FIELDS_Gfn2ShellOwnershipProjectionView(MUTATE)
#undef MUTATE
    }
  }
}

void check_ao_matrix(const T& t, bool mutations) {
  auto out = poisoned_Gfn2AOMatrixProjectionView();
  const auto d =
      invoke("project_ao_matrix", [&] { return project_gfn2_ao_matrix_projection_host(t, out); });
  Gfn2AOMatrixProjectionView expected{};
  if (d.error == E::kSuccess) {
#define COPY(f) expected.f = t.f;
    FIELDS_Gfn2AOMatrixProjectionView(COPY)
#undef COPY
  }
  require(equal(out, expected), "ao_matrix pointer/count/token identity or failure clearing");
  if (d.error == E::kSuccess) {
    require(invoke(
                "binding_ao_matrix",
                [&] {
                  return validate_gfn2_ao_matrix_projection_binding(t, out, M::kHost);
                }).error == E::kSuccess,
            "valid ao_matrix projection");
    if (mutations) {
#define MUTATE(f)                                                                          \
  {                                                                                        \
    auto changed = out;                                                                    \
    corrupt(changed.f);                                                                    \
    require(invoke(                                                                        \
                "ao_matrix." #f,                                                           \
                [&] {                                                                      \
                  return validate_gfn2_ao_matrix_projection_binding(t, changed, M::kHost); \
                }).error != E::kSuccess,                                                   \
            "mutated ao_matrix." #f);                                                      \
  }
      FIELDS_Gfn2AOMatrixProjectionView(MUTATE)
#undef MUTATE
    }
  }
}

void check_packed_all_pair(const T& t, bool mutations) {
  auto out = poisoned_Gfn2PackedAllPairProjectionView();
  const auto d = invoke("project_packed_all_pair",
                        [&] { return project_gfn2_packed_all_pair_projection_host(t, out); });
  Gfn2PackedAllPairProjectionView expected{};
  if (d.error == E::kSuccess) {
#define COPY(f) expected.f = t.f;
    FIELDS_Gfn2PackedAllPairProjectionView(COPY)
#undef COPY
  }
  require(equal(out, expected), "packed_all_pair pointer/count/token identity or failure clearing");
  if (d.error == E::kSuccess) {
    require(invoke(
                "binding_packed_all_pair",
                [&] {
                  return validate_gfn2_packed_all_pair_projection_binding(t, out, M::kHost);
                }).error == E::kSuccess,
            "valid packed_all_pair projection");
    if (mutations) {
#define MUTATE(f)                                                                                \
  {                                                                                              \
    auto changed = out;                                                                          \
    corrupt(changed.f);                                                                          \
    require(invoke(                                                                              \
                "packed_all_pair." #f,                                                           \
                [&] {                                                                            \
                  return validate_gfn2_packed_all_pair_projection_binding(t, changed, M::kHost); \
                }).error != E::kSuccess,                                                         \
            "mutated packed_all_pair." #f);                                                      \
  }
      FIELDS_Gfn2PackedAllPairProjectionView(MUTATE)
#undef MUTATE
    }
  }
}

void check_ao_bucket(const T& t, bool mutations) {
  auto out = poisoned_Gfn2AOBucketProjectionView();
  const auto d =
      invoke("project_ao_bucket", [&] { return project_gfn2_ao_bucket_projection_host(t, out); });
  Gfn2AOBucketProjectionView expected{};
  if (d.error == E::kSuccess) {
#define COPY(f) expected.f = t.f;
    FIELDS_Gfn2AOBucketProjectionView(COPY)
#undef COPY
  }
  require(equal(out, expected), "ao_bucket pointer/count/token identity or failure clearing");
  if (d.error == E::kSuccess) {
    require(invoke(
                "binding_ao_bucket",
                [&] {
                  return validate_gfn2_ao_bucket_projection_binding(t, out, M::kHost);
                }).error == E::kSuccess,
            "valid ao_bucket projection");
    if (mutations) {
#define MUTATE(f)                                                                          \
  {                                                                                        \
    auto changed = out;                                                                    \
    corrupt(changed.f);                                                                    \
    require(invoke(                                                                        \
                "ao_bucket." #f,                                                           \
                [&] {                                                                      \
                  return validate_gfn2_ao_bucket_projection_binding(t, changed, M::kHost); \
                }).error != E::kSuccess,                                                   \
            "mutated ao_bucket." #f);                                                      \
  }
      FIELDS_Gfn2AOBucketProjectionView(MUTATE)
#undef MUTATE
    }
  }
}

void topology(const char* name, const T& t, bool mutations = false) {
  std::printf("case %s\n", name);
  const T before = t;
  invoke("binding", [&] { return validate_gfn2_topology_binding(t, M::kHost); });
  const auto d = invoke("host", [&] { return validate_gfn2_topology_host(t); });
  auto out = poisoned_Gfn2RaggedTopologyView();
  const auto bound = invoke("bind", [&] { return bind_gfn2_topology_host(t, out); });
  require(bound.error == d.error && bound.field == d.field && bound.index == d.index,
          "binder changed diagnostic");
  require(equal(out, d.error == E::kSuccess ? t : T{}), "binder identity or failure clearing");
  check_atom(t, mutations);
  check_shell_ownership(t, mutations);
  check_ao_matrix(t, mutations);
  check_packed_all_pair(t, mutations);
  check_ao_bucket(t, mutations);
  require(equal(t, before), "input topology changed");
}

void host_cases() {
  Storage s;
  for (auto kind : {Gfn2PairMapKind::kNone, Gfn2PairMapKind::kPackedLowerTriangle,
                    Gfn2PairMapKind::kExplicit}) {
    auto t = s.view(kind);
    require(no_alloc([&] { return validate_gfn2_topology_host(t); }).error == E::kSuccess,
            "ragged fixture must be independently valid");
    topology("ragged", t, true);
  }
  auto t = s.view();
  t.bucket_count = t.bucket_offset_count = t.bucket_system_count = t.bucket_orbital_count = 0;
  t.bucket_offsets = nullptr;
  t.bucket_systems = t.bucket_orbital_counts = nullptr;
  topology("no-buckets", t, true);

  // A one-system empty topology still owns distinct one-entry partition arrays.
  std::array<I, 2> a{}, b{}, c{}, d{}, pair{};
  I atom_shell = 0, shell_orbital = 0;
  T empty{};
  empty.plan_token = 7;
  empty.batch_size = 1;
  empty.atom_offset_count = empty.batch_shell_offset_count = 2;
  empty.batch_orbital_offset_count = empty.matrix_offset_count = 2;
  empty.atom_shell_offset_count = empty.shell_orbital_offset_count = 1;
  empty.atom_offsets = a.data();
  empty.batch_shell_offsets = b.data();
  empty.batch_orbital_offsets = c.data();
  empty.matrix_offsets = d.data();
  empty.atom_shell_offsets = &atom_shell;
  empty.shell_orbital_offsets = &shell_orbital;
  topology("empty-none", empty, true);
  empty.pair_map_kind = Gfn2PairMapKind::kPackedLowerTriangle;
  empty.pair_offset_count = 2;
  empty.pair_offsets = pair.data();
  topology("empty-packed", empty, true);
  empty.pair_map_kind = Gfn2PairMapKind::kExplicit;
  topology("empty-explicit", empty, true);
#define NULLABLE(f)                                    \
  {                                                    \
    auto bad = empty;                                  \
    bad.f = reinterpret_cast<decltype(bad.f)>(0x1000); \
    topology("zero-count-nonnull-" #f, bad);           \
  }
  NULLABLE(shell_to_atom)
  NULLABLE(orbital_to_shell)
  NULLABLE(orbital_to_atom)
  NULLABLE(atom_pairs)
  NULLABLE(bucket_offsets)
  NULLABLE(bucket_systems)
  NULLABLE(bucket_orbital_counts)
#undef NULLABLE
  empty.pair_map_kind = Gfn2PairMapKind::kNone;
  empty.pair_offset_count = 0;
  empty.pair_offsets = nullptr;
  auto forbidden_pairs = empty;
  forbidden_pairs.pair_offsets = pair.data();
  topology("zero-count-nonnull-pair-offsets", forbidden_pairs);

  t = s.view();
  invoke("same-object-bind", [&] { return bind_gfn2_topology_host(t, t); });
  require(equal(t, T{}), "same-object bind must retain frozen fail-closed behavior");

  t = s.view();
  t.plan_token = 0;
  topology("zero-token", t);
  t = s.view();
  t.plan_token = 1;
  topology("one-token", t);
  t = s.view();
  t.plan_token = ~std::uint64_t{0};
  topology("maximum-token", t);
  for (auto space : {M::kCudaDevice, M::kHipDevice, static_cast<M>(99)}) {
    t = s.view();
    t.memory_space = space;
    topology("wrong-host-space", t);
  }
  t = s.view();
  t.pair_map_kind = static_cast<Gfn2PairMapKind>(99);
  topology("unknown-pair-kind", t);

#define SCALAR(f)                          \
  for (I value : {I{-1}, I{0}, maximum}) { \
    t = s.view();                          \
    t.f = value;                           \
    topology(#f, t);                       \
  }
  SCALAR(batch_size)
  SCALAR(total_atoms)
  SCALAR(total_shells)
  SCALAR(total_orbitals)
  SCALAR(total_matrix_elements)
  SCALAR(total_pairs)
  SCALAR(bucket_count)
  SCALAR(atom_offset_count)
  SCALAR(batch_shell_offset_count)
  SCALAR(batch_orbital_offset_count)
  SCALAR(matrix_offset_count)
  SCALAR(atom_shell_offset_count)
  SCALAR(shell_orbital_offset_count)
  SCALAR(shell_to_atom_count)
  SCALAR(orbital_to_shell_count)
  SCALAR(orbital_to_atom_count)
  SCALAR(pair_offset_count)
  SCALAR(atom_pair_count)
  SCALAR(bucket_offset_count)
  SCALAR(bucket_system_count)
  SCALAR(bucket_orbital_count)
#undef SCALAR

#define POINTER(f)                                                                               \
  for (std::uintptr_t address :                                                                  \
       {std::uintptr_t{0}, std::uintptr_t{1}, std::numeric_limits<std::uintptr_t>::max() - 7}) { \
    t = s.view(Gfn2PairMapKind::kExplicit);                                                      \
    t.f = reinterpret_cast<decltype(t.f)>(address);                                              \
    topology(#f, t);                                                                             \
  }
  POINTER(atom_offsets)
  POINTER(batch_shell_offsets)
  POINTER(batch_orbital_offsets)
  POINTER(matrix_offsets)
  POINTER(atom_shell_offsets)
  POINTER(shell_orbital_offsets)
  POINTER(shell_to_atom)
  POINTER(orbital_to_shell)
  POINTER(orbital_to_atom)
  POINTER(pair_offsets)
  POINTER(atom_pairs)
  POINTER(bucket_offsets)
  POINTER(bucket_systems)
  POINTER(bucket_orbital_counts)
#undef POINTER

  // Error precedence is observable: memory, token, scalar counts, then arrays.
  t = s.view();
  t.memory_space = static_cast<M>(99);
  t.plan_token = 0;
  t.batch_size = -1;
  t.atom_offsets = nullptr;
  topology("precedence-memory", t);
  t.memory_space = M::kHost;
  topology("precedence-token", t);
  t.plan_token = 1;
  topology("precedence-count", t);
  t.batch_size = 4;
  topology("precedence-pointer", t);

#define ARRAY(field)                                                                \
  for (std::size_t i = 0; i < s.field.size(); ++i) {                                \
    for (I value : {I{-1}, I{0}, maximum}) {                                        \
      Storage changed = s;                                                          \
      changed.field[i] = static_cast<decltype(changed.field)::value_type>(value);   \
      std::printf("array " #field " %zu %lld\n", i, static_cast<long long>(value)); \
      topology("array-mutation", changed.view());                                   \
    }                                                                               \
  }
  ARRAY(atom);
  ARRAY(shell);
  ARRAY(orbital);
  ARRAY(matrix);
  ARRAY(atom_shell);
  ARRAY(shell_orbital);
  ARRAY(shell_atom);
  ARRAY(orbital_shell);
  ARRAY(orbital_atom);
  ARRAY(pair);
  ARRAY(bucket);
  ARRAY(systems);
  ARRAY(sizes);
#undef ARRAY

  for (std::size_t i = 0; i < 2; ++i) {
    for (I value : {I{-1}, I{0}, I{1}, I{4}, maximum}) {
      Storage changed = s;
      changed.pairs[i].first = value;
      topology("explicit-first", changed.view(Gfn2PairMapKind::kExplicit));
      changed = s;
      changed.pairs[i].second = value;
      topology("explicit-second", changed.view(Gfn2PairMapKind::kExplicit));
    }
  }
  Storage duplicate = s;
  duplicate.pair = {0, 0, 2, 2, 3};
  duplicate.pairs = {{{0, 1}, {0, 1}, {3, 4}}};
  t = duplicate.view(Gfn2PairMapKind::kExplicit);
  t.total_pairs = t.atom_pair_count = 3;
  topology("duplicate-explicit-pair", t);

  Storage sparse = s;
  sparse.pair.fill(0);
  t = sparse.view(Gfn2PairMapKind::kExplicit);
  t.total_pairs = t.atom_pair_count = 0;
  t.atom_pairs = nullptr;
  topology("empty-explicit-pair-list", t);

  // Repack all five atoms into one system so first-major and second-major
  // ordering differ. The sparse pair list is not a physical cutoff policy.
  Storage ordered = s;
  ordered.atom = {0, 5, 5, 5, 5};
  ordered.shell = {0, 6, 6, 6, 6};
  ordered.orbital = {0, 9, 9, 9, 9};
  ordered.matrix = {0, 81, 81, 81, 81};
  ordered.pair = {0, 3, 3, 3, 3};
  ordered.pairs = {{{0, 2}, {0, 3}, {1, 2}}};
  t = ordered.view(Gfn2PairMapKind::kExplicit);
  t.total_matrix_elements = 81;
  t.total_pairs = t.atom_pair_count = 3;
  t.bucket_count = t.bucket_offset_count = t.bucket_system_count = t.bucket_orbital_count = 0;
  t.bucket_offsets = nullptr;
  t.bucket_systems = t.bucket_orbital_counts = nullptr;
  require(no_alloc([&] { return validate_gfn2_topology_host(t); }).error == E::kSuccess,
          "first-major sparse pair order");
  topology("first-major-pairs", t);
  std::swap(ordered.pairs[1], ordered.pairs[2]);
  require(no_alloc([&] { return validate_gfn2_topology_host(t); }).error == E::kInvalidPairMap,
          "second-major sparse pair order must be rejected");
  topology("second-major-pairs", t);
}

// A legal overlap through an int64 scalar subobject makes clearing order
// observable without treating unrelated complete descriptor types as aliases.
void descriptor_overlap_cases() {
  std::array<I, 2> a{0, 1}, b{0, 1}, c{0, 1}, d{0, 1};
  std::array<I, 2> atom_shell{0, 1}, shell_orbital{0, 1}, pairs{0, 0};
  I orbital_shell = 0, orbital_atom = 0;
  T t{};
  t.plan_token = 37;
  t.batch_size = t.total_atoms = t.total_shells = t.total_orbitals = 1;
  t.total_matrix_elements = 1;
  t.atom_offset_count = t.batch_shell_offset_count = t.batch_orbital_offset_count = 2;
  t.matrix_offset_count = t.atom_shell_offset_count = t.shell_orbital_offset_count = 2;
  t.shell_to_atom_count = t.orbital_to_shell_count = t.orbital_to_atom_count = 1;
  t.atom_offsets = a.data();
  t.batch_shell_offsets = b.data();
  t.batch_orbital_offsets = c.data();
  t.matrix_offsets = d.data();
  t.atom_shell_offsets = atom_shell.data();
  t.shell_orbital_offsets = shell_orbital.data();
  t.orbital_to_shell = &orbital_shell;
  t.orbital_to_atom = &orbital_atom;
  t.pair_map_kind = Gfn2PairMapKind::kPackedLowerTriangle;
  t.pair_offset_count = 2;
  t.pair_offsets = pairs.data();
  {
    Gfn2AtomProjectionView out{};
    out.batch_size = 99;
    t.shell_to_atom = &out.batch_size;
    require(no_alloc([&] { return validate_gfn2_topology_host(t); }).error == E::kInvalidShellMap,
            "overlap fixture before clearing");
    require(
        invoke("overlap-atom", [&] { return project_gfn2_atom_projection_host(t, out); }).error ==
            E::kSuccess,
        "output must be cleared before topology inspection");
    Gfn2AtomProjectionView expected{};
#define COPY(f) expected.f = t.f;
    FIELDS_Gfn2AtomProjectionView(COPY)
#undef COPY
        require(equal(out, expected), "overlapping projection preserves every member");
  }
  {
    Gfn2ShellOwnershipProjectionView out{};
    out.batch_size = 99;
    t.shell_to_atom = &out.batch_size;
    require(no_alloc([&] { return validate_gfn2_topology_host(t); }).error == E::kInvalidShellMap,
            "overlap fixture before clearing");
    require(invoke(
                "overlap-shell_ownership",
                [&] {
                  return project_gfn2_shell_ownership_projection_host(t, out);
                }).error == E::kSuccess,
            "output must be cleared before topology inspection");
    Gfn2ShellOwnershipProjectionView expected{};
#define COPY(f) expected.f = t.f;
    FIELDS_Gfn2ShellOwnershipProjectionView(COPY)
#undef COPY
        require(equal(out, expected), "overlapping projection preserves every member");
  }
  {
    Gfn2AOMatrixProjectionView out{};
    out.batch_size = 99;
    t.shell_to_atom = &out.batch_size;
    require(no_alloc([&] { return validate_gfn2_topology_host(t); }).error == E::kInvalidShellMap,
            "overlap fixture before clearing");
    require(invoke(
                "overlap-ao_matrix",
                [&] {
                  return project_gfn2_ao_matrix_projection_host(t, out);
                }).error == E::kSuccess,
            "output must be cleared before topology inspection");
    Gfn2AOMatrixProjectionView expected{};
#define COPY(f) expected.f = t.f;
    FIELDS_Gfn2AOMatrixProjectionView(COPY)
#undef COPY
        require(equal(out, expected), "overlapping projection preserves every member");
  }
  {
    Gfn2PackedAllPairProjectionView out{};
    out.batch_size = 99;
    t.shell_to_atom = &out.batch_size;
    require(no_alloc([&] { return validate_gfn2_topology_host(t); }).error == E::kInvalidShellMap,
            "overlap fixture before clearing");
    require(invoke(
                "overlap-packed_all_pair",
                [&] {
                  return project_gfn2_packed_all_pair_projection_host(t, out);
                }).error == E::kSuccess,
            "output must be cleared before topology inspection");
    Gfn2PackedAllPairProjectionView expected{};
#define COPY(f) expected.f = t.f;
    FIELDS_Gfn2PackedAllPairProjectionView(COPY)
#undef COPY
        require(equal(out, expected), "overlapping projection preserves every member");
  }
  {
    Gfn2AOBucketProjectionView out{};
    out.batch_size = 99;
    t.shell_to_atom = &out.batch_size;
    require(no_alloc([&] { return validate_gfn2_topology_host(t); }).error == E::kInvalidShellMap,
            "overlap fixture before clearing");
    require(invoke(
                "overlap-ao_bucket",
                [&] {
                  return project_gfn2_ao_bucket_projection_host(t, out);
                }).error == E::kSuccess,
            "output must be cleared before topology inspection");
    Gfn2AOBucketProjectionView expected{};
#define COPY(f) expected.f = t.f;
    FIELDS_Gfn2AOBucketProjectionView(COPY)
#undef COPY
        require(equal(out, expected), "overlapping projection preserves every member");
  }
  {
    T out{};
    out.batch_size = 99;
    t.shell_to_atom = &out.batch_size;
    require(invoke("overlap-bind", [&] { return bind_gfn2_topology_host(t, out); }).error ==
                E::kSuccess,
            "binder clearing before topology inspection");
    require(equal(out, t), "overlapping binder exact copy");
  }
}

struct Inaccessible {
  std::size_t stride = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  void* base = mmap(nullptr, 16 * stride, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  Inaccessible() { require(base != MAP_FAILED, "PROT_NONE device sentinel reservation"); }
  ~Inaccessible() { munmap(base, 16 * stride); }
  std::uintptr_t at(std::size_t index) const {
    return reinterpret_cast<std::uintptr_t>(base) + index * stride;
  }
};

struct PointerField {
  const char* name;
  F field;
  void (*assign)(T&, std::uintptr_t);
};
const PointerField pointer_fields[] = {
    {"atom_offsets", F::kAtomOffsets,
     [](T& t, std::uintptr_t p) {
       t.atom_offsets = reinterpret_cast<decltype(t.atom_offsets)>(p);
     }},
    {"batch_shell_offsets", F::kBatchShellOffsets,
     [](T& t, std::uintptr_t p) {
       t.batch_shell_offsets = reinterpret_cast<decltype(t.batch_shell_offsets)>(p);
     }},
    {"batch_orbital_offsets", F::kBatchOrbitalOffsets,
     [](T& t, std::uintptr_t p) {
       t.batch_orbital_offsets = reinterpret_cast<decltype(t.batch_orbital_offsets)>(p);
     }},
    {"matrix_offsets", F::kMatrixOffsets,
     [](T& t, std::uintptr_t p) {
       t.matrix_offsets = reinterpret_cast<decltype(t.matrix_offsets)>(p);
     }},
    {"atom_shell_offsets", F::kAtomShellOffsets,
     [](T& t, std::uintptr_t p) {
       t.atom_shell_offsets = reinterpret_cast<decltype(t.atom_shell_offsets)>(p);
     }},
    {"shell_orbital_offsets", F::kShellOrbitalOffsets,
     [](T& t, std::uintptr_t p) {
       t.shell_orbital_offsets = reinterpret_cast<decltype(t.shell_orbital_offsets)>(p);
     }},
    {"shell_to_atom", F::kShellToAtom,
     [](T& t, std::uintptr_t p) {
       t.shell_to_atom = reinterpret_cast<decltype(t.shell_to_atom)>(p);
     }},
    {"orbital_to_shell", F::kOrbitalToShell,
     [](T& t, std::uintptr_t p) {
       t.orbital_to_shell = reinterpret_cast<decltype(t.orbital_to_shell)>(p);
     }},
    {"orbital_to_atom", F::kOrbitalToAtom,
     [](T& t, std::uintptr_t p) {
       t.orbital_to_atom = reinterpret_cast<decltype(t.orbital_to_atom)>(p);
     }},
    {"pair_offsets", F::kPairOffsets,
     [](T& t, std::uintptr_t p) {
       t.pair_offsets = reinterpret_cast<decltype(t.pair_offsets)>(p);
     }},
    {"atom_pairs", F::kAtomPairs,
     [](T& t, std::uintptr_t p) { t.atom_pairs = reinterpret_cast<decltype(t.atom_pairs)>(p); }},
    {"bucket_offsets", F::kBucketOffsets,
     [](T& t, std::uintptr_t p) {
       t.bucket_offsets = reinterpret_cast<decltype(t.bucket_offsets)>(p);
     }},
    {"bucket_systems", F::kBucketSystems,
     [](T& t, std::uintptr_t p) {
       t.bucket_systems = reinterpret_cast<decltype(t.bucket_systems)>(p);
     }},
    {"bucket_orbital_counts", F::kBucketOrbitalCounts,
     [](T& t, std::uintptr_t p) {
       t.bucket_orbital_counts = reinterpret_cast<decltype(t.bucket_orbital_counts)>(p);
     }},
};
void device_cases() {
  Inaccessible memory;
  Storage storage;
  for (auto space : {M::kCudaDevice, M::kHipDevice}) {
    auto t = storage.view(Gfn2PairMapKind::kExplicit);
    t.memory_space = space;
    for (std::size_t i = 0; i < std::size(pointer_fields); ++i)
      pointer_fields[i].assign(t, memory.at(i));
    require(
        invoke("device-topology", [&] { return validate_gfn2_topology_binding(t, space); }).error ==
            E::kSuccess,
        "structural validator dereferenced device pointers");
    for (std::size_t first = 0; first < std::size(pointer_fields); ++first) {
      for (std::size_t second = first + 1; second < std::size(pointer_fields); ++second) {
        auto alias = t;
        pointer_fields[second].assign(alias, memory.at(first));
        std::printf("alias %s %s\n", pointer_fields[first].name, pointer_fields[second].name);
        const auto d =
            invoke("device-alias", [&] { return validate_gfn2_topology_binding(alias, space); });
        require(
            d.error == E::kAliasedRange && d.field == pointer_fields[second].field && d.index == -1,
            "first aliased field changed");
      }
    }
    auto multiple = t;
    pointer_fields[7].assign(multiple, memory.at(0));
    pointer_fields[2].assign(multiple, memory.at(1));
    const auto first =
        invoke("alias-precedence", [&] { return validate_gfn2_topology_binding(multiple, space); });
    require(first.field == F::kOrbitalToShell, "pairwise first-conflict ordering changed");

    // Packed projections require no explicit pairs; all other array addresses
    // stay inside inaccessible pages throughout every binding call.
    t.pair_map_kind = Gfn2PairMapKind::kPackedLowerTriangle;
    t.atom_pair_count = 0;
    t.atom_pairs = nullptr;
    {
      Gfn2AtomProjectionView projection{};
#define COPY(f) projection.f = t.f;
      FIELDS_Gfn2AtomProjectionView(COPY)
#undef COPY
          require(invoke(
                      "device-atom",
                      [&] {
                        return validate_gfn2_atom_projection_binding(t, projection, space);
                      }).error == E::kSuccess,
                  "device atom validation");
#define MUTATE(f)                                                                  \
  {                                                                                \
    auto changed = projection;                                                     \
    corrupt(changed.f);                                                            \
    require(invoke(                                                                \
                "device-atom." #f,                                                 \
                [&] {                                                              \
                  return validate_gfn2_atom_projection_binding(t, changed, space); \
                }).error != E::kSuccess,                                           \
            "device projection mutation accepted");                                \
  }
      FIELDS_Gfn2AtomProjectionView(MUTATE)
#undef MUTATE
    }
    {
      Gfn2ShellOwnershipProjectionView projection{};
#define COPY(f) projection.f = t.f;
      FIELDS_Gfn2ShellOwnershipProjectionView(COPY)
#undef COPY
          require(invoke(
                      "device-shell_ownership",
                      [&] {
                        return validate_gfn2_shell_ownership_projection_binding(t, projection,
                                                                                space);
                      }).error == E::kSuccess,
                  "device shell_ownership validation");
#define MUTATE(f)                                                                             \
  {                                                                                           \
    auto changed = projection;                                                                \
    corrupt(changed.f);                                                                       \
    require(invoke(                                                                           \
                "device-shell_ownership." #f,                                                 \
                [&] {                                                                         \
                  return validate_gfn2_shell_ownership_projection_binding(t, changed, space); \
                }).error != E::kSuccess,                                                      \
            "device projection mutation accepted");                                           \
  }
      FIELDS_Gfn2ShellOwnershipProjectionView(MUTATE)
#undef MUTATE
    }
    {
      Gfn2AOMatrixProjectionView projection{};
#define COPY(f) projection.f = t.f;
      FIELDS_Gfn2AOMatrixProjectionView(COPY)
#undef COPY
          require(invoke(
                      "device-ao_matrix",
                      [&] {
                        return validate_gfn2_ao_matrix_projection_binding(t, projection, space);
                      }).error == E::kSuccess,
                  "device ao_matrix validation");
#define MUTATE(f)                                                                       \
  {                                                                                     \
    auto changed = projection;                                                          \
    corrupt(changed.f);                                                                 \
    require(invoke(                                                                     \
                "device-ao_matrix." #f,                                                 \
                [&] {                                                                   \
                  return validate_gfn2_ao_matrix_projection_binding(t, changed, space); \
                }).error != E::kSuccess,                                                \
            "device projection mutation accepted");                                     \
  }
      FIELDS_Gfn2AOMatrixProjectionView(MUTATE)
#undef MUTATE
    }
    {
      Gfn2PackedAllPairProjectionView projection{};
#define COPY(f) projection.f = t.f;
      FIELDS_Gfn2PackedAllPairProjectionView(COPY)
#undef COPY
          require(invoke(
                      "device-packed_all_pair",
                      [&] {
                        return validate_gfn2_packed_all_pair_projection_binding(t, projection,
                                                                                space);
                      }).error == E::kSuccess,
                  "device packed_all_pair validation");
#define MUTATE(f)                                                                             \
  {                                                                                           \
    auto changed = projection;                                                                \
    corrupt(changed.f);                                                                       \
    require(invoke(                                                                           \
                "device-packed_all_pair." #f,                                                 \
                [&] {                                                                         \
                  return validate_gfn2_packed_all_pair_projection_binding(t, changed, space); \
                }).error != E::kSuccess,                                                      \
            "device projection mutation accepted");                                           \
  }
      FIELDS_Gfn2PackedAllPairProjectionView(MUTATE)
#undef MUTATE
    }
    {
      Gfn2AOBucketProjectionView projection{};
#define COPY(f) projection.f = t.f;
      FIELDS_Gfn2AOBucketProjectionView(COPY)
#undef COPY
          require(invoke(
                      "device-ao_bucket",
                      [&] {
                        return validate_gfn2_ao_bucket_projection_binding(t, projection, space);
                      }).error == E::kSuccess,
                  "device ao_bucket validation");
#define MUTATE(f)                                                                       \
  {                                                                                     \
    auto changed = projection;                                                          \
    corrupt(changed.f);                                                                 \
    require(invoke(                                                                     \
                "device-ao_bucket." #f,                                                 \
                [&] {                                                                   \
                  return validate_gfn2_ao_bucket_projection_binding(t, changed, space); \
                }).error != E::kSuccess,                                                \
            "device projection mutation accepted");                                     \
  }
      FIELDS_Gfn2AOBucketProjectionView(MUTATE)
#undef MUTATE
    }

    Gfn2ElementIdentityProjectionView element{};
    element.memory_space = space;
    element.plan_token = t.plan_token;
    element.total_atoms = element.atomic_number_count = 5;
    element.element_fingerprint = 0xf00dcafe;
    element.atomic_numbers = reinterpret_cast<const std::int32_t*>(memory.at(14));
    require(invoke(
                "device-element",
                [&] {
                  return validate_gfn2_element_identity_projection_binding(element, space);
                }).error == E::kSuccess,
            "element device seal must not dereference");
    require(no_alloc([&] { return gfn2_element_identity_fingerprint_host(element); }) == 0,
            "host fingerprint must reject device storage without reading it");
    for (std::uintptr_t address :
         {std::uintptr_t{0}, std::uintptr_t{1}, std::numeric_limits<std::uintptr_t>::max() - 3}) {
      auto changed = element;
      changed.atomic_numbers = reinterpret_cast<const std::int32_t*>(address);
      invoke("device-element-pointer",
             [&] { return validate_gfn2_element_identity_projection_binding(changed, space); });
    }
    element.total_atoms = element.atomic_number_count = maximum;
    const auto overflow = invoke("element-count-overflow", [&] {
      return validate_gfn2_element_identity_projection_binding(element, space);
    });
    require(overflow.error == E::kCountOverflow, "element byte count overflow");
    t = storage.view();
    t.memory_space = space;
    t.total_orbitals = t.orbital_to_shell_count = t.orbital_to_atom_count = maximum;
    const auto range = invoke("topology-byte-count-overflow",
                              [&] { return validate_gfn2_topology_binding(t, space); });
    require(range.error == E::kCountOverflow && range.field == F::kOrbitalToShell,
            "topology byte count overflow precedence");
  }
}

void element_case(const char* name, const std::int32_t* numbers, I count, std::uint64_t token,
                  bool inspect = true) {
  std::printf("element %s\n", name);
  auto out = poisoned_Gfn2ElementIdentityProjectionView();
  const auto d = invoke("element-project", [&] {
    return project_gfn2_element_identity_projection_host(numbers, count, token, out);
  });
  if (d.error != E::kSuccess) {
    require(equal(out, Gfn2ElementIdentityProjectionView{}), "element failure clearing");
    return;
  }
  require(out.memory_space == M::kHost && out.plan_token == token && out.total_atoms == count &&
              out.atomic_number_count == count && out.atomic_numbers == numbers &&
              out.element_fingerprint != 0,
          "element identity must remain borrowed and exact");
  std::printf("fingerprint %016llx\n", static_cast<unsigned long long>(out.element_fingerprint));
  invoke("element-binding",
         [&] { return validate_gfn2_element_identity_projection_binding(out, M::kHost); });
  if (inspect) {
    require(no_alloc([&] { return gfn2_element_identity_fingerprint_host(out); }) ==
                out.element_fingerprint,
            "recomputed fingerprint");
    auto copied = out;
    copied.element_fingerprint ^= 1;
    invoke("corrupt-element-seal",
           [&] { return validate_gfn2_element_identity_projection_binding(copied, M::kHost); });
    copied = out;
    copied.plan_token ^= 1;
    invoke("cross-plan-element",
           [&] { return validate_gfn2_element_identity_projection_binding(copied, M::kHost); });
    copied = out;
    copied.atomic_number_count += 1;
    invoke("element-count-mismatch",
           [&] { return validate_gfn2_element_identity_projection_binding(copied, M::kHost); });
  }
}
void element_cases() {
  std::array<std::int32_t, 5> numbers{8, 1, 1, 6, 7};
  const auto unchanged = numbers;
  element_case("ordered", numbers.data(), numbers.size(), 0x123456789abcdef0ull);
  require(numbers == unchanged, "element builder modified input");
  auto copy = numbers;
  Gfn2ElementIdentityProjectionView first{}, other{};
  require(no_alloc([&] {
            return project_gfn2_element_identity_projection_host(numbers.data(), 5, 91, first);
          }).error == E::kSuccess,
          "element baseline");
  require(no_alloc([&] {
            return project_gfn2_element_identity_projection_host(copy.data(), 5, 91, other);
          }).error == E::kSuccess,
          "relocated element baseline");
  require(first.element_fingerprint == other.element_fingerprint,
          "element fingerprint must exclude address");
  std::swap(copy[0], copy[1]);
  element_case("permuted", copy.data(), copy.size(), 91);
  require(no_alloc([&] {
            return project_gfn2_element_identity_projection_host(copy.data(), 5, 91, other);
          }).error == E::kSuccess,
          "permuted element baseline");
  require(first.element_fingerprint != other.element_fingerprint, "order-sensitive hash");
  for (std::uint64_t token : {std::uint64_t{0}, std::uint64_t{1}, ~std::uint64_t{0}})
    element_case("token", numbers.data(), numbers.size(), token);
  element_case("empty", nullptr, 0, 17);
  element_case("empty-nonnull", numbers.data(), 0, 17);
  element_case("negative", numbers.data(), -1, 17);
  element_case("null", nullptr, 5, 17);
  element_case("misaligned", reinterpret_cast<const std::int32_t*>(1), 5, 17);
  element_case(
      "address-overflow",
      reinterpret_cast<const std::int32_t*>(std::numeric_limits<std::uintptr_t>::max() - 3), 5, 17);
  element_case("byte-overflow", numbers.data(), maximum, 17);
  numbers = {0, -1, std::numeric_limits<std::int32_t>::min(),
             std::numeric_limits<std::int32_t>::max(), 119};
  element_case("full-int32-domain", numbers.data(), numbers.size(), 17);
  Gfn2ElementIdentityProjectionView bad{};
  for (auto space : {M::kHost, M::kCudaDevice, M::kHipDevice, static_cast<M>(0)}) {
    bad.memory_space = space;
    invoke("invalid-element-space-token",
           [&] { return validate_gfn2_element_identity_projection_binding(bad, space); });
    std::printf("invalid-fingerprint %llu\n", static_cast<unsigned long long>(no_alloc([&] {
                  return gfn2_element_identity_fingerprint_host(bad);
                })));
  }
}
}  // namespace

int main(int argc, char** argv) {
  require(argc == 2, "one scenario argument required");
  if (std::strcmp(argv[1], "abi") == 0)
    abi();
  else if (std::strcmp(argv[1], "host") == 0) {
    host_cases();
    descriptor_overlap_cases();
  } else if (std::strcmp(argv[1], "device") == 0)
    device_cases();
  else if (std::strcmp(argv[1], "element") == 0)
    element_cases();
  else
    require(false, "unknown scenario");
  require(allocations == 0, "qualification must make zero production allocations");
  std::puts("zero production allocations");
}
