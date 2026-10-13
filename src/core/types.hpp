#ifndef GENERATIVEQC_CORE_TYPES_HPP
#define GENERATIVEQC_CORE_TYPES_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace generativeqc::core {

// Native numerical tags do not depend on C handle layouts or public headers.
// api/native_method_inputs.hpp verifies them against the versioned ABI.
using Backend = std::int32_t;
inline constexpr Backend kCpuReferenceBackend = 0;
inline constexpr Backend kCudaBackend = 1;
inline constexpr Backend kHybridCudaBackend = 2;

using BasisRepresentation = std::int32_t;
inline constexpr BasisRepresentation kCartesianBasis = 0;
inline constexpr BasisRepresentation kSphericalBasis = 1;

struct Atom {
  int atomic_number{};
  std::array<double, 3> position{};
  int ecp_core{};
  [[nodiscard]] int ionic_charge() const noexcept { return atomic_number - ecp_core; }
};

struct EcpTerm {
  unsigned atom_index{};
  int channel{-1};
  unsigned power{};
  double exponent{}, coefficient{};
  bool operator==(const EcpTerm&) const = default;
};

struct Primitive {
  double exponent{};
  double coefficient{};
};

struct Shell {
  std::uint32_t atom_index{};
  std::uint32_t angular_momentum{};
  std::vector<Primitive> primitives;
};

struct System {
  std::vector<Atom> atoms;
  std::vector<Shell> shells;
  int charge{};
  unsigned multiplicity{1};
  int electron_count{};
  BasisRepresentation basis_representation{kCartesianBasis};
  std::vector<EcpTerm> ecp_terms;
};

/** One context-owned execution workspace, independent of method interfaces.
 *
 * A method may retain at most one workspace here instead of rebuilding its
 * backend for every prepared calculation. Prepared owners also hold a strong
 * reference, so replacement cannot invalidate an outstanding calculation.
 * Acquisition/replacement is protected by the owning API context's mutex;
 * each concrete workspace owns its scientific identity and execution lock.
 */
class ContextWorkspace {
 public:
  virtual ~ContextWorkspace() = default;
};

struct ContextState {
  Backend requested_backend{kCpuReferenceBackend};
  Backend executed_backend{kCpuReferenceBackend};
  int device_id{0};
  std::string device_name;
  int compute_capability_major{};
  int compute_capability_minor{};
  int warp_size{};
  int maximum_threads_per_sm{};
  int maximum_blocks_per_sm{};
  int registers_per_sm{};
  int maximum_registers_per_thread{};
  std::size_t shared_memory_per_block{};
  std::size_t shared_memory_per_sm{};
  int multiprocessor_count{};
  std::string aot_profile_name{"generic_cuda"};
  bool aot_profile_tuned{};
  bool aot_profile_portable{true};
  bool aot_profile_compatible{};
  std::shared_ptr<ContextWorkspace> workspace;
};

}  // namespace generativeqc::core

#endif
