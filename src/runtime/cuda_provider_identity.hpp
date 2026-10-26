#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "runtime/cuda_provider.hpp"
#include "runtime/loaded_provider_identity_types.hpp"

namespace generativeqc::runtime {

enum class ContextIdentityStatus { available, unavailable, query_error, changed };
enum class IdentityErrorDomain { driver, cuda };

struct CurrentContextIdentity {
  int device{-1};
  std::uint64_t context_id{};
  std::array<unsigned char, 16> device_uuid{};
  int driver_version{}, runtime_version{};
  bool operator==(const CurrentContextIdentity&) const = default;
};

/** HOST seam for the current-context query only. A pointer is transport to
 * cuCtxGetId, never a context generation or identity. No callback may create,
 * select, retain or reset a context. Raw API error domain/status are preserved. */
struct CurrentContextQueries {
  void* state{};
  std::uint32_t (*current)(void*, std::uintptr_t&){};
  std::uint32_t (*id)(void*, std::uintptr_t, std::uint64_t&){};
  std::uint32_t (*device)(void*, int&){};
  std::uint32_t (*uuid)(void*, int, std::array<unsigned char, 16>&){};
  std::uint32_t (*driver_version)(void*, int&){};
  std::uint32_t (*runtime_version)(void*, int&){};
};

struct CurrentContextObservation {
  ContextIdentityStatus status{ContextIdentityStatus::unavailable};
  std::optional<CurrentContextIdentity> identity;
  IdentityErrorDomain error_domain{IdentityErrorDomain::driver};
  std::uint32_t error_status{};
};

inline CurrentContextObservation observe_current_context(const CurrentContextQueries& queries) {
  if (!queries.current || !queries.id || !queries.device || !queries.uuid ||
      !queries.driver_version || !queries.runtime_version)
    return {};
  const auto failure = [](IdentityErrorDomain domain, std::uint32_t status) {
    return CurrentContextObservation{ContextIdentityStatus::query_error, {}, domain, status};
  };
  std::uintptr_t context{};
  auto status = queries.current(queries.state, context);
  if (status) return failure(IdentityErrorDomain::driver, status);
  if (!context) return {};
  CurrentContextIdentity identity;
  status = queries.id(queries.state, context, identity.context_id);
  if (status) return failure(IdentityErrorDomain::driver, status);
  status = queries.device(queries.state, identity.device);
  if (status) return failure(IdentityErrorDomain::driver, status);
  status = queries.uuid(queries.state, identity.device, identity.device_uuid);
  if (status) return failure(IdentityErrorDomain::driver, status);
  status = queries.driver_version(queries.state, identity.driver_version);
  if (status) return failure(IdentityErrorDomain::driver, status);
  status = queries.runtime_version(queries.state, identity.runtime_version);
  if (status) return failure(IdentityErrorDomain::cuda, status);
  std::uintptr_t after_context{};
  status = queries.current(queries.state, after_context);
  if (status) return failure(IdentityErrorDomain::driver, status);
  if (after_context != context)
    return {ContextIdentityStatus::changed, {}, IdentityErrorDomain::driver, 0};
  std::uint64_t after_id{};
  status = queries.id(queries.state, after_context, after_id);
  if (status) return failure(IdentityErrorDomain::driver, status);
  if (after_id != identity.context_id)
    return {ContextIdentityStatus::changed, {}, IdentityErrorDomain::driver, 0};
  if (identity.device < 0 || identity.driver_version <= 0 || identity.runtime_version <= 0)
    return {};
  return {ContextIdentityStatus::available, identity, IdentityErrorDomain::driver, 0};
}

struct CudaProviderIdentity {
  CudaProviderKind provider{CudaProviderKind::None};
  CurrentContextIdentity context;
  std::array<std::optional<std::string>, 4> build_ids;
  std::uint8_t providers{};
  bool operator==(const CudaProviderIdentity&) const = default;
};

struct CudaProviderIdentityObservation {
  ContextIdentityStatus status{ContextIdentityStatus::unavailable};
  std::optional<CudaProviderIdentity> identity;
  IdentityErrorDomain error_domain{IdentityErrorDomain::driver};
  std::uint32_t error_status{};
  std::optional<LoadedIdentityStatus> loaded_status;
  std::array<std::shared_ptr<void>, 4> library_pins;
};

/** Observe already established current NVIDIA context and already loaded direct
 * cudart/BLAS/solver/driver objects. Disabled BLAS/solver bits skip those objects;
 * runtime and driver identities remain mandatory.
 * This is build/context metadata, never a complete transitive allocation bound. */
CudaProviderIdentityObservation observe_cuda_provider_identity(
    std::uint8_t providers, const std::array<const void*, 3>& execution_symbols);

}  // namespace generativeqc::runtime
