#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/df_source_kernels.hpp"
#include "scf/cuda_density_fitting.hpp"

namespace generativeqc::scf {

/** Immutable single-system normalized DF basis, not mutable J/K/SCF scratch.
 * Orbital shells/primitives/components precede auxiliary ones. Public expansion
 * Cartesian indices are local to their respective basis, while component
 * normalization uses the combined orbital+auxiliary Cartesian array. The
 * producer's dummy constant shell is deliberately excluded from all counts.
 * Auxiliary g retains the actual six-term-capable public expansion ABI.
 *
 * A live owner must authenticate this entire view and its original stream;
 * obtaining a view alone does not prove a stationary KS state. No geometry,
 * normalization or projection numbers are downloaded or reconstructed on CPU.
 */
struct CudaDfReadonlyBasis {
  int device_id{-1};
  std::size_t atoms{}, primitives{}, orbital_shells{}, auxiliary_shells{}, nao{}, naux{},
      cartesian_nao{}, cartesian_naux{};
  const double *positions{}, *exponents{}, *coefficients{}, *component_coefficients{};
  const std::int32_t* shell_atoms{};
  const std::uint8_t* shell_angular{};
  const std::int64_t *shell_primitive_offsets{}, *shell_cartesian_offsets{};
  const cuda_execution::DfPublicAoExpansion *orbital_expansions{}, *auxiliary_expansions{};
  bool operator==(const CudaDfReadonlyBasis&) const = default;
  explicit operator bool() const noexcept {
    return device_id >= 0 && atoms && primitives && orbital_shells && auxiliary_shells && nao &&
           naux && cartesian_nao && cartesian_naux && positions && exponents && coefficients &&
           component_coefficients && shell_atoms && shell_angular && shell_primitive_offsets &&
           shell_cartesian_offsets && orbital_expansions && auxiliary_expansions;
  }
};

/** Empty for missing, batched or wrong-current-device sources. No allocation,
 * numerical transfer, device switch, compilation or scratch mutation occurs. */
CudaDfReadonlyBasis cuda_density_fitting_readonly_basis(
    const CudaDensityFittingIntegralSource* source) noexcept;

}  // namespace generativeqc::scf
