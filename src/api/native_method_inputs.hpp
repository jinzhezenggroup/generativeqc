#pragma once

#include "api/handles.hpp"

namespace generativeqc::api {

// The C ABI deliberately uses signed 32-bit integer tags. Keep its stable
// published values checked at the ingress boundary, never in core/types.hpp.
static_assert(core::kCpuReferenceBackend == GENERATIVEQC_BACKEND_CPU_REFERENCE);
static_assert(core::kCudaBackend == GENERATIVEQC_BACKEND_CUDA);
static_assert(core::kHybridCudaBackend == GENERATIVEQC_BACKEND_HYBRID_CUDA);
static_assert(core::kCartesianBasis == GENERATIVEQC_BASIS_CARTESIAN);
static_assert(core::kSphericalBasis == GENERATIVEQC_BASIS_SPHERICAL);

[[nodiscard]] constexpr core::Backend native_backend(generativeqc_backend backend) noexcept {
  return backend;
}

[[nodiscard]] constexpr core::BasisRepresentation native_basis_representation(
    generativeqc_basis_representation representation) noexcept {
  return representation;
}

// Borrowed only for the synchronous prepare call. Prepared method owners must
// deep-copy an explicitly supplied auxiliary system before returning.
[[nodiscard]] inline const core::System* native_auxiliary_basis(
    const generativeqc_method_descriptor& descriptor) noexcept {
  return descriptor.density_fitting_auxiliary_basis
             ? &descriptor.density_fitting_auxiliary_basis->data
             : nullptr;
}

}  // namespace generativeqc::api
