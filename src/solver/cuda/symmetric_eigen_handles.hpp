#pragma once

#include <cstdint>

#include "solver/cuda/symmetric_eigen_provider.hpp"

namespace generativeqc::solver::cuda {

/** Unique ownership of the provider's prepared handle and optional descriptors.
 * Staged preparation preserves each consumer's existing library call order and
 * numerical settings. Failed preparation retains acquired resources for reset
 * or destruction; no partially prepared owner is published by the caller.
 *
 * The enclosing device/stream owner must settle queued work and select its device
 * before reset/destruction. This class never selects a device, synchronizes,
 * allocates numeric workspace, or owns a borrowed stream. Explicit reset keeps
 * teardown at the original position relative to arenas, BLAS and stream release.
 * The vendor ABI stays private to the implementation, including for GFN2's
 * independent host declarations. */
class PreparedSymmetricEigenHandles {
 public:
  PreparedSymmetricEigenHandles() noexcept = default;
  ~PreparedSymmetricEigenHandles();
  PreparedSymmetricEigenHandles(const PreparedSymmetricEigenHandles&) = delete;
  PreparedSymmetricEigenHandles& operator=(const PreparedSymmetricEigenHandles&) = delete;
  PreparedSymmetricEigenHandles(PreparedSymmetricEigenHandles&& other) noexcept;
  PreparedSymmetricEigenHandles& operator=(PreparedSymmetricEigenHandles&& other) noexcept;

  /** Return the unmodified provider status; repeated acquisition is rejected
   * without overwriting a live resource. Configuration stops at first failure. */
  std::uint32_t create() noexcept;
  std::uint32_t bind_stream(void* stream) noexcept;
  std::uint32_t create_parameters() noexcept;
  std::uint32_t configure_jacobi(double tolerance, int max_sweeps, int sort) noexcept;
  void reset() noexcept;

  /** Immutable borrowed handles. Numeric workspace is supplied by its existing
   * arena owner at query/submission; this view never transfers ownership. */
  SymmetricEigenResources view() const noexcept { return {solver_, parameters_, jacobi_}; }

 private:
  void* solver_{};
  void* parameters_{};
  void* jacobi_{};
};

}  // namespace generativeqc::solver::cuda
