#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string_view>

#include "generativeqc/generativeqc.h"

namespace generativeqc::runtime {
struct NativeLoweringRequest;
struct NativeLoweringCandidate;
}  // namespace generativeqc::runtime

namespace generativeqc::tensor {

/** Immutable operation bindings borrowing an already prepared FP64 context.
 * The enclosing resource owner still owns handles, scratch and stream lifetime.
 * This migration preserves its qualified incumbent, without inventing costs or
 * promoting another precision/provider. Execution cannot change the recipe.
 * Shape bounds and borrowed context are separate from symbolic AOT identities.
 */
class CudaMatrixProductBinding {
 public:
  static constexpr std::size_t kPreparationHostBytes = 16U << 10;
  struct PreparationFailure : std::runtime_error {
    generativeqc_status status;
    PreparationFailure(generativeqc_status code, const char* message)
        : std::runtime_error(message), status(code) {}
  };
  struct Context {
    int device{}, runtime_version{}, provider_version{};
  };
  struct Site {
    /** Canonical AOT metadata; candidate() adds resolved resource/capture facts. */
    const runtime::NativeLoweringRequest* request{};
    const runtime::NativeLoweringCandidate* selected{};
    std::string_view alternative_rejection;
    bool retained_incumbent{};
    std::size_t provider_bytes{}, temporary_bytes{}, host_bytes{};
    runtime::NativeLoweringCandidate candidate() const;
  };

  CudaMatrixProductBinding(cudaStream_t stream, cublasHandle_t handle, int dimension, int systems,
                           int spins);
  bool compatible(cudaStream_t stream, cublasHandle_t handle, int dimension, int systems,
                  int spins) const noexcept;
  bool requires_provider_context() const noexcept { return recipe_ == Recipe::Library; }
  std::size_t scratch_bytes() const noexcept {
    return requires_provider_context() ? scratch_bytes_ : 0;
  }
  const Site& site(bool spin, bool left_spin, bool right_spin, bool transpose) const;
  const Context& context() const noexcept { return context_; }

 private:
  enum class Recipe { Generated, Library };
  Recipe recipe_{Recipe::Generated};
  cudaStream_t stream_{};
  cublasHandle_t handle_{};
  int dimension_{}, systems_{}, spins_{};
  std::size_t scratch_bytes_{};
  Context context_;
  std::array<Site, 10> sites_{};
};

namespace cuda {
/** Already-bound column-major GEMM. Singleton systems use ordinary GEMM;
 * physical operands broadcast across spins without pointer-list allocation.
 * Masked publication remains a separate stream-ordered executor obligation. */
generativeqc_status matrix_product(cublasHandle_t handle, int systems, int spins, int dimension,
                                   const double* left, bool left_spin, bool transpose_left,
                                   const double* right, bool right_spin, double* output,
                                   double scale);
}  // namespace cuda
}  // namespace generativeqc::tensor
