#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <array>

#include "runtime/lowering_binding.hpp"
#include "tensor/cuda_matrix_product.hpp"

namespace generativeqc::tensor {

/** Canonical regions, not method/vendor implementation switches. */
enum class StructuredOperation : std::size_t {
  Gram,
  SignedGram,
  InverseRoot,
  DensityRight,
  DensityLeft,
  FockLeft,
  FockRight
};

/** Immutable bindings borrowing the existing owner's context and charged arena.
 * The finite owner must outlive bindings and all captured invocations. Sparse
 * shell-span domains retain their existing executor, not an implicit dense pack.
 */
class CudaStructuredProductBinding {
 public:
  static constexpr std::size_t kPreparationHostBytes = 32U << 10;
  /** Primitive evidence cannot promote a new scientific endpoint by default. */
  static constexpr bool production_library_qualified = false;
  CudaStructuredProductBinding(cudaStream_t stream, cublasHandle_t handle, int dimension,
                               int direct_dimension, int systems, int spins);
  bool compatible(cudaStream_t stream, cublasHandle_t handle, int dimension, int systems,
                  int spins) const noexcept;
  const runtime::NativeLoweringCandidate& site(StructuredOperation operation) const;
  /** Complete bounded offers with resolved byte charges and explicit rejections.
   * Unknown endpoint costs remain unknown; enumeration performs no selection. */
  std::array<runtime::NativeLoweringCandidate, 7> candidates(StructuredOperation operation) const;
  bool library(StructuredOperation operation) const;
  int dimension() const noexcept { return dimension_; }
  int direct_dimension() const noexcept { return direct_dimension_; }
  int systems() const noexcept { return systems_; }
  int spins() const noexcept { return spins_; }
#if defined(GENERATIVEQC_TEST_HOOKS)
  /** Diagnostic-only ablation; production owners cannot pin an implementation. */
  static CudaStructuredProductBinding qualification(cudaStream_t stream, cublasHandle_t handle,
                                                    int dimension, int direct_dimension,
                                                    int systems, int spins) {
    return CudaStructuredProductBinding(stream, handle, dimension, direct_dimension, systems, spins,
                                        true);
  }
#endif

 private:
  CudaStructuredProductBinding(cudaStream_t stream, cublasHandle_t handle, int dimension,
                               int direct_dimension, int systems, int spins,
                               bool primitive_qualification);
  cudaStream_t stream_{};
  cublasHandle_t handle_{};
  int dimension_{}, direct_dimension_{}, systems_{}, spins_{};
  bool primitive_qualification_{};
  std::array<runtime::NativeLoweringCandidate, 7> sites_{};
};

namespace cuda {
/** Full-square positive Gram or signed column-scale reconstruction. The caller
 * applies a device mask after the contraction; only scratch is written here. */
generativeqc_status structured_gram(cublasHandle_t handle, int dimension, int states,
                                    const double* coefficients, const double* scaled,
                                    double* scratch, double occupancy, bool unsigned_gram);
/** One rectangular column-major stage; transform broadcasts across spins. */
generativeqc_status basis_stage(cublasHandle_t handle, StructuredOperation operation, int dimension,
                                int direct_dimension, int systems, int spins,
                                const double* transform, const double* input, double* scratch);
}  // namespace cuda
}  // namespace generativeqc::tensor
