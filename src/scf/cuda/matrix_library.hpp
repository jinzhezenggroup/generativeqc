#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "generativeqc/generativeqc.h"

namespace generativeqc::scf::cuda_execution {

/** Non-owning matrix-library execution context, independent of provider/plan storage. */
struct MatrixLibraryResources {
  cudaStream_t stream_{};
  cublasHandle_t blas_{};
  /** Caller-owned, precharged scratch for masked library results. It must be
   * disjoint from both operands, output and mask, and live through stream/Graph
   * completion. Native and unmasked library products do not use it. */
  double* masked_output_{};
  std::size_t masked_output_elements_{};
};

/** Prepared host-safe owner for the shared SCF matrix adapter.
 *
 * Provider selection is frozen at preparation. Small AO spaces keep the
 * generated matrix kernel; larger spaces may retain one cuBLAS handle on the
 * caller-owned stream. Allocation failure falls back to generated execution,
 * while other provider/runtime failures are reported to the caller.
 */
class MatrixLibraryOwner {
 public:
  static constexpr std::size_t kProviderAllowance = 96ULL << 20;
  // One shape-only policy serves preparation and the public resource inventory.
  // Preserve the qualified small-SCF crossover until #1890 replaces it.
  static constexpr std::size_t provider_allowance(int nbf) noexcept {
    return nbf >= 17 ? kProviderAllowance : 0;
  }

  MatrixLibraryOwner() = default;
  ~MatrixLibraryOwner();
  MatrixLibraryOwner(const MatrixLibraryOwner&) = delete;
  MatrixLibraryOwner& operator=(const MatrixLibraryOwner&) = delete;

  generativeqc_status prepare(cudaStream_t stream, int nbf);
  void reset() noexcept;

  bool library_enabled() const noexcept { return blas_ != nullptr; }
  MatrixLibraryResources view(double* masked_output = nullptr,
                              std::size_t masked_output_elements = 0) const noexcept {
    return {stream_, blas_, masked_output, masked_output_elements};
  }
  std::size_t retained_bytes() const noexcept { return retained_bytes_; }

 private:
  int device_{-1};
  cudaStream_t stream_{};
  cublasHandle_t blas_{};
  std::size_t retained_bytes_{};
  bool prepared_{};
};

/** Use the resolved native/library route, preserving masks and column-major strides.
 * Spin products broadcast physical operands through one library submission per spin;
 * singleton physical batches use ordinary GEMM and true multi-system batches use
 * strided-batched GEMM. The caller owns every input/output allocation and borrowed
 * library handles.
 * A non-null device mask requires batch_size * spin_count * nbf * nbf scratch
 * elements for the library route (spin_count is one for plain products). GEMM
 * writes only scratch; a stream-ordered masked copy preserves inactive output
 * bytes, including on Graph replay. Missing/undersized/overlapping scratch is
 * rejected before submission; explicit library selection never becomes native.
 * The optional scale is applied by both the native and cuBLAS routes, allowing
 * occupation normalization without a separate matrix pass.
 */
generativeqc_status launch_matrix_product(MatrixLibraryResources resources, int batch_size, int nbf,
                                          const double* left, bool transpose_left,
                                          const double* right, const std::uint8_t* active,
                                          double* output, bool use_cublas, double scale = 1.0);

generativeqc_status launch_spin_matrix_product(MatrixLibraryResources resources, int batch_size,
                                               int spin_count, int nbf, const double* left,
                                               bool left_is_spin, bool transpose_left,
                                               const double* right, bool right_is_spin,
                                               const std::uint8_t* active, double* output,
                                               bool use_cublas);

}  // namespace generativeqc::scf::cuda_execution
