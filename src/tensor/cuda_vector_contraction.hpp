#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <array>
#include <memory>

#include "generativeqc/generativeqc.h"
#include "runtime/lowering_binding.hpp"
#include "tensor/native_contraction.hpp"

namespace generativeqc::tensor {

/** Shared prepared implementation of a binary TensorIR contraction with a
 * vector output per batch. The enclosing provider exclusively owns the borrowed
 * handle/stream and preserves their modes until this binding is destroyed.
 * It also owns any opaque BLAS storage; this binding allocates no numeric data.
 * FP64 Dgemv/Dgemm retain the created handle's default math policy. Canonical
 * request/candidate identity strings must have static generated storage.
 * The constructor is for generated backend adapters, not a method selection API.
 */
class CudaVectorContraction {
 public:
  struct Diagnostic {
    runtime::NativeLoweringRequest request;
    std::array<runtime::NativeLoweringCandidate, 2> candidates;
    std::array<std::string_view, 2> rejections;
    std::size_t selected{};
    bool retained_incumbent{};
    int device{}, compute_major{}, compute_minor{}, provider_version{}, runtime_version{};
    std::uint64_t prepare_ns{};
    // Explicit binding-owned workspace is zero. Borrowed library-internal
    // allocations are not queryable here and are never claimed to be zero.
    bool borrowed_provider_bytes_known{};
  };

  CudaVectorContraction(const runtime::NativeLoweringRequest& request,
                        std::array<runtime::NativeLoweringCandidate, 2> candidates,
                        std::string_view target, std::string_view compilation,
                        const ContractionRequest& resolved, std::size_t qualified_incumbent,
                        cublasHandle_t handle, cudaStream_t stream);
  CudaVectorContraction(const CudaVectorContraction&) = delete;
  CudaVectorContraction& operator=(const CudaVectorContraction&) = delete;

  /** Borrow disjoint contiguous FP64 tensors until stream completion. Capture
   * records device work only; its enclosing owner accounts physical replays. */
  generativeqc_status launch(const double* matrix, const double* vector, double* output) const;
  const Diagnostic& diagnostic() const noexcept { return diagnostic_; }
  const ContractionRequest& resolved() const noexcept { return resolved_; }

 private:
  ContractionRequest resolved_;
  std::array<std::size_t, 3> operand_bytes_{};
  cublasHandle_t handle_{};
  cudaStream_t stream_{};
  std::array<char, 32> version_{};
  Diagnostic diagnostic_{};
  bool gemv_{};
};

}  // namespace generativeqc::tensor
