#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

#include "runtime/lowering_binding.hpp"

namespace generativeqc::tensor {

/** Borrowed packed [reduction, column] panels and full row-major output.
 * B is batched; A is shared by all batches. Empty/indexed domains retain the
 * generated consumer, whose callback also owns its finite/epilogue contract.
 * Callbacks enqueue synchronously and may not retain the host state pointer.
 */
struct SymmetricProductInvocation {
  std::size_t columns{}, reduction{}, batches{};
  const double *a{}, *b{};
  double* output{};
  int* error{};
  bool accumulate{}, indexed{};
  void* state{};
  void (*generated)(void*){};
  void (*materialize)(void*){};
  void (*epilogue)(void*){};
};

struct SymmetricProductDiagnostic {
  runtime::NativeLoweringCandidate candidate;
  // Dense qualification never admits indexed scatter. Keep its actual binding
  // provenance explicit even when the same owner also retains a dense handle.
  runtime::NativeLoweringCandidate indexed_candidate;
  std::string_view alternative_rejection;
  std::size_t host_bytes{}, provider_allowance{}, retained_provider_bytes{};
  int provider_version{};
  double prepare_seconds{};
};

/** Prepared resource owner. Execution enqueues only device work, including
 * during capture; the enclosing physical replay publisher owns work counts.
 * An invocation must not alias either input with output. The caller's stream
 * and all captured graphs must be drained/destroyed before this binding dies.
 */
class PreparedSymmetricProduct {
 public:
  static constexpr std::size_t host_reservation = 16U << 10;
  virtual ~PreparedSymmetricProduct() = default;
  virtual void execute(cudaStream_t, const SymmetricProductInvocation&) const = 0;
  virtual const SymmetricProductDiagnostic& diagnostic() const noexcept = 0;
};
}  // namespace generativeqc::tensor
