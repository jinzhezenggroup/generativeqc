#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string_view>

#include "scf/initial_guess/eigen_operation.hpp"
#include "scf/reference/linalg.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::scf::solver {

/** Host physical validation keeps its original matrices, equations and gates.
 * An owner may prefer the shared FP64 provider for qualified cold preparation;
 * other owners remain default-off. Neither schedule replaces the reference
 * eigensolver or gates. Capture the selector once per validation; an explicit
 * off/invalid selector or unavailable provider keeps the scalar incumbent. */
class ValidationProducts {
 public:
  explicit ValidationProducts(const initial_guess::MatrixProductOperation& operation = {},
                              bool prefer_provider = false)
      : operation_(operation) {
#if defined(GENERATIVEQC_HAS_OPENBLAS) && GENERATIVEQC_HAS_OPENBLAS
    const auto* selection = std::getenv("GENERATIVEQC_SCF_VALIDATION_GEMM");
    const bool requested = selection ? std::string_view(selection) == "1" : prefer_provider;
    provider_ =
        requested && tensor::resolve_cpu_linalg_provider({}) == tensor::CpuLinalgProvider::openblas;
#else
    (void)prefer_provider;
#endif
  }

  /** Return op(A) * op(B) in row-major storage. Callers validate square shapes
   * before entering this internal adapter. Provider transpose flags eliminate
   * explicit transpose temporaries without changing the contracted equation. */
  reference::Matrix operator()(const reference::Matrix& a, const reference::Matrix& b,
                               std::size_t n, bool transpose_a = false,
                               bool transpose_b = false) const {
    if (operation_) {
      auto result = operation_(a, b, n, transpose_a, transpose_b);
      if (result.size() != n * n)
        throw std::runtime_error("borrowed validation product returned an invalid shape");
      return result;
    }
#if defined(GENERATIVEQC_HAS_OPENBLAS) && GENERATIVEQC_HAS_OPENBLAS
    if (provider_) {
      reference::Matrix result(n * n);
      tensor::cpu_gemm(transpose_a ? 'T' : 'N', transpose_b ? 'T' : 'N', n, n, n, a.data(),
                       b.data(), result.data());
      return result;
    }
#endif
    // Keep the old scalar reduction and temporary lifetimes for the disabled
    // path and unavailable-provider fallback; do not redirect the oracle itself.
    if (transpose_a && transpose_b)
      return reference::multiply(reference::transpose(a, n), reference::transpose(b, n), n);
    if (transpose_a) return reference::multiply(reference::transpose(a, n), b, n);
    if (transpose_b) return reference::multiply(a, reference::transpose(b, n), n);
    return reference::multiply(a, b, n);
  }

 private:
  initial_guess::MatrixProductOperation operation_;
#if defined(GENERATIVEQC_HAS_OPENBLAS) && GENERATIVEQC_HAS_OPENBLAS
  bool provider_{};
#endif
};

}  // namespace generativeqc::scf::solver
