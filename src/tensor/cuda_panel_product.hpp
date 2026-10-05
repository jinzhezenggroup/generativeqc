#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

#include "runtime/lowering_binding.hpp"

namespace generativeqc::tensor {

/** Prepared left-matrix times shared panel. The compiler materializes each
 * column-major square matrix at the recipe's lifetime boundary: once per
 * evaluation for a dense domain, or per map for indexed views. No scientific
 * symmetrization or precision policy lives here.
 * The owner supplies an independent allowance beyond its ordinary numeric arena.
 */
struct PanelProductDiagnostic {
  runtime::NativeLoweringCandidate candidate;
  std::string_view alternative_rejection;
  std::size_t host_bytes{}, matrix_bytes{}, provider_allowance{}, retained_provider_bytes{};
  int provider_version{};
  double prepare_seconds{};
};

class PreparedPanelProduct {
 public:
  static constexpr std::size_t host_reservation = 16U << 10;
  virtual ~PreparedPanelProduct() = default;
  virtual bool enabled() const noexcept = 0;
  virtual double* materialized_matrices() const noexcept = 0;
  /** Panels are row-major [rows, columns], shared by batches; outputs are
   * [batch, rows, columns]. All buffers must be disjoint and live on the
   * prepared stream/device until completion, including captured replay.
   * Bounded-column recipes require freshly materialized compact matrices of
   * [batch, columns, columns] for each call; no global leading stride is used.
   * Empty domains stay with the caller's generated zero-domain operation. */
  virtual void execute(cudaStream_t, std::size_t columns, std::size_t rows, const double* panel,
                       double* output, int* error) const = 0;
  virtual const PanelProductDiagnostic& diagnostic() const noexcept = 0;
};
}  // namespace generativeqc::tensor
