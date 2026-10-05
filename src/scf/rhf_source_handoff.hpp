#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace generativeqc::core {
struct System;
}

namespace generativeqc::integrals {
class ElectronInteractionSource;
}

namespace generativeqc::scf {
struct ScfOptions;
struct ScfResult;

/** Optional exact interaction source detached from a successful CUDA RHF.
 *
 * Only immutable public-AO basis/geometry metadata survives. The SCF arena,
 * density, DIIS, J/K scratch and screening state are not retained. Raw ERI
 * reads remain unscreened FP64 on the consumer's stream. The consumer must
 * finish that stream before releasing its last source reference.
 *
 * Compaction copies device metadata once because RHF currently places it in
 * the iteration arena. Admission includes both allocations at the handoff.
 * Resource rejection leaves source empty; it never changes RHF convergence.
 * required_peak_bytes is a capacity query, not an observed allocation peak.
 */
struct CudaRhfSourceHandoff {
  std::shared_ptr<const integrals::ElectronInteractionSource> source;
  std::size_t retained_numeric_bytes{};
  std::size_t required_peak_bytes{};
  std::size_t numeric_peak_bytes{};
  std::size_t device_copy_bytes{};
  double seconds{};
  bool resource_fallback{};
};

/** Execute the same physical CUDA RHF solve, then optionally detach its exact
 * source within ScfOptions::reference_memory_budget_bytes. No source is
 * published for a failed/nonconverged reference or rejected optional storage. */
ScfResult run_rhf_cuda_with_source(const core::System& system, const ScfOptions& options,
                                   int device_id, const std::vector<double>* initial_density,
                                   CudaRhfSourceHandoff& handoff);

}  // namespace generativeqc::scf
