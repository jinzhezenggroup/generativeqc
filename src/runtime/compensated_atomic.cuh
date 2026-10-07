#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace generativeqc::runtime {

/** Two separately owned planes for a compensated unordered FP64 scatter.
 * Each atomic returns the previous sum, so its addition's rounding residual
 * can be recovered without a lock. Fold the correction only after every
 * writer completes on the owning stream. This improves accuracy; it is not
 * a bitwise-reproducible summation contract.
 */
struct CompensatedAddress {
  double* sum;
  double* correction;

  __device__ CompensatedAddress operator+(std::size_t offset) const {
    return {sum + offset, correction ? correction + offset : nullptr};
  }
};

struct CompensatedOutput {
  double* sum;
  double* correction;

  __device__ CompensatedAddress operator+(std::size_t offset) const {
    return {sum + offset, correction ? correction + offset : nullptr};
  }
};

/** Neumaier's magnitude-ordered residual, with explicit rounded additions.
 * The second plane sums rounding errors, not a second copy of the values.
 * A null correction preserves the ordinary atomic scatter. Nonfinite values
 * propagate through the fold and remain subject to the consumer's audit.
 */
__device__ __forceinline__ void atomicAdd(CompensatedAddress output, double value) {
  const double previous = ::atomicAdd(output.sum, value);
  if (!output.correction) return;
  const double updated = __dadd_rn(previous, value);
  const double residual = fabs(previous) >= fabs(value)
                              ? __dadd_rn(__dadd_rn(previous, -updated), value)
                              : __dadd_rn(__dadd_rn(value, -updated), previous);
  ::atomicAdd(output.correction, residual);
}

}  // namespace generativeqc::runtime
