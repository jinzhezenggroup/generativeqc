#pragma once

#include <memory>

#include "scf/cuda_df_gradient.hpp"

namespace generativeqc::scf {

/** Persistent consumer of borrowed device DF cotangent tiles.
 * Construction uploads normalized orbital-through-f/auxiliary-through-g metadata
 * once. The first consume binds one nonnull producer stream; every later tile
 * must use that stream. The producer owns weight lifetime until queued reads
 * finish, and may reuse its buffer in stream order immediately after consume.
 *
 * The caller must invoke finish ONLY after the complete producer has succeeded.
 * Destruction drains queued reads but publishes nothing; exceptions cannot turn
 * a partial contraction into a certified gradient. The stream owner outlives
 * this sink. A sink is single-use and is not shared between host threads.
 * Numeric capacity includes its conservative host setup peak and device storage,
 * excluding caller-owned systems, weights, stream and other live source state.
 * Charge numeric_capacity_bytes() to the producer's combined admission.
 */
class CudaDfNuclearSink {
 public:
  CudaDfNuclearSink(int device, const core::System& orbital, const core::System& auxiliary,
                    std::size_t maximum_bytes);
  ~CudaDfNuclearSink();
  CudaDfNuclearSink(const CudaDfNuclearSink&) = delete;
  CudaDfNuclearSink& operator=(const CudaDfNuclearSink&) = delete;
  void consume(unsigned kind, runtime::StridedRange range, const double* weights, std::size_t count,
               void* producer_stream);
  std::vector<double> finish();
  std::size_t numeric_capacity_bytes() const noexcept;
  DfGradientResources resources() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
};

}  // namespace generativeqc::scf
