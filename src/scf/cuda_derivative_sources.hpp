#pragma once

#include <cstddef>
#include <functional>

namespace generativeqc::scf {

/** The two existing provider protocols retain their separate source channels.
 * Convention is a transport property, not a new coefficient or force equation. */
enum class CudaDerivativeSourceKind {
  OneElectron,
  OverlapPulay,
  FullRangeJk,
  RangeSeparatedJk,
  Coulomb,
  Exchange,
  RangeCorrection,
  /** Already weighted J+K derivative in the declared transport convention.
   * Only a total-gradient consumer
   * whose compiler proves a unit sum may accept this coalesced source; it
   * cannot recover independent J/K diagnostics or range-separated channels. */
  FullRangeJkSum
};
enum class CudaDerivativeConvention { EnergyGradient, Force };

/** A callback-scoped source-major device span, never a persistent force lease.
 * Each channel contains coordinates doubles. The consumer must enqueue its
 * use/copy on producer_stream before returning; subsequent provider work may
 * overwrite values. It owns the destination and final drain/publication gate.
 * The enclosing prepared calculation authenticates the physical D/W token. */
struct CudaDerivativeSourceView {
  CudaDerivativeSourceKind kind{};
  CudaDerivativeConvention convention{};
  int device{-1};
  void* producer_stream{};
  const double* values{};
  std::size_t coordinates{}, channels{};
};

using CudaDerivativeSourceConsumer = std::function<void(const CudaDerivativeSourceView&)>;

}  // namespace generativeqc::scf
