#pragma once

#include <cstddef>
#include <cstdint>

namespace generativeqc::dft {
/** Geometry-owned sampled-AO work. Point/AO counts describe one full XC
 * traversal; density changes reuse the immutable maps. No field certifies a
 * numerical error or claims an elapsed-time speedup. */
struct CudaXcAoSelectionWork {
  bool requested{}, selected{};
  double cutoff{}, discovery_seconds{};
  std::size_t tiles{}, empty_tiles{}, min_active{}, max_active{}, active_sum{};
  std::uint64_t discovery_ao_jet_values{}, point_ao_visits{}, point_ao_square_sum{},
      dense_point_ao_square_sum{}, discovery_d2h_bytes{};
  std::size_t reserved_device_bytes{}, host_peak_bytes{};
  std::uint64_t xc_evaluations{};
  /** Submitted discovery transfers/fences, separate from later XC evaluation.
   * They are available in the opt-in preparation trace without changing v1 ABI.
   */
  std::uint64_t discovery_h2d_bytes{}, discovery_synchronizations{}, discovery_chunks{};
  bool device_compaction{};
};
}  // namespace generativeqc::dft
