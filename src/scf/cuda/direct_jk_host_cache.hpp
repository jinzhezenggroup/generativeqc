#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <memory>

#include "runtime/temporary_cache_file.hpp"

namespace generativeqc::scf {

/** Experimental geometry-bound FP64 backing for omitted full/LR buckets.
 * Pageable values never enter a device kernel. Two pinned/device tile pairs
 * overlap sequential H2D transfers with the existing compensated contraction.
 * ready protects pinned reuse; consumed protects device reuse. The provider
 * drains its consumer stream before destroying this owner and its metadata.
 */
struct DirectJkHostCache {
  std::unique_ptr<double[]> values;
  std::array<std::size_t, 28> offsets{}, disk_offsets{};
  std::unique_ptr<runtime::TemporaryCacheFile> disk;
  std::size_t disk_count{};
  std::size_t count{}, tile_elements{}, next_slot{};
  cudaStream_t copy_stream{}, consumer_stream{};
  std::array<double*, 2> pinned{}, device{};
  std::array<cudaEvent_t, 2> ready{}, consumed{};
  std::array<bool, 2> used{};
  ~DirectJkHostCache();
  std::size_t staging_bytes() const noexcept { return 2 * tile_elements * sizeof(double); }
};

}  // namespace generativeqc::scf
