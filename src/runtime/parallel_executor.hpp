#pragma once

#include <cstddef>

namespace generativeqc::runtime {

/** Borrowed synchronous chunk dispatcher; a null callback means serial work.
 * Dispatch must run every chunk exactly once before returning, without retaining
 * body context. Bodies must be noexcept, allocation-free and write disjoint output.
 * Supplying the executor does not create workers or permit reentrant pool use;
 * the caller must establish idle worker capacity and keep the pool alive.
 */
struct ChunkExecutor {
  void* pool_context = nullptr;
  std::size_t worker_count = 1;
  void (*dispatch_chunks)(void* pool_context, std::size_t chunk_count,
                          void (*body)(void* body_context, std::size_t chunk) noexcept,
                          void* body_context) = nullptr;

  [[nodiscard]] bool enabled() const noexcept { return dispatch_chunks != nullptr; }
};

}  // namespace generativeqc::runtime
