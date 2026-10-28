#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "runtime/df_progress_trace.hpp"
#include "tensor/ring_gram.hpp"

namespace generativeqc::scf {

/** The existing optional progress journal requests terminal DIIS observations.
 * Freeze this choice with the prepared arena/Graph; ordinary execution owns no
 * counter storage, readback, or journal scope. */
inline bool diis_work_requested() noexcept {
  const char* path = std::getenv("GENERATIVEQC_DF_PROGRESS_TRACE");
  return path && *path;
}

/** Host destination retained by the same owner as its asynchronous copies.
 * Work accumulates across wrap/retirement within one solve, while count/head
 * describe the terminal live ring. A new solve clears the work on device. */
struct DiisWorkReadback {
  tensor::RingGramWork work{};
  std::uint32_t count{}, head{};
};

/** Publish only after the owner's terminal stream boundary. The counters are
 * logical ordered dot reductions, not kernel launches, elapsed time, or DRAM
 * traffic. Other providers retain an explicit unavailable work census. */
inline void publish_diis_work(const DiisWorkReadback& receipt, const char* method,
                              std::uint64_t owner, std::uint64_t epoch, std::size_t item,
                              std::size_t nbf, unsigned spins, std::size_t capacity, bool ordered,
                              bool incremental, bool converged, bool failed,
                              std::uint64_t graph_launches = 0) noexcept {
  runtime::df_progress::Scope trace("cuda_diis_work", "cuda_completed");
  if (!trace.enabled()) return;
  using Journal = runtime::df_progress::Scope;
  Journal::label("method", method);
  Journal::label("route", ordered ? (nbf >= 512 ? "ordered-staged" : "ordered-warp")
                                  : (incremental ? "incremental-unordered" : "serial"));
  Journal::number("owner", owner);
  Journal::number("solve_epoch", epoch);
  Journal::number("item", item);
  Journal::number("nbf", nbf);
  Journal::number("spins", spins);
  Journal::number("history_capacity", capacity);
  Journal::number("live_history", receipt.count);
  Journal::number("history_head", receipt.head);
  Journal::number("converged", converged);
  Journal::number("failed", failed);
  // Graph owners count successful submissions over their retained lifetime,
  // separately from the per-attempt device-observed Gram work below.
  Journal::number("graph_launches_lifetime", graph_launches);
  constexpr auto limit = std::numeric_limits<std::uint64_t>::max();
  const bool valid_shape = nbf && spins && nbf <= limit / nbf && nbf * nbf <= limit / spins;
  const auto vector_size = valid_shape ? static_cast<std::uint64_t>(nbf) * nbf * spins : 0;
  const bool valid = receipt.count <= capacity && receipt.head < capacity && vector_size &&
                     receipt.work.dots <= std::numeric_limits<std::uint64_t>::max() / vector_size &&
                     receipt.work.vector_elements == receipt.work.dots * vector_size;
  Journal::label("work_coverage", ordered && valid ? "observed" : "unavailable");
  if (ordered && valid) {
    Journal::number("gram_dot_reductions", receipt.work.dots);
    Journal::number("residual_vector_elements", receipt.work.vector_elements);
    Journal::number("blas_calls", 0);
  }
  Journal::number("device_counter_bytes", sizeof(tensor::RingGramWork));
  Journal::number("host_readback_bytes", sizeof(DiisWorkReadback));
  trace.finish(valid ? "completed" : "invalid-history-or-work");
}

}  // namespace generativeqc::scf
