#pragma once

#include <algorithm>
#include <cstddef>

namespace generativeqc::scf {

/** Select a bounded charge traversal without changing its source domain.
 * A zero request denotes automatic selection; explicit 1..64 controls retain
 * their existing meaning. Long source classes are partitioned independently
 * of GPU model. At least 8192 primitives means the original
 * CTA traverses at least 256 source groups (8..32 source lanes). Short classes
 * retain that single-CTA schedule. The same complete FP64 contraction is
 * reduced in deterministic split order; no source or numerical gate changes.
 *
 * Complete partials must fit the dead orbital-potential allocation. A zero
 * capacity cannot hold partitioned output and retains the allocation-free
 * incumbent, whose output lives in the auxiliary view instead. */
constexpr unsigned df_md_charge_splits(unsigned request, std::size_t source_primitives,
                                       std::size_t partial_capacity) noexcept {
  const unsigned preferred = request ? request : (source_primitives >= 8192 ? 32 : 1);
  return static_cast<unsigned>(
      std::max<std::size_t>(1, std::min<std::size_t>(preferred, partial_capacity)));
}

}  // namespace generativeqc::scf
