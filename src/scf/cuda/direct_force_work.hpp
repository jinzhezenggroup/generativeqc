#pragma once

#include "scf/direct_task_layout.hpp"

namespace generativeqc::scf::cuda_execution {

/** Optional counters at the actual shared-recurrence execution sites. A
 * Coulomb preparation is one executed bra/ket primitive-pair product, after
 * center, AO Schwarz and nonzero-source admission. It is not a theoretical
 * Cartesian-component-times-primitive cardinality.
 */
struct MaterializedDirectPairWork {
  unsigned long long bra_preparations{}, ket_preparations{}, coulomb_preparations{};
  unsigned long long component_contractions{}, published_components{};
};

struct DensityDirectPairWork {
  unsigned long long attempted_shells{}, accepted_shells{}, mask_rejections{};
  unsigned long long numerical_rejections{}, density_terms{}, hermite_products{};
  MaterializedDirectPairWork recurrence;
};

/** Intrusive scalar-consumer attribution, not kernel or endpoint elapsed time.
 * Inclusive thread cycles are sampled by physical task identity. Their sum
 * may overlap across lanes/SMs; compare class cost only within this census.
 * Separable work additionally requires full AO admission, zero exchange and
 * no bilinear seed. These counters never authorize changing those predicates.
 */
struct ScalarDirectForceWork {
  unsigned long long calls{}, separable_calls{}, separable_primitive_products{};
  unsigned long long sampled_calls{}, sampled_thread_cycles{};
  unsigned long long separable_sampled_calls{}, separable_sampled_thread_cycles{};
};

/** Actual execution of the default-off scalar pair candidate, separately
 * from incumbent eligibility. Rejected work is retained before strict retry. */
struct ScalarSeparableDirectForceWork {
  unsigned long long attempted_shells{}, accepted_shells{}, mask_rejections{};
  unsigned long long numerical_rejections{}, checked_ao_quartets{};
  unsigned long long bra_density_terms{}, ket_density_terms{}, primitive_products{};
  /** Pair-factor loads and exact minimum-product decisions, not AO quartets.
   * Invalid factors retain the incumbent per-quartet comparison semantics. */
  unsigned long long screening_pair_terms{}, screening_pair_proofs{}, screening_pair_fallbacks{};
};

/** Diagnostic-only census of one complete force dispatch. Queue visits and
 * actual source work have distinct fields; no shell inventory is called an
 * executed integral. Null borrowed pointers retain the ordinary route.
 */
struct DirectForceClassWork {
  unsigned long long enumerated{}, shell_screened{}, generated_owned{}, admitted{};
  unsigned long long executed_shells{}, executed_components{}, primitive_pair_products{};
  MaterializedDirectPairWork recurrence;
  DensityDirectPairWork density;
  ScalarDirectForceWork scalar;
  ScalarSeparableDirectForceWork scalar_pair;
};

struct DirectForceTraversalWork {
  unsigned long long pages{}, inactive_pages{}, block_rejected_pages{}, candidates{};
  unsigned long long angular_rejected{};
};

struct DirectForceWork {
  // 0: low/spatial or generic; 1: cooperative order seven; 2: materialized
  // order eight. A repeated candidate traversal is charged to its own route.
  DirectForceTraversalWork traversal[3];
  DirectForceClassWork classes[detail::kDirectQuartetShellClassCount];
};

}  // namespace generativeqc::scf::cuda_execution
