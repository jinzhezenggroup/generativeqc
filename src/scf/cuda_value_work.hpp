#ifndef GENERATIVEQC_SCF_CUDA_VALUE_WORK_HPP
#define GENERATIVEQC_SCF_CUDA_VALUE_WORK_HPP

#include <cstdint>

#include "scf/coulomb_exchange_work.hpp"

namespace generativeqc::scf {

/** Observe one known value-provider dispatch, never a requested iteration.
 * J consumes the total density; each K term consumes every executed spin.
 * Failed/unknown submissions and unsupported spin/term domains revoke the
 * execution-local census. A later success cannot repair that missing work.
 * Graph capture/replay callers must leave their census unavailable unless
 * they separately observe the applications that actually executed.
 */
inline void observe_cuda_value_dispatch(CoulombExchangeWork& work, bool submitted, bool coulomb,
                                        unsigned exchange_terms, unsigned spins) noexcept {
  if (!submitted || (spins != 1 && spins != 2) || exchange_terms > 2) {
    work.complete = false;
    return;
  }
  work.add({true, coulomb ? 1U : 0U, static_cast<std::uint64_t>(exchange_terms) * spins});
}

}  // namespace generativeqc::scf

#endif
