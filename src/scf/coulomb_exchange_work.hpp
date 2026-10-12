#ifndef GENERATIVEQC_SCF_COULOMB_EXCHANGE_WORK_HPP
#define GENERATIVEQC_SCF_COULOMB_EXCHANGE_WORK_HPP

#include <cstdint>
#include <limits>

namespace generativeqc::scf {

/** Observed value-provider applications, not ERI recurrences or FLOPs.
 * J applies once to the total density; K applies once per executed spin.
 * Producers certify only their actually dispatched contractions. An unknown
 * provider or overflow permanently revokes the combined census. This narrow
 * record does not certify the separate, full precision/operator inventory.
 */
struct CoulombExchangeWork {
  bool complete{};
  std::uint64_t coulomb_applications{};
  std::uint64_t exchange_applications{};

  static CoulombExchangeWork empty() noexcept { return {true, 0, 0}; }

  void add(const CoulombExchangeWork& other) noexcept {
    if (!complete || !other.complete ||
        other.coulomb_applications >
            std::numeric_limits<std::uint64_t>::max() - coulomb_applications ||
        other.exchange_applications >
            std::numeric_limits<std::uint64_t>::max() - exchange_applications) {
      complete = false;
      return;
    }
    coulomb_applications += other.coulomb_applications;
    exchange_applications += other.exchange_applications;
  }
};

}  // namespace generativeqc::scf

#endif
