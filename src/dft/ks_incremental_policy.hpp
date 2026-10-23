#pragma once

#include <cstddef>

namespace generativeqc::dft {

/** Bound a guarded automatic cold Direct hybrid default separately from the
 * explicit incremental controller. The caller already owns default Standard
 * FP64 controls; no public resource ledger can borrow uncounted anchor space. */
struct KsIncrementalColdAutoDomain {
  bool requested{};
  std::size_t public_aos{};
  std::size_t atoms{};
  bool exact_fp64_direct_hybrid{};
  bool standard_default_grid{};
  bool through_d_shells{};
  bool all_electron{};
  bool no_resource_ledger{};
};

constexpr bool admit_ks_incremental_cold_auto(KsIncrementalColdAutoDomain domain) noexcept {
  return domain.requested && domain.exact_fp64_direct_hybrid && domain.standard_default_grid &&
         domain.through_d_shells && domain.all_electron && domain.no_resource_ledger &&
         domain.public_aos >= 64 && domain.public_aos <= 768 && domain.atoms >= 2 &&
         domain.atoms <= 96;
}

/** Select each solve from a prepared capability without confusing a native
 * cold guess with an imported density. Only the initial-guess owner may attest
 * native provenance. Retained warm proof stays ordinary, and explicit opt-in
 * preserves its broader external-seed contract. */
constexpr bool ks_incremental_solve_active(bool prepared_active, bool automatic, bool ordinary_warm,
                                           bool supplied_density, bool native_cold_guess) noexcept {
  return prepared_active && !ordinary_warm &&
         (!automatic || !supplied_density || native_cold_guess);
}

/** Bound an incremental proposal epoch whose consumer requires a fresh physical
 * full-density closure. These are provisional SCF operators, not a certificate
 * of the accumulated screening error. The closing full-Fock energy pair and
 * canonical proof remain mandatory; callers without that ownership retain the
 * exact controller's ordinary screened cadence. Zero may not create an
 * unbounded screened proposal chain, even when periodic exact refresh is off.
 */
constexpr unsigned ks_incremental_proposal_interval(bool audited_proposals,
                                                    unsigned requested_interval,
                                                    unsigned ordinary_interval) noexcept {
  if (!audited_proposals) return ordinary_interval;
  constexpr unsigned maximum_proposals = 8;
  return requested_interval && requested_interval < maximum_proposals ? requested_interval
                                                                      : maximum_proposals;
}

/** A coarser cutoff accelerates only the first eight provisional delta actions
 * in a solve, across refresh epochs. Exhaustion restores the prepared cutoff;
 * neither warm/full builds nor unscreened requests may inherit this policy.
 * This is a proposal heuristic, not a bound on accumulated Fock error. Only
 * the independent full-density tail may certify the final state.
 */
constexpr double ks_incremental_proposal_screening(bool audited_delta, unsigned delta_builds,
                                                   double prepared_screening) noexcept {
  return audited_delta && delta_builds > 0 && delta_builds <= 8 && prepared_screening > 0.0 &&
                 prepared_screening < 1e-9
             ? 1e-9
             : prepared_screening;
}

}  // namespace generativeqc::dft
