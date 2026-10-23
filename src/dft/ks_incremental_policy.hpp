#pragma once

namespace generativeqc::dft {

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
