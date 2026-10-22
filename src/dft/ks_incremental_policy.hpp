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

}  // namespace generativeqc::dft
