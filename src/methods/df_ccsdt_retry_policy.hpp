#pragma once

#include <algorithm>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <utility>

namespace generativeqc::methods::detail::df_retry {

/** Preserve a caller-supplied reference seed across success, resource retry and failure. */
template <class Experiment>
struct RestoreSeed {
  Experiment* diagnostic;
  decltype(std::declval<Experiment&>().initial_density) seed;

  explicit RestoreSeed(Experiment& experiment)
      : diagnostic(&experiment), seed(experiment.initial_density) {}
  ~RestoreSeed() { diagnostic->initial_density = seed; }
  RestoreSeed(const RestoreSeed&) = delete;
  RestoreSeed& operator=(const RestoreSeed&) = delete;
};

/** Charge seed, recycling and diagnostic output while they are simultaneously live. */
template <class CheckedAdd>
std::size_t attempt_reservation(std::size_t seed, std::size_t recycling,
                                std::size_t diagnostic_output, CheckedAdd checked_add) {
  return checked_add(seed, checked_add(recycling, diagnostic_output));
}

/** Retry one complete cold endpoint only after a resource refusal and cache release.
 *
 * The attempt owns/drains its phase resources. Both attempts use the same
 * prepared guess/diagnostic supplied by the enclosing endpoint; the second
 * failure propagates, and only successful retry output is flagged as having
 * unavailable precursor work. Finish is inside the first exception boundary,
 * exactly as in the original endpoint. No CUDA or scientific equation lives here.
 */
template <class MethodError, class Cache, class Attempt, class Finish>
auto retry_retained_cache_once(Cache* recycling, bool had_retained_cache,
                               int out_of_memory_status, Attempt&& attempt, Finish&& finish) {
  try {
    return finish(attempt());
  } catch (const MethodError& error) {
    if (!had_retained_cache || !recycling || !recycling->storage_bytes() ||
        error.status() != out_of_memory_status)
      throw;
  } catch (const std::length_error&) {
    if (!had_retained_cache || !recycling || !recycling->storage_bytes()) throw;
  } catch (const std::bad_alloc&) {
    if (!had_retained_cache || !recycling || !recycling->storage_bytes()) throw;
  }
  // The failed attempt has unwound its source, force and stream owners.
  // Never catch a second resource failure or another physical error here.
  recycling->clear();
  auto result = attempt();
  result.recycling_discarded_primal_attempt = true;
  return finish(std::move(result));
}

struct GroupSelection {
  std::size_t lambda_batch_limit;
  bool mixed_ccsd;
  bool mixed_lambda;
  double mixed_switch_residual;
};

/** Select the preconvergence policy without changing MP2/CC companion demand. */
template <class Preconvergence, class Convergence, class AutomaticShape>
GroupSelection select_group_schedule(bool forces, bool mp2, std::size_t nao,
                                     int electron_count, Preconvergence preconvergence,
                                     const Convergence& convergence, double strict_switch,
                                     AutomaticShape&& automatic_shape) {
  const std::size_t occupied =
      electron_count > 0 ? static_cast<std::size_t>(electron_count / 2) : 0;
  const auto virtuals = nao > occupied ? nao - occupied : 0;
  const bool default_shape = forces ? occupied >= 9 && virtuals >= 48
                                    : automatic_shape(occupied, virtuals);
  const bool mixed = !mp2 &&
                     (preconvergence == Preconvergence::Mixed ||
                      (preconvergence == Preconvergence::Legacy && convergence.relaxed()) ||
                      (preconvergence == Preconvergence::Automatic &&
                       (convergence.relaxed() || default_shape)));
  const bool lambda_mixed = mixed && forces;
  return {lambda_mixed ? 64U : 32U, mixed, lambda_mixed,
          std::max(strict_switch, 100.0 * convergence.cc_residual())};
}

}  // namespace generativeqc::methods::detail::df_retry
