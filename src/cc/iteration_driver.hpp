#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "cc/solver.hpp"
#include "solver/iteration_control.hpp"

namespace generativeqc::cc {

// The backend owns evaluation, storage, synchronization, diagnostics and failure
// recovery. This host-only driver owns the shared scientific iteration policy.
// Evaluate observes a carried output or computes a fresh one. Advance may return
// a retained output only when its final state equals the evaluated trial and the
// referenced storage remains valid until the next evaluation. Replay never uses
// that retained output. No callback may invalidate an output before consuming it.
struct IterationMetrics {
  double energy, r1, r2;
};

template <class Output, class Evaluate, class Publish, class Replay, class Advance, class Converged,
          class Failure>
void run_cc_iterations(const SolverOptions& options, Evaluate&& evaluate, Publish&& publish,
                       Replay&& replay, Advance&& advance, Converged&& converged,
                       Failure&& failure) {
  double previous = std::numeric_limits<double>::quiet_NaN();
  std::optional<Output> carried;
  const auto step = [&](unsigned iteration) {
    try {
      const auto [output, status] = evaluate(carried);
      carried.reset();
      const double delta = std::isfinite(previous) ? std::abs(status.energy - previous)
                                                   : std::numeric_limits<double>::infinity();
      const unsigned observations =
          iteration == std::numeric_limits<unsigned>::max() ? iteration : iteration + 1;
      publish(observations, status, delta);
      if (std::isfinite(previous) && delta <= options.energy_tolerance &&
          std::max(status.r1, status.r2) <= options.residual_tolerance) {
        // Physical replay is always fresh, even when the iteration output was
        // retained from an unchanged DIIS trial.
        const auto physical = replay();
        if (std::max(physical.r1, physical.r2) <= options.residual_tolerance &&
            std::abs(physical.energy - status.energy) <= options.energy_tolerance) {
          converged();
          return false;
        }
      }
      // max_iterations counts updates; always observe the final updated state.
      if (iteration == options.max_iterations) return false;
      carried = advance(output);
      previous = status.energy;
      return true;
    } catch (const std::runtime_error& error) {
      failure(error);
      return false;
    }
  };
  const auto loop = solver::run_bounded_iterations(
      options.max_iterations, [&](unsigned ordinal) { return step(ordinal - 1); });
  // Keep the last observation separate instead of overflowing max_iterations+1.
  if (!loop.stopped_by_callback) step(options.max_iterations);
}

}  // namespace generativeqc::cc
