#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "solver/diis.hpp"
#include "solver/diis_coefficients.hpp"
#include "solver/self_consistent.hpp"

namespace {

using generativeqc::solver::run_bounded_iterations;
using generativeqc::solver::run_self_consistent;
using generativeqc::solver::SelfConsistentPolicy;
using generativeqc::solver::SelfConsistentProgress;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct ScalarEvaluation {
  double proposed_state{};
  double energy{};
  double state_rms{};
  double residual_rms{};
};

void verify_bounded_iteration_control() {
  unsigned calls = 0;
  const auto stopped = run_bounded_iterations(8, [&](unsigned iteration) {
    ++calls;
    require(iteration == calls, "bounded iteration numbering changed");
    return iteration < 3;
  });
  require(stopped.completed_iterations == 3, "bounded early-stop count changed");
  require(stopped.stopped_by_callback, "bounded callback stop was not reported");
  require(calls == 3, "bounded driver executed after stop");

  calls = 0;
  const auto exhausted = run_bounded_iterations(2, [&](unsigned) {
    ++calls;
    return true;
  });
  require(exhausted.completed_iterations == 2, "bounded maximum was not enforced");
  require(!exhausted.stopped_by_callback, "bounded exhaustion was marked callback-stopped");
  require(calls == 2, "bounded driver call count changed");

  calls = 0;
  const auto empty = run_bounded_iterations(0, [&](unsigned) {
    ++calls;
    return true;
  });
  require(empty.completed_iterations == 0, "zero iteration budget changed");
  require(!empty.stopped_by_callback, "zero iteration budget was marked callback-stopped");
  require(calls == 0, "zero iteration budget invoked the callback");
}

void verify_basic_convergence() {
  const SelfConsistentPolicy policy{8, 1.0e-12, 1.0e-12, 1.0e-12, false};
  unsigned records = 0;
  const auto outcome = run_self_consistent(
      0.0, policy,
      [](double state, unsigned) {
        const double next = 1.0;
        return ScalarEvaluation{next, (state - 1.0) * (state - 1.0), std::abs(next - state), 99.0};
      },
      [](double, ScalarEvaluation evaluation, const SelfConsistentProgress&) {
        return evaluation.proposed_state;
      },
      [&](const SelfConsistentProgress& progress, const ScalarEvaluation&) {
        ++records;
        require(progress.iteration == records, "progress iteration order changed");
      });

  require(outcome.converged, "fixed-point driver failed a convergent scalar problem");
  require(outcome.progress.iteration == 3, "fixed-point convergence iteration changed");
  require(outcome.state == 1.0, "fixed-point driver did not retain the accepted state");
  require(records == 3, "fixed-point recorder call count changed");
}

void verify_residual_gate() {
  const SelfConsistentPolicy policy{8, 1.0e-12, 1.0e-12, 1.0e-6, true};
  const auto outcome = run_self_consistent(
      0.0, policy,
      [](double state, unsigned iteration) {
        const double next = 1.0;
        const double residual = iteration < 4 ? 1.0 : 0.0;
        return ScalarEvaluation{next, (state - 1.0) * (state - 1.0), std::abs(next - state),
                                residual};
      },
      [](double, ScalarEvaluation evaluation, const SelfConsistentProgress&) {
        return evaluation.proposed_state;
      },
      [](const SelfConsistentProgress&, const ScalarEvaluation&) {});

  require(outcome.converged, "residual-gated fixed point did not converge");
  require(outcome.progress.iteration == 4, "residual gate was not enforced");
}

void verify_nonconverged_state_retention() {
  const SelfConsistentPolicy policy{2, 1.0e-12, 1.0e-12, 1.0e-12, false};
  const auto outcome = run_self_consistent(
      0.0, policy,
      [](double state, unsigned) {
        const double next = state + 1.0;
        return ScalarEvaluation{next, 1.0, 1.0, 0.0};
      },
      [](double, ScalarEvaluation evaluation, const SelfConsistentProgress&) {
        return evaluation.proposed_state;
      },
      [](const SelfConsistentProgress&, const ScalarEvaluation&) {});

  require(!outcome.converged, "nonconvergent fixed point was marked converged");
  require(outcome.progress.iteration == 2, "max-iteration bookkeeping changed");
  require(outcome.state == 2.0, "last accepted nonconverged state was not retained");
}

void verify_accept_owns_update_policy() {
  const SelfConsistentPolicy policy{1, 0.0, 0.0, 0.0, false};
  const auto outcome = run_self_consistent(
      0.0, policy, [](double, unsigned) { return ScalarEvaluation{1.0, 0.0, 1.0, 0.0}; },
      [](double, ScalarEvaluation evaluation, const SelfConsistentProgress&) {
        return evaluation.proposed_state + 0.5;
      },
      [](const SelfConsistentProgress&, const ScalarEvaluation&) {});

  require(!outcome.converged, "one-step acceptance unexpectedly converged");
  require(outcome.state == 1.5, "driver bypassed method-owned acceptance policy");
}

void verify_diis_shape_rejection_preserves_history() {
  generativeqc::solver::Diis diis(3, 2);
  diis.update({0.0, 0.0}, {1.0, 0.0});
  for (const auto& sizes :
       {std::pair{1U, 2U}, std::pair{3U, 2U}, std::pair{2U, 1U}, std::pair{2U, 3U}}) {
    bool rejected = false;
    try {
      diis.update(std::vector<double>(sizes.first, 1.0), std::vector<double>(sizes.second, 1.0));
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "shared DIIS accepted inconsistent vector/error dimensions");
  }
  const auto next = diis.update({2.0, 4.0}, {0.0, 1.0});
  require(next.size() == 2 && std::abs(next[0] - 1.0) < 1e-14 && std::abs(next[1] - 2.0) < 1e-14 &&
              diis.restarts() == 0,
          "invalid DIIS update poisoned the retained history");
}

void verify_shared_diis_coefficient_policy() {
  using generativeqc::solver::detail::DiisCoefficientAction;
  using generativeqc::solver::detail::DiisCoefficientPolicy;
  using generativeqc::solver::detail::DiisMetricScaling;
  using generativeqc::solver::detail::solve_diis_coefficients;

  DiisCoefficientPolicy shared;
  shared.metric_scaling = DiisMetricScaling::MaximumAbsoluteEntry;
  shared.failure_retirement_floor = 1;
  shared.maximum_abs_coefficient = 1.0e6;

  std::vector<double> coefficients;
  auto action = solve_diis_coefficients({1.0, 0.0, 0.0, 1.0}, 2, shared, coefficients);
  require(action == DiisCoefficientAction::Extrapolate && coefficients.size() == 2 &&
              std::abs(coefficients[0] - 0.5) < 1.0e-14 &&
              std::abs(coefficients[1] - 0.5) < 1.0e-14,
          "shared DIIS coefficient solve changed");

  const double delta = 4.0e-7;
  const double off_diagonal = 1.0 + delta;
  const double second_diagonal = (1.0 + delta) * (1.0 + delta) + delta * delta;
  action = solve_diis_coefficients({1.0, off_diagonal, off_diagonal, second_diagonal}, 2, shared,
                                   coefficients);
  require(action == DiisCoefficientAction::RetireOldest,
          "shared DIIS coefficient guard stopped retiring unstable history");

  DiisCoefficientPolicy hf;
  action = solve_diis_coefficients({1.0, 1.0, 1.0, 1.0}, 2, hf, coefficients);
  require(action == DiisCoefficientAction::RetainCurrent,
          "HF-style singular DIIS unexpectedly retired history");

  DiisCoefficientPolicy ks;
  ks.metric_scaling = DiisMetricScaling::MaximumDiagonal;
  ks.failure_retirement_floor = 2;
  action = solve_diis_coefficients({1.0, 1.0, 1.0, 1.0}, 2, ks, coefficients);
  require(action == DiisCoefficientAction::RetainCurrent,
          "two-state KS-style singular DIIS unexpectedly retired history");
  action =
      solve_diis_coefficients({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 3, ks, coefficients);
  require(action == DiisCoefficientAction::RetireOldest,
          "KS-style singular DIIS stopped retiring older history");
}

void verify_method_neutral_diis() {
  generativeqc::solver::Diis disabled(0, 2);
  const std::vector<double> original{2.0, 4.0};
  require(disabled.update(original, {1.0, 0.0}) == original,
          "disabled shared DIIS changed the input state");
  require(disabled.restarts() == 0, "disabled shared DIIS reported a restart");

  generativeqc::solver::Diis diis(2, 2);
  require(diis.update({0.0, 0.0}, {1.0, 0.0}) == std::vector<double>({0.0, 0.0}),
          "first shared DIIS state changed");
  const auto extrapolated = diis.update({2.0, 4.0}, {0.0, 1.0});
  require(std::abs(extrapolated[0] - 1.0) < 1.0e-14 && std::abs(extrapolated[1] - 2.0) < 1.0e-14,
          "shared DIIS Pulay extrapolation changed");

  generativeqc::solver::Diis dependent(2, 2);
  dependent.update({0.0, 0.0}, {1.0, 1.0});
  const auto retained = dependent.update({3.0, 5.0}, {1.0, 1.0});
  require(retained == std::vector<double>({3.0, 5.0}),
          "shared DIIS did not retain the latest state after a singular history");
  require(dependent.restarts() == 1, "shared DIIS did not count dependent-history retirement");
}

void verify_three_history_diis_gram_symmetry() {
  generativeqc::solver::Diis diis(3, 3);
  require(diis.update({1.0, 2.0, 3.0}, {1.0, 0.0, 0.0}) == std::vector<double>({1.0, 2.0, 3.0}),
          "first three-history DIIS state changed");
  (void)diis.update({3.0, 5.0, 7.0}, {1.0, 1.0, 0.0});
  const auto extrapolated = diis.update({2.0, 4.0, 8.0}, {0.0, 1.0, 1.0});
  require(extrapolated.size() == 3 && std::abs(extrapolated[0] - 0.5) < 1e-14 &&
              std::abs(extrapolated[1] - 1.5) < 1e-14 && std::abs(extrapolated[2] - 3.5) < 1e-14 &&
              diis.restarts() == 0,
          "three-history shared DIIS Gram/extrapolation changed");
}

void verify_terminal_accept_can_keep_current_state() {
  const SelfConsistentPolicy policy{4, 1.0e-12, 1.0e-12, 1.0e-12, false};
  const auto outcome = run_self_consistent(
      0.0, policy, [](double, unsigned) { return ScalarEvaluation{42.0, 0.0, 0.0, 0.0}; },
      [](double& current, ScalarEvaluation evaluation, const SelfConsistentProgress& progress) {
        if (progress.converged) return std::move(current);
        return current + 1.0 + 0.0 * evaluation.proposed_state;
      },
      [](const SelfConsistentProgress&, const ScalarEvaluation&) {});

  require(outcome.converged, "terminal-retention problem did not converge");
  require(outcome.progress.iteration == 2, "terminal-retention convergence iteration changed");
  require(outcome.state == 1.0, "driver replaced a method-retained terminal state");
}

struct MoveOnlyState {
  int value;
  inline static unsigned moves = 0;
  explicit MoveOnlyState(int initial) : value(initial) {}
  MoveOnlyState(const MoveOnlyState&) = delete;
  MoveOnlyState& operator=(const MoveOnlyState&) = delete;
  MoveOnlyState(MoveOnlyState&& other) noexcept : value(other.value) { ++moves; }
  MoveOnlyState& operator=(MoveOnlyState&& other) noexcept {
    value = other.value;
    ++moves;
    return *this;
  }
};

void verify_default_eligibility_preserves_moves_and_order() {
  const SelfConsistentPolicy policy{4, 1e-12, 1e-12, 1e-12, true};
  std::vector<unsigned> default_trace;
  for (bool explicit_gate : {false, true}) {
    MoveOnlyState::moves = 0;
    std::vector<unsigned> trace;
    unsigned eligible_calls = 0;
    const auto evaluate = [&](const MoveOnlyState& state, unsigned iteration) {
      trace.push_back(10 * iteration + 1);
      require(state.value == static_cast<int>(iteration - 1), "state update order changed");
      return ScalarEvaluation{double(iteration), 0.0, 0.0, 0.0};
    };
    const auto record = [&](const SelfConsistentProgress& progress, const ScalarEvaluation&) {
      trace.push_back(10 * progress.iteration + 2);
      require(progress.converged == (progress.iteration == 2), "record convergence changed");
    };
    const auto accept = [&](MoveOnlyState&, ScalarEvaluation evaluation,
                            const SelfConsistentProgress& progress) {
      trace.push_back(10 * progress.iteration + 3);
      return MoveOnlyState(static_cast<int>(evaluation.proposed_state));
    };
    const auto eligible = [&](const SelfConsistentProgress& progress, const ScalarEvaluation&) {
      ++eligible_calls;
      require(progress.iteration == 2 && progress.converged, "eligibility ran before scalar pass");
      require(trace.back() == 21, "eligibility ran after record/accept");
      return true;
    };
    auto outcome =
        explicit_gate
            ? run_self_consistent(MoveOnlyState(0), policy, evaluate, accept, record, eligible)
            : run_self_consistent(MoveOnlyState(0), policy, evaluate, accept, record);
    require(outcome.converged && outcome.progress.iteration == 2 && outcome.state.value == 2,
            "default/always-true state or outcome changed");
    require(MoveOnlyState::moves == 4, "eligibility introduced a state move");
    require(eligible_calls == unsigned(explicit_gate), "wrong eligibility invocation count");
    if (!explicit_gate) default_trace = trace;
    require(trace == default_trace && trace == std::vector<unsigned>({11, 12, 13, 21, 22, 23}),
            "default/always-true evaluate-record-accept ordering changed");
  }
}

void verify_eligibility_veto_and_current_retention() {
  for (unsigned allow_at : {3U, 99U}) {
    const SelfConsistentPolicy policy{4, 1e-12, 1e-12, 1e-12, true};
    unsigned eligible_calls = 0, records = 0, accepts = 0;
    const auto outcome = run_self_consistent(
        0.0, policy,
        [](double state, unsigned) { return ScalarEvaluation{state + 1.0, 0.0, 0.0, 0.0}; },
        [&](double& current, ScalarEvaluation evaluation, const SelfConsistentProgress& progress) {
          ++accepts;
          require(records == accepts, "accept ran before record");
          if (progress.converged || progress.iteration == policy.max_iterations) return current;
          return evaluation.proposed_state;
        },
        [&](const SelfConsistentProgress& progress, const ScalarEvaluation&) {
          ++records;
          require(progress.converged == (progress.iteration >= allow_at), "record missed veto");
        },
        [&](const SelfConsistentProgress& progress, const ScalarEvaluation& evaluation) {
          ++eligible_calls;
          require(progress.iteration > 1 && progress.converged, "first/scalar failure eligible");
          require(records + 1 == progress.iteration && accepts == records,
                  "eligibility order changed");
          require(evaluation.proposed_state == double(progress.iteration),
                  "eligibility did not receive CURRENT evaluation");
          return progress.iteration >= allow_at;
        });
    const unsigned expected = std::min(allow_at, policy.max_iterations);
    require(outcome.converged == (allow_at <= policy.max_iterations), "veto convergence wrong");
    require(outcome.progress.iteration == expected && records == expected && accepts == expected &&
                eligible_calls == expected - 1,
            "veto extended budget or skipped controller work");
    require(outcome.state == double(expected - 1), "terminal CURRENT state was replaced");
  }
}

void verify_eligibility_cannot_override_scalar_failure() {
  for (unsigned failure = 0; failure < 4; ++failure) {
    const SelfConsistentPolicy policy{3, 1.0, 1.0, 1.0, true};
    unsigned calls = 0;
    const auto outcome = run_self_consistent(
        0.0, policy,
        [failure](double, unsigned iteration) {
          return ScalarEvaluation{
              42.0, failure == 0 ? double(iteration) : 0.0, failure == 1 ? 1.0 : 0.0,
              failure == 2 ? 1.0 : (failure == 3 ? std::numeric_limits<double>::quiet_NaN() : 0.0)};
        },
        [](double state, ScalarEvaluation, const SelfConsistentProgress&) { return state; },
        [](const SelfConsistentProgress&, const ScalarEvaluation&) {},
        [&](const SelfConsistentProgress&, const ScalarEvaluation&) {
          ++calls;
          return true;
        });
    require(!outcome.converged && outcome.progress.iteration == 3 && calls == 0,
            "eligibility overrode a scalar gate or ran on iteration one");
  }
  for (unsigned budget : {0U, 1U}) {
    unsigned calls = 0;
    const auto outcome = run_self_consistent(
        0.0, SelfConsistentPolicy{budget, 1.0, 1.0, 1.0, false},
        [](double, unsigned) { return ScalarEvaluation{}; },
        [](double state, ScalarEvaluation, const SelfConsistentProgress&) { return state; },
        [](const SelfConsistentProgress&, const ScalarEvaluation&) {},
        [&](const SelfConsistentProgress&, const ScalarEvaluation&) {
          ++calls;
          return true;
        });
    require(!outcome.converged && outcome.progress.iteration == budget && calls == 0,
            "zero/first iteration invoked eligibility");
  }
}

}  // namespace

int main() {
  try {
    verify_bounded_iteration_control();
    verify_default_eligibility_preserves_moves_and_order();
    verify_eligibility_veto_and_current_retention();
    verify_eligibility_cannot_override_scalar_failure();
    verify_basic_convergence();
    verify_residual_gate();
    verify_nonconverged_state_retention();
    verify_accept_owns_update_policy();
    verify_terminal_accept_can_keep_current_state();
    verify_shared_diis_coefficient_policy();
    verify_method_neutral_diis();
    verify_three_history_diis_gram_symmetry();
    verify_diis_shape_rejection_preserves_history();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
