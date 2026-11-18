#include "response/native_gmres.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "runtime/df_progress_trace.hpp"

namespace generativeqc::response {
namespace {
using WorkspaceVector = runtime::TrackedVector<double>;

std::size_t checked_add(std::size_t first, std::size_t second) {
  if (second > std::numeric_limits<std::size_t>::max() - first)
    throw std::overflow_error("GMRES workspace size overflow");
  return first + second;
}

std::size_t checked_multiply(std::size_t first, std::size_t second) {
  if (first && second > std::numeric_limits<std::size_t>::max() / first)
    throw std::overflow_error("GMRES workspace size overflow");
  return first * second;
}

bool finite(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

double relative_residual(double residual, double rhs) {
  if (rhs > 0.0) return residual / rhs;
  return residual == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
}

GmresResult build_result(const GmresPlan& plan, WorkspaceVector solution, GmresStatus status,
                         double residual, double rhs_norm, std::size_t iterations,
                         std::size_t restarts, std::size_t operator_actions,
                         std::size_t preconditioner_actions) {
  GmresResult result;
  result.solution = std::move(solution);
  result.status = status;
  result.residual_norm = residual;
  result.relative_residual = relative_residual(residual, rhs_norm);
  result.iterations = iterations;
  result.restarts = restarts;
  result.operator_actions = operator_actions;
  result.preconditioner_actions = preconditioner_actions;
  result.workspace_bytes = plan.workspace_bytes;
  if (const auto counter = result.solution.get_allocator().counter()) {
    const auto measured = counter->snapshot();
    result.measured_workspace_peak_bytes = measured.peak_bytes;
    result.workspace_allocation_count = measured.allocation_count;
  }
  return result;
}

bool solve_upper(std::span<const double> hessenberg, std::size_t stride,
                 std::span<const double> transformed_rhs, std::size_t count,
                 std::span<double> solution) {
  std::fill(solution.begin(), solution.end(), 0.0);
  for (std::size_t reverse = count; reverse > 0; --reverse) {
    const auto row = reverse - 1;
    double value = transformed_rhs[row];
    for (std::size_t column = row + 1; column < count; ++column)
      value -= hessenberg[row * stride + column] * solution[column];
    const double diagonal = hessenberg[row * stride + row];
    // Attempt every finite, nonzero pivot; a scale-dependent pivot floor would
    // reject valid systems. Finite coefficients and true residuals gate success.
    if (!std::isfinite(diagonal) || diagonal == 0.0) return false;
    solution[row] = value / diagonal;
    if (!std::isfinite(solution[row])) return false;
  }
  return true;
}

void validate_plan(const GmresPlan& plan) {
  const auto expected = prepare_gmres(plan.dimension, plan.options);
  if (plan.restart != expected.restart || plan.workspace_bytes != expected.workspace_bytes)
    throw std::invalid_argument("GMRES plan does not match its dimensions and options");
}

/** The single numerical owner for serial and independently batched solves.
 * Only physical operator calls suspend execution. All arrays retain an explicit
 * per-solve allocation domain; no coroutine frame, thread or borrowed numerical
 * state hides additional Krylov payload outside prepare_gmres's array bound.
 */
class GmresState {
 public:
  GmresState(const GmresPlan& plan, const ResponseSolveRequest& request, bool has_operator,
             const LinearOperator& right_preconditioner)
      : plan_(plan),
        request_(request),
        right_preconditioner_(right_preconditioner),
        allocator_(std::make_shared<runtime::AllocationCounter>()),
        solution_(allocator_),
        image_(allocator_),
        residual_(allocator_),
        basis_(allocator_),
        preconditioned_(allocator_),
        hessenberg_(allocator_),
        cosine_(allocator_),
        sine_(allocator_),
        transformed_(allocator_),
        work_(allocator_),
        candidate_(allocator_),
        candidate_image_(allocator_),
        candidate_residual_(allocator_),
        best_solution_(allocator_),
        best_residual_(allocator_),
        coefficients_(allocator_),
        base_solution_(allocator_),
        history_(allocator_),
        best_basis_(allocator_) {
    const auto dimension = plan_.dimension;
    solution_.assign(dimension, 0.0);
    if (request_.rhs.size() != dimension || !finite(request_.rhs) ||
        (!request_.initial_guess.empty() &&
         (request_.initial_guess.size() != dimension || !finite(request_.initial_guess))) ||
        (!request_.diagonal_preconditioner.empty() &&
         (request_.diagonal_preconditioner.size() != dimension ||
          !finite(request_.diagonal_preconditioner)))) {
      finish(std::move(solution_), GmresStatus::nonfinite_input,
             std::numeric_limits<double>::infinity());
      return;
    }
    if (!has_operator) throw std::invalid_argument("GMRES operator callback is empty");
    if (right_preconditioner_ && !request_.diagonal_preconditioner.empty())
      throw std::invalid_argument(
          "GMRES preconditioner callback and diagonal are mutually exclusive");
    for (double value : request_.diagonal_preconditioner)
      if (std::abs(value) <= plan_.options.breakdown_tolerance)
        throw std::invalid_argument("GMRES diagonal preconditioner contains a zero entry");
    if (!request_.initial_guess.empty())
      std::copy(request_.initial_guess.begin(), request_.initial_guess.end(), solution_.begin());
    if (plan_.options.collect_history) history_.reserve(plan_.options.max_iterations + 2);
    try {
      rhs_norm_ = stable_norm(request_.rhs);
    } catch (const std::exception&) {
      finish(std::move(solution_), GmresStatus::nonfinite_input,
             std::numeric_limits<double>::infinity());
      return;
    }
    target_ =
        std::max(plan_.options.absolute_tolerance, plan_.options.relative_tolerance * rhs_norm_);
    image_.assign(dimension, 0.0);
    residual_.assign(request_.rhs.begin(), request_.rhs.end());
    if (!request_.initial_guess.empty()) {
      suspend(Phase::initial, solution_, image_);
    } else {
      initialize_residual();
    }
  }

  [[nodiscard]] bool done() const noexcept { return phase_ == Phase::done; }
  [[nodiscard]] LinearAction action() const noexcept { return action_; }
  [[nodiscard]] const char* action_name() const noexcept {
    switch (phase_) {
      case Phase::initial:
        return "gmres_initial_residual";
      case Phase::arnoldi:
        return "gmres_arnoldi_action";
      case Phase::candidate:
        return "gmres_exact_residual_replay";
      case Phase::done:
        return "gmres_complete";
    }
    return "gmres_unknown";
  }
  [[nodiscard]] bool collect_timing() const noexcept { return plan_.options.collect_timing; }
  GmresResult take_result() { return std::move(result_); }

  /** Resume only after a newly evaluated physical action, including every
   * requested true residual. Finite/preconditioner gates stay per RHS.
   */
  void complete_action(double elapsed) {
    ++operator_actions_;
    if (collect_timing()) operator_seconds_ += elapsed;
    if (!finite(action_.output)) {
      const bool initial = phase_ == Phase::initial;
      finish(initial ? std::move(solution_) : std::move(best_solution_),
             GmresStatus::nonfinite_operator,
             initial ? std::numeric_limits<double>::infinity() : best_norm_);
      return;
    }
    switch (phase_) {
      case Phase::initial:
        for (std::size_t index = 0; index < plan_.dimension; ++index)
          residual_[index] -= image_[index];
        initialize_residual();
        break;
      case Phase::arnoldi:
        complete_arnoldi();
        break;
      case Phase::candidate:
        complete_candidate();
        break;
      case Phase::done:
        throw std::logic_error("completed GMRES state cannot resume an action");
    }
  }

 private:
  enum class Phase { initial, arnoldi, candidate, done };

  void finish(WorkspaceVector solution, GmresStatus status, double residual) {
    result_ = build_result(plan_, std::move(solution), status, residual, rhs_norm_, iterations_,
                           restarts_, operator_actions_, preconditioner_actions_);
    result_.history = std::move(history_);
    result_.basis = std::move(best_basis_);
    result_.basis_columns = best_basis_columns_;
    result_.orthogonalization_seconds = orthogonalization_seconds_;
    result_.operator_seconds = operator_seconds_;
    phase_ = Phase::done;
    action_ = {};
  }

  void suspend(Phase phase, std::span<const double> input, std::span<double> output) {
    std::fill(output.begin(), output.end(), 0.0);
    action_ = {input, output};
    phase_ = phase;
  }

  void initialize_residual() {
    try {
      beta_ = stable_norm(residual_);
    } catch (const std::exception&) {
      finish(std::move(solution_), GmresStatus::nonfinite_input,
             std::numeric_limits<double>::infinity());
      return;
    }
    if (plan_.options.collect_history) history_.push_back(beta_);
    if (beta_ <= target_) {
      finish(std::move(solution_), GmresStatus::initial_residual, beta_);
      return;
    }
    const auto dimension = plan_.dimension;
    const auto restart = plan_.restart;
    basis_.assign((restart + 1) * dimension, 0.0);
    preconditioned_.assign(restart * dimension, 0.0);
    hessenberg_.assign((restart + 1) * restart, 0.0);
    cosine_.assign(restart, 0.0);
    sine_.assign(restart, 0.0);
    transformed_.assign(restart + 1, 0.0);
    work_.assign(dimension, 0.0);
    candidate_.assign(dimension, 0.0);
    candidate_image_.assign(dimension, 0.0);
    candidate_residual_.assign(dimension, 0.0);
    best_solution_ = solution_;
    best_residual_ = residual_;
    coefficients_.assign(restart, 0.0);
    if (plan_.options.collect_basis) best_basis_.reserve(dimension * restart);
    begin_cycle();
  }

  void begin_cycle() {
    std::fill(basis_.begin(), basis_.end(), 0.0);
    std::fill(preconditioned_.begin(), preconditioned_.end(), 0.0);
    std::fill(hessenberg_.begin(), hessenberg_.end(), 0.0);
    std::fill(transformed_.begin(), transformed_.end(), 0.0);
    transformed_[0] = beta_;
    for (std::size_t index = 0; index < plan_.dimension; ++index)
      basis_[index] = residual_[index] / beta_;
    base_solution_ = solution_;
    best_norm_ = beta_;
    column_ = 0;
    prepare_column();
  }

  void prepare_column() {
    const auto dimension = plan_.dimension;
    const auto basis_column =
        std::span<const double>(basis_.data() + column_ * dimension, dimension);
    auto z_column = std::span<double>(preconditioned_.data() + column_ * dimension, dimension);
    if (right_preconditioner_) {
      ++preconditioner_actions_;
      std::fill(z_column.begin(), z_column.end(), 0.0);
      right_preconditioner_(basis_column, z_column);
    } else if (request_.diagonal_preconditioner.empty()) {
      std::copy(basis_column.begin(), basis_column.end(), z_column.begin());
    } else {
      ++preconditioner_actions_;
      for (std::size_t index = 0; index < dimension; ++index)
        z_column[index] = basis_column[index] / request_.diagonal_preconditioner[index];
    }
    if (!finite(z_column)) {
      finish(std::move(best_solution_), GmresStatus::nonfinite_preconditioner, best_norm_);
      return;
    }
    suspend(Phase::arnoldi, z_column, work_);
  }

  void complete_arnoldi() {
    const auto dimension = plan_.dimension;
    const auto restart = plan_.restart;
    const auto begin = collect_timing() ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
    for (unsigned pass = 0; pass < plan_.options.reorthogonalize; ++pass)
      for (std::size_t row = 0; row <= column_; ++row) {
        double projection = 0.0;
        for (std::size_t index = 0; index < dimension; ++index)
          projection += basis_[row * dimension + index] * work_[index];
        hessenberg_[row * restart + column_] += projection;
        for (std::size_t index = 0; index < dimension; ++index)
          work_[index] -= projection * basis_[row * dimension + index];
      }
    double next_norm = 0.0;
    try {
      next_norm = stable_norm(work_);
    } catch (const std::exception&) {
      finish(std::move(best_solution_), GmresStatus::nonfinite_operator, best_norm_);
      return;
    }
    if (collect_timing())
      orthogonalization_seconds_ +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    hessenberg_[(column_ + 1) * restart + column_] = next_norm;
    broke_down_ = next_norm <= plan_.options.breakdown_tolerance;
    if (!broke_down_)
      for (std::size_t index = 0; index < dimension; ++index)
        basis_[(column_ + 1) * dimension + index] = work_[index] / next_norm;
    for (std::size_t row = 0; row < column_; ++row) {
      const double upper = hessenberg_[row * restart + column_];
      const double lower = hessenberg_[(row + 1) * restart + column_];
      hessenberg_[row * restart + column_] = cosine_[row] * upper + sine_[row] * lower;
      hessenberg_[(row + 1) * restart + column_] = -sine_[row] * upper + cosine_[row] * lower;
    }
    const double upper = hessenberg_[column_ * restart + column_];
    const double lower = hessenberg_[(column_ + 1) * restart + column_];
    const double rotation_norm = std::hypot(upper, lower);
    if (rotation_norm > plan_.options.breakdown_tolerance) {
      cosine_[column_] = upper / rotation_norm;
      sine_[column_] = lower / rotation_norm;
    } else {
      cosine_[column_] = 1.0;
      sine_[column_] = 0.0;
    }
    hessenberg_[column_ * restart + column_] = cosine_[column_] * upper + sine_[column_] * lower;
    hessenberg_[(column_ + 1) * restart + column_] = 0.0;
    const double transformed_upper = transformed_[column_];
    transformed_[column_] = cosine_[column_] * transformed_upper;
    transformed_[column_ + 1] = -sine_[column_] * transformed_upper;
    ++iterations_;
    // Predicted convergence requests, but never replaces, a new physical
    // candidate-residual action. The numerical checkpoint policy is unchanged.
    const bool checkpoint = std::abs(transformed_[column_ + 1]) <= target_ || broke_down_ ||
                            column_ + 1 == restart || iterations_ == plan_.options.max_iterations ||
                            iterations_ % plan_.options.true_residual_every == 0;
    if (!checkpoint) {
      ++column_;
      prepare_column();
      return;
    }
    const auto columns = column_ + 1;
    solvable_ = solve_upper(hessenberg_, restart, transformed_, columns, coefficients_);
    candidate_ = base_solution_;
    if (solvable_)
      for (std::size_t vector = 0; vector < columns; ++vector)
        for (std::size_t index = 0; index < dimension; ++index)
          candidate_[index] += preconditioned_[vector * dimension + index] * coefficients_[vector];
    // Sparse operators need not observe an overflowing coordinate. Refuse it
    // here rather than allowing such an operator to certify a false residual.
    if (!finite(candidate_)) {
      finish(std::move(best_solution_), GmresStatus::breakdown, best_norm_);
      return;
    }
    suspend(Phase::candidate, candidate_, candidate_image_);
  }

  void complete_candidate() {
    for (std::size_t index = 0; index < plan_.dimension; ++index)
      candidate_residual_[index] = request_.rhs[index] - candidate_image_[index];
    double candidate_norm = 0.0;
    try {
      candidate_norm = stable_norm(candidate_residual_);
    } catch (const std::exception&) {
      finish(std::move(best_solution_), GmresStatus::nonfinite_operator, best_norm_);
      return;
    }
    if (plan_.options.collect_history) history_.push_back(candidate_norm);
    if (candidate_norm < best_norm_) {
      best_norm_ = candidate_norm;
      best_solution_ = candidate_;
      best_residual_ = candidate_residual_;
      stagnation_ = 0;
      if (plan_.options.collect_basis) {
        best_basis_.assign(basis_.begin(), basis_.begin() + (column_ + 1) * plan_.dimension);
        best_basis_columns_ = column_ + 1;
      }
    } else if (candidate_norm >= best_norm_ * (1.0 - plan_.options.stagnation_tolerance)) {
      ++stagnation_;
    }
    if (candidate_norm <= target_) {
      finish(std::move(candidate_), GmresStatus::converged, candidate_norm);
    } else if (broke_down_ || !solvable_) {
      finish(std::move(best_solution_), GmresStatus::breakdown, best_norm_);
    } else if (stagnation_ >= plan_.options.stagnation_window) {
      finish(std::move(best_solution_), GmresStatus::stagnation, best_norm_);
    } else if (column_ + 1 == plan_.restart || iterations_ == plan_.options.max_iterations) {
      solution_ = best_solution_;
      residual_ = best_residual_;
      beta_ = best_norm_;
      if (iterations_ >= plan_.options.max_iterations || beta_ == 0.0) {
        finish(std::move(best_solution_), GmresStatus::max_iterations, beta_);
      } else {
        ++restarts_;
        begin_cycle();
      }
    } else {
      ++column_;
      prepare_column();
    }
  }

  const GmresPlan& plan_;
  const ResponseSolveRequest& request_;
  const LinearOperator& right_preconditioner_;
  runtime::TrackedAllocator<double> allocator_;
  WorkspaceVector solution_, image_, residual_, basis_, preconditioned_, hessenberg_, cosine_,
      sine_, transformed_, work_, candidate_, candidate_image_, candidate_residual_, best_solution_,
      best_residual_, coefficients_, base_solution_, history_, best_basis_;
  GmresResult result_;
  LinearAction action_;
  Phase phase_{Phase::done};
  double rhs_norm_{}, target_{}, beta_{}, best_norm_{};
  double orthogonalization_seconds_{}, operator_seconds_{};
  std::size_t column_{}, iterations_{}, restarts_{}, stagnation_{}, operator_actions_{};
  std::size_t preconditioner_actions_{}, best_basis_columns_{};
  bool broke_down_{}, solvable_{};
};
}  // namespace

double stable_norm(std::span<const double> values) {
  double scale = 0.0;
  for (double value : values) {
    if (!std::isfinite(value))
      throw std::invalid_argument("GMRES norm requires finite vector values");
    scale = std::max(scale, std::abs(value));
  }
  if (scale == 0.0) return 0.0;
  double squares = 0.0;
  for (double value : values) {
    const double scaled = value / scale;
    squares += scaled * scaled;
  }
  const double result = scale * std::sqrt(squares);
  if (!std::isfinite(result)) throw std::overflow_error("GMRES vector norm overflows FP64");
  return result;
}

GmresPlan prepare_gmres(std::size_t dimension, const GmresOptions& options) {
  if (!dimension || !options.restart || !options.max_iterations || !options.max_workspace_bytes ||
      !options.reorthogonalize || !options.true_residual_every || !options.stagnation_window ||
      !std::isfinite(options.relative_tolerance) || options.relative_tolerance < 0.0 ||
      options.relative_tolerance >= 1.0 || !std::isfinite(options.absolute_tolerance) ||
      options.absolute_tolerance < 0.0 || !std::isfinite(options.breakdown_tolerance) ||
      options.breakdown_tolerance < 0.0 || !std::isfinite(options.stagnation_tolerance) ||
      options.stagnation_tolerance < 0.0)
    throw std::invalid_argument("invalid bounded GMRES options");
  const auto restart = std::min({dimension, options.restart, options.max_iterations});
  const auto restart_plus_one = checked_add(restart, 1);
  const auto iterations_plus_two = checked_add(options.max_iterations, 2);
  // The Arnoldi/preconditioned bases and ten vector panels, plus publication
  // headroom. Every independently suspended state owns and prices this payload.
  const auto vector_slots = checked_add(checked_multiply(2, restart), 12);
  std::size_t elements = checked_multiply(vector_slots, dimension);
  elements = checked_add(elements, checked_multiply(restart_plus_one, restart));
  elements = checked_add(elements, checked_add(checked_multiply(4, restart), 1));
  if (options.collect_basis) elements = checked_add(elements, checked_multiply(dimension, restart));
  if (options.collect_history) elements = checked_add(elements, iterations_plus_two);
  return {dimension, restart, checked_multiply(elements, sizeof(double)), options};
}

GmresResult solve_gmres(const GmresPlan& plan, const LinearOperator& apply,
                        std::span<const double> rhs, std::span<const double> initial_guess,
                        std::span<const double> diagonal_preconditioner,
                        const LinearOperator& right_preconditioner) {
  validate_plan(plan);
  if (plan.options.max_workspace_bytes < plan.workspace_bytes)
    return build_result(plan, {}, GmresStatus::workspace_limit,
                        std::numeric_limits<double>::infinity(), 0.0, 0, 0, 0, 0);
  const ResponseSolveRequest request{rhs, initial_guess, diagonal_preconditioner};
  GmresState state(plan, request, static_cast<bool>(apply), right_preconditioner);
  std::size_t actions = 0;
  while (!state.done()) {
    runtime::df_progress::Scope action(state.action_name());
    const auto begin = state.collect_timing() ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
    const auto pending = state.action();
    apply(pending.input, pending.output);
    const double elapsed =
        state.collect_timing()
            ? std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
            : 0.0;
    if (action.enabled()) runtime::df_progress::Scope::number("operator_action", ++actions);
    action.finish();
    state.complete_action(elapsed);
  }
  return state.take_result();
}

std::size_t gmres_batch_workspace_bytes(const GmresPlan& plan, std::size_t request_count,
                                        std::size_t concurrent_requests) {
  validate_plan(plan);
  if (!concurrent_requests || concurrent_requests > 4)
    throw std::invalid_argument("independent GMRES batch width must be one through four");
  if (!request_count) return 0;
  auto retained = plan.dimension;
  if (plan.options.collect_basis)
    retained = checked_add(retained, checked_multiply(plan.dimension, plan.restart));
  if (plan.options.collect_history)
    retained = checked_add(retained, checked_add(plan.options.max_iterations, 2));
  retained = checked_multiply(retained, sizeof(double));
  const auto active = std::min(concurrent_requests, request_count);
  return checked_add(checked_multiply(active, plan.workspace_bytes),
                     checked_multiply(request_count - active, retained));
}

std::vector<GmresResult> solve_gmres_batch(const GmresPlan& plan, const BatchLinearOperator& apply,
                                           std::span<const ResponseSolveRequest> requests,
                                           std::size_t concurrent_requests,
                                           std::size_t max_numeric_bytes) {
  const auto needed = gmres_batch_workspace_bytes(plan, requests.size(), concurrent_requests);
  if (needed > max_numeric_bytes ||
      (!requests.empty() && plan.workspace_bytes > plan.options.max_workspace_bytes))
    return {};
  std::vector<GmresResult> results(requests.size());
  // Fixed-size descriptors contain only spans/control data, never numerical
  // panels. Completed states release their full workspaces before another cohort.
  std::array<std::unique_ptr<GmresState>, 4> states;
  std::array<LinearAction, 4> actions;
  std::array<std::size_t, 4> owners;
  for (std::size_t first = 0; first < requests.size();) {
    const auto count = std::min(concurrent_requests, requests.size() - first);
    for (std::size_t index = 0; index < count; ++index)
      states[index] =
          std::make_unique<GmresState>(plan, requests[first + index], static_cast<bool>(apply),
                                       requests[first + index].right_preconditioner);
    while (true) {
      std::size_t pending = 0;
      for (std::size_t index = 0; index < count; ++index) {
        if (!states[index]) continue;
        if (states[index]->done()) {
          results[first + index] = states[index]->take_result();
          states[index].reset();
        } else {
          owners[pending] = index;
          actions[pending++] = states[index]->action();
        }
      }
      if (!pending) break;
      runtime::df_progress::Scope action("gmres_independent_actions");
      if (action.enabled()) runtime::df_progress::Scope::number("physical_actions", pending);
      const auto begin = plan.options.collect_timing ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point{};
      apply(std::span<const LinearAction>(actions.data(), pending));
      const double elapsed =
          plan.options.collect_timing
              ? std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
              : 0.0;
      action.finish();
      for (std::size_t index = 0; index < pending; ++index)
        states[owners[index]]->complete_action(elapsed);
    }
    first += count;
  }
  return results;
}

}  // namespace generativeqc::response
