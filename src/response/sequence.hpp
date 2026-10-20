#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

#include "response/solve.hpp"

namespace generativeqc::response {

/** Shape-only capacity for streaming independent RHS requests.
 *
 * Unlike the retaining/all-or-nothing batch API, this owner releases each native
 * result after synchronous publication. Earlier published solution/basis arrays
 * remain charged to the sequence. Publisher-owned history/object bookkeeping and
 * physical operator allocations have separate budgets. Each solve still uses a
 * problem-bound native response batch; no new GMRES equations live here.
 */
struct ResponseSequencePlan {
  GmresPlan solve_plan;
  std::size_t request_count{}, row_stride{}, column_stride{}, storage_count{};
  std::size_t staging_bytes{}, first_workspace_bytes{}, complete_workspace_bytes{};
  std::size_t max_numeric_bytes{};
};

struct ResponseSequenceResult {
  std::size_t published_count{}, converged_count{}, operator_actions{}, preconditioner_actions{};
  std::size_t peak_workspace_bytes{}, retained_solution_basis_bytes{}, staging_peak_bytes{};
  bool workspace_limited{};
};

using ResponsePublisher = std::function<void(std::size_t, const GmresResult&)>;
using ResponseValidator = std::function<void()>;

inline ResponseSequencePlan prepare_response_sequence(
    std::size_t dimension, const GmresOptions& options, std::size_t request_count,
    std::size_t row_stride, std::size_t column_stride, std::size_t storage_count,
    std::size_t max_numeric_bytes) {
  const auto solve_plan = prepare_gmres(dimension, options);
  ResponseSequencePlan plan{solve_plan, request_count, row_stride, column_stride, storage_count};
  plan.max_numeric_bytes = max_numeric_bytes;
  if (!request_count) return plan;
  if (!row_stride) throw std::invalid_argument("response RHS row stride must be positive");
  (void)detail::response_checked_multiply(storage_count, sizeof(double));
  const auto last = detail::response_checked_add(
      detail::response_checked_multiply(dimension - 1, row_stride),
      detail::response_checked_multiply(request_count - 1, column_stride));
  if (last >= storage_count) throw std::invalid_argument("response RHS storage is too small");
  const auto vector_bytes = detail::response_checked_multiply(dimension, sizeof(double));
  plan.staging_bytes = row_stride == 1 ? 0 : vector_bytes;
  plan.first_workspace_bytes =
      detail::response_checked_add(solve_plan.workspace_bytes, plan.staging_bytes);
  const auto published_bytes = detail::response_checked_add(
      vector_bytes, options.collect_basis
                        ? detail::response_checked_multiply(vector_bytes, solve_plan.restart)
                        : 0);
  plan.complete_workspace_bytes = detail::response_checked_add(
      plan.first_workspace_bytes,
      detail::response_checked_multiply(request_count - 1, published_bytes));
  return plan;
}

/** Stream a budget-admitted prefix; publish a shared refusal for the remaining tail.
 *
 * Input storage is borrowed read-only for the call. A strided column uses one
 * reusable tracked staging vector, allocated only after first admission. The
 * publisher may copy borrowed arrays but must not retain native spans. The GMRES
 * workspace polynomial also bounds the current result/publication-copy overlap;
 * earlier published solution/basis arrays are charged by their actual sizes.
 * Optional validation protects even zero RHS and resource-refused requests.
 */
inline ResponseSequenceResult solve_response_sequence(const ResponseSequencePlan& plan,
                                                      const LinearResponseProblem& problem,
                                                      std::span<const double> storage,
                                                      const ResponsePublisher& publish,
                                                      const LinearOperator& precondition = {},
                                                      const ResponseValidator& validate = {}) {
  const auto expected = prepare_response_sequence(
      problem.dimension(), plan.solve_plan.options, plan.request_count, plan.row_stride,
      plan.column_stride, plan.storage_count, plan.max_numeric_bytes);
  if (plan.solve_plan.dimension != expected.solve_plan.dimension ||
      plan.solve_plan.restart != expected.solve_plan.restart ||
      plan.solve_plan.workspace_bytes != expected.solve_plan.workspace_bytes ||
      plan.staging_bytes != expected.staging_bytes ||
      plan.first_workspace_bytes != expected.first_workspace_bytes ||
      plan.complete_workspace_bytes != expected.complete_workspace_bytes ||
      storage.size() != plan.storage_count)
    throw std::invalid_argument("response sequence plan does not match its inputs");
  if (plan.request_count && !publish)
    throw std::invalid_argument("response sequence publisher is empty");
  ResponseSequenceResult result;
  runtime::TrackedVector<double> staging;
  const auto check = [&] {
    if (validate) validate();
  };
  if (!plan.request_count) {
    check();
    return result;
  }
  for (std::size_t column = 0; column < plan.request_count; ++column) {
    check();
    const auto required = detail::response_checked_add(plan.first_workspace_bytes,
                                                       result.retained_solution_basis_bytes);
    result.peak_workspace_bytes = std::max(result.peak_workspace_bytes, required);
    if (plan.solve_plan.options.max_workspace_bytes < plan.solve_plan.workspace_bytes ||
        required > plan.max_numeric_bytes) {
      GmresResult refused;
      refused.status = GmresStatus::workspace_limit;
      refused.residual_norm = refused.relative_residual = std::numeric_limits<double>::infinity();
      refused.workspace_bytes = plan.solve_plan.workspace_bytes;
      result.workspace_limited = true;
      for (; column < plan.request_count; ++column) {
        check();
        publish(column, refused);
        ++result.published_count;
        check();
      }
      break;
    }
    const auto* source = storage.data() + column * plan.column_stride;
    std::span<const double> rhs(source, problem.dimension());
    if (plan.staging_bytes) {
      if (staging.empty()) {
        staging = runtime::TrackedVector<double>(
            problem.dimension(), 0.0,
            runtime::TrackedAllocator<double>(std::make_shared<runtime::AllocationCounter>()));
        result.staging_peak_bytes = staging.get_allocator().counter()->snapshot().peak_bytes;
      }
      for (std::size_t row = 0; row < problem.dimension(); ++row)
        staging[row] = source[row * plan.row_stride];
      rhs = staging;
    }
    const auto batch_plan = prepare_response_batch(problem, plan.solve_plan.options, 1,
                                                   plan.solve_plan.workspace_bytes);
    const std::array<ResponseSolveRequest, 1> request{
        ResponseSolveRequest{rhs, {}, {}, precondition}};
    auto batch = solve_response_batch(batch_plan, problem, request);
    const auto& solved = batch.responses.front();
    check();
    publish(column, solved);
    ++result.published_count;
    result.converged_count += solved.converged();
    result.operator_actions += solved.operator_actions;
    result.preconditioner_actions += solved.preconditioner_actions;
    result.retained_solution_basis_bytes = detail::response_checked_add(
        result.retained_solution_basis_bytes,
        detail::response_checked_multiply(
            detail::response_checked_add(solved.solution.size(), solved.basis.size()),
            sizeof(double)));
    check();
  }
  return result;
}

}  // namespace generativeqc::response
