#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "generativeqc/generativeqc.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Operator {
  std::array<double, 9> matrix{{4, 1, -1, 0, 3, 2, 1, 0, 2}};
  std::uint64_t actions{};
  bool fail{}, nonfinite{};
};

generativeqc_status apply(void* data, const double* input, double* output, std::size_t dimension) {
  auto& action = *static_cast<Operator*>(data);
  ++action.actions;
  if (action.fail) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  if (dimension != 3) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  for (std::size_t row = 0; row < dimension; ++row) {
    output[row] = 0;
    for (std::size_t column = 0; column < dimension; ++column)
      output[row] += action.matrix[row * dimension + column] * input[column];
  }
  if (action.nonfinite) output[0] = std::numeric_limits<double>::quiet_NaN();
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_response_gmres_descriptor_v1 request() {
  generativeqc_response_gmres_descriptor_v1 descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.dimension = 3;
  descriptor.rtol = 1e-12;
  descriptor.restart = 3;
  descriptor.max_iterations = 16;
  descriptor.max_workspace_bytes = 1 << 20;
  descriptor.reorthogonalize = 2;
  descriptor.true_residual_every = 1;
  descriptor.stagnation_window = 25;
  descriptor.breakdown_tolerance = 1e-14;
  descriptor.stagnation_tolerance = 1e-14;
  descriptor.flags = GENERATIVEQC_RESPONSE_COLLECT_HISTORY | GENERATIVEQC_RESPONSE_COLLECT_BASIS |
                     GENERATIVEQC_RESPONSE_COLLECT_TIMING;
  return descriptor;
}

void run() {
  auto descriptor = request();
  std::uint64_t required = 0;
  generativeqc::check(generativeqc_response_gmres_workspace_v1(&descriptor, &required));
  require(required > 0, "shape-only query failed without arrays/callbacks");
  Operator action;
  std::array<double, 3> rhs{{1.5, -5, 2}};
  descriptor.rhs = rhs.data();
  descriptor.apply = apply;
  descriptor.apply_data = &action;
  descriptor.max_workspace_bytes = required;
  generativeqc::ResponseResult solved(descriptor);
  const auto data = solved.data();
  require(data.converged && data.reason == GENERATIVEQC_RESPONSE_CONVERGED && data.dimension == 3 &&
              data.solution_count == 3,
          "public nonsymmetric response did not converge");
  require(std::abs(data.solution[0] - 1) < 2e-12 && std::abs(data.solution[1] + 2) < 2e-12 &&
              std::abs(data.solution[2] - 0.5) < 2e-12,
          "response differs from independent exact solution");
  require(data.history_count > 0 && data.history[data.history_count - 1] == data.residual_norm &&
              data.operator_actions == action.actions && data.basis_columns > 0 &&
              data.basis_columns <= 3 && data.basis && data.measured_workspace_peak_bytes > 0 &&
              data.measured_workspace_peak_bytes <= required,
          "public numeric/work/trace or exact-budget contract changed");
  for (std::size_t column = 0; column < data.basis_columns; ++column)
    for (std::size_t other = 0; other < data.basis_columns; ++other) {
      double product = 0;
      for (std::size_t row = 0; row < 3; ++row)
        product += data.basis[3 * column + row] * data.basis[3 * other + row];
      require(std::abs(product - (column == other ? 1.0 : 0.0)) < 2e-12,
              "borrowed column-major basis is not orthonormal");
    }

  descriptor.max_workspace_bytes = required - 1;
  const auto before = action.actions;
  generativeqc::ResponseResult refused(descriptor);
  const auto shortage = refused.data();
  require(!shortage.converged && shortage.reason == GENERATIVEQC_RESPONSE_WORKSPACE_LIMIT &&
              !shortage.solution && shortage.solution_count == 0 && !shortage.history &&
              !shortage.basis && shortage.measured_workspace_peak_bytes == 0 &&
              action.actions == before,
          "workspace refusal allocated vectors or invoked the physical callback");
  descriptor.max_workspace_bytes = required;

  auto failure = [&](generativeqc_status expected) {
    auto* output = reinterpret_cast<generativeqc_response_result*>(std::uintptr_t{1});
    const auto status = generativeqc_response_solve_v1(&descriptor, &output);
    require(status == expected && output == nullptr, "failed solve published a result");
  };
  descriptor.abi_version += 1;
  failure(GENERATIVEQC_STATUS_ABI_MISMATCH);
  descriptor.abi_version -= 1;
  descriptor.flags |= 1u << 20;
  failure(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  descriptor.flags &= ~(1u << 20);
  descriptor.rtol = std::numeric_limits<double>::quiet_NaN();
  failure(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  descriptor.rtol = 1e-12;
  action.fail = true;
  failure(GENERATIVEQC_STATUS_NUMERICAL_FAILURE);
  action.fail = false;
  action.nonfinite = true;
  generativeqc::ResponseResult nonfinite(descriptor);
  require(nonfinite.data().reason == GENERATIVEQC_RESPONSE_NONFINITE_OPERATOR &&
              !nonfinite.data().converged,
          "nonfinite action falsely converged");
  action.nonfinite = false;
  descriptor.flags = 0;
  generativeqc::ResponseResult recovered(descriptor);
  require(recovered.data().converged && !recovered.data().history &&
              recovered.data().history_count == 0 && !recovered.data().basis &&
              recovered.data().basis_columns == 0,
          "independent recovery retained an optional trace or failed");

  auto overflow = request();
  overflow.dimension = std::numeric_limits<std::uint64_t>::max();
  required = 123;
  require(generativeqc_response_gmres_workspace_v1(&overflow, &required) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              required == 123,
          "overflowing metadata query changed output or allocated dimension-sized state");

  std::unique_ptr<generativeqc::ResponseResult> retained;
  {
    Operator temporary;
    std::array<double, 3> values{{1.5, -5, 2}};
    auto local = request();
    local.rhs = values.data();
    local.apply = apply;
    local.apply_data = &temporary;
    generativeqc::ResponseResult owner(local);
    retained = std::make_unique<generativeqc::ResponseResult>(std::move(owner));
    values.fill(99);
    temporary.matrix.fill(0);
  }
  require(retained->data().converged && std::abs(retained->data().solution[1] + 2) < 2e-12,
          "native result borrowed destroyed input/callback state or lost move ownership");
}

void small_pivot_contracts() {
  for (double scale : {1e-170, -1e-170, 1e-15, 1.0, -1e170, 1e170}) {
    for (bool identity : {false, true}) {
      auto descriptor = request();
      Operator action;
      std::array<double, 3> rhs{{1.5 * scale, -5 * scale, 2 * scale}};
      if (identity) {
        action.matrix = {{scale, 0, 0, 0, scale, 0, 0, 0, scale}};
        rhs = {{scale, -2 * scale, 0.5 * scale}};
      } else {
        // General small operators still need an explicit exact-zero Arnoldi
        // threshold; this repair changes only triangular pivot admission.
        descriptor.breakdown_tolerance = 0;
        for (double& entry : action.matrix) entry *= scale;
      }
      descriptor.rhs = rhs.data();
      descriptor.apply = apply;
      descriptor.apply_data = &action;
      generativeqc::ResponseResult result(descriptor);
      const auto data = result.data();
      require(data.converged && data.relative_residual <= descriptor.rtol &&
                  data.operator_actions == action.actions,
              "finite small pivots changed convergence or actual work");
      require(std::abs(data.solution[0] - 1) < 2e-12 && std::abs(data.solution[1] + 2) < 2e-12 &&
                  std::abs(data.solution[2] - 0.5) < 2e-12,
              "small-pivot solve differs from the independent exact solution");
    }
  }

  auto descriptor = request();
  Operator action;
  descriptor.apply = apply;
  descriptor.apply_data = &action;
  for (bool nonfinite_coefficient : {false, true}) {
    action.matrix.fill(0);
    if (nonfinite_coefficient) action.matrix[0] = 1e-309;
    std::array<double, 3> rhs{{1, 0, 0}};
    descriptor.rhs = rhs.data();
    generativeqc::ResponseResult failed(descriptor);
    require(!failed.data().converged && failed.data().relative_residual == 1 &&
                std::isfinite(failed.data().solution[0]),
            "singular or nonfinite backsolve falsely converged");
  }

  // A sparse physical action can ignore an overflowing nullspace coordinate.
  // Its finite residual must never certify an infinite preconditioned solution.
  std::uint64_t actions = 0;
  std::array<double, 3> rhs{{1, 0, 0}};
  descriptor.rhs = rhs.data();
  descriptor.apply_data = &actions;
  descriptor.apply = [](void* data, const double* input, double* output,
                        std::size_t n) -> generativeqc_status {
    ++*static_cast<std::uint64_t*>(data);
    for (std::size_t i = 0; i < n; ++i)
      require(std::isfinite(input[i]), "operator received a nonfinite solution candidate");
    output[0] = 1e-15 * input[0];
    output[1] = output[2] = 0;
    return GENERATIVEQC_STATUS_SUCCESS;
  };
  std::uint64_t preconditioner_actions = 0;
  descriptor.precondition_data = &preconditioner_actions;
  descriptor.precondition = [](void* data, const double* input, double* output,
                               std::size_t) -> generativeqc_status {
    ++*static_cast<std::uint64_t*>(data);
    output[0] = input[0];
    output[1] = 1e300 * input[0];
    output[2] = 0;
    return GENERATIVEQC_STATUS_SUCCESS;
  };
  for (bool with_guess : {false, true}) {
    actions = preconditioner_actions = 0;
    const std::array<double, 3> guess{{0, 3, 0}};
    descriptor.initial_guess = with_guess ? guess.data() : nullptr;
    generativeqc::ResponseResult overflowed_candidate(descriptor);
    const auto failed = overflowed_candidate.data();
    require(!failed.converged && failed.reason == GENERATIVEQC_RESPONSE_BREAKDOWN &&
                failed.relative_residual == 1 && failed.residual_norm == 1 &&
                failed.operator_actions == actions &&
                failed.preconditioner_actions == preconditioner_actions,
            "overflowed candidate changed failure status, residual or actual work");
    for (std::size_t index = 0; index < guess.size(); ++index)
      require(failed.solution[index] == (with_guess ? guess[index] : 0),
              "overflowed candidate did not retain the last finite iterate");
  }
}

}  // namespace

int main() {
  try {
    run();
    small_pivot_contracts();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
