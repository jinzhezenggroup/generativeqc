// Tooling bridge to the actual native Lambda GMRES controller. The supplied
// operator and inverse are synchronous callbacks; no borrowed pointer escapes.
// This is CPU control qualification, not a CUDA scientific-action endpoint.
#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>

#include "response/native_gmres.hpp"

using Action = int (*)(const double*, double*);

extern "C" int multisecant_gmres(std::size_t dimension, Action transpose, Action inverse,
                                 const double* rhs, double* solution, std::size_t* counts,
                                 double* residual, char* error, std::size_t error_size) noexcept {
  try {
    if (!transpose || !inverse || !rhs || !solution || !counts || !residual)
      throw std::invalid_argument("missing multisecant GMRES bridge buffer/callback");
    namespace response = generativeqc::response;
    response::GmresOptions options;
    options.absolute_tolerance = 1e-12;
    options.relative_tolerance = 0.0;
    options.max_iterations = 150;
    options.true_residual_every = 30;
    const auto bind = [](Action callback) -> response::LinearOperator {
      return [callback](std::span<const double> input, std::span<double> output) {
        if (callback(input.data(), output.data()))
          throw std::runtime_error("multisecant screening callback failed");
      };
    };
    const auto result =
        response::solve_gmres(response::prepare_gmres(dimension, options), bind(transpose),
                              {rhs, dimension}, {}, {}, bind(inverse));
    if (!result.converged()) throw std::runtime_error("native screening GMRES did not converge");
    std::copy(result.solution.begin(), result.solution.end(), solution);
    counts[0] = result.iterations;
    counts[1] = result.operator_actions;
    counts[2] = result.preconditioner_actions;
    *residual = result.residual_norm;
    return 0;
  } catch (const std::exception& exception) {
    if (error && error_size) std::snprintf(error, error_size, "%s", exception.what());
    return 1;
  }
}
