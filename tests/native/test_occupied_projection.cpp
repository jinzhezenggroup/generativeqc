#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scf/occupied_projection.hpp"

namespace {
using namespace generativeqc::scf::projection;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, double tolerance = 2e-12) {
  require(std::abs(actual - expected) <= tolerance, "independent occupied projection gate failed");
}

template <class Operation>
void rejected(Operation operation) {
  bool refused = false;
  try {
    operation();
  } catch (const Rejected&) {
    refused = true;
  }
  require(refused, "invalid occupied projection was accepted");
}

void run() {
  const std::array<double, 4> source{{1, 0, 0, 1}};
  const std::array<double, 9> target{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
  const std::array<double, 6> cross{{0.8, 0, 0, 0.6, 0, 0.8}};
  const std::array<double, 9> expected{{2, 0, 0, 0, 0.72, 0.96, 0, 0.96, 1.28}};
  const Policy permissive{1e-10, 1e-7, 1.0, 4096};
  Input input{2, 3, 2, 2, source, target, cross, source};
  auto result = project_occupied(input, permissive);
  require(result.density.size() == expected.size() && result.coefficients.size() == 6,
          "detached occupied projection shapes differ");
  for (std::size_t index = 0; index < expected.size(); ++index)
    close(result.density[index], expected[index]);
  close(result.diagnostics.minimum_projected_norm_squared, 0.64);
  close(result.diagnostics.projection_residual, 0.6);
  close(result.diagnostics.electron_trace, 4);
  close(result.diagnostics.metric_idempotency_error, 0);
  require(result.diagnostics.source_metric_rank == 2 && result.diagnostics.target_metric_rank == 3,
          "metric ranks were not retained");
  const auto detached = result;
  const std::array<double, 4> rotated{{0.6, -0.8, 0.8, 0.6}};
  input.source_state = rotated;
  result = project_occupied(input, permissive);
  for (std::size_t index = 0; index < expected.size(); ++index)
    close(result.density[index], detached.density[index]);
  input.source_state = source;
  rejected([&] { (void)project_occupied(input); });

  const std::array<double, 4> density{{2, 0, 0, 2}};
  input.source_state = density;
  result = project_density(input, permissive);
  for (std::size_t index = 0; index < expected.size(); ++index)
    close(result.density[index], expected[index]);
  const std::array<double, 4> fractional{{2, 0, 0, 1.8}};
  input.source_state = fractional;
  rejected([&] { (void)project_density(input, permissive); });

  input.occupied = 0;
  input.occupation = 1;
  const std::array<double, 4> zero{};
  input.source_state = zero;
  result = project_density(input, permissive);
  require(result.coefficients.empty(), "empty spin channel invented orbitals");
  for (const auto value : result.density) close(value, 0);
  close(result.diagnostics.electron_trace, 0);
  close(result.diagnostics.minimum_projected_norm_squared, 1);
  close(result.diagnostics.projection_residual, 0);
  auto all_discarded = permissive;
  all_discarded.relative_threshold = 1;
  result = project_density(input, all_discarded);
  require(result.diagnostics.source_metric_rank == 0 && result.diagnostics.target_metric_rank == 0,
          "inclusive metric cutoff retained a boundary mode");

  const std::array<double, 4> ill_conditioned{{1, 0, 0, 1e-10}};
  input = {2, 2, 2, 2, ill_conditioned, source, source, source};
  rejected([&] { (void)project_occupied(input, permissive); });
  const std::array<double, 4> asymmetric{{1, 0.01, 0, 1}};
  input = {2, 2, 2, 2, asymmetric, source, source, source};
  rejected([&] { (void)project_occupied(input, permissive); });
  const std::array<double, 4> indefinite{{1, 0, 0, -1}};
  input.source_overlap = indefinite;
  rejected([&] { (void)project_occupied(input, permissive); });
  input = {2, 2, 2, 2, source, source, source, source};
  const std::array<double, 4> inconsistent{{1.1, 0, 0, 1}};
  input.cross_overlap = inconsistent;
  rejected([&] { (void)project_occupied(input, permissive); });
  const std::array<double, 4> nonfinite{{1, 0, 0, std::numeric_limits<double>::quiet_NaN()}};
  input.cross_overlap = nonfinite;
  rejected([&] { (void)project_occupied(input, permissive); });
  input.cross_overlap = std::span<const double>(source).first(3);
  rejected([&] { (void)project_occupied(input, permissive); });

  const std::array<double, 4> nontrivial_metric{{4, 0, 0, 9}};
  const std::array<double, 4> normalized{{0.5, 0, 0, 1.0 / 3.0}};
  input = {2, 2, 2, 2, nontrivial_metric, nontrivial_metric, nontrivial_metric, normalized};
  result = project_occupied(input);
  close(result.density[0], 0.5);
  close(result.density[3], 2.0 / 9.0);
  close(result.diagnostics.electron_trace, 4);
  close(result.diagnostics.normal_equation_residual, 0);

  const std::array<double, 4> half_cross{{0.5, 0, 0, 0.5}};
  input = {2, 2, 2, 2, source, source, half_cross, source};
  auto inclusive = permissive;
  inclusive.relative_threshold = 0.25;
  rejected([&] { (void)project_occupied(input, inclusive); });
  inclusive.relative_threshold = std::nextafter(0.25, 0.0);
  result = project_occupied(input, inclusive);
  close(result.diagnostics.minimum_projected_norm_squared, 0.25);
  inclusive.maximum_residual = std::sqrt(0.75);
  result = project_occupied(input, inclusive);
  close(result.diagnostics.projection_residual, inclusive.maximum_residual);
  inclusive.maximum_residual = std::nextafter(inclusive.maximum_residual, 0.0);
  rejected([&] { (void)project_occupied(input, inclusive); });

  for (const double scale : {1e-200, 1e200}) {
    const std::array<double, 4> scaled_metric{{scale, 0, 0, scale}};
    const double coefficient = 1.0 / std::sqrt(scale);
    const std::array<double, 4> scaled_coefficients{{coefficient, 0, 0, coefficient}};
    input = {2, 2, 2, 2, scaled_metric, scaled_metric, scaled_metric, scaled_coefficients};
    result = project_occupied(input);
    close(result.density[0] * scale, 2);
    close(result.density[3] * scale, 2);
    close(result.diagnostics.electron_trace, 4);
    close(result.diagnostics.minimum_projected_norm_squared, 1);
  }
}
}  // namespace

int main() {
  try {
    run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
