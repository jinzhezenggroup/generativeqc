#include "posthf/mp2_force.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>

#include "hf/reference.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "posthf/mp2_derivative.hpp"
#include "posthf/mp2_force_workspace.hpp"
#include "posthf/mp2_gradient.hpp"
#include "posthf/native_provider.hpp"
#include "posthf/raw_source.hpp"
#include "posthf/source_reuse_schedule_generated.hpp"
#include "response/solve.hpp"
#include "runtime/df_progress_trace.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::mp2 {
namespace {
std::size_t square(std::size_t value) { return posthf::checked_mul(value, value); }

std::vector<std::size_t> range(std::size_t begin, std::size_t end) {
  std::vector<std::size_t> result(end - begin);
  std::iota(result.begin(), result.end(), begin);
  return result;
}

std::vector<double> hcore_mo(const hf::PhysicalReference& reference) {
  const auto n = reference.nbf;
  const auto n2 = square(n);
  if (reference.hcore.size() != n2 || reference.coefficients.size() != n2)
    throw std::invalid_argument("MP2 force reference has inconsistent one-electron data");
  std::vector<double> result(n2), workspace(n2);
  tensor::cpu_congruence('T', n, reference.coefficients.data(), reference.hcore.data(),
                         result.data(), workspace.data());
  return result;
}

EnergyAdjoint energy_adjoint(const hf::PhysicalReference& reference,
                             const posthf::MOBlockProvider& provider, double denominator_threshold,
                             bool cuda, int device_id) {
  const auto no = reference.nocc, n = reference.nbf, nv = n - no;
  const auto occupied = range(0, no);
  const auto virtuals = range(no, n);
  const auto raw = provider.get({occupied, virtuals, occupied, virtuals}, cuda, device_id);
  std::vector<double> ordered(posthf::checked_mul(square(no), square(nv)));
  for (std::size_t i = 0; i < no; ++i)
    for (std::size_t a = 0; a < nv; ++a)
      for (std::size_t j = 0; j < no; ++j)
        for (std::size_t b = 0; b < nv; ++b)
          ordered[((i * no + j) * nv + a) * nv + b] = raw[((i * nv + a) * no + j) * nv + b];
  return canonical_energy_adjoint(ordered, reference.orbital_energies, no, denominator_threshold);
}

response::LinearResponseProblem response_problem(const hf::PhysicalReference& reference,
                                                 const posthf::MOBlockProvider& provider, bool cuda,
                                                 int device_id) {
  const auto no = reference.nocc, n = reference.nbf, nv = n - no;
  const auto occupied = range(0, no);
  const auto virtuals = range(no, n);
  return {posthf::checked_mul(no, nv),
          [&reference, &provider, no, nv, occupied, virtuals, cuda, device_id](
              std::span<const double> input, std::span<double> output) {
            if (input.size() != no * nv || output.size() != input.size())
              throw std::invalid_argument("RHF response vector has the wrong shape");
            for (std::size_t i = 0; i < no; ++i)
              for (std::size_t a = 0; a < nv; ++a) {
                const std::vector<std::size_t> ai{no + a};
                const std::vector<std::size_t> oi{i};
                const auto ovov = provider.get({ai, oi, virtuals, occupied}, cuda, device_id);
                const auto vvoo = provider.get({ai, virtuals, oi, occupied}, cuda, device_id);
                const auto voov = provider.get({ai, occupied, oi, virtuals}, cuda, device_id);
                double value =
                    (reference.orbital_energies[no + a] - reference.orbital_energies[i]) *
                    input[i * nv + a];
                for (std::size_t j = 0; j < no; ++j)
                  for (std::size_t b = 0; b < nv; ++b)
                    value += (4.0 * ovov[b * no + j] - vvoo[b * no + j] - voov[j * nv + b]) *
                             input[j * nv + b];
                output[i * nv + a] = value;
              }
          },
          response::LinearResponseSymmetry::Symmetric};
}

ConventionalForceResult conventional_force_impl(
    const hf::PhysicalReference& reference, const integrals::ElectronInteractionSource& source,
    std::size_t budget_bytes, double denominator_threshold, double same_space_threshold,
    const response::GmresOptions& response_options, bool cuda, int device_id) {
#if !GENERATIVEQC_HAS_CUDA
  if (cuda) throw std::runtime_error("CUDA conventional force is unavailable in this build");
#endif
  if (!reference.nocc || reference.nocc >= reference.nbf || source.nbf() != reference.nbf ||
      !budget_bytes || !std::isfinite(denominator_threshold) || denominator_threshold <= 0.0 ||
      !std::isfinite(same_space_threshold) || same_space_threshold <= 0.0 || device_id < 0)
    throw std::invalid_argument("invalid conventional MP2 force request");
  auto provider = std::make_unique<posthf::NativeBlockProvider>(source, reference, budget_bytes);
  auto problem = response_problem(reference, *provider, cuda, device_id);
  const auto dimension = problem.dimension();
  auto plan = response::prepare_response(problem, response_options);
  std::size_t maximum_shell = 0;
  for (const auto& shell : source.orbital().shells) {
    const auto count = source.orbital().basis_representation == GENERATIVEQC_BASIS_SPHERICAL
                           ? 2 * shell.angular_momentum + 1
                           : (shell.angular_momentum + 1) * (shell.angular_momentum + 2) / 2;
    maximum_shell = std::max(maximum_shell, static_cast<std::size_t>(count));
  }
  const auto coordinate_count = posthf::checked_mul(source.orbital().atoms.size(), 3);
  const auto workspace_for = [&](const posthf::NativeBlockProvider& candidate) {
    return detail::streamed_force_workspace_bytes(
        reference.nbf, reference.nocc, [&](const std::array<std::size_t, 4>& shape) {
          const auto block = candidate.plan(shape, cuda);
          const auto peak = posthf::checked_add(block.host_bytes, block.device_bytes);
          if (peak < candidate.provider_bytes())
            throw std::logic_error("MP2 provider workspace accounting underflow");
          return peak - candidate.provider_bytes();
        });
  };
  auto streamed_workspace = workspace_for(*provider);
  const auto legacy_budget = detail::remaining_force_budget(budget_bytes, streamed_workspace);
  const auto legacy_resources = conventional_gradient_plan(
      reference.nbf, reference.nocc, provider->provider_bytes(), plan, maximum_shell,
      coordinate_count, posthf::checked_mul(coordinate_count, sizeof(double)), legacy_budget);
  const auto* tile_override = std::getenv("GENERATIVEQC_MP2_FORCE_SOURCE_TILING");
  if (cuda && source.supports_device_read(integrals::ElectronInteractionOperator::eri, device_id) &&
      (!tile_override || std::string_view(tile_override) != "0")) {
    // Reuse the compiler's cross-shell source planner from CC/MP2 energy.
    // Every candidate covers all force consumers, not just OVOV. Preserve the
    // legacy derivative allowance up to its actual bounded scratch ceiling.
    const auto derivative_reserve =
        std::min(legacy_budget - legacy_resources.peak_bytes, posthf::source_scratch_bytes);
    std::vector<posthf::generated::SourceTileCandidate> candidates;
    for (std::size_t axis = 1; axis <= reference.nbf; ++axis) {
      try {
        posthf::NativeBlockProvider candidate(source, reference, budget_bytes,
                                              static_cast<unsigned>(axis),
                                              posthf::AOTileDomain::Basis);
        const auto workspace = workspace_for(candidate);
        const auto remaining = detail::remaining_force_budget(budget_bytes, workspace);
        const auto resources = conventional_gradient_plan(
            reference.nbf, reference.nocc, candidate.provider_bytes(), plan, maximum_shell,
            coordinate_count, posthf::checked_mul(coordinate_count, sizeof(double)), remaining,
            derivative_reserve);
        candidates.push_back({axis, 1, posthf::checked_add(resources.peak_bytes, workspace)});
      } catch (const std::length_error&) {
        continue;
      }
    }
    if (!candidates.empty()) {
      const auto selected = posthf::generated::select_source_tile(reference.nbf, candidates);
      provider = std::make_unique<posthf::NativeBlockProvider>(
          source, reference, budget_bytes, static_cast<unsigned>(selected.axis_tile),
          posthf::AOTileDomain::Basis);
      streamed_workspace = workspace_for(*provider);
      problem = response_problem(reference, *provider, cuda, device_id);
      plan = response::prepare_response(problem, response_options);
    }
  }
  runtime::df_progress::Scope::number("mp2_force_source_axis_tile", provider->tile_shape()[0]);
  const auto provider_bytes = provider->provider_bytes();
  const auto endpoint_budget = detail::remaining_force_budget(budget_bytes, streamed_workspace);
  const auto base_resources = conventional_gradient_plan(
      reference.nbf, reference.nocc, provider_bytes, plan, maximum_shell, coordinate_count,
      posthf::checked_mul(coordinate_count, sizeof(double)), endpoint_budget);
  const auto backend_stage_bytes = cuda ? endpoint_budget - base_resources.peak_bytes : 0;
  if (cuda && !backend_stage_bytes)
    throw std::length_error("conventional MP2 CUDA derivative has no staging budget");
  const auto resources = conventional_gradient_plan(
      reference.nbf, reference.nocc, provider_bytes, plan, maximum_shell, coordinate_count,
      posthf::checked_mul(coordinate_count, sizeof(double)), endpoint_budget, backend_stage_bytes);
  const auto h = hcore_mo(reference);
  auto adjoint = energy_adjoint(reference, *provider, denominator_threshold, cuda, device_id);
  auto orbital = canonical_orbital_rhs_streamed(reference, h, *provider, adjoint,
                                                same_space_threshold, cuda, device_id);
  std::vector<double> diagonal(dimension);
  for (std::size_t i = 0; i < reference.nocc; ++i)
    for (std::size_t a = 0; a < reference.nbf - reference.nocc; ++a)
      diagonal[i * (reference.nbf - reference.nocc) + a] =
          reference.orbital_energies[reference.nocc + a] - reference.orbital_energies[i];
  auto response_result =
      response::solve_response(plan, problem, orbital.response_rhs, {}, diagonal);
  if (!response_result.converged())
    throw std::runtime_error("canonical MP2 orbital response did not converge");
  auto weights = canonical_lagrangian_weights_streamed(reference, h, *provider, std::move(adjoint),
                                                       std::move(orbital), response_result.solution,
                                                       same_space_threshold, cuda, device_id);
  if (!std::isfinite(weights.stationarity_residual) || weights.stationarity_residual > 1e-7)
    throw std::runtime_error("canonical MP2 relaxed Lagrangian is not stationary");
  auto derivative = cuda ? conventional_derivative_cuda(source.orbital(), reference, weights,
                                                        device_id, backend_stage_bytes)
                         : conventional_derivative_cpu(source.orbital(), reference, weights);
  for (double& value : derivative) value = -value;
  ConventionalForceResult result;
  result.forces = std::move(derivative);
  result.response = std::move(response_result);
  result.stationarity_residual = weights.stationarity_residual;
  const auto shells = source.orbital().shells.size();
  result.weighted_eri_shell_tiles = square(square(shells));
  result.derivative_workspace_bytes = posthf::checked_add(
      resources.derivative_staging_bytes, resources.derivative_backend_staging_bytes);
  result.planned_endpoint_peak_bytes =
      posthf::checked_add(resources.peak_bytes, streamed_workspace);
  // No allocator-level endpoint telemetry is available in this slice. Zero
  // means unmeasured, not zero allocation; never copy the plan into a measurement.
  result.measured_endpoint_peak_bytes = 0;
  return result;
}
}  // namespace

ConventionalForceResult conventional_force_cpu(const hf::PhysicalReference& reference,
                                               const integrals::ElectronInteractionSource& source,
                                               std::size_t budget_bytes,
                                               double denominator_threshold,
                                               double same_space_threshold,
                                               const response::GmresOptions& response_options) {
  return conventional_force_impl(reference, source, budget_bytes, denominator_threshold,
                                 same_space_threshold, response_options, false, 0);
}

ConventionalForceResult conventional_force_cuda(
    const hf::PhysicalReference& reference, const integrals::ElectronInteractionSource& source,
    std::size_t budget_bytes, double denominator_threshold, double same_space_threshold,
    const response::GmresOptions& response_options, int device_id) {
  return conventional_force_impl(reference, source, budget_bytes, denominator_threshold,
                                 same_space_threshold, response_options, true, device_id);
}

ConventionalForceResult density_fitted_force_cpu(
    const hf::PhysicalReference& reference, const posthf::RawSource& source,
    std::size_t budget_bytes, double denominator_threshold, double metric_relative_threshold,
    double same_space_threshold, const response::GmresOptions& response_options) {
  if (!reference.nocc || reference.nocc >= reference.nbf || source.nbf() != reference.nbf ||
      !source.naux() || !budget_bytes || !std::isfinite(denominator_threshold) ||
      denominator_threshold <= 0.0 || !std::isfinite(metric_relative_threshold) ||
      metric_relative_threshold <= 0.0 || metric_relative_threshold >= 1.0 ||
      !std::isfinite(same_space_threshold) || same_space_threshold <= 0.0)
    throw std::invalid_argument("invalid RI-MP2 force request");

  posthf::DensityFittedBlockProvider provider(source, reference, budget_bytes,
                                              metric_relative_threshold);
  const auto problem = response_problem(reference, provider, false, 0);
  const auto response_plan = response::prepare_response(problem, response_options);
  const auto coordinate_count = posthf::checked_mul(source.orbital().atoms.size(), std::size_t{3});
  const auto candidate_output_bytes = posthf::checked_mul(coordinate_count, sizeof(double));
  // The RI provider reconstructs each block directly into its output vector;
  // it owns no per-call transform scratch beyond that returned allocation.
  const auto streamed_workspace = detail::streamed_force_workspace_bytes(
      reference.nbf, reference.nocc, [](const std::array<std::size_t, 4>& shape) {
        std::size_t bytes = sizeof(double);
        for (const auto extent : shape) bytes = posthf::checked_mul(bytes, extent);
        return bytes;
      });
  const auto endpoint_budget = detail::remaining_force_budget(budget_bytes, streamed_workspace);
  const auto resources = density_fitted_gradient_plan(
      reference.nbf, reference.nocc, provider.auxiliary_count(), provider.provider_bytes(),
      response_plan, molecule::cartesian_ao_count(source.orbital()),
      molecule::cartesian_ao_count(source.auxiliary()), coordinate_count, candidate_output_bytes,
      endpoint_budget);

  const auto h = hcore_mo(reference);
  auto adjoint = energy_adjoint(reference, provider, denominator_threshold, false, 0);
  auto orbital = canonical_orbital_rhs_streamed(reference, h, provider, adjoint,
                                                same_space_threshold, false, 0);
  const auto virtuals = reference.nbf - reference.nocc;
  std::vector<double> diagonal(posthf::checked_mul(reference.nocc, virtuals));
  for (std::size_t i = 0; i < reference.nocc; ++i)
    for (std::size_t a = 0; a < virtuals; ++a)
      diagonal[i * virtuals + a] =
          reference.orbital_energies[reference.nocc + a] - reference.orbital_energies[i];
  auto response_result =
      response::solve_response(response_plan, problem, orbital.response_rhs, {}, diagonal);
  if (!response_result.converged())
    throw std::runtime_error("RI-MP2 orbital response did not converge");

  auto weights = canonical_lagrangian_weights_streamed(reference, h, provider, std::move(adjoint),
                                                       std::move(orbital), response_result.solution,
                                                       same_space_threshold, false, 0);
  if (!std::isfinite(weights.stationarity_residual) || weights.stationarity_residual > 1e-7)
    throw std::runtime_error("RI-MP2 relaxed Lagrangian is not stationary");
  auto fitted = density_fitted_lagrangian_weights(reference, provider, weights, endpoint_budget);
  if (fitted.planned_peak_bytes > resources.peak_bytes)
    throw std::runtime_error("RI-MP2 reverse exceeded its composed resource plan");

  auto derivative = density_fitted_derivative_cpu(source.orbital(), source.auxiliary(), fitted,
                                                  resources.derivative_staging_bytes);
  for (double& value : derivative) value = -value;

  ConventionalForceResult result;
  result.forces = std::move(derivative);
  result.response = std::move(response_result);
  result.stationarity_residual = weights.stationarity_residual;
  result.derivative_workspace_bytes =
      posthf::checked_add(fitted.workspace_bytes, resources.derivative_staging_bytes);
  result.planned_endpoint_peak_bytes =
      posthf::checked_add(resources.peak_bytes, streamed_workspace);
  result.measured_endpoint_peak_bytes = 0;
  return result;
}

}  // namespace generativeqc::mp2
