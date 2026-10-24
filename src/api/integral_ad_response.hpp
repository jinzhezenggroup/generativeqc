#pragma once

#include <array>
#include <vector>

#include "response/unrestricted.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/reference/mean_field.hpp"

namespace generativeqc::api::integral_ad {
using Matrix = scf::reference::Matrix;
using SpinMatrices = std::array<Matrix, 2>;

/** Differentiate the physical HF stationary equations for arbitrary integral
 * motion, including overlap transport. Both spin channels have unit occupation;
 * restricted response is the symmetric spin subspace of the shared UHF action.
 * No eigengauge derivative or displaced SCF history enters the calculation. */
inline void response(const scf::PreparedFockPlan& plan, const scf::PreparedFockPlan& direction_plan,
                     const SpinMatrices& density, const SpinMatrices& fock,
                     std::array<std::size_t, 2> occupied, bool restricted,
                     std::size_t workspace_bytes, SpinMatrices& density_direction,
                     SpinMatrices& weighted_direction, double* receipt) {
  using namespace scf::reference;
  const auto& ints = plan.one_electron();
  const auto& motion = direction_plan.one_electron();
  const auto n = ints.nbf, n2 = n * n;
  const auto orthogonalizer = plan.overlap_orthogonalizer();
  std::array<EigenResult, 2> orbitals;
  SpinMatrices metric, delta, explicit_fock;
  const auto induced = [&](const SpinMatrices& d) -> SpinMatrices {
    auto result = scf::assemble_fock(plan.strategy(), Matrix(n2, 0), plan.build(d[0], d[1]));
    receipt[3] += 1;
    return {std::move(result.alpha), std::move(result.beta)};
  };
  auto explicit_result = scf::assemble_fock(direction_plan.strategy(), motion.hcore,
                                            direction_plan.build(density[0], density[1]));
  receipt[3] += 1;
  explicit_fock = {std::move(explicit_result.alpha), std::move(explicit_result.beta)};
  core::ElectronicReferenceView reference{n, 2, ints.overlap, ints.hcore};
  std::size_t dimension = 0;
  for (unsigned spin = 0; spin < 2; ++spin) {
    const auto eigen = plan.eigen_operation(scf::PreparedFockPlan::EigenUse::Finalization);
    orbitals[spin] = eigen ? eigen(fock[spin], &ints.overlap, &orthogonalizer, n)
                           : generalized_eigen(fock[spin], orthogonalizer, n);
    const auto& c = orbitals[spin].vectors;
    // Reconstruct the density in this physical canonical frame before admitting
    // its response; the final iterate and the response must be the same state.
    if (density_rms(density_from_orbitals(c, n, occupied[spin], 1), density[spin]) > 1e-8)
      throw std::runtime_error("integral AD response frame is not stationary");
    metric[spin] = multiply(transpose(c, n), multiply(motion.overlap, c, n), n);
    Matrix transport(n2);
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        transport[p * n + q] = -0.5 * metric[spin][p * n + q] *
                               (double(p < occupied[spin]) + double(q < occupied[spin]));
    delta[spin] = multiply(c, multiply(transport, transpose(c, n), n), n);
    reference.channels[spin] = {occupied[spin], c,          orbitals[spin].values,
                                density[spin],  fock[spin], {}};
    dimension += occupied[spin] * (n - occupied[spin]);
  }
  const auto metric_fock = induced(delta);
  Matrix rhs(dimension), diagonal(dimension);
  std::size_t offset = 0;
  for (unsigned spin = 0; spin < 2; ++spin) {
    Matrix fixed = explicit_fock[spin];
    for (std::size_t k = 0; k < n2; ++k) fixed[k] += metric_fock[spin][k];
    const auto& c = orbitals[spin].vectors;
    const auto mo = multiply(transpose(c, n), multiply(fixed, c, n), n);
    const auto& energies = orbitals[spin].values;
    const auto no = occupied[spin];
    for (std::size_t i = 0; i < no; ++i)
      for (std::size_t a = no; a < n; ++a) {
        rhs[offset] = -mo[a * n + i] + 0.5 * (energies[a] + energies[i]) * metric[spin][a * n + i];
        diagonal[offset++] = energies[a] - energies[i];
      }
  }
  Matrix rotations(dimension, 0);
  if (dimension) {
    auto full = generativeqc::response::spin_orbital_problem(reference, induced, [] {});
    const auto restricted_dimension = occupied[0] * (n - occupied[0]);
    // Collapse the shared spin action to the RHF singlet sector, so an unrelated
    // triplet instability cannot enter the restricted derivative solve.
    auto action = [&](std::span<const double> x, std::span<double> y) {
      if (!restricted) {
        full.apply()(x, y);
        return;
      }
      Matrix joined(dimension), result(dimension);
      std::copy(x.begin(), x.end(), joined.begin());
      std::copy(x.begin(), x.end(), joined.begin() + restricted_dimension);
      full.apply()(joined, result);
      for (std::size_t i = 0; i < restricted_dimension; ++i)
        y[i] = 0.5 * (result[i] + result[restricted_dimension + i]);
    };
    generativeqc::response::LinearResponseProblem problem(
        restricted ? restricted_dimension : dimension, action,
        generativeqc::response::LinearResponseSymmetry::Symmetric);
    generativeqc::response::GmresOptions options;
    options.relative_tolerance = 1e-11;
    options.absolute_tolerance = 1e-12;
    options.max_workspace_bytes = workspace_bytes;
    const auto solve_plan = generativeqc::response::prepare_response(problem, options);
    const auto count = problem.dimension();
    // Gap preconditioning is optional; a vanishing gap is not a license to
    // divide by zero or publish an unverified response.
    bool regular = true;
    for (std::size_t i = 0; i < count; ++i) regular = regular && std::abs(diagonal[i]) > 1e-10;
    const auto result = generativeqc::response::solve_response(
        solve_plan, problem, std::span(rhs).first(count), {},
        regular ? std::span<const double>(diagonal).first(count) : std::span<const double>{});
    if (!result.converged())
      throw std::runtime_error("integral AD orbital response did not converge");
    std::copy(result.solution.begin(), result.solution.end(), rotations.begin());
    if (restricted)
      std::copy(result.solution.begin(), result.solution.end(), rotations.begin() + count);
    receipt[6] = result.residual_norm;
    receipt[7] = result.operator_actions;
  }
  offset = 0;
  for (unsigned spin = 0; spin < 2; ++spin) {
    Matrix mo(n2, 0);
    for (std::size_t i = 0; i < occupied[spin]; ++i)
      for (std::size_t a = occupied[spin]; a < n; ++a)
        mo[a * n + i] = mo[i * n + a] = rotations[offset++];
    const auto& c = orbitals[spin].vectors;
    density_direction[spin] = multiply(c, multiply(mo, transpose(c, n), n), n);
    for (std::size_t k = 0; k < n2; ++k) density_direction[spin][k] += delta[spin][k];
  }
  const auto response_fock = induced(density_direction);
  for (unsigned spin = 0; spin < 2; ++spin) {
    Matrix df = explicit_fock[spin];
    for (std::size_t k = 0; k < n2; ++k) df[k] += response_fock[spin][k];
    const auto left = multiply(density_direction[spin], multiply(fock[spin], density[spin], n), n);
    const auto right = multiply(density[spin], multiply(fock[spin], density_direction[spin], n), n);
    weighted_direction[spin] = multiply(density[spin], multiply(df, density[spin], n), n);
    for (std::size_t k = 0; k < n2; ++k) weighted_direction[spin][k] += left[k] + right[k];
  }
}
}  // namespace generativeqc::api::integral_ad
