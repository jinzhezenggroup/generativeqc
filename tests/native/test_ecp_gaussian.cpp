#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "core/work_budget.hpp"
#include "ecp_reference.hpp"
#include "integrals/ecp_gaussian.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_usage.hpp"

namespace {
namespace integrals = generativeqc::integrals;
using generativeqc::core::System;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const char* message) {
  require(
      std::isfinite(actual) && std::isfinite(expected) && std::abs(actual - expected) <= tolerance,
      message);
}

System fixture(generativeqc::core::BasisRepresentation representation, bool high_angular) {
  System system;
  system.atoms = {
      {11, {0.13, -0.21, 0.17}, 10}, {1, {0.43, 0.19, 3.2}, 0}, {11, {-0.63, 0.24, -1.6}, 10}};
  system.shells = {{0, 0, {{0.73, 0.8}, {0.21, -0.12}}}, {1, 1, {{0.45, 0.9}, {1.15, -0.12}}}};
  if (high_angular) {
    system.shells.push_back({0, 2, {{0.35, 1.0}}});
    system.shells.push_back({1, 3, {{0.55, 0.81}, {0.17, 0.23}}});
  }
  system.basis_representation = representation;
  system.multiplicity = 2;
  for (unsigned center : {0U, 2U})
    for (int channel = -1; channel <= 2; ++channel) {
      system.ecp_terms.push_back({center, channel, 2, 0.67, -0.12});
      system.ecp_terms.push_back({center, channel, 2, 1.23, 0.081});
    }
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      detail.c_str());
  return system;
}

std::vector<double> weights_for(std::size_t count) {
  std::vector<double> weights(count * count);
  for (std::size_t index = 0; index < weights.size(); ++index)
    weights[index] = std::sin(0.7 + 1.3 * index);
  return weights;
}

std::vector<double> contract_gradient(const integrals::EcpData& data,
                                      const std::vector<double>& weights, unsigned part) {
  std::vector<double> result(data.ncoord);
  const auto& derivatives = part == 0 ? data.local_derivative : data.nonlocal_derivative;
  for (std::size_t coordinate = 0; coordinate < data.ncoord; ++coordinate)
    for (std::size_t index = 0; index < weights.size(); ++index)
      result[coordinate] += weights[index] * derivatives[coordinate * weights.size() + index];
  return result;
}

void compare(const integrals::EcpData& actual, const integrals::EcpData& expected) {
  require(actual.nbf == expected.nbf && actual.ncoord == expected.ncoord,
          "Gaussian ECP output shape");
  const std::vector<double>* left[] = {&actual.local, &actual.nonlocal, &actual.local_derivative,
                                       &actual.nonlocal_derivative};
  const std::vector<double>* right[] = {&expected.local, &expected.nonlocal,
                                        &expected.local_derivative, &expected.nonlocal_derivative};
  for (unsigned part = 0; part < 4; ++part) {
    require(left[part]->size() == right[part]->size(), "Gaussian ECP output panel size");
    for (std::size_t index = 0; index < left[part]->size(); ++index)
      near((*left[part])[index], (*right[part])[index], part < 2 ? 2e-10 : 2e-8,
           "Gaussian ECP differs from independent real-space projector quadrature");
  }
}

void check_raw(generativeqc::core::BasisRepresentation representation) {
  const auto system = fixture(representation, true);
  integrals::EcpGaussianWork work;
  generativeqc::runtime::cpu_resource_observation = {};
  generativeqc::runtime::cpu_resource_observation.active = true;
  const auto actual = integrals::gaussian_ecp_integrals(system, true, &work);
  const auto measured = generativeqc::runtime::cpu_resource_observation;
  generativeqc::runtime::cpu_resource_observation.active = false;
  require(measured.samples && measured.peak_bytes <= work.workspace_bound_bytes,
          "Gaussian ECP workspace must bound observed live numeric capacity");
  require(
      work.kernel_blocks && work.kernel_blocks <= work.kernel_block_bound && work.radial_frontiers,
      "Gaussian ECP must report bounded actual semantic work");
  compare(actual, generativeqc::testing::ecp_integrals(system, 160, 32, true));
  compare(actual, integrals::checked_ecp_integrals(system, true));
  const auto primal = integrals::gaussian_ecp_integrals(system, false);
  for (std::size_t index = 0; index < primal.local.size(); ++index) {
    near(primal.local[index], actual.local[index], 2e-12,
         "Gaussian primal/derivative local operator mismatch");
    near(primal.nonlocal[index], actual.nonlocal[index], 2e-12,
         "Gaussian primal/derivative projector mismatch");
  }
  auto moved = system;
  moved.atoms[1].position[2] += 0.17;
  compare(integrals::gaussian_ecp_integrals(moved, true),
          generativeqc::testing::ecp_integrals(moved, 160, 32, true));
  compare(actual, integrals::gaussian_ecp_integrals(system, true));
}

void check_curvature(unsigned part) {
  const auto system = fixture(generativeqc::core::kSphericalBasis, false);
  const auto count = generativeqc::molecule::ao_count(system);
  const auto weights = weights_for(count);
  const auto complete = integrals::gaussian_ecp_weighted_second_order(system, weights);
  const auto& result = complete.parts[part];
  const auto reference = generativeqc::testing::ecp_integrals(system, 160, 32, true);
  const auto gradient = contract_gradient(reference, weights, part);
  double value = 0;
  for (std::size_t index = 0; index < weights.size(); ++index)
    value += weights[index] * (part == 0 ? reference.local[index] : reference.nonlocal[index]);
  near(result.value, value, 2e-10, "Gaussian weighted ECP value");
  const auto coordinates = gradient.size();
  require(
      result.gradient.size() == coordinates && result.hessian.size() == coordinates * coordinates,
      "only the contracted coordinate Hessian is published");
  for (std::size_t row = 0; row < coordinates; ++row) {
    near(result.gradient[row], gradient[row], 2e-8, "Gaussian weighted ECP gradient");
    for (std::size_t column = 0; column < coordinates; ++column)
      near(result.hessian[row * coordinates + column], result.hessian[column * coordinates + row],
           2e-8, "Gaussian contracted Hessian symmetry");
    for (unsigned axis = 0; axis < 3; ++axis) {
      double translation = 0;
      for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
        translation += result.hessian[row * coordinates + 3 * atom + axis];
      near(translation, 0, 2e-8, "Gaussian contracted Hessian translation nullspace");
    }
  }
  // Differentiate the independent real-space oracle, not the generated provider.
  // Every coordinate includes the basis-free ECP center and coincident scatter.
  for (double step : {0.003, 0.001})
    for (std::size_t column = 0; column < coordinates; ++column) {
      std::array<std::vector<double>, 4> displaced;
      const std::array<int, 4> offsets{-2, -1, 1, 2};
      for (unsigned sample = 0; sample < 4; ++sample) {
        auto shifted = system;
        shifted.atoms[column / 3].position[column % 3] += offsets[sample] * step;
        displaced[sample] = contract_gradient(
            generativeqc::testing::ecp_integrals(shifted, 160, 32, true), weights, part);
      }
      for (std::size_t row = 0; row < coordinates; ++row) {
        const double finite_difference = (displaced[0][row] - 8 * displaced[1][row] +
                                          8 * displaced[2][row] - displaced[3][row]) /
                                         (12 * step);
        near(result.hessian[row * coordinates + column], finite_difference, 2e-8,
             "Gaussian complete contracted Hessian differs from independent multistep oracle");
      }
    }
}

void check_boundaries() {
  const auto system = fixture(generativeqc::core::kCartesianBasis, false);
  const auto weights = weights_for(generativeqc::molecule::ao_count(system));
  for (const auto limits : {integrals::EcpGaussianLimits{1, 1000000},
                            integrals::EcpGaussianLimits{512ULL * 1024 * 1024, 1}}) {
    bool rejected = false;
    try {
      (void)integrals::gaussian_ecp_weighted_second_order(system, weights, limits);
    } catch (const std::length_error&) {
      rejected = true;
    } catch (const generativeqc::core::WorkLimitError&) {
      rejected = true;
    }
    require(rejected, "Gaussian ECP budget must reject before publishing any result");
  }
  auto invalid = weights;
  invalid[0] = std::numeric_limits<double>::quiet_NaN();
  bool rejected = false;
  try {
    (void)integrals::gaussian_ecp_weighted_second_order(system, invalid);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "Gaussian ECP rejects nonfinite weights");
  for (unsigned boundary = 0; boundary < 2; ++boundary) {
    auto other = system;
    if (boundary == 0)
      other.ecp_terms[0].power = 1;
    else
      other.ecp_terms[0].channel = 3;
    require(!integrals::gaussian_ecp_supported(other), "unsupported radial/projector domain");
    compare(integrals::checked_ecp_integrals(other, true),
            generativeqc::testing::checked_ecp_integrals(other, true));
  }
  compare(integrals::gaussian_ecp_integrals(system, true),
          generativeqc::testing::ecp_integrals(system, 160, 32, true));
}
}  // namespace

int main() {
  try {
    check_raw(generativeqc::core::kCartesianBasis);
    check_raw(generativeqc::core::kSphericalBasis);
    check_curvature(0);
    check_curvature(1);
    check_boundaries();
    std::cout << "Gaussian ECP primal, complete contracted curvature and bounded fallback pass\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
