#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "integrals/ecp.hpp"
#include "integrals/ecp_gaussian.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

core::System fixture(bool general, bool spherical) {
  core::System system;
  system.atoms = {{11, {.13, -.21, .17}, 10}, {1, {.43, .19, 3.2}}, {11, {-.63, .24, -1.6}, 10}};
  system.shells = {{0, 1, {{.9, .7}, {.25, .3}}}, {1, 0, {{.6, .7}, {.18, .3}}}};
  system.multiplicity = 2;
  system.basis_representation = spherical ? core::kSphericalBasis : core::kCartesianBasis;
  for (const auto center : {0U, 2U})
    for (int channel = -1; channel <= (general ? 3 : 2); ++channel)
      for (unsigned power = 0; power <= 4; ++power) {
        if (!general && power != 2) continue;
        const auto scale = center ? -.8 : 1;
        system.ecp_terms.push_back({center, channel, power, .67, -.012 * scale});
        system.ecp_terms.push_back({center, channel, power, 1.23, .0081 * scale});
      }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  return system;
}
}  // namespace

int main() {
  try {
    const std::array<double, 9> direction{.16, -.09, .13, -.12, .07, -.11, .03, -.08, .14};
    const std::array<double, 9> translation{.13, -.21, .17, .13, -.21, .17, .13, -.21, .17};
    for (const bool general : {false, true})
      for (const bool spherical : {false, true}) {
        const auto system = fixture(general, spherical);
        const auto plan = integrals::ecp_direction_plan(system);
        require(plan.backend == (general ? integrals::EcpDirectionBackend::CheckedGrid
                                         : integrals::EcpDirectionBackend::Gaussian),
                "wrong ECP source backend");
        const auto result = integrals::checked_ecp_direction(system, direction, plan.limits, [] {});
        double norm{};
        for (const auto* panel : {&result.local, &result.nonlocal})
          for (const auto value : *panel) {
            require(std::isfinite(value), "nonfinite native ECP source");
            norm += value * value;
          }
        require(norm > 1e-10, "empty native ECP source");
        const auto count = molecule::ao_count(system);
        std::vector<double> weights(count * count, .3);
        const auto column_plan = integrals::ecp_curvature_plan(system);
        const auto column = integrals::checked_ecp_curvature(system, weights, {}, direction,
                                                             column_plan.limits, [] {});
        const auto rigid_column = integrals::checked_ecp_curvature(system, weights, {}, translation,
                                                                   column_plan.limits, [] {});
        double curvature_norm{};
        for (std::size_t part = 0; part < 2; ++part) {
          double projection{};
          for (std::size_t coordinate = 0; coordinate < direction.size(); ++coordinate) {
            const auto curvature = column.parts[part][2 + direction.size() + coordinate];
            require(std::isfinite(curvature), "nonfinite ECP column");
            curvature_norm += curvature * curvature;
            projection += direction[coordinate] * column.parts[part][2 + coordinate];
            require(rigid_column.parts[part][2 + direction.size() + coordinate] == 0,
                    "nonzero ECP rigid column");
          }
          require(std::abs(projection - column.parts[part][1]) < 3e-12,
                  "inconsistent ECP column right derivative");
          require(rigid_column.parts[part][1] == 0, "nonzero ECP rigid right derivative");
        }
        require(curvature_norm > 1e-10, "empty native ECP column");
        std::vector<double> moving_weights(weights.size(), -.04), zero(direction.size());
        const auto response_column = integrals::checked_ecp_curvature(
            system, weights, moving_weights, direction, column_plan.limits, [] {});
        const auto density_column = integrals::checked_ecp_curvature(
            system, weights, moving_weights, zero, column_plan.limits, [] {});
        double response_norm{};
        for (std::size_t part = 0; part < 2; ++part) {
          require(std::abs(response_column.parts[part][1] - column.parts[part][1] -
                           density_column.parts[part][1]) < 3e-12,
                  "inconsistent ECP weight-right scalar");
          for (std::size_t coordinate = 0; coordinate < direction.size(); ++coordinate) {
            const auto field = 2 + direction.size() + coordinate;
            require(
                response_column.parts[part][2 + coordinate] == column.parts[part][2 + coordinate],
                "ECP right weights changed the fixed-weight left partial");
            require(std::abs(response_column.parts[part][field] - column.parts[part][field] -
                             density_column.parts[part][field]) < 3e-12,
                    "inconsistent ECP weight-right action");
            response_norm += std::abs(density_column.parts[part][field]);
          }
        }
        require(response_norm > 1e-8, "empty ECP density response");
        if (!general) {
          const auto limits = integrals::gaussian_ecp_weighted_mixed_plan(system);
          const auto mixed = integrals::gaussian_ecp_weighted_mixed(system, weights, direction,
                                                                    direction, limits, [] {});
          const auto rigid = integrals::gaussian_ecp_weighted_mixed(system, weights, translation,
                                                                    direction, limits, [] {});
          require(std::abs(mixed.parts[0][3]) + std::abs(mixed.parts[1][3]) > 1e-10,
                  "empty native ECP mixed source");
          for (const auto& part : rigid.parts)
            require(part[1] == 0 && part[3] == 0, "nonzero ECP mixed translation");
        }
        require(result.work.kernel_blocks <= plan.limits.gaussian_kernel_blocks &&
                    result.work.radial_layers <= plan.limits.grid_radial_layers &&
                    result.work.ao_primitive_points <= plan.limits.grid_ao_primitive_points &&
                    result.work.contraction_terms <= plan.limits.grid_contraction_terms,
                "ECP source exceeded its work envelope");
        const auto translated =
            integrals::checked_ecp_direction(system, translation, plan.limits, [] {});
        for (const auto* panel : {&translated.local, &translated.nonlocal})
          require(
              std::all_of(panel->begin(), panel->end(), [](double value) { return value == 0; }),
              "native ECP translation is nonzero");
      }
    std::cout << "native ECP directions/columns: analytic/checked-grid, Cartesian/spherical\n";
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    return 1;
  }
}
