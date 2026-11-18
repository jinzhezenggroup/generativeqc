#pragma once

#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "generated_ecp_ao.cuh"
#include "integrals/ecp.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_usage.hpp"

namespace generativeqc::integrals::ecp_curvature_cpu {

struct Sum {
  double value{}, correction{};
  void add(double term) {
    const auto adjusted = term - correction;
    const auto total = value + adjusted;
    correction = (total - value) - adjusted;
    if (!std::isfinite(term) || !std::isfinite(total) || !std::isfinite(correction))
      throw std::domain_error("nonfinite residual ECP column accumulation");
    value = total;
  }
};

/** Shared transactional reduction of canonical relative integral jets.
 * Both analytic and quadrature providers retain the same ordered-weight and
 * physical-slot contract. Only compensated output fields grow with atoms.
 */
class Column {
 public:
  explicit Column(std::size_t atoms)
      : coordinates(runtime::size_mul(3, atoms)),
        fields(runtime::size_add(2, runtime::size_mul(2, coordinates))),
        values{std::vector<Sum>(fields), std::vector<Sum>(fields)},
        retained(runtime::vector_capacities(values[0], values[1])) {}

  void add(std::size_t part, double weight, double weight_right, const double* jet,
           std::size_t first, std::size_t second, std::size_t center,
           std::span<const double> right) {
    std::array<double, 14> projected{};
    generated::ecp_weighted_column(weight, weight_right, jet, right.data() + 3 * first,
                                   right.data() + 3 * second, right.data() + 3 * center,
                                   projected.data());
    for (const auto value : projected)
      if (!std::isfinite(value))
        throw std::domain_error("nonfinite residual ECP column projection");
    values[part][0].add(projected[0]);
    values[part][1].add(projected[1]);
    const std::array<std::size_t, 2> atoms{first, second};
    for (std::size_t slot = 0; slot < 2; ++slot) {
      if (atoms[slot] == center) continue;
      for (std::size_t axis = 0; axis < 3; ++axis)
        for (std::size_t leg = 0; leg < 2; ++leg) {
          const auto value = projected[2 + 6 * leg + 3 * slot + axis];
          values[part][2 + leg * coordinates + 3 * atoms[slot] + axis].add(value);
          values[part][2 + leg * coordinates + 3 * center + axis].add(-value);
        }
    }
  }

  EcpCurvatureData publish(std::size_t basis, EcpDirectionWork work) const {
    EcpCurvatureData result{
        basis, {std::vector<double>(fields), std::vector<double>(fields)}, work};
    runtime::sample_cpu_capacity(runtime::vector_capacities(result.parts[0], result.parts[1]));
    for (std::size_t part = 0; part < 2; ++part)
      for (std::size_t field = 0; field < fields; ++field)
        result.parts[part][field] = values[part][field].value;
    return result;
  }

 private:
  std::size_t coordinates, fields;
  std::array<std::vector<Sum>, 2> values;
  runtime::CpuRetainedCapacity retained;
};

}  // namespace generativeqc::integrals::ecp_curvature_cpu
