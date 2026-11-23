#pragma once

#include <array>
#include <cmath>
#include <limits>

#include "dft/host_hessian_aot.hpp"

// Some generated jet variants leave optional DAG intermediates and parameters
// unused. Keep strict warnings on the handwritten point code and this adapter.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "dft/generated_point_ad.hpp"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include "dft/xc_point_response.hpp"

namespace generativeqc::dft::host_hessian_point {

using host_hessian_aot::PointDomain;

template <std::size_t Width = 4>
inline std::array<double, Width> lane(const double* panel, std::size_t count, std::size_t index) {
  std::array<double, Width> result{};
  for (std::size_t component = 0; component < Width; ++component)
    result[component] = panel[component * count + index];
  return result;
}

template <PointDomain Domain>
inline constexpr std::size_t width = 4 * host_hessian_aot::physical_spins(Domain);

template <PointDomain Domain>
inline constexpr bool pbe = Domain != PointDomain::lda_rks && Domain != PointDomain::lda_uks;

/** RKS maps carry total fields, whereas UKS maps carry separate physical
 * alpha and beta blocks without scaling. The shared point code owns the
 * tail domain and never materializes uncontracted rho/sigma Hessians. */
template <PointDomain Domain>
inline point::Value value(const std::array<double, width<Domain>>& field) {
  if constexpr (width<Domain> == 8) {
    const double density[]{field[0], field[4]};
    const double gradients[2][3]{{field[1], field[2], field[3]}, {field[5], field[6], field[7]}};
    return point::evaluate(pbe<Domain>, density, gradients,
                           Domain == PointDomain::scaled_pbe0_uks ? 0.75 : 1.0, 1.0);
  }
  const double gradient[]{field[1] / 2, field[2] / 2, field[3] / 2};
  if constexpr (Domain == PointDomain::scaled_pbe0_rks) {
    return point::evaluate_pbe0_restricted_bound(field[0] / 2, gradient);
  } else {
    const double density[]{field[0] / 2, field[0] / 2};
    const double spin_gradient[2][3]{{gradient[0], gradient[1], gradient[2]},
                                     {gradient[0], gradient[1], gradient[2]}};
    return point::evaluate_with_channels<4, true>(pbe<Domain>, density, spin_gradient, 1.0, 1.0);
  }
}

template <PointDomain Domain>
inline point::Value response(const std::array<double, width<Domain>>& field,
                             const std::array<double, width<Domain>>& direction) {
  if constexpr (width<Domain> == 8) {
    const double density[]{field[0], field[4]}, delta_density[]{direction[0], direction[4]};
    const double gradients[2][3]{{field[1], field[2], field[3]}, {field[5], field[6], field[7]}};
    const double delta_gradients[2][3]{{direction[1], direction[2], direction[3]},
                                       {direction[5], direction[6], direction[7]}};
    return point::unrestricted_response(pbe<Domain>, density, gradients, delta_density,
                                        delta_gradients,
                                        Domain == PointDomain::scaled_pbe0_uks ? 0.75 : 1.0, 1.0);
  }
  return point::restricted_response(pbe<Domain>, field[0], field.data() + 1, direction[0],
                                    direction.data() + 1,
                                    Domain == PointDomain::scaled_pbe0_rks ? 0.75 : 1.0, 1.0);
}

inline point::Value response(PointDomain domain, const std::array<double, 4>& field,
                             const std::array<double, 4>& direction) {
  switch (domain) {
    case PointDomain::lda_rks:
      return response<PointDomain::lda_rks>(field, direction);
    case PointDomain::pbe_rks:
      return response<PointDomain::pbe_rks>(field, direction);
    case PointDomain::scaled_pbe0_rks:
      return response<PointDomain::scaled_pbe0_rks>(field, direction);
    case PointDomain::linked_ad:
    case PointDomain::scaled_pbe0_uks:
    case PointDomain::pbe_uks:
    case PointDomain::lda_uks:
      break;
  }
  point::Value invalid{};
  invalid.valid = false;
  return invalid;
}

template <std::size_t Width = 4>
inline std::array<double, Width> potential(const point::Value& value) {
  std::array<double, Width> coefficients{};
  for (std::size_t spin = 0; spin < Width / 4; ++spin) {
    coefficients[4 * spin] = value.rho[spin];
    for (std::size_t axis = 0; axis < 3; ++axis)
      coefficients[4 * spin + 1 + axis] = value.gradient[spin][axis];
  }
  return coefficients;
}

template <std::size_t Width>
inline double dot(const std::array<double, Width>& left, const std::array<double, Width>& right) {
  double result = left[0] * right[0];
  for (std::size_t component = 1; component < Width; ++component)
    result += left[component] * right[component];
  return result;
}

/** Match the tile ABI of the linked AD leaves. None of these adapters owns
 * output publication; a failing lane invalidates the entire staged tile. */
template <std::size_t Inputs, std::size_t Outputs, class Action>
int evaluate(const double* input, std::size_t input_count, double* output, std::size_t output_count,
             std::size_t count, Action action) noexcept {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(double) / Inputs ||
      input_count != Inputs * count || output_count != Outputs * count ||
      (count && (!input || !output)))
    return -1;
  for (std::size_t index = 0; index < count; ++index) {
    std::array<double, Outputs> values{};
    if (!action(input, count, index, values)) return 1;
    for (std::size_t component = 0; component < Outputs; ++component) {
      if (!std::isfinite(values[component])) return static_cast<int>(component + 1);
      output[component * count + index] = values[component];
    }
  }
  return 0;
}

template <PointDomain Domain>
inline int field_response(const double* input, std::size_t input_count, double* output,
                          std::size_t output_count, std::size_t count) noexcept {
  constexpr auto fields = width<Domain>;
  return evaluate<2 * fields + 2, 2 * fields>(
      input, input_count, output, output_count, count,
      [](const double* panel, auto lanes, auto index, auto& values) {
        const auto field = lane<fields>(panel, lanes, index);
        const auto base = value<Domain>(field);
        const auto tangent =
            response<Domain>(field, lane<fields>(panel + fields * lanes, lanes, index));
        if (!base.valid || !tangent.valid) return false;
        const auto first = potential<fields>(base), delta = potential<fields>(tangent);
        const auto weight = panel[2 * fields * lanes + index];
        const auto moving_weight = panel[(2 * fields + 1) * lanes + index];
        for (std::size_t component = 0; component < fields; ++component) {
          values[component] = weight * first[component];
          values[fields + component] = weight * delta[component] + moving_weight * first[component];
        }
        return true;
      });
}

/** Apply the quadrature product rule to the shared physical partials. The
 * right field direction is physical even when the column pullback uses a
 * unit seed on the left to recover an adjoint. */
template <PointDomain Domain>
inline int energy_mixed(const double* input, std::size_t input_count, double* output,
                        std::size_t output_count, std::size_t count) noexcept {
  constexpr auto fields = width<Domain>;
  return evaluate<4 * fields + 4, 4>(
      input, input_count, output, output_count, count,
      [](const double* panel, auto lanes, auto index, auto& values) {
        const auto field = lane<fields>(panel, lanes, index);
        const auto left = lane<fields>(panel + fields * lanes, lanes, index);
        const auto right = lane<fields>(panel + 2 * fields * lanes, lanes, index);
        const auto mixed = lane<fields>(panel + 3 * fields * lanes, lanes, index);
        const auto base = value<Domain>(field), tangent = response<Domain>(field, right);
        if (!base.valid || !tangent.valid) return false;
        const auto first = potential<fields>(base);
        const auto weight = lane(panel + 4 * fields * lanes, lanes, index);
        const auto energy_left = dot(first, left), energy_right = dot(first, right);
        const auto energy_mixed = dot(first, mixed) + dot(potential<fields>(tangent), left);
        values = {weight[0] * base.energy, weight[0] * energy_left + weight[1] * base.energy,
                  weight[0] * energy_right + weight[2] * base.energy,
                  weight[0] * energy_mixed + weight[1] * energy_right + weight[2] * energy_left +
                      weight[3] * base.energy};
        return true;
      });
}

template <PointDomain Domain>
inline const host_hessian_aot::Program* physical_field_program() {
  static constexpr std::array<std::string_view, 10> restricted{
      "f0", "f1", "f2", "f3", "d0", "d1", "d2", "d3", "weight", "dweight"};
  static constexpr std::array<std::string_view, 18> unrestricted{
      "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",     "d0",
      "d1", "d2", "d3", "d4", "d5", "d6", "d7", "weight", "dweight"};
  static constexpr bool polarized = width<Domain> == 8;
  static const host_hessian_aot::Program program{
      polarized ? "gga_field_response_polarized" : "gga_field_response_unpolarized", point::kDomain,
      polarized ? std::span<const std::string_view>(unrestricted)
                : std::span<const std::string_view>(restricted),
      2 * width<Domain>, field_response<Domain>};
  return &program;
}

inline const host_hessian_aot::Program* field_program(
    const host_hessian_aot::MethodBinding& method) {
  if (method.point_domain != PointDomain::linked_ad) {
    if (method.spins != host_hessian_aot::physical_spins(method.point_domain)) return nullptr;
    switch (method.point_domain) {
      case PointDomain::lda_rks:
        return physical_field_program<PointDomain::lda_rks>();
      case PointDomain::pbe_rks:
        return physical_field_program<PointDomain::pbe_rks>();
      case PointDomain::scaled_pbe0_rks:
        return physical_field_program<PointDomain::scaled_pbe0_rks>();
      case PointDomain::scaled_pbe0_uks:
        return physical_field_program<PointDomain::scaled_pbe0_uks>();
      case PointDomain::pbe_uks:
        return physical_field_program<PointDomain::pbe_uks>();
      case PointDomain::lda_uks:
        return physical_field_program<PointDomain::lda_uks>();
      case PointDomain::linked_ad:
        break;
    }
  }
  return host_hessian_aot::find(
      method.spins == 1 ? "gga_field_response_unpolarized" : "gga_field_response_polarized",
      method.field_response_identity);
}

template <PointDomain Domain>
inline const host_hessian_aot::Program* physical_mixed_program() {
  static constexpr std::array<std::string_view, 20> restricted{
      "f0",  "f1",  "f2",   "f3",   "lf0",  "lf1",  "lf2",    "lf3",     "rf0",     "rf1",
      "rf2", "rf3", "lrf0", "lrf1", "lrf2", "lrf3", "weight", "lweight", "rweight", "lrweight"};
  static constexpr std::array<std::string_view, 36> unrestricted{
      "f0",   "f1",   "f2",   "f3",   "f4",   "f5",     "f6",      "f7",      "lf0",
      "lf1",  "lf2",  "lf3",  "lf4",  "lf5",  "lf6",    "lf7",     "rf0",     "rf1",
      "rf2",  "rf3",  "rf4",  "rf5",  "rf6",  "rf7",    "lrf0",    "lrf1",    "lrf2",
      "lrf3", "lrf4", "lrf5", "lrf6", "lrf7", "weight", "lweight", "rweight", "lrweight"};
  static constexpr bool polarized = width<Domain> == 8;
  static const host_hessian_aot::Program program{
      polarized ? "gga_energy_mixed_polarized" : "gga_energy_mixed_unpolarized", point::kDomain,
      polarized ? std::span<const std::string_view>(unrestricted)
                : std::span<const std::string_view>(restricted),
      4, energy_mixed<Domain>};
  return &program;
}

inline const host_hessian_aot::Program* mixed_program(
    const host_hessian_aot::MethodBinding& method) {
  if (method.point_domain != PointDomain::linked_ad) {
    if (method.spins != host_hessian_aot::physical_spins(method.point_domain)) return nullptr;
    switch (method.point_domain) {
      case PointDomain::lda_rks:
        return physical_mixed_program<PointDomain::lda_rks>();
      case PointDomain::pbe_rks:
        return physical_mixed_program<PointDomain::pbe_rks>();
      case PointDomain::scaled_pbe0_rks:
        return physical_mixed_program<PointDomain::scaled_pbe0_rks>();
      case PointDomain::scaled_pbe0_uks:
        return physical_mixed_program<PointDomain::scaled_pbe0_uks>();
      case PointDomain::pbe_uks:
        return physical_mixed_program<PointDomain::pbe_uks>();
      case PointDomain::lda_uks:
        return physical_mixed_program<PointDomain::lda_uks>();
      case PointDomain::linked_ad:
        break;
    }
  }
  return host_hessian_aot::find(
      method.spins == 1 ? "gga_energy_mixed_unpolarized" : "gga_energy_mixed_polarized",
      method.energy_mixed_identity);
}

}  // namespace generativeqc::dft::host_hessian_point
