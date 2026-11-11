#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace generativeqc::integrals {

/** Bounded CPU consumer of compiler-generated factorized ERI jets.
 * Independent mathematical slots remain independent even at repeated centers.
 * Records contain four exponents, twelve coordinates and one radial scale.
 * No scientific recurrence, density policy or molecular tensor lives here.
 * Operations select raw value/gradient (1), weighted gradient-JVP (2), or
 * raw value/scalar-direction (3). A zero direction never changes the operation.
 * Invalid input and late arithmetic failure leave output and counters intact.
 */
template <class Program>
int factorized_eri_records(unsigned operation, const unsigned* angular, const unsigned* components,
                           std::size_t component_count, const double* records, std::size_t count,
                           std::size_t stride, const double* weights, const double* direction,
                           double* output, std::size_t output_size, std::uint64_t* visits) {
  constexpr std::size_t maximum_components = 64;
  constexpr std::size_t first_width = 13;
  const std::size_t component_width = operation == 3 ? 2 : first_width;
  if ((operation != 1 && operation != 2 && operation != 3) || !angular || !components || !output ||
      !visits || !component_count || component_count > maximum_components || count > 128 ||
      (count && !records) || stride != 17 ||
      output_size != (operation == 2 ? 12 : component_width * component_count) ||
      (operation == 2 && !weights) || (operation != 1 && !direction))
    return 1;
  std::size_t total_components = 1;
  for (unsigned slot = 0; slot < 4; ++slot) {
    if (angular[slot] > 3) return 1;
    total_components *= (angular[slot] + 1) * (angular[slot] + 2) / 2;
  }
  for (std::size_t component = 0; component < component_count; ++component) {
    if (components[component] >= total_components ||
        (operation == 2 && !std::isfinite(weights[component])))
      return 1;
    for (std::size_t previous = 0; previous < component; ++previous)
      if (components[previous] == components[component]) return 1;
  }
  std::array<double, 12> motion{};
  if (operation != 1)
    for (unsigned coordinate = 0; coordinate < motion.size(); ++coordinate) {
      if (!std::isfinite(direction[coordinate])) return 1;
      motion[coordinate] = direction[coordinate];
    }
  std::array<double, maximum_components * first_width> candidate{};
  for (std::size_t record = 0; record < count; ++record) {
    const auto* input = records + stride * record;
    for (std::size_t field = 0; field < stride; ++field)
      if (!std::isfinite(input[field]) || (field < 4 && input[field] <= 0)) return 1;
    // Geometry and radial moments are invariant across this record's component
    // page. Keep the generated binding local so another record cannot reuse it.
    typename Program::Prepared geometry{};
    if (!Program::prepare(angular, input, geometry)) return 5;
    for (std::size_t component = 0; component < component_count; ++component) {
      std::array<double, 25> jet{};
      if (!Program::evaluate(angular, components[component], input, motion.data(), geometry,
                             jet.data()))
        return 5;
      const double scale = input[16] * (operation == 2 ? weights[component] : 1.0);
      if (!std::isfinite(scale)) return 5;
      const auto width = operation == 2 ? 12 : component_width;
      for (std::size_t coordinate = 0; coordinate < width; ++coordinate) {
        const auto target = operation == 2 ? coordinate : component_width * component + coordinate;
        const double term = scale * jet[(operation == 2 ? 13 : 0) + coordinate];
        candidate[target] += term;
        if (!std::isfinite(term) || !std::isfinite(candidate[target])) return 5;
      }
    }
  }
  for (std::size_t coordinate = 0; coordinate < output_size; ++coordinate)
    output[coordinate] = candidate[coordinate];
  visits[0] = count;
  visits[1] = count * component_count;
  return 0;
}

}  // namespace generativeqc::integrals
