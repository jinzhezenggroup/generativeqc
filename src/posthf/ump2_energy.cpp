#include "posthf/ump2_energy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "posthf/native_provider.hpp"
#include "posthf/ump2_cpu_generated.hpp"

namespace generativeqc::mp2 {
namespace {

constexpr std::array<std::array<unsigned, 2>, 3> kChannels{{{0, 0}, {1, 1}, {0, 1}}};

double checked_denominators(const core::ElectronicReferenceView& reference, double threshold) {
  if (!std::isfinite(threshold) || threshold <= 0)
    throw std::invalid_argument("UMP2 denominator threshold must be finite and positive");
  double minimum = std::numeric_limits<double>::infinity();
  const auto n = reference.basis_functions;
  for (const auto [left, right] : kChannels) {
    const auto& a = reference.channels[left];
    const auto& b = reference.channels[right];
    if (!a.occupied || !b.occupied || a.occupied == n || b.occupied == n) continue;
    const auto occupied_a = a.orbital_energies.first(a.occupied);
    const auto occupied_b = b.orbital_energies.first(b.occupied);
    const auto virtual_a = a.orbital_energies.subspan(a.occupied);
    const auto virtual_b = b.orbital_energies.subspan(b.occupied);
    const double nearest = *std::max_element(occupied_a.begin(), occupied_a.end()) +
                           *std::max_element(occupied_b.begin(), occupied_b.end()) -
                           *std::min_element(virtual_a.begin(), virtual_a.end()) -
                           *std::min_element(virtual_b.begin(), virtual_b.end());
    const double farthest = *std::min_element(occupied_a.begin(), occupied_a.end()) +
                            *std::min_element(occupied_b.begin(), occupied_b.end()) -
                            *std::max_element(virtual_a.begin(), virtual_a.end()) -
                            *std::max_element(virtual_b.begin(), virtual_b.end());
    if (!std::isfinite(nearest) || !std::isfinite(farthest))
      throw std::invalid_argument("nonfinite UMP2 denominator extrema");
    if (nearest >= 0)
      throw std::invalid_argument("UMP2 occupied energies must be below virtual energies");
    if (-nearest <= threshold)
      throw std::invalid_argument("near-zero UMP2 denominator; no regularization applied");
    minimum = std::min(minimum, -nearest);
  }
  return std::isinf(minimum) ? 0.0 : minimum;
}

}  // namespace

UnrestrictedEnergy conventional_unrestricted_energy(
    const hf::UnrestrictedPhysicalReference& owned,
    const integrals::ElectronInteractionSource& source, std::size_t budget, double threshold) {
  const auto reference = owned.electronic_reference();
  core::validate_electronic_reference_shape(reference);
  if (reference.spin_channels != 2 || !std::isfinite(reference.energy))
    throw std::invalid_argument("UMP2 requires a finite validated UHF reference");
  const auto& source_system = source.orbital();
  if (!owned.source_identity || !owned.source_identity->matches(source_system) ||
      owned.source_charge != source_system.charge ||
      owned.source_electrons != source_system.electron_count ||
      owned.source_multiplicity != source_system.multiplicity)
    throw std::invalid_argument("stale UMP2 reference/source identity");
  for (const auto& channel : reference.channels) {
    if (!std::all_of(channel.orbital_energies.begin(), channel.orbital_energies.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("nonfinite UMP2 orbital energies");
  }
  // Check every active spin denominator before the first AO source read.
  UnrestrictedEnergy result;
  result.minimum_denominator = checked_denominators(reference, threshold);
  if (budget == 0) throw std::length_error("UMP2 numeric memory budget is zero");
  posthf::NativeBlockProvider provider(source, owned, budget, 2);
  const std::array<std::size_t, 4> shape{1, 1, 1, 1};
  const auto provider_peak = provider.batch_bytes(shape, 1, false);
  const auto n = reference.basis_functions;
  for (unsigned channel = 0; channel < 3; ++channel) {
    const auto program = generated::ump2_cpu_plan(channel);
    result.equation_hashes[channel] = program.equation_hash;
    result.numeric_capacity_bytes = std::max(
        result.numeric_capacity_bytes,
        posthf::checked_add(provider_peak, posthf::checked_add(program.numeric_bytes, 80)));
  }
  if (result.numeric_capacity_bytes > budget)
    throw std::length_error("UMP2 reference/source/transform/tile exceed numeric memory budget");

  std::array<double, 3> corrections{};
  for (unsigned channel = 0; channel < 3; ++channel) {
    const auto [left, right] = kChannels[channel];
    const auto& a = reference.channels[left];
    const auto& b = reference.channels[right];
    if (!a.occupied || !b.occupied || a.occupied == n || b.occupied == n) continue;
    const auto program = generated::ump2_cpu_plan(channel);
    for (std::size_t i = 0; i < a.occupied; ++i)
      for (std::size_t j = 0; j < b.occupied; ++j)
        for (std::size_t av = a.occupied; av < n; ++av)
          for (std::size_t bv = b.occupied; bv < n; ++bv) {
            const posthf::SpinMOSlots direct_request{
                posthf::MOSlots{std::vector<std::size_t>{i}, std::vector<std::size_t>{av},
                                std::vector<std::size_t>{j}, std::vector<std::size_t>{bv}},
                {left, left, right, right}};
            const auto direct = provider.get_spin(direct_request);
            const double g = direct.at(0);
            double x = 0.0;
            if (left == right) {
              const posthf::SpinMOSlots exchange_request{
                  posthf::MOSlots{std::vector<std::size_t>{i}, std::vector<std::size_t>{bv},
                                  std::vector<std::size_t>{j}, std::vector<std::size_t>{av}},
                  {left, left, right, right}};
              x = provider.get_spin(exchange_request).at(0);
            }
            double value = 0.0;
            program.run(g, x, a.orbital_energies[i], b.orbital_energies[j], a.orbital_energies[av],
                        b.orbital_energies[bv], &value);
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite UMP2 tile energy");
            const auto adjusted = value - corrections[channel];
            const auto updated = result.channels[channel] + adjusted;
            corrections[channel] = (updated - result.channels[channel]) - adjusted;
            result.channels[channel] = updated;
            if (!std::isfinite(updated))
              throw std::runtime_error("nonfinite UMP2 accumulated energy");
            ++result.tiles;
          }
  }
  return result;
}

}  // namespace generativeqc::mp2
