#include "posthf/ump2_energy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "cc/uccsd_executor.hpp"
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

double validate_unrestricted_energy_input(const hf::UnrestrictedPhysicalReference& owned,
                                          const integrals::ElectronInteractionSource& source,
                                          double threshold) {
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
  return checked_denominators(reference, threshold);
}

std::size_t unrestricted_cuda_energy_host_capacity(
    const hf::UnrestrictedPhysicalReference& reference,
    const posthf::NativeBlockProvider& provider) {
  const auto basis_count = reference.nbf;
  if (reference.nocc[0] > basis_count || reference.nocc[1] > basis_count)
    throw std::invalid_argument("UMP2 occupied count exceeds the orbital frame");
  auto peak = provider.batch_bytes({1, 1, 1, 1}, 1, true);
  for (const auto& channel : kChannels) {
    const auto left = channel[0], right = channel[1];
    const auto occupied_left = reference.nocc[left], occupied_right = reference.nocc[right];
    const auto virtual_left = basis_count - occupied_left;
    const auto virtual_right = basis_count - occupied_right;
    if (!occupied_left || !occupied_right || !virtual_left || !virtual_right) continue;
    const std::array<std::size_t, 4> extents{occupied_left, virtual_left, occupied_right,
                                             virtual_right};
    auto elements = std::size_t{1};
    for (const auto extent : extents) elements = posthf::checked_mul(elements, extent);
    // g and, for equal spins, x coexist with four orbital-energy feeds and
    // the returned scalar. The transformed block is already in batch_bytes.
    const auto feeds =
        posthf::checked_add(posthf::checked_mul(left == right ? 2 : 1, elements),
                            posthf::checked_add(posthf::checked_mul(2, basis_count), 1));
    peak = std::max(peak, posthf::checked_add(provider.batch_bytes(extents, 1, true),
                                              posthf::checked_mul(sizeof(double), feeds)));
  }
  return peak;
}

UnrestrictedEnergy conventional_unrestricted_energy(
    const hf::UnrestrictedPhysicalReference& owned,
    const integrals::ElectronInteractionSource& source, std::size_t budget, double threshold,
    bool cuda, int device) {
  UnrestrictedEnergy result;
  result.minimum_denominator = validate_unrestricted_energy_input(owned, source, threshold);
  const auto reference = owned.electronic_reference();
  if (budget == 0) throw std::length_error("UMP2 numeric memory budget is zero");
  posthf::NativeBlockProvider provider(
      source, owned, budget, cuda ? 8 : 2,
      cuda ? posthf::AOTileDomain::Basis : posthf::AOTileDomain::Shell);
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

  if (cuda) {
#if GENERATIVEQC_HAS_CUDA
    const auto peak = unrestricted_cuda_energy_host_capacity(owned, provider);
    if (peak >= budget) throw std::length_error("UMP2 CUDA host/provider capacity exceeds budget");
    auto executor = cc::make_unrestricted_cuda_executor(n, owned.nocc[0], owned.nocc[1], device,
                                                        budget - peak, false, true);
    result.numeric_capacity_bytes = posthf::checked_add(peak, executor->numeric_capacity_bytes());
    for (unsigned channel = 0; channel < 3; ++channel) {
      const auto [left, right] = kChannels[channel];
      const auto no_left = owned.nocc[left], no_right = owned.nocc[right], nv_left = n - no_left,
                 nv_right = n - no_right;
      if (!no_left || !no_right || !nv_left || !nv_right) continue;
      const std::array<std::size_t, 4> extents{no_left, nv_left, no_right, nv_right};
      posthf::SpinMOSlots request;
      request.spins = {left, left, right, right};
      for (unsigned axis = 0; axis < 4; ++axis) {
        request.slots[axis].resize(extents[axis]);
        std::iota(request.slots[axis].begin(), request.slots[axis].end(),
                  axis % 2 ? owned.nocc[request.spins[axis]] : 0);
      }
      const auto block = provider.get_spin(request, nullptr, true, device);
      cc::UnrestrictedFeeds inputs;
      auto& direct = inputs["g"];
      direct.resize(block.size());
      for (std::size_t occupied_left = 0; occupied_left < no_left; ++occupied_left)
        for (std::size_t occupied_right = 0; occupied_right < no_right; ++occupied_right)
          for (std::size_t virtual_left = 0; virtual_left < nv_left; ++virtual_left)
            for (std::size_t virtual_right = 0; virtual_right < nv_right; ++virtual_right)
              direct[((occupied_left * no_right + occupied_right) * nv_left + virtual_left) *
                         nv_right +
                     virtual_right] =
                  block[((occupied_left * nv_left + virtual_left) * no_right + occupied_right) *
                            nv_right +
                        virtual_right];
      if (left == right) {
        auto& exchange = inputs["x"];
        exchange.resize(direct.size());
        for (std::size_t first = 0; first < no_left; ++first)
          for (std::size_t second = 0; second < no_right; ++second)
            for (std::size_t first_virtual = 0; first_virtual < nv_left; ++first_virtual)
              for (std::size_t second_virtual = 0; second_virtual < nv_right; ++second_virtual)
                exchange[((first * no_right + second) * nv_left + first_virtual) * nv_right +
                         second_virtual] =
                    direct[((first * no_right + second) * nv_left + second_virtual) * nv_right +
                           first_virtual];
      }
      inputs["ei"].assign(owned.orbital_energies[left].begin(),
                          owned.orbital_energies[left].begin() + no_left);
      inputs["ej"].assign(owned.orbital_energies[right].begin(),
                          owned.orbital_energies[right].begin() + no_right);
      inputs["ea"].assign(owned.orbital_energies[left].begin() + no_left,
                          owned.orbital_energies[left].end());
      inputs["eb"].assign(owned.orbital_energies[right].begin() + no_right,
                          owned.orbital_energies[right].end());
      const std::array<std::string, 3> names{"alpha_alpha", "beta_beta", "alpha_beta"};
      result.channels[channel] =
          executor->evaluate("mp2_" + names[channel], inputs).at("energy")[0];
      result.tiles = posthf::checked_add(result.tiles, direct.size());
    }
    return result;
#else
    throw std::invalid_argument("CUDA UMP2 requires a CUDA build");
#endif
  }

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
