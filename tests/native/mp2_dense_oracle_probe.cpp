/** Execute the real dense oracle with scoped allocation receipts.
 * Inputs and JSON serialization are outside the numeric-payload observation.
 * Failed guards may allocate exception storage; no zero-heap claim is made.
 */
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "posthf/mp2_gradient.hpp"

namespace allocation {
struct alignas(std::max_align_t) Header {
  std::size_t bytes;
  bool tracked;
};
bool active{};
std::size_t live{}, peak{}, calls{}, largest{};
std::array<std::size_t, 32> requests{};
}  // namespace allocation

void* operator new(std::size_t bytes) {
  if (bytes > std::numeric_limits<std::size_t>::max() - sizeof(allocation::Header))
    throw std::bad_alloc();
  auto* header = static_cast<allocation::Header*>(
      std::malloc(sizeof(allocation::Header) + std::max(bytes, std::size_t{1})));
  if (!header) throw std::bad_alloc();
  *header = {bytes, allocation::active};
  if (header->tracked) {
    allocation::live += bytes;
    allocation::peak = std::max(allocation::peak, allocation::live);
    allocation::largest = std::max(allocation::largest, bytes);
    if (allocation::calls < allocation::requests.size())
      allocation::requests[allocation::calls] = bytes;
    ++allocation::calls;
  }
  return header + 1;
}
void operator delete(void* pointer) noexcept {
  if (!pointer) return;
  auto* header = static_cast<allocation::Header*>(pointer) - 1;
  if (header->tracked) allocation::live -= header->bytes;
  std::free(header);
}
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete(pointer); }

namespace {
void read(std::span<double> values) {
  for (auto& value : values)
    if (!(std::cin >> value)) throw std::invalid_argument("missing finite probe input");
}
void vector_json(std::span<const double> values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}
}  // namespace

int main(int count, char** arguments) {
  using namespace generativeqc::mp2;
  if (count != 5) return 2;
  const std::string mode = arguments[1];
  const auto orbitals = static_cast<std::size_t>(std::stoull(arguments[2]));
  const auto occupied = static_cast<std::size_t>(std::stoull(arguments[3]));
  const auto budget = static_cast<std::size_t>(std::stoull(arguments[4]));
  const bool relaxed = mode == "weights" || mode == "legacy-weights" ||
                       mode == "invalid-response" || mode == "nonfinite-response";
  const bool legacy = mode == "legacy-rhs" || mode == "legacy-weights";
  const bool plan_only = mode == "plan-rhs" || mode == "plan-weights";
  const bool oversized = mode == "oversized-rhs" || mode == "oversized-weights" ||
                         mode == "oversized-budget-rhs" || mode == "oversized-budget-weights";
  DenseOrbitalOraclePlan plan;
  EnergyAdjoint adjoint;
  adjoint.orbitals = orbitals;
  adjoint.occupied = occupied;
  std::vector<double> hcore, eri, response;
  OrbitalRhs rhs;
  LagrangianWeights weights;
  if (!plan_only && !oversized) {
    std::vector<double> energies(orbitals);
    hcore.resize(orbitals * orbitals);
    eri.resize(orbitals * orbitals * orbitals * orbitals);
    response.resize(occupied * (orbitals - occupied));
    read(hcore);
    read(eri);
    read(energies);
    read(response);
    if (mode == "invalid-response") response.resize(response.size() - 1);
    if (mode == "nonfinite-response") response.front() = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> integrals(occupied * occupied * (orbitals - occupied) *
                                  (orbitals - occupied));
    for (std::size_t first = 0; first < occupied; ++first)
      for (std::size_t second = 0; second < occupied; ++second)
        for (std::size_t left = 0; left < orbitals - occupied; ++left)
          for (std::size_t right = 0; right < orbitals - occupied; ++right) {
            const auto input =
                ((first * orbitals + occupied + left) * orbitals + second) * orbitals + occupied +
                right;
            const auto output = ((first * occupied + second) * (orbitals - occupied) + left) *
                                    (orbitals - occupied) +
                                right;
            integrals[output] = eri[input];
          }
    adjoint = canonical_energy_adjoint(integrals, energies, occupied, 1e-10);
  }
  const char* error = nullptr;
  allocation::active = true;
  try {
    if (plan_only) {
      plan = mode == "plan-rhs" ? dense_orbital_rhs_plan(orbitals, occupied, budget)
                                : dense_lagrangian_weights_plan(orbitals, occupied, budget);
    } else if (oversized) {
      if (mode == "oversized-rhs")
        rhs = canonical_orbital_rhs(hcore, eri, adjoint, 1e-10);
      else if (mode == "oversized-weights")
        weights = canonical_lagrangian_weights(hcore, eri, adjoint, response, 1e-10);
      else if (mode == "oversized-budget-rhs")
        rhs = canonical_orbital_rhs_with_budget(hcore, eri, adjoint, 1e-10, budget);
      else
        weights =
            canonical_lagrangian_weights_with_budget(hcore, eri, adjoint, response, 1e-10, budget);
    } else if (relaxed) {
      weights = legacy ? canonical_lagrangian_weights(hcore, eri, adjoint, response, 1e-10)
                       : canonical_lagrangian_weights_with_budget(hcore, eri, adjoint, response,
                                                                  1e-10, budget);
    } else {
      rhs = legacy ? canonical_orbital_rhs(hcore, eri, adjoint, 1e-10)
                   : canonical_orbital_rhs_with_budget(hcore, eri, adjoint, 1e-10, budget);
    }
  } catch (const std::length_error&) {
    error = "length_error";
  } catch (const std::invalid_argument&) {
    error = "invalid_argument";
  }
  allocation::active = false;
  const auto retained = allocation::live;
  std::cout << std::setprecision(17) << "{\"error\":";
  if (error)
    std::cout << '"' << error << '"';
  else
    std::cout << "null";
  std::cout << ",\"measured_peak_bytes\":" << allocation::peak
            << ",\"measured_retained_bytes\":" << retained
            << ",\"allocation_calls\":" << allocation::calls
            << ",\"largest_request\":" << allocation::largest << ",\"allocation_requests\":[";
  for (std::size_t index = 0; index < std::min(allocation::calls, allocation::requests.size());
       ++index) {
    if (index) std::cout << ',';
    std::cout << allocation::requests[index];
  }
  std::cout << "],\"dropped_requests\":"
            << (allocation::calls > allocation::requests.size()
                    ? allocation::calls - allocation::requests.size()
                    : 0)
            << ",\"plan\":";
  if (plan_only && !error)
    std::cout << "{\"peak_bytes\":" << plan.peak_bytes
              << ",\"retained_bytes\":" << plan.retained_bytes
              << ",\"dense_weight_bytes\":" << plan.dense_weight_bytes
              << ",\"budget_bytes\":" << plan.budget_bytes << '}';
  else
    std::cout << "null";
  std::cout << ",\"one\":";
  vector_json(relaxed ? std::span<const double>(weights.one_electron)
                      : std::span<const double>(rhs.one_electron));
  std::cout << ",\"two\":";
  vector_json(relaxed ? std::span<const double>(weights.two_electron)
                      : std::span<const double>(rhs.two_electron));
  std::cout << ",\"energy_gradient\":";
  vector_json(rhs.energy_gradient);
  std::cout << ",\"response_rhs\":";
  vector_json(rhs.response_rhs);
  std::cout << ",\"overlap\":";
  vector_json(weights.overlap);
  std::cout << ",\"stationarity_residual\":" << weights.stationarity_residual;
  rhs = {};
  weights = {};
  std::cout << ",\"final_live_bytes\":" << allocation::live << "}\n";
  return 0;
}
