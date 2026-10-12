#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "scf/ks_host_resources.hpp"

extern "C" int generativeqc_resource_ks_cpu_host_v1(const std::uint64_t*, std::size_t,
                                                    std::uint64_t*, std::size_t);

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
  const std::array<std::uint64_t, 17> shape{2,  2, 2,   2, 6,   1, 0, 0, 6144,
                                            24, 8, 256, 8, 100, 1, 0, 0};
  std::array<std::uint64_t, 16> inventory{};
  require(generativeqc_resource_ks_cpu_host_v1(shape.data(), shape.size(), inventory.data(),
                                               inventory.size()) == 0,
          "CPU host shape query failed");
  require(inventory[0] == 15360 && inventory[13] == 263184 && inventory[14] == 32864 &&
              inventory[15] == 415600 && inventory[12] == 393216,
          "CPU host H2 lifetime bounds changed");
  auto unrestricted = shape;
  unrestricted[5] = 2;
  require(generativeqc_resource_ks_cpu_host_v1(unrestricted.data(), unrestricted.size(),
                                               inventory.data(), inventory.size()) == 0 &&
              inventory[12] == 0,
          "UKS must not retain the restricted AO cache");
  auto nonlocal = shape;
  nonlocal[15] = 1;
  nonlocal[16] = 768;
  require(generativeqc_resource_ks_cpu_host_v1(nonlocal.data(), nonlocal.size(), inventory.data(),
                                               inventory.size()) == 0 &&
              inventory[11] == 48 * shape[8] && inventory[14] == 32864 + 768,
          "nonlocal/ECP numeric owners are missing");
  for (const auto slot : {0u, 1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 13u}) {
    auto invalid = shape;
    invalid[slot] = 0;
    inventory.fill(314159);
    require(generativeqc_resource_ks_cpu_host_v1(invalid.data(), invalid.size(), inventory.data(),
                                                 inventory.size()) != 0 &&
                std::all_of(inventory.begin(), inventory.end(),
                            [](auto value) { return value == 314159; }),
            "invalid topology published partial capacities");
  }
  for (const auto slot : {0u, 1u, 6u, 12u, 16u}) {
    auto invalid = shape;
    invalid[slot] = std::numeric_limits<std::uint64_t>::max();
    inventory.fill(314159);
    require(generativeqc_resource_ks_cpu_host_v1(invalid.data(), invalid.size(), inventory.data(),
                                                 inventory.size()) != 0 &&
                inventory[0] == 314159,
            "overflow or unsupported angular topology was accepted");
  }
  inventory.fill(314159);
  for (const auto lengths : {std::array<std::size_t, 2>{16, 16}, {17, 15}, {18, 16}})
    require(generativeqc_resource_ks_cpu_host_v1(shape.data(), lengths[0], inventory.data(),
                                                 lengths[1]) != 0 &&
                inventory[0] == 314159,
            "wrong bridge schema length was accepted");
  auto invalid = shape;
  invalid[14] = 2;
  require(generativeqc_resource_ks_cpu_host_v1(invalid.data(), invalid.size(), inventory.data(),
                                               inventory.size()) != 0,
          "noncanonical boolean was accepted");
  require(generativeqc_resource_ks_cpu_host_v1(nullptr, 17, inventory.data(), 16) != 0 &&
              generativeqc_resource_ks_cpu_host_v1(shape.data(), 17, nullptr, 16) != 0,
          "null shape/publication storage was accepted");
}
