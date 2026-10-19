#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "scf/ks_host_resources.hpp"

extern "C" int generativeqc_resource_ks_cpu_host_v1(const std::uint64_t*, std::size_t,
                                                    std::uint64_t*, std::size_t);
extern "C" int generativeqc_resource_ks_cpu_fleet_v1(const std::uint64_t*, std::size_t,
                                                     std::uint64_t, std::uint64_t*, std::size_t);

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void test_fleet_owner() {
  using generativeqc::scf::CpuKsHostFleetResources;
  const generativeqc::scf::CpuKsHostResourceShape shape{2,  2, 2,   2, 6,   1,    0,     0, 6144,
                                                        24, 8, 256, 8, 100, true, false, 0};
  CpuKsHostFleetResources fleet;
  const auto item = fleet.append(shape, 3);
  auto aggregate = fleet.inventory();
  require(aggregate[12] == 3 * 263184 && aggregate[13] == 415600 &&
              aggregate[14] == 3 * 263184 + 415600,
          "KS fleet must sum retained owners and share serialized workspace");
  for (std::size_t index = 0; index < 12; ++index)
    require(aggregate[index] == 3 * item[index], "KS persistent component sum is incomplete");
  CpuKsHostFleetResources repeated;
  for (std::size_t index = 0; index < 3; ++index) repeated.append(shape);
  require(repeated.inventory() == aggregate, "KS repetition differs from descriptor fleet");
  auto invalid = shape;
  invalid.primitives = 0;
  for (const auto copies : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
    try {
      fleet.append(shape, copies);
      require(false, "invalid KS repetition was accepted");
    } catch (const std::invalid_argument&) {
    } catch (const std::overflow_error&) {
    }
    require(fleet.inventory() == aggregate, "failed KS repetition mutated fleet");
  }
  try {
    fleet.append(invalid);
    require(false, "invalid KS item was accepted");
  } catch (const std::invalid_argument&) {
  }
  require(fleet.inventory() == aggregate, "invalid KS item mutated fleet");
  constexpr auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  CpuKsHostFleetResources exact(maximum - item[13]);
  exact.append(shape);
  aggregate = exact.inventory();
  require(aggregate[14] == maximum, "exact signed-int64 KS fleet peak was rejected");
  try {
    exact.append(shape);
    require(false, "KS retained-plus-workspace overflow was accepted");
  } catch (const std::overflow_error&) {
  }
  require(exact.inventory() == aggregate, "KS peak overflow mutated fleet");
  try {
    CpuKsHostFleetResources excessive(maximum + 1);
    require(false, "out-of-domain KS workspace floor was accepted");
  } catch (const std::overflow_error&) {
  }
}

void test_fleet_bridge() {
  std::array<std::uint64_t, 34> shapes{};
  const std::array<std::uint64_t, 17> first{2,  2, 2,   2, 6,   1, 0, 0, 6144,
                                            24, 8, 256, 8, 100, 1, 0, 0};
  std::copy(first.begin(), first.end(), shapes.begin());
  std::copy(first.begin(), first.end(), shapes.begin() + 17);
  shapes[17 + 5] = 2;
  shapes[17 + 15] = 1;
  shapes[17 + 16] = 768;
  std::array<std::uint64_t, 47> values{};
  require(generativeqc_resource_ks_cpu_fleet_v1(shapes.data(), shapes.size(), 1 << 20,
                                                values.data(), values.size()) == 0,
          "ragged KS fleet bridge failed");
  for (std::size_t index = 0; index < 12; ++index)
    require(values[index] == values[15 + index] + values[31 + index],
            "ragged KS component total is incorrect");
  require(values[12] == values[28] + values[44] && values[13] == 1 << 20 &&
              values[14] == values[12] + values[13] && values[43] == 0,
          "KS fleet floor, retained lifetime or UKS cache is incorrect");
  const auto preserves = [&] {
    return std::all_of(values.begin(), values.end(), [](auto value) { return value == 314159; });
  };
  for (const auto slot : {0u, 4u, 14u, 15u, 16u}) {
    auto invalid = shapes;
    invalid[17 + slot] = (slot == 14 || slot == 15) ? 2 : std::numeric_limits<std::uint64_t>::max();
    values.fill(314159);
    require(generativeqc_resource_ks_cpu_fleet_v1(invalid.data(), invalid.size(), 0, values.data(),
                                                  values.size()) != 0 &&
                preserves(),
            "invalid later KS item published partial output");
  }
  values.fill(314159);
  for (const auto lengths : {std::array<std::size_t, 2>{0, 47}, {33, 47}, {34, 46}, {34, 48}})
    require(generativeqc_resource_ks_cpu_fleet_v1(shapes.data(), lengths[0], 0, values.data(),
                                                  lengths[1]) != 0 &&
                preserves(),
            "malformed KS fleet schema published output");
  require(generativeqc_resource_ks_cpu_fleet_v1(nullptr, 34, 0, values.data(), 47) != 0 &&
              generativeqc_resource_ks_cpu_fleet_v1(shapes.data(), 34, 0, nullptr, 47) != 0,
          "null KS fleet storage was accepted");
  std::array<std::uint64_t, 64> overlap{};
  std::copy(first.begin(), first.end(), overlap.begin());
  const auto unchanged = overlap;
  for (const auto offset : {0u, 1u})
    require(generativeqc_resource_ks_cpu_fleet_v1(overlap.data(), 17, 0, overlap.data() + offset,
                                                  31) != 0 &&
                overlap == unchanged,
            "overlapping KS buffers were accepted or modified");
  std::array<std::uint64_t, 48> adjacent{};
  std::copy(first.begin(), first.end(), adjacent.begin());
  require(generativeqc_resource_ks_cpu_fleet_v1(adjacent.data(), 17, 0, adjacent.data() + 17, 31) ==
                  0 &&
              std::equal(first.begin(), first.end(), adjacent.begin()),
          "disjoint adjacent KS buffers were rejected or input was modified");
  const auto huge_items = std::numeric_limits<std::size_t>::max() / 17;
  require(generativeqc_resource_ks_cpu_fleet_v1(shapes.data(), huge_items * 17, 0, values.data(),
                                                15 + 16 * huge_items) != 0 &&
              preserves(),
          "overflowing KS byte ranges were accepted");
  constexpr auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  const auto retained = generativeqc::scf::cpu_ks_host_resources(
      {2, 2, 2, 2, 6, 1, 0, 0, 6144, 24, 8, 256, 8, 100, true, false, 0})[13];
  std::array<std::uint64_t, 31> single{};
  require(generativeqc_resource_ks_cpu_fleet_v1(first.data(), 17, maximum - retained, single.data(),
                                                single.size()) == 0 &&
              single[14] == maximum,
          "exact KS bridge peak was rejected");
  single.fill(314159);
  require(generativeqc_resource_ks_cpu_fleet_v1(first.data(), 17, maximum + 1, single.data(),
                                                single.size()) != 0 &&
              std::all_of(single.begin(), single.end(), [](auto value) { return value == 314159; }),
          "out-of-domain KS bridge floor published output");
  require(generativeqc_resource_ks_cpu_fleet_v1(first.data(), 17, maximum - retained + 1,
                                                single.data(), single.size()) != 0 &&
              std::all_of(single.begin(), single.end(), [](auto value) { return value == 314159; }),
          "overflowing KS bridge peak published output");
}
}  // namespace

int main() {
  test_fleet_owner();
  test_fleet_bridge();
  const std::array<std::uint64_t, 17> shape{2,  2, 2,   2, 6,   1, 0, 0, 6144,
                                            24, 8, 256, 8, 100, 1, 0, 0};
  std::array<std::uint64_t, 16> inventory{};
  require(generativeqc_resource_ks_cpu_host_v1(shape.data(), shape.size(), inventory.data(),
                                               inventory.size()) == 0,
          "CPU host shape query failed");
  require(inventory[0] == 15360 && inventory[13] == 263184 && inventory[14] == 33504 &&
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
              inventory[11] == 48 * shape[8] && inventory[14] == 33504 + 768,
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
