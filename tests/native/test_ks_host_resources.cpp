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
extern "C" int generativeqc_resource_ks_cpu_df_fleet_v1(const std::uint64_t*, std::size_t,
                                                        std::uint64_t, std::uint64_t*, std::size_t);
extern "C" int generativeqc_resource_ks_cpu_df_force_fleet_v1(const std::uint64_t*, std::size_t,
                                                              std::uint64_t, std::uint64_t*,
                                                              std::size_t);
extern "C" int generativeqc_resource_ks_cpu_packed_fleet_v1(const std::uint64_t*, std::size_t,
                                                            std::uint64_t, std::uint64_t*,
                                                            std::size_t);
extern "C" int generativeqc_resource_ks_cpu_packed_force_fleet_v1(const std::uint64_t*, std::size_t,
                                                                  std::uint64_t, std::uint64_t*,
                                                                  std::size_t);
extern "C" int generativeqc_resource_ks_cpu_force_fleet_v1(const std::uint64_t*, std::size_t,
                                                           std::uint64_t*, std::size_t);

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
  CpuKsHostFleetResources packed;
  constexpr auto packed_execution = generativeqc::scf::KsHostExecution::cpu_packed;
  const auto packed_item = packed.append(shape, 3, packed_execution);
  constexpr std::uint64_t dense_values = 16 * sizeof(double);
  constexpr std::uint64_t packed_values = 6 * sizeof(double);
  const auto savings = dense_values - packed_values;
  require(item[5] - packed_item[5] == savings && item[13] - packed_item[13] == savings &&
              item[14] == packed_item[14] && item[15] == packed_item[15],
          "packed KS changes only the retained ERI values, not transient workspace");
  const auto packed_aggregate = packed.inventory();
  require(packed_aggregate[12] == aggregate[12] - 3 * savings &&
              packed_aggregate[13] == aggregate[13] &&
              packed_aggregate[14] == aggregate[14] - 3 * savings,
          "packed KS values must remain resident for each item");
  auto through_g = shape;
  through_g.maximum_angular = 4;
  try {
    packed.append(through_g, 1, packed_execution);
    require(false, "packed KS accepted a g shell");
  } catch (const std::invalid_argument&) {
  }
  require(packed.inventory() == packed_aggregate, "failed packed request mutated fleet");
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

void test_packed_fleet_bridge() {
  const std::array<std::uint64_t, 17> first{2,  2, 2,   2, 6,   1, 0, 0, 6144,
                                            24, 8, 256, 8, 100, 1, 0, 0};
  std::array<std::uint64_t, 34> shapes{};
  std::copy(first.begin(), first.end(), shapes.begin());
  std::copy(first.begin(), first.end(), shapes.begin() + first.size());
  shapes[18] = shapes[19] = 3;
  auto query = [&](std::uint64_t cap, auto& output) {
    return generativeqc_resource_ks_cpu_packed_fleet_v1(shapes.data(), shapes.size(), cap,
                                                        output.data(), output.size());
  };
  constexpr auto cap = 8 * (6 + 21);
  std::array<std::uint64_t, 47> values{};
  require(query(cap, values) == 0, "exact-fit packed KS fleet denied");
  require(values[5] == 16 * (4 + 9) + cap && values[14] == values[12] + values[13],
          "packed KS fleet returned the wrong provider/peak");
  const auto committed = values;
  require(query(cap + 1, values) == 0 && values == committed,
          "packed KS fleet charged the caller ceiling");
  require(query(cap - 1, values) == 1 && values == committed,
          "short packed KS capacity mutated output or returned the wrong status");
  shapes[23] = 4;
  require(query(cap, values) == -1 && values == committed,
          "invalid packed KS item mutated output or returned the wrong status");
}

void test_packed_force_fleet_bridge() {
  const std::array<std::uint64_t, 34> shapes{2, 2,    2,  2, 6,   1, 0,   0, 6144, 24, 8, 256,
                                             8, 100,  1,  0, 0,   3, 3,   3, 3,    9,  2, 0,
                                             0, 9216, 24, 8, 256, 8, 100, 1, 0,    0};
  constexpr std::uint64_t cap = 8 * (6 + 21);
  std::array<std::uint64_t, 47> energy{}, force{};
  require(generativeqc_resource_ks_cpu_packed_fleet_v1(shapes.data(), shapes.size(), cap,
                                                       energy.data(), energy.size()) == 0,
          "exact packed energy fleet was rejected");
  const auto query = [&](const auto& input, std::uint64_t bytes) {
    return generativeqc_resource_ks_cpu_packed_force_fleet_v1(input.data(), input.size(), bytes,
                                                              force.data(), force.size());
  };
  const auto status = query(shapes, cap);
  std::array<std::uint64_t, 47> dense_force{};
  const auto dense_query = [&](const auto& input) {
    return generativeqc_resource_ks_cpu_force_fleet_v1(input.data(), input.size(),
                                                       dense_force.data(), dense_force.size());
  };
  if (status == 2) {
    require(force == decltype(force){}, "unlinked force inventory published output");
    require(dense_query(shapes) == 2 && dense_force == decltype(dense_force){},
            "unlinked dense force inventory published output");
    return;
  }
  require(status == 0, "exact packed force fleet was rejected");
  const auto cap_and_frame = (64u << 20) + 8 * 2 * (4 * 9 + 3);
  require(force[5] == energy[5] && force[12] == energy[12] + 24 * (2 + 3) &&
              force[13] == cap_and_frame && force[14] == force[12] + force[13],
          "force fleet changed the provider or summed serialized force scratch");
  const auto admitted = force;
  require(query(shapes, cap - 1) == 1 && force == admitted,
          "short packed force cap published output");
  auto invalid = shapes;
  invalid[17 + 15] = 1;
  require(query(invalid, cap) == -1 && force == admitted,
          "nonlocal packed force query published partial output");
  require(dense_query(shapes) == 0 && dense_force[5] == force[5] + 8 * (16 + 81) - cap &&
              dense_force[12] == force[12] + 8 * (16 + 81) - cap && dense_force[13] == force[13] &&
              dense_force[14] == dense_force[12] + dense_force[13],
          "dense force bridge did not replace the packed provider with dense values");
  const auto dense_admitted = dense_force;
  require(dense_query(invalid) == -1 && dense_force == dense_admitted,
          "invalid later dense force item modified committed output");

  // A large explicit AO tile can dominate the integral publication cap.
  auto large_tile = shapes;
  large_tile[17 + 8] = large_tile[17 + 11] = 300'000;
  require(query(large_tile, cap) == 0, "large valid force tile was rejected");
  const auto expected_xc = 8 * (10 * 3 * 300'000 + 6 * 3 + 4 * 3 + 300'000 + 3 * 9 + 5 * 3);
  require(force[13] == expected_xc + 96 * 3 + 8 * 2 * (4 * 9 + 3),
          "force fleet did not price the overlapping AO and Becke buffers");
}
void test_df_fleet_bridge() {
  const std::array<std::uint64_t, 22> shape{2, 7, 8,  4, 12, 1, 2,  0,  64, 4,  4,
                                            8, 6, 40, 1, 0,  0, 12, 14, 7,  20, 2};
  std::array<std::uint64_t, 44> shapes{};
  std::copy(shape.begin(), shape.end(), shapes.begin());
  std::copy(shape.begin(), shape.end(), shapes.begin() + shape.size());
  shapes[22 + 17] = 14;
  std::array<std::uint64_t, 47> values{};
  const auto query = [&] {
    return generativeqc_resource_ks_cpu_df_fleet_v1(shapes.data(), shapes.size(), 0, values.data(),
                                                    values.size());
  };
  require(query() == 0, "valid heterogeneous CPU DF fleet refused");
  const auto one_electron = 16 * 7 * 7 * 7 + 8 * 6;
  const auto provider = [&](std::uint64_t auxiliary) {
    return one_electron + 8 * auxiliary * auxiliary + 3 * 8 * 7 * 7 * auxiliary;
  };
  require(values[15 + 5] == provider(12) && values[31 + 5] == provider(14),
          "DF provider did not price raw and whitened factors independently");
  require(values[12] == values[15 + 13] + values[31 + 13] &&
              values[13] == std::max({values[29], values[30], values[45], values[46]}) &&
              values[14] == values[12] + values[13],
          "DF fleet did not preserve retained versus serialized lifetimes");
  values.fill(7);
  shapes[22 + 18] = 13;
  require(query() != 0 &&
              std::all_of(values.begin(), values.end(), [](auto value) { return value == 7; }),
          "invalid late auxiliary topology published partial capacity");
  shapes[22 + 18] = shapes[22 + 17] = std::numeric_limits<std::uint64_t>::max();
  require(query() != 0 && values[0] == 7, "DF capacity overflow was not rejected");
}

void test_df_force_fleet_bridge() {
  // With a large auxiliary basis, both the metric VJP and Cartesian lift
  // exceed the ordinary SCF phase. The larger one, not their sum, dominates.
  std::array<std::uint64_t, 22> shape{3, 7, 8,  4, 12, 2, 2,   0,   64,  4,   4,
                                      8, 6, 40, 1, 0,  0, 256, 300, 160, 370, 3};
  std::array<std::uint64_t, 31> force{}, energy{};
  const auto force_query = [&] {
    return generativeqc_resource_ks_cpu_df_force_fleet_v1(shape.data(), shape.size(), 0,
                                                          force.data(), force.size());
  };
  require(
      force_query() == 0 && generativeqc_resource_ks_cpu_df_fleet_v1(
                                shape.data(), shape.size(), 0, energy.data(), energy.size()) == 0,
      "valid fitted force topology refused");
  const auto reverse = 3 * 8 * 49 * 256 + 12 * 8 * 256 * 256 + 128 * 256;
  const auto projection = 128 * (8 + 7 + 256 + 300 + 7 * 8 + 256 * 300);
  const auto lift = 8 * (256 * 256 + 49 * 256 + 300 * 300 + 64 * 300) + projection;
  const auto expected_source = std::max(reverse, lift) + 8 * 49 + 64 * 9;
  const auto frame = 8 * 2 * (4 * 49 + 7);
  require(force[30] == expected_source + frame && force[30] > energy[30] &&
              force[13] == std::max(force[29], force[30]) && force[15 + 5] == energy[15 + 5] &&
              force[12] == energy[12] + 24 * 3,
          "DF forces did not price the metric response phase independently of the retained fit");
  force.fill(9);
  shape[21] = 4;
  require(force_query() != 0 &&
              std::all_of(force.begin(), force.end(), [](auto value) { return value == 9; }),
          "DF derivative fallback topology published a bounded-force inventory");
}
}  // namespace

int main() {
  test_fleet_owner();
  test_fleet_bridge();
  test_df_fleet_bridge();
  test_df_force_fleet_bridge();
  test_packed_fleet_bridge();
  test_packed_force_fleet_bridge();
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
